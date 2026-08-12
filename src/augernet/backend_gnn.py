"""
GNN Backend — model-specific routines for train_driver.py
===============================================================

Supports both model types from a single module:
  - ``cebe-gnn``   : CEBE binding-energy prediction (scalar, 1 model/fold)
  - ``auger-gnn``  : Auger spectrum prediction

Provides the routines for train_driver:
  load_data, train_single_run, load_saved_model,
  run_evaluation, run_unit_tests, run_predict
"""

from __future__ import annotations

import os
import json
import numpy as np
import torch
from typing import Any, Dict, List, Tuple
from sklearn.model_selection import KFold, GroupKFold

from augernet import gnn_train_utils as gtu
from augernet.feature_assembly import (
    assemble_dataset, compute_feature_tag, describe_features,
    parse_feature_keys, compute_feature_stats
)
from augernet.spec_utils import fit_spectrum_to_grid

from augernet import DATA_DIR, DATA_PROCESSED_DIR


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_overrides(cfg, overrides: dict) -> dict:
    """Resolve hyperparameters from cfg + per-config overrides.

    ``hp`` contains every value the training loop reads, not just the
    searchable ones.  ``OVERRIDABLE_FIELDS`` is the ``param_grid`` whitelist


    Parameters
    ----------
    cfg : AugerNetConfig
    overrides : dict
        Per-config values from a param search.  Every key must be in
        ``OVERRIDABLE_FIELDS``; anything else is a programming error rather
        than a config error, since ``load_config`` already validates
        ``param_grid`` against the same set.
    """
    from augernet.config import OVERRIDABLE_FIELDS

    # Start from the complete config, so the training loop can never be handed
    # a partial set regardless of what is currently searchable.
    hp = {name: getattr(cfg, name) for name in cfg.__dataclass_fields__}

    for key, value in overrides.items():
        if key not in OVERRIDABLE_FIELDS:
            raise ValueError(
                f"'{key}' was passed as a per-config override but is not in "
                f"OVERRIDABLE_FIELDS, so it cannot be varied by param search.\n"
                f"  Add it to OVERRIDABLE_FIELDS in config.py to make it "
                f"searchable, or remove it from the override."
            )
        hp[key] = value

    return hp


def _get_fold_split(calc_data, fold, n_folds, split_method, random_seed,
                    cutoff=0.65, verbose=False):
    """Compute molecule-level train/val indices for a single fold.

    Returns ``(train_idx, val_idx)`` as Python lists.
    """
    n_molecules = len(calc_data)
    if split_method == 'random':
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        folds = list(kf.split(np.arange(n_molecules)))
    elif split_method == 'butina':
        from augernet.build_molecular_graphs import get_butina_clusters
        smiles_list = [d.smiles for d in calc_data]
        cluster_ids = get_butina_clusters(smiles_list, cutoff=cutoff)
        if verbose:
            print(f"  Butina clustering: {len(set(cluster_ids))} clusters "
                  f"(cutoff=0.65)")
        gkf = GroupKFold(n_splits=n_folds)
        folds = list(gkf.split(np.arange(n_molecules), groups=cluster_ids))
    else:
        raise ValueError(
            f"Unknown split_method '{split_method}'. "
            f"Supported: 'random', 'butina'."
        )
    # folds from GroupKFold and KFold contain two lists [0] train and [1] val
    train_idx, val_idx = folds[fold - 1]  # fold is 1-indexed
    train_idx = train_idx.tolist() if hasattr(train_idx, 'tolist') else list(train_idx)
    val_idx = val_idx.tolist() if hasattr(val_idx, 'tolist') else list(val_idx)
    return train_idx, val_idx


def _extract_results(train_results):
    """Extract best-epoch metrics from a train_results list.

    Returns ``(best_val_loss, best_train_loss, best_val_epoch,
    final_train_loss, final_val_loss, n_epochs)``.
    """
    train_losses = [r[1] for r in train_results]
    val_losses   = [r[2] for r in train_results]
    best_idx     = int(np.argmin(val_losses))
    return (
        val_losses[best_idx],       # best_val_loss
        train_losses[best_idx],     # best_train_loss
        best_idx + 1,               # best_val_epoch (1-indexed)
        train_losses[-1],           # final_train_loss
        val_losses[-1],             # final_val_loss
        len(train_results),         # n_epochs
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Per-fold normalisation  
# ─────────────────────────────────────────────────────────────────────────────

_SPECTRUM_KEYS = ('fwhm', 'n_points', 'min_ke', 'max_ke', 'max_spec_len')


def _spectrum_params(cfg, hp):
    """Resolve the spectrum-grid parameters, honouring param-search overrides."""
    import types
    spec = types.SimpleNamespace()
    for k in _SPECTRUM_KEYS:
        spec.__dict__[k] = hp.get(k, getattr(cfg, k)) if hp else getattr(cfg, k)
    return spec


# ── CEBE target scale ────────────────────────────────────────────────────────

def _cebe_delta(d):
    """Raw CEBE target in eV: isolated-atom 1s BE minus the molecular CEBE.

    Reconstructed from ``atomic_be_eV`` and ``true_cebe``, both stored raw at
    graph-build time.  This is the same quantity ``build_graphs`` forms before
    normalising, so the target can be re-normalised per fold without
    regenerating any ``.pt`` file — and the value stored in ``cebe_y`` at prep
    time is never trusted.
    """
    return d.atomic_be_eV.view(-1).float() - d.true_cebe.view(-1).float()


def _fit_cebe_norm(calc_data, train_idx):
    """Fit the CEBE target mean/std on the training molecules of this fold."""
    vals = []
    for i in train_idx:
        d = calc_data[i]
        mask = d.node_mask.view(-1) > 0.5
        vals.append(_cebe_delta(d)[mask])
    if not vals:
        raise ValueError("_fit_cebe_norm: no training molecules supplied.")
    v = torch.cat(vals)
    if v.numel() < 2:
        raise ValueError(
            f"_fit_cebe_norm: only {v.numel()} carbon(s) in the training fold — "
            f"cannot fit a standard deviation."
        )
    # ddof=1, matching the historical np.std(..., ddof=1)
    return {'mean': float(v.mean()), 'std': float(v.std(unbiased=True)),
            'n_carbons': int(v.numel())}


def _apply_cebe_norm(data_lists, norm):
    """Rewrite ``cebe_y`` in place from the raw delta using *norm*.

    Non-carbon nodes keep the -1 sentinel that downstream code masks on.
    """
    mean, std = norm['mean'], norm['std']
    for lst in data_lists:
        for d in lst or ():
            mask = d.node_mask.view(-1) > 0.5
            y = torch.full((d.node_mask.numel(),), -1.0, dtype=torch.float32)
            y[mask] = (_cebe_delta(d)[mask] - mean) / std
            d.cebe_y = y.view(-1, 1)


# ── Auger target scale ───────────────────────────────────────────────────────

def _broaden_sticks(data_list, spec):
    """Gaussian-broaden singlet+triplet sticks onto the common grid.

    Returns ``[(data_obj, E_fitted, I_fitted), ...]`` with RAW intensities — no
    cross-molecule constant is involved at this stage.
    """
    out = []
    for d in data_list or ():
        n_atoms = d.node_mask.numel()
        s_y, t_y = d.sing_y, d.trip_y
        E_fitted = np.zeros((n_atoms, spec.n_points), dtype=np.float32)
        I_fitted = np.zeros((n_atoms, spec.n_points), dtype=np.float32)
        for c in d.node_mask.nonzero(as_tuple=True)[0].tolist():
            energy_stick = np.concatenate([s_y[c, :, 0], t_y[c, :, 0]])
            intensity_stick = np.concatenate([s_y[c, :, 1], t_y[c, :, 1]])
            E_fitted[c], I_fitted[c] = fit_spectrum_to_grid(
                energy_stick, intensity_stick, fwhm=spec.fwhm,
                energy_min=spec.min_ke, energy_max=spec.max_ke,
                n_points=spec.n_points, normalize=False)
        out.append((d, E_fitted, I_fitted))
    return out


def _fit_apply_auger_maxI(calc_data, train_idx, other_lists, spec, verbose=True):
    """Fit the Auger intensity scale on training molecules; apply it everywhere.

    ``maxI`` is the largest broadened per-carbon peak height over the TRAINING
    molecules.  Every split — including the validation fold and the calculated
    hold-out — is then divided by that one number, so all ``y_fitted`` tensors
    share a single scale.
    """
    fitted = _broaden_sticks(calc_data, spec)
    others = [_broaden_sticks(lst, spec) for lst in other_lists if lst]

    train_set = set(train_idx)
    maxI, where = 0.0, None
    for i, (d, _E, I) in enumerate(fitted):
        if i not in train_set:
            continue
        for c in d.node_mask.nonzero(as_tuple=True)[0].tolist():
            peak = float(I[c].max())
            if peak > maxI:
                maxI, where = peak, (getattr(d, 'mol_name', '?'), c)
    if maxI <= 0.0:
        raise ValueError(
            "_fit_apply_auger_maxI: fitted maxI is 0 — the training molecules "
            "produced no intensity on the "
            f"[{spec.min_ke}, {spec.max_ke}] eV grid."
        )

    for group in [fitted, *others]:
        for d, E, I in group:
            d.y_fitted = torch.tensor(I / maxI, dtype=torch.float32)
            d.e_fitted = torch.tensor(E, dtype=torch.float32)

    if verbose:
        n_other = sum(len(g) for g in others)
        print(f"  Auger maxI = {maxI:.6e}  (fitted on {len(train_set)} training "
              f"molecules; set by {where[0]} carbon {where[1]})")
        print(f"  Applied to {len(fitted)} train+val and {n_other} hold-out "
              f"molecules; training targets in (0, 1]")
    return maxI


# ── Node-feature statistics + assembly ───────────────────────────────────────

def _serialise_feature_stats(fs):
    """``{name: (mu, sigma)}`` tensors -> JSON-safe nested lists."""
    if not fs:
        return None
    return {name: {'mu': mu.view(-1).tolist(), 'sigma': sigma.view(-1).tolist()}
            for name, (mu, sigma) in fs.items()}


def _deserialise_feature_stats(blob):
    """Inverse of ``_serialise_feature_stats``."""
    if not blob:
        return None
    return {name: (torch.tensor(v['mu'], dtype=torch.float).view(1, -1),
                   torch.tensor(v['sigma'], dtype=torch.float).view(1, -1))
            for name, v in blob.items()}


# ── The one fitting site ─────────────────────────────────────────────────────

def _fit_fold_norm(data, cfg, hp, train_idx, verbose=True):
    """Fit every cross-molecule constant on this fold's training molecules and
    apply it to every split held in *data*.

    Returns a JSON-serialisable dict describing the transforms, which
    ``train_single_run`` writes beside the checkpoint.
    """
    calc_data = data['calc_data']
    feature_keys = parse_feature_keys(hp.get('feature_keys', cfg.feature_keys))
    fk_tag = compute_feature_tag(feature_keys)
    mode = hp.get('node_feature_norm', cfg.node_feature_norm)

    # Every other split that must be transformed by the fitted constants.
    aux = [data.get('test_data'), data.get('exp_data')]
    aux = [lst for lst in aux if lst]

    norm: Dict[str, Any] = {
        'model':               cfg.model,
        'feature_keys':        fk_tag,
        'node_feature_norm':   mode,
        'n_train_molecules':   len(train_idx),
    }

    # 1. CEBE target scale — the training target for cebe-gnn and for the CEBE
    #    head of a multi-task auger-gnn; a reporting scale otherwise.
    cebe = _fit_cebe_norm(calc_data, train_idx)
    _apply_cebe_norm([calc_data, *aux], cebe)
    norm['cebe'] = cebe
    if verbose:
        print(f"  CEBE norm: mean={cebe['mean']:.6f} std={cebe['std']:.6f} "
              f"(fitted on {cebe['n_carbons']} carbons in {len(train_idx)} "
              f"training molecules)")

    # 2. Auger target scale.
    if cfg.model == 'auger-gnn':
        spec = _spectrum_params(cfg, hp)
        stick_lists = [lst for lst in aux if lst and hasattr(lst[0], 'sing_y')]
        norm['auger_maxI'] = _fit_apply_auger_maxI(
            calc_data, train_idx, stick_lists, spec, verbose=verbose)
        norm['spectrum'] = {k: getattr(spec, k) for k in _SPECTRUM_KEYS}
        data['auger_maxI'] = norm['auger_maxI']

    # 3. Node features.  'graph' scaling is per-molecule and involves no
    #    cross-molecule statistic, so there is nothing to leak in that mode.
    fs = None
    if mode == 'data':
        fs = compute_feature_stats([calc_data[i] for i in train_idx], feature_keys)
    assemble_dataset(calc_data, feature_keys, scale_mode=mode, feature_stats=fs)
    for lst in aux:
        assemble_dataset(lst, feature_keys, scale_mode=mode, feature_stats=fs)

    data['feature_stats'] = fs
    data['assembled_feature_keys'] = fk_tag
    data['node_feature_norm'] = mode
    data['cebe_norm'] = cebe
    norm['feature_stats'] = _serialise_feature_stats(fs)

    if verbose:
        print(f"  Features: {fk_tag} ({describe_features(feature_keys)}), "
              f"{mode} scaling, x.shape[1]={calc_data[0].x.size(1)}")
    return norm


# ── Sidecar persistence ──────────────────────────────────────────────────────
# Mechanics live in augernet.norm_sidecar so backend_gnn and backend_cnn share
# one implementation of the naming and the mismatch reporting.  Re-exported
# here because callers (and tests) import them from the backend.

from augernet.norm_sidecar import (            # noqa: E402  (re-export)
    norm_sidecar_path,
    save_norm_sidecar,
    collect_mismatches,
    raise_on_mismatch,
)
from augernet.norm_sidecar import load_norm_sidecar as _load_sidecar

# Every GNN sidecar carries a 'cebe' block: it is the target scale for cebe-gnn
# and for the CEBE head of a multi-task auger-gnn, and the reporting scale
# otherwise.  Requiring it rejects a CNN sidecar handed to a GNN by mistake.
_GNN_SIDECAR_BLOCKS = ('cebe',)


def load_norm_sidecar(model_path: str) -> dict:
    """Load the normalisation constants fitted when *model_path* was trained."""
    return _load_sidecar(model_path, require=_GNN_SIDECAR_BLOCKS)


def apply_saved_norm(data, cfg, model_path, verbose=True):
    """Apply a checkpoint's saved constants to freshly loaded data.

    Used by ``evaluate`` mode, which loads raw graphs and has no training split
    of its own.  Mirrors ``_fit_fold_norm`` exactly, minus the fitting.
    """
    norm = load_norm_sidecar(model_path)
    feature_keys = parse_feature_keys(norm.get('feature_keys', cfg.feature_keys))
    mode = norm.get('node_feature_norm', 'graph')
    fs = _deserialise_feature_stats(norm.get('feature_stats'))

    lists = [data.get('calc_data'), data.get('test_data'), data.get('exp_data')]
    lists = [lst for lst in lists if lst]

    _apply_cebe_norm(lists, norm['cebe'])

    if cfg.model == 'auger-gnn' and 'auger_maxI' in norm:
        import types
        spec = types.SimpleNamespace(**norm.get(
            'spectrum', {k: getattr(cfg, k) for k in _SPECTRUM_KEYS}))
        maxI = float(norm['auger_maxI'])
        for lst in lists:
            if not lst or not hasattr(lst[0], 'sing_y'):
                continue
            for d, E, I in _broaden_sticks(lst, spec):
                d.y_fitted = torch.tensor(I / maxI, dtype=torch.float32)
                d.e_fitted = torch.tensor(E, dtype=torch.float32)
        data['auger_maxI'] = maxI

    for lst in lists:
        assemble_dataset(lst, feature_keys, scale_mode=mode, feature_stats=fs)

    data['feature_stats'] = fs
    data['node_feature_norm'] = mode
    data['assembled_feature_keys'] = compute_feature_tag(feature_keys)
    data['cebe_norm'] = norm['cebe']
    if verbose:
        print(f"  Loaded fold normalisation from {norm_sidecar_path(model_path)}")
        print(f"    CEBE mean={norm['cebe']['mean']:.6f} "
              f"std={norm['cebe']['std']:.6f}"
              + (f", auger maxI={norm['auger_maxI']:.6e}"
                 if 'auger_maxI' in norm else ''))
    return norm

def _train_one_model(train_data, val_data, in_channels, edge_dim, device, hp,
                     pred_type='CEBE', spectrum_dim=300, task_type='single',
                     out_dir=None, run_tag=None):

    """Build, train, and return a single MPNN model + train_results.

    ``out_dir`` / ``run_tag`` are forwarded to ``gtu.train_loop`` and control
    the per-epoch loss-history CSV.  If either is None no history is written.
    """
    # n_var: number of learnable log-variance terms for uncertainty weighting.
    # 3 when alpha_weight='uw' (CEBE + Auger + alpha), 2 otherwise (CEBE + Auger).
    # Not a config field -- derived here so the MPNN state_dict is always
    # self-consistent with the loss used during training.
    n_var =  2
    model = gtu.MPNN(
        num_layers=hp['n_layers'], emb_dim=hp['hidden_channels'],
        in_dim=in_channels, edge_dim=edge_dim,
        out_dim=1, layer_type=hp['layer_type'], pred_type=pred_type,
        dropout=hp['dropout'],
        spectrum_dim=spectrum_dim,
        task_type=task_type,
        n_var=n_var,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:  {n_params:,}")

    loop_kwargs = dict(
        num_epochs=hp['num_epochs'], batch_size=hp['batch_size'],
        max_lr=hp['learning_rate'],
        verbose=True, pred_type=pred_type,
        cebe_loss=hp['cebe_loss'], 
        patience=hp['patience'],
        random_seed=hp['random_seed'],
        optimizer_type=hp['optimizer_type'], weight_decay=hp['weight_decay'],
        gradient_clip_norm=hp['gradient_clip_norm'],
        warmup_epochs=hp['warmup_epochs'], min_lr=hp['min_lr'],
        scheduler_type=hp.get('scheduler_type', 'cosine'),
        pct_start=hp.get('pct_start', 0.3),
        task_type=task_type,
        out_dir=out_dir, run_tag=run_tag,
    )
    if pred_type == 'AUGER':
        loop_kwargs['auger_loss'] = hp.get('auger_loss', 'mse')
    if task_type == 'multi':
        loop_kwargs['mt_warmup_epochs']           = hp.get('mt_warmup_epochs', 10)
        loop_kwargs['mt_finetune_auger']           = hp.get('mt_finetune_auger', False)
        loop_kwargs['mt_finetune_epochs']          = hp.get('mt_finetune_epochs', 50)

    train_results = gtu.train_loop(train_data, val_data, model, device, **loop_kwargs)
    model.eval()
    return model, train_results

# ─────────────────────────────────────────────────────────────────────────────
#  CEBE experimental data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_exp_split_names(split):
    """Return hardcoded mol-name sets for the experimental val/eval split.

    Returns ``(val_names, eval_names)`` as sets.
    """
    _VAL_NAMES = {
        "fluoromethane",
        "1-3-5-trifluorobenzene",
        "1-2-3-5-tetrafluorobenzene",
        "pentafluorobenzene",
        "hexafluorobenzene",
        "nitrobenzene",
        "aniline",
        "tetracyanoethylene",
        "tetracyanoethylene-oxide",
        "4-fluorobenzonitrile",
        "benzotrifluoride",
        "benzonitrile",
        "benzaldehyde",
        "4-dimethylamino-aniline",
        "nitromethane",
        "cinnamaldehyde",
        "2-5-dimethylacetophenone",
        "nitrosyl-cyanide",
        "carbon-dioxide",
        "difluoromethane",
        "1-1-difluoroethylene",
        "trifluoroacetic-acid",
        "1-1-1-trifluoroethane",
        "carbonyl-fluoride",
        "hexafluoroethane",
        "bis-trifluoromethyl-ether",
        "bis-trifluoromethyl-peroxide",
        "hexafluoroacetone",
        "octafluoropropane",
        "dimethyl-carbonate",
        "2-nitropropane",
        "trimethylamine",
        "carbon-suboxide",
        "ethyl-fluoroacetate",
        "octafluoro-2-butene",
        "perfluoro-tert-butanol",
        "pyrrole",
        "2-butyne",
        "acetic-anhydride",
        "2-methyl-2-nitropropane",
        "1-1-3-trimethylurea",
        "hexafluoroacetylacetone",
        "cyanamide",
        "fluorobenzene",
        "p-fluoroaniline",
        "p-fluorophenol",
        "o-difluorobenzene",
        "acetic-acid",
        "acetone",
        "acrylic-acid",
        "benzene",
        "butane",
        "benzophenone",
        "cyclobutane",
        "cyclohexane",
        "ethyl-trifluoroacetate-esca",
        "hexane",
        "m-bis-trifluoromethyl-benzene",
        "methyl-methacrylate",
        "p-bis-trifluoromethyl-benzene",
        "pentane",
        "3-3-3-trifluoropropyne",
        "acetylacetone",
    }

    _EVAL_NAMES = {
        "1-2-3-4-tetrafluorobenzene",
        "1-2-4-5-tetrafluorobenzene",
        "phenol",
        "4-aminobenzonitrile",
        "toluene",
        "acetophenone",
        "octane",
        "decane",
        "tridecane",
        "methylamine",
        "vinyl-fluoride",
        "ethyl-fluoride",
        "difluoroacetic-acid",
        "trifluoroethylene",
        "tetrafluoroethylene",
        "bis-trifluoromethyl-trioxide",
        "ketene",
        "trifluoromethane",
        "cyanoguanidine",
        "3-3-3-trifluoropropene",
        "trifluoronitrosomethane",
        "hexafluoropropene",
        "ethyl-formate",
        "ethyl-difluoroacetate",
        "hexafluorocyclobutene",
        "hexafluoro-2-butyne",
        "octafluorocyclobutane",
        "isocyanic-acid",
        "ethyl-acetate",
        "trimethylacetonitrile",
        "p-fluoronitrobenzene",
        "p-difluorobenzene",
        "acrylonitrile",
        "adamantane",
        "anthrone",
        "2-4-6-trimethylacetophenone",
        "diphenyl-carbonate",
        "4-nitrobenzaldehyde",
        "4-trifluoromethyl-benzonitrile",
        "cyclopentane",
        "ethylene",
        "fluorenone",
        "formic-acid",
        "indole",
        "methyl-acrylate",
        "methyl-isobutyrate",
        "pyrimidine",
        "allene",
        "ketoavobenzone",
        "enolavobenzone",
    }

    return _VAL_NAMES, _EVAL_NAMES


def _split_exp_data(exp_data_all, cfg):
    """Partition experimental data according to ``cfg.cebe_exp_split``.

    Returns ``(exp_val_data, exp_eval_data)``.  Depending on the mode:

    - ``'all'``  — both lists contain all 113 molecules 
    - ``'val'``  — val = 63 validation, eval = empty
    - ``'eval'`` — val = empty, eval = 50 evaluation
    - ``'both'`` — val = 63, eval = 50  (run evaluation on each separately)
    """
    split = cfg.cebe_exp_split

    if split == 'all':
        return exp_data_all, exp_data_all

    val_names, eval_names = _load_exp_split_names(split)

    exp_val  = [d for d in exp_data_all if d.mol_name in val_names]
    exp_eval = [d for d in exp_data_all if d.mol_name in eval_names]

    missing_val  = val_names  - {d.mol_name for d in exp_val}
    missing_eval = eval_names - {d.mol_name for d in exp_eval}
    if missing_val:
        print(f"  Val split: {len(missing_val)} names not found in data: "
              f"{sorted(missing_val)[:5]}")
    if missing_eval:
        print(f"  Eval split: {len(missing_eval)} names not found in data: "
              f"{sorted(missing_eval)[:5]}")

    if split == 'val':
        return exp_val, []
    elif split == 'eval':
        return [], exp_eval
    elif split == 'both':
        return exp_val, exp_eval
    else:
        raise ValueError(
            f"Unknown exp_split '{split}'. "
            f"Supported: 'all', 'val', 'eval', 'both'."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(cfg) -> Dict[str, Any]:
    """Load training data for any model type.

    cebe-gnn: calculated + experimental data with val/eval split.
    auger-gnn: singlet + triplet stick data; y_fitted built per fold.

    Consequently the returned graphs carry an unassembled ``x`` and no
    ``y_fitted``; both appear once ``_fit_fold_norm`` has run.
    """
    print(f"\nLoading training data from: {DATA_PROCESSED_DIR}")
    print(f"Feature keys: {cfg.feature_keys}  ({describe_features(cfg.feature_keys_parsed)})")
    print(f"Model ID:     {cfg.model_id}")

    # ── CEBE-GNN ─────────────────────────────────────────────────────────
    if cfg.model == 'cebe-gnn':

        ds = gtu.LoadDataset(DATA_DIR, file_name=cfg.train_data_file)
        calc_data = [ds[i] for i in range(len(ds))]
        print(f"  Loaded calculated data: {len(calc_data)} molecules")

        # Experimental data — load all, then split
        exp_ds = gtu.LoadDataset(DATA_DIR, file_name=cfg.cebe_eval_data_file)
        exp_data_all = [exp_ds[i] for i in range(len(exp_ds))]
        exp_val, exp_eval = _split_exp_data(exp_data_all, cfg)

        exp_split = cfg.cebe_exp_split
        if exp_split == 'all':
            print(f"  Exp split: all ({len(exp_data_all)} molecules)")
        else:
            print(f"  Exp split: {exp_split}  "
                  f"(val={len(exp_val)}, eval={len(exp_eval)})")

        print(f"  Calculated data: {len(calc_data)} molecules "
              f"(features and target scale fitted per fold)")
        return {
            'calc_data': calc_data,
            'exp_data': exp_data_all,
            'exp_val_data': exp_val,
            'exp_eval_data': exp_eval,
        }

    # ── Auger-GNN ────────────────────────────────────────────────────────
    if cfg.model == 'auger-gnn':

        ds = gtu.LoadDataset(DATA_DIR, file_name=cfg.train_data_file)
        calc_data = [ds[i] for i in range(len(ds))]

        print(f"  Loaded {len(calc_data)} molecules "
              f"(spectra broadened and scaled per fold)")
        return {
                'calc_data': calc_data,
        }
    else:
        raise ValueError(
            f"Unknown model '{cfg.model}'. "
            f"Supported: 'cebe-gnn', 'auger-gnn'."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Single training run
# ─────────────────────────────────────────────────────────────────────────────

def train_single_run(
    data: Dict[str, Any],
    fold: int,
    n_folds: int,
    *,
    save_paths: Dict[str, str],
    output_dir: str,
    cfg,
    verbose: bool = True,
    **overrides,
) -> Dict[str, Any]:
    """Train a single GNN model (or model pair) on one fold.

    Returns a result dict compatible with train_driver expectations.

    Parameters
    ----------
    save_paths : dict
        Pre-built mapping of logical name to absolute ``.pth`` path.
        Built by ``train_driver._build_save_paths``

        Examples::
            {'model': '/…/cebe_gnn_…_fold1.pth'}  
    """
    hp = _extract_overrides(cfg, overrides)

    model_id = cfg.model_id

    gtu.seed(hp['random_seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure all target directories exist
    for p in save_paths.values():
        os.makedirs(os.path.dirname(p), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # train val splitting
    
    calc_data = data['calc_data']
    train_idx, val_idx = _get_fold_split(
        calc_data, fold, n_folds,
        hp['split_method'], hp['random_seed'], 
        cfg.butina_cutoff,
        verbose=verbose,
    )

    # training-set subsampling for data-efficiency sweep
    frac = hp.get('train_frac', 1.0)
    if frac < 1.0:
        rng = np.random.default_rng(hp.get('train_subsample_seed', 0))
        n = len(train_idx)
        k = max(1, int(round(frac * len(train_idx))))
        perm = rng.permutation(n)
        train_idx = [train_idx[perm[i]] for i in range(k)]
        if verbose:
            print(f"  [data-eff] train_frac={frac}  seed={hp.get('train_subsample_seed',0)}  "
              f" {k}/{n} train molecules kept")

    # ── Fit every cross-molecule constant on THIS fold's training molecules ──
    # Runs after the fold split and after the train_frac subsample, so the
    # molecules that define the constants are exactly the ones trained on.
    # Also builds y_fitted and assembles node features, so it must precede the
    # in_channels read below.
    if verbose:
        print(f"\n  Fitting fold normalisation ({len(train_idx)} training molecules)")
    norm = _fit_fold_norm(data, cfg, hp, train_idx, verbose=verbose)
    norm['fold'] = fold

    if verbose:
        print(f"\n{'=' * 70}")
        label = cfg.model.upper().replace('-', ' ')
        print(f"{label} TRAINING — Fold {fold}/{n_folds}")
        print(f"{'=' * 70}")
        print(f"  Model ID:    {model_id}")
        print(f"  Layer type:  {hp['layer_type']}")
        print(f"  Hidden:      {hp['hidden_channels']}")
        print(f"  Layers:      {hp['n_layers']}")
        print(f"  LR:          {hp['learning_rate']}")
        print(f"  Dropout:     {hp['dropout']}")
        print(f"  Batch:       {hp['batch_size']}")
        print(f"  Split:       {hp['split_method']}, "
              f"{len(train_idx)} train / {len(val_idx)} val molecules")
        print(f"  Save path(s):")
        for _label, _path in save_paths.items():
            print(f"    {_label}: {_path}")
        print(f"{'=' * 70}")

    in_channels = calc_data[0].x.size(1)
    edge_dim = calc_data[0].edge_attr.size(1)

    # ── Dispatch to model-specific training ──────────────────────────────
    # Each _train_* helper returns a result dict with trained model(s) and
    # metrics but does NOT save anything to disk — saving is done below.
    # Loss-history tag: derived from the model .pth stem rather than rebuilt
    # from model_id/fold, so the CSV always matches its checkpoint — including
    # the param-search `prefix` and `config_id` that _build_save_paths adds.
    run_tag = os.path.splitext(os.path.basename(save_paths['model']))[0]

    if cfg.model == 'cebe-gnn':
        result = _train_cebe(
            data, train_idx, val_idx, in_channels, edge_dim,
            device, hp, fold, verbose,
            out_dir=output_dir, run_tag=run_tag,
        )
    else:
        result = _train_auger(
            data, train_idx, val_idx, in_channels, edge_dim,
            device, hp, fold, verbose, cfg,
            out_dir=output_dir, run_tag=run_tag,
        )

    # ── Save model(s) to disk ────────────────────────────────────────────
    # All file I/O happens here regardless of model type, so the naming
    # convention is enforced in exactly one place.
    torch.save(result['model'].state_dict(), save_paths['model'])
    print(f"  Saved model to {save_paths['model']}")
    result['model_path'] = save_paths['model']

    # The fold's normalisation constants travel with the checkpoint.  evaluate
    # and predict have no training split and cannot re-derive them, and there
    # is deliberately no dataset-wide fallback to substitute.
    sidecar = save_norm_sidecar(save_paths['model'], norm)
    print(f"  Saved fold normalisation to {sidecar}")
    result['norm'] = norm
    result['norm_path'] = sidecar

    result['model_id'] = model_id
    return result


# ── CEBE training ────────────────────────────────────────────────────────────

def _train_cebe(data, train_idx, val_idx, in_channels, edge_dim,
                device, hp, fold, verbose, out_dir=None, run_tag=None):
    """Train a single CEBE GNN and return metrics.

    The only file written here is the loss-history CSV (streamed per epoch by
    ``gtu.train_loop``); model saving still happens in ``train_single_run``.
    """
    calc_data = data['calc_data']
    train_data = [calc_data[i] for i in train_idx]
    val_data   = [calc_data[i] for i in val_idx]

    model, train_results = _train_one_model(
        train_data, val_data, in_channels, edge_dim, device, hp,
        pred_type='CEBE', out_dir=out_dir, run_tag=run_tag,
    )

    bvl, btl, bve, ftl, fvl, n_ep = _extract_results(train_results)

    if verbose:
        print(f"\n  Fold {fold} complete — {n_ep} epochs, "
              f"best val loss {bvl:.6f} (epoch {bve})")

    return {
        'model': model, 'device': device, 'fold': fold,
        'best_val_loss': bvl, 'best_train_loss': btl,
        'best_val_epoch': bve,
        'final_train_loss': ftl, 'final_val_loss': fvl,
        'n_epochs': n_ep,
        'train_results': train_results,
    }

def _train_auger(data, train_idx, val_idx, in_channels, edge_dim,
                        device, hp, fold, verbose, cfg,
                        out_dir=None, run_tag=None):
    """Train a single auger GNN on one fold."""
    calc_data = data['calc_data']
    train_data = [calc_data[i] for i in train_idx]
    val_data   = [calc_data[i] for i in val_idx]

    model, train_results = _train_one_model(
        train_data, val_data, in_channels, edge_dim, device, hp,
        pred_type='AUGER', spectrum_dim=cfg.n_points, task_type=cfg.task_type,
        out_dir=out_dir, run_tag=run_tag,
    )

    bvl, btl, bve, ftl, fvl, n_ep = _extract_results(train_results)

    if verbose:
        print(f"\n  Fold {fold} complete — {n_ep} epochs, "
              f"best val loss {bvl:.6f} (epoch {bve})")

    return {
        'model': model, 'device': device, 'fold': fold,
        'best_val_loss': bvl, 'best_train_loss': btl,
        'best_val_epoch': bve,
        'final_train_loss': ftl, 'final_val_loss': fvl,
        'n_epochs': n_ep,
        'train_results': train_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_model_from_path(
    model_path: str,
    calc_data: list,
    *,
    layer_type: str,
    hidden_channels: int,
    n_layers: int,
    dropout: float = 0.0,
    pred_type: str = 'CEBE',
    spectrum_dim: int = 300,
    task_type: str = 'single',
    n_var: int = 2,
) -> Tuple[torch.nn.Module, torch.device]:
    """Load any GNN model from a .pth file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = calc_data[0].x.size(1)
    edge_dim = calc_data[0].edge_attr.size(1)

    model = gtu.MPNN(
        num_layers=n_layers, emb_dim=hidden_channels,
        in_dim=in_channels, edge_dim=edge_dim,
        out_dim=1, layer_type=layer_type, pred_type=pred_type,
        dropout=dropout,
        spectrum_dim=spectrum_dim, task_type=task_type,
        n_var=n_var,
    ).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    missing, unexpected = model.load_state_dict(
        torch.load(model_path, map_location=device), strict=False
    )
    if missing:
        print(f"  [load] Missing keys (will use init values): {missing}")
    if unexpected:
        print(f"  [load] Unexpected keys (ignored): {unexpected}")
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f" Loaded model from {model_path}  ({n_params:,} params)")
    return model, device


def _model_load_kwargs(cfg):
    """Return the extra kwargs for _load_model_from_path based on model type.

    Maps the high-level config (model name) to the MPNN constructor arguments
    needed to reconstruct the architecture at load time.

    n_var is derived here from cfg.alpha_weight (same rule as _train_one_model)
    so that the loaded model's log_var tensor has the correct dimension.
    """
    if cfg.model == 'cebe-gnn':
        return dict(pred_type='CEBE')
    elif cfg.model == 'auger-gnn':
        n_var = 2 
        kw = dict(pred_type='AUGER', task_type=cfg.task_type, n_var=n_var)
        kw['spectrum_dim'] = cfg.n_points
        return kw
    return {}


def load_saved_model(save_paths, data, cfg):
    """Load saved model(s) from pre-built paths.

    Parameters
    ----------
    save_paths : dict
        Mapping of logical name to absolute ``.pth`` path, as produced
        by ``train_driver._build_save_paths``.  Same dict that was
        passed to ``train_single_run`` at save time.

    Returns a result dict matching ``train_single_run`` output so
    downstream code (evaluation, unit tests) can consume either.
    """
    calc_data = data['calc_data']
    load_kw = dict(
        layer_type=cfg.layer_type,
        hidden_channels=cfg.hidden_channels,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        **_model_load_kwargs(cfg),
    )

    model_id = cfg.model_id

    model_path = save_paths['model']
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved model found:\n  {model_path}"
        )
    model, device = _load_model_from_path(model_path, calc_data, **load_kw)
    return {
        'model': model, 'device': device,
        'model_id': model_id,
    }

# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
    # ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(model_result, data, fold, output_dir, png_dir, cfg,
                   train_results=None, config_id=None, param_file_prefix=None,
                   exp_split=None):
    """Run evaluation after training.

    CEBE-GNN: evaluates on experimental data via evaluate_cebe_model.
    Auger-GNN: evaluates on experimental spectra via evaluate_auger_model.
    Auger-GNN (multi-task): additionally evaluates CEBE head on the
                            experimental CEBE dataset via evaluate_cebe_model,
                            identical to the CEBE-GNN evaluation path.
    """
    # Constants fitted for this fold.  Present because train_single_run ran
    # _fit_fold_norm, or because _run_evaluate applied a saved sidecar.
    cebe_norm = data.get('cebe_norm')
    if cebe_norm is None:
        raise RuntimeError(
            "Normalization stats for the given fold are not available on the "
            "data dict.  run_evaluation must be preceded by _fit_fold_norm "
            "(training) or apply_saved_norm (evaluate mode)."
        )

    # ── Auger-GNN evaluation ─────────────────────────────────────────────
    if cfg.model == 'auger-gnn':
        from .evaluation_scripts.evaluate_auger_model import (
            run_evaluation as _run_auger_eval,
        )

        model_dict = model_result
        device_a = model_result['device']
        train_results = model_result.get('train_results', train_results)
        model_id = model_result.get('model_id', cfg.model_id)

        auger_metrics = _run_auger_eval(
            model_dict, device_a,
            output_dir=output_dir, png_dir=png_dir, cfg=cfg,
            fold=fold, train_results=train_results,
            model_id=model_id, config_id=config_id,
            param_file_prefix=param_file_prefix,
            train_calc_data=data['calc_data'],
            test_calc_data=data['test_data'],
            maxI=data.get('auger_maxI'),
            scale_mode=data.get('node_feature_norm','graph'),
            feature_stats=data.get('feature_stats')
        )

        # ── Multi-task: also evaluate CEBE head on experimental CEBE data ──
        if getattr(cfg, 'task_type', 'single') == 'multi':
            cebe_model = model_dict.get('model')
            from .evaluation_scripts.evaluate_cebe_model import (
                run_evaluation as _run_cebe_eval,
            )
            # Load experimental CEBE data on-the-fly, then transform it with
            # THIS fold's constants — the same ones the CEBE head was trained
            # against.
            exp_ds = gtu.LoadDataset(DATA_DIR, file_name=cfg.cebe_eval_data_file)
            exp_data_mt = [exp_ds[i] for i in range(len(exp_ds))]

            _apply_cebe_norm([exp_data_mt], cebe_norm)
            assemble_dataset(exp_data_mt, cfg.feature_keys_parsed,
                            scale_mode=data.get('node_feature_norm', 'graph'),
                            feature_stats=data.get('feature_stats'))

            cebe_metrics = _run_cebe_eval(
                cebe_model, device_a, exp_data_mt,
                output_dir=output_dir, fold=fold,
                png_dir=png_dir,
                train_results=train_results,
                norm_stats=cebe_norm,
                model_id=model_id,
                config_id=config_id,
                param_file_prefix=param_file_prefix,
            )
            if isinstance(cebe_metrics, dict):
                auger_metrics.update({
                    'eval_cebe_mae': cebe_metrics.get('mae'),
                    'eval_cebe_r2':  cebe_metrics.get('r2'),
                    'eval_cebe_std': cebe_metrics.get('std'),
                })

        return auger_metrics

    # ── CEBE-GNN evaluation ──────────────────────────────────────────────
    from .evaluation_scripts.evaluate_cebe_model import run_evaluation as _run_eval

    model = model_result['model']
    device = model_result['device']
    train_results = model_result.get('train_results', train_results)
    model_id = model_result.get('model_id', cfg.model_id)

    split = exp_split if exp_split is not None else cfg.cebe_exp_split

    def _call(exp_data, suffix=''):
        pfx = param_file_prefix
        if suffix and pfx:
            pfx = f"{pfx}_{suffix}"
        elif suffix:
            pfx = suffix
        return _run_eval(
            model, device, exp_data,
            output_dir=output_dir, fold=fold,
            png_dir=png_dir,
            train_results=train_results,
            norm_stats=cebe_norm,
            model_id=model_id,
            config_id=config_id,
            param_file_prefix=pfx or None,
            alpha=cfg.cp_alpha, #split-CP alpha
        )

    if split == 'val' and data.get('exp_val_data'):
        return _call(data['exp_val_data'])
    elif split == 'eval' and data.get('exp_eval_data'):
        return _call(data['exp_eval_data'])
    elif split == 'both' and data.get('exp_val_data') and data.get('exp_eval_data'):
        # Both splits are evaluated, so both must be reported.  Returning only
        # the val metrics meant eval_mae / eval_r2 in the CV summary -- and the
        # mean +/- std quoted from it -- were the 63-molecule VALIDATION
        # numbers, while the 50-molecule held-out evaluation existed only in
        # the per-fold *_expeval_*_labels.txt files.
        #
        # Keys: eval_*     = validation split   (legacy names, unchanged so
        #                     existing summary JSONs stay readable)
        #       expeval_*  = held-out evaluation split
        # 'expeval_' is registered in train_driver._EVAL_PREFIXES, so these
        # aggregate to mean +/- std over folds exactly like eval_*.
        val_metrics  = _call(data['exp_val_data'],  suffix='expval')
        eval_metrics = _call(data['exp_eval_data'], suffix='expeval')

        merged = dict(val_metrics or {})
        for key, value in (eval_metrics or {}).items():
            merged[f'expeval_{key}'] = value
        return merged
    else:
        # 'all' or lists not available use full experimental set
        if data.get('exp_data'):
            return _call(data['exp_data'])
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Unit tests
# ─────────────────────────────────────────────────────────────────────────────

def run_unit_tests(model, data, cfg):
    """Run GNN symmetry tests."""
    # Unwrap if caller passed a (model, device) tuple
    if isinstance(model, tuple):
        model = model[0]

    if isinstance(model, dict):
        model = model['model']
    model.eval()
    gtu.run_unit_tests(model, data['calc_data'],
                        layer_type=cfg.layer_type)


# ─────────────────────────────────────────────────────────────────────────────
#  Predict: from user defined .xyz dir
# ─────────────────────────────────────────────────────────────────────────────

def _check_predict_config(cfg, norm, model_path):
    """Fail if the predict config disagrees with the checkpoint's sidecar.

    Only some mismatches are caught by the state-dict load: a different
    ``feature_keys`` of a DIFFERENT total width, or a different ``n_points``,
    change a layer shape and raise.  Same-width feature sets (e.g. '035' vs
    '034', both 202 columns) and the broadening parameters load cleanly and
    then silently predict against the wrong representation — this closes that
    gap.

    ``ke_shift_calc`` is deliberately not checked.  It is a display-only
    calibration applied to the output energy axis; it never enters training
    (``_broaden_sticks`` ignores it), so it is a labelling choice at predict
    time rather than something that must match.
    """
    pairs = [
        ('feature_keys',
         compute_feature_tag(cfg.feature_keys_parsed), norm.get('feature_keys')),
        ('node_feature_norm',
         getattr(cfg, 'node_feature_norm', 'graph'), norm.get('node_feature_norm')),
    ]
    # Spectrum grid — auger only; the sidecar records what training used.
    pairs += [(key, getattr(cfg, key, None), value)
              for key, value in (norm.get('spectrum') or {}).items()]

    problems = collect_mismatches(pairs)
    if problems and norm.get('feature_keys'):
        # Expand the feature-key line with what each set actually contains.
        problems = [
            p + (f"\n      config     = {describe_features(cfg.feature_keys_parsed)}"
                 f"\n      checkpoint = "
                 f"{describe_features(parse_feature_keys(norm['feature_keys']))}")
            if p.lstrip().startswith('feature_keys:') else p
            for p in problems
        ]
    raise_on_mismatch(problems, model_path=model_path, context='Predict config')


def run_predict(*, model_path: str, predict_dir: str, cfg):
    """
    Build graphs from .xyz files, run inference, and write output.
    """
    from augernet.build_molecular_graphs import (
        _mol_from_xyz_order,
        _build_node_and_edge_features,
        _initialize_all_atom_encoders,
    )
    from augernet import DATA_RAW_DIR
    from torch_geometric.data import Data

    # ── Check the config against what the checkpoint was trained with ────
    # Done before anything expensive so a mismatch fails immediately.
    norm = load_norm_sidecar(model_path)
    _check_predict_config(cfg, norm, model_path)

    # ── Discover .xyz files ──────────────────────────────────────────────
    xyz_files = sorted(
        f for f in os.listdir(predict_dir) if f.endswith('.xyz')
    )
    if not xyz_files:
        raise FileNotFoundError(f"No .xyz files found in: {predict_dir}")

    mol_names = [os.path.splitext(f)[0] for f in xyz_files]

    print(f"\n  Predict directory: {predict_dir}")
    print(f"  Found {len(xyz_files)} .xyz files")

    # ── Build graphs ─────────────────────────────────────────────────────
    skipatom_dir = os.path.join(DATA_RAW_DIR, 'skipatom')
    all_encoders = _initialize_all_atom_encoders(skipatom_dir)

    feature_keys = cfg.feature_keys_parsed

    print("  Building molecular graphs...")
    data_list = []
    for xyz_file, mol_name in zip(xyz_files, mol_names):
        xyz_path = os.path.join(predict_dir, xyz_file)
        mol, xyz_symbols, pos, smiles = _mol_from_xyz_order(
            xyz_path, labeled_atoms=False)

        n_atoms = mol.GetNumAtoms()
        dummy_cebe = np.full(n_atoms, -1.0)

        node_features, x, edge_index, edge_attr, atomic_be, _, _ = \
            _build_node_and_edge_features(
                mol, all_encoders, dummy_cebe)
        n_atoms = mol.GetNumAtoms()
        ###### cat feature debug check
        #category_feature=np.array([1, 0, 0])
        #cat_feat = np.tile(category_feature, (n_atoms, 1))
        #x = torch.tensor(cat_feat, dtype=torch.float)

        d = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            pos=torch.tensor(pos, dtype=torch.float),
            atomic_be_eV=atomic_be,
            atom_symbols=xyz_symbols,
            smiles=smiles,
            mol_name=mol_name,
        )
        for attr_name, tensor in node_features.items():
            setattr(d, attr_name, tensor)
        data_list.append(d)

    print(f"  Assembling features {cfg.feature_keys}")
    if getattr(cfg, 'node_feature_norm', 'graph') == 'data':
        raise ValueError(
            "run_predict cannot assemble features with node_feature_norm='data' "
            "— no calculated training set is loaded in predict mode."
        )
    from augernet.feature_assembly import assemble_dataset
    assemble_dataset(data_list, feature_keys)

    output_dir = cfg.outputs_dir
    os.makedirs(output_dir, exist_ok=True)
    file_stem = cfg.model_id

    if cfg.model == 'cebe-gnn':
        _predict_cebe(
            model_path, data_list, mol_names,
            cfg=cfg, output_dir=output_dir, file_stem=file_stem, norm=norm,
        )
    else: #auger-gnn
        load_kw = dict(
            layer_type=cfg.layer_type,
            hidden_channels=cfg.hidden_channels,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            **_model_load_kwargs(cfg),
        )
        model, device = _load_model_from_path(model_path, data_list, **load_kw)
        _predict_auger(
            model, device, data_list, mol_names,
            cfg=cfg, output_dir=output_dir, file_stem=file_stem,
        )

def _predict_cebe(model_path, data_list, mol_names, *, cfg, output_dir,
                  file_stem, norm=None):
    """Run CEBE inference and write per-atom output files."""
    from torch_geometric.loader import DataLoader

    # Denormalise with the constants fitted for THIS checkpoint's fold.
    # load_norm_sidecar raises if they are missing — predicting physical
    # binding energies with any other mean/std silently shifts every value.
    norm_stats = (norm or load_norm_sidecar(model_path))['cebe']
    mean = norm_stats['mean']
    std  = norm_stats['std']
    print(f"  CEBE denormalisation: mean={mean:.6f}  std={std:.6f}")

    model, device = _load_model_from_path(
        model_path, data_list,
        layer_type=cfg.layer_type,
        hidden_channels=cfg.hidden_channels,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    )

    print(f"\n{'=' * 80}")
    print(f"  PREDICT: Running CEBE inference on {len(data_list)} molecules")
    print(f"{'=' * 80}")

    loader = DataLoader(data_list, batch_size=1, shuffle=False)
    all_pred, all_atoms = [], []
    molecule_results = {}

    for i, d in enumerate(loader):
        n_nodes = d.x.size(0)
        nodes_in_edges = set(d.edge_index[0].tolist() + d.edge_index[1].tolist())
        mol_name = mol_names[i]

        if len(nodes_in_edges) < n_nodes:
            atom_syms = [str(s).strip() for s in (d.atom_symbols[0]
                         if isinstance(d.atom_symbols, list) else d.atom_symbols)]
            molecule_results[mol_name] = [(sym, float('nan')) for sym in atom_syms]
            all_pred.extend([float('nan')] * n_nodes)
            all_atoms.extend(atom_syms)
            print(f"  Skipping {mol_name}: disconnected graph")
            continue

        d = d.to(device)
        with torch.no_grad():
            out = model(d)

        pred_out = out.cpu().numpy()
        atomic_be_vals = d.atomic_be_eV.cpu().numpy()
        atom_syms = [str(s).strip() for s in (d.atom_symbols[0]
                     if isinstance(d.atom_symbols, list) else d.atom_symbols)]

        mol_rows = []
        for j in range(len(pred_out)):
            sym = atom_syms[j] if j < len(atom_syms) else '?'
            pred_be = float(atomic_be_vals[j] - (pred_out[j] * std + mean))
            mol_rows.append((sym, pred_be))
            all_pred.append(pred_be)
            all_atoms.append(sym)
        molecule_results[mol_name] = mol_rows

    label_path = os.path.join(output_dir, f"{file_stem}_labels.txt")
    with open(label_path, 'w') as f:
        f.write("# CEBE Predictions\n")
        f.write(f"# Model: {cfg.model_id}\n")
        f.write("# Note: Only carbon (C) predictions are meaningful.\n")
        f.write("# Columns: atom_symbol  pred_BE(eV)\n#\n")
        for mol_name, rows in molecule_results.items():
            f.write(f"# --- {mol_name} ---\n")
            for sym, pred_be in rows:
                marker = ' ' if sym == 'C' else '*'
                f.write(f"{sym:>3s}{marker}   {pred_be:10.4f}\n")
            f.write("\n")
    print(f"  Label results saved to {label_path}")

    carbon_preds = [p for s, p in zip(all_atoms, all_pred) if s == 'C']
    results_path = os.path.join(output_dir, f"{file_stem}_results.txt")
    np.savetxt(results_path, np.array(carbon_preds).reshape(-1, 1))
    print(f"  Numeric results saved to {results_path}")


def _predict_auger(model, device, data_list, mol_names,
                           *, cfg, output_dir, file_stem):
    """Run auger-gnn inference and write per-molecule spectrum files.
        Predicts normalized spectra with relative intensities between 
        carbons and molecules."""
    from torch_geometric.loader import DataLoader

    print(f"\n{'=' * 80}")
    print(f"  PREDICT: Running Auger inference on {len(data_list)} molecules")
    print(f"  FWHM: {cfg.fwhm} eV  |  Grid: [{cfg.min_ke}, {cfg.max_ke}] eV  "
          f"|  {cfg.n_points} points")
    print(f"{'=' * 80}")

    # Same KE calibration as evaluation, so predict output shares the axis of
    # the reported metrics and the published figures.
    energy_grid = np.linspace(cfg.min_ke, cfg.max_ke, cfg.n_points) + cfg.ke_shift_calc
    loader = DataLoader(data_list, batch_size=1, shuffle=False)

    model.eval()
    spectra = {}
    with torch.no_grad():
        for mol_idx, d in enumerate(loader):
            d = d.to(device)
            out = model(d)
            if getattr(model, 'task_type', 'single') == 'multi':
                out = out[1]

            # predict-mode graphs carry no node_mask (no reference CEBE exists),
            # so identify carbons from the atom symbols instead
            atom_syms = [str(s).strip() for s in (d.atom_symbols[0]
                         if isinstance(d.atom_symbols, list) else d.atom_symbols)]
            valid_nodes = [j for j, s in enumerate(atom_syms) if s == 'C']

            mol_spectrum = np.zeros(cfg.n_points)
            for nidx in valid_nodes:
                mol_spectrum += out[nidx].cpu().numpy()

            spectra[mol_names[mol_idx]] = mol_spectrum

    # Write one output file per molecule: two-column [energy, intensity]
    print(f"\n  Writing spectra to {output_dir}/")
    for mol_name, spectrum in spectra.items():
        out_path = os.path.join(output_dir, f"{file_stem}_{mol_name}_spectrum.txt")
        np.savetxt(out_path,
                   np.column_stack([energy_grid, spectrum]),
                   header=(f"energy_eV  intensity  (model={cfg.model_id}, "
                           f"fwhm={cfg.fwhm}, ke_shift={cfg.ke_shift_calc}, "))

    # Summary table
    print(f"\n{'Molecule':<22s} {'N_C':>5s} {'Peak KE (eV)':>14s}")
    print("-" * 45)
    for mol_name, spectrum in spectra.items():
        d = data_list[mol_names.index(mol_name)]
        n_c = sum(1 for s in (d.atom_symbols[0] if isinstance(d.atom_symbols, list)
                              else d.atom_symbols) if str(s).strip() == 'C')
        peak_ke = float(energy_grid[np.argmax(spectrum)]) if spectrum.max() > 0 else float('nan')
        print(f"{mol_name:<22s} {n_c:>5d} {peak_ke:>14.2f}")

    print(f"\n  {len(spectra)} spectra written to {output_dir}/")

