"""
GNN Backend — model-specific routines for train_driver.py
===============================================================

Supports both model types from a single module:
  - ``cebe-gnn``   : CEBE binding-energy prediction (scalar, 1 model/fold)
  - ``auger-gnn``  : Auger spectrum prediction
      - ``stick``   : separate singlet + triplet stick spectra (2 models/fold)
      - ``fitted``  : combined broadened spectrum (1 model/fold)

Provides the routines for train_driver:
  load_data, train_single_run, load_saved_model,
  run_evaluation, run_unit_tests, run_predict
"""

from __future__ import annotations

import os
import numpy as np
import torch
from typing import Any, Dict, List, Tuple
from sklearn.model_selection import KFold, GroupKFold

from augernet import gnn_train_utils as gtu
from augernet.feature_assembly import (
    assemble_dataset, compute_feature_tag, describe_features,
    parse_feature_keys,
)
from augernet.spec_utils import fit_spectrum_to_grid

from augernet import PROJECT_ROOT, DATA_DIR, DATA_PROCESSED_DIR


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_overrides(cfg, overrides: dict) -> dict:
    """Resolve hyperparameters from cfg + per-config overrides.

    Returns a flat dict of all HP values needed by the training loop.
    Any key listed in ``OVERRIDABLE_FIELDS`` can be overridden; values
    fall back to the base config.
    """
    from augernet.config import OVERRIDABLE_FIELDS

    hp = {}
    for key in OVERRIDABLE_FIELDS:
        if key in overrides:
            hp[key] = overrides[key]
        elif hasattr(cfg, key):
            hp[key] = getattr(cfg, key)
    return hp


def _get_fold_split(calc_data, fold, n_folds, split_method, random_seed,
                    verbose=False):
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
        cluster_ids = get_butina_clusters(smiles_list, cutoff=0.65)
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


def _handle_feature_override(data, cfg, overrides):
    """Re-assemble node features in-place if param search overrides feature_keys.

    Compares the requested feature_keys against what is currently assembled
    on the data objects.  If they differ, calls ``assemble_dataset`` to
    rebuild ``x`` on every graph in-place.  This is a side-effect-only
    helper — it modifies *data* and returns nothing.
    """
    override_fk = overrides.get('feature_keys')
    if override_fk is not None:
        fk_parsed = parse_feature_keys(override_fk)
        fk_tag = compute_feature_tag(fk_parsed)
        if fk_tag != data.get('assembled_feature_keys', cfg.feature_keys):
            print(f"  Re-assembling features for key override: {fk_tag}")
            ns = data.get('norm_stats')
            assemble_dataset(data['calc_data'], fk_parsed)
            if data.get('exp_data'):
                assemble_dataset(data['exp_data'], fk_parsed)
            data['assembled_feature_keys'] = fk_tag


def _rebuild_y_fitted(data, cfg, hp):
    """Rebuild ``y_fitted`` in-place when any spectrum parameter is overridden.

    Called from ``train_single_run`` for ``auger-gnn fitted`` mode so that
    param searches over ``fwhm``, ``n_points``, ``min_ke``, or ``max_ke``
    train against targets built with the correct overridden values instead
    of the base ``cfg`` values that were used in ``load_data``.

    The rebuild mutates the shared ``data['calc_data']`` list in-place (each
    Data object's ``y_fitted`` attribute is replaced).  Subsequent configs
    will call this function again with their own overrides, so each config
    always trains against self-consistent targets.
    """
    spectrum_keys = ('fwhm', 'n_points', 'min_ke', 'max_ke', 'max_spec_len')
    needs_rebuild = any(
        hp.get(k, getattr(cfg, k)) != getattr(cfg, k)
        for k in spectrum_keys
        if k in hp
    )
    if not needs_rebuild:
        return

    import types
    tmp = types.SimpleNamespace()
    for k in spectrum_keys:
        setattr(tmp, k, hp.get(k, getattr(cfg, k)))

    auger_norm_stats = data['auger_norm_stats']
    print(f"  [param override] Rebuilding y_fitted "
          f"(fwhm={tmp.fwhm}, n_points={tmp.n_points}, "
          f"ke=[{tmp.min_ke}, {tmp.max_ke}])")
    _attach_y_fitted(data['calc_data'], auger_norm_stats, tmp)


def _train_one_model(train_data, val_data, in_channels, edge_dim, device, hp,
                     pred_type='CEBE', spectrum_type='stick', spectrum_dim=300,
                     task_type='single'):
    """Build, train, and return a single MPNN model + train_results."""
    model = gtu.MPNN(
        num_layers=hp['n_layers'], emb_dim=hp['hidden_channels'],
        in_dim=in_channels, edge_dim=edge_dim,
        out_dim=1, layer_type=hp['layer_type'], pred_type=pred_type,
        dropout=hp['dropout'],
        spectrum_type=spectrum_type, spectrum_dim=spectrum_dim,
        task_type=task_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:  {n_params:,}")

    loop_kwargs = dict(
        num_epochs=hp['num_epochs'], batch_size=hp['batch_size'],
        max_lr=hp['learning_rate'],
        verbose=True, layer_type=hp['layer_type'], pred_type=pred_type,
        val_data_list=val_data, patience=hp['patience'],
        optimizer_type=hp['optimizer_type'], weight_decay=hp['weight_decay'],
        gradient_clip_norm=hp['gradient_clip_norm'],
        warmup_epochs=hp['warmup_epochs'], min_lr=hp['min_lr'],
        scheduler_type=hp.get('scheduler_type', 'cosine'),
        pct_start=hp.get('pct_start', 0.3),
        task_type=task_type,
    )
    if pred_type == 'AUGER':
        loop_kwargs['spectrum_type'] = spectrum_type
    if task_type == 'multi':
        loop_kwargs['mt_warmup_epochs']  = hp.get('mt_warmup_epochs', 10)
        loop_kwargs['mt_log_grad_cosine'] = hp.get('mt_log_grad_cosine', False)
        loop_kwargs['mt_finetune_auger'] = hp.get('mt_finetune_auger', False)
        loop_kwargs['mt_finetune_epochs'] = hp.get('mt_finetune_epochs', 50)

    train_results = gtu.train_loop(train_data, model, device, **loop_kwargs)
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
    """Partition experimental data according to ``cfg.exp_split``.

    Returns ``(exp_val_data, exp_eval_data)``.  Depending on the mode:

    - ``'all'``  — both lists contain all 113 molecules (legacy behaviour)
    - ``'val'``  — val = 63 validation, eval = empty
    - ``'eval'`` — val = empty, eval = 50 evaluation
    - ``'both'`` — val = 63, eval = 50  (run evaluation on each separately)
    """
    split = cfg.exp_split

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
#  On-the-fly stick to fitted conversion
# ─────────────────────────────────────────────────────────────────────────────

def _attach_y_fitted(calc_data, auger_norm_stats, cfg):

    """Create ``y_fitted`` on each singlet Data object by combining singlet +
    triplet stick spectra and Gaussian-broadening onto a common energy grid.

    Each atom's ``y`` is a 600-vector = [energies(300), intensities(300)].
    Energies are normalised by ``max_ke``; intensities by per-atom max.
    ``spec_len`` gives the number of valid entries in each half.

    After this call every element in *sing_data* has a new attribute
    ``y_fitted`` of shape ``(n_atoms, cfg.n_points)``.
    """
    max_spec_len = cfg.max_spec_len
    maxE = auger_norm_stats['maxE']

    for data in calc_data:
        n_atoms = data.x.size(0)
        s_y = data.sing_y.view(n_atoms, 2*max_spec_len).numpy()
        t_y = data.trip_y.view(n_atoms, 2*max_spec_len).numpy()
        fitted = np.zeros((n_atoms, cfg.n_points), dtype=np.float32)

        for ai in data.node_mask.nonzero(as_tuple=True)[0].tolist():
            # Un-normalise energies and concatenate singlet + triplet sticks
            e = np.concatenate([s_y[ai, :max_spec_len] * maxE,
                                t_y[ai, :max_spec_len] * maxE])
            inten = np.concatenate([s_y[ai, max_spec_len:],
                                    t_y[ai, max_spec_len:]])
            _, fitted[ai] = fit_spectrum_to_grid(
                e, inten, fwhm=cfg.fwhm,
                energy_min=cfg.min_ke, energy_max=cfg.max_ke,
                n_points=cfg.n_points,
            )
        data.y_fitted = torch.tensor(fitted, dtype=torch.float32)
    print(f"  Built y_fitted on-the-fly ({cfg.n_points}-pt grid, "
          f"fwhm={cfg.fwhm})")


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(cfg) -> Dict[str, Any]:
    """Load training data for any model type.

    CEBE-GNN: calculated + experimental data with val/eval split.
    Auger-GNN stick:  singlet + triplet calculated data.
    Auger-GNN fitted: singlet + triplet stick data to y_fitted built on-the-fly.

    Feature assembly is deferred to ``train_single_run`` / ``run_evaluation``
    so that param-search can override ``feature_keys`` per configuration.
    """
    print(f"\nLoading training data from: {DATA_PROCESSED_DIR}")
    print(f"Feature keys: {cfg.feature_keys}  ({describe_features(cfg.feature_keys_parsed)})")
    print(f"Model ID:     {cfg.model_id}")

    feature_keys = cfg.feature_keys_parsed

    # Load dataset-wide CEBE norm stats for mol_be scaling (key 4)
    cebe_norm_stats = torch.load(cfg.cebe_norm_stats_file, weights_only=False)

    # ── CEBE-GNN ─────────────────────────────────────────────────────────
    if cfg.model == 'cebe-gnn':

        ds = gtu.LoadDataset(DATA_DIR, file_name='gnn_calc_cebe_data.pt')
        calc_data = [ds[i] for i in range(len(ds))]
        print(f"  Loaded calculated data: {len(calc_data)} molecules")

        # Experimental data — load all, then split
        exp_ds = gtu.LoadDataset(DATA_DIR, file_name='gnn_exp_cebe_data.pt')
        exp_data_all = [exp_ds[i] for i in range(len(exp_ds))]
        exp_val, exp_eval = _split_exp_data(exp_data_all, cfg)

        exp_split = cfg.exp_split
        if exp_split == 'all':
            print(f"  Exp split: all ({len(exp_data_all)} molecules)")
        else:
            print(f"  Exp split: {exp_split}  "
                  f"(val={len(exp_val)}, eval={len(exp_eval)})")

        print(f"  Assembling features {cfg.feature_keys}")
        assemble_dataset(calc_data, feature_keys)
        assemble_dataset(exp_data_all, feature_keys)
        print(f"  Calculated data: {len(calc_data)} molecules, "
              f"x.shape[1]={calc_data[0].x.size(1)}")
        return {
            'calc_data': calc_data,
            'exp_data': exp_data_all,
            'exp_val_data': exp_val,
            'exp_eval_data': exp_eval,
            'assembled_feature_keys': cfg.feature_keys,
            'norm_stats': cebe_norm_stats,
        }

    # ── Auger-GNN ────────────────────────────────────────────────────────
    if cfg.model == 'auger-gnn':
        spec_type = cfg.spectrum_type

        # Singlet and triplet are always loaded
        ds = gtu.LoadDataset(DATA_DIR, file_name='gnn_calc_auger_data.pt')
        calc_data = [ds[i] for i in range(len(ds))]

        auger_norm_stats = torch.load(cfg.auger_norm_stats_file, weights_only=False)
        # Reshape y from (n_atoms*600, 1) to (n_atoms, 600) so that PyG
        # DataLoader batching concatenates correctly along the atom axis.
        for d in calc_data:
            n_atoms = d.x.size(0)
            d.sing_y = d.sing_y.view(n_atoms, 2*cfg.max_spec_len)
            d.trip_y = d.trip_y.view(n_atoms, 2*cfg.max_spec_len)

        print(f"  Loaded {len(calc_data)} molecules ({spec_type})")
        if spec_type == 'stick':
            print(f"  Assembling features {cfg.feature_keys}")
            assemble_dataset(calc_data, feature_keys)
            print(f"  x.shape[1]={calc_data[0].x.size(1)}")

            return {
                'calc_data': calc_data,  # used for splitting
                'assembled_feature_keys': cfg.feature_keys,
                'cebe_norm_stats': cebe_norm_stats,
                'auger_norm_stats': auger_norm_stats,
            }
        else:  # fitted — build y_fitted on-the-fly from stick data
            _attach_y_fitted(calc_data, auger_norm_stats, cfg)
            print(f"  Assembling features {cfg.feature_keys}")
            assemble_dataset(calc_data, feature_keys)
            print(f"  x.shape[1]={calc_data[0].x.size(1)}, "
                  f"y_fitted.shape={calc_data[0].y_fitted.shape}")
            # Note sing_calc_data is both singlet and triplet
            return {
                    'calc_data': calc_data,
                    'assembled_feature_keys': cfg.feature_keys,
                    'cebe_norm_stats': cebe_norm_stats,
                    'auger_norm_stats': auger_norm_stats
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

    Works for CEBE-GNN, Auger-GNN stick, and Auger-GNN fitted.
    Returns a result dict compatible with train_driver expectations.

    Parameters
    ----------
    save_paths : dict
        Pre-built mapping of logical name to absolute ``.pth`` path.
        Built by ``train_driver._build_save_paths``

        Examples::

            {'model': '/…/cebe_gnn_…_fold1.pth'}           # cebe / fitted
            {'singlet': '/…/…_singlet_fold1.pth',           # auger stick
             'triplet': '/…/…_triplet_fold1.pth'}
    """
    hp = _extract_overrides(cfg, overrides)

    # If param search overrides feature_keys, re-assemble node features
    # in-place on the shared data dict before training starts.
    _handle_feature_override(data, cfg, overrides)

    # For auger-gnn fitted: rebuild y_fitted targets if spectrum params
    # (fwhm, n_points, min_ke, max_ke) were overridden for this config.
    if cfg.model == 'auger-gnn' and getattr(cfg, 'spectrum_type', 'stick') == 'fitted':
        _rebuild_y_fitted(data, cfg, hp)

    model_id = cfg.model_id

    gtu.seed(hp['random_seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure all target directories exist
    for p in save_paths.values():
        os.makedirs(os.path.dirname(p), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # ── Splitting (shared) ───────────────────────────────────────────────
    calc_data = data['calc_data']
    train_idx, val_idx = _get_fold_split(
        calc_data, fold, n_folds,
        hp['split_method'], hp['random_seed'], verbose=verbose,
    )

    if verbose:
        print(f"\n{'=' * 70}")
        label = cfg.model.upper().replace('-', ' ')
        if cfg.model == 'auger-gnn':
            label += f" ({cfg.spectrum_type})"
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
    if cfg.model == 'cebe-gnn':
        result = _train_cebe(
            data, train_idx, val_idx, in_channels, edge_dim,
            device, hp, fold, verbose,
        )
    elif cfg.spectrum_type == 'stick':
        result = _train_auger_stick(
            data, train_idx, val_idx, in_channels, edge_dim,
            device, hp, fold, verbose, cfg,
        )
    else:
        result = _train_auger_fitted(
            data, train_idx, val_idx, in_channels, edge_dim,
            device, hp, fold, verbose, cfg,
        )

    # ── Save model(s) to disk ────────────────────────────────────────────
    # All file I/O happens here regardless of model type, so the naming
    # convention is enforced in exactly one place.
    if cfg.model == 'auger-gnn' and cfg.spectrum_type == 'stick':
        torch.save(result['sing_model'].state_dict(), save_paths['singlet'])
        torch.save(result['trip_model'].state_dict(), save_paths['triplet'])
        print(f"  Saved singlet to {save_paths['singlet']}")
        print(f"  Saved triplet to {save_paths['triplet']}")
        result['model_path'] = save_paths['singlet']
        result['sing_model_path'] = save_paths['singlet']
        result['trip_model_path'] = save_paths['triplet']
    else:
        torch.save(result['model'].state_dict(), save_paths['model'])
        print(f"  Saved model to {save_paths['model']}")
        result['model_path'] = save_paths['model']

    result['model_id'] = model_id
    return result


# ── CEBE training ────────────────────────────────────────────────────────────

def _train_cebe(data, train_idx, val_idx, in_channels, edge_dim,
                device, hp, fold, verbose):
    """Train a single CEBE GNN and return metrics (no file I/O)."""
    calc_data = data['calc_data']
    train_data = [calc_data[i] for i in train_idx]
    val_data   = [calc_data[i] for i in val_idx]

    model, train_results = _train_one_model(
        train_data, val_data, in_channels, edge_dim, device, hp,
        pred_type='CEBE',
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


# ── Auger stick training (singlet + triplet) ─────────────────────────────────

def _train_auger_stick(data, train_idx, val_idx, in_channels, edge_dim,
                       device, hp, fold, verbose, cfg):
    """Train singlet + triplet GNN models on one fold."""
    import copy
    calc_data = data['calc_data']
    spec_dim = 2 * cfg.max_spec_len

    # Build singlet and triplet views from the combined data objects.
    # Shallow copies let us set d.y / d.mask_bin independently without
    # touching the shared tensors in the original list.
    sing_calc = []
    trip_calc = []
    for d in calc_data:
        n_atoms = d.x.size(0)
        ds = copy.copy(d)
        ds.y = d.sing_y.view(n_atoms, spec_dim)
        ds.mask_bin = d.sing_mask_bin
        sing_calc.append(ds)

        dt = copy.copy(d)
        dt.y = d.trip_y.view(n_atoms, spec_dim)
        dt.mask_bin = d.trip_mask_bin
        trip_calc.append(dt)

    sing_train = [sing_calc[i] for i in train_idx]
    sing_val   = [sing_calc[i] for i in val_idx]
    trip_train = [trip_calc[i] for i in train_idx]
    trip_val   = [trip_calc[i] for i in val_idx]

    # ── Singlet ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}\n  SINGLET MODEL\n{'─' * 70}")
    sing_model, sing_results = _train_one_model(
        sing_train, sing_val, in_channels, edge_dim, device, hp,
        pred_type='AUGER', spectrum_type='stick',
        spectrum_dim=cfg.max_spec_len,
        task_type=cfg.task_type,
    )

    # ── Triplet ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}\n  TRIPLET MODEL\n{'─' * 70}")
    trip_model, trip_results = _train_one_model(
        trip_train, trip_val, in_channels, edge_dim, device, hp,
        pred_type='AUGER', spectrum_type='stick',
        spectrum_dim=cfg.max_spec_len,
        task_type=cfg.task_type,
    )

    # ── Combine results ──────────────────────────────────────────────────
    s_bvl, s_btl, s_bve, s_ftl, s_fvl, s_nep = _extract_results(sing_results)
    t_bvl, t_btl, t_bve, t_ftl, t_fvl, t_nep = _extract_results(trip_results)

    combined_val_loss = (s_bvl + t_bvl) / 2

    if verbose:
        print(f"\n  Fold {fold} complete")
        print(f"    Singlet: {s_nep} epochs, best val loss {s_bvl:.6f}")
        print(f"    Triplet: {t_nep} epochs, best val loss {t_bvl:.6f}")
        print(f"    Combined val loss: {combined_val_loss:.6f}")

    return {
        'model': sing_model, 'device': device, 'fold': fold,
        'sing_model': sing_model, 'trip_model': trip_model,
        'best_val_loss': combined_val_loss,
        'best_train_loss': (s_btl + t_btl) / 2,
        'best_val_epoch': max(s_bve, t_bve),
        'final_train_loss': (s_ftl + t_ftl) / 2,
        'final_val_loss': (s_fvl + t_fvl) / 2,
        'n_epochs': max(s_nep, t_nep),
        'train_results': sing_results,  # driver uses for loss curves
    }


# ── Auger fitted training (single combined model) ────────────────────────────

def _train_auger_fitted(data, train_idx, val_idx, in_channels, edge_dim,

                        device, hp, fold, verbose, cfg):
    """Train a single fitted-spectrum GNN on one fold."""
    calc_data = data['calc_data']
    train_data = [calc_data[i] for i in train_idx]
    val_data   = [calc_data[i] for i in val_idx]

    model, train_results = _train_one_model(
        train_data, val_data, in_channels, edge_dim, device, hp,
        pred_type='AUGER', spectrum_type='fitted',
        spectrum_dim=cfg.n_points,
        task_type=cfg.task_type,
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
    spectrum_type: str = 'stick',
    spectrum_dim: int = 300,
    task_type: str = 'single',
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
        spectrum_type=spectrum_type, spectrum_dim=spectrum_dim,
        task_type=task_type,
    ).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f" Loaded model from {model_path}  ({n_params:,} params)")
    return model, device


def _model_load_kwargs(cfg):
    """Return the extra kwargs for _load_model_from_path based on model type.

    Maps the high-level config (model name, spectrum_type) to the MPNN
    constructor arguments needed to reconstruct the architecture.
    """
    if cfg.model == 'cebe-gnn':
        return dict(pred_type='CEBE')
    elif cfg.model == 'auger-gnn':
        kw = dict(pred_type='AUGER', spectrum_type=cfg.spectrum_type,
                  task_type=cfg.task_type)
        if cfg.spectrum_type == 'fitted':
            kw['spectrum_dim'] = cfg.n_points
        else:
            kw['spectrum_dim'] = cfg.max_spec_len
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

    # ── Auger stick: load singlet + triplet pair ─────────────────────────
    if cfg.model == 'auger-gnn' and cfg.spectrum_type == 'stick':
        sing_path = save_paths['singlet']
        trip_path = save_paths['triplet']
        if not os.path.exists(sing_path):
            raise FileNotFoundError(
                f"No saved singlet model found:\n  {sing_path}"
            )
        sing_model, device = _load_model_from_path(sing_path, calc_data, **load_kw)
        trip_model = None
        if os.path.exists(trip_path):
            trip_model, _ = _load_model_from_path(trip_path, calc_data, **load_kw)
        return {
            'model': sing_model, 'device': device,
            'sing_model': sing_model, 'trip_model': trip_model,
            'model_id': model_id,
        }

    # ── All other model types: single file ───────────────────────────────
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
    # ── Auger-GNN evaluation ─────────────────────────────────────────────
    if cfg.model == 'auger-gnn':
        from .evaluation_scripts.evaluate_auger_model import (
            run_evaluation as _run_auger_eval,
        )

        if isinstance(model_result, dict):
            model_dict = model_result
            device_a = model_result['device']
            train_results = model_result.get('train_results', train_results)
            model_id = model_result.get('model_id', cfg.model_id)
        elif isinstance(model_result, tuple):
            model_dict = {'model': model_result[0]}
            device_a = model_result[1]
            model_id = cfg.model_id
        else:
            model_dict = {'model': model_result}
            device_a = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_id = cfg.model_id

        auger_metrics = _run_auger_eval(
            model_dict, device_a,
            output_dir=output_dir, png_dir=png_dir, cfg=cfg,
            fold=fold, train_results=train_results,
            model_id=model_id, config_id=config_id,
            param_file_prefix=param_file_prefix,
        )

        # ── Multi-task: also evaluate CEBE head on experimental CEBE data ──
        if getattr(cfg, 'task_type', 'single') == 'multi':
            # Use singlet model (both models share the same CEBE head weights)
            cebe_model = model_dict.get('sing_model', model_dict.get('model'))
            if cebe_model is not None:
                from .evaluation_scripts.evaluate_cebe_model import (
                    run_evaluation as _run_cebe_eval,
                )
                # Load and assemble experimental CEBE data on-the-fly
                exp_ds = gtu.LoadDataset(DATA_DIR, file_name='gnn_exp_cebe_data.pt')
                exp_data_mt = [exp_ds[i] for i in range(len(exp_ds))]
                assemble_dataset(exp_data_mt, cfg.feature_keys_parsed)
                _run_cebe_eval(
                    cebe_model, device_a, exp_data_mt,
                    output_dir=output_dir, fold=fold,
                    png_dir=png_dir,
                    train_results=train_results,
                    norm_stats_file=cfg.cebe_norm_stats_file,
                    model_id=model_id,
                    config_id=config_id,
                    param_file_prefix=param_file_prefix,
                )

        return auger_metrics

    # ── CEBE-GNN evaluation ──────────────────────────────────────────────
    from .evaluation_scripts.evaluate_cebe_model import run_evaluation as _run_eval

    if isinstance(model_result, dict):
        model = model_result['model']
        device = model_result['device']
        train_results = model_result.get('train_results', train_results)
        model_id = model_result.get('model_id', cfg.model_id)
    elif isinstance(model_result, tuple):
        model, device = model_result
        model_id = cfg.model_id
    else:
        model = model_result
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_id = cfg.model_id

    split = exp_split if exp_split is not None else cfg.exp_split

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
            norm_stats_file=cfg.cebe_norm_stats_file,
            model_id=model_id,
            config_id=config_id,
            param_file_prefix=pfx or None,
            alpha=cfg.cp_alpha,
        )

    if split == 'val' and data.get('exp_val_data'):
        return _call(data['exp_val_data'])
    elif split == 'eval' and data.get('exp_eval_data'):
        return _call(data['exp_eval_data'])
    elif split == 'both' and data.get('exp_val_data') and data.get('exp_eval_data'):
        val_metrics = _call(data['exp_val_data'], suffix='expval')
        _call(data['exp_eval_data'], suffix='expeval')
        return val_metrics
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

    if cfg.model == 'auger-gnn' and cfg.spectrum_type == 'stick':
        # Test both singlet and triplet models
        if isinstance(model, dict):
            sing = model.get('sing_model', model.get('model'))
            trip = model.get('trip_model')
        else:
            sing, trip = model, None

        if sing:
            sing.eval()
            print("\n  [singlet model]")
            gtu.run_unit_tests(sing, data['sing_calc_data'],
                               layer_type=cfg.layer_type)
        if trip:
            trip.eval()
            print("\n  [triplet model]")
            gtu.run_unit_tests(trip, data['trip_calc_data'],
                               layer_type=cfg.layer_type)
    else:
        # CEBE or Auger fitted — single model
        if isinstance(model, dict):
            model = model['model']
        model.eval()
        gtu.run_unit_tests(model, data['calc_data'],
                           layer_type=cfg.layer_type)


# ─────────────────────────────────────────────────────────────────────────────
#  Predict: from user defined .xyz dir
# ─────────────────────────────────────────────────────────────────────────────

def run_predict(*, model_path: str, predict_dir: str, cfg):
    """Build graphs from .xyz files, run inference, and write output.

    Supports ``cebe-gnn`` (scalar per-atom CEBE) and ``auger-gnn``
    (per-molecule Auger spectrum, fitted or stick).

    For ``auger-gnn stick``, both ``model_path`` (singlet) and
    ``cfg.trip_model_path`` (triplet) must be set in the YAML.
    """
    from augernet.build_molecular_graphs import (
        _mol_from_xyz_order,
        _build_node_and_edge_features,
        _initialize_all_atom_encoders,
    )
    from augernet import DATA_RAW_DIR
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

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

    # Category feature: choose CEBE category for node features
    category_feature = np.array([1, 0, 0])

    print("  Building molecular graphs...")
    data_list = []
    for xyz_file, mol_name in zip(xyz_files, mol_names):
        xyz_path = os.path.join(predict_dir, xyz_file)
        mol, xyz_symbols, pos, smiles = _mol_from_xyz_order(
            xyz_path, labeled_atoms=False)

        n_atoms = mol.GetNumAtoms()
        dummy_cebe = np.full(n_atoms, -1.0)

        node_features, x, edge_index, edge_attr, atomic_be, _ = \
            _build_node_and_edge_features(
                mol, all_encoders, category_feature, dummy_cebe)

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
    from augernet.feature_assembly import assemble_dataset
    assemble_dataset(data_list, feature_keys)

    output_dir = cfg.outputs_dir
    os.makedirs(output_dir, exist_ok=True)
    file_stem = cfg.model_id

    if cfg.model == 'cebe-gnn':
        _predict_cebe(
            model_path, data_list, mol_names,
            cfg=cfg, output_dir=output_dir, file_stem=file_stem,
        )

    elif cfg.model == 'auger-gnn':
        load_kw = dict(
            layer_type=cfg.layer_type,
            hidden_channels=cfg.hidden_channels,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            **_model_load_kwargs(cfg),
        )
        if cfg.spectrum_type == 'fitted':
            model, device = _load_model_from_path(model_path, data_list, **load_kw)
            _predict_auger_fitted(
                model, device, data_list, mol_names,
                cfg=cfg, output_dir=output_dir, file_stem=file_stem,
            )
        else:  # stick
            sing_model, device = _load_model_from_path(model_path, data_list, **load_kw)
            trip_model = None
            trip_path = getattr(cfg, 'trip_model_path', '')
            if trip_path:
                trip_path = os.path.abspath(trip_path)
                trip_model, _ = _load_model_from_path(trip_path, data_list, **load_kw)
            else:
                print("  WARNING: trip_model_path not set — only singlet spectra will be written.")
            _predict_auger_stick(
                sing_model, trip_model, device, data_list, mol_names,
                cfg=cfg, output_dir=output_dir, file_stem=file_stem,
            )
    else:
        raise NotImplementedError(
            f"Predict mode not implemented for model '{cfg.model}'."
        )


def _predict_cebe(model_path, data_list, mol_names, *, cfg, output_dir, file_stem):
    """Run CEBE inference and write per-atom output files."""
    from torch_geometric.loader import DataLoader

    norm_stats = torch.load(cfg.cebe_norm_stats_file, weights_only=False)
    mean = norm_stats['mean']
    std  = norm_stats['std']

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


def _predict_auger_fitted(model, device, data_list, mol_names,
                           *, cfg, output_dir, file_stem):
    """Run auger-gnn fitted inference and write per-molecule spectrum files."""
    from torch_geometric.loader import DataLoader

    print(f"\n{'=' * 80}")
    print(f"  PREDICT: Running Auger fitted inference on {len(data_list)} molecules")
    print(f"  FWHM: {cfg.fwhm} eV  |  Grid: [{cfg.min_ke}, {cfg.max_ke}] eV  "
          f"|  {cfg.n_points} points")
    print(f"{'=' * 80}")

    energy_grid = np.linspace(cfg.min_ke, cfg.max_ke, cfg.n_points)
    loader = DataLoader(data_list, batch_size=1, shuffle=False)

    model.eval()
    spectra = {}
    with torch.no_grad():
        for mol_idx, d in enumerate(loader):
            d = d.to(device)
            out = model(d)
            if getattr(model, 'task_type', 'single') == 'multi':
                out = out[1]

            # node_mask identifies carbon atoms with predictions
            node_mask = d.node_mask.squeeze()
            valid_nodes = node_mask.nonzero(as_tuple=True)[0]

            mol_spectrum = np.zeros(cfg.n_points)
            for nidx in valid_nodes:
                mol_spectrum += out[nidx].cpu().numpy()

            if mol_spectrum.max() > 0:
                mol_spectrum /= mol_spectrum.max()
            spectra[mol_names[mol_idx]] = mol_spectrum

    # Write one output file per molecule: two-column [energy, intensity]
    print(f"\n  Writing spectra to {output_dir}/")
    for mol_name, spectrum in spectra.items():
        out_path = os.path.join(output_dir, f"{file_stem}_{mol_name}_spectrum.txt")
        np.savetxt(out_path,
                   np.column_stack([energy_grid, spectrum]),
                   header=f"energy_eV  intensity  (model={cfg.model_id}, fwhm={cfg.fwhm})",
                   fmt="%.6f")

    # Summary table
    print(f"\n{'Molecule':<22s} {'N_C':>5s} {'Peak KE (eV)':>14s}")
    print("-" * 45)
    for mol_name, spectrum in spectra.items():
        d = data_list[mol_names.index(mol_name)]
        n_c = int(d.node_mask.sum().item()) if hasattr(d, 'node_mask') else 0
        peak_ke = float(energy_grid[np.argmax(spectrum)]) if spectrum.max() > 0 else float('nan')
        print(f"{mol_name:<22s} {n_c:>5d} {peak_ke:>14.2f}")

    print(f"\n  {len(spectra)} spectra written to {output_dir}/")


def _predict_auger_stick(sing_model, trip_model, device, data_list, mol_names,
                          *, cfg, output_dir, file_stem):
    """Run auger-gnn stick inference and write broadened per-molecule spectra.

    Singlet and triplet intensities are summed onto a common energy grid
    using the fwhm from cfg.
    """
    from torch_geometric.loader import DataLoader
    from augernet.spec_utils import fit_spectrum_to_grid

    print(f"\n{'=' * 80}")
    print(f"  PREDICT: Running Auger stick inference on {len(data_list)} molecules")
    print(f"  FWHM: {cfg.fwhm} eV  |  Grid: [{cfg.min_ke}, {cfg.max_ke}] eV")
    print(f"{'=' * 80}")

    energy_grid = np.linspace(cfg.min_ke, cfg.max_ke, cfg.n_points)
    loader = DataLoader(data_list, batch_size=1, shuffle=False)

    # Load auger norm stats for energy de-normalisation
    auger_norm_stats = torch.load(cfg.auger_norm_stats_file, weights_only=False)
    maxE = auger_norm_stats['maxE']
    max_spec_len = cfg.max_spec_len

    def _run_model(model, data_list, device):
        """Return {mol_idx: array(n_valid_atoms, 2*max_spec_len)} raw outputs."""
        loader = DataLoader(data_list, batch_size=1, shuffle=False)
        outputs = {}
        model.eval()
        with torch.no_grad():
            for mol_idx, d in enumerate(loader):
                d = d.to(device)
                out = model(d)
                if getattr(model, 'task_type', 'single') == 'multi':
                    out = out[1]
                node_mask = d.node_mask.squeeze()
                valid = node_mask.nonzero(as_tuple=True)[0]
                outputs[mol_idx] = {
                    int(ai): out[ai].cpu().numpy() for ai in valid
                }
        return outputs

    sing_out = _run_model(sing_model, data_list, device)
    trip_out = _run_model(trip_model, data_list, device) if trip_model else {}

    spectra = {}
    for mol_idx, mol_name in enumerate(mol_names):
        mol_spectrum = np.zeros(cfg.n_points)
        for ai, s_vec in sing_out.get(mol_idx, {}).items():
            # s_vec shape: (2*max_spec_len,) = [energies_norm, intensities]
            e_s = s_vec[:max_spec_len] * maxE
            i_s = s_vec[max_spec_len:]
            # add ke_shift to align calculated → experimental frame
            e_s += cfg.ke_shift_calc
            _, broadened = fit_spectrum_to_grid(
                e_s, i_s, fwhm=cfg.fwhm,
                energy_min=cfg.min_ke, energy_max=cfg.max_ke,
                n_points=cfg.n_points,
            )
            mol_spectrum += broadened

        for ai, t_vec in trip_out.get(mol_idx, {}).items():
            e_t = t_vec[:max_spec_len] * maxE + cfg.ke_shift_calc
            i_t = t_vec[max_spec_len:]
            _, broadened = fit_spectrum_to_grid(
                e_t, i_t, fwhm=cfg.fwhm,
                energy_min=cfg.min_ke, energy_max=cfg.max_ke,
                n_points=cfg.n_points,
            )
            mol_spectrum += broadened

        if mol_spectrum.max() > 0:
            mol_spectrum /= mol_spectrum.max()
        spectra[mol_name] = mol_spectrum

    print(f"\n  Writing spectra to {output_dir}/")
    for mol_name, spectrum in spectra.items():
        out_path = os.path.join(output_dir, f"{file_stem}_{mol_name}_spectrum.txt")
        np.savetxt(out_path,
                   np.column_stack([energy_grid, spectrum]),
                   header=f"energy_eV  intensity  (model={cfg.model_id}, fwhm={cfg.fwhm})",
                   fmt="%.6f")

    print(f"\n{'Molecule':<22s} {'Peak KE (eV)':>14s}")
    print("-" * 38)
    for mol_name, spectrum in spectra.items():
        peak_ke = float(energy_grid[np.argmax(spectrum)]) if spectrum.max() > 0 else float('nan')
        print(f"{mol_name:<22s} {peak_ke:>14.2f}")

    print(f"\n  {len(spectra)} spectra written to {output_dir}/")
