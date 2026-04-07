    """
Unified GNN Backend — model-specific hooks for train_driver.py
===============================================================

Supports both model types from a single module:
  - ``cebe-gnn``   : CEBE binding-energy prediction (scalar, 1 model/fold)
  - ``auger-gnn``  : Auger spectrum prediction
      - ``stick``   : separate singlet + triplet stick spectra (2 models/fold)
      - ``fitted``  : combined broadened spectrum (1 model/fold)

Provides the hooks that train_driver needs:
  load_data, train_single_run, load_saved_model,
  load_param_model, run_evaluation, run_unit_tests, run_predict
"""

from __future__ import annotations

import os
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple
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

    Returns a flat dict of all HP values and pops ``param_file_prefix``
    from *overrides* so it does not leak into downstream calls.
    """
    hp = {
        'layer_type':       overrides.get('layer_type',       cfg.layer_type),
        'hidden_channels':  overrides.get('hidden_channels',  cfg.hidden_channels),
        'n_layers':         overrides.get('n_layers',         cfg.n_layers),
        'num_epochs':       overrides.get('num_epochs',       cfg.num_epochs),
        'patience':         overrides.get('patience',         cfg.patience),
        'batch_size':       overrides.get('batch_size',       cfg.batch_size),
        'learning_rate':    overrides.get('learning_rate',    cfg.learning_rate),
        'random_seed':      overrides.get('random_seed',      cfg.random_seed),
        'split_method':     overrides.get('split_method',     cfg.split_method),
        'optimizer_type':   overrides.get('optimizer_type',   cfg.optimizer_type),
        'weight_decay':     overrides.get('weight_decay',     cfg.weight_decay),
        'gradient_clip_norm': overrides.get('gradient_clip_norm', cfg.gradient_clip_norm),
        'warmup_epochs':    overrides.get('warmup_epochs',    cfg.warmup_epochs),
        'min_lr':           overrides.get('min_lr',           cfg.min_lr),
        'dropout':          overrides.get('dropout',          cfg.dropout),
    }
    hp['param_file_prefix'] = overrides.pop('param_file_prefix', None)
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
    """Re-assemble features if param search overrides feature_keys.

    Returns the resolved feature keys string.
    """
    override_fk = overrides.get('feature_keys')
    if override_fk is not None:
        fk_parsed = parse_feature_keys(override_fk)
        fk_tag = compute_feature_tag(fk_parsed)
        if fk_tag != data.get('assembled_feature_keys', cfg.feature_keys):
            print(f"  Re-assembling features for key override: {fk_tag}")
            assemble_dataset(data['calc_data'], fk_parsed)
            if data.get('exp_data'):
                assemble_dataset(data['exp_data'], fk_parsed)
            data['assembled_feature_keys'] = fk_tag
        return fk_tag
    return cfg.feature_keys


def _build_model_id(cfg, hp, fk_tag):
    """Build the per-config model_id string."""
    prefix = 'cebe_gnn' if cfg.model == 'cebe-gnn' else 'auger_gnn'
    return (
        f"{prefix}_{fk_tag}_{hp['split_method']}"
        f"_{hp['layer_type']}{hp['n_layers']}_h{hp['hidden_channels']}"
    )


def _train_one_model(train_data, val_data, in_channels, edge_dim, device, hp,
                     pred_type='CEBE', spectrum_type='stick', spectrum_dim=300):
    """Build, train, and return a single MPNN model + train_results."""
    model = gtu.MPNN(
        num_layers=hp['n_layers'], emb_dim=hp['hidden_channels'],
        in_dim=in_channels, edge_dim=edge_dim,
        out_dim=1, layer_type=hp['layer_type'], pred_type=pred_type,
        dropout=hp['dropout'],
        spectrum_type=spectrum_type, spectrum_dim=spectrum_dim,
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
    )
    if pred_type == 'AUGER':
        loop_kwargs['spectrum_type'] = spectrum_type

    train_results = gtu.train_loop(train_data, model, device, **loop_kwargs)
    model.eval()
    return model, train_results


def _save_model(model, model_id, fold, save_dir, param_file_prefix=None,
                tag=None):
    """Save model state dict and return the path."""
    if tag:
        filename = f"{tag}_{model_id}_fold{fold}.pth"
    else:
        filename = f"{model_id}_fold{fold}.pth"
    if param_file_prefix:
        filename = f"{param_file_prefix}_{filename}"
    path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), path)
    print(f" Saved model from {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  CEBE experimental data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_exp_split_names(split):
    """Load mol-name sets for the experimental val/eval split.

    Reads ``mol_list_val.txt`` and ``mol_list_eval.txt`` from the raw
    exp_cebe directory.  Returns ``(val_names, eval_names)`` as sets.
    """
    from augernet import DATA_RAW_DIR
    exp_dir = os.path.join(DATA_RAW_DIR, 'exp_cebe')

    def _read(fname):
        path = os.path.join(exp_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Exp split file not found: {path}\n"
                f"Run  generate_exp_split.py  to create it."
            )
        with open(path) as f:
            return {line.strip() for line in f if line.strip()}

    return _read('mol_list_val.txt'), _read('mol_list_eval.txt')


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
        print(f"  ⚠ Val split: {len(missing_val)} names not found in data: "
              f"{sorted(missing_val)[:5]}")
    if missing_eval:
        print(f"  ⚠ Eval split: {len(missing_eval)} names not found in data: "
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
#  On-the-fly stick → fitted conversion
# ─────────────────────────────────────────────────────────────────────────────

def _attach_y_fitted(sing_data, trip_data, cfg):
    """Create ``y_fitted`` on each singlet Data object by combining singlet +
    triplet stick spectra and Gaussian-broadening onto a common energy grid.

    Each atom's ``y`` is a 600-vector = [energies(300), intensities(300)].
    Energies are normalised by ``max_ke``; intensities by per-atom max.
    ``spec_len`` gives the number of valid entries in each half.

    After this call every element in *sing_data* has a new attribute
    ``y_fitted`` of shape ``(n_atoms, cfg.n_points)``.
    """
    max_ke = cfg.max_ke
    for s_data, t_data in zip(sing_data, trip_data):
        n_atoms = s_data.x.size(0)
        s_y = s_data.y.view(n_atoms, 600).numpy()
        t_y = t_data.y.view(n_atoms, 600).numpy()
        s_sl = int(s_data.spec_len.item())
        t_sl = int(t_data.spec_len.item())
        fitted = np.zeros((n_atoms, cfg.n_points), dtype=np.float32)

        for ai in s_data.node_mask.nonzero(as_tuple=True)[0].tolist():
            # Un-normalise energies and concatenate singlet + triplet sticks
            e = np.concatenate([s_y[ai, :s_sl] * max_ke,
                                t_y[ai, :t_sl] * max_ke])
            inten = np.concatenate([s_y[ai, 300:300 + s_sl],
                                    t_y[ai, 300:300 + t_sl]])
            _, fitted[ai] = fit_spectrum_to_grid(
                e, inten, fwhm=cfg.fwhm,
                energy_min=cfg.min_ke, energy_max=cfg.max_ke,
                n_points=cfg.n_points,
            )
        s_data.y_fitted = torch.tensor(fitted, dtype=torch.float32)
    print(f"  ✓ Built y_fitted on-the-fly ({cfg.n_points}-pt grid, "
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
        }

    # ── Auger-GNN ────────────────────────────────────────────────────────
    elif cfg.model == 'auger-gnn':
        spec_type = cfg.spectrum_type

        # Both singlet and triplet are always loaded
        sing_ds = gtu.LoadDataset(DATA_DIR, file_name='gnn_calc_auger_sing_data.pt')
        sing_calc_data = [sing_ds[i] for i in range(len(sing_ds))]
        trip_ds = gtu.LoadDataset(DATA_DIR, file_name='gnn_calc_auger_trip_data.pt')
        trip_calc_data = [trip_ds[i] for i in range(len(trip_ds))]

        # Reshape y from (n_atoms*600, 1) → (n_atoms, 600) so that PyG
        # DataLoader batching concatenates correctly along the atom axis.
        for dlist in (sing_calc_data, trip_calc_data):
            for d in dlist:
                n_atoms = d.x.size(0)
                d.y = d.y.view(n_atoms, 600)

        print(f"  Loaded {len(sing_calc_data)} singlet, "
              f"{len(trip_calc_data)} triplet molecules ({spec_type})")

        if spec_type == 'stick':
            print(f"  Assembling features {cfg.feature_keys}")
            assemble_dataset(sing_calc_data, feature_keys)
            assemble_dataset(trip_calc_data, feature_keys)
            print(f"  x.shape[1]={sing_calc_data[0].x.size(1)}")

            return {
                'calc_data': sing_calc_data,  # used for splitting
                'sing_calc_data': sing_calc_data,
                'trip_calc_data': trip_calc_data,
                'assembled_feature_keys': cfg.feature_keys,
            }
        else:  # fitted — build y_fitted on-the-fly from stick data
            _attach_y_fitted(sing_calc_data, trip_calc_data, cfg)
            print(f"  Assembling features {cfg.feature_keys}")
            assemble_dataset(sing_calc_data, feature_keys)
            print(f"  x.shape[1]={sing_calc_data[0].x.size(1)}, "
                  f"y_fitted.shape={sing_calc_data[0].y_fitted.shape}")

            return {
                'calc_data': sing_calc_data,
                'assembled_feature_keys': cfg.feature_keys,
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
    save_dir: str,
    output_dir: str,
    cfg,
    verbose: bool = True,
    **overrides,
) -> Dict[str, Any]:
    """Train a single GNN model (or model pair) on one fold.

    Works for CEBE-GNN, Auger-GNN stick, and Auger-GNN fitted.
    Returns a result dict compatible with train_driver expectations.
    """
    hp = _extract_overrides(cfg, overrides)
    fk_tag = _handle_feature_override(data, cfg, overrides)
    model_id = _build_model_id(cfg, hp, fk_tag)

    gtu.seed(hp['random_seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir,   exist_ok=True)
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
        print(f"  Layer type:  {hp['layer_type']}")
        print(f"  Hidden:      {hp['hidden_channels']}")
        print(f"  Layers:      {hp['n_layers']}")
        print(f"  LR:          {hp['learning_rate']}")
        print(f"  Dropout:     {hp['dropout']}")
        print(f"  Batch:       {hp['batch_size']}")
        print(f"  Split method:       {hp['split_method']}")
        print(f"  Split:       {len(train_idx)} train / {len(val_idx)} val molecules")
        print(f"{'=' * 70}")

    in_channels = calc_data[0].x.size(1)
    edge_dim = calc_data[0].edge_attr.size(1)

    # ── Dispatch to model-specific training ──────────────────────────────
    if cfg.model == 'cebe-gnn':
        return _train_cebe(
            data, train_idx, val_idx, in_channels, edge_dim,
            device, hp, model_id, fold, save_dir, verbose,
        )
    elif cfg.spectrum_type == 'stick':
        return _train_auger_stick(
            data, train_idx, val_idx, in_channels, edge_dim,
            device, hp, model_id, fold, save_dir, verbose, cfg,
        )
    else:
        return _train_auger_fitted(
            data, train_idx, val_idx, in_channels, edge_dim,
            device, hp, model_id, fold, save_dir, verbose, cfg,
        )


# ── CEBE training ────────────────────────────────────────────────────────────

def _train_cebe(data, train_idx, val_idx, in_channels, edge_dim,
                device, hp, model_id, fold, save_dir, verbose):
    """Train a single CEBE GNN."""
    calc_data = data['calc_data']
    train_data = [calc_data[i] for i in train_idx]
    val_data   = [calc_data[i] for i in val_idx]

    model, train_results = _train_one_model(
        train_data, val_data, in_channels, edge_dim, device, hp,
        pred_type='CEBE',
    )

    model_path = _save_model(model, model_id, fold, save_dir,
                             hp['param_file_prefix'])

    bvl, btl, bve, ftl, fvl, n_ep = _extract_results(train_results)

    if verbose:
        print(f"\n  Fold {fold} complete — {n_ep} epochs, "
              f"best val loss {bvl:.6f} (epoch {bve})")

    return {
        'model': model, 'device': device, 'fold': fold,
        'best_val_loss': bvl, 'best_train_loss': btl,
        'best_val_epoch': bve,
        'final_train_loss': ftl, 'final_val_loss': fvl,
        'n_epochs': n_ep, 'model_path': model_path,
        'train_results': train_results, 'model_id': model_id,
    }


# ── Auger stick training (singlet + triplet) ─────────────────────────────────

def _train_auger_stick(data, train_idx, val_idx, in_channels, edge_dim,
                       device, hp, model_id, fold, save_dir, verbose, cfg):
    """Train singlet + triplet GNN models on one fold (stick spectra)."""
    sing_calc = data['sing_calc_data']
    trip_calc = data['trip_calc_data']

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
    )
    sing_path = _save_model(sing_model, model_id, fold, save_dir,
                            hp['param_file_prefix'], tag='singlet')

    # ── Triplet ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}\n  TRIPLET MODEL\n{'─' * 70}")
    trip_model, trip_results = _train_one_model(
        trip_train, trip_val, in_channels, edge_dim, device, hp,
        pred_type='AUGER', spectrum_type='stick',
        spectrum_dim=cfg.max_spec_len,
    )
    trip_path = _save_model(trip_model, model_id, fold, save_dir,
                            hp['param_file_prefix'], tag='triplet')

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
        'model_path': sing_path,
        'sing_model_path': sing_path,
        'trip_model_path': trip_path,
        'train_results': sing_results,  # driver uses for loss curves
        'model_id': model_id,
    }


# ── Auger fitted training (single combined model) ────────────────────────────

def _train_auger_fitted(data, train_idx, val_idx, in_channels, edge_dim,
                        device, hp, model_id, fold, save_dir, verbose, cfg):
    """Train a single fitted-spectrum GNN on one fold."""
    calc_data = data['calc_data']
    train_data = [calc_data[i] for i in train_idx]
    val_data   = [calc_data[i] for i in val_idx]

    model, train_results = _train_one_model(
        train_data, val_data, in_channels, edge_dim, device, hp,
        pred_type='AUGER', spectrum_type='fitted',
        spectrum_dim=cfg.n_points,
    )

    model_path = _save_model(model, model_id, fold, save_dir,
                             hp['param_file_prefix'], tag='fitted')

    bvl, btl, bve, ftl, fvl, n_ep = _extract_results(train_results)

    if verbose:
        print(f"\n  Fold {fold} complete — {n_ep} epochs, "
              f"best val loss {bvl:.6f} (epoch {bve})")

    return {
        'model': model, 'device': device, 'fold': fold,
        'best_val_loss': bvl, 'best_train_loss': btl,
        'best_val_epoch': bve,
        'final_train_loss': ftl, 'final_val_loss': fvl,
        'n_epochs': n_ep, 'model_path': model_path,
        'train_results': train_results, 'model_id': model_id,
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
    """Return the extra kwargs for _load_model_from_path based on model type."""
    if cfg.model == 'cebe-gnn':
        return dict(pred_type='CEBE')
    elif cfg.model == 'auger-gnn':
        kw = dict(pred_type='AUGER', spectrum_type=cfg.spectrum_type)
        if cfg.spectrum_type == 'fitted':
            kw['spectrum_dim'] = cfg.n_points
        else:
            kw['spectrum_dim'] = cfg.max_spec_len
        return kw
    return {}


def load_saved_model(save_dir, fold, data, cfg):
    """Load saved model(s) by fold number.

    For auger-gnn stick mode two tagged models (singlet_ / triplet_) are
    loaded and returned as a dict matching ``train_single_run`` output.
    For auger-gnn fitted a single fitted_ model is loaded.
    For cebe-gnn the bare model_id filename is used.
    """
    calc_data = data['calc_data']
    load_kw = dict(
        layer_type=cfg.layer_type,
        hidden_channels=cfg.hidden_channels,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        **_model_load_kwargs(cfg),
    )

    if cfg.model == 'auger-gnn' and cfg.spectrum_type == 'stick':
        sing_path = os.path.join(save_dir, f"singlet_{cfg.model_id}_fold{fold}.pth")
        trip_path = os.path.join(save_dir, f"triplet_{cfg.model_id}_fold{fold}.pth")
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
            'model_id': cfg.model_id, 'fold': fold,
        }

    elif cfg.model == 'auger-gnn' and cfg.spectrum_type == 'fitted':
        fitted_path = os.path.join(save_dir, f"fitted_{cfg.model_id}_fold{fold}.pth")
        if not os.path.exists(fitted_path):
            raise FileNotFoundError(
                f"No saved fitted model found:\n  {fitted_path}"
            )
        model, device = _load_model_from_path(fitted_path, calc_data, **load_kw)
        return {
            'model': model, 'device': device,
            'model_id': cfg.model_id, 'fold': fold,
        }

    else:  # cebe-gnn — bare filename, no tag
        model_filename = f"{cfg.model_id}_fold{fold}.pth"
        model_path = os.path.join(save_dir, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No saved model '{model_filename}' found in:\n  {save_dir}"
            )
        return _load_model_from_path(
            model_path, calc_data, **load_kw,
        )


def load_param_model(model_path, data, cfg, best_params):
    """Load a model from a param-search result path."""
    calc_data = data['calc_data']

    # If best_params overrode feature_keys, ensure data is assembled correctly
    fk_override = best_params.get('feature_keys')
    if fk_override is not None:
        fk_parsed = parse_feature_keys(fk_override)
        fk_tag = compute_feature_tag(fk_parsed)
        if fk_tag != data.get('assembled_feature_keys', cfg.feature_keys):
            assemble_dataset(calc_data, fk_parsed)
            if data.get('exp_data'):
                assemble_dataset(data['exp_data'], fk_parsed)
            data['assembled_feature_keys'] = fk_tag

    return _load_model_from_path(
        model_path, calc_data,
        layer_type=best_params.get('layer_type', cfg.layer_type),
        hidden_channels=best_params.get('hidden_channels', cfg.hidden_channels),
        n_layers=best_params.get('n_layers', cfg.n_layers),
        dropout=best_params.get('dropout', cfg.dropout),
        **_model_load_kwargs(cfg),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(model_result, data, fold, output_dir, png_dir, cfg,
                   train_results=None, config_id=None, param_file_prefix=None,
                   exp_split=None):
    """Run evaluation after training.

    CEBE-GNN: evaluates on experimental data via evaluate_cebe_model.
    Auger-GNN: evaluates on experimental spectra via evaluate_auger_model.
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

        return _run_auger_eval(
            model_dict, device_a,
            output_dir=output_dir, png_dir=png_dir, cfg=cfg,
            fold=fold, train_results=train_results,
            model_id=model_id, config_id=config_id,
            param_file_prefix=param_file_prefix,
        )

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
            norm_stats_file=cfg.norm_stats_file,
            model_id=model_id,
            config_id=config_id,
            param_file_prefix=pfx or None,
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
        # 'all' or lists not available → full experimental set
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
#  Predict (inference on arbitrary .xyz files) — CEBE only
# ─────────────────────────────────────────────────────────────────────────────

def run_predict(*, model_path: str, predict_dir: str, cfg):

    """Build graphs from .xyz files, run CEBE inference, and write output.

    Currently only supports CEBE-GNN.  Auger predict requires a different
    workflow and will be added separately.
    """
    if cfg.model != 'cebe-gnn':
        raise NotImplementedError(
            f"Predict mode is not yet implemented for model '{cfg.model}'."
        )

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
        raise FileNotFoundError(
            f"No .xyz files found in: {predict_dir}"
        )

    mol_names = [os.path.splitext(f)[0] for f in xyz_files]

    print(f"\n  Predict directory: {predict_dir}")
    print(f"  Found {len(xyz_files)} .xyz files")

    # ── Build graphs ─────────────────────────────────────────────────────
    skipatom_dir = os.path.join(DATA_RAW_DIR, 'skipatom')
    all_encoders = _initialize_all_atom_encoders(skipatom_dir)

    norm_stats = torch.load(cfg.norm_stats_file, weights_only=False)
    mean = norm_stats['mean']
    std  = norm_stats['std']

    category_feature = np.array([1, 0, 0])   # CEBE category
    feature_keys = cfg.feature_keys_parsed
    data_list = []

    print(f"  Building molecular graphs...")
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
            atomic_be=atomic_be,
            atom_symbols=xyz_symbols,
            smiles=smiles,
            mol_name=mol_name,
        )
        for attr_name, tensor in node_features.items():
            setattr(d, attr_name, tensor)
        data_list.append(d)

    print(f"  Assembled {len(data_list)} graphs")

    print(f"  Assembling features {cfg.feature_keys}")
    assemble_dataset(data_list, feature_keys)

    # ── Load model ───────────────────────────────────────────────────────
    model, device = _load_model_from_path(
        model_path, data_list,
        layer_type=cfg.layer_type,
        hidden_channels=cfg.hidden_channels,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    )

    # ── Inference ────────────────────────────────────────────────────────
    output_dir = cfg.outputs_dir

    #file_stem = f"{cfg.model_id}_fold{fold}" if fold is not None else cfg.model_id
    file_stem = cfg.model_id

    print(f"\n{'=' * 80}")
    print(f"  PREDICT: Running inference on {len(data_list)} molecules")
    print(f"  NOTE: Model is trained on carbon 1s CEBEs only.")
    print(f"        Predictions for non-carbon atoms are not meaningful.")
    print(f"{'=' * 80}")

    loader = DataLoader(data_list, batch_size=1, shuffle=False)

    all_pred = []
    all_atoms = []
    molecule_results = {}

    for i, d in enumerate(loader):
        n_nodes = d.x.size(0)
        nodes_in_edges = set(
            d.edge_index[0].tolist() + d.edge_index[1].tolist()
        )
        if len(nodes_in_edges) < n_nodes:
            mol_name_raw = d.mol_name
            if isinstance(mol_name_raw, list):
                mol_name_raw = mol_name_raw[0]
            atom_syms = d.atom_symbols
            if isinstance(atom_syms, list):
                atom_syms = atom_syms[0]
            atom_syms = [str(s).strip() for s in atom_syms]
            mol_rows = [(sym, float('nan')) for sym in atom_syms]
            molecule_results[mol_name_raw] = mol_rows
            all_pred.extend([float('nan')] * n_nodes)
            all_atoms.extend(atom_syms)
            print(f"  Skipping {mol_name_raw}: disconnected graph "
                  f"({len(nodes_in_edges)}/{n_nodes} nodes in edges)")
            continue

        d = d.to(device)
        with torch.no_grad():
            out = model(d)

        pred_out = out.cpu().numpy()
        atomic_be_vals = d.atomic_be.cpu().numpy()

        atom_syms = d.atom_symbols
        if isinstance(atom_syms, list):
            atom_syms = atom_syms[0]
        atom_syms = [str(s).strip() for s in atom_syms]

        mol_name_raw = d.mol_name
        if isinstance(mol_name_raw, list):
            mol_name_raw = mol_name_raw[0]
        mol_rows = []

        for j in range(len(pred_out)):
            sym = atom_syms[j] if j < len(atom_syms) else '?'
            ref = atomic_be_vals[j]
            pred_be = float(ref - (pred_out[j] * std + mean))
            mol_rows.append((sym, pred_be))
            all_pred.append(pred_be)
            all_atoms.append(sym)

        molecule_results[mol_name_raw] = mol_rows

    # ── Write _labels.txt ────────────────────────────────────────────────
    label_path = os.path.join(output_dir, f"{file_stem}_labels.txt")
    with open(label_path, 'w') as f:
        f.write(f"# CEBE Predictions\n")
        f.write(f"# Model: {cfg.model_id}\n")
        f.write(f"# Note: Only carbon (C) predictions are meaningful.\n")
        f.write(f"#       Non-carbon rows are marked with * and should be ignored.\n")
        f.write(f"# Columns: atom_symbol  pred_BE(eV)\n#\n")
        for mol_name, rows in molecule_results.items():
            f.write(f"# --- {mol_name} ---\n")
            for sym, pred_be in rows:
                marker = ' ' if sym == 'C' else '*'
                f.write(f"{sym:>3s}{marker}   {pred_be:10.4f}\n")
            f.write(f"\n")
    print(f"  Label results saved to {label_path}")

    # ── Write _results.txt (numeric, carbon atoms only) ──────────────────
    carbon_preds = [p for s, p in zip(all_atoms, all_pred) if s == 'C']
    results_path = os.path.join(output_dir, f"{file_stem}_results.txt")
    np.savetxt(results_path, np.array(carbon_preds).reshape(-1, 1))
    print(f"  Numeric results saved to {results_path}")

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'Molecule':<22s} {'N_C':>5s} {'N_atoms':>8s} {'Mean C_1s (eV)':>15s}")
    print("-" * 55)
    for mol_name, rows in molecule_results.items():
        c_preds = [p for s, p in rows if s == 'C']
        n_c = len(c_preds)
        n_tot = len(rows)
        mean_c = np.mean(c_preds) if c_preds else float('nan')
        print(f"{mol_name:<22s} {n_c:>5d} {n_tot:>8d} {mean_c:>15.4f}")
