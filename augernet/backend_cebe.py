"""
CEBE-GNN Backend — model-specific hooks for train_driver.py
============================================================

Provides the five hooks that train_driver needs:
  load_data, train_single_run, load_saved_model,
  load_param_model, run_evaluation, run_unit_tests
"""

from __future__ import annotations

import os
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple
from sklearn.model_selection import KFold

from augernet import gnn_train_utils as gtu
from augernet.feature_assembly import (
    assemble_dataset, compute_feature_tag, describe_features,
)

from augernet import PROJECT_ROOT, DATA_DIR, DATA_PROCESSED_DIR


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(cfg) -> Dict[str, Any]:
    """Load CEBE calculated + experimental data, assemble features."""
    print(f"\nLoading training data from: {DATA_PROCESSED_DIR}")
    print(f"Feature keys: {cfg.feature_keys}  ({describe_features(cfg.feature_keys)})")
    print(f"Feature tag:  {cfg.feature_tag}")
    print(f"Model tag:    {cfg.model_tag}")
    print(f"Feature scale: {cfg.feature_scale}")

    ds = gtu.LoadDataset(DATA_DIR, file_name='gnn_calc_cebe_data.pt')
    calc_data = [ds[i] for i in range(len(ds))]
    print(f"Loaded feature-store data: {len(calc_data)} molecules")
    print(f"Assembling features {cfg.feature_keys} with scale={cfg.feature_scale}...")
    assemble_dataset(calc_data, cfg.feature_keys, scale=cfg.feature_scale)
    print(f"Calculated data: {len(calc_data)} molecules, "
          f"x.shape[1]={calc_data[0].x.size(1)}")

    # Experimental data
    exp_ds = gtu.LoadDataset(DATA_DIR, file_name='gnn_exp_cebe_data.pt')
    exp_data = [exp_ds[i] for i in range(len(exp_ds))]
    assemble_dataset(exp_data, cfg.feature_keys, scale=cfg.feature_scale)

    exp_list_path = os.path.join(cfg.exp_dir, 'mol_list.txt')
    with open(exp_list_path) as f:
        exp_list = [line.strip() for line in f]

    return {
        'calc_data': calc_data,
        'exp_data': exp_data,
        'exp_list': exp_list,
    }


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
    """
    Train a single CEBE GNN on one fold.

    Returns dict with keys: model, device, fold, best_val_loss, n_epochs,
    model_path, combined_val_loss, train_results
    """
    # Merge cfg with any per-config overrides (from param search)
    layer_type      = overrides.get('layer_type',      cfg.layer_type)
    hidden_channels = overrides.get('hidden_channels', cfg.hidden_channels)
    n_layers        = overrides.get('n_layers',        cfg.n_layers)
    num_epochs      = overrides.get('num_epochs',      cfg.num_epochs)
    patience        = overrides.get('patience',        cfg.patience)
    batch_size      = overrides.get('batch_size',      cfg.batch_size)
    learning_rate   = overrides.get('learning_rate',   cfg.learning_rate)
    random_seed     = overrides.get('random_seed',     cfg.random_seed)
    split_method    = overrides.get('split_method',    cfg.split_method)
    optimizer_type  = overrides.get('optimizer_type',  cfg.optimizer_type)
    weight_decay    = overrides.get('weight_decay',    cfg.weight_decay)
    gradient_clip_norm = overrides.get('gradient_clip_norm', cfg.gradient_clip_norm)
    warmup_epochs   = overrides.get('warmup_epochs',   cfg.warmup_epochs)
    min_lr          = overrides.get('min_lr',          cfg.min_lr)

    calc_data = data['calc_data']

    gtu.seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(save_dir,   exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    in_channels = calc_data[0].x.size(1)
    edge_dim = calc_data[0].edge_attr.size(1)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"CEBE MODEL TRAINING — Fold {fold}/{n_folds}")
        print(f"{'=' * 70}")
        print(f"  Layer type:  {layer_type}")
        print(f"  Hidden:      {hidden_channels}")
        print(f"  Layers:      {n_layers}")
        print(f"  LR:          {learning_rate}")
        print(f"  Batch:       {batch_size}")
        print(f"  Split:       {split_method}")
        print(f"{'=' * 70}")

    n_molecules = len(calc_data)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    folds = list(kf.split(np.arange(n_molecules)))
    train_idx, val_idx = folds[fold - 1]  # fold is 1-indexed
    train_idx = train_idx.tolist() if hasattr(train_idx, 'tolist') else list(train_idx)
    val_idx = val_idx.tolist() if hasattr(val_idx, 'tolist') else list(val_idx)

    train_data = [calc_data[i] for i in train_idx]
    val_data   = [calc_data[i] for i in val_idx]

    if verbose:
        print(f"  Split → {len(train_data)} train / {len(val_data)} val molecules")

    # ── build model ──────────────────────────────────────────────────────
    model = gtu.MPNN(
        num_layers=n_layers, emb_dim=hidden_channels,
        in_dim=in_channels, edge_dim=edge_dim,
        out_dim=1, layer_type=layer_type, pred_type='CEBE',
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  Parameters:  {n_params:,}")

    # ── train ────────────────────────────────────────────────────────────
    train_results = gtu.train_loop(
        train_data, model, device,
        num_epochs=num_epochs, batch_size=batch_size,
        max_lr=learning_rate,
        verbose=verbose, layer_type=layer_type, pred_type='CEBE',
        val_data_list=val_data, patience=patience,
        optimizer_type=optimizer_type, weight_decay=weight_decay,
        gradient_clip_norm=gradient_clip_norm,
        warmup_epochs=warmup_epochs, min_lr=min_lr,
    )
    model.eval()

    # ── save ─────────────────────────────────────────────────────────────
    model_filename = (f"cebe_model_fold{fold}_{cfg.model_tag}"
                      f"_{layer_type}_{n_layers}.pth")
    model_path = os.path.join(save_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    if verbose:
        print(f"\n✓ Saved model  → {model_path}")

    # ── results ──────────────────────────────────────────────────────────
    train_losses = [r[1] for r in train_results]
    val_losses   = [r[2] for r in train_results]
    best_val_loss = min(val_losses)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"FOLD {fold} COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Epochs run:    {len(train_results)}")
        print(f"  Best Val Loss: {best_val_loss:.6f}")

    return {
        'model': model,
        'device': device,
        'fold': fold,
        'best_val_loss': best_val_loss,
        'combined_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'n_epochs': len(train_results),
        'model_path': model_path,
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
) -> Tuple[torch.nn.Module, torch.device]:
    """Load a CEBE GNN from a .pth file (thin wrapper around evaluate script)."""
    # Import here to avoid circular imports at module load time
    import sys
    script_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'cebe_pred',
    )
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from evaluate_cebe_model import load_model as load_eval_model

    in_channels = calc_data[0].x.size(1)
    edge_dim = calc_data[0].edge_attr.size(1)
    return load_eval_model(
        model_path, in_channels=in_channels, edge_dim=edge_dim,
        layer_type=layer_type, hidden_channels=hidden_channels,
        n_layers=n_layers,
    )


def load_saved_model(save_dir, fold, data, cfg):
    """Load a saved CEBE model by fold number."""
    calc_data = data['calc_data']
    model_filename = (f"cebe_model_fold{fold}_{cfg.model_tag}"
                      f"_{cfg.layer_type}_{cfg.n_layers}.pth")
    model_path = os.path.join(save_dir, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved model '{model_filename}' found in:\n  {save_dir}"
        )
    return _load_model_from_path(
        model_path, calc_data,
        layer_type=cfg.layer_type,
        hidden_channels=cfg.hidden_channels,
        n_layers=cfg.n_layers,
    )


def load_param_model(model_path, data, cfg, best_params):
    """Load a CEBE model from a param-search result path."""
    calc_data = data['calc_data']
    return _load_model_from_path(
        model_path, calc_data,
        layer_type=cfg.layer_type,
        hidden_channels=best_params.get('hidden_channels', cfg.hidden_channels),
        n_layers=best_params.get('n_layers', cfg.n_layers),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(model_result, data, fold, output_dir, png_dir, cfg,
                   train_results=None):
    """Run CEBE evaluation (calls evaluate_cebe_model.run_evaluation)."""

    from .evaluation_scripts.evaluate_cebe_model import run_evaluation as _run_eval

    if isinstance(model_result, dict):
        model = model_result['model']
        device = model_result['device']
        train_results = model_result.get('train_results', train_results)
    elif isinstance(model_result, tuple):
        model, device = model_result
    else:
        model = model_result
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _run_eval(
        model, device, data['exp_data'], data['exp_list'],
        output_dir=output_dir, fold=fold,
        png_dir=png_dir,
        train_results=train_results,
        out_scale=cfg.out_scale,
        norm_stats_file=cfg.norm_stats_file,
        feature_tag=cfg.model_tag,
        n_layers=cfg.n_layers,
        layer_type=cfg.layer_type,
        exp_dir=cfg.exp_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Unit tests
# ─────────────────────────────────────────────────────────────────────────────

def run_unit_tests(model, data, cfg):
    """Run GNN symmetry tests on CEBE model."""
    if isinstance(model, dict):
        model = model['model']
    model.eval()
    gtu.run_unit_tests(model, data['calc_data'], layer_type=cfg.layer_type)
