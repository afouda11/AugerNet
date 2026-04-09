"""
Auger-CNN Backend — model-specific routines for train_driver.py
=============================================================

Carbon environment classification using 1D CNN on Auger spectra.

Provides the same routine signatures as ``backend_gnn.py``:
  load_data, train_single_run, load_saved_model,
  run_evaluation, run_unit_tests, run_predict

Dependencies:
  augernet.cnn_train_utils   — AugerCNN1D, CNNTrainer, etc.
  augernet.carbon_dataframe  — CarbonDataset, load_carbon_dataframe
  augernet.class_merging     — apply_label_merging, get_num_classes, etc.
"""

from __future__ import annotations

import os
import numpy as np
import torch
import pandas as pd
from typing import Any, Dict

from sklearn.model_selection import KFold

from augernet import cnn_train_utils as ctu
from augernet.cnn_train_utils import AugerCNN1D
from augernet import carbon_dataframe as cdf
from augernet.class_merging import (
    apply_label_merging,
    get_num_classes,
    print_scheme_summary,
)

from augernet import DATA_PROCESSED_DIR


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mol_kfold_split(carbon_df, *, n_folds, fold, random_state=42,
                     verbose=True):
    """Random molecule-level k-fold split for a carbon DataFrame.

    Splits by unique ``mol_name`` so that all carbon atoms from the same
    molecule stay in the same fold (prevents data leakage).

    Parameters
    ----------
    carbon_df : pd.DataFrame
        One row per carbon atom; must have a ``mol_name`` column.
    n_folds, fold : int
        Standard CV parameters.  ``fold`` is **1-indexed**.
    random_state : int
        Seed for reproducibility.
    verbose : bool
        Print a one-line summary.

    Returns
    -------
    (train_row_idx, val_row_idx) : Tuple[List[int], List[int]]
        Carbon-row indices into the DataFrame.
    """
    # Ordered unique molecules
    seen: dict = {}
    for name in carbon_df['mol_name']:
        if name not in seen:
            seen[name] = len(seen)
    mol_names = list(seen.keys())

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = list(kf.split(np.arange(len(mol_names))))
    mol_train_idx, mol_val_idx = folds[fold - 1]

    # Map molecule indices → carbon-row indices
    train_mols = {mol_names[i] for i in mol_train_idx}
    val_mols = {mol_names[i] for i in mol_val_idx}

    train_rows, val_rows = [], []
    for row_idx, name in enumerate(carbon_df['mol_name']):
        if name in train_mols:
            train_rows.append(row_idx)
        elif name in val_mols:
            val_rows.append(row_idx)

    if verbose:
        print(f"  Random molecule split (fold {fold}/{n_folds}): "
              f"{len(mol_train_idx)} train / {len(mol_val_idx)} val molecules")
        print(f"  Carbon rows: {len(train_rows)} train / {len(val_rows)} val")

    return train_rows, val_rows


def _get_input_length(df, cfg, *, use_augmented=None):
    """Determine CNN input_length from config (spectra are always broadened).

    Optional keyword arg overrides ``cfg`` value — needed when called from
    param search where each config may toggle ``use_augmented``.
    """
    n_spec = getattr(cfg, 'n_points', 731)
    use_aug = use_augmented if use_augmented is not None else getattr(cfg, 'use_augmented', False)
    return n_spec + (1 if use_aug else 0)


def _resolve_architecture(cfg, overrides=None):
    """Resolve architecture dict from cfg or overrides.

    Priority: overrides['architecture'] > cfg.architecture > recommended preset.
    """
    overrides = overrides or {}

    arch = overrides.get('architecture') or getattr(cfg, 'architecture', None)
    if arch is None or not arch:
        arch = ctu.ARCHITECTURE_PRESETS.get('recommended',
                                            ctu.ARCHITECTURE_PRESETS.get('legacy_3block'))
    return arch


def _resolve_num_classes(cfg, merge_scheme_override=None):
    """Determine number of output classes based on merge scheme."""
    merge_scheme = merge_scheme_override or getattr(cfg, 'merge_scheme', 'none')
    if merge_scheme != 'none':
        return get_num_classes(merge_scheme)
    return ctu.NUM_CARBON_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(cfg) -> Dict[str, Any]:
    """Load CNN training data (+ path for eval).

    Always stores the *unmerged* DataFrame as ``'train_df_raw'`` so that
    param search can re-apply different merge schemes per config.  The
    ``'train_df'`` entry has the base ``cfg.merge_scheme`` already applied
    (used by cv / train modes that don't change the scheme).
    """
    train_data = os.path.join(DATA_PROCESSED_DIR, 'cnn_auger_calc.pkl')
    eval_data  = os.path.join(DATA_PROCESSED_DIR, 'cnn_auger_eval.pkl')

    print(f"\nLoading training data: {train_data}")
    train_df = cdf.load_carbon_dataframe(train_data)
    ctu.diagnose_dataframe(train_df)

    # Keep unmerged copy for param search
    train_df_raw = train_df.copy()

    merge_scheme = getattr(cfg, 'merge_scheme', 'none')
    if merge_scheme != 'none':
        print_scheme_summary(merge_scheme)
        train_df = apply_label_merging(train_df, merge_scheme)

    return {
        'train_df': train_df,
        'train_df_raw': train_df_raw,
        'eval_data_path': eval_data,
    }


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
    """Train CNN on a single fold.

    Parameters
    ----------
    save_paths : dict
        Pre-built mapping ``{'model': '/abs/path/to/file.pth'}``.
        Built by ``train_driver._build_save_paths``.
    """
    from torch.utils.data import DataLoader, Subset

    # ── Resolve hyper-parameters from cfg + overrides ─────────────────────
    _g = lambda k, d=None: overrides.get(k, getattr(cfg, k, d))

    architecture     = _resolve_architecture(cfg, overrides)
    num_epochs       = _g('num_epochs', 500)
    patience         = _g('patience', 40)
    batch_size       = _g('batch_size', 64)
    learning_rate    = _g('learning_rate', 3e-4)
    weight_decay     = _g('weight_decay', 1e-4)
    use_augmented    = _g('use_augmented', False)
    device_str       = _g('device', 'auto')
    random_seed      = _g('random_seed', 42)
    scheduler_type   = _g('scheduler_type', 'cosine')
    broadening_fwhm  = _g('fwhm', 1.6)
    energy_min       = _g('min_ke', 200.0)
    energy_max       = _g('max_ke', 273.0)
    n_spectrum_points = _g('n_points', 731)
    merge_scheme     = _g('merge_scheme', 'none')

    # ── Resolve training DataFrame (re-merge if scheme differs) ──────────
    base_merge = getattr(cfg, 'merge_scheme', 'none')
    if merge_scheme != base_merge and 'train_df_raw' in data:
        # Param search is testing a different merge scheme → re-merge
        train_df = data['train_df_raw'].copy()
        if merge_scheme != 'none':
            train_df = apply_label_merging(train_df, merge_scheme)
    else:
        train_df = data['train_df']

    ctu.seed(random_seed)
    device = ctu.get_device(device_str, verbose=verbose)

    # Ensure all target directories exist
    for p in save_paths.values():
        os.makedirs(os.path.dirname(p), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    num_classes = _resolve_num_classes(cfg, merge_scheme_override=merge_scheme)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"AUGER CNN TRAINING — Fold {fold}/{n_folds}")
        print(f"{'=' * 70}")
        print(f"  Arch:        {architecture}")
        print(f"  LR:          {learning_rate}")
        print(f"  Batch:       {batch_size}")
        print(f"  Classes:     {num_classes}  (merge={merge_scheme})")
        print(f"  FWHM:        {broadening_fwhm} eV")
        print(f"  Augmented:   {use_augmented}")
        print(f"{'=' * 70}")

    # ── Dataset ───────────────────────────────────────────────────────────
    base_dataset = cdf.CarbonDataset(
        train_df,
        include_augmentation=use_augmented,
        broadening_fwhm=broadening_fwhm,
        energy_min=energy_min, energy_max=energy_max,
        n_points=n_spectrum_points,
    )
    dataset = ctu.CarbonLabelDataset(base_dataset, train_df)

    # ── Molecule-level split ──────────────────────────────────────────────
    train_idx, val_idx = _mol_kfold_split(
        train_df, n_folds=n_folds, fold=fold,
        random_state=random_seed, verbose=verbose,
    )

    train_loader = DataLoader(Subset(dataset, train_idx),
                              batch_size=batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx),
                            batch_size=batch_size, shuffle=False,
                            num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    input_length = _get_input_length(train_df, cfg,
                                     use_augmented=use_augmented)
    model = AugerCNN1D(input_length, num_classes, architecture)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Input length: {input_length}  |  Parameters: {n_params:,}")

    # ── Class weights + trainer ───────────────────────────────────────────
    class_weights, _ = ctu.get_class_weights_and_counts(
        train_df, num_classes=num_classes,
    )

    trainer = ctu.CNNTrainer(
        model=model, device=device,
        learning_rate=learning_rate, weight_decay=weight_decay,
        patience=patience,
        scheduler_type=scheduler_type,
        cosine_T_max=num_epochs,
        class_weights=class_weights,
    )

    if verbose:
        print("\nStarting training...")
    history = trainer.fit(train_loader, val_loader,
                          num_epochs=num_epochs, verbose=verbose)

    # ── Save model + history ──────────────────────────────────────────────
    model_path = save_paths['model']
    torch.save(model.state_dict(), model_path)
    if verbose:
        print(f"\n Saved model to {model_path}")

    hist_df = pd.DataFrame(history)
    hist_path = os.path.join(output_dir,
                             f'training_history_fold{fold}.csv')
    hist_df.to_csv(hist_path, index=False)

    ctu.plot_training_history(history, output_dir)
    generic_plot = os.path.join(output_dir, 'training_plots.png')
    fold_plot = os.path.join(output_dir,
                             f'training_plots_fold{fold}.png')
    if os.path.exists(generic_plot):
        os.replace(generic_plot, fold_plot)

    # ── Results ───────────────────────────────────────────────────────────
    best_val_loss   = min(history['val_loss'])
    best_val_acc    = max(history['val_acc'])
    final_train_acc = history['train_acc'][-1]
    final_val_acc   = history['val_acc'][-1]
    n_epochs_run    = len(history['train_loss'])

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"FOLD {fold} COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Epochs run:    {n_epochs_run}")
        print(f"  Final Train:   {final_train_acc:.2f}%")
        print(f"  Final Val:     {final_val_acc:.2f}%")
        print(f"  Best Val Loss: {best_val_loss:.4f}")
        print(f"  Best Val Acc:  {best_val_acc:.2f}%")

    return {
        'model': model,
        'device': device,
        'fold': fold,
        'best_val_loss': best_val_loss,
        'combined_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'final_train_acc': final_train_acc,
        'final_val_acc': final_val_acc,
        'n_epochs': n_epochs_run,
        'model_path': model_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_model_from_path(model_path, data, cfg, *, architecture=None,
                          merge_scheme=None, use_augmented=None):
    """Load a CNN model from a ``.pth`` file.

    Reconstructs the model architecture from config, loads state dict,
    and returns ``(model, device)``.
    """
    train_df = data.get('train_df_raw', data['train_df'])
    input_length = _get_input_length(train_df, cfg,
                                     use_augmented=use_augmented)
    ms = merge_scheme or getattr(cfg, 'merge_scheme', 'none')
    arch = architecture or _resolve_architecture(cfg)
    device_str = getattr(cfg, 'device', 'auto')

    device = ctu.get_device(device_str, verbose=True)
    num_classes = _resolve_num_classes(cfg, merge_scheme_override=ms)

    model = AugerCNN1D(input_length, num_classes, arch)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model from {model_path}  ({n_params:,} params)")
    return model, device


def load_saved_model(save_paths, data, cfg):
    """Load a saved CNN model from pre-built paths.

    Parameters
    ----------
    save_paths : dict
        Mapping ``{'model': '/abs/path/to/file.pth'}``, as produced
        by ``train_driver._build_save_paths``.
    """
    model_path = save_paths['model']

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved model found:\n  {model_path}"
        )

    return _load_model_from_path(model_path, data, cfg)


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation  (placeholder — full implementation not included)
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(model_result, data, fold, output_dir, png_dir, cfg,
                   train_results=None, **_extra):
    """Evaluate a trained CNN model.

    .. note::
       Full evaluation (per-molecule detail, experimental data comparison)
       is not included in this release.  This stub prints a reminder.
    """
    print("  ⚠ CNN evaluation is not included in this release.")


# ─────────────────────────────────────────────────────────────────────────────
#  Unit tests & predict  (CNN has no unit tests; predict not yet implemented)
# ─────────────────────────────────────────────────────────────────────────────

def run_unit_tests(model, data, cfg):
    """No unit tests for CNN model."""
    print("  (no unit tests for CNN model)")


def run_predict(*, model_path: str, predict_dir: str, fold, cfg):
    """Predict mode is not yet implemented for CNN."""
    raise NotImplementedError(
        "Predict mode is not yet implemented for model 'auger-cnn'."
    )