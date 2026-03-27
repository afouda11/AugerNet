"""
Auger-CNN Backend — model-specific hooks for train_driver.py
=============================================================

Carbon environment classification using 1D CNN on Auger spectra.

Provides the same hook signatures as ``backend.py`` (GNN backend):
  load_data, train_single_run, load_saved_model,
  load_param_model, run_evaluation, run_unit_tests, run_predict

Dependencies (to be ported from AUGER-NET-DEV):
  augernet.cnn_train_utils   — AugerCNN1D, UnifiedCNNTrainer, etc.
  augernet.carbon_dataframe  — CarbonDataset, load_carbon_dataframe
  augernet.class_merging     — apply_label_merging, get_num_classes, etc.
  augernet.unified_split     — get_stratified_kfold_split, mol_indices_to_carbon_indices
"""

from __future__ import annotations

import os
import torch
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from augernet import cnn_train_utils as ctu
from augernet.cnn_train_utils import AugerCNN1D
from augernet import carbon_dataframe as cdf
from augernet.unified_split import (
    get_stratified_kfold_split,
    mol_indices_to_carbon_indices,
)
from augernet.class_merging import (
    apply_label_merging,
    get_num_classes,
    get_merged_class_names,
    print_scheme_summary,
)

from augernet import DATA_PROCESSED_DIR


# ─────────────────────────────────────────────────────────────────────────────
#  Architecture catalog  (named variants for param search)
# ─────────────────────────────────────────────────────────────────────────────

ARCH_CATALOG = {
    'current': {
        'conv_filters': [32, 64, 128],
        'conv_kernels': [21, 11, 5],
        'pool_size': 4,
        'fc_hidden': [256, 128],
        'use_batch_norm': True,
        'dropout': 0.3,
        'dropout_conv': 0.1,
    },
    'spectral': {
        'conv_filters': [32, 64, 128, 128],
        'conv_kernels': [41, 21, 11, 7],
        'pool_size': 3,
        'fc_hidden': [256, 128],
        'use_batch_norm': True,
        'dropout': 0.3,
        'dropout_conv': 0.1,
    },
    'spectral_wide': {
        'conv_filters': [32, 64, 128, 128],
        'conv_kernels': [51, 31, 15, 7],
        'pool_size': 3,
        'fc_hidden': [256, 128],
        'use_batch_norm': True,
        'dropout': 0.3,
        'dropout_conv': 0.1,
    },
    'spectral_deep': {
        'conv_filters': [32, 64, 128, 256],
        'conv_kernels': [41, 21, 11, 5],
        'pool_size': 3,
        'fc_hidden': [256, 128],
        'use_batch_norm': True,
        'dropout': 0.3,
        'dropout_conv': 0.1,
    },
    'spectral_light': {
        'conv_filters': [16, 32, 64, 64],
        'conv_kernels': [41, 21, 11, 7],
        'pool_size': 3,
        'fc_hidden': [128, 64],
        'use_batch_norm': True,
        'dropout': 0.4,
        'dropout_conv': 0.15,
    },
    'deep5': {
        'conv_filters': [16, 32, 64, 128, 128],
        'conv_kernels': [31, 21, 15, 11, 5],
        'pool_size': 2,
        'fc_hidden': [256, 128],
        'use_batch_norm': True,
        'dropout': 0.3,
        'dropout_conv': 0.1,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_input_length(df, cfg, *, use_augmented=None, augmented_scaled=None):
    """Determine CNN input_length from config (spectra are always broadened).

    Optional keyword args override ``cfg`` values — needed when called from
    param search where each config may toggle ``use_augmented``.
    """
    n_spec = getattr(cfg, 'n_spectrum_points', 731)
    use_aug = use_augmented if use_augmented is not None else getattr(cfg, 'use_augmented', False)
    aug_scaled = augmented_scaled if augmented_scaled is not None else getattr(cfg, 'augmented_scaled', False)
    return n_spec + (1 if (use_aug or aug_scaled) else 0)


def _resolve_architecture(cfg, overrides=None):
    """Resolve architecture dict from cfg + optional overrides."""
    overrides = overrides or {}

    # Check if overrides specify an arch_name from the catalog
    arch_name = overrides.get('arch_name')
    if arch_name and arch_name in ARCH_CATALOG:
        arch = dict(ARCH_CATALOG[arch_name])
        if 'dropout' in overrides:
            arch['dropout'] = overrides['dropout']
        return arch

    arch = getattr(cfg, 'architecture', None) or overrides.get('architecture')
    if arch is None or not arch:
        arch = ctu.ARCHITECTURE_PRESETS.get('recommended',
                                            ctu.ARCHITECTURE_PRESETS.get('light'))
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
    train_data = getattr(cfg, 'train_data', None)
    if not train_data:
        train_data = os.path.join(DATA_PROCESSED_DIR,
                                  'cnn_auger_calc.pkl')
    eval_data = getattr(cfg, 'eval_data', None)
    if not eval_data:
        eval_data = os.path.join(DATA_PROCESSED_DIR,
                                 'cnn_auger_eval.pkl')

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
    save_dir: str,
    output_dir: str,
    cfg,
    verbose: bool = True,
    **overrides,
) -> Dict[str, Any]:
    """Train CNN on a single fold."""
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
    augmented_scaled = _g('augmented_scaled', False)
    delta_be_scale   = _g('delta_be_scale', 100.0)
    device_str       = _g('device', 'auto')
    random_seed      = _g('random_seed', 42)
    split_method     = _g('split_method', 'size')
    reuse_splits     = _g('reuse_splits', True)
    use_cosine       = _g('use_cosine_schedule', True)
    late_fusion      = _g('late_fusion', False)
    broadening_fwhm  = _g('broadening_fwhm', 1.6)
    energy_min       = _g('energy_min', 200.0)
    energy_max       = _g('energy_max', 273.0)
    n_spectrum_points = _g('n_spectrum_points', 731)
    label_smoothing  = _g('label_smoothing', 0.0)
    mixup_alpha      = _g('mixup_alpha', 0.0)
    merge_scheme     = _g('merge_scheme', 'none')
    cv_suffix        = getattr(cfg, 'cv_suffix', '')
    split_file       = getattr(cfg, 'split_file', None)

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

    os.makedirs(save_dir,   exist_ok=True)
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
    augmentation_type = ('normalized' if use_augmented
                         else ('scaled' if augmented_scaled else 'normalized'))
    base_dataset = cdf.CarbonDataset(
        train_df,
        include_augmentation=use_augmented or augmented_scaled,
        augmentation_type=augmentation_type,
        delta_be_scale=delta_be_scale, normalize_delta_be=True,
        broadening_fwhm=broadening_fwhm,
        energy_min=energy_min, energy_max=energy_max,
        n_points=n_spectrum_points,
    )
    dataset = ctu.CarbonLabelDataset(base_dataset, train_df)

    # ── Molecule-level split ──────────────────────────────────────────────
    mol_train_idx, mol_val_idx = get_stratified_kfold_split(
        carbon_df=train_df,
        n_folds=n_folds, fold=fold,
        random_state=random_seed,
        split_file=split_file,
        split_method=split_method,
        force_recompute=not reuse_splits,
        verbose=verbose,
    )
    train_idx, val_idx = mol_indices_to_carbon_indices(
        train_df, mol_train_idx, mol_val_idx,
    )

    if verbose:
        print(f"  Molecule split → {len(mol_train_idx)} train / "
              f"{len(mol_val_idx)} val")
        print(f"  Carbon split   → {len(train_idx)} train / "
              f"{len(val_idx)} val")

    train_loader = DataLoader(Subset(dataset, train_idx),
                              batch_size=batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx),
                            batch_size=batch_size, shuffle=False,
                            num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    input_length = _get_input_length(train_df, cfg,
                                     use_augmented=use_augmented,
                                     augmented_scaled=augmented_scaled)
    n_late = 1 if (late_fusion and (use_augmented or augmented_scaled)) else 0
    model = AugerCNN1D(input_length, num_classes, architecture,
                       late_fusion_features=n_late)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        fusion_mode = "late fusion" if n_late > 0 else "early fusion"
        print(f"  Input length: {input_length}  |  Parameters: {n_params:,}"
              f"  |  {fusion_mode}")

    # ── Class weights + trainer ───────────────────────────────────────────
    class_weights, _ = ctu.get_class_weights_and_counts(
        train_df, num_classes=num_classes,
    )

    trainer = ctu.UnifiedCNNTrainer(
        model=model, device=device,
        learning_rate=learning_rate, weight_decay=weight_decay,
        patience=patience,
        use_cosine_schedule=use_cosine,
        cosine_T_max=num_epochs,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
        mixup_alpha=mixup_alpha,
    )

    if verbose:
        print("\nStarting training...")
    history = trainer.fit(train_loader, val_loader,
                          num_epochs=num_epochs, verbose=verbose)

    # ── Save model + history ──────────────────────────────────────────────
    model_filename = f"cnn_fold{fold}{cv_suffix}.pth"
    model_path = os.path.join(save_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    if verbose:
        print(f"\n✓ Saved model  → {model_path}")

    hist_df = pd.DataFrame(history)
    hist_path = os.path.join(output_dir,
                             f'training_history_fold{fold}{cv_suffix}.csv')
    hist_df.to_csv(hist_path, index=False)

    ctu.plot_training_history(history, output_dir)
    generic_plot = os.path.join(output_dir, 'training_plots.png')
    fold_plot = os.path.join(output_dir,
                             f'training_plots_fold{fold}{cv_suffix}.png')
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
                          merge_scheme=None, use_augmented=None,
                          augmented_scaled=None):
    """Load a CNN model from a .pth file.

    This is the CNN equivalent of ``backend._load_model_from_path``.
    """
    from augernet.evaluation_scripts.evaluate_auger_cnn_model import load_model

    train_df = data.get('train_df_raw', data['train_df'])
    input_length = _get_input_length(train_df, cfg,
                                     use_augmented=use_augmented,
                                     augmented_scaled=augmented_scaled)
    ms = merge_scheme or getattr(cfg, 'merge_scheme', 'none')
    arch = architecture or _resolve_architecture(cfg)
    device_str = getattr(cfg, 'device', 'auto')

    return load_model(
        model_path,
        architecture=arch,
        device_str=device_str,
        input_length=input_length,
        merge_scheme=ms,
    )


def _model_load_kwargs(cfg):
    """Return extra kwargs for _load_model_from_path (CNN has none)."""
    return {}


def load_saved_model(save_dir, fold, data, cfg):
    """Load a saved CNN model by fold number."""
    cv_suffix = getattr(cfg, 'cv_suffix', '')
    model_filename = f"cnn_fold{fold}{cv_suffix}.pth"
    model_path = os.path.join(save_dir, model_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved model '{model_filename}' found in:\n  {save_dir}"
        )

    return _load_model_from_path(model_path, data, cfg)


def load_param_model(model_path, data, cfg, best_params):
    """Load a CNN model from a param-search result."""
    arch = _resolve_architecture(cfg, best_params)
    ms = best_params.get('merge_scheme', getattr(cfg, 'merge_scheme', 'none'))
    use_aug = best_params.get('use_augmented')
    aug_scaled = best_params.get('augmented_scaled')
    return _load_model_from_path(model_path, data, cfg,
                                 architecture=arch, merge_scheme=ms,
                                 use_augmented=use_aug,
                                 augmented_scaled=aug_scaled)


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(model_result, data, fold, output_dir, png_dir, cfg,
                   train_results=None, **_extra):
    """Run CNN evaluation on calc + experimental data.

    Accepts same arguments as ``backend.run_evaluation`` so train_driver
    can call either backend uniformly.  Extra kwargs are checked first for
    param-search overrides (``broadening_fwhm``, ``use_augmented``, etc.),
    falling back to the base ``cfg`` values.
    """
    from augernet.evaluation_scripts.evaluate_auger_cnn_model import (
        run_evaluation as _run_eval,
    )

    if isinstance(model_result, dict):
        model = model_result['model']
        device = model_result['device']
    elif isinstance(model_result, tuple):
        model, device = model_result
    else:
        model = model_result
        device = torch.device("cpu")

    # Helper: check _extra overrides first, then cfg, then default.
    # This ensures param-search overrides (broadening_fwhm, use_augmented,
    # etc.) flow through to eval instead of always using the base config.
    def _get(key, default=None):
        if key in _extra:
            return _extra[key]
        return getattr(cfg, key, default)

    # Build a cv_suffix that reflects the *actual* config for this eval run.
    # If called from param search, include config_id so CSVs don't overwrite.
    config_id = _extra.get('config_id', '')
    base_suffix = getattr(cfg, 'cv_suffix', '')
    if config_id:
        # Param search: encode the key param values in the suffix
        merge = _get('merge_scheme', 'none')
        fwhm = _get('broadening_fwhm', 1.6)
        fwhm_str = str(fwhm).replace('.', 'pt')
        aug = 'aug' if _get('use_augmented', False) else 'noaug'
        cv_suffix = f"_{config_id}_{merge}_fwhm{fwhm_str}_{aug}"
    else:
        cv_suffix = base_suffix

    eval_kwargs = dict(
        use_augmented=_get('use_augmented', False),
        augmented_scaled=_get('augmented_scaled', False),
        delta_be_scale=_get('delta_be_scale', 100.0),
        broadening_fwhm=_get('broadening_fwhm', 1.6),
        energy_min=_get('energy_min', 200.0),
        energy_max=_get('energy_max', 273.0),
        n_spectrum_points=_get('n_spectrum_points', 731),
        merge_scheme=_get('merge_scheme', 'none'),
        cv_suffix=cv_suffix,
    )

    _run_eval(
        model, device,
        eval_data_path=data['eval_data_path'],
        output_dir=output_dir, fold=fold,
        **eval_kwargs,
    )


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
