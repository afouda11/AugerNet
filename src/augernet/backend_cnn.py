"""
Auger-CNN Backend — model-specific routines for train_driver.py
=============================================================

Carbon environment classification using 1D CNN on Auger spectra.

This version combines the calc + eval pickles into a single dataset,
does a molecule-level 3-way train/val/test split (random or Butina),
trains on train, validates for early stopping on val, then runs the
per-molecule evaluation on the held-out test split inside
``train_single_run``.

Public surface (called by train_driver):
  load_data, train_single_run, load_saved_model,
  run_evaluation, run_unit_tests, run_predict
"""

from __future__ import annotations

import os
import numpy as np
import torch
import pandas as pd
from typing import Any, Dict, List, Tuple

from sklearn.model_selection import train_test_split

from augernet import cnn_train_utils as ctu
from augernet import carbon_dataframe as cdf
from augernet.class_merging import (
    apply_label_merging,
    get_merged_class_names,
    get_num_classes,
    print_scheme_summary,
)

from augernet import DATA_PROCESSED_DIR


# =============================================================================
#  Molecule-level 3-way split  (train / val / test)
# =============================================================================

def _molecule_groups(carbon_df: pd.DataFrame) -> List[str]:
    """Ordered list of unique mol_names as they first appear in the df."""
    seen: dict = {}
    for name in carbon_df['mol_name']:
        if name not in seen:
            seen[name] = len(seen)
    return list(seen.keys())


def _butina_cluster_ids_per_molecule(mol_names: List[str],
                                     carbon_df: pd.DataFrame,
                                     cutoff: float = 0.65,
                                     verbose: bool = True) -> List[int]:
    """Run Butina clustering on the unique molecules and return a cluster
    ID per molecule (same order as ``mol_names``).

    Requires the carbon DataFrame to have a 'smiles' column (one SMILES
    per row; rows belonging to the same molecule must share the SMILES).
    The Tanimoto/Butina implementation itself is re-used from the GNN
    backend so the two models cluster molecules the same way.
    """
    if 'smiles' not in carbon_df.columns:
        raise ValueError(
            "Butina splitting requires a 'smiles' column in the carbon "
            "DataFrame. Either add one upstream (one SMILES per atom row, "
            "shared across atoms of the same molecule) or set "
            "split_method='random'."
        )
    from augernet.build_molecular_graphs import get_butina_clusters

    # One SMILES per unique molecule, in the same order as mol_names
    first_idx = carbon_df.drop_duplicates('mol_name').set_index('mol_name')
    smiles_per_mol = [first_idx.loc[name, 'smiles'] for name in mol_names]
    cluster_ids = get_butina_clusters(smiles_per_mol, cutoff=cutoff)
    if verbose:
        print(f"  Butina clustering: {len(set(cluster_ids))} clusters "
              f"across {len(mol_names)} molecules (cutoff={cutoff})")
    return list(cluster_ids)


def _three_way_split(carbon_df: pd.DataFrame,
                     *,
                     train_frac: float = 0.70,
                     val_frac:   float = 0.15,
                     test_frac:  float = 0.15,
                     split_method: str = 'random',
                     random_seed: int = 42,
                     butina_cutoff: float = 0.65,
                     verbose: bool = True
                     ) -> Tuple[List[int], List[int], List[int]]:
    """Split carbon-atom rows into train/val/test at the molecule (or
    Butina-cluster) level, so atoms from one molecule never straddle
    a split boundary.

    Returns three lists of row indices into ``carbon_df``.
    """
    eps = 1e-6
    if not abs(train_frac + val_frac + test_frac - 1.0) < eps:
        raise ValueError(
            f"Split fractions must sum to 1.0; got "
            f"{train_frac}+{val_frac}+{test_frac}={train_frac+val_frac+test_frac}"
        )

    mol_names = _molecule_groups(carbon_df)

    if split_method == 'random':
        # Treat each molecule as its own "group"; group_ids == index in mol_names
        groups = np.arange(len(mol_names))
    elif split_method == 'butina':
        groups = np.array(_butina_cluster_ids_per_molecule(
            mol_names, carbon_df, cutoff=butina_cutoff, verbose=verbose
        ))
    else:
        raise ValueError(f"Unknown split_method '{split_method}'. "
                         f"Supported: 'random', 'butina'.")

    # We split *cluster IDs* (random: one cluster per molecule;
    # butina: real clusters). All molecules in a cluster move together.
    unique_clusters = np.array(sorted(set(groups.tolist())))

    # First peel off the test set, then split the remainder into train+val.
    test_size_rel = test_frac
    val_size_rel  = val_frac / (train_frac + val_frac)  # of the trainval pool

    trainval_clust, test_clust = train_test_split(
        unique_clusters,
        test_size=test_size_rel,
        random_state=random_seed,
        shuffle=True,
    )
    train_clust, val_clust = train_test_split(
        trainval_clust,
        test_size=val_size_rel,
        random_state=random_seed,
        shuffle=True,
    )

    train_clusters = set(train_clust.tolist())
    val_clusters   = set(val_clust.tolist())
    test_clusters  = set(test_clust.tolist())

    # Map cluster membership back to row indices.
    # groups[i] is the cluster of the i-th *unique molecule*, so we need
    # a per-row mapping mol_name -> cluster_id.
    mol_to_cluster = {mol_names[i]: int(groups[i]) for i in range(len(mol_names))}

    train_rows, val_rows, test_rows = [], [], []
    for row_idx, name in enumerate(carbon_df['mol_name']):
        c = mol_to_cluster[name]
        if c in train_clusters:
            train_rows.append(row_idx)
        elif c in val_clusters:
            val_rows.append(row_idx)
        elif c in test_clusters:
            test_rows.append(row_idx)
        # (every cluster is in exactly one bucket — no else needed)

    if verbose:
        n_mol = len(mol_names)
        n_train_mol = sum(1 for n in mol_names if mol_to_cluster[n] in train_clusters)
        n_val_mol   = sum(1 for n in mol_names if mol_to_cluster[n] in val_clusters)
        n_test_mol  = sum(1 for n in mol_names if mol_to_cluster[n] in test_clusters)
        print(f"  Split ({split_method}, seed={random_seed}): "
              f"{n_train_mol}/{n_val_mol}/{n_test_mol} molecules "
              f"({n_mol} total)")
        print(f"  Carbon rows: {len(train_rows)}/{len(val_rows)}/{len(test_rows)} "
              f"(train/val/test)")

    return train_rows, val_rows, test_rows


# =============================================================================
#  Per-environment summary helpers
# =============================================================================

def _per_class_counts(df: pd.DataFrame, row_indices: List[int],
                      class_names: List[str]) -> Dict[str, int]:
    """Count atoms per class within a row subset."""
    labels = df.iloc[row_indices]['carbon_env_index'].to_numpy()
    counts = {name: 0 for name in class_names}
    for lbl in labels:
        if 0 <= lbl < len(class_names):
            counts[class_names[lbl]] += 1
    return counts


def _per_class_accuracy(df: pd.DataFrame, row_indices: List[int],
                        dataset, model, device,
                        class_names: List[str]) -> Dict[str, Tuple[int, int]]:
    """Predict on a subset and return {class_name: (n_correct, n_total)}."""
    from torch.utils.data import DataLoader, Subset

    model.eval()
    loader = DataLoader(Subset(dataset, row_indices),
                        batch_size=64, shuffle=False, num_workers=0)
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            spectra, delta_be, mol_size, y = batch
            spectra = spectra.to(device, dtype=torch.float32)
            delta_be = delta_be.to(device, dtype=torch.float32)
            mol_size = mol_size.to(device, dtype=torch.float32)
            if spectra.dim() == 2:
                spectra = spectra.unsqueeze(1)
            film_cond = torch.stack([delta_be, mol_size], dim=1)  # (B, 2)
            logits = model(spectra, film_cond)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(y.numpy())
    preds  = np.concatenate(preds)  if preds  else np.array([], dtype=int)
    labels = np.concatenate(labels) if labels else np.array([], dtype=int)

    out = {name: (0, 0) for name in class_names}
    for lbl, prd in zip(labels, preds):
        if 0 <= lbl < len(class_names):
            name = class_names[lbl]
            correct, total = out[name]
            out[name] = (correct + int(prd == lbl), total + 1)
    return out, preds, labels


def _print_environment_table(class_names: List[str],
                             counts: Dict[str, Dict[str, int]],
                             accs:   Dict[str, Dict[str, Tuple[int, int]]]):
    """Print a single table: per-class counts and accuracies for each split."""
    print("\n" + "=" * 100)
    print("PER-ENVIRONMENT BREAKDOWN (atoms per split, accuracy per split)")
    print("=" * 100)
    header = (f"  {'environment':<22} "
              f"{'train n':>8} {'val n':>7} {'test n':>7}   "
              f"{'train acc':>10} {'val acc':>10} {'test acc':>10}")
    print(header)
    print("-" * 100)
    for name in class_names:
        n_tr = counts['train'].get(name, 0)
        n_va = counts['val'].get(name, 0)
        n_te = counts['test'].get(name, 0)

        def fmt(stats):
            c, t = stats.get(name, (0, 0))
            return f"{(c/t*100):8.1f}%" if t > 0 else "      —  "

        print(f"  {name:<22} "
              f"{n_tr:>8} {n_va:>7} {n_te:>7}   "
              f"{fmt(accs['train']):>10} {fmt(accs['val']):>10} "
              f"{fmt(accs['test']):>10}")
    print("=" * 100)


# =============================================================================
#  Data loading — combine calc + eval into one DataFrame
# =============================================================================

def load_data(cfg) -> Dict[str, Any]:
    """Load CNN training data, concatenating the previously-separate
    calc and eval pickles into one DataFrame.
    """
    calc_path = os.path.join(DATA_PROCESSED_DIR, 'cnn_auger_calc.pkl')
    eval_path = os.path.join(DATA_PROCESSED_DIR, 'cnn_auger_eval.pkl')

    print(f"\nLoading calc data: {calc_path}")
    calc_df = pd.read_pickle(calc_path)
    print(f"  Loaded {len(calc_df)} carbon atoms "
          f"({calc_df['mol_name'].nunique()} molecules)")

    eval_df = None
    if os.path.exists(eval_path):
        print(f"Loading eval data: {eval_path}")
        eval_df = pd.read_pickle(eval_path)
        print(f"  Loaded {len(eval_df)} carbon atoms "
              f"({eval_df['mol_name'].nunique()} molecules)")

    # Concatenate; tag rows so we can sanity-check splits later
    if eval_df is not None:
        calc_df = calc_df.assign(source='calc')
        eval_df = eval_df.assign(source='eval')
        combined = pd.concat([calc_df, eval_df], ignore_index=True)
        print(f"Combined dataset: {len(combined)} atoms, "
              f"{combined['mol_name'].nunique()} molecules")
    else:
        combined = calc_df.assign(source='calc')

    # Keep raw copy for param search re-merging
    combined_raw = combined.copy()

    merge_scheme = getattr(cfg, 'merge_scheme', 'none')
    if merge_scheme != 'none':
        print_scheme_summary(merge_scheme)
        combined = apply_label_merging(combined, merge_scheme)
        # Drop rows whose original label didn't map to any merged class
        n_before = len(combined)
        combined = combined[combined['carbon_env_index'] >= 0].reset_index(drop=True)
        if len(combined) != n_before:
            print(f"  Dropped {n_before - len(combined)} unmapped atoms after merging")

    return {
        'train_df':     combined,
        'train_df_raw': combined_raw,
        # Kept for backwards compatibility — no longer used by training,
        # but run_evaluation still accepts an external eval pkl path if
        # the user wants the old behaviour.
        'eval_data_path': eval_path if os.path.exists(eval_path) else None,
    }


# =============================================================================
#  Architecture / class-count resolution  (unchanged from previous version)
# =============================================================================

def _get_input_length(cfg, *, use_augmented=None) -> int:
    n_spec = getattr(cfg, 'n_points', 731)
    use_aug = use_augmented if use_augmented is not None \
              else getattr(cfg, 'use_augmented', False)
    return n_spec + (1 if use_aug else 0)


def _resolve_architecture(cfg, overrides=None):
    overrides = overrides or {}
    arch = overrides.get('architecture') or getattr(cfg, 'architecture', None)
    if arch is None or not arch:
        arch = ctu.ARCHITECTURE_PRESETS.get(
            'recommended', ctu.ARCHITECTURE_PRESETS.get('legacy_3block')
        )
    return arch


def _resolve_num_classes(cfg, merge_scheme_override=None) -> int:
    ms = merge_scheme_override or getattr(cfg, 'merge_scheme', 'none')
    if ms != 'none':
        return get_num_classes(ms)
    return ctu.NUM_CARBON_CLASSES


# =============================================================================
#  Single training run  (now includes internal test evaluation)
# =============================================================================

def train_single_run(data: Dict[str, Any],
                     fold: int,
                     n_folds: int,
                     *,
                     save_paths: Dict[str, str],
                     output_dir: str,
                     cfg,
                     verbose: bool = True,
                     **overrides) -> Dict[str, Any]:
    """Train on the train split, early-stop on val, evaluate on the held-out
    test split, all from a single combined DataFrame.

    The ``fold`` and ``n_folds`` parameters are kept for compatibility with
    the train_driver interface; they are not used for k-fold CV in this
    version. Different folds simply correspond to different random seeds
    so the driver's existing fold loop produces multiple seeded runs.
    """
    from torch.utils.data import DataLoader, Subset

    # ── Resolve hyper-parameters from cfg + overrides ─────────────────────
    _g = lambda k, d=None: overrides.get(k, getattr(cfg, k, d))

    architecture        = _resolve_architecture(cfg, overrides)
    num_epochs          = _g('num_epochs', 500)
    patience            = _g('patience', 40)
    batch_size          = _g('batch_size', 64)
    learning_rate       = _g('learning_rate', 3e-4)
    weight_decay        = _g('weight_decay', 1e-4)
    cebe_augment        = _g('cebe_augment', True)
    device_str          = _g('device', 'auto')
    random_seed         = _g('random_seed', 42)
    scheduler_type      = _g('scheduler_type', 'cosine')
    broadening_fwhm     = _g('fwhm', 1.6)
    energy_min          = _g('min_ke', 200.0)
    energy_max          = _g('max_ke', 273.0)
    n_spectrum_points   = _g('n_points', 731)
    merge_scheme        = _g('merge_scheme', 'none')
    normalize_intensity = _g('normalize_intensity', True)
    label_smoothing     = _g('label_smoothing', 0.1)
    noise_std           = _g('augment_noise_std', 0.0)
    film_inputs         = _g('film_inputs', 'both')

    # New: splitting params
    split_method   = _g('split_method', 'random')
    butina_cutoff  = _g('butina_cutoff', 0.65)
    train_frac     = _g('train_frac', 0.70)
    val_frac       = _g('val_frac', 0.15)
    test_frac      = _g('test_frac', 0.15)

    # Different "folds" use different seeds — gives the driver's existing
    # fold loop a meaningful interpretation under fixed-split training.
    split_seed = random_seed + (fold - 1)

    # ── Resolve training DataFrame (re-merge if scheme differs) ──────────
    base_merge = getattr(cfg, 'merge_scheme', 'none')
    if merge_scheme != base_merge and 'train_df_raw' in data:
        df = data['train_df_raw'].copy()
        if merge_scheme != 'none':
            df = apply_label_merging(df, merge_scheme)
            df = df[df['carbon_env_index'] >= 0].reset_index(drop=True)
    else:
        df = data['train_df']

    ctu.seed(random_seed)
    device = ctu.get_device(device_str, verbose=verbose)

    for p in save_paths.values():
        os.makedirs(os.path.dirname(p), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    num_classes = _resolve_num_classes(cfg, merge_scheme_override=merge_scheme)
    if merge_scheme != 'none':
        class_names = get_merged_class_names(merge_scheme)
    else:
        class_names = ctu.CARBON_ENVIRONMENT_NAMES

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"AUGER CNN TRAINING — Run {fold}/{n_folds}  (split seed={split_seed})")
        print(f"{'=' * 70}")
        print(f"  Arch:          {architecture}")
        print(f"  LR:            {learning_rate}")
        print(f"  Batch:         {batch_size}")
        print(f"  Classes:       {num_classes}  (merge={merge_scheme})")
        print(f"  FWHM:          {broadening_fwhm} eV")
        print(f"  CEBE augment:  {cebe_augment}")
        print(f"  Normalize int: {normalize_intensity}")
        print(f"  Split method:  {split_method}"
              f"{f' (Butina cutoff={butina_cutoff})' if split_method=='butina' else ''}")
        print(f"  Split fracs:   {train_frac:.2f} / {val_frac:.2f} / {test_frac:.2f}")
        print(f"{'=' * 70}")

    # ── Build the dataset once over the entire combined df ────────────────
    dataset = cdf.CarbonDataset(
        df,
        include_augmentation=cebe_augment,
        normalize_intensity=normalize_intensity,
        broadening_fwhm=broadening_fwhm,
        energy_min=energy_min, energy_max=energy_max,
        n_points=n_spectrum_points,
    )

    # ── 3-way molecule-level split ────────────────────────────────────────
    train_idx, val_idx, test_idx = _three_way_split(
        df,
        train_frac=train_frac, val_frac=val_frac, test_frac=test_frac,
        split_method=split_method, random_seed=split_seed,
        butina_cutoff=butina_cutoff, verbose=verbose,
    )

    train_loader = DataLoader(Subset(dataset, train_idx),
                              batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(Subset(dataset, val_idx),
                              batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    input_length = _get_input_length(cfg, use_augmented=cebe_augment)
    model = ctu.AugerCNN1D_FiLMd(input_length, num_classes, film_inputs=film_inputs)
    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Input length: {input_length}  |  Parameters: {n_params:,}")

    # ── Class weights + trainer (weights from train split only!) ─────────
    train_df_subset = df.iloc[train_idx].reset_index(drop=True)
    train_dataset_for_weights = cdf.CarbonDataset(
        train_df_subset,
        include_augmentation=cebe_augment,
        normalize_intensity=normalize_intensity,
        broadening_fwhm=broadening_fwhm,
        energy_min=energy_min, energy_max=energy_max,
        n_points=n_spectrum_points,
    )
    class_weights, _ = train_dataset_for_weights.get_class_weights_and_counts(
        num_classes=num_classes
    )

    trainer = ctu.CNNTrainer(
        model=model, device=device,
        learning_rate=learning_rate, weight_decay=weight_decay,
        patience=patience,
        scheduler_type=scheduler_type,
        cosine_T_max=patience * 2,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
        noise_std=noise_std,
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

    pd.DataFrame(history).to_csv(
        os.path.join(output_dir, f'training_history_fold{fold}.csv'),
        index=False,
    )
    ctu.plot_training_history(history, output_dir)
    generic_plot = os.path.join(output_dir, 'training_plots.png')
    fold_plot    = os.path.join(output_dir, f'training_plots_fold{fold}.png')
    if os.path.exists(generic_plot):
        os.replace(generic_plot, fold_plot)

    # ── Per-split per-environment summary ─────────────────────────────────
    if verbose:
        counts = {
            'train': _per_class_counts(df, train_idx, class_names),
            'val':   _per_class_counts(df, val_idx,   class_names),
            'test':  _per_class_counts(df, test_idx,  class_names),
        }
        accs_train, _, _ = _per_class_accuracy(df, train_idx, dataset, model,
                                               device, class_names)
        accs_val,   _, _ = _per_class_accuracy(df, val_idx,   dataset, model,
                                               device, class_names)
        accs_test, test_preds, test_labels = _per_class_accuracy(
            df, test_idx, dataset, model, device, class_names
        )
        accs = {'train': accs_train, 'val': accs_val, 'test': accs_test}
        _print_environment_table(class_names, counts, accs)

        # ── Per-molecule test evaluation (rich CSV + summary) ─────────────
        test_df = df.iloc[test_idx].reset_index(drop=True)
        test_dataset = cdf.CarbonDataset(
            test_df,
            include_augmentation=cebe_augment,
            normalize_intensity=normalize_intensity,
            broadening_fwhm=broadening_fwhm,
            energy_min=energy_min, energy_max=energy_max,
            n_points=n_spectrum_points,
        )
        test_results = ctu.evaluate_with_molecule_details(
            df=test_df, model=model, device=device,
            dataset=test_dataset,
            output_dir=output_dir,
            eval_type='test',
            csv_suffix=f'_fold{fold}',
            class_names_override=class_names,
            num_classes_override=num_classes,
        )
    else:
        test_results = None

    # ── Results ───────────────────────────────────────────────────────────
    best_val_loss   = min(history['val_loss'])
    best_val_acc    = max(history['val_acc'])
    best_val_f1     = max(history['val_f1'])
    final_train_acc = history['train_acc'][-1]
    final_val_acc   = history['val_acc'][-1]
    n_epochs_run    = len(history['train_loss'])

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"RUN {fold} COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Epochs run:    {n_epochs_run}")
        print(f"  Final Train:   {final_train_acc:.2f}%")
        print(f"  Final Val:     {final_val_acc:.2f}%")
        print(f"  Best Val Loss: {best_val_loss:.4f}")
        print(f"  Best Val Acc:  {best_val_acc:.2f}%")
        print(f"  Best Val F1:   {best_val_f1:.4f}")
        if test_results is not None:
            print(f"  Test Acc:      "
                  f"{test_results.get('accuracy', 0)*100:.2f}%  "
                  f"(F1-macro={test_results.get('f1_macro', 0):.4f})")

    return {
        'model': model,
        'device': device,
        'fold': fold,
        'best_val_loss': best_val_loss,
        'combined_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1,
        'final_train_acc': final_train_acc,
        'final_val_acc': final_val_acc,
        'n_epochs': n_epochs_run,
        'model_path': model_path,
        'test_results': test_results,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
    }


# =============================================================================
#  Model loading
# =============================================================================

def _load_model_from_path(model_path, data, cfg, *, architecture=None,
                          merge_scheme=None, use_augmented=None):
    input_length = _get_input_length(cfg, use_augmented=use_augmented)
    ms = merge_scheme or getattr(cfg, 'merge_scheme', 'none')
    arch = architecture or _resolve_architecture(cfg)
    device_str = getattr(cfg, 'device', 'auto')

    device = ctu.get_device(device_str, verbose=True)
    num_classes = _resolve_num_classes(cfg, merge_scheme_override=ms)
    film_inputs = getattr(cfg, 'film_inputs', 'both')
    model = ctu.AugerCNN1D_FiLMd(input_length, num_classes, film_inputs=film_inputs)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model from {model_path}  ({n_params:,} params)")
    return model, device


def load_saved_model(save_paths, data, cfg):
    model_path = save_paths['model']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model found:\n  {model_path}")
    return _load_model_from_path(model_path, data, cfg)


# =============================================================================
#  run_evaluation — kept as a thin shim for the driver
# =============================================================================

def run_evaluation(model_result, data, fold, output_dir, png_dir, cfg,
                   train_results=None, **_extra):
    """The test split is now evaluated inside ``train_single_run``.

    This stub is preserved so train_driver doesn't error out; it simply
    surfaces the test results already attached to ``model_result``.
    """
    test_results = model_result.get('test_results')
    if test_results is None:
        print("\n(No held-out test results were captured during training; "
              "skipping run_evaluation.)")
        return {}
    print(f"\nTest-split metrics from fold {fold}:")
    print(f"  Accuracy:    {test_results.get('accuracy', 0)*100:.2f}%")
    print(f"  F1-macro:    {test_results.get('f1_macro', 0):.4f}")
    print(f"  F1-weighted: {test_results.get('f1_weighted', 0):.4f}")
    return test_results


# =============================================================================
#  Unit tests & predict
# =============================================================================

def run_unit_tests(model, data, cfg):
    print("  (no unit tests for CNN model)")


def run_predict(*, model_path: str, predict_dir: str, fold, cfg):
    raise NotImplementedError(
        "Predict mode is not yet implemented for model 'auger-cnn'."
    )