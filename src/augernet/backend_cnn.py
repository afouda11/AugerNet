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

from sklearn.model_selection import KFold, GroupKFold, train_test_split

from augernet import cnn_train_utils as ctu
from augernet import carbon_dataframe as cdf
from augernet.class_merging import (
    apply_label_merging,
    get_merged_class_names,
    restrict_to_present,
    print_scheme_summary,
)
from augernet.norm_sidecar import (
    collect_mismatches,
    load_norm_sidecar,
    norm_sidecar_path,
    raise_on_mismatch,
    save_norm_sidecar,
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


def _cnn_fold_split(calc_mol_names, calc_df, fold, n_folds,
                    split_method, random_seed, butina_cutoff=0.65,
                    verbose=False):
    """Compute train/val molecule-name lists for the CNN, mirroring
    ``backend_gnn._get_fold_split`` exactly so both models use the same
    fold boundaries.

    Parameters
    ----------
    calc_mol_names : list[str]
        Ordered unique calc molecule names (hold-out mols already excluded).
    calc_df : pd.DataFrame
        Carbon-atom rows for those molecules (used only for Butina SMILES).
    fold : int
        1-indexed fold number.
    n_folds : int
        Total number of folds (= ``cfg.n_folds``).
    split_method : str
        ``'random'`` (KFold) or ``'butina'`` (GroupKFold on Butina clusters).
    random_seed : int
        Passed directly to ``KFold(random_state=...)``.
    butina_cutoff : float
        Tanimoto distance threshold for Butina clustering.

    Returns
    -------
    train_mol_names, val_mol_names : list[str]
    """
    n_molecules = len(calc_mol_names)
    if split_method == 'random':
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        folds = list(kf.split(range(n_molecules)))
    elif split_method == 'butina':
        cluster_ids = _butina_cluster_ids_per_molecule(
            calc_mol_names, calc_df, cutoff=butina_cutoff, verbose=verbose
        )
        gkf = GroupKFold(n_splits=n_folds)
        folds = list(gkf.split(range(n_molecules), groups=cluster_ids))
    else:
        raise ValueError(f"Unknown split_method '{split_method}'. "
                         f"Supported: 'random', 'butina'.")

    train_idx, val_idx = folds[fold - 1]
    train_mol_names = [calc_mol_names[i] for i in train_idx]
    val_mol_names   = [calc_mol_names[i] for i in val_idx]

    if verbose:
        print(f"  CNN fold {fold}/{n_folds} ({split_method}, seed={random_seed}): "
              f"{len(train_mol_names)} train, {len(val_mol_names)} val molecules")

    return train_mol_names, val_mol_names

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
            spectra, delta_be, y = batch
            spectra = spectra.to(device, dtype=torch.float32)
            delta_be = delta_be.to(device, dtype=torch.float32)
            if spectra.dim() == 2:
                spectra = spectra.unsqueeze(1)
            film_cond = torch.stack([delta_be], dim=1)  
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
    """Print a single table: per-class counts and accuracies for each split.

    ``counts`` and ``accs`` are keyed by split name (e.g. ``'train'``,
    ``'val'``, ``'holdout'``, ``'eval_auger'``).  Any subset of splits is
    accepted — columns are generated dynamically.
    """
    split_names = list(counts.keys())  # preserves insertion order (Python 3.7+)
    col_lbl = {s: s.replace('_', '-')[:9] for s in split_names}

    # Dynamic column widths
    n_col  = max(7, max(len(col_lbl[s]) + 2 for s in split_names))
    a_col  = max(9, max(len(col_lbl[s]) + 4 for s in split_names))
    row_w  = 22 + len(split_names) * (n_col + 1) + 3 + len(split_names) * (a_col + 1)
    row_w  = max(row_w, 80)

    print("\n" + "=" * row_w)
    print("PER-ENVIRONMENT BREAKDOWN (atoms per split, accuracy per split)")
    print("=" * row_w)

    hdr = f"  {'environment':<22}"
    for s in split_names:
        hdr += f" {(col_lbl[s]+' n'):>{n_col}}"
    hdr += "  "
    for s in split_names:
        hdr += f" {(col_lbl[s]+' acc'):>{a_col}}"
    print(hdr)
    print("-" * row_w)

    def fmt(stats_dict, name):
        c, t = stats_dict.get(name, (0, 0))
        return f"{(c/t*100):6.1f}%" if t > 0 else "      —"

    for name in class_names:
        row = f"  {name:<22}"
        for s in split_names:
            row += f" {counts[s].get(name, 0):>{n_col}}"
        row += "  "
        for s in split_names:
            row += f" {fmt(accs[s], name):>{a_col}}"
        print(row)
    print("=" * row_w)


# =============================================================================
#  Data loading — combine calc + eval into one DataFrame
# =============================================================================

def load_data(cfg) -> Dict[str, Any]:
    """Load CNN training data, concatenating the previously-separate
    calc and eval pickles into one DataFrame.
    """
    calc_path = os.path.join(
        DATA_PROCESSED_DIR, getattr(cfg, 'cnn_calc_data_file', 'cnn_auger_calc.pkl'))
    eval_path = os.path.join(
        DATA_PROCESSED_DIR, getattr(cfg, 'cnn_eval_data_file', 'cnn_auger_eval.pkl'))

    if not os.path.isfile(calc_path):
        raise FileNotFoundError(
            f"CNN calc data not found: {calc_path}\n"
            f"  Set 'cnn_calc_data_file' in the config, or generate it with "
            f"'python scripts/prepare_data.py'."
        )

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

    # Label space is restricted to the classes this dataset actually contains.
    # Computed once over the whole combined set

    present = set(combined_raw['carbon_env_label'])
    merge_scheme = getattr(cfg, 'merge_scheme', 'none')
    if merge_scheme != 'none':
        print_scheme_summary(merge_scheme)

    # Project onto the restricted label space for EVERY scheme, 'none' included.
    # This must not be skipped for merge_scheme='none': _resolve_num_classes and
    # _resolve_class_names return the restricted space regardless, so leaving the
    # frame in global indices trains a 30-wide head against labels up to 34 --
    # which MPS does not bounds-check, so it fails silently rather than raising.
    # data['test_df'] and the eval split are both sliced from this frame by
    # train_driver, so remapping here is what keeps all three consistent.
    n_before = len(combined)
    combined = _apply_scheme(combined, merge_scheme, present)
    if len(combined) != n_before:
        print(f"  Dropped {n_before - len(combined)} atoms whose class is "
              f"absent from the '{merge_scheme}' label space")

    n_classes = len(restrict_to_present(present, merge_scheme)[0])
    max_label = int(combined['carbon_env_index'].max())
    if max_label >= n_classes:
        raise AssertionError(
            f"label {max_label} is outside the {n_classes}-class '{merge_scheme}' "
            f"space -- the frame was not projected onto the restricted labels."
        )
    print(f"  Label space: {n_classes} classes (merge={merge_scheme}), "
          f"labels 0..{max_label}")

    return {
        'train_df':     combined,
        'train_df_raw': combined_raw,
        'present':      present,
        'eval_data_path': eval_path if os.path.exists(eval_path) else None,
    }


# =============================================================================
#  Architecture / class-count resolution  (unchanged from previous version)
# =============================================================================

def _resolve_architecture(cfg, overrides=None):
    overrides = overrides or {}
    arch = overrides.get('architecture') or getattr(cfg, 'architecture', None)
    if arch is None or not arch:
        arch = ctu.ARCHITECTURE_PRESETS['legacy_3block']
    return arch


def _resolve_num_classes(cfg, present, merge_scheme_override=None) -> int:
    ms = merge_scheme_override or getattr(cfg, 'merge_scheme', 'none')
    return len(restrict_to_present(present, ms)[0])

def _resolve_class_names(merge_scheme, present):
    return restrict_to_present(present, merge_scheme)[0]

def _apply_scheme(frame, merge_scheme, present):
    """Map *frame*'s global labels into the restricted label space."""
    frame = frame.copy()
    _, label_map = restrict_to_present(present, merge_scheme)
    frame['carbon_env_index'] = frame['carbon_env_index'].map(
        lambda i: label_map.get(i, -1))
    return frame[frame['carbon_env_index'] >= 0].reset_index(drop=True)

def _present(data):
    p = data.get('present')
    if p is None:
        raise KeyError(
            "data['present'] is missing. It is populated by load_data(); "
            "set it too if constructing the data dict directly."
        )
    return p

def _resolve_holdout_df(data, merge_scheme, scheme_overridden):
    """The calc hold-out, in the label space this run actually predicts in."""
    if scheme_overridden:
        raw = data.get('test_df_raw')
        if raw is None:
            raise KeyError(
                "merge_scheme was overridden but data['test_df_raw'] is missing. "
                "It is populated by train_driver.run alongside data['test_df']; "
                "set it too if calling train_single_run directly."
            )
        return _apply_scheme(raw, merge_scheme, _present(data))
    return data.get('test_df')


def _dataset_params(cfg, overrides=None) -> Dict[str, Any]:
    """CarbonDataset construction parameters, honouring param-search overrides."""
    o = overrides or {}
    g = lambda k, d=None: o.get(k, getattr(cfg, k, d))
    return dict(
        include_augmentation=g('cebe_augment', True),
        normalize_intensity=g('normalize_intensity', False),
        broadening_fwhm=g('fwhm', 1.6),
        energy_min=g('min_ke', 200.0),
        energy_max=g('max_ke', 273.0),
        n_points=g('n_points', 731),
    )


def _make_dataset(df_sub, ds_params, norm_stats=None):
    return cdf.CarbonDataset(df_sub, norm_stats=norm_stats, **ds_params)


def _check_evaluate_config(cfg, norm, model_path):
    """Fail if the evaluate config disagrees with the checkpoint's sidecar.

    The strict ``load_state_dict`` already catches anything that changes a
    tensor shape — filters, kernel sizes, ``pool_kernel``, ``num_classes`` (so
    ``merge_scheme``), and ``film_inputs``.  It catches none of the input-build
    settings, because ``AugerCNN1D_FiLMd`` is length-agnostic: a spectrum
    broadened with a different FWHM, on a different grid, or with
    ``cebe_augment`` toggled is consumed silently.  Those are exactly the
    settings checked here.
    """
    ds_cfg = _dataset_params(cfg)
    ds_ckpt = norm.get('dataset') or {}

    pairs = [(f'{key}', ds_cfg[key], ds_ckpt.get(key)) for key in ds_cfg]
    pairs += [
        ('merge_scheme', getattr(cfg, 'merge_scheme', 'none'),
         norm.get('merge_scheme')),
        ('film_inputs', getattr(cfg, 'film_inputs', 'none'),
         norm.get('film_inputs')),
        ('architecture', _resolve_architecture(cfg), norm.get('architecture')),
    ]
    raise_on_mismatch(collect_mismatches(pairs),
                      model_path=model_path, context='Evaluate config')


def _single_env_exp_frame(eval_df):
    """One row per eval_auger molecule whose carbons are all the same class.

    Membership is computed, not hard-coded, so it tracks the class scheme.
    """
    if eval_df is None or len(eval_df) == 0 or 'exp_spec' not in eval_df.columns:
        return pd.DataFrame()
    n_types = eval_df.groupby('mol_name')['carbon_env_index'].nunique()
    keep = set(n_types[n_types == 1].index)
    return (eval_df[eval_df['mol_name'].isin(keep)]
            .drop_duplicates('mol_name').reset_index(drop=True))


def _exp_spectrum_on_grid(exp_spec, ds_params):
    """Interpolate a measured (energy, intensity) trace onto the model grid.

    The model consumes spectra broadened onto ``energy_min..energy_max`` with
    ``n_points`` samples, so a measured trace must be resampled onto exactly
    that grid, and normalised the same way, before it can be fed in.
    """
    arr = np.asarray(exp_spec, dtype=np.float64)
    e, y = arr[:, 0], arr[:, 1]
    order = np.argsort(e)
    grid = np.linspace(ds_params['energy_min'], ds_params['energy_max'],
                       ds_params['n_points'])
    g = np.clip(np.interp(grid, e[order], y[order], left=0.0, right=0.0), 0.0, None)
    if ds_params.get('normalize_intensity', False) and g.max() > 0:
        g = g / g.max()
    return g.astype(np.float32)


def _evaluate_single_env_exp(model, device, *, eval_df, ds_params, norm_stats,
                             class_names, output_dir, fold):
    """Classify carbon environment from *measured* spectra.

    Restricted to eval_auger molecules containing a single carbon environment
    (methane, benzene, cyclopropane, ...).  For those the measured molecular
    spectrum *is* that environment's spectrum, so it can be fed to the model
    directly and scored against an unambiguous label -- the only place in the
    pipeline where the classifier meets experimental data with a known target.

    Reported for interest only.  See the storage note in ``run_evaluation``:
    this must never become a model-selection signal.
    """
    frame = _single_env_exp_frame(eval_df)
    if frame.empty:
        return None

    if not ds_params.get('normalize_intensity', False):
        print("  NOTE: normalize_intensity is off, so the calculated spectra the "
              "model trained on and these measured spectra sit on different "
              "intensity scales -- treat the result as indicative only.")

    # Build the dataset normally, then swap the broadened *calculated* spectra
    # for the measured ones.  Everything else -- delta_be z-scoring, the
    # optional augmentation prepend, labels -- stays exactly as training built
    # it, so the spectrum itself is the only thing that differs.
    ds = cdf.CarbonDataset(frame, norm_stats=norm_stats, **ds_params)
    for i in range(len(frame)):
        ds._spectra[i] = _exp_spectrum_on_grid(frame.iloc[i]['exp_spec'], ds_params)

    rows, n_correct = [], 0
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            spec, dbe, label = ds[i]
            spec = spec.to(device, dtype=torch.float32).unsqueeze(0)
            if spec.dim() == 2:
                spec = spec.unsqueeze(1)
            film_cond = torch.tensor([[float(dbe)]], device=device,
                                     dtype=torch.float32)
            logits = model(spec, film_cond)
            probs = torch.softmax(logits, dim=1)[0]
            pred, true = int(logits.argmax(1).item()), int(label.item())
            n_correct += (pred == true)
            rows.append({'mol_name':   frame.iloc[i]['mol_name'],
                         'true':       class_names[true],
                         'pred':       class_names[pred],
                         'confidence': float(probs[pred]),
                         'p_true':     float(probs[true]),
                         'correct':    pred == true})

    out = pd.DataFrame(rows)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out.to_csv(os.path.join(output_dir, f'single_env_exp_fold{fold}.csv'),
                   index=False)
    return {'accuracy':  n_correct / len(out),
            'n':         len(out),
            'n_classes': out['true'].nunique(),
            'rows':      out}


def _evaluate_splits(model, device, *, holdout_df, eval_df, ds_params,
                     norm_stats, class_names, num_classes, output_dir, fold):
    """Run *model* over the calc hold-out and the eval_auger split.

    Returns ``{'holdout': {...}, 'eval_auger': {...}, 'single_env_exp': ...}``.
    The first two hold ``df`` / ``idx`` / ``accs`` / ``results`` and are scored
    on *calculated* spectra; empty splits yield a zeroed entry with
    ``results=None`` rather than being skipped, so the caller can build the
    per-environment table unconditionally.  ``single_env_exp`` is scored on
    *measured* spectra and is ``None`` when no molecule qualifies.
    """
    out: Dict[str, Any] = {}
    for key, frame, eval_type in (('holdout', holdout_df, 'calc_holdout'),
                                  ('eval_auger', eval_df, 'eval_auger')):
        if frame is not None and len(frame) > 0:
            sub = frame.reset_index(drop=True)
            idx = list(range(len(sub)))
            dataset = _make_dataset(sub, ds_params, norm_stats)
            accs, _, _ = _per_class_accuracy(
                sub, idx, dataset, model, device, class_names)
            results = ctu.evaluate_with_molecule_details(
                df=sub, model=model, device=device, dataset=dataset,
                output_dir=output_dir, eval_type=eval_type,
                csv_suffix=f'_fold{fold}',
                class_names_override=class_names,
                num_classes_override=num_classes,
            )
            out[key] = dict(df=sub, idx=idx, accs=accs, results=results)
        else:
            out[key] = dict(df=pd.DataFrame(), idx=[],
                            accs={n: (0, 0) for n in class_names}, results=None)

    # Measured spectra, single-carbon-environment molecules only.
    out['single_env_exp'] = _evaluate_single_env_exp(
        model, device, eval_df=eval_df, ds_params=ds_params,
        norm_stats=norm_stats, class_names=class_names,
        output_dir=output_dir, fold=fold)
    return out


def _evaluate_checkpoint(model_result, data, cfg, fold, output_dir):
    """Evaluate a checkpoint loaded from disk — the ``mode: evaluate`` path.

    The delta_be normalisation and the spectrum-build settings come from the
    checkpoint's sidecar, not from the config, so evaluation reproduces exactly
    the representation the model was trained on.  The config is still compared
    against the sidecar first, so a YAML that disagrees is reported rather than
    quietly overridden.

    An earlier version re-derived ``be_mu``/``be_std`` by reproducing the fold
    split from the config.  That worked but made the result depend on
    ``n_folds`` / ``split_method`` / ``random_seed`` / ``butina_cutoff`` still
    matching the training run, with nothing enforcing it.
    """
    model = model_result['model']
    device = model_result['device']
    model_path = model_result.get('model_path') or getattr(cfg, 'model_path', '')

    norm = load_norm_sidecar(model_path, require=('delta_be',))
    _check_evaluate_config(cfg, norm, model_path)

    merge_scheme = norm.get('merge_scheme', getattr(cfg, 'merge_scheme', 'none'))
    class_names = norm.get('class_names') or _resolve_class_names(merge_scheme, _present(data))
    num_classes = norm.get('num_classes') or _resolve_num_classes(cfg, _present(data))
    ds_params = norm.get('dataset') or _dataset_params(cfg)
    norm_stats = {'be_mu':  norm['delta_be']['be_mu'],
                  'be_std': norm['delta_be']['be_std']}

    df = data['train_df']                      # already in cfg's merge scheme
    eval_df = df[df['source'] == 'eval'].reset_index(drop=True)
    holdout_df = _resolve_holdout_df(data, merge_scheme, scheme_overridden=False)

    print(f"  Loaded fold normalisation from {norm_sidecar_path(model_path)}")
    print(f"    fitted on fold {norm.get('fold', '?')} "
          f"({norm.get('n_train_molecules', '?')} training molecules)")
    print(f"    delta_be: be_mu={norm_stats['be_mu']:.6f} "
          f"be_std={norm_stats['be_std']:.6f}")

    return _evaluate_splits(
        model, device, holdout_df=holdout_df, eval_df=eval_df,
        ds_params=ds_params, norm_stats=norm_stats,
        class_names=class_names, num_classes=num_classes,
        output_dir=output_dir, fold=norm.get('fold', fold),
    )


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
    normalize_intensity = _g('normalize_intensity', False)
    label_smoothing     = _g('label_smoothing', 0.1)
    noise_std           = _g('augment_noise_std', 0.0)
    film_inputs         = _g('film_inputs', 'none')

    # New: splitting params
    split_method   = _g('split_method', 'random')
    butina_cutoff  = _g('butina_cutoff', 0.65)

    # Different "folds" use different seeds — gives the driver's existing
    # fold loop a meaningful interpretation under fixed-split training.
    split_seed = random_seed + (fold - 1)

    # ── Resolve training DataFrame (re-merge if scheme differs) ──────────
    base_merge = getattr(cfg, 'merge_scheme', 'none')

    # True when this run's label space differs from the one data['train_df'] and
    # data['test_df'] were built in (i.e. a param-search merge_scheme override).
    # Both frames then have to be rebuilt from raw — see _resolve_holdout_df.
    scheme_overridden = (merge_scheme != base_merge and 'train_df_raw' in data)

    if scheme_overridden:
        df = _apply_scheme(data['train_df_raw'], merge_scheme, _present(data))
    else:
        df = data['train_df']

    ctu.seed(random_seed)
    device = ctu.get_device(device_str, verbose=verbose)

    for p in save_paths.values():
        os.makedirs(os.path.dirname(p), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    num_classes = _resolve_num_classes(cfg, _present(data), merge_scheme_override=merge_scheme)
    class_names = _resolve_class_names(merge_scheme, _present(data)) 

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
        print(f"{'=' * 70}")

    # ── Separate calc training pool from eval_auger molecules ────────────
    # After train_driver's hold-out filtering, data['train_df'] contains:
    #   source=='calc'  non-holdout calc molecules  (for train/val)
    #   source=='eval'  eval_auger molecules         (evaluation only)
    calc_df = df[df['source'] == 'calc'].reset_index(drop=True)
    eval_df = df[df['source'] == 'eval'].reset_index(drop=True)

    calc_mol_names = list(dict.fromkeys(calc_df['mol_name']))

    # ── GNN-consistent fold split (mirrors backend_gnn._get_fold_split) ──
    train_mol_names, val_mol_names = _cnn_fold_split(
        calc_mol_names, calc_df, fold, n_folds,
        split_method, random_seed, butina_cutoff=butina_cutoff, verbose=verbose,
    )
    train_mol_set = set(train_mol_names)
    val_mol_set   = set(val_mol_names)

    train_idx = calc_df.index[calc_df['mol_name'].isin(train_mol_set)].tolist()
    val_idx   = calc_df.index[calc_df['mol_name'].isin(val_mol_set)].tolist()

    train_df_subset = calc_df.iloc[train_idx].reset_index(drop=True)
    # calc z norm stats for film args, by not defining nom_stats in CarbonDataset call 
    train_ds = cdf.CarbonDataset(
        train_df_subset,
        include_augmentation=cebe_augment,
        normalize_intensity=normalize_intensity,
        broadening_fwhm=broadening_fwhm,
        energy_min=energy_min, energy_max=energy_max,
        n_points=n_spectrum_points,
    )
    # get z norm stats from train data to apply across data
    norm_stats = train_ds.norm_stats

    # ── Build dataset on calc pool only ──────────────────────────────────
    dataset = cdf.CarbonDataset(
        calc_df,
        include_augmentation=cebe_augment,
        normalize_intensity=normalize_intensity,
        broadening_fwhm=broadening_fwhm,
        energy_min=energy_min, energy_max=energy_max,
        n_points=n_spectrum_points,
        norm_stats=norm_stats,
    )

    train_loader = DataLoader(Subset(dataset, train_idx),
                              batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(Subset(dataset, val_idx),
                              batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    # AugerCNN1D_FiLMd is length-agnostic (AdaptiveAvgPool1d), so it takes no
    # input_length.  The width below is read off the dataset the model will
    # actually be fed, not derived from cfg, so it is a real diagnostic rather
    # than a restatement of the config.
    ctu.validate_architecture(architecture)
    model = ctu.AugerCNN1D_FiLMd(
        num_classes,
        film_inputs=film_inputs,
        **architecture,
    )
    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        sample_width = int(dataset[0][0].numel()) if len(dataset) else 0
        print(f"  Input width: {sample_width}  |  Parameters: {n_params:,}")

    # ── Class weights + trainer (weights from train split only!) ─────────
    class_weights, _ = train_ds.get_class_weights_and_counts(
        num_classes=num_classes
    )

    trainer = ctu.CNNTrainer(
        model=model, device=device,
        learning_rate=learning_rate, weight_decay=weight_decay,
        patience=patience,
        scheduler_type=scheduler_type,
        # The schedule spans the training budget, not the early-stopping
        # patience.  patience * 2 made the cosine period end at 2*patience
        # epochs, after which the LR climbed back up (torch's cosine is
        # periodic), and under-sized OneCycleLR into a ValueError.
        cosine_T_max=num_epochs,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
        noise_std=noise_std,
        # CarbonDataset prepends one z-scored delta_be element when
        # cebe_augment is on; keep noise augmentation off it.
        augment_offset=1 if cebe_augment else 0,
    )

    if verbose:
        print("\nStarting training...")
    history = trainer.fit(train_loader, val_loader,
                          num_epochs=num_epochs, verbose=verbose)

    # ── Save model + sidecar + history ───────────────────────────────────
    model_path = save_paths['model']
    torch.save(model.state_dict(), model_path)
    if verbose:
        print(f"\n Saved model to {model_path}")

    # The checkpoint is not self-describing: delta_be is z-scored with constants
    # fitted on THIS fold's training split, and the spectra were broadened with
    # these grid settings.  None of that changes a tensor shape, so a mismatch at
    # evaluate time cannot be caught by the state_dict load — the sidecar is the
    # only thing standing between a wrong config and plausible wrong numbers.
    ds_params = _dataset_params(cfg, overrides)
    sidecar = save_norm_sidecar(model_path, {
        'model':             'auger-cnn',
        'fold':              fold,
        'n_train_molecules': len(train_mol_names),
        'delta_be':          {'be_mu':  float(norm_stats['be_mu']),
                              'be_std': float(norm_stats['be_std'])},
        'merge_scheme':      merge_scheme,
        'num_classes':       num_classes,
        'film_inputs':       film_inputs,
        'dataset':           ds_params,
        'architecture':      architecture,
    })
    if verbose:
        print(f" Saved fold normalisation to {sidecar}")

    pd.DataFrame(history).to_csv(
        os.path.join(output_dir, f'training_history_fold{fold}.csv'),
        index=False,
    )
    ctu.plot_training_history(history, output_dir)
    generic_plot = os.path.join(output_dir, 'training_plots.png')
    fold_plot    = os.path.join(output_dir, f'training_plots_fold{fold}.png')
    if os.path.exists(generic_plot):
        os.replace(generic_plot, fold_plot)

    # ── Hold-out + eval_auger evaluation ─────────────────────────────────
    # Shared with run_evaluation so `mode: evaluate` produces identical
    # results — see _evaluate_splits.
    splits = _evaluate_splits(
        model, device,
        holdout_df=_resolve_holdout_df(data, merge_scheme, scheme_overridden),
        eval_df=eval_df,
        ds_params=ds_params,
        norm_stats=norm_stats,
        class_names=class_names, num_classes=num_classes,
        output_dir=output_dir, fold=fold,
    )
    single_env_exp     = splits['single_env_exp']
    holdout_df         = splits['holdout']['df']
    holdout_idx        = splits['holdout']['idx']
    accs_holdout       = splits['holdout']['accs']
    holdout_results    = splits['holdout']['results']
    eval_auger_idx     = splits['eval_auger']['idx']
    accs_eval          = splits['eval_auger']['accs']
    eval_auger_results = splits['eval_auger']['results']

    if verbose:
        counts = {
            'train':    _per_class_counts(calc_df, train_idx,   class_names),
            'val':      _per_class_counts(calc_df, val_idx,     class_names),
            'holdout':  _per_class_counts(holdout_df, holdout_idx, class_names)
                        if len(holdout_df) > 0 else {n: 0 for n in class_names},
            'eval-aug': _per_class_counts(eval_df, eval_auger_idx, class_names)
                        if len(eval_df) > 0 else {n: 0 for n in class_names},
        }
        accs_train, _, _ = _per_class_accuracy(
            calc_df, train_idx, dataset, model, device, class_names)
        accs_val, _, _   = _per_class_accuracy(
            calc_df, val_idx,   dataset, model, device, class_names)
        accs = {
            'train':    accs_train,
            'val':      accs_val,
            'holdout':  accs_holdout,
            'eval-aug': accs_eval,
        }
        _print_environment_table(class_names, counts, accs)

    # ── Results ───────────────────────────────────────────────────────────
    # Report the epoch whose weights were actually checkpointed and saved.
    # trainer.fit selects on max val F1, so min(val_loss) / max(val_acc) taken
    # independently over the history generally describe a DIFFERENT epoch than
    # the .pth on disk — and the driver ranks folds/configs on best_val_loss.
    sel = getattr(trainer, 'best_epoch', -1)
    if sel < 0:
        sel = int(np.argmax(history['val_f1']))

    best_val_epoch  = sel + 1                     # 1-indexed, matches the GNN
    best_val_loss   = history['val_loss'][sel]
    best_val_acc    = history['val_acc'][sel]
    best_val_f1     = history['val_f1'][sel]
    best_train_loss = history['train_loss'][sel]
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
        print(f"  Selected epoch:{best_val_epoch}  (max val F1 — saved weights)")
        print(f"  Val Loss:      {best_val_loss:.4f}")
        print(f"  Val Acc:       {best_val_acc:.2f}%")
        print(f"  Val F1:        {best_val_f1:.4f}")
        if holdout_results is not None:
            print(f"  Calc holdout:  "
                  f"{holdout_results.get('accuracy', 0)*100:.2f}% acc  "
                  f"F1-macro={holdout_results.get('f1_macro', 0):.4f}  "
                  f"({holdout_df['mol_name'].nunique()} mols)")
        if eval_auger_results is not None:
            print(f"  Eval-auger calc: "
                  f"{eval_auger_results.get('accuracy', 0)*100:.2f}% acc  "
                  f"F1-macro={eval_auger_results.get('f1_macro', 0):.4f}  "
                  f"({eval_df['mol_name'].nunique()} mols)")
        if single_env_exp is not None:
            print(f"  Eval-auger exp:  "
                  f"{single_env_exp['accuracy']*100:.2f}% acc  "
                  f"({single_env_exp['n']} single-environment mols)")

    return {
        'model': model,
        'device': device,
        'fold': fold,
        # All four are read off best_val_epoch, so they describe one model.
        'best_val_loss': best_val_loss,
        'combined_val_loss': best_val_loss,
        'best_train_loss': best_train_loss,
        'best_val_epoch': best_val_epoch,
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1,
        'final_train_acc': final_train_acc,
        'final_val_acc': final_val_acc,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'n_epochs': n_epochs_run,
        'model_path': model_path,
        'holdout_results': holdout_results,      # calc hold-out (GNN-consistent)
        'eval_auger_results': eval_auger_results, # eval_auger, calculated spectra
        'single_env_exp': single_env_exp,        # eval_auger, measured spectra
        # Tells run_evaluation the splits have already been scored, so it
        # reports rather than re-running inference.  Absent on the bare
        # {'model', 'device'} dict that mode: evaluate builds.
        'evaluated': True,
        'train_idx': train_idx,
        'val_idx':   val_idx,
    }


# =============================================================================
#  Model loading
# =============================================================================

def _load_model_from_path(model_path, data, cfg, *, architecture=None,
                          merge_scheme=None):
    ms = merge_scheme or getattr(cfg, 'merge_scheme', 'none')
    arch = architecture or _resolve_architecture(cfg)
    device_str = getattr(cfg, 'device', 'auto')

    device = ctu.get_device(device_str, verbose=True)
    num_classes = _resolve_num_classes(cfg, _present(data), merge_scheme_override=ms)
    film_inputs = getattr(cfg, 'film_inputs', 'none')
    ctu.validate_architecture(arch)
    model = ctu.AugerCNN1D_FiLMd(
        num_classes,
        film_inputs=film_inputs,
        **arch,
    )

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
    """Report hold-out and eval_auger performance.

    ``mode: train`` passes the result dict from ``train_single_run``, which
    already carries both — this then just reports them.  ``mode: evaluate``
    passes a bare ``{'model', 'device'}`` loaded from disk, in which case the
    evaluation is run here.  Previously the latter case silently produced
    nothing at all.
    """
    holdout_results     = model_result.get('holdout_results')
    eval_auger_results  = model_result.get('eval_auger_results')
    single_env_exp      = model_result.get('single_env_exp')

    # 'evaluated' is set by train_single_run.  Keyed on that rather than on the
    # results being None, because a run with no hold-out and no eval_auger data
    # legitimately yields two Nones and must not be re-evaluated.
    if not model_result.get('evaluated', False):
        print(f"\n{'=' * 70}")
        print("CNN EVALUATION — running inference on the loaded checkpoint")
        print(f"{'=' * 70}")
        splits = _evaluate_checkpoint(model_result, data, cfg, fold, output_dir)
        holdout_results    = splits['holdout']['results']
        eval_auger_results = splits['eval_auger']['results']
        single_env_exp     = splits['single_env_exp']

    print(f"\n{'=' * 70}")
    print(f"CNN EVALUATION SUMMARY  (fold {fold})")
    print(f"{'=' * 70}")

    def _print_block(label, res):
        if res is None:
            print(f"\n  {label}: (no results)")
            return
        print(f"\n  {label}:")
        print(f"    Accuracy:    {res.get('accuracy', 0)*100:.2f}%")
        print(f"    F1-macro:    {res.get('f1_macro',    0):.4f}")
        print(f"    F1-weighted: {res.get('f1_weighted', 0):.4f}")
        print(f"    Prec-macro:  {res.get('precision_macro', 0):.4f}")
        print(f"    Rec-macro:   {res.get('recall_macro',    0):.4f}")
        if res.get('per_class'):
            print(f"    {'Class':<22} {'N':>6} {'Correct':>9} {'Acc':>8}")
            print(f"    {'-'*50}")
            for cls, info in res['per_class'].items():
                n   = info.get('n_total', 0)
                cor = info.get('n_correct', 0)
                acc = f"{cor/n*100:.1f}%" if n > 0 else '—'
                print(f"    {cls:<22} {n:>6} {cor:>9} {acc:>8}")

    _print_block('Calc hold-out (GNN-consistent, 50 molecules)', holdout_results)
    _print_block('Eval-auger (calc spectra)', eval_auger_results)

    # Measured spectra.  Only molecules with one carbon environment qualify,
    # so this is a small set -- reported per molecule rather than aggregated.
    if single_env_exp:
        se = single_env_exp
        print(f"\n  Eval-auger (exp spectra):")
        print(f"    Single-environment molecules only "
              f"({se['n']} molecules, {se['n_classes']} classes)")
        print(f"    Accuracy:    {se['accuracy']*100:.2f}%")
        print(f"    {'':2}{'molecule':<22}{'true':<22}{'predicted':<22}{'conf':>6}")
        print(f"    {'-' * 74}")
        for _, r in se['rows'].iterrows():
            print(f"    {' ' if r['correct'] else '*'} {r['mol_name']:<22}"
                  f"{r['true']:<22}{r['pred']:<22}{r['confidence']:>6.2f}")
    else:
        print(f"\n  Eval-auger (exp spectra): (no single-environment molecules)")

    print(f"\n{'=' * 70}")

    # Flatten to the scalar contract train_driver._collect_eval_metrics expects:
    # only keys prefixed 'eval_'/'test_' whose value is a scalar are recorded on
    # the fold/config entry and aggregated into the CV / param summary JSON.
    # Nested dicts (per_class, predictions, ...) are dropped there, so returning
    # only those meant every CNN accuracy/F1 existed solely in stdout.
    #   test_ = calc hold-out          eval_ = eval_auger (experimental)
    # Adding a key here is all that is needed for a metric to reach the CV /
    # param summary JSON: _aggregate_eval_metrics picks up any scalar carrying
    # an eval_/test_ prefix, so no change is required in train_driver.
    _SCALARS = {
        'accuracy':              'acc',
        'f1_macro':              'f1_macro',
        'f1_weighted':           'f1_weighted',
        'precision_macro':       'prec_macro',
        'precision_weighted':    'prec_weighted',
        'recall_macro':          'rec_macro',
        'recall_weighted':       'rec_weighted',
        'dedup_accuracy':        'dedup_acc',
        'dedup_f1_macro':        'dedup_f1_macro',
        'dedup_f1_weighted':     'dedup_f1_weighted',
        'dedup_precision_macro': 'dedup_prec_macro',
        'dedup_recall_macro':    'dedup_rec_macro',
    }

    metrics = {}
    for prefix, res in (('test', holdout_results), ('eval', eval_auger_results)):
        if not isinstance(res, dict):
            continue
        for src, short in _SCALARS.items():
            v = res.get(src)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                metrics[f'{prefix}_{short}'] = float(v)

    # Nested detail kept for callers that want it; filtered out downstream.
    metrics['holdout'] = holdout_results
    metrics['eval_auger'] = eval_auger_results

    # Deliberately NOT prefixed 'eval_'/'test_' and deliberately not a scalar:
    # _collect_eval_metrics records only scalars under those prefixes, so this
    # reports per fold but never enters CV aggregation or param-search
    # selection.  Renaming it to an eval_* scalar would silently make it a
    # model-selection signal.
    metrics['single_env_exp'] = single_env_exp
    return metrics


# =============================================================================
#  Unit tests & predict
# =============================================================================

def run_unit_tests(model, data, cfg):
    print("  (no unit tests for CNN model)")


def run_predict(*, model_path: str, predict_dir: str, fold, cfg):
    raise NotImplementedError(
        "Predict mode is not yet implemented for model 'auger-cnn'."
    )