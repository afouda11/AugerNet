"""
AugerNet Training Driver
=========================

Contains run_kfold_cv, run_param_search, _build_param_configs,
and the mode-dispatch logic for the GNN and CNN training.

Model-specific behaviour is provided by the backend module:
  - augernet.backend_gnn  (CEBE and Auger prediction GNN)
  - augernet.backend_cnn  (bond environment classification CNN)

The backend exports hooks:
  load_data(cfg)                : data dict
  train_single_run(data, …)     : result dict  (receives save_paths from driver)
  load_saved_model(save_paths, …): (model, device) or result dict
  run_evaluation(…)             : eval metrics dict
  run_unit_tests(…)             : None
  run_predict(…)                : None

"""

from __future__ import annotations

import os
import json
import time
import itertools
import numpy as np
from typing import Any, Dict, List

from augernet.config import AugerNetConfig

# ─────────────────────────────────────────────────────────────────────────────
#  Backend registry
# ─────────────────────────────────────────────────────────────────────────────

def _get_backend(cfg):
    """Return the backend module for the given model type."""
    if cfg.model == 'auger-cnn':
        from augernet import backend_cnn
        return backend_cnn
    else:
        from augernet import backend_gnn
        return backend_gnn


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation-metric contract
# ─────────────────────────────────────────────────────────────────────────────
#  Backends return a flat dict of scalars from ``run_evaluation``.  Every key
#  prefixed 'eval_' or 'test_' is recorded verbatim on the fold/config entry by
#  ``_run_entry`` and aggregated to mean +/- std by ``_build_summary``.
#
#    cebe-gnn   evaluate_cebe_model  -> mae, r2, std  (legacy names, mapped below)
#    auger-gnn  evaluate_auger_model -> eval_gvx_pcc, test_gvc_mse, ...
#                 set          eval = experimental evaluation molecules
#                              test = calc hold-out molecules
#                 comparison   gvx  = GNN vs experiment
#                              gvc  = GNN vs calc
#                              cvx  = calc vs experiment (reference ceiling)
#
#  Adding a metric downstream requires no change here: it is picked up
#  automatically provided it is a scalar carrying one of the prefixes.

_EVAL_PREFIXES = ('eval_', 'test_')

# Historical CEBE key names, kept so existing summary JSONs stay readable.
_LEGACY_EVAL_ALIASES = {'mae': 'eval_mae', 'r2': 'eval_r2', 'std': 'eval_std'}

# Console columns per model type: (entry key, header, width, format)
#
# Printout only -- this does NOT control what is recorded.  Every scalar metric
# returned by run_evaluation is still written to the fold entries and to the CV
# summary JSON by _aggregate_eval_metrics, which filters on _EVAL_PREFIXES and
# ignores this table.  Add a column here purely to make a metric visible on the
# console; remove one and the metric is still in the JSON.
#
# Auger columns are deliberately calc-referenced only.  The experiment-referenced
# metrics (eval_gvx_*, eval_cvx_*) are recorded but not printed: the models
# predict broader spectra than the 1.6 eV-broadened calculation, and the
# experimental spectra are broader again, so agreement with experiment is
# confounded by that broadening and is not a basis for ranking models.
_EVAL_COLUMNS = {
    'cebe-gnn':  [('eval_mae',     'Exp MAE (eV)', 12, '.4f'),
                  ('eval_r2',      'Exp R2',        8, '.4f')],
    'auger-gnn': [('eval_gvc_pcc', 'PCC G-Calc',    10, '.4f'),
                  ('test_gvc_pcc', 'PCC G-Calc HO', 13, '.4f'),
                  ('test_gvc_mse', 'MSE G-Calc HO', 13, '.5f')],
}


def _collect_eval_metrics(eval_metrics) -> dict:
    """Extract the flat scalar metrics from a backend evaluation result.

    Non-scalars (``per_env``, ``per_molecule``, the nested eval/test summaries)
    are dropped -- they already live in ``{file_stem}_eval_results.json``.
    NaN becomes None so ``json.dump`` stays valid and the aggregation in
    ``_aggregate_eval_metrics`` can skip it.
    """
    if not isinstance(eval_metrics, dict):
        return {}
    out = {}
    for key, val in eval_metrics.items():
        key = _LEGACY_EVAL_ALIASES.get(key, key)
        if not key.startswith(_EVAL_PREFIXES):
            continue
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        val = float(val)
        out[key] = None if val != val else val        # NaN -> None
    return out


def _eval_columns(cfg, entries):
    """Console columns for the metrics actually present in *entries*."""
    return [c for c in _EVAL_COLUMNS.get(cfg.model, [])
            if any(isinstance(r.get(c[0]), float) for r in entries)]


# ─────────────────────────────────────────────────────────────────────────────
#  Filename construction (single source of truth for .pth paths)
# ─────────────────────────────────────────────────────────────────────────────

def _build_save_paths(
    cfg,
    fold: int,
    save_dir: str,
    *,
    prefix: str | None = None,
    config_id: str | None = None,
) -> Dict[str, str]:
    """Build the complete dict of ``.pth`` save paths for one training run.

    Parameters
    ----------
    cfg : AugerNetConfig
        Must have ``model_id`` already resolved.
    fold : int
        Current fold number (1-indexed).
    save_dir : str
        Directory where ``.pth`` files are written.
    prefix : str, optional
        Param-search identifier (e.g. ``"search_layer_type2_n_layers3"``).
    config_id : str, optional
        Per-configuration label (e.g. ``"cfg003"``).

    Returns
    -------
    dict
        Mapping of logical name to absolute path, e.g.::
            {'model': '/path/to/cebe_gnn_…_fold1.pth'}  

    Naming convention
    -----------------
    Normal:       ``{model_id}[_{tag}]_fold{fold}.pth``
    Param search: ``{prefix}_{model_id}[_{tag}]_fold{fold}_{config_id}.pth``
    """
    model_id = cfg.model_id

    def _fname(tag: str | None = None) -> str:
        stem = f"{model_id}_{tag}_fold{fold}" if tag else f"{model_id}_fold{fold}"
        if config_id:
            stem = f"{stem}_{config_id}"
        if prefix:
            stem = f"{prefix}_{stem}"
        return os.path.join(save_dir, f"{stem}.pth")

    return {'model': _fname()}

# ─────────────────────────────────────────────────────────────────────────────
#  Cartesian-product param grid expansion
# ─────────────────────────────────────────────────────────────────────────────

def _build_param_configs(param_grid: dict) -> List[dict]:
    """Expand a parameter grid dict into a list of config dicts."""
    keys = sorted(param_grid.keys())
    values = [param_grid[k] for k in keys]
    return [dict(zip(keys, combo))
            for combo in itertools.product(*values)]


def _cfg_with_overrides(cfg, overrides: dict):
    """Return a shallow copy of *cfg* with per-config override fields applied.

    Used by ``run_param_search`` to ensure that evaluation of each config
    uses the same parameter values (e.g. ``fwhm``, ``n_points``) that were
    used during training — not the base config values.
    """
    import copy
    cfg_copy = copy.copy(cfg)
    for k, v in overrides.items():
        if hasattr(cfg_copy, k):
            setattr(cfg_copy, k, v)
    return cfg_copy


# ─────────────────────────────────────────────────────────────────────────────
#  k-fold cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def run_kfold_cv(data, cfg) -> Dict[str, Any]:
    """
    Run full k-fold cross-validation.

    Trains one model per fold via backend.train_single_run, saves each
    model, and writes a JSON summary identifying the best fold.
    """

    be = _get_backend(cfg)
    n_folds  = cfg.n_folds

    fold_results = []

    print(f"\n{'#' * 80}")
    print(f"#  K-FOLD CROSS-VALIDATION  ({n_folds} folds)")
    print(f"{'#' * 80}")

    for fold in range(1, n_folds + 1):
        save_paths = _build_save_paths(cfg, fold, cfg.models_dir)
        result = be.train_single_run(
            data, fold, n_folds,
            save_paths=save_paths,
            output_dir=cfg.outputs_dir,
            cfg=cfg,
            verbose=True,
        )

        # Save loss curve and run evaluation for this fold
        eval_metrics = None
        if cfg.run_evaluation:
            eval_metrics = be.run_evaluation(
                result, data, fold,
                output_dir=cfg.outputs_dir, png_dir=cfg.pngs_dir, cfg=cfg,
                train_results=result.get('train_results'),
                exp_split=cfg.cebe_exp_split,  
            )

        # Build a JSON-serialisable record
        entry = _run_entry(result, eval_metrics=eval_metrics)
        entry['fold'] = fold
        fold_results.append(entry)

    # ── Identify best fold ───────────────────────────────────────────────
    best = min(fold_results, key=lambda r: r['best_val_loss'])

    # ── Print summary table ──────────────────────────────────────────────
    eval_cols = _eval_columns(cfg, fold_results)
    _print_cv_summary(fold_results, n_folds, best, eval_cols=eval_cols)

    # Mean +/- std over folds -- these are the manuscript CV-table values.
    combined = [r['best_val_loss'] for r in fold_results]
    print(f"\n  {'Val Loss':<16s} {np.mean(combined):.6f} +/- "
          f"{np.std(combined, ddof=1):.6f}  (n={len(combined)})")
    for key, header, _w, fmt in eval_cols:
        vals = [r[key] for r in fold_results if isinstance(r.get(key), float)]
        if vals:
            print(f"  {header:<16s} {np.mean(vals):{fmt}} +/- "
                  f"{np.std(vals, ddof=1):{fmt}}  (n={len(vals)})")
    print(f"  Best fold: Fold {best['fold']}  (loss={best['best_val_loss']:.6f})")

    # ── Save JSON summary ────────────────────────────────────────────────
    cv_summary = _build_summary(fold_results, cfg)
    cv_summary['n_folds'] = n_folds
    cv_summary['best_fold'] = best['fold']

    summary_path = os.path.join(cfg.result_dir, f'{cfg.model_id}_cv_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(cv_summary, f, indent=2, default=str)
    print(f"\nCV summary saved to: {summary_path}")

    return cv_summary


# ─────────────────────────────────────────────────────────────────────────────
#  Unified hyperparameter search
# ─────────────────────────────────────────────────────────────────────────────

def run_param_search(data, cfg) -> Dict[str, Any]:
    """
    Run hyperparameter search.

    For each combination in ``cfg.param_grid``, trains one fold via
    ``backend.train_single_run`` with overrides, records the best
    validation loss, and writes a sorted leaderboard JSON.
    """
    be = _get_backend(cfg)

    param_grid = cfg.param_grid
    if not param_grid:
        raise ValueError(
            "No param_grid defined in config. "
            "Add a 'param_grid' section to your YAML file."
        )

    fold      = cfg.train_fold
    n_folds   = cfg.n_folds

    configs = _build_param_configs(param_grid)
    n_configs = len(configs)

    # Build a unique search identifier for this grid
    search_id = _param_search_id(param_grid)

    # Cap epochs for search speed
    search_epochs  = min(cfg.num_epochs, 300)
    search_patience = min(cfg.patience, 40)

    print(f"\n{'#' * 80}")
    print(f"#  HYPERPARAMETER SEARCH  ({n_configs} configurations)")
    print(f"#  Fold {fold}/{n_folds}  |  max {search_epochs} epochs  |  patience {search_patience}")
    print(f"{'#' * 80}")

    print(f"\nSearch grid:")
    for k, v in sorted(param_grid.items()):
        print(f"  {k}: {v}")
    print()

    results = []
    t0_total = time.time()

    for i, config in enumerate(configs):
        config_id = f"cfg{i:03d}"

        # Overrides for this config (includes capped epochs)
        overrides = dict(config)
        overrides['num_epochs'] = search_epochs
        overrides['patience'] = search_patience

        print(f"\n{'─' * 70}")
        print(f"  Config {i+1}/{n_configs}  [{config_id}]")
        for k, v in sorted(config.items()):
            print(f"    {k}: {v}")
        print(f"{'─' * 70}")

        t0 = time.time()
        try:
            save_paths = _build_save_paths(
                cfg, fold, cfg.models_dir,
                prefix=search_id, config_id=config_id,
            )
            result = be.train_single_run(
                data, fold, n_folds,
                save_paths=save_paths,
                output_dir=cfg.models_dir,
                cfg=cfg,
                verbose=True,
                **overrides,
            )
            elapsed = time.time() - t0

            # Save loss curve and run evaluation for this fold.
            # Build a per-config cfg copy so spectrum params (e.g. fwhm)
            # used during evaluation match those used during training.
            eval_metrics = None
            if cfg.run_evaluation:
                eval_cfg = _cfg_with_overrides(cfg, config)
                eval_metrics = be.run_evaluation(
                    result, data, fold,
                    output_dir=cfg.outputs_dir, png_dir=cfg.pngs_dir, cfg=eval_cfg,
                    train_results=result.get('train_results'),
                    config_id=config_id,
                    param_file_prefix=search_id,
                    exp_split=cfg.cebe_exp_split,  
                )

            entry = _run_entry(result, eval_metrics=eval_metrics)
            entry.update({
                'config_id': config_id,
                'rank': 0,
                **config,
                'elapsed_sec': round(elapsed, 1),
                'status': 'ok',
            })

        except Exception as e:
            elapsed = time.time() - t0
            entry = {
                'model_id': cfg.model_id,
                'best_val_loss': float('inf'),
                'best_train_loss': None,
                'best_val_epoch': None,
                'n_epochs': 0,
                'model_path': None,
                'final_train_loss': None,
                'final_val_loss': None,
                'config_id': config_id,
                'rank': 999,
                **config,
                'elapsed_sec': round(elapsed, 1),
                'status': f'error: {e}',
            }
            print(f"ERROR: {e}")

        results.append(entry)

    total_elapsed = time.time() - t0_total

    # Sort by best_val_loss
    results.sort(key=lambda r: r['best_val_loss'])
    for rank, r in enumerate(results):
        r['rank'] = rank + 1

    # ── Leaderboard ──────────────────────────────────────────────────────
    eval_cols = _eval_columns(cfg, results)
    _print_param_leaderboard(results, n_configs, total_elapsed, param_grid,
                             eval_cols=eval_cols)

    best = results[0]
    print(f"\n  Best config: {best['config_id']}")
    for k in sorted(param_grid.keys()):
        print(f"      {k}: {best.get(k)}")
    print(f"      val_loss: {best['best_val_loss']:.6f}")
    for key, header, _w, fmt in eval_cols:
        v = best.get(key)
        if isinstance(v, float):
            print(f"      {header}: {v:{fmt}}")

    # ── Save JSON summary ────────────────────────────────────────────────
    summary = _build_summary(results, cfg)
    summary['search_id'] = search_id
    summary['n_configs'] = n_configs
    summary['search_epochs'] = search_epochs
    summary['search_patience'] = search_patience
    summary['total_elapsed_min'] = round(total_elapsed / 60, 1)
    summary['param_grid'] = {
        k: [str(v) if isinstance(v, float) else v for v in vals]
        for k, vals in param_grid.items()
    }
    summary['best_config_id'] = best['config_id']
    summary['best_params'] = {k: best.get(k)
                              for k in sorted(param_grid.keys())}

    summary_path = os.path.join(cfg.result_dir,
                                f'{search_id}_{cfg.model_id}_param_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved param search summary to: {summary_path}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: AugerNetConfig):
    """
    Execute a full training / evaluation / prediction run.

    Parameters
    ----------
    cfg : AugerNetConfig: resolved configuration from yml.
    """
    mode = cfg.mode
    model_name = cfg.model

    print(f"\n{'=' * 80}")
    print(f"  AugerNet: model={model_name}  mode={mode}")
    if cfg.model_id:
        print(f"  Model ID: {cfg.model_id}")
    print(f"{'=' * 80}")

    be = _get_backend(cfg)

    if mode == 'predict':
        _run_predict(cfg)
        print("\n Predictions Complete.")
        return

    # ── Load data ────────────────────────────────────────────────────────
    data = be.load_data(cfg)

    if cfg.model in ('auger-gnn', 'auger-cnn'):
        # Calc hold-out: 50 randomly selected molecules for both GNN and CNN,
        if cfg.model == 'auger-gnn':
            calc_data = data['calc_data']
            mol_order = [d.mol_name for d in calc_data]
        else:  # auger-cnn
            calc_mask = data['train_df']['source'] == 'calc'
            mol_order = list(dict.fromkeys(
                data['train_df'].loc[calc_mask, 'mol_name']
            ))

        from sklearn.model_selection import ShuffleSplit
        test_splitter = ShuffleSplit(n_splits=1, test_size=50, random_state=0)
        tr_arr, te_arr = next(test_splitter.split(mol_order))
        test_mol_names = {mol_order[i] for i in te_arr}

        if cfg.model == 'auger-gnn':
            data['calc_data'] = [calc_data[i] for i in tr_arr]
            data['test_data'] = [calc_data[i] for i in te_arr]

            print(f"\nGNN:\n  {len(data['calc_data'])} train+val mol, " 
                    f"{sum(s == 'C' for d in data['calc_data'] for s in d.atom_symbols)} carbons\n")

            print(f"  {len(data['test_data'])} calc test hold-mol (ShuffleSplit random_state=0), " 
                    f"{sum(s == 'C' for d in data['test_data'] for s in d.atom_symbols)} carbons\n")
        else:
            df = data['train_df']
            is_calc_test = (df['source'] == 'calc') & df['mol_name'].isin(test_mol_names)
            data['test_df']  = df[is_calc_test].reset_index(drop=True)
            data['train_df'] = df[~is_calc_test].reset_index(drop=True)
            if 'train_df_raw' in data:
                raw = data['train_df_raw']
                data['train_df_raw'] = raw[
                    ~((raw['source'] == 'calc') & raw['mol_name'].isin(test_mol_names))
                ].reset_index(drop=True)
            n_holdout_mols = data['test_df']['mol_name'].nunique()
            n_train_mols   = data['train_df'][
                data['train_df']['source'] == 'calc']['mol_name'].nunique()
            print(f"  CNN: {n_holdout_mols} hold-out mols "
                  f"({len(data['test_df'])} carbons), "
                  f"{n_train_mols} train+val calc mols remaining")

    result = None  # Set by train/cv for unit tests

    if mode == 'cv':
        cv_summary = run_kfold_cv(data, cfg)
        # Load the best-fold model for unit tests
        if getattr(cfg, 'run_unit_tests', False):
            best_fold = cv_summary['best_fold']
            save_paths = _build_save_paths(cfg, best_fold, cfg.models_dir)
            result = be.load_saved_model(save_paths, data, cfg)

    elif mode == 'train':
        save_paths = _build_save_paths(cfg, cfg.train_fold, cfg.models_dir)
        result = be.train_single_run(
            data, cfg.train_fold, cfg.n_folds,
            save_paths=save_paths,
            output_dir=cfg.outputs_dir,
            cfg=cfg,
            verbose=True,
        )

        if cfg.run_evaluation:
            be.run_evaluation(
                result, data, cfg.train_fold,
                output_dir=cfg.outputs_dir,
                png_dir=cfg.pngs_dir, cfg=cfg,
                train_results=result.get('train_results'),
            )

    elif mode == 'param':
        run_param_search(data, cfg)

    elif mode == 'evaluate':
        _run_evaluate(data, cfg)

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. "
            f"Choose from: cv, train, param, evaluate, predict"
        )

    # ── Unit tests ───────────────────────────────────────────────────────
    if getattr(cfg, 'run_unit_tests', False) and mode in ('train', 'cv'):
        if result is not None:
            try:
                be.run_unit_tests(result, data, cfg)
            except Exception:
                pass  # unit tests are optional

    print("\n AugerNet run complete\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluate mode (load existing model, run evaluation only)
# ─────────────────────────────────────────────────────────────────────────────

def _run_evaluate(data, cfg):
    """Load a saved model from ``model_path`` and evaluate on experimental data.

    The user specifies ``model_path`` (relative to cwd) in the YAML config.
    Results are written to ``evaluate_results/``.
    """

    be = _get_backend(cfg)

    model_path = cfg.model_path
    if not model_path:
        raise ValueError(
            "evaluate mode requires 'model_path' in the config YAML.\n"
            "  Example:  model_path: train_results/models/cebe_gnn_035_random_EQ3_h64_fold3.pth"
        )
    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"\n  Loading model from: {model_path}")

    if cfg.model == 'auger-cnn':
        # CNN backend: _load_model_from_path takes (path, data, cfg)
        model, device = be._load_model_from_path(model_path, data, cfg)
        #result = (model, device)
        result = {'model': model, 'device': device}
    else:
        # GNN backend (cebe-gnn or auger-gnn): 
        calc_data = data['calc_data']
        model, device = be._load_model_from_path(
            model_path, calc_data,
            layer_type=cfg.layer_type,
            hidden_channels=cfg.hidden_channels,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            **be._model_load_kwargs(cfg),
        )
        #result = (model, device)
        result = {'model': model, 'device': device}

    # Try to infer fold from filename (e.g. …_fold3.pth → 3)
    fold = _infer_fold_from_path(model_path)

    be.run_evaluation(
        result, data, fold,
        output_dir=cfg.outputs_dir, png_dir=cfg.pngs_dir, cfg=cfg,
    )


def _infer_fold_from_path(model_path: str):
    """Extract fold number from a model filename, or return None."""
    import re
    base = os.path.basename(model_path)
    m = re.search(r'_fold(\d+)', base)
    return int(m.group(1)) if m else None


# ─────────────────────────────────────────────────────────────────────────────
#  Predict mode (inference on arbitrary .xyz files)
# ─────────────────────────────────────────────────────────────────────────────

def _run_predict(cfg):
    """Run predictions on a directory of .xyz files using a saved model.

    Requires ``model_path`` and ``predict_dir`` in the YAML config.
    Builds molecular graphs on the fly from the .xyz files, runs inference,
    and writes ``_labels.txt`` and ``_results.txt`` output files.
    """

    be = _get_backend(cfg)

    model_path = cfg.model_path
    predict_dir = cfg.predict_dir

    if not model_path:
        raise ValueError(
            "predict mode requires 'model_path' in the config YAML.\n"
            "  Example:  model_path: train_results/models/cebe_gnn_035_random_EQ3_h64_fold3.pth"
        )
    if not predict_dir:
        raise ValueError(
            "predict mode requires 'predict_dir' in the config YAML.\n"
            "  Example:  predict_dir: my_molecules/"
        )

    model_path = os.path.abspath(model_path)
    predict_dir = os.path.abspath(predict_dir)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.isdir(predict_dir):
        raise FileNotFoundError(f"Predict directory not found: {predict_dir}")

    be.run_predict(
        model_path=model_path,
        predict_dir=predict_dir,
        cfg=cfg,
    )


def _aggregate_eval_metrics(entries: List[dict]) -> dict:
    """Mean / std / n over folds (or configs) for every recorded eval metric.

    Picks up any scalar key carrying an ``_EVAL_PREFIXES`` prefix, so a metric
    added downstream needs no change here.  The prefix filter is also what
    keeps param-search grid values (merged into the entry by ``**config``)
    out of the aggregation.

    Notes
    -----
    - ``ddof=1``: a k-fold spread is a sample estimate, not a population one.
      With ``n_folds=10`` the population std understates it by ~5%.
    - Folds where a metric is missing or NaN are skipped, and ``n_<key>`` is
      reported so a partially-populated metric is visible rather than silent.
    """
    keys: List[str] = []
    for r in entries:
        for k, v in r.items():
            if (k.startswith(_EVAL_PREFIXES) and k not in keys
                    and isinstance(v, (int, float)) and not isinstance(v, bool)):
                keys.append(k)

    agg: Dict[str, Any] = {}
    for k in keys:
        vals = [float(r[k]) for r in entries
                if isinstance(r.get(k), (int, float))
                and not isinstance(r[k], bool)
                and r[k] == r[k]]                       # drop NaN
        if not vals:
            continue
        agg[f'mean_{k}'] = float(np.mean(vals))
        agg[f'std_{k}']  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        agg[f'n_{k}']    = len(vals)
    return agg


def _build_summary(entries: List[dict], cfg) -> dict:
    """Build the common top-level JSON summary shared by CV and param search.

    Computes aggregate statistics (mean/std of val loss, train loss, and every
    evaluation metric recorded by ``_collect_eval_metrics``) and returns an
    ``OrderedDict``-style dict.  Callers add mode-specific keys (``n_folds``,
    ``param_grid``, etc.) after this returns.

    All spreads use ``ddof=1`` (sample std) -- these are the mean +/- std
    values reported in the manuscript CV table.
    """
    val_losses = [r['best_val_loss'] for r in entries]
    train_losses = [r['best_train_loss'] for r in entries
                    if r.get('best_train_loss') is not None]

    summary: Dict[str, Any] = {
        'model': cfg.model,
        'model_id': cfg.model_id,
        'feature_keys': cfg.feature_keys,
        'split_method': cfg.split_method,
        'n_runs':          len(entries),
        'mean_val_loss':   float(np.mean(val_losses)),
        'std_val_loss':    float(np.std(val_losses, ddof=1)) if len(val_losses) > 1 else 0.0,
        'mean_train_loss': float(np.mean(train_losses)) if train_losses else None,
        'std_train_loss':  (float(np.std(train_losses, ddof=1))
                            if len(train_losses) > 1 else
                            (0.0 if train_losses else None)),
        'best_val_loss':   float(min(val_losses)),
        'best_train_loss': None,
    }

    # best_train_loss corresponding to the run with the lowest val loss
    best_idx = int(np.argmin(val_losses))
    summary['best_train_loss'] = entries[best_idx].get('best_train_loss')

    # ── Aggregate every recorded evaluation metric (mean +/- std over folds) ──
    summary.update(_aggregate_eval_metrics(entries))

    summary['runs'] = entries
    return summary


def _run_entry(result: dict, eval_metrics: dict = None) -> dict:
    """Build the common JSON-serialisable record from a training result.

    Both CV folds and param-search configs share this base structure.
    Callers add ``fold`` or ``config_id`` / ``rank`` as needed.
"""
    entry = {
        'model_id': result.get('model_id', ''),
        'best_val_loss': result.get('best_val_loss', float('inf')),
        'best_train_loss': result.get('best_train_loss'),
        'best_val_epoch': result.get('best_val_epoch'),
        'n_epochs': result.get('n_epochs', 0),
        'model_path': result.get('model_path'),
        'final_train_loss': result.get('final_train_loss'),
        'final_val_loss': result.get('final_val_loss'),
    }

    entry.update(_collect_eval_metrics(eval_metrics))

    return entry


def _print_cv_summary(fold_results, n_folds, best, eval_cols=()):
    """Print CV summary table.

    ``eval_cols`` comes from ``_eval_columns(cfg, fold_results)`` -- a list of
    ``(entry_key, header, width, format)`` for the metrics actually populated,
    so the table adapts to cebe-gnn / auger-gnn without a boolean flag.
    """
    width = max(90, 40 + sum(w + 2 for _k, _h, w, _f in eval_cols))
    print(f"\n{'=' * width}")
    print(f"  K-FOLD CROSS-VALIDATION SUMMARY  ({n_folds} folds)")
    print(f"{'=' * width}")

    hdr = f"  {'Fold':>4}  {'Epochs':>6}  {'TrnLoss':>12}  {'ValLoss':>12}"
    sep = f"  {'─'*4}  {'─'*6}  {'─'*12}  {'─'*12}"
    for _key, header, w, _fmt in eval_cols:
        hdr += f"  {header:>{w}}"
        sep += f"  {'─'*w}"
    print(hdr)
    print(sep)

    for r in fold_results:
        m = ' best' if r['fold'] == best['fold'] else ''
        trn = f"{r['best_train_loss']:>12.6f}" if r.get('best_train_loss') is not None else f"{'—':>12}"
        line = (f"  {r['fold']:>4}  {r['n_epochs']:>6}  "
                f"{trn}  {r['best_val_loss']:>12.6f}")
        for key, _h, w, fmt in eval_cols:
            v = r.get(key)
            line += f"  {v:>{w}{fmt}}" if isinstance(v, float) else f"  {'—':>{w}}"
        print(f"{line}{m}")

    print(f"{'=' * width}")


def _print_param_leaderboard(results, n_configs, total_elapsed, param_grid,
                             eval_cols=()):
    """Print the top results from a param search."""
    width = max(110, 60 + 12 * len(param_grid)
                + sum(w + 2 for _k, _h, w, _f in eval_cols))
    print(f"\n{'=' * width}")
    print(f"  HYPERPARAMETER SEARCH LEADERBOARD  ({n_configs} configs)")
    print(f"  Total time: {total_elapsed/60:.1f} minutes")
    print(f"{'=' * width}")

    grid_keys = sorted(param_grid.keys())
    header = f"  {'Rank':>4}  {'ID':>6}"
    for k in grid_keys:
        header += f"  {k:>10}"
    header += f"  {'TrnLoss':>10}  {'ValLoss':>10}  {'Epochs':>6}  {'Time':>6}"
    sep = (f"  {'─'*4}  {'─'*6}" +
           ''.join(f"  {'─'*10}" for _ in grid_keys) +
           f"  {'─'*10}  {'─'*10}  {'─'*6}  {'─'*6}")
    for _key, hdr_label, w, _fmt in eval_cols:
        header += f"  {hdr_label:>{w}}"
        sep    += f"  {'─'*w}"
    print(header)
    print(sep)

    for r in results:
        if r['status'] != 'ok':
            continue
        line = f"  {r['rank']:>4}  {r['config_id']:>6}"
        for k in grid_keys:
            v = r.get(k, '')
            if isinstance(v, float):
                line += f"  {v:>10.5f}"
            else:
                line += f"  {str(v):>10}"
        trn = f"{r['best_train_loss']:>10.6f}" if r.get('best_train_loss') is not None else f"{'—':>10}"
        line += (f"  {trn}  {r['best_val_loss']:>10.6f}  "
                 f"{r.get('n_epochs',0):>6}  "
                 f"{r['elapsed_sec']:>5.0f}s")
        for key, _h, w, fmt in eval_cols:
            v = r.get(key)
            line += f"  {v:>{w}{fmt}}" if isinstance(v, float) else f"  {'—':>{w}}"
        print(line)

    print(f"{'=' * width}")


def _param_search_id(param_grid: dict) -> str:
    """Build a unique search identifier for param-search output filenames.

    The id encodes the searched dimensions only — each searched parameter
    name and the number of values explored.  The fixed hyperparameters are
    already captured in the per-config ``model_id``, so repeating them here
    would cause duplication in filenames.

    Example
    -------
    Searching ``layer_type`` (2 values) and ``n_layers`` (3 values):

    → ``search_layer_type2_n_layers3``

    The searched parameter names are sorted alphabetically.
    """
    grid_keys = sorted(param_grid.keys())

    search_parts = []
    for k in grid_keys:
        n_vals = len(param_grid[k])
        search_parts.append(f"{k}{n_vals}")

    return f"search_{'_'.join(search_parts)}"
