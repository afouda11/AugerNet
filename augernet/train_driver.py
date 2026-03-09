"""
AugerNet Unified Training Driver
=================================

Contains the **ONLY** instances of ``run_kfold_cv``, ``run_param_search``,
``_build_param_configs``, and the mode-dispatch logic.  All three model
types (cebe-gnn, auger-gnn, auger-cnn) share this single implementation.

Model-specific behaviour is provided by backend modules:
  - ``augernet.backend_cebe``       (CEBE binding-energy GNN)
  - ``augernet.backend_auger_gnn``  (Auger spectrum GNN, stick + fitted)
  - ``augernet.backend_auger_cnn``  (Auger CNN carbon-env classifier)

Each backend exports six hooks:
  load_data(cfg)                → data dict
  train_single_run(data, …)    → result dict
  load_saved_model(…)           → (model, device) or result dict
  load_param_model(…)           → (model, device) or result dict
  run_evaluation(…)             → None
  run_unit_tests(…)             → None

Usage (from ``__main__.py``)::

    from augernet.train_driver import run
    from augernet.config import load_config

    cfg = load_config('configs/cebe_default.yml', overrides={'mode': 'cv'})
    run(cfg)
"""

from __future__ import annotations

import os
import sys
import json
import time
import itertools
import numpy as np
from typing import Any, Dict, List

from augernet.config import AugerNetConfig

from augernet import PROJECT_ROOT, DATA_RAW_DIR, DATA_PROCESSED_DIR

# ─────────────────────────────────────────────────────────────────────────────
#  Backend registry
# ─────────────────────────────────────────────────────────────────────────────

def _get_backend(model_name: str):
    """Return the backend module for the given model type."""
    if model_name == 'cebe-gnn':
        from augernet import backend_cebe as backend
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Only cebe-gnn currently available"
            #f"Choose from: cebe-gnn, auger-gnn, auger-cnn"
        )
    return backend


# ─────────────────────────────────────────────────────────────────────────────
#  Cartesian-product param grid expansion
# ─────────────────────────────────────────────────────────────────────────────

def _build_param_configs(param_grid: dict) -> List[dict]:
    """Expand a parameter grid dict into a list of config dicts."""
    keys = sorted(param_grid.keys())
    values = [param_grid[k] for k in keys]
    return [dict(zip(keys, combo))
            for combo in itertools.product(*values)]


# ─────────────────────────────────────────────────────────────────────────────
#  Unified k-fold cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def run_kfold_cv(backend, data, cfg) -> Dict[str, Any]:
    """
    Run full k-fold cross-validation.

    Trains one model per fold via ``backend.train_single_run``, saves each
    model, and writes a JSON summary identifying the best fold.
    """
    cv_dir   = cfg.cv_dir
    n_folds  = cfg.n_folds
    os.makedirs(cv_dir, exist_ok=True)

    models_dir  = os.path.join(cv_dir, 'models')
    outputs_dir = os.path.join(cv_dir, 'outputs')
    os.makedirs(models_dir,  exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    pngs_dir = os.path.join(cv_dir, 'pngs')
    os.makedirs(pngs_dir, exist_ok=True)

    fold_results = []

    print(f"\n{'#' * 80}")
    print(f"#  K-FOLD CROSS-VALIDATION  ({n_folds} folds)")
    print(f"{'#' * 80}")

    for fold in range(1, n_folds + 1):
        result = backend.train_single_run(
            data, fold, n_folds,
            save_dir=models_dir,
            output_dir=outputs_dir,
            cfg=cfg,
            verbose=True,
        )

        # Save loss curve and run evaluation for this fold
        if cfg.run_evaluation:
            backend.run_evaluation(
                result, data, fold,
                output_dir=outputs_dir, png_dir=pngs_dir, cfg=cfg,
                train_results=result.get('train_results'),
            )

        # Build a JSON-serialisable record
        entry = _fold_entry(result, fold)
        fold_results.append(entry)

    # ── Identify best fold ───────────────────────────────────────────────
    best = min(fold_results, key=lambda r: r['combined_val_loss'])

    # ── Print summary table ──────────────────────────────────────────────
    _print_cv_summary(fold_results, n_folds, best)

    combined = [r['combined_val_loss'] for r in fold_results]
    print(f"\n  Mean Val Loss: {np.mean(combined):.6f} ± {np.std(combined):.6f}")
    print(f"  Best fold: Fold {best['fold']}  (loss={best['combined_val_loss']:.6f})")

    # ── Save JSON summary ────────────────────────────────────────────────
    cv_summary = {
        'n_folds': n_folds,
        'model': cfg.model,
        'feature_tag': cfg.feature_tag,
        'model_tag': cfg.model_tag,
        'split_method': cfg.split_method,
        'best_fold_by_loss': best['fold'],
        'mean_val_loss': float(np.mean(combined)),
        'std_val_loss':  float(np.std(combined)),
        'folds': fold_results,
    }
    tag = cfg.model_tag or cfg.feature_tag or cfg.model
    summary_path = os.path.join(cv_dir, f'cv_summary_{tag}.json')
    with open(summary_path, 'w') as f:
        json.dump(cv_summary, f, indent=2, default=str)
    print(f"\n✓ Saved CV summary → {summary_path}")

    return cv_summary


# ─────────────────────────────────────────────────────────────────────────────
#  Unified hyperparameter search
# ─────────────────────────────────────────────────────────────────────────────

def run_param_search(backend, data, cfg) -> Dict[str, Any]:
    """
    Run hyperparameter search.

    For each combination in ``cfg.param_grid``, trains one fold via
    ``backend.train_single_run`` with overrides, records the best
    validation loss, and writes a sorted leaderboard JSON.
    """
    param_grid = cfg.param_grid
    if not param_grid:
        raise ValueError(
            "No param_grid defined in config. "
            "Add a 'param_grid' section to your YAML file."
        )

    param_dir = cfg.param_dir
    fold      = cfg.train_fold
    n_folds   = cfg.n_folds
    os.makedirs(param_dir, exist_ok=True)

    models_dir = os.path.join(param_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    outputs_dir = os.path.join(param_dir, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)

    pngs_dir = os.path.join(param_dir, 'pngs')
    os.makedirs(pngs_dir, exist_ok=True)

    configs = _build_param_configs(param_grid)
    n_configs = len(configs)

    # Cap epochs for search speed
    search_epochs  = min(cfg.num_epochs, 300)
    search_patience = min(cfg.patience, 40)

    print(f"\n{'#' * 80}")
    print(f"#  HYPERPARAMETER SEARCH  ({n_configs} configurations)")
    print(f"#  Fold {fold}/{n_folds}  •  max {search_epochs} epochs  •  patience {search_patience}")
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
            result = backend.train_single_run(
                data, fold, n_folds,
                save_dir=models_dir,
                output_dir=models_dir,
                cfg=cfg,
                verbose=False,
                **overrides,
            )
            elapsed = time.time() - t0

            # Rename model file to include config_id (avoid overwrites)
            _rename_model_file(result, config_id)

            # Save loss curve and run evaluation for this fold
            if cfg.run_evaluation:
                backend.run_evaluation(
                    result, data, fold,
                    output_dir=outputs_dir, png_dir=pngs_dir, cfg=cfg,
                    train_results=result.get('train_results'),
                )

            entry = {
                'config_id': config_id,
                'rank': 0,
                **config,
                'combined_val_loss': result['combined_val_loss'],
                'best_val_loss': result.get('best_val_loss',
                                            result['combined_val_loss']),
                'n_epochs': result.get('n_epochs', 0),
                'model_path': result.get('model_path'),
                'elapsed_sec': round(elapsed, 1),
                'status': 'ok',
            }

        except Exception as e:
            elapsed = time.time() - t0
            entry = {
                'config_id': config_id,
                'rank': 999,
                **config,
                'combined_val_loss': float('inf'),
                'best_val_loss': float('inf'),
                'n_epochs': 0,
                'model_path': None,
                'elapsed_sec': round(elapsed, 1),
                'status': f'error: {e}',
            }
            print(f"  ✗ ERROR: {e}")

        results.append(entry)

    total_elapsed = time.time() - t0_total

    # Sort by combined_val_loss
    results.sort(key=lambda r: r['combined_val_loss'])
    for rank, r in enumerate(results):
        r['rank'] = rank + 1

    # ── Leaderboard ──────────────────────────────────────────────────────
    _print_param_leaderboard(results, n_configs, total_elapsed, param_grid)

    best = results[0]
    print(f"\n  ★ Best config: {best['config_id']}")
    for k in sorted(param_grid.keys()):
        print(f"      {k}: {best.get(k)}")
    print(f"      val_loss: {best['combined_val_loss']:.6f}")

    # ── Save JSON summary ────────────────────────────────────────────────
    summary = {
        'model': cfg.model,
        'fold': fold,
        'n_folds': n_folds,
        'n_configs': n_configs,
        'search_epochs': search_epochs,
        'search_patience': search_patience,
        'total_elapsed_min': round(total_elapsed / 60, 1),
        'param_grid': {k: [str(v) if isinstance(v, float) else v
                           for v in vals]
                       for k, vals in param_grid.items()},
        'best_config_id': best['config_id'],
        'best_val_loss': best['combined_val_loss'],
        'best_params': {k: best.get(k) for k in sorted(param_grid.keys())},
        'feature_tag': cfg.feature_tag,
        'model_tag': cfg.model_tag,
        'leaderboard': results,
    }
    tag = cfg.model_tag or cfg.feature_tag or cfg.model
    summary_path = os.path.join(param_dir,
                                f'param_search_summary_{tag}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n✓ Saved param search summary → {summary_path}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: AugerNetConfig):
    """
    Execute a full training / evaluation / param-search run.

    Parameters
    ----------
    cfg : AugerNetConfig
        Fully resolved configuration.
    """
    mode = cfg.mode
    model_name = cfg.model

    print(f"\n{'=' * 80}")
    print(f"  AugerNet — model={model_name}  mode={mode}")
    if cfg.model_tag:
        print(f"  Model tag: {cfg.model_tag}")
    print(f"{'=' * 80}")

    backend = _get_backend(model_name)

    # ── Load data ────────────────────────────────────────────────────────
    data = backend.load_data(cfg)

    # ── Dispatch ─────────────────────────────────────────────────────────
    result = None  # may be set by train/cv for unit tests

    if mode == 'cv':
        cv_summary = run_kfold_cv(backend, data, cfg)
        # Evaluation per fold is handled inside run_kfold_cv now
        # Load the best-fold model for unit tests
        if getattr(cfg, 'run_unit_tests', False):
            best_fold = cv_summary['best_fold_by_loss']
            models_dir = os.path.join(cfg.cv_dir, 'models')
            result = backend.load_saved_model(models_dir, best_fold, data, cfg)

    elif mode == 'train':
        train_models_dir  = os.path.join(cfg.train_dir, 'models')
        train_outputs_dir = os.path.join(cfg.train_dir, 'outputs')
        train_pngs_dir    = os.path.join(cfg.train_dir, 'pngs')
        os.makedirs(train_models_dir,  exist_ok=True)
        os.makedirs(train_outputs_dir, exist_ok=True)
        os.makedirs(train_pngs_dir,    exist_ok=True)

        result = backend.train_single_run(
            data, cfg.train_fold, cfg.n_folds,
            save_dir=train_models_dir,
            output_dir=train_outputs_dir,
            cfg=cfg,
            verbose=True,
        )

        if cfg.run_evaluation:
            backend.run_evaluation(
                result, data, cfg.train_fold,
                output_dir=train_outputs_dir,
                png_dir=train_pngs_dir, cfg=cfg,
                train_results=result.get('train_results'),
            )

    elif mode == 'param':
        summary = run_param_search(backend, data, cfg)

        # Auto-evaluate the best config
        best = summary['leaderboard'][0]
        model_path = best.get('model_path') 
        if model_path and os.path.exists(model_path):
            print(f"\n{'=' * 70}")
            print(f"  AUTO-EVALUATING BEST CONFIG: {best['config_id']}")
            print(f"{'=' * 70}")
            best_params = summary['best_params']

            model_result = backend.load_param_model(
                model_path, data, cfg, best_params)
            param_outputs = os.path.join(cfg.param_dir, 'outputs')
            param_pngs    = os.path.join(cfg.param_dir, 'pngs')
            os.makedirs(param_outputs, exist_ok=True)
            os.makedirs(param_pngs, exist_ok=True)
            backend.run_evaluation(
                model_result, data, cfg.train_fold,
                output_dir=param_outputs, png_dir=param_pngs, cfg=cfg,
            )

    elif mode == 'evaluate_cv':
        _run_evaluate(backend, data, cfg, results_dir=cfg.cv_dir)

    elif mode == 'evaluate_train':
        _run_evaluate(backend, data, cfg, results_dir=cfg.train_dir)

    elif mode == 'evaluate_param':
        _run_evaluate(backend, data, cfg, results_dir=cfg.param_dir)

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. "
            f"Choose from: cv, train, param, evaluate_cv, "
            f"evaluate_train, evaluate_param"
        )

    # ── Unit tests ───────────────────────────────────────────────────────
    if getattr(cfg, 'run_unit_tests', False) and mode in ('train', 'cv'):
        if result is not None:
            try:
                backend.run_unit_tests(result, data, cfg)
            except Exception:
                pass  # unit tests are optional

    print("\n✓ Done.")


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluate modes (load existing model, run evaluation only)
# ─────────────────────────────────────────────────────────────────────────────

def _run_evaluate(backend, data, cfg, results_dir: str):
    """Load a saved model from results_dir and evaluate."""
    models_dir  = os.path.join(results_dir, 'models')
    outputs_dir = os.path.join(results_dir, 'outputs')
    png_dir     = os.path.join(results_dir, 'pngs')
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(png_dir,     exist_ok=True)

    model_result = backend.load_saved_model(
        models_dir, cfg.train_fold, data, cfg)
    backend.run_evaluation(
        model_result, data, cfg.train_fold,
        output_dir=outputs_dir, png_dir=png_dir, cfg=cfg,
    )


def _fold_entry(result: dict, fold: int) -> dict:
    """Extract a JSON-serialisable fold record from a training result dict."""
    entry = {
        'fold': fold,
        'combined_val_loss': result.get('combined_val_loss',
                                        result.get('best_val_loss', float('inf'))),
        'best_val_loss': result.get('best_val_loss', float('inf')),
        'n_epochs': result.get('n_epochs', 0),
        'model_path': result.get('model_path'),
    }
    # Carry through any extra keys the backend provides
    for extra in ('sing_best_val_loss', 'trip_best_val_loss',
                  'n_epochs_sing', 'n_epochs_trip',
                  'sing_model_path', 'trip_model_path',
                  'final_train_loss', 'final_val_loss',
                  'best_val_acc', 'final_train_acc', 'final_val_acc'):
        if extra in result:
            entry[extra] = result[extra]
    return entry


def _print_cv_summary(fold_results, n_folds, best):
    """Print a human-readable CV summary table."""
    print(f"\n{'=' * 90}")
    print(f"  K-FOLD CROSS-VALIDATION SUMMARY  ({n_folds} folds)")
    print(f"{'=' * 90}")

    # Detect which columns are available
    has_acc = any('best_val_acc' in r for r in fold_results)
    has_stick = any('sing_best_val_loss' in r for r in fold_results)

    if has_stick:
        print(f"  {'Fold':>4}  {'SingEp':>6}  {'TripEp':>6}  "
              f"{'Sing Loss':>10}  {'Trip Loss':>10}  {'Combined':>10}")
        print(f"  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}")
        for r in fold_results:
            m = ' ◀ best' if r['fold'] == best['fold'] else ''
            print(f"  {r['fold']:>4}  {r.get('n_epochs_sing',0):>6}  "
                  f"{r.get('n_epochs_trip',0):>6}  "
                  f"{r.get('sing_best_val_loss',0):>10.6f}  "
                  f"{r.get('trip_best_val_loss',0):>10.6f}  "
                  f"{r['combined_val_loss']:>10.6f}{m}")
    elif has_acc:
        print(f"  {'Fold':>4}  {'Epochs':>6}  {'Train Acc':>10}  {'Val Acc':>10}  "
              f"{'Val Loss':>10}")
        print(f"  {'─'*4}  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}")
        for r in fold_results:
            m = ' ◀ best' if r['fold'] == best['fold'] else ''
            print(f"  {r['fold']:>4}  {r['n_epochs']:>6}  "
                  f"{r.get('final_train_acc',0):>9.2f}%  "
                  f"{r.get('final_val_acc',0):>9.2f}%  "
                  f"{r['best_val_loss']:>10.4f}{m}")
    else:
        print(f"  {'Fold':>4}  {'Epochs':>6}  {'Val Loss':>12}")
        print(f"  {'─'*4}  {'─'*6}  {'─'*12}")
        for r in fold_results:
            m = ' ◀ best' if r['fold'] == best['fold'] else ''
            print(f"  {r['fold']:>4}  {r['n_epochs']:>6}  "
                  f"{r['best_val_loss']:>12.6f}{m}")

    print(f"{'=' * 90}")


def _print_param_leaderboard(results, n_configs, total_elapsed, param_grid):
    """Print the top results from a param search."""
    print(f"\n{'=' * 100}")
    print(f"  HYPERPARAMETER SEARCH — LEADERBOARD  ({n_configs} configs)")
    print(f"  Total time: {total_elapsed/60:.1f} minutes")
    print(f"{'=' * 100}")

    grid_keys = sorted(param_grid.keys())
    header = f"  {'Rank':>4}  {'ID':>6}"
    for k in grid_keys:
        header += f"  {k:>10}"
    header += f"  {'ValLoss':>10}  {'Epochs':>6}  {'Time':>6}"
    print(header)
    print(f"  {'─'*4}  {'─'*6}" +
          ''.join(f"  {'─'*10}" for _ in grid_keys) +
          f"  {'─'*10}  {'─'*6}  {'─'*6}")

    for r in results[:20]:
        if r['status'] != 'ok':
            continue
        line = f"  {r['rank']:>4}  {r['config_id']:>6}"
        for k in grid_keys:
            v = r.get(k, '')
            if isinstance(v, float):
                line += f"  {v:>10.5f}"
            else:
                line += f"  {str(v):>10}"
        line += (f"  {r['combined_val_loss']:>10.6f}  "
                 f"{r.get('n_epochs',0):>6}  "
                 f"{r['elapsed_sec']:>5.0f}s")
        print(line)

    print(f"{'=' * 100}")


def _rename_model_file(result: dict, config_id: str):
    """Rename model file(s) to include config_id, preventing overwrites."""
    for key in ('model_path', 'sing_model_path', 'trip_model_path'):
        path = result.get(key)
        if not path or not os.path.exists(path):
            continue
        base, ext = os.path.splitext(path)
        new_path = f"{base}_{config_id}{ext}"
        if path != new_path:
            os.rename(path, new_path)
            result[key] = new_path
