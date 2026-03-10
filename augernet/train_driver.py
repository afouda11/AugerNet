"""
AugerNet Training Driver
=========================

Contains ``run_kfold_cv``, ``run_param_search``, ``_build_param_configs``,
and the mode-dispatch logic for CEBE-GNN training.

Model-specific behaviour is provided by the backend module:
  - ``augernet.backend_cebe``  (CEBE binding-energy GNN)

The backend exports hooks:
  load_data(cfg)                → data dict
  train_single_run(data, …)    → result dict
  load_saved_model(…)           → (model, device) or result dict
  run_evaluation(…)             → None
  run_unit_tests(…)             → None
  run_predict(…)                → None

Usage (from ``__main__.py``)::

    from augernet.train_driver import run
    from augernet.config import load_config

    cfg = load_config('configs/cebe_default.yml')
    run(cfg)
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
#  k-fold cross-validation
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
    os.makedirs(models_dir,  exist_ok=True)

    outputs_dir = os.path.join(cv_dir, 'outputs')
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
        'model_id': cfg.model_id,
        'feature_tag': cfg.feature_tag,
        'split_method': cfg.split_method,
        'best_fold_by_loss': best['fold'],
        'mean_val_loss': float(np.mean(combined)),
        'std_val_loss':  float(np.std(combined)),
        'folds': fold_results,
    }
    summary_path = os.path.join(cv_dir, f'{cfg.model_id}_cv_summary.json')
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
                'model_id': result.get('model_id', cfg.model_id),
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
        'model_id': cfg.model_id,
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
        'leaderboard': results,
    }
    summary_path = os.path.join(param_dir,
                                f'{cfg.model_id}_param_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n✓ Saved param search summary → {summary_path}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: AugerNetConfig):
    """
    Execute a full training / evaluation / prediction run.

    Parameters
    ----------
    cfg : AugerNetConfig
        Fully resolved configuration.
    """
    mode = cfg.mode
    model_name = cfg.model

    print(f"\n{'=' * 80}")
    print(f"  AugerNet — model={model_name}  mode={mode}")
    if cfg.model_id:
        print(f"  Model ID: {cfg.model_id}")
    print(f"{'=' * 80}")

    backend = _get_backend(model_name)

    # ── Modes that do NOT need the full training dataset ─────────────────
    if mode == 'predict':
        _run_predict(backend, cfg)
        print("\n✓ Done.")
        return

    # ── Load data ────────────────────────────────────────────────────────
    data = backend.load_data(cfg)

    # ── Dispatch ─────────────────────────────────────────────────────────
    result = None  # may be set by train/cv for unit tests

    if mode == 'cv':
        cv_summary = run_kfold_cv(backend, data, cfg)
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
        run_param_search(backend, data, cfg)

    elif mode == 'evaluate':
        _run_evaluate(backend, data, cfg)

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. "
            f"Choose from: cv, train, param, evaluate, predict"
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
#  Evaluate mode (load existing model, run evaluation only)
# ─────────────────────────────────────────────────────────────────────────────

def _run_evaluate(backend, data, cfg):
    """Load a saved model from ``model_path`` and evaluate on experimental data.

    The user specifies ``model_path`` (relative to cwd) in the YAML config.
    Results are written to ``evaluate_results/``.
    """
    model_path = cfg.model_path
    if not model_path:
        raise ValueError(
            "evaluate mode requires 'model_path' in the config YAML.\n"
            "  Example:  model_path: train_results/models/cebe_035_random_EQ3_h64_fold3.pth"
        )
    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    evaluate_dir = cfg.evaluate_dir
    outputs_dir = os.path.join(evaluate_dir, 'outputs')
    png_dir     = os.path.join(evaluate_dir, 'pngs')
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(png_dir,     exist_ok=True)

    print(f"\n  Loading model from: {model_path}")

    calc_data = data['calc_data']
    model, device = backend._load_model_from_path(
        model_path, calc_data,
        layer_type=cfg.layer_type,
        hidden_channels=cfg.hidden_channels,
        n_layers=cfg.n_layers,
    )

    # Try to infer fold from filename (e.g. …_fold3.pth → 3)
    fold = _infer_fold_from_path(model_path)

    backend.run_evaluation(
        (model, device), data, fold,
        output_dir=outputs_dir, png_dir=png_dir, cfg=cfg,
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

def _run_predict(backend, cfg):
    """Run predictions on a directory of .xyz files using a saved model.

    Requires ``model_path`` and ``predict_dir`` in the YAML config.
    Builds molecular graphs on the fly from the .xyz files, runs inference,
    and writes ``_labels.txt`` and ``_results.txt`` output files.
    """
    model_path = cfg.model_path
    predict_dir = cfg.predict_dir

    if not model_path:
        raise ValueError(
            "predict mode requires 'model_path' in the config YAML.\n"
            "  Example:  model_path: train_results/models/cebe_035_random_EQ3_h64_fold3.pth"
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

    fold = _infer_fold_from_path(model_path)

    backend.run_predict(
        model_path=model_path,
        predict_dir=predict_dir,
        fold=fold,
        cfg=cfg,
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
        'final_train_loss': result.get('final_train_loss'),
        'final_val_loss': result.get('final_val_loss'),
    }
    return entry


def _print_cv_summary(fold_results, n_folds, best):
    """Print a human-readable CV summary table."""
    print(f"\n{'=' * 90}")
    print(f"  K-FOLD CROSS-VALIDATION SUMMARY  ({n_folds} folds)")
    print(f"{'=' * 90}")

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
    """Rename model file to include config_id, preventing overwrites."""
    path = result.get('model_path')
    if not path or not os.path.exists(path):
        return
    base, ext = os.path.splitext(path)
    new_path = f"{base}_{config_id}{ext}"
    if path != new_path:
        os.rename(path, new_path)
        result['model_path'] = new_path
