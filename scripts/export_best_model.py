"""
Export Best Model
=================

Locates the best model weights from a completed training run and exports
them, along with diagnostic plots and the training config, into the tracked
``artifacts/`` directory for release.

The results directory type is auto-detected from the summary JSON present:

  ``*_cv_summary.json``    → CV mode    (best fold by val loss)
  ``*_param_summary.json`` → Param mode (best config by val loss)
  no summary               → Train mode (single model in models/)

Usage
-----
    # Auto-select best fold from a CV run:
    uv run python scripts/export_best_model.py

    # Auto-select from param search or train results:
    uv run python scripts/export_best_model.py --results-dir results/param
    uv run python scripts/export_best_model.py --results-dir results/train

    # Overwrite existing artifacts:
    uv run python scripts/export_best_model.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
import re


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='export_best_model'examples/gnn_cebe_configs/train.yml,
        description='Export the best model weights and plots to artifacts/.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--results-dir', default='results/cv',
        help=(
            'Path to the results directory to export from '
            '(default: results/cv). Accepts cv, param, or train directories.'
        ),
    )
    parser.add_argument(
        '--out-dir', default='artifacts',
        help='Destination root directory (default: artifacts)',
    )
    parser.add_argument(
        '--config', default='config_examples/cv.yml',
        help='Path to the config YAML to include in artifacts (default: config_examples/cv.yml)',
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Overwrite existing artifacts without error',
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Auto-detection
# ─────────────────────────────────────────────────────────────────────────────

def _find_best_from_results(results_dir: Path):
    """Detect the results type and dispatch to the appropriate resolver.

    Returns ``(weights_path, stem, info, plots_required)``.
    ``info`` is a dict with display fields that varies by mode.
    ``plots_required`` controls whether missing plot files are a hard error.
    """
    if sorted(results_dir.glob('*_cv_summary.json')):
        return _resolve_cv(results_dir)
    if sorted(results_dir.glob('*_param_summary.json')):
        return _resolve_param(results_dir)
    return _resolve_train(results_dir)


def _resolve_cv(results_dir: Path):
    summaries = sorted(results_dir.glob('*_cv_summary.json'))
    if len(summaries) > 1:
        names = ', '.join(s.name for s in summaries)
        sys.exit(
            f"ERROR: Multiple CV summary files in {results_dir}:\n  {names}\n"
            f"Remove all but the one you want to export."
        )
    summary   = json.loads(summaries[0].read_text())
    model_id  = summary['model_id']
    best_fold = summary['best_fold']
    stem         = f'{model_id}_fold{best_fold}'
    weights_path = results_dir / 'models' / f'{stem}.pth'
    info = {
        'mode':         'cv',
        'summary_path': summaries[0],
        'model_id':     model_id,
        'best_fold':    best_fold,
        'val_loss':     summary['best_val_loss'],
        'eval_mae':     summary.get('mean_eval_mae'),
    }
    return weights_path, stem, info, True


def _resolve_param(results_dir: Path):
    summaries = sorted(results_dir.glob('*_param_summary.json'))
    if len(summaries) > 1:
        names = ', '.join(s.name for s in summaries)
        sys.exit(
            f"ERROR: Multiple param summary files in {results_dir}:\n  {names}\n"
            f"Remove all but the one you want to export."
        )
    summary  = json.loads(summaries[0].read_text())
    model_id = summary['model_id']
    # runs are sorted by val_loss ascending; index 0 is the best config
    best_run    = summary['runs'][0]
    stored_path = best_run.get('model_path', '')
    if not stored_path:
        sys.exit("ERROR: best run in param summary has no model_path recorded.")
    # Use only the filename — the stored absolute path may be stale
    stem         = Path(stored_path).stem
    weights_path = results_dir / 'models' / f'{stem}.pth'
    info = {
        'mode':           'param',
        'summary_path':   summaries[0],
        'model_id':       model_id,
        'best_config_id': summary.get('best_config_id'),
        'best_params':    summary.get('best_params', {}),
        'val_loss':       summary['best_val_loss'],
        'eval_mae':       summary.get('mean_eval_mae'),
    }
    return weights_path, stem, info, True


def _resolve_train(results_dir: Path):
    models_dir = results_dir / 'models'
    if not models_dir.exists():
        sys.exit(
            f"ERROR: No summary JSON and no models/ directory found in {results_dir}.\n"
            f"Is this a valid results directory?"
        )
    models = sorted(models_dir.glob('*.pth'))
    if not models:
        sys.exit(f"ERROR: No .pth files found in {models_dir}")
    if len(models) > 1:
        names = ', '.join(m.name for m in models)
        sys.exit(
            f"ERROR: Multiple .pth files in {models_dir}:\n  {names}\n"
            f"Pass --results-dir pointing to a single-model directory."
        )
    weights_path = models[0]
    stem = weights_path.stem
    info = {
        'mode':       'train',
        'model_file': weights_path,
    }
    return weights_path, stem, info, False  # plots optional for train


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_info(info: dict) -> None:
    mode = info['mode']
    if mode == 'cv':
        print('  Export Best CV Model')
        print('=' * 60)
        print(f"  Summary:   {info['summary_path']}")
        print(f"  Model ID:  {info['model_id']}")
        print(f"  Best fold: {info['best_fold']}")
        print(f"  Val loss:  {info['val_loss']:.6f}")
        if info.get('eval_mae') is not None:
            print(f"  Exp MAE:   {info['eval_mae']:.4f} eV")
    elif mode == 'param':
        print('  Export Best Param Search Model')
        print('=' * 60)
        print(f"  Summary:     {info['summary_path']}")
        print(f"  Model ID:    {info['model_id']}")
        print(f"  Best config: {info['best_config_id']}")
        for k, v in sorted((info.get('best_params') or {}).items()):
            print(f"    {k}: {v}")
        print(f"  Val loss:    {info['val_loss']:.6f}")
        if info.get('eval_mae') is not None:
            print(f"  Exp MAE:     {info['eval_mae']:.4f} eV")
    elif mode == 'train':
        print('  Export Train Model')
        print('=' * 60)
        print(f"  Model file: {info['model_file']}")


def _copy(src: Path, dst: Path, overwrite: bool) -> None:
    if not src.exists():
        sys.exit(f"ERROR: Expected file not found: {src}")
    if dst.exists() and not overwrite:
        sys.exit(
            f"ERROR: Destination already exists: {dst}\n"
            f"Use --overwrite to replace it."
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  copied  {src.name}  →  {dst}")


def _try_copy(src: Path, dst: Path, overwrite: bool) -> None:
    """Like _copy, but prints a warning instead of exiting when src is absent."""
    if not src.exists():
        print(f"  skipped {src.name}  (not found)")
        return
    _copy(src, dst, overwrite)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    config   = Path(args.config)

    print()
    print('=' * 60)

    weights_path, stem, info, plots_required = _find_best_from_results(results_dir)
    plots_src_dir = results_dir / 'pngs'
    copy_plot     = _copy if plots_required else _try_copy
    _print_info(info)

    print()

    # ── Model weights ────────────────────────────────────────────────────────
    print('Model weights:')
    _copy(
        src=weights_path,
        dst=out_dir / 'model_weights' / weights_path.name,
        overwrite=args.overwrite,
    )

    # ── Plots ────────────────────────────────────────────────────────────────
    print('\nPlots:')
    # loss plot has no exp-split prefix
    loss_stem = re.sub(r'^(expval|expeval)_', '', stem)
    for filename in [
        f'{loss_stem}_loss.png',
        f'expval_{loss_stem}_scatter.png',
        f'expeval_{loss_stem}_scatter.png',
    ]:
        copy_plot(
            src=plots_src_dir / filename,
            dst=out_dir / 'plots' / filename,
            overwrite=args.overwrite,
        )

    # ── Config ───────────────────────────────────────────────────────────────
    print('\nConfig:')
    _copy(
        src=config,
        dst=out_dir / 'config' / config.name,
        overwrite=args.overwrite,
    )

    print()
    print(f'Done. Artifacts written to: {out_dir.resolve()}')
    print()


if __name__ == '__main__':
    main()
