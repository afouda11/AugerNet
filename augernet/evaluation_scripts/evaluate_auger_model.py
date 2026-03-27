"""
Evaluate Auger GNN Model
=========================

Compares GNN-predicted Auger spectra with calculated and experimental
reference spectra via Pearson correlation coefficients (PCC).

Supports two spectrum modes:

- **stick**  — separate singlet + triplet models predict stick peaks,
  which are broadened with Gaussians and summed.
- **fitted** — a single model predicts intensity on a common energy grid
  (n_points-dim), summed over carbon atoms.

Can be used in two ways:

1. **Imported by backend.py** — called automatically after training
   when ``run_evaluation: true``::

       from .evaluation_scripts.evaluate_auger_model import run_evaluation

2. **Standalone CLI**::

       python -m augernet.evaluation_scripts.evaluate_auger_model \\
           --sing-model /path/to/singlet.pth \\
           --trip-model /path/to/triplet.pth \\
           --mode stick --feature-keys 035
"""

from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate, stats
from torch_geometric.loader import DataLoader
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Allow imports when run as a script
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from augernet import gnn_train_utils as gtu
from augernet.feature_assembly import (
    assemble_dataset, compute_feature_tag, describe_features,
    parse_feature_keys,
)
from augernet import DATA_DIR, DATA_RAW_DIR


# ─────────────────────────────────────────────────────────────────────────────
#  Spectrum utilities
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian1D(yo, xo, x, sig):
    """Single Gaussian peak centred at *xo* with height *yo*."""
    return yo * np.exp(-((x - xo) ** 2) / (2 * sig ** 2))


def _broaden_sticks(energies, intensities, fwhm, e_min, e_max, n_points):
    """Broaden stick peaks into a continuous spectrum.

    Returns ``np.ndarray`` of shape ``(n_points, 2)``  — ``[energy, intensity]``.
    """
    sig = fwhm / 2.355
    xbase = np.linspace(e_min, e_max, n_points)
    yfit = np.zeros(n_points)
    for e, y in zip(energies, intensities):
        yfit += _gaussian1D(y, e, xbase, sig)
    return np.column_stack((xbase, yfit))


def _unflatten_spectrum(flattened, max_ke, max_spec_len):
    """Convert a flat stick-spectrum tensor to ``(energy, intensity)`` arrays."""
    eng = flattened[:max_spec_len]
    intensity = flattened[max_spec_len:]
    energy = eng * max_ke
    return energy, intensity


# ─────────────────────────────────────────────────────────────────────────────
#  Prediction generators
# ─────────────────────────────────────────────────────────────────────────────

def _predict_stick(model, eval_data, device, max_ke, max_spec_len):
    """Run inference on eval_data and return per-atom stick spectra.

    Returns
    -------
    dict[int, dict[int, np.ndarray]]
        ``{mol_idx: {atom_idx: array(N, 2)}}``  — energy / intensity pairs.
    """
    loader = DataLoader(eval_data, batch_size=1, shuffle=False)
    predictions = {}

    model.eval()
    with torch.no_grad():
        for mol_idx, data in enumerate(loader):
            data = data.to(device)
            out = model(data)

            node_mask = data.node_mask.squeeze()
            valid_nodes = node_mask.nonzero(as_tuple=True)[0]

            predictions[mol_idx] = {}
            for atom_idx, node_idx in enumerate(valid_nodes):
                spec = out[node_idx].cpu().numpy()
                energy, intensity = _unflatten_spectrum(spec, max_ke, max_spec_len)

                mask_bin = data.mask_bin[node_idx].cpu().numpy()
                mask_indices = mask_bin[:max_spec_len] > 0.5

                if mask_indices.sum() > 0:
                    predictions[mol_idx][atom_idx] = np.column_stack(
                        (energy[mask_indices], intensity[mask_indices])
                    )

    return predictions


def _predict_fitted(model, eval_data, device):
    """Run inference on eval_data and return per-atom fitted intensity vectors.

    Returns
    -------
    dict[int, dict[int, np.ndarray]]
        ``{mol_idx: {atom_idx: array(n_points,)}}``  — intensity vectors.
    """
    loader = DataLoader(eval_data, batch_size=1, shuffle=False)
    predictions = {}

    model.eval()
    with torch.no_grad():
        for mol_idx, data in enumerate(loader):
            data = data.to(device)
            out = model(data)

            node_mask = data.node_mask.squeeze()
            valid_nodes = node_mask.nonzero(as_tuple=True)[0]

            predictions[mol_idx] = {}
            for atom_idx, node_idx in enumerate(valid_nodes):
                predictions[mol_idx][atom_idx] = out[node_idx].cpu().numpy()

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
#  Shared evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_eval_metadata(eval_dir):
    """Return ``(mol_list, c_num)`` from the eval directory."""
    mol_list_path = os.path.join(eval_dir, 'mol_list.txt')
    with open(mol_list_path) as f:
        mol_list = [line.strip() for line in f if line.strip()]

    c_num_path = os.path.join(eval_dir, 'carbon_numbers.txt')
    c_num = np.loadtxt(c_num_path)

    return mol_list, c_num


def _load_experimental_spectrum(eval_dir, mol_id):
    """Load ``{mol_id}_exp.txt`` or return ``None``."""
    path = os.path.join(eval_dir, f'{mol_id}_exp.txt')
    return np.loadtxt(path) if os.path.exists(path) else None


def _load_calc_spectrum(eval_dir, mol_id, carbon_idx, state,
                        method='mcpdft_hybrid_rcc'):
    """Load a single calculated stick-spectrum file or return ``None``."""
    path = os.path.join(
        eval_dir,
        f'{mol_id}_{method}_{state}_c{carbon_idx}.auger.spectrum.out',
    )
    return np.loadtxt(path) if os.path.exists(path) else None


def _load_all_calc_sticks(eval_dir, mol_id, carbon_count):
    """Collect all singlet + triplet calculated sticks for a molecule."""
    sticks = []
    for c in range(1, carbon_count + 1):
        for state in ('singlet', 'triplet'):
            spec = _load_calc_spectrum(eval_dir, mol_id, c, state)
            if spec is not None:
                sticks.append(spec)
    return sticks


def _compute_pcc(a, b):
    """Pearson r between two vectors, or ``None`` if invalid."""
    if np.isfinite(a).all() and np.isfinite(b).all() and a.std() > 0 and b.std() > 0:
        return stats.pearsonr(a, b)[0]
    return None


def _print_pcc_summary(title, pcc_data):
    """Print a Pearson-correlation summary table."""
    print(f"\n{'=' * 80}")
    print(f"EVALUATION SUMMARY — {title}")
    print(f"{'=' * 80}")
    print(f"{'Molecule':<20} {'Calc vs Exp':>15} {'GNN vs Calc':>15} {'GNN vs Exp':>15}")
    print('-' * 80)

    gnn_pccs, calc_pccs, gvc_pccs = [], [], []

    for entry in pcc_data:
        mol = entry['molecule'][:19]
        gnn = entry['gnn_pcc']
        calc = entry['calc_pcc']
        gvc = entry['gnn_vs_calc_pcc']

        if gnn is not None:
            gnn_pccs.append(gnn)
        if calc is not None:
            calc_pccs.append(calc)
        if gvc is not None:
            gvc_pccs.append(gvc)

        _f = lambda v: f'{v:.4f}' if v is not None else 'N/A'
        print(f"{mol:<20} {_f(calc):>15} {_f(gvc):>15} {_f(gnn):>15}")

    print('-' * 80)
    _avg = lambda lst: f'{np.mean(lst):.4f}' if lst else 'N/A'
    print(f"{'AVERAGE':<20} {_avg(calc_pccs):>15} {_avg(gvc_pccs):>15} {_avg(gnn_pccs):>15}")
    print('=' * 80)

    metrics = {
        'mean_gnn_pcc':      float(np.mean(gnn_pccs))  if gnn_pccs  else None,
        'mean_calc_pcc':     float(np.mean(calc_pccs))  if calc_pccs else None,
        'mean_gvc_pcc':      float(np.mean(gvc_pccs))   if gvc_pccs  else None,
        'per_molecule':      pcc_data,
    }
    return metrics


def _add_pcc_annotation(ax, pcc_calc, pcc_gvc, pcc_gnn):
    """Add a small text box with PCC values to a subplot."""
    if pcc_calc is None and pcc_gvc is None and pcc_gnn is None:
        return
    lines = ['Pearson Corr.:']
    if pcc_calc is not None:
        lines.append(f'Calc vs Exp: {pcc_calc:.3f}')
    if pcc_gvc is not None:
        lines.append(f'GNN vs Calc: {pcc_gvc:.3f}')
    if pcc_gnn is not None:
        lines.append(f'GNN vs Exp: {pcc_gnn:.3f}')
    ax.text(
        0.05, 0.50, '\n'.join(lines),
        transform=ax.transAxes, fontsize=13,
        va='center', ha='left',
        bbox=dict(boxstyle='round', facecolor='none', edgecolor='gray'),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Stick evaluation  (singlet + triplet)
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_stick(
    sing_model, trip_model, device,
    eval_data_sing, eval_data_trip,
    *,
    output_dir, png_dir, file_stem,
    max_ke, max_spec_len, n_points, fwhm, ke_shift,
    train_results=None,
):
    """Evaluate stick-mode singlet + triplet models.

    Generates predictions, combines singlet + triplet, broadens with
    Gaussians, and compares to experimental + calculated reference
    spectra via Pearson correlation.

    Returns a dict of PCC metrics.
    """
    print('\nGenerating singlet predictions...')
    sing_preds = _predict_stick(sing_model, eval_data_sing, device,
                                max_ke, max_spec_len)
    print(f'  {len(sing_preds)} molecules')

    print('Generating triplet predictions...')
    trip_preds = _predict_stick(trip_model, eval_data_trip, device,
                                max_ke, max_spec_len)
    print(f'  {len(trip_preds)} molecules')

    n_molecules = len(eval_data_sing)
    eval_dir = os.path.join(DATA_RAW_DIR, 'eval_auger')
    mol_list, c_num = _load_eval_metadata(eval_dir)

    # Combine singlet + triplet GNN sticks per molecule
    gnn_combined = {}
    for mol_idx in range(n_molecules):
        specs = []
        for preds in (sing_preds, trip_preds):
            if mol_idx in preds:
                for a_idx in sorted(preds[mol_idx]):
                    specs.append(preds[mol_idx][a_idx])
        if specs:
            combined = np.row_stack(specs)
            if combined.shape[0] > 0:
                gnn_combined[mol_idx] = combined

    # ── Loss curves ──────────────────────────────────────────────────────
    if train_results is not None and len(train_results) > 0:
        _plot_loss_curves(train_results, png_dir, file_stem)

    # ── Overview plot ────────────────────────────────────────────────────
    n_plot = min(16, n_molecules)
    fig, axes = plt.subplots(8, 2, figsize=(24, 22), sharex=True, sharey=True)
    ax = axes.ravel()
    pcc_data = []

    for i in range(n_plot):
        mol_id = mol_list[i]
        exp_spec = _load_experimental_spectrum(eval_dir, mol_id)

        if exp_spec is None:
            print(f'  ⚠ No experimental spectrum: {mol_id}')
            pcc_data.append(dict(molecule=mol_id, gnn_pcc=None,
                                 calc_pcc=None, gnn_vs_calc_pcc=None))
            continue

        exp_min, exp_max = exp_spec[:, 0].min(), exp_spec[:, 0].max()
        exp_base = np.linspace(exp_min, exp_max, n_points)
        fit_exp = interpolate.interp1d(exp_spec[:, 0], exp_spec[:, 1])(exp_base)
        fit_exp_norm = fit_exp / fit_exp.max()

        # Plot experimental
        ax[i].plot(exp_spec[:, 0], exp_spec[:, 1] / exp_spec[:, 1].max(),
                   lw=2.5, color='k', ls='-', label='Experimental', alpha=0.8)

        # Calculated reference
        calc_sticks = _load_all_calc_sticks(eval_dir, mol_id, int(c_num[i]))
        fit_calc_norm = None
        if calc_sticks:
            calc_all = np.row_stack(calc_sticks)
            calc_all[:, 0] += ke_shift
            fit_calc = _broaden_sticks(
                calc_all[:, 0], calc_all[:, 1],
                fwhm, exp_min - 1, exp_max + 1, n_points,
            )
            fit_calc_norm = fit_calc[:, 1] / fit_calc[:, 1].max()
            ax[i].plot(fit_calc[:, 0], fit_calc_norm,
                       lw=2.5, color='g', ls='--', label='Calculated', alpha=0.8)
            ax[i].vlines(calc_all[:, 0], 0,
                         calc_all[:, 1] / fit_calc[:, 1].max(),
                         lw=1.5, color='g', ls='--', alpha=0.6)

        # GNN prediction
        gnn_fit_norm = None
        if i in gnn_combined:
            gnn_spec = gnn_combined[i].copy()
            gnn_spec[:, 0] += ke_shift
            gnn_fit = _broaden_sticks(
                gnn_spec[:, 0], gnn_spec[:, 1],
                fwhm, exp_min - 1, exp_max + 1, n_points,
            )
            gnn_max = gnn_fit[:, 1].max()
            if gnn_max > 0:
                gnn_fit_norm = gnn_fit[:, 1] / gnn_max
                ax[i].plot(gnn_fit[:, 0], gnn_fit_norm,
                           lw=2.5, color='b', ls=':', label='GNN Predicted', alpha=0.8)
                ax[i].vlines(gnn_spec[:, 0], 0,
                             gnn_spec[:, 1] / gnn_max,
                             lw=1.5, color='b', alpha=0.6)
            else:
                print(f'  ⚠ GNN predicted zero intensities for {mol_id}')

        # PCC
        pcc_gnn = pcc_calc = pcc_gvc = None
        try:
            if gnn_fit_norm is not None:
                gnn_interp = np.interp(exp_base, gnn_fit[:, 0], gnn_fit_norm)
                pcc_gnn = _compute_pcc(gnn_interp, fit_exp_norm)
            if fit_calc_norm is not None:
                calc_interp = np.interp(exp_base, fit_calc[:, 0], fit_calc_norm)
                pcc_calc = _compute_pcc(calc_interp, fit_exp_norm)
            if gnn_fit_norm is not None and fit_calc_norm is not None:
                gnn_on_calc = np.interp(fit_calc[:, 0], gnn_fit[:, 0], gnn_fit_norm)
                pcc_gvc = _compute_pcc(gnn_on_calc, fit_calc_norm)
        except Exception as e:
            print(f'  ⚠ PCC error for {mol_id}: {e}')

        # Annotations
        ax[i].text(0.05, 0.95, mol_id, transform=ax[i].transAxes,
                   fontsize=18, va='top',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        _add_pcc_annotation(ax[i], pcc_calc, pcc_gvc, pcc_gnn)

        pcc_data.append(dict(molecule=mol_id, gnn_pcc=pcc_gnn,
                             calc_pcc=pcc_calc, gnn_vs_calc_pcc=pcc_gvc))

        ax[i].set_xlim(220, 275)
        ax[i].set_ylim(0, 1.1)
        ax[i].tick_params(axis='y', labelleft=False)

    ax[0].legend(loc='upper right', fontsize=12)
    ax[6].set_ylabel('Normalized Intensity (arb. units)', fontsize=24)
    ax[12].set_xlabel('Kinetic Energy (eV)', fontsize=24)
    ax[13].set_xlabel('Kinetic Energy (eV)', fontsize=24)
    plt.subplots_adjust(hspace=0)
    plt.tight_layout()

    plot_path = os.path.join(png_dir, f'{file_stem}_stick_overview.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f'  Overview plot saved to {plot_path}')
    plt.close(fig)

    return _print_pcc_summary('STICK (singlet + triplet)', pcc_data)


# ─────────────────────────────────────────────────────────────────────────────
#  Fitted evaluation  (single combined model)
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_fitted(
    model, device, eval_data,
    *,
    output_dir, png_dir, file_stem,
    n_points, min_ke, max_ke, fwhm, ke_shift,
    train_results=None,
):
    """Evaluate a fitted-spectrum model.

    The model predicts per-atom intensity on a common energy grid.
    We sum per carbon, then compare to experimental spectra via PCC.

    Returns a dict of PCC metrics.
    """
    print('\nGenerating fitted predictions...')
    predictions = _predict_fitted(model, eval_data, device)
    print(f'  {len(predictions)} molecules')

    n_molecules = len(eval_data)
    eval_dir = os.path.join(DATA_RAW_DIR, 'eval_auger')
    mol_list, c_num = _load_eval_metadata(eval_dir)
    energy_grid = np.linspace(min_ke, max_ke, n_points)

    # ── Loss curves ──────────────────────────────────────────────────────
    if train_results is not None and len(train_results) > 0:
        _plot_loss_curves(train_results, png_dir, file_stem)

    # ── Overview plot ────────────────────────────────────────────────────
    n_plot = min(16, n_molecules)
    fig, axes = plt.subplots(8, 2, figsize=(24, 22), sharex=True, sharey=True)
    ax = axes.ravel()
    pcc_data = []

    for i in range(n_plot):
        mol_id = mol_list[i]
        exp_spec = _load_experimental_spectrum(eval_dir, mol_id)

        if exp_spec is None:
            print(f'  ⚠ No experimental spectrum: {mol_id}')
            pcc_data.append(dict(molecule=mol_id, gnn_pcc=None,
                                 calc_pcc=None, gnn_vs_calc_pcc=None))
            continue

        exp_min, exp_max = exp_spec[:, 0].min(), exp_spec[:, 0].max()
        exp_base = np.linspace(exp_min, exp_max, n_points)
        fit_exp = interpolate.interp1d(exp_spec[:, 0], exp_spec[:, 1])(exp_base)
        fit_exp_norm = fit_exp / fit_exp.max()

        ax[i].plot(exp_spec[:, 0], exp_spec[:, 1] / exp_spec[:, 1].max(),
                   lw=2.5, color='k', ls='-', label='Experimental', alpha=0.8)

        # Calculated reference (broaden stick spectra)
        calc_sticks = _load_all_calc_sticks(eval_dir, mol_id, int(c_num[i]))
        fit_calc_norm = None
        calc_base = None
        if calc_sticks:
            calc_all = np.row_stack(calc_sticks)
            calc_all[:, 0] += ke_shift
            fit_calc = _broaden_sticks(
                calc_all[:, 0], calc_all[:, 1],
                fwhm, exp_min - 1, exp_max + 1, n_points,
            )
            fit_calc_norm = fit_calc[:, 1] / fit_calc[:, 1].max()
            calc_base = fit_calc[:, 0]
            ax[i].plot(calc_base, fit_calc_norm,
                       lw=2.5, color='g', ls='--', label='Calculated', alpha=0.8)

        # GNN fitted prediction — sum per-carbon intensities
        gnn_total = np.zeros(n_points)
        if i in predictions:
            for a_idx in predictions[i]:
                gnn_total += predictions[i][a_idx]

        gnn_fit_norm = None
        if gnn_total.max() > 0:
            gnn_fit_norm = gnn_total / gnn_total.max()
            ax[i].plot(energy_grid, gnn_fit_norm,
                       lw=2.5, color='b', ls=':', label='GNN Fitted', alpha=0.8)

        # PCC
        pcc_gnn = pcc_calc = pcc_gvc = None
        try:
            if gnn_fit_norm is not None:
                gnn_interp = np.interp(exp_base, energy_grid, gnn_fit_norm)
                pcc_gnn = _compute_pcc(gnn_interp, fit_exp_norm)
            if fit_calc_norm is not None and calc_base is not None:
                calc_interp = np.interp(exp_base, calc_base, fit_calc_norm)
                pcc_calc = _compute_pcc(calc_interp, fit_exp_norm)
            if gnn_fit_norm is not None and fit_calc_norm is not None and calc_base is not None:
                gnn_on_calc = np.interp(calc_base, energy_grid, gnn_fit_norm)
                pcc_gvc = _compute_pcc(gnn_on_calc, fit_calc_norm)
        except Exception as e:
            print(f'  ⚠ PCC error for {mol_id}: {e}')

        ax[i].text(0.05, 0.95, mol_id, transform=ax[i].transAxes,
                   fontsize=18, va='top',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        _add_pcc_annotation(ax[i], pcc_calc, pcc_gvc, pcc_gnn)

        pcc_data.append(dict(molecule=mol_id, gnn_pcc=pcc_gnn,
                             calc_pcc=pcc_calc, gnn_vs_calc_pcc=pcc_gvc))

        ax[i].set_xlim(220, 275)
        ax[i].set_ylim(0, 1.1)
        ax[i].tick_params(axis='y', labelleft=False)

    ax[0].legend(loc='upper right', fontsize=12)
    ax[6].set_ylabel('Normalized Intensity (arb. units)', fontsize=24)
    ax[12].set_xlabel('Kinetic Energy (eV)', fontsize=24)
    ax[13].set_xlabel('Kinetic Energy (eV)', fontsize=24)
    plt.subplots_adjust(hspace=0)
    plt.tight_layout()

    plot_path = os.path.join(png_dir, f'{file_stem}_fitted_overview.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f'  Overview plot saved to {plot_path}')
    plt.close(fig)

    return _print_pcc_summary('FITTED SPECTRUM', pcc_data)


# ─────────────────────────────────────────────────────────────────────────────
#  Loss-curve plotting (shared)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_loss_curves(train_results, png_dir, file_stem):
    """Plot train/val loss curves (same style as CEBE evaluation)."""
    epochs     = np.array([r[0] for r in train_results])
    train_loss = np.array([r[1] for r in train_results])
    val_loss   = np.array([r[2] for r in train_results])

    best_epoch = int(np.argmin(val_loss))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.semilogy(epochs, train_loss, color='#0072B2', lw=1.6,
                label='Train', zorder=3)
    ax.semilogy(epochs, val_loss,   color='#E69F00', lw=1.6,
                label='Validation', alpha=0.92, zorder=3)
    ax.axvline(best_epoch, color='#d62728', ls='--', lw=1.3, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.85, loc='lower left')
    ax.tick_params(axis='both', labelsize=9)
    ax.set_xlim(0, epochs[-1] + 2)
    ax.grid(True, alpha=0.3, lw=1.0, zorder=0)

    for ext in ('png', 'pdf'):
        path = os.path.join(png_dir, f'{file_stem}_loss.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f'  Loss curves saved to {png_dir}/{file_stem}_loss.{{png,pdf}}')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point  (called by backend.run_evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    model_result,
    device,
    *,
    output_dir: str,
    png_dir: str,
    cfg,
    fold: Optional[int] = None,
    train_results=None,
    model_id: str = 'auger',
    config_id: str = None,
    param_file_prefix: str = None,
):
    """Evaluate an Auger GNN model after training.

    Parameters
    ----------
    model_result : dict
        Training result dict from ``backend.train_single_run``.
        For stick mode, must contain ``'sing_model'`` and ``'trip_model'``.
        For fitted mode, must contain ``'model'``.
    device : torch.device
    output_dir, png_dir : str
        Output directories.
    cfg : AugerNetConfig
        Full config — used for spectrum_type, feature_keys, and spectrum
        parameters (max_spec_len, max_ke, n_points, fwhm, etc.).
    fold : int, optional
    train_results : list, optional
        ``[epoch, train_loss, val_loss]`` for loss-curve plotting.
    model_id : str
    config_id : str, optional
    param_file_prefix : str, optional
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    # Build file stem
    file_stem = f'{model_id}_fold{fold}' if fold is not None else model_id
    if config_id is not None:
        file_stem = f'{file_stem}_{config_id}'
    if param_file_prefix is not None:
        file_stem = f'{param_file_prefix}_{file_stem}'

    print(f"\n{'=' * 80}")
    print(f"AUGER EVALUATION — {cfg.spectrum_type.upper()}"
          f"{f'  (fold {fold})' if fold else ''}")
    print(f"{'=' * 80}")

    # Load evaluation data
    feature_keys = cfg.feature_keys_parsed

    if cfg.spectrum_type == 'stick':
        sing_ds = gtu.LoadDataset(DATA_DIR, file_name='gnn_eval_auger_sing_data.pt')
        trip_ds = gtu.LoadDataset(DATA_DIR, file_name='gnn_eval_auger_trip_data.pt')
        eval_sing = [sing_ds[i] for i in range(len(sing_ds))]
        eval_trip = [trip_ds[i] for i in range(len(trip_ds))]
        # Reshape y from (n_atoms*600, 1) → (n_atoms, 600) for correct batching
        for dlist in (eval_sing, eval_trip):
            for d in dlist:
                n_atoms = d.x.size(0)
                d.y = d.y.view(n_atoms, 600)
        assemble_dataset(eval_sing, feature_keys)
        assemble_dataset(eval_trip, feature_keys)
        print(f'  Loaded {len(eval_sing)} singlet + {len(eval_trip)} triplet eval molecules')

        # Extract models
        sing_model = model_result.get('sing_model', model_result.get('model'))
        trip_model = model_result.get('trip_model')
        if trip_model is None:
            print('  ⚠ No triplet model — evaluating singlet only')

        return _evaluate_stick(
            sing_model, trip_model, device,
            eval_sing, eval_trip,
            output_dir=output_dir, png_dir=png_dir, file_stem=file_stem,
            max_ke=cfg.max_ke, max_spec_len=cfg.max_spec_len,
            n_points=cfg.n_points, fwhm=cfg.fwhm, ke_shift=cfg.ke_shift_calc,
            train_results=train_results,
        )

    else:  # fitted
        sing_ds = gtu.LoadDataset(DATA_DIR, file_name='gnn_eval_auger_sing_data.pt')
        eval_data = [sing_ds[i] for i in range(len(sing_ds))]
        assemble_dataset(eval_data, feature_keys)
        print(f'  Loaded {len(eval_data)} eval molecules (fitted)')

        model = model_result.get('model') if isinstance(model_result, dict) else model_result

        return _evaluate_fitted(
            model, device, eval_data,
            output_dir=output_dir, png_dir=png_dir, file_stem=file_stem,
            n_points=cfg.n_points, min_ke=cfg.min_ke, max_ke=cfg.max_ke,
            fwhm=cfg.fwhm, ke_shift=cfg.ke_shift_calc,
            train_results=train_results,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a saved Auger GNN model on experimental data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stick mode (singlet + triplet)
  python -m augernet.evaluation_scripts.evaluate_auger_model \\
      --sing-model models/singlet_fold3.pth \\
      --trip-model models/triplet_fold3.pth \\
      --mode stick --feature-keys 035

  # Fitted mode
  python -m augernet.evaluation_scripts.evaluate_auger_model \\
      --model models/fitted_fold3.pth \\
      --mode fitted --feature-keys 035
        """,
    )
    parser.add_argument('--mode', '-m', default='stick',
                        choices=['stick', 'fitted'])
    parser.add_argument('--model', type=str, default=None,
                        help='Model .pth (fitted mode)')
    parser.add_argument('--sing-model', type=str, default=None,
                        help='Singlet model .pth (stick mode)')
    parser.add_argument('--trip-model', type=str, default=None,
                        help='Triplet model .pth (stick mode)')
    parser.add_argument('--feature-keys', '-fk', type=str, default='035',
                        help="Feature key string (e.g. '035')")
    parser.add_argument('--layer-type', '-lt', default='EQ',
                        choices=['IN', 'EQ'])
    parser.add_argument('--hidden-channels', '-hc', type=int, default=64)
    parser.add_argument('--n-layers', '-nl', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--fwhm', type=float, default=3.768)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--png-dir', type=str, default=None)
    parser.add_argument('--fold', type=int, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()

    output_dir = args.output_dir or os.path.join(SCRIPT_DIR, 'eval_results', 'outputs')
    png_dir    = args.png_dir    or os.path.join(SCRIPT_DIR, 'eval_results', 'pngs')

    feature_keys = parse_feature_keys(args.feature_keys)
    feature_tag  = compute_feature_tag(feature_keys)

    print('=' * 80)
    print(f'  AUGER GNN EVALUATION — {args.mode.upper()} (standalone)')
    print('=' * 80)
    print(f'  Feature keys:   {feature_keys}  ({describe_features(feature_keys)})')
    print(f'  Architecture:   {args.layer_type}, {args.hidden_channels}h, {args.n_layers}L')
    print(f'  FWHM:           {args.fwhm} eV')
    print('=' * 80)

    # Spectrum parameters (defaults matching config.py)
    max_spec_len = 300
    max_ke = 273
    min_ke = 200
    n_points = 731
    ke_shift = -2.0

    # Load sample data for dimensions
    sing_ds = gtu.LoadDataset(DATA_DIR, file_name='gnn_sing_eval_auger_data.pt')
    sample = [sing_ds[0]]
    assemble_dataset(sample, feature_keys)
    in_channels = sample[0].x.size(1)
    edge_dim = sample[0].edge_attr.size(1)
    print(f'  x.shape[1]={in_channels}, edge_attr.shape[1]={edge_dim}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.mode == 'stick':
        if not args.sing_model or not args.trip_model:
            raise ValueError('Stick mode requires --sing-model and --trip-model.')

        # Build model_id
        model_id = f'auger_{feature_tag}_standalone_{args.layer_type}{args.n_layers}_h{args.hidden_channels}'

        def _load(path, spec_type='stick', spec_dim=max_spec_len):
            m = gtu.MPNN(
                num_layers=args.n_layers, emb_dim=args.hidden_channels,
                in_dim=in_channels, edge_dim=edge_dim,
                out_dim=1, layer_type=args.layer_type, pred_type='AUGER',
                dropout=args.dropout,
                spectrum_type=spec_type, spectrum_dim=spec_dim,
            ).to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            print(f'  ✓ Loaded ← {path}')
            return m

        sing_model = _load(args.sing_model)
        trip_model = _load(args.trip_model)

        # Load eval data
        trip_ds = gtu.LoadDataset(DATA_DIR, file_name='gnn_trip_eval_auger_data.pt')
        eval_sing = [sing_ds[i] for i in range(len(sing_ds))]
        eval_trip = [trip_ds[i] for i in range(len(trip_ds))]
        assemble_dataset(eval_sing, feature_keys)
        assemble_dataset(eval_trip, feature_keys)

        file_stem = f'{model_id}_fold{args.fold}' if args.fold else model_id
        _evaluate_stick(
            sing_model, trip_model, device,
            eval_sing, eval_trip,
            output_dir=output_dir, png_dir=png_dir, file_stem=file_stem,
            max_ke=max_ke, max_spec_len=max_spec_len,
            n_points=n_points, fwhm=args.fwhm, ke_shift=ke_shift,
        )

    elif args.mode == 'fitted':
        if not args.model:
            raise ValueError('Fitted mode requires --model.')

        model_id = f'auger_{feature_tag}_standalone_{args.layer_type}{args.n_layers}_h{args.hidden_channels}'

        m = gtu.MPNN(
            num_layers=args.n_layers, emb_dim=args.hidden_channels,
            in_dim=in_channels, edge_dim=edge_dim,
            out_dim=1, layer_type=args.layer_type, pred_type='AUGER',
            dropout=args.dropout,
            spectrum_type='fitted', spectrum_dim=n_points,
        ).to(device)
        m.load_state_dict(torch.load(args.model, map_location=device))
        m.eval()
        print(f'  ✓ Loaded ← {args.model}')

        eval_data = [sing_ds[i] for i in range(len(sing_ds))]
        assemble_dataset(eval_data, feature_keys)

        file_stem = f'{model_id}_fold{args.fold}' if args.fold else model_id
        _evaluate_fitted(
            m, device, eval_data,
            output_dir=output_dir, png_dir=png_dir, file_stem=file_stem,
            n_points=n_points, min_ke=min_ke, max_ke=max_ke,
            fwhm=args.fwhm, ke_shift=ke_shift,
        )

    print('\n✓ Done.')


if __name__ == '__main__':
    main()
