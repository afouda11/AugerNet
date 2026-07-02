"""
Evaluate Auger GNN Model
=========================

Compares GNN-predicted Auger spectra with calculated and experimental
reference spectra via Pearson correlation coefficients (PCC).

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
    assemble_dataset, 
)
from augernet import DATA_DIR, DATA_RAW_DIR
from augernet import spec_utils as su


# ─────────────────────────────────────────────────────────────────────────────
#  Shared evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

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

def _carbon_spec_idx(data_obj):
    """Return the OpenMolcas spectrum-file index (c1, c2, ...) for each carbon.
    """
    mask = data_obj.node_mask.squeeze().tolist()
    idx = data_obj.carbon_spec_idx.squeeze().tolist()
    if not isinstance(idx, list):        # single-carbon edge case
        idx = [idx]
    return [int(ci) for ci, m in zip(idx, mask) if m > 0.5]


def _compute_pcc(a, b):
    """Pearson r between two vectors, or ``None`` if invalid."""
    if np.isfinite(a).all() and np.isfinite(b).all() and a.std() > 0 and b.std() > 0:
        return stats.pearsonr(a, b)[0]
    return None


def _compute_mse(a, b):
    """Mean squared error between two normalized vectors, or ``None`` if invalid."""
    if np.isfinite(a).all() and np.isfinite(b).all():
        return float(np.mean((a - b) ** 2))
    return None


def _compute_mae(a, b):
    """Mean absolute error between two normalized vectors, or ``None`` if invalid."""
    if np.isfinite(a).all() and np.isfinite(b).all():
        return float(np.mean(np.abs(a - b)))
    return None


def _print_pcc_summary(title, pcc_data, show_exp=True, train_env_counts=None):
    """Print a PCC + MSE + MAE summary table.

    Parameters
    ----------
    show_exp : bool
        If False, suppress the ``Calc vs Exp`` and ``GNN vs Exp`` columns
        (used for the calc hold-out test set where no experimental data exist).
    train_env_counts : dict[str, int] or None
        Number of carbons of each environment type in the training set.
    """
    # Cell format: PCC(MSE/MAE) -- 22 chars wide when right-aligned
    def _fmt(pcc, mse, mae):
        if pcc is None:
            return 'N/A'
        p = f'{pcc:.4f}'
        m = f'{mse:.3f}' if mse is not None else 'N/A'
        a = f'{mae:.3f}' if mae is not None else 'N/A'
        return f'{p}({m}/{a})'

    C = 22   # cell column width
    if show_exp:
        sep  = '=' * (22 + 3 * (C + 2))
        dash = '-' * (22 + 3 * (C + 2))
        hdr1 = f"{'Molecule':<22}  {'Calc vs Exp':^{C}}  {'GNN vs Calc':^{C}}  {'GNN vs Exp':^{C}}"
        hdr2 = f"{'':22}  {'PCC(MSE/MAE)':^{C}}  {'PCC(MSE/MAE)':^{C}}  {'PCC(MSE/MAE)':^{C}}"
        pc_off = 22 + 2 + C + 2   # label indent to align with GNN vs Calc column
    else:
        sep  = '=' * (22 + C + 2)
        dash = '-' * (22 + C + 2)
        hdr1 = f"{'Molecule':<22}  {'GNN vs Calc':^{C}}"
        hdr2 = f"{'':22}  {'PCC(MSE/MAE)':^{C}}"
        pc_off = 22 + 2

    print(f"\n{sep}")
    print(f"EVALUATION SUMMARY -- {title}")
    print(f"{sep}")
    print(hdr1)
    print(hdr2)
    print(dash)

    gnn_pccs,  calc_pccs,  gvc_pccs  = [], [], []
    gnn_mses,  calc_mses,  gvc_mses  = [], [], []
    gnn_maes,  calc_maes,  gvc_maes  = [], [], []

    for entry in pcc_data:
        mol   = entry['molecule'][:21]
        g_pcc = entry.get('gnn_pcc');          g_mse = entry.get('gnn_mse');          g_mae = entry.get('gnn_mae')
        c_pcc = entry.get('calc_pcc');         c_mse = entry.get('calc_mse');         c_mae = entry.get('calc_mae')
        v_pcc = entry.get('gnn_vs_calc_pcc');  v_mse = entry.get('gnn_vs_calc_mse');  v_mae = entry.get('gnn_vs_calc_mae')

        for lst, v in [(gnn_pccs, g_pcc), (calc_pccs, c_pcc), (gvc_pccs, v_pcc)]:
            if v is not None: lst.append(v)
        for lst, v in [(gnn_mses, g_mse), (calc_mses, c_mse), (gvc_mses, v_mse)]:
            if v is not None: lst.append(v)
        for lst, v in [(gnn_maes, g_mae), (calc_maes, c_mae), (gvc_maes, v_mae)]:
            if v is not None: lst.append(v)

        if show_exp:
            print(f"{mol:<22}  {_fmt(c_pcc,c_mse,c_mae):>{C}}  {_fmt(v_pcc,v_mse,v_mae):>{C}}  {_fmt(g_pcc,g_mse,g_mae):>{C}}")
        else:
            print(f"{mol:<22}  {_fmt(v_pcc,v_mse,v_mae):>{C}}")

        # Per-carbon rows (GNN vs Calc only, aligned under GNN vs Calc column)
        for item in entry.get('per_carbon_pccs', []):
            c, env_label = item[0], item[1]
            pcc = item[2]
            mse = item[3] if len(item) > 3 else None
            mae = item[4] if len(item) > 4 else None
            lbl = f'  C{c}'
            if env_label:
                lbl += f' ({env_label.removeprefix("C_")})'
            print(f"{lbl:<{pc_off}}{_fmt(pcc, mse, mae):>{C}}")

    print(dash)
    _avg = lambda lst: float(np.mean(lst))   if lst else None
    _med = lambda lst: float(np.median(lst)) if lst else None
    a_c_p, a_c_s, a_c_a = _avg(calc_pccs), _avg(calc_mses), _avg(calc_maes)
    a_v_p, a_v_s, a_v_a = _avg(gvc_pccs),  _avg(gvc_mses),  _avg(gvc_maes)
    a_g_p, a_g_s, a_g_a = _avg(gnn_pccs),  _avg(gnn_mses),  _avg(gnn_maes)
    m_c_p, m_c_s, m_c_a = _med(calc_pccs), _med(calc_mses), _med(calc_maes)
    m_v_p, m_v_s, m_v_a = _med(gvc_pccs),  _med(gvc_mses),  _med(gvc_maes)
    m_g_p, m_g_s, m_g_a = _med(gnn_pccs),  _med(gnn_mses),  _med(gnn_maes)
    if show_exp:
        print(f"{'AVERAGE':<22}  {_fmt(a_c_p,a_c_s,a_c_a):>{C}}  {_fmt(a_v_p,a_v_s,a_v_a):>{C}}  {_fmt(a_g_p,a_g_s,a_g_a):>{C}}")
        print(f"{'MEDIAN':<22}  {_fmt(m_c_p,m_c_s,m_c_a):>{C}}  {_fmt(m_v_p,m_v_s,m_v_a):>{C}}  {_fmt(m_g_p,m_g_s,m_g_a):>{C}}")
    else:
        print(f"{'AVERAGE':<22}  {_fmt(a_v_p,a_v_s,a_v_a):>{C}}")
        print(f"{'MEDIAN':<22}  {_fmt(m_v_p,m_v_s,m_v_a):>{C}}")
    print(sep)

    # -- Per-environment-type summary (GNN vs Calc) --
    env_pcc_lists: dict = {}
    env_mse_lists: dict = {}
    env_mae_lists: dict = {}
    for entry in pcc_data:
        for item in entry.get('per_carbon_pccs', []):
            c, env_label, pcc = item[0], item[1], item[2]
            mse = item[3] if len(item) > 3 else None
            mae = item[4] if len(item) > 4 else None
            if pcc is None:
                continue
            key = env_label.removeprefix('C_') if env_label else 'unknown'
            env_pcc_lists.setdefault(key, []).append(pcc)
            if mse is not None: env_mse_lists.setdefault(key, []).append(mse)
            if mae is not None: env_mae_lists.setdefault(key, []).append(mae)

    if env_pcc_lists:
        has_train = train_env_counts is not None and len(train_env_counts) > 0
        if has_train:
            env_sep = '-' * 95
            print(f"\n{'Environment type':<28} {'N test':>7} {'N train':>8} {'Mean PCC(MSE/MAE)':>{C}} {'Std PCC':>9}")
        else:
            env_sep = '-' * 78
            print(f"\n{'Environment type':<28} {'N test':>7} {'Mean PCC(MSE/MAE)':>{C}} {'Std PCC':>9}")
        print(env_sep)

        env_means = {}
        for env_name in sorted(env_pcc_lists):
            vals     = env_pcc_lists[env_name]
            mse_vals = env_mse_lists.get(env_name, [])
            mae_vals = env_mae_lists.get(env_name, [])
            mean_pcc = np.mean(vals)
            std_pcc  = np.std(vals) if len(vals) > 1 else 0.0
            mean_mse = float(np.mean(mse_vals)) if mse_vals else None
            mean_mae = float(np.mean(mae_vals)) if mae_vals else None
            env_means[env_name] = mean_pcc
            cell = _fmt(mean_pcc, mean_mse, mean_mae)
            n_train = train_env_counts.get(env_name, 0) if has_train else None
            if has_train:
                print(f"{env_name:<28} {len(vals):>7} {n_train:>8} {cell:>{C}} {std_pcc:>9.4f}")
            else:
                print(f"{env_name:<28} {len(vals):>7} {cell:>{C}} {std_pcc:>9.4f}")

        print(env_sep)
        means = list(env_means.values())
        macro_avg = np.mean(means)
        macro_std = np.std(means)
        print(f"  Macro-avg PCC (unweighted):   {macro_avg:.4f}  (std across types: {macro_std:.4f})")

        inv_freq_pcc = None
        if has_train:
            weights = {}
            for env_name in env_means:
                n_tr = train_env_counts.get(env_name, 0)
                weights[env_name] = 1.0 / n_tr if n_tr > 0 else 0.0
            total_w = sum(weights.values())
            if total_w > 0:
                inv_freq_pcc = sum(weights[e] * env_means[e] for e in env_means) / total_w
                print(f"  Inv-freq-weighted PCC (by train N): {inv_freq_pcc:.4f}")

        print('=' * (95 if has_train else 78))

        per_env_out = {
            env_name: {
                'n_test':   len(env_pcc_lists[env_name]),
                'n_train':  train_env_counts.get(env_name, None) if has_train else None,
                'mean_pcc': float(env_means[env_name]),
                'std_pcc':  float(np.std(env_pcc_lists[env_name])
                            if len(env_pcc_lists[env_name]) > 1 else 0.0),
                'mean_mse': float(np.mean(env_mse_lists[env_name])) if env_mse_lists.get(env_name) else None,
                'mean_mae': float(np.mean(env_mae_lists[env_name])) if env_mae_lists.get(env_name) else None,
            }
            for env_name in sorted(env_means)
        }
    else:
        macro_avg = macro_std = inv_freq_pcc = None
        per_env_out = {}

    return {
        'mean_gnn_pcc':  _avg(gnn_pccs),  'mean_gnn_mse':  _avg(gnn_mses),  'mean_gnn_mae':  _avg(gnn_maes),
        'mean_calc_pcc': _avg(calc_pccs), 'mean_calc_mse': _avg(calc_mses), 'mean_calc_mae': _avg(calc_maes),
        'mean_gvc_pcc':  _avg(gvc_pccs),  'mean_gvc_mse':  _avg(gvc_mses),  'mean_gvc_mae':  _avg(gvc_maes),
        'macro_avg_pcc':           float(macro_avg) if macro_avg is not None else None,
        'macro_std_pcc':           float(macro_std) if macro_std is not None else None,
        'inv_freq_weighted_pcc':   float(inv_freq_pcc) if inv_freq_pcc is not None else None,
        'per_env':                 per_env_out,
        'per_molecule': [
            {
                'molecule':        e['molecule'],
                'calc_pcc':        e.get('calc_pcc'),  'calc_mse':        e.get('calc_mse'),  'calc_mae':        e.get('calc_mae'),
                'gnn_pcc':         e.get('gnn_pcc'),   'gnn_mse':         e.get('gnn_mse'),   'gnn_mae':         e.get('gnn_mae'),
                'gnn_vs_calc_pcc': e.get('gnn_vs_calc_pcc'),
                'gnn_vs_calc_mse': e.get('gnn_vs_calc_mse'),
                'gnn_vs_calc_mae': e.get('gnn_vs_calc_mae'),
                'per_carbon': [
                    {'carbon': item[0],
                     'env':    item[1].removeprefix('C_') if item[1] else '',
                     'pcc':    item[2],
                     'mse':    item[3] if len(item) > 3 else None,
                     'mae':    item[4] if len(item) > 4 else None}
                    for item in e.get('per_carbon_pccs', [])
                ],
            }
            for e in pcc_data
        ],
    }


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

def _compute_molecule_results(
    mol_idx, mol_id, predictions, n_carbons,
    eval_dir, energy_grid, n_points, fwhm, ke_shift, min_ke, max_ke,
    carbon_env_labels=None,
    calc_method='mcpdft_hybrid_rcc',
    carbon_spec_idx=None,
):
    """Compute broadened spectra and PCCs for one molecule.

    All spectra are placed on ``display_grid = energy_grid + ke_shift`` so
    GNN predictions and broadened calc sticks share the same axis.

    Parameters
    ----------
    carbon_env_labels : list[str] or None
        Ordered environment class label for each carbon (0-based), e.g.
        ``['C_methyl', 'C_ketone', ...]``.  If provided, each per-carbon
        PCC entry carries the label alongside the index.
    carbon_spec_idx : list[int] or None
        XYZ to openmolcas carbon index map, same logic as data prep

    Returns
    -------
    dict with keys:
        mol_id, n_carbons, display_grid,
        gnn_total, calc_total, gnn_per_carbon, calc_per_carbon,
        exp_spec,
        gnn_pcc, calc_pcc, gnn_vs_calc_pcc,   (total vs experimental)
        per_carbon_pccs   (list of (c, env_label, pcc), GNN vs calc)
    """
    display_grid = energy_grid + ke_shift     # common axis for GNN + calc

    # -- GNN: accumulate per-atom predictions --
    gnn_per_carbon = {}   # {0-based atom_idx: array(n_points)}
    gnn_total = np.zeros(n_points)
    if mol_idx in predictions:
        for a_idx in sorted(predictions[mol_idx]):
            spec = predictions[mol_idx][a_idx]
            gnn_per_carbon[a_idx] = spec
            gnn_total += spec

    # Calc: broaden per-carbon sticks onto display_grid 
    # Uses openmoclas to xyz carbon index mapping from data prep
    calc_per_carbon = {}   
    calc_total = np.zeros(n_points)
    for c in range(1, n_carbons + 1):
        file_idx = carbon_spec_idx[c - 1]
        sticks = []
        for state in ('singlet', 'triplet'):
            s = _load_calc_spectrum(eval_dir, mol_id, file_idx, state, method=calc_method)
            if s is not None:
                sticks.append(s)
        if sticks:
            calc_c = np.vstack(sticks)
            _, calc_c_i = su.fit_spectrum_to_grid(
                calc_c[:, 0] + ke_shift, calc_c[:, 1],
                fwhm, display_grid[0], display_grid[-1], n_points,
            )
            calc_per_carbon[c] = calc_c_i
            calc_total += calc_c_i

    exp_spec = _load_experimental_spectrum(eval_dir, mol_id)

    result = dict(
        mol_id=mol_id, n_carbons=n_carbons, display_grid=display_grid,
        gnn_total=gnn_total, calc_total=calc_total,
        gnn_per_carbon=gnn_per_carbon, calc_per_carbon=calc_per_carbon,
        exp_spec=exp_spec,
        gnn_pcc=None, gnn_mse=None, gnn_mae=None,
        calc_pcc=None, calc_mse=None, calc_mae=None,
        gnn_vs_calc_pcc=None, gnn_vs_calc_mse=None, gnn_vs_calc_mae=None,
        per_carbon_pccs=[],
    )

    # -- Total PCCs (interpolated onto exp energy axis) --
    if gnn_total.max() > 0 and calc_total.max() > 0:
        gnn_norm  = gnn_total  / gnn_total.max()
        calc_norm = calc_total / calc_total.max()
        result['gnn_vs_calc_pcc'] = _compute_pcc(gnn_norm, calc_norm)
        result['gnn_vs_calc_mse'] = _compute_mse(gnn_norm, calc_norm)
        result['gnn_vs_calc_mae'] = _compute_mae(gnn_norm, calc_norm)

        if exp_spec is not None:
            exp_min, exp_max = exp_spec[:, 0].min(), exp_spec[:, 0].max()
            exp_base = np.linspace(exp_min, exp_max, n_points)
            try:
                fit_exp = interpolate.interp1d(exp_spec[:, 0], exp_spec[:, 1])(exp_base)
                fit_exp_norm = fit_exp / fit_exp.max()
                gnn_on_exp  = np.interp(exp_base, display_grid, gnn_norm)
                calc_on_exp = np.interp(exp_base, display_grid, calc_norm)
                result['gnn_pcc']  = _compute_pcc(gnn_on_exp,  fit_exp_norm)
                result['gnn_mse']  = _compute_mse(gnn_on_exp,  fit_exp_norm)
                result['gnn_mae']  = _compute_mae(gnn_on_exp,  fit_exp_norm)
                result['calc_pcc'] = _compute_pcc(calc_on_exp, fit_exp_norm)
                result['calc_mse'] = _compute_mse(calc_on_exp, fit_exp_norm)
                result['calc_mae'] = _compute_mae(calc_on_exp, fit_exp_norm)
            except Exception as e:
                print(f'  PCC error for {mol_id}: {e}')

    # -- Per-carbon PCCs (GNN vs calc on display_grid) --
    per_carbon_pccs = []
    for c in range(1, n_carbons + 1):
        a_idx = c - 1
        env_label = (carbon_env_labels[a_idx]
                     if carbon_env_labels is not None and a_idx < len(carbon_env_labels)
                     else '')
        g = gnn_per_carbon.get(a_idx)
        k = calc_per_carbon.get(c)
        pcc = mse = mae = None
        if g is not None and k is not None and g.max() > 0 and k.max() > 0:
            g_n = g / g.max()
            k_n = k / k.max()
            pcc = _compute_pcc(g_n, k_n)
            mse = _compute_mse(g_n, k_n)
            mae = _compute_mae(g_n, k_n)
        per_carbon_pccs.append((c, env_label, pcc, mse, mae))
    result['per_carbon_pccs'] = per_carbon_pccs

    return result


def _plot_overview(results, png_dir, file_stem, n_plot=16):
    """8x2 grid overview: GNN vs calc vs exp for up to 16 eval molecules."""
    n_plot = min(n_plot, len(results))
    fig, axes = plt.subplots(8, 2, figsize=(24, 22), sharex=True, sharey=True)
    ax = axes.ravel()

    for i in range(n_plot):
        r   = results[i]
        dg  = r['display_grid']
        exp = r['exp_spec']

        if exp is None:
            ax[i].text(0.5, 0.5, f"{r['mol_id']}\nNo exp",
                       transform=ax[i].transAxes, ha='center', va='center')
            continue

        ax[i].plot(exp[:, 0], exp[:, 1] / exp[:, 1].max(),
                   lw=2.5, color='k', ls='-', label='Experimental', alpha=0.8)
        if r['calc_total'].max() > 0:
            ax[i].plot(dg, r['calc_total'] / r['calc_total'].max(),
                       lw=2.5, color='g', ls='--', label='Calculated', alpha=0.8)
        if r['gnn_total'].max() > 0:
            ax[i].plot(dg, r['gnn_total'] / r['gnn_total'].max(),
                       lw=2.5, color='b', ls=':', label='GNN Fitted', alpha=0.8)

        ax[i].text(0.05, 0.95, r['mol_id'], transform=ax[i].transAxes,
                   fontsize=18, va='top',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        _add_pcc_annotation(ax[i], r['calc_pcc'], r['gnn_vs_calc_pcc'], r['gnn_pcc'])
        ax[i].set_xlim(220, 275)
        ax[i].set_ylim(0, 1.1)
        ax[i].tick_params(axis='y', labelleft=False)

    ax[0].legend(loc='upper right', fontsize=12)
    ax[6].set_ylabel('Normalized Intensity (arb. units)', fontsize=24)
    ax[12].set_xlabel('Kinetic Energy (eV)', fontsize=24)
    ax[13].set_xlabel('Kinetic Energy (eV)', fontsize=24)
    plt.subplots_adjust(hspace=0)
    plt.tight_layout()

    path = os.path.join(png_dir, f'{file_stem}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f'  Overview plot saved to {path}')
    plt.close(fig)


def _plot_per_molecule_carbons(results, png_dir, file_stem):
    """For each molecule: n_carbons rows x 1 col, GNN vs calc per carbon."""
    for r in results:
        mol_id   = r['mol_id']
        n_carbons = r['n_carbons']
        dg       = r['display_grid']
        gnn_pc   = r['gnn_per_carbon']
        calc_pc  = r['calc_per_carbon']
        pc_pccs  = {c: (env_label, pcc) for c, env_label, pcc, *_ in r['per_carbon_pccs']}

        if not calc_pc:
            continue

        fig, axes = plt.subplots(n_carbons, 1,
                                 figsize=(10, 2.5 * n_carbons),
                                 sharex=True, squeeze=False)
        axes = axes[:, 0]

        carbon_plot_path = os.path.join(png_dir, f'{file_stem}_carbons_plots')
        os.makedirs(carbon_plot_path, exist_ok=True)

        for row, c in enumerate(range(1, n_carbons + 1)):
            ax    = axes[row]
            a_idx = c - 1

            if c in calc_pc and calc_pc[c].max() > 0:
                ax.plot(dg, calc_pc[c] / calc_pc[c].max(),
                        lw=2.0, color='g', ls='--', label='Calculated', alpha=0.8)
            if a_idx in gnn_pc and gnn_pc[a_idx].max() > 0:
                ax.plot(dg, gnn_pc[a_idx] / gnn_pc[a_idx].max(),
                        lw=2.0, color='b', ls=':', label='GNN', alpha=0.8)

            pcc     = pc_pccs.get(c, (None, None))[1]
            env_lbl = pc_pccs.get(c, (None, None))[0] or ''
            pcc_str = f'PCC: {pcc:.3f}' if pcc is not None else 'PCC: N/A'
            ylabel  = f'C{c}' + (f'\n{env_lbl.removeprefix("C_")}' if env_lbl else '')
            ax.set_ylabel(ylabel, fontsize=11, fontweight='bold',
                          rotation=0, labelpad=28)
            ax.text(0.98, 0.78, pcc_str, transform=ax.transAxes,
                    ha='right', fontsize=10,
                    bbox=dict(facecolor='none', edgecolor='gray', boxstyle='round'))
            ax.set_ylim(0, 1.25)
            ax.tick_params(labelsize=9)

        axes[0].set_title(f'{mol_id}',
                          fontsize=13, fontweight='bold')
        axes[0].legend(loc='upper left', fontsize=9)
        axes[-1].set_xlabel('Kinetic Energy (eV)', fontsize=11)
        plt.tight_layout()

        safe_id = mol_id.replace('/', '_').replace(' ', '_')
        
        path = os.path.join(carbon_plot_path, f'{safe_id}.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)

    print(f' Per carbon plots save to: {carbon_plot_path}')

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

def _predict_spectra(model, eval_data, device):
    """Run inference on eval_data and return per-atom fitted intensity vectors.

    Returns
    -------
    dict[int, dict[int, np.ndarray]]
        ``{mol_idx: {atom_idx: array(n_points,)}}``  — intensity vectors.
    """
    loader = DataLoader(eval_data, batch_size=1, shuffle=False)
    predictions     = {}
    carbon_counts   = {}
    mol_list        = {}

    model.eval()
    with torch.no_grad():
        for mol_idx, data in enumerate(loader):
            data = data.to(device)
            out = model(data)
            # Multi-task models return (cebe_out, auger_out) — take the Auger head
            if getattr(model, 'task_type', 'single') == 'multi':
                out = out[1]

            node_mask = data.node_mask.squeeze()
            valid_nodes = node_mask.nonzero(as_tuple=True)[0]
            num_carbons = int(node_mask.sum().item())
            carbon_counts[mol_idx] = num_carbons
            name = data.mol_name
            mol_list[mol_idx] = name[0] if isinstance(name, list) else name

            predictions[mol_idx] = {}
            for atom_idx, node_idx in enumerate(valid_nodes):
                predictions[mol_idx][atom_idx] = out[node_idx].cpu().numpy()

    return predictions, carbon_counts, mol_list

def _evaluate_spectra(
    model, device, eval_data, test_data, train_data,
    *,
    output_dir, png_dir, file_stem,
    n_points, min_ke, max_ke, fwhm, ke_shift,
    train_results=None,
):
    """
    The model predicts per-atom intensity on a common energy grid.

    Returns a dict of PCC metrics.
    """
    print('\nGenerating predictions...')
    eval_predictions, eval_c_num, eval_mol_list = _predict_spectra(model, eval_data, device)
    print(f'  {len(eval_predictions)} eval molecules')
    test_predictions, test_c_num, test_mol_list = _predict_spectra(model, test_data, device)
    print(f'  {len(test_predictions)} test molecules')

    eval_dir = os.path.join(DATA_RAW_DIR, 'eval_auger')
    calc_dir = os.path.join(DATA_RAW_DIR, 'calc_auger')
    energy_grid = np.linspace(min_ke, max_ke, n_points)

    # Build training-set class counts from carbon_env_labels.
    # carbon_env_labels is per-ALL-atom; filter to carbons only via node_mask
    # to avoid counting 'non_carbon' entries for H, O, N, F atoms.
    train_env_counts: dict = {}
    if train_data is not None:
        for data_obj in train_data:
            if hasattr(data_obj, 'carbon_env_labels'):
                mask = data_obj.node_mask.squeeze().tolist()
                for lbl, m in zip(data_obj.carbon_env_labels, mask):
                    if m < 0.5:
                        continue   # skip non-carbon atoms
                    key = lbl.removeprefix('C_') if lbl else 'unknown'
                    train_env_counts[key] = train_env_counts.get(key, 0) + 1
    if train_env_counts:
        print(f'  Training env counts: {len(train_env_counts)} types, '
              f'{sum(train_env_counts.values())} total carbons')

    # Loss curves
    if train_results is not None and len(train_results) > 0:
        _plot_loss_curves(train_results, png_dir, file_stem)

    # Compute spectra + PCCs for every eval molecule
    results = []
    for i in range(len(eval_data)):
        # Filter carbon_env_labels to carbon atoms only (node_mask == 1).
        # The stored list is per-ALL-atom; using it directly with carbon
        # counter indices would pick up non-carbon ('non_carbon') labels.
        data_obj = eval_data[i]
        if hasattr(data_obj, 'carbon_env_labels'):
            mask = data_obj.node_mask.squeeze().tolist()
            env_labels_i = [
                lbl for lbl, m in zip(data_obj.carbon_env_labels, mask)
                if m > 0.5
            ]
        else:
            env_labels_i = None
        spec_idx_i = _carbon_spec_idx(data_obj)
        r = _compute_molecule_results(
            i, eval_mol_list[i], eval_predictions, int(eval_c_num[i]),
            eval_dir, energy_grid, n_points, fwhm, ke_shift, min_ke, max_ke,
            carbon_env_labels=env_labels_i,
            calc_method='mcpdft_hybrid_rcc',
            carbon_spec_idx=spec_idx_i,
        )
        results.append(r)

    # Plots
    _plot_overview(results, png_dir, file_stem)
    _plot_per_molecule_carbons(results, png_dir, file_stem)

    # Summary table
    pcc_data = [
        dict(
            molecule=r['mol_id'],
            gnn_pcc=r['gnn_pcc'],   gnn_mse=r.get('gnn_mse'),   gnn_mae=r.get('gnn_mae'),
            calc_pcc=r['calc_pcc'], calc_mse=r.get('calc_mse'),  calc_mae=r.get('calc_mae'),
            gnn_vs_calc_pcc=r['gnn_vs_calc_pcc'],
            gnn_vs_calc_mse=r.get('gnn_vs_calc_mse'),
            gnn_vs_calc_mae=r.get('gnn_vs_calc_mae'),
            per_carbon_pccs=r['per_carbon_pccs'],
        )
        for r in results
    ]
    eval_summary = _print_pcc_summary('eval set', pcc_data,
                                      train_env_counts=train_env_counts or None)

    # -- Test-set analysis (calc hold-out, no experimental data) --
    test_summary = None
    if test_data is not None and len(test_data) > 0:
        print('\nRunning test-set (calc hold-out) analysis...')
        test_results = []
        for i in range(len(test_data)):
            data_obj = test_data[i]
            if hasattr(data_obj, 'carbon_env_labels'):
                mask = data_obj.node_mask.squeeze().tolist()
                env_labels_i = [
                    lbl for lbl, m in zip(data_obj.carbon_env_labels, mask)
                    if m > 0.5
                ]
            else:
                env_labels_i = None
            spec_idx_i = _carbon_spec_idx(data_obj)
            r = _compute_molecule_results(
                i, test_mol_list[i], test_predictions, int(test_c_num[i]),
                calc_dir, energy_grid, n_points, fwhm, ke_shift, min_ke, max_ke,
                carbon_env_labels=env_labels_i,
                calc_method='auger',
                carbon_spec_idx=spec_idx_i,
            )
            test_results.append(r)

        # Per-molecule carbon plots 
        _plot_per_molecule_carbons(test_results, png_dir, f'{file_stem}_test')

        test_pcc_data = [
            dict(
                molecule=r['mol_id'],
                gnn_pcc=None, gnn_mse=None, gnn_mae=None,
                calc_pcc=None, calc_mse=None, calc_mae=None,
                gnn_vs_calc_pcc=r['gnn_vs_calc_pcc'],
                gnn_vs_calc_mse=r.get('gnn_vs_calc_mse'),
                gnn_vs_calc_mae=r.get('gnn_vs_calc_mae'),
                per_carbon_pccs=r['per_carbon_pccs'],
            )
            for r in test_results
        ]
        test_summary = _print_pcc_summary('calc test set (GNN vs Calc only)', test_pcc_data,
                           show_exp=False, train_env_counts=train_env_counts or None)

    # -- Save full evaluation results to JSON --
    import json

    def _make_serialisable(obj):
        """Recursively convert numpy scalars / NaN to JSON-safe types."""
        if isinstance(obj, dict):
            return {k: _make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_serialisable(v) for v in obj]
        if isinstance(obj, float) and (obj != obj):   # NaN check
            return None
        if hasattr(obj, 'item'):                       # numpy scalar
            return obj.item()
        return obj

    json_payload = {
        'file_stem':  file_stem,
        'eval_set':   _make_serialisable(eval_summary),
        'test_set':   _make_serialisable(test_summary) if test_data is not None and len(test_data) > 0 else None,
    }
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f'{file_stem}_eval_results.json')
    with open(json_path, 'w') as fh:
        json.dump(json_payload, fh, indent=2)
    print(f'  Evaluation results saved to {json_path}')

    return eval_summary

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
    train_calc_data=None,
    test_calc_data=None
):
    """Evaluate an Auger GNN model after training.

    Parameters
    ----------
    model_result : dict
        Training result dict from ``backend.train_single_run``.
    device : torch.device
    output_dir, png_dir : str
        Output directories.
    cfg : AugerNetConfig
        Full config — used for feature_keys, and spectrum
        parameters (max_spec_len, max_ke, n_points, fwhm, etc.).
    fold : int, optional
    train_results : list, optional
        ``[epoch, train_loss, val_loss]`` for loss-curve plotting.
    model_id : str
    config_id : str, optional
    param_file_prefix : str, optional
    """

    # Build file stem
    file_stem = f'{model_id}_fold{fold}' if fold is not None else model_id
    if config_id is not None:
        file_stem = f'{file_stem}_{config_id}'
    if param_file_prefix is not None:
        file_stem = f'{param_file_prefix}_{file_stem}'

    print(f"\n{'=' * 80}")
    print(f"AUGER EVALUATION"
          f"{f'  (fold {fold})' if fold else ''}")
    print(f"{'=' * 80}")

    # Load evaluation data
    feature_keys = cfg.feature_keys_parsed

    eval_ds = gtu.LoadDataset(DATA_DIR, file_name=cfg.auger_eval_data_file)
    eval_data = [eval_ds[i] for i in range(len(eval_ds))]
    test_data = test_calc_data
    train_data = train_calc_data
    assemble_dataset(eval_data, feature_keys)
    print(f'  Loaded {len(eval_data)} eval molecules')

    model = model_result.get('model') if isinstance(model_result, dict) else model_result

    return _evaluate_spectra(
        model, device, eval_data, test_data, train_data,
        output_dir=output_dir, png_dir=png_dir, file_stem=file_stem,
        n_points=cfg.n_points, min_ke=cfg.min_ke, max_ke=cfg.max_ke,
        fwhm=cfg.fwhm, ke_shift=cfg.ke_shift_calc,
        train_results=train_results,
    )