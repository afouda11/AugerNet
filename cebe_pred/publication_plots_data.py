#!/usr/bin/env python3
"""
Dataset-composition figure for the CEBE GNN paper.

Produces a 4-panel figure (1 row × 4 columns):

  Col 1 — molecule-size histogram: Train
  Col 2 — molecule-size histogram: Val
  Col 3 — molecule-size histogram: Exp. Eval
  Col 4 — grouped bar chart of atom-type (H / C / N / O / F) counts

Output:  ``pngs/data_analysis.{png,pdf}``

Usage::

    cd cebe_pred && python publication_plots_data.py
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.model_selection import KFold

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PNG_DIR      = os.path.join(SCRIPT_DIR, 'pngs')
os.makedirs(PNG_DIR, exist_ok=True)

# ── Parameters (match the trained model) ─────────────────────────────────────
N_FOLDS     = 5
FOLD        = 1        # 1-indexed, matches cebe_035_random_EQ4_h32_fold1
RANDOM_SEED = 42
ATOM_TYPES  = ['H', 'C', 'N', 'O', 'F']


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper: per-molecule atom count from slices
# ═══════════════════════════════════════════════════════════════════════════════

def _mol_sizes(slices):
    """Return an int array of per-molecule total-atom counts."""
    node_slices = slices['x']                          # shape (n_mols + 1,)
    return (node_slices[1:] - node_slices[:-1]).numpy()


def _atom_type_counts(feat_onehot, slices, mol_indices):
    """Sum one-hot atom features over the selected molecules.

    Returns an int array of length ``len(ATOM_TYPES)`` (H, C, N, O, F).
    """
    idx = []
    node_slices = slices['x']
    for mi in mol_indices:
        s = node_slices[mi].item()
        e = node_slices[mi + 1].item()
        idx.extend(range(s, e))
    return feat_onehot[idx].sum(dim=0).int().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main figure
# ═══════════════════════════════════════════════════════════════════════════════

def plot_combined_size_and_atomtype():
    """Build the 4-panel dataset-composition figure."""

    CALC_PT = os.path.join(PROJECT_ROOT, 'data', 'processed',
                           'gnn_calc_cebe_data.pt')
    EXP_PT  = os.path.join(PROJECT_ROOT, 'data', 'processed',
                           'gnn_exp_cebe_data.pt')

    # ── Load data ─────────────────────────────────────────────────────────
    data_calc, slices_calc = torch.load(CALC_PT, weights_only=False)
    data_exp,  slices_exp  = torch.load(EXP_PT,  weights_only=False)

    n_calc = len(slices_calc['x']) - 1
    n_exp  = len(slices_exp['x'])  - 1

    # ── Reproduce the train/val split ─────────────────────────────────────
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    folds = list(kf.split(np.arange(n_calc)))
    train_idx, val_idx = folds[FOLD - 1]

    # ── Per-molecule sizes ────────────────────────────────────────────────
    all_sizes_calc = _mol_sizes(slices_calc)
    all_sizes_exp  = _mol_sizes(slices_exp)

    n_train = all_sizes_calc[train_idx]
    n_val   = all_sizes_calc[val_idx]
    n_eval  = all_sizes_exp

    # ── Per-split atom-type counts ────────────────────────────────────────
    feat_calc = data_calc.feat_onehot
    feat_exp  = data_exp.feat_onehot

    cts_train = _atom_type_counts(feat_calc, slices_calc, train_idx)
    cts_val   = _atom_type_counts(feat_calc, slices_calc, val_idx)
    cts_eval  = feat_exp.sum(dim=0).int().numpy()

    # ── Colour scheme ─────────────────────────────────────────────────────
    colors = ['#0072B2', '#E69F00', '#4daf4a']    # train / val / eval

    AXIS_FONT   = 14
    LEGEND_FONT = 13
    TICK_FONT   = 14
    LABEL_FONT  = 9.5

    # ── Figure layout: 4 columns ──────────────────────────────────────────
    sns.set_style('whitegrid')
    plt.rcParams.update({
        'font.size':       10,
        'axes.linewidth':  1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size':  5,
        'ytick.major.size':  5,
    })

    fig = plt.figure(figsize=(17, 5))
    gs  = fig.add_gridspec(
        1, 4,
        width_ratios=[1.1, 1.1, 1.1, 1.1],
        wspace=0.13,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    # Push col 4 further right
    pos3 = axes[3].get_position()
    axes[3].set_position([pos3.x0 + 0.015, pos3.y0, pos3.width, pos3.height])

    global_max = int(max(n_train.max(), n_val.max(), n_eval.max()))

    size_panels = [
        (axes[0], n_train, f'Train ({len(n_train)})', colors[0]),
        (axes[1], n_val,   f'Val. ({len(n_val)})',   colors[1]),
        (axes[2], n_eval,  f'Exp. Eval. ({len(n_eval)})',  colors[2]),
    ]

    # ── Cols 1–3: molecule-size bar charts ────────────────────────────────
    for ax, data, title, color in size_panels:
        x_vals   = np.arange(1, global_max + 1)
        bar_vals = np.bincount(data, minlength=global_max + 1)[1:global_max + 1]
        ax.bar(x_vals, bar_vals, width=0.85,
               color=color, alpha=0.80, edgecolor='white',
               label=title, linewidth=0.3)

        ax.set_xlabel('Molecule Size', fontsize=AXIS_FONT, fontweight='bold')
        ax.legend(fontsize=LEGEND_FONT, framealpha=0.8, fancybox='round')
        ax.tick_params(axis='x', labelsize=TICK_FONT)
        ax.tick_params(axis='y', labelsize=TICK_FONT)
        ax.grid(True, alpha=0.35, linewidth=1.0, axis='both', zorder=0)

    axes[0].text(-0.22, 1.0, '(a)', transform=axes[0].transAxes,
                 fontsize=14, fontweight='bold', va='top')
    axes[0].set_xlim(3, 16.5)
    axes[1].set_xlim(3, 16.5)
    axes[2].set_xlim(2, 46)
    axes[0].xaxis.set_major_locator(ticker.MultipleLocator(2))
    axes[0].xaxis.set_minor_locator(ticker.MultipleLocator(2))
    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(2))
    axes[1].xaxis.set_minor_locator(ticker.MultipleLocator(2))
    axes[2].xaxis.set_major_locator(ticker.MultipleLocator(5))
    axes[2].xaxis.set_minor_locator(ticker.MultipleLocator(5))
    axes[0].set_ylabel('Molecule Count', fontsize=AXIS_FONT, fontweight='bold')

    # ── Col 4: atom-type grouped bar chart ────────────────────────────────
    ax4   = axes[3]
    x     = np.arange(len(ATOM_TYPES))
    width = 0.26

    b1 = ax4.bar(x - width, cts_train, width,
                 label='Train',
                 color=colors[0], edgecolor='white', alpha=0.80, linewidth=0.5)
    b2 = ax4.bar(x,          cts_val,   width,
                 label='Val.',
                 color=colors[1], edgecolor='white', alpha=0.80, linewidth=0.5)
    b3 = ax4.bar(x + width,  cts_eval,  width,
                 label='Exp. Eval.',
                 color=colors[2], edgecolor='white', alpha=0.80, linewidth=0.5)

    max_count = max(cts_train.max(), cts_val.max(), cts_eval.max())
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     h + max_count * 0.012,
                     f'{h:,}', ha='center', va='bottom',
                     fontsize=LABEL_FONT, fontweight='bold')

    ax4.text(3.275, 1.0, '(b)', transform=axes[0].transAxes,
             fontsize=14, fontweight='bold', va='top')
    ax4.set_xticks(x)
    ax4.set_xticklabels(ATOM_TYPES, fontsize=10)
    ax4.set_xlabel('Atom Type', fontsize=AXIS_FONT, fontweight='bold')
    ax4.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f'{v / 1000:.1f}'))
    ax4.set_ylabel('Atom Count (×10³)', fontsize=AXIS_FONT, fontweight='bold')
    ax4.tick_params(axis='x', labelsize=TICK_FONT)
    ax4.tick_params(axis='y', labelsize=TICK_FONT)
    ax4.set_ylim(0, max_count * 1.20)
    ax4.legend(framealpha=0.85, fontsize=LEGEND_FONT,
               loc='upper right', fancybox='round')
    ax4.grid(True, alpha=0.35, linewidth=1.0, axis='both', zorder=0)

    # ── Save ──────────────────────────────────────────────────────────────
    png_path = os.path.join(PNG_DIR, 'data_analysis.png')
    pdf_path = os.path.join(PNG_DIR, 'data_analysis.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f'  ✓ Combined size + atom-type → {png_path}')
    print(f'  ✓ Combined size + atom-type → {pdf_path}')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 72)
    print('  Dataset Composition Figure')
    print('=' * 72)
    plot_combined_size_and_atomtype()
    print('\n  ✅  Done.')