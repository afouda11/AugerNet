#!/usr/bin/env python3
"""
Dataset-composition figure for the CEBE GNN paper.

Produces a 2-row × 2-column figure (LaTeX \\columnwidth ≈ 3.5 in):

  Row 1 — Calc Train vs  Calc Val          (molecule size │ atom type)
  Row 2 — Exp Val    vs  Exp Eval          (molecule size │ atom type)

Left column:  molecule-size histograms  (two series side-by-side per panel)
Right column: grouped bar of atom-type counts  (H / C / N / O / F)

Output:  ``pngs/data_analysis.{png,pdf}``

Usage::

    cd cebe_pred && python publication_plots_data.py
"""

import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.model_selection import GroupKFold

# ── Project imports (for Butina + exp split) ─────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from augernet.build_molecular_graphs import get_butina_clusters

PNG_DIR = os.path.join(SCRIPT_DIR, 'pngs')
os.makedirs(PNG_DIR, exist_ok=True)

# ── Parameters (match the 10-fold Butina CV, fold 5) ────────────────────────
N_FOLDS    = 10
FOLD       = 5          # 1-indexed
ATOM_TYPES = ['H', 'C', 'N', 'O', 'F']


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _mol_sizes(slices):
    """Per-molecule total-atom counts."""
    s = slices['x']
    return (s[1:] - s[:-1]).numpy()


def _atom_type_counts(feat_onehot, slices, mol_indices):
    """Sum one-hot atom features for selected molecules → (5,) int array."""
    idx = []
    ns = slices['x']
    for mi in mol_indices:
        s, e = ns[mi].item(), ns[mi + 1].item()
        idx.extend(range(s, e))
    return feat_onehot[idx].sum(dim=0).int().numpy()


def _load_exp_split_names():
    """Read mol_list_val.txt / mol_list_eval.txt → (set, set)."""
    exp_dir = os.path.join(PROJECT_ROOT, 'data', 'raw', 'exp_cebe')
    def _read(fname):
        with open(os.path.join(exp_dir, fname)) as f:
            return {line.strip() for line in f if line.strip()}
    return _read('mol_list_val.txt'), _read('mol_list_eval.txt')


def _exp_split_indices(data_exp, slices_exp):
    """Return (val_indices, eval_indices) into the exp dataset."""
    n_exp = len(slices_exp['x']) - 1
    val_names, eval_names = _load_exp_split_names()
    val_idx, eval_idx = [], []
    for i in range(n_exp):
        name = data_exp.mol_name[i]
        if name in val_names:
            val_idx.append(i)
        elif name in eval_names:
            eval_idx.append(i)
    return np.array(val_idx), np.array(eval_idx)


# ═════════════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ═════════════════════════════════════════════════════════════════════════════

def _size_histogram(ax, sizes_a, sizes_b, label_a, label_b, color_a, color_b,
                    x_title=None, y_title=None,
                    xlim=None, AXIS_FONT=16, TICK_FONT=17, LEG_FONT=14):
    """Side-by-side bar histogram of molecule sizes."""
    lo = min(sizes_a.min(), sizes_b.min())
    hi = max(sizes_a.max(), sizes_b.max())
    bins = np.arange(lo, hi + 2) - 0.5
    counts_a, _ = np.histogram(sizes_a, bins=bins)
    counts_b, _ = np.histogram(sizes_b, bins=bins)
    centres = 0.5 * (bins[:-1] + bins[1:])
    w = 0.38
    ax.bar(centres - w/2, counts_a, width=w, color=color_a, alpha=0.85,
           edgecolor='white', linewidth=0.3, label=label_a, zorder=3)
    ax.bar(centres + w/2, counts_b, width=w, color=color_b, alpha=0.85,
           edgecolor='white', linewidth=0.3, label=label_b, zorder=3)
    if xlim:
        ax.set_xlim(*xlim)
    if x_title:
        ax.set_xlabel(x_title, fontsize=AXIS_FONT, fontweight='bold')
    if y_title:
        ax.set_ylabel(y_title, fontsize=AXIS_FONT, fontweight='bold')


def _atom_bar(ax, cts_a, cts_b, label_a, label_b, color_a, color_b,
              x_title=None, y_title=None,
              AXIS_FONT=16, TICK_FONT=17, LABEL_FONT=10,
              label_nudge=None):
    """Grouped bar chart for atom-type counts.

    label_nudge : dict, optional
        Extra vertical offset for specific bar labels, keyed by
        ``(series, atom_idx)`` where series 0 = cts_a, 1 = cts_b
        and atom_idx follows ATOM_TYPES order (0=H, 1=C, 2=N, …).
        Values are fractions of max_count, e.g. ``{(1, 1): 0.06}``.
    """
    if label_nudge is None:
        label_nudge = {}
    x = np.arange(len(ATOM_TYPES))
    w = 0.35
    b1 = ax.bar(x - w/2, cts_a, w, color=color_a, alpha=0.85,
                edgecolor='white', linewidth=0.3, label=label_a, zorder=3)
    b2 = ax.bar(x + w/2, cts_b, w, color=color_b, alpha=0.85,
                edgecolor='white', linewidth=0.3, label=label_b, zorder=3)

    max_count = max(cts_a.max(), cts_b.max())
    for si, bars in enumerate((b1, b2)):
        for ai, bar in enumerate(bars):
            h = bar.get_height()
            if h > 0:
                extra = label_nudge.get((si, ai), 0.0) * max_count
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + max_count * 0.015 + extra,
                        f'{h:,}', ha='center', va='bottom',
                        fontsize=LABEL_FONT, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(ATOM_TYPES, fontsize=TICK_FONT)
    if x_title:
        ax.set_xlabel(x_title, fontsize=AXIS_FONT, fontweight='bold')
    if y_title:
        ax.set_ylabel(y_title, fontsize=AXIS_FONT, fontweight='bold')
    ax.set_ylim(0, max_count * 1.22)


# ═════════════════════════════════════════════════════════════════════════════
#  Main figure
# ═════════════════════════════════════════════════════════════════════════════

def plot_data_composition():
    """Build the 3×2 dataset-composition figure."""

    CALC_PT = os.path.join(PROJECT_ROOT, 'data', 'processed',
                           'gnn_calc_cebe_data.pt')
    EXP_PT  = os.path.join(PROJECT_ROOT, 'data', 'processed',
                           'gnn_exp_cebe_data.pt')

    # ── Load ──────────────────────────────────────────────────────────────
    data_calc, sl_calc = torch.load(CALC_PT, weights_only=False)
    data_exp,  sl_exp  = torch.load(EXP_PT,  weights_only=False)

    n_calc = len(sl_calc['x']) - 1
    n_exp  = len(sl_exp['x'])  - 1

    # ── Calc train/val split (Butina, 10-fold, fold 5) ────────────────────
    smiles_list = [data_calc.smiles[i] for i in range(n_calc)]
    cluster_ids = get_butina_clusters(smiles_list, cutoff=0.65)
    n_clusters = len(set(cluster_ids))
    print(f"  Butina clustering → {n_clusters} clusters")

    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = list(gkf.split(np.arange(n_calc), groups=cluster_ids))
    train_idx, val_idx = folds[FOLD - 1]
    print(f"  Fold {FOLD}: {len(train_idx)} train / {len(val_idx)} val (calc)")

    # ── Exp val/eval split ────────────────────────────────────────────────
    exp_val_idx, exp_eval_idx = _exp_split_indices(data_exp, sl_exp)
    print(f"  Exp: {len(exp_val_idx)} val / {len(exp_eval_idx)} eval")

    # ── Sizes ─────────────────────────────────────────────────────────────
    sizes_calc = _mol_sizes(sl_calc)
    sizes_exp  = _mol_sizes(sl_exp)

    sizes_calc_train = sizes_calc[train_idx]
    sizes_calc_val   = sizes_calc[val_idx]
    sizes_exp_val    = sizes_exp[exp_val_idx]
    sizes_exp_eval   = sizes_exp[exp_eval_idx]

    # ── Atom-type counts ──────────────────────────────────────────────────
    feat_calc = data_calc.feat_onehot
    feat_exp  = data_exp.feat_onehot

    cts_calc_train = _atom_type_counts(feat_calc, sl_calc, train_idx)
    cts_calc_val   = _atom_type_counts(feat_calc, sl_calc, val_idx)
    cts_exp_val    = _atom_type_counts(feat_exp,  sl_exp,  exp_val_idx)
    cts_exp_eval   = _atom_type_counts(feat_exp,  sl_exp,  exp_eval_idx)

    # ── Colours ───────────────────────────────────────────────────────────
    c_train = '#0072B2'   # blue
    c_val   = '#E69F00'   # orange
    c_expv  = '#4daf4a'   # green
    c_expe  = '#984ea3'   # purple

    # ── Figure: 2 rows × 2 cols (double-column, ~7.2 in wide) ──────────
    sns.set_style('whitegrid')
    plt.rcParams.update({
        'font.size':        10,
        'axes.linewidth':   1.2,
        'xtick.direction':  'in',
        'ytick.direction':  'in',
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size':  5,
        'ytick.major.size':  5,
        'legend.frameon':   True,
    })

    AXIS_FONT  = 14
    TICK_FONT  = 13
    STATS_FONT = 10
    LEG_FONT   = 11
    LABEL_FONT = 9                # bar-top counts

    fig, axes = plt.subplots(
        2, 2, figsize=(7.2, 7.0),
        gridspec_kw={'hspace': 0.2, 'wspace': 0.2},
    )

    # ── Row 0: Calc Train vs Calc Val ─────────────────────────────────────
    _size_histogram(axes[0, 0], sizes_calc_train, sizes_calc_val,
                    f'Train ({len(train_idx)})', f'Calc-val. ({len(val_idx)})',
                    c_train, c_val, xlim=(2, 16.5), x_title=None, y_title="Count",
                    AXIS_FONT=AXIS_FONT, TICK_FONT=TICK_FONT, LEG_FONT=LEG_FONT)
    axes[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(2))

    _atom_bar(axes[0, 1], cts_calc_train, cts_calc_val,
                #'Train', 'Val', 
                f'Train ({len(train_idx)})', f'Calc-val. ({len(val_idx)})',
                c_train, c_val, x_title=None, y_title=None,
                AXIS_FONT=AXIS_FONT, TICK_FONT=TICK_FONT, LABEL_FONT=LABEL_FONT)
    axes[0, 1].yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f'{v/1000:.0f}k' if v >= 1000
                             else f'{v:.0f}'))

    # ── Row 1: Exp Val vs Exp Eval ────────────────────────────────────────
    _size_histogram(axes[1, 0], sizes_exp_val, sizes_exp_eval,
                    f'Exp-val. ({len(exp_val_idx)})',
                    f'Eval. ({len(exp_eval_idx)})',
                    c_expv, c_expe, xlim=(2, 46), x_title="Molecule size (atoms)", y_title="Count",
                    AXIS_FONT=AXIS_FONT, TICK_FONT=TICK_FONT, LEG_FONT=LEG_FONT)
    axes[1, 0].xaxis.set_major_locator(ticker.MultipleLocator(5))

    _atom_bar(axes[1, 1], cts_exp_val, cts_exp_eval,
              #'Exp Val', 'Exp Eval', 
                f'Exp-val. ({len(exp_val_idx)})',
                f'Eval. ({len(exp_eval_idx)})',
                c_expv, c_expe, x_title="Atom type", y_title=None,
                AXIS_FONT=AXIS_FONT, TICK_FONT=TICK_FONT, LABEL_FONT=LABEL_FONT,
                label_nudge={(1, 1): 0.05})

    # ── Cosmetics ─────────────────────────────────────────────────────────
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    for i, (ax, lbl) in enumerate(zip(axes.flat, panel_labels)):
        ax.tick_params(axis='both', labelsize=TICK_FONT)
        if lbl == '(b)' or lbl == '(d)':
            ax.legend(fontsize=LEG_FONT, framealpha=0.85, fancybox=True,
                  handlelength=1.0, handletextpad=0.4, borderpad=0.3)
        ax.grid(True, alpha=0.3, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        if lbl == '(b)' or lbl == '(d)':
            ax.text(-0.15, 1.08, lbl, transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top')
        else:
            ax.text(-0.23, 1.08, lbl, transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top')

    # ── Save ──────────────────────────────────────────────────────────────
    png_path = os.path.join(PNG_DIR, 'data_analysis.png')
    pdf_path = os.path.join(PNG_DIR, 'data_analysis.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f'  ✓ {png_path}')
    print(f'  ✓ {pdf_path}')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 60)
    print('  Dataset Composition Figure')
    print('=' * 60)
    plot_data_composition()
    print('\n  ✅  Done.')