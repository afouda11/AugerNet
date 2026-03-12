import os
import re
import sys
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import seaborn as sns
from collections import OrderedDict, Counter
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, rankdata


os.makedirs('pngs', exist_ok=True)

PARAM_SUMMARY_PATH = 'param_results/search_hidden_channels2_n_layers8_param_summary.json'

def load_results_data_and_compute_stats(data_path):
    results = np.loadtxt(data_path)           # (N, 2)
    exp  = results[:, 0]
    pred = results[:, 1]

    residuals = pred - exp
    mae = np.mean(np.abs(residuals))
    std = np.std(residuals)
    r2  = r2_score(exp, pred)

    stats = {'mae': mae, 'std': std, 'r2': r2}
    return results, stats

def load_label_data(data_path):
    molecules = []
    current_name = None
    true_vals, pred_vals = [], []

    def _flush():
        if current_name and true_vals:
            molecules.append({
                'name': current_name,
                'true': np.array(true_vals),
                'pred': np.array(pred_vals),
            })

    with open(data_path) as f:
        for line in f:
            line = line.rstrip('\n')
            # Molecule header: # --- name ---
            m = re.match(r'^# --- (.+) ---$', line)
            if m:
                _flush()
                current_name = m.group(1).strip()
                true_vals, pred_vals = [], []
                continue
            # Carbon data line:  C   293.7000   293.4897   -0.2103
            parts = line.split()
            if len(parts) == 4 and parts[0] == 'C' and parts[1] != '—':
                true_vals.append(float(parts[1]))
                pred_vals.append(float(parts[2]))
        _flush()

    return molecules


def compute_ranking_stats(molecules):
    """Compute per-molecule Spearman ρ and perfect-ranking flag.

    Only molecules with ≥ 2 *unique* experimental carbon CEBEs are
    included (single-carbon or all-equivalent molecules have no
    meaningful ranking).

    Parameters
    ----------
    molecules : list[dict]
        Output of :func:`load_label_data`.

    Returns
    -------
    rank_stats : list[dict]
        Each dict has keys ``'name'``, ``'n_carbons'``, ``'rho'``,
        ``'perfect'`` (bool).
    """
    rank_stats = []
    for mol in molecules:
        true, pred = mol['true'], mol['pred']
        # Need ≥ 2 distinct experimental values for a ranking
        if len(true) < 2 or len(np.unique(true)) < 2:
            continue
        rho, _ = spearmanr(true, pred)
        # Perfect ranking: dense ranks match (ties share the same rank,
        # so swapping two atoms with equal true CEBE is not penalised)
        true_ranks = rankdata(true, method='dense')
        pred_ranks = rankdata(pred, method='dense')
        perfect = np.array_equal(true_ranks, pred_ranks)
        rank_stats.append({
            'name':      mol['name'],
            'n_carbons': len(true),
            'rho':       rho,
            'perfect':   perfect,
        })
    return rank_stats

# ─────────────────────────────────────────────────────────────────────────────
#  Layer-sweep data loader + reusable axes-level plot
# ─────────────────────────────────────────────────────────────────────────────


def _grouping_key(run):
    """Return the sweep grouping key for a param-search run.

    If the run dict contains ``'layer_type'`` (layer-type sweep), use
    that (e.g. ``'EQ'``).  Otherwise fall back to
    ``'h{hidden_channels}'`` (hidden-dim sweep, e.g. ``'h32'``).
    """
    if 'layer_type' in run:
        return run['layer_type']
    return f"h{run['hidden_channels']}"


def load_layer_sweep_data(json_path=PARAM_SUMMARY_PATH):
    """Load layer-sweep param-search JSON and return per-group arrays.

    The grouping key is auto-detected: ``layer_type`` when present,
    otherwise ``h{hidden_channels}``.

    Returns
    -------
    sweep : dict
        ``{group_key: {'layers': np.array, 'train_loss': …,
        'val_loss': …, 'mae': …}}`` sorted by n_layers.
    """
    with open(json_path) as f:
        summary = json.load(f)

    sweep = {}
    for run in summary['runs']:
        gk = _grouping_key(run)
        sweep.setdefault(gk, {'layers': [], 'train_loss': [], 'val_loss': [], 'mae': []})
        sweep[gk]['layers'].append(run['n_layers'])
        sweep[gk]['train_loss'].append(run['best_train_loss'])
        sweep[gk]['val_loss'].append(run['best_val_loss'])
        sweep[gk]['mae'].append(run['eval_mae'])

    for gk in sweep:
        order = np.argsort(sweep[gk]['layers'])
        for k in ('layers', 'train_loss', 'val_loss', 'mae'):
            sweep[gk][k] = np.array(sweep[gk][k])[order]

    return sweep


def layer_sweep_plot(ax, sweep, model_type, LINE_WIDTH, STATS_FONT,
                     AXIS_FONT, TICK_FONT, subplot=True, panel_label='(b)',
                     log_scale=False):
    """Draw a dual-axis layer-sweep on *ax* for a single model type.

    Left y-axis: train loss + val loss.  Right y-axis: eval MAE.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes (left y-axis = loss).
    sweep : dict
        Output of :func:`load_layer_sweep_data`.
    model_type : str
        Grouping key — ``'EQ'``, ``'IN'`` (layer-type sweep) or
        ``'h16'``, ``'h32'`` (hidden-dim sweep).
        If True, use log₁₀ scale on the left (loss) axis.  Makes the
        growing train/val gap (overfitting) much more visible.
    """
    d = sweep[model_type]
    color_train = '#0072B2'   # blue
    color_val   = '#E69F00'   # orange
    color_mae   = '#4daf4a'   # green

    # Left axis — train & val loss
    ax.plot(d['layers'], d['train_loss'], 'o-', color=color_train,
            linewidth=LINE_WIDTH, markersize=5, label='Train loss', zorder=3)
    ax.plot(d['layers'], d['val_loss'], 's-', color=color_val,
            linewidth=LINE_WIDTH, markersize=5, label='Val loss', zorder=3)

    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel('Number of GNN Layers', fontsize=AXIS_FONT, fontweight='bold')
    ax.set_ylabel('Best Loss', fontsize=AXIS_FONT, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_FONT)
    ax.set_xticks(d['layers'])
    ax.grid(True, alpha=0.2, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)

    # Right axis — eval MAE
    ax2 = ax.twinx()
    ax2.plot(d['layers'], d['mae'], '^--', color=color_mae,
             linewidth=LINE_WIDTH, markersize=5, label='Exp. MAE', zorder=3)

    ax2.set_ylabel('Exp. MAE (eV)', fontsize=AXIS_FONT, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=TICK_FONT)

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    legend = ax.legend(h1 + h2, l1 + l2, fontsize=STATS_FONT, loc='upper right',
              framealpha=0.85)
    legend.set_zorder(10)

    if subplot:
        ax.text(-0.12, 1.05, panel_label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')

    return ax, ax2


# ─────────────────────────────────────────────────────────────────────────────
#  One-panel figure: layer sweep
# ─────────────────────────────────────────────────────────────────────────────

def layer_sweep(model_type='EQ'):
    """Single-panel dual-axis layer sweep (train/val loss + MAE vs n_layers)."""
    sweep = load_layer_sweep_data()

    AXIS_FONT  = 10
    TICK_FONT  = 10
    STATS_FONT = 9
    LINE_WIDTH = 1.6

    fig, ax = plt.subplots(figsize=(6, 4))
    layer_sweep_plot(ax, sweep, model_type, LINE_WIDTH, STATS_FONT,
                     AXIS_FONT, TICK_FONT, subplot=False, log_scale=True)

    fig.tight_layout()
    png_path = f'pngs/layer_sweep_{model_type}.png'
    pdf_path = f'pngs/layer_sweep_{model_type}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  ✓ {png_path}")
    print(f"  ✓ {pdf_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Layer-sweep ranking data loader + reusable axes-level plot
# ─────────────────────────────────────────────────────────────────────────────

def load_layer_sweep_rank_data(json_path=PARAM_SUMMARY_PATH):
    """Load per-molecule Spearman ρ for every layer-sweep config.

    The grouping key is auto-detected: ``layer_type`` when present,
    otherwise ``h{hidden_channels}``.

    Returns
    -------
    rank_sweep : dict
        ``{group_key: {'layers': np.array, 'mae': np.array,
        'mean_rho': np.array, 'median_rho': np.array,
        'frac_perfect': np.array}}`` sorted by n_layers.
    """
    with open(json_path) as f:
        summary = json.load(f)

    search_id = summary['search_id']
    output_dir = 'param_results/outputs'

    rank_sweep = {}
    for run in summary['runs']:
        gk = _grouping_key(run)
        rank_sweep.setdefault(gk, {
            'layers': [], 'mae': [],
            'mean_rho': [], 'median_rho': [], 'frac_perfect': [],
        })

        labels_path = os.path.join(
            output_dir,
            f"{search_id}_{run['model_id']}_fold1_{run['config_id']}_labels.txt",
        )
        molecules   = load_label_data(labels_path)
        rank_stats  = compute_ranking_stats(molecules)
        rhos        = np.array([r['rho'] for r in rank_stats])
        n_perfect   = sum(r['perfect'] for r in rank_stats)
        n_mols      = len(rank_stats)

        rank_sweep[gk]['layers'].append(run['n_layers'])
        rank_sweep[gk]['mae'].append(run['eval_mae'])
        rank_sweep[gk]['mean_rho'].append(rhos.mean() if len(rhos) else 0)
        rank_sweep[gk]['median_rho'].append(np.median(rhos) if len(rhos) else 0)
        rank_sweep[gk]['frac_perfect'].append(
            n_perfect / n_mols if n_mols else 0)

    for gk in rank_sweep:
        order = np.argsort(rank_sweep[gk]['layers'])
        for k in rank_sweep[gk]:
            rank_sweep[gk][k] = np.array(rank_sweep[gk][k])[order]

    return rank_sweep


def layer_sweep_rank_plot(ax, rank_sweep, model_type, LINE_WIDTH,
                          STATS_FONT, AXIS_FONT, TICK_FONT,
                          subplot=True, panel_label='(b)'):
    """Dual-axis layer-sweep: Spearman ρ (left) vs Exp. MAE (right).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes (left y-axis = Spearman ρ).
    rank_sweep : dict
        Output of :func:`load_layer_sweep_rank_data`.
    model_type : str
        Grouping key — ``'EQ'``, ``'IN'`` (layer-type sweep) or
        ``'h16'``, ``'h32'`` (hidden-dim sweep).
    """
    d = rank_sweep[model_type]
    color_mean   = '#0072B2'   # blue
    color_median = '#E69F00'   # orange
    color_mae    = '#4daf4a'   # green

    # Left axis — Spearman ρ
    ax.plot(d['layers'], d['mean_rho'], 'o-', color=color_mean,
            linewidth=LINE_WIDTH, markersize=5,
            label=f'Mean $\\rho$', zorder=3)
#     ax.plot(d['layers'], d['median_rho'], 's-', color=color_median,
#             linewidth=LINE_WIDTH, markersize=5,
#             label=f'Median $\\rho$', zorder=3)

    ax.set_xlabel('Number of GNN Layers', fontsize=AXIS_FONT,
                  fontweight='bold')
    ax.set_ylabel(f'Spearman $\\rho$ (per molecule)', fontsize=AXIS_FONT,
                  fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_FONT)
    ax.set_xticks(d['layers'])
    ax.grid(True, alpha=0.3, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)

    # Right axis — eval MAE
    ax2 = ax.twinx()
    ax2.plot(d['layers'], d['mae'], '^--', color=color_mae,
             linewidth=LINE_WIDTH, markersize=5,
             label='Exp. MAE', zorder=3)
    ax2.set_ylabel('Exp. MAE (eV)', fontsize=AXIS_FONT, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=TICK_FONT)

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    legend = ax.legend(h1 + h2, l1 + l2, fontsize=STATS_FONT, loc='upper right',
              framealpha=0.85)
    legend.set_zorder(10) 

    if subplot:
        ax.text(-0.12, 1.05, panel_label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')

    return ax, ax2


# ─────────────────────────────────────────────────────────────────────────────
#  One-panel figure: layer sweep ranking
# ─────────────────────────────────────────────────────────────────────────────

def layer_sweep_rank(model_type='EQ'):
    """Single-panel: Spearman ρ (mean/median) + Exp. MAE vs n_layers."""
    rank_sweep = load_layer_sweep_rank_data()

    AXIS_FONT  = 10
    TICK_FONT  = 10
    STATS_FONT = 9
    LINE_WIDTH = 1.6

    fig, ax = plt.subplots(figsize=(6, 4))
    layer_sweep_rank_plot(ax, rank_sweep, model_type, LINE_WIDTH,
                          STATS_FONT, AXIS_FONT, TICK_FONT, subplot=False)

    fig.tight_layout()
    png_path = f'pngs/layer_sweep_rank_{model_type}.png'
    pdf_path = f'pngs/layer_sweep_rank_{model_type}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  ✓ {png_path}")
    print(f"  ✓ {pdf_path}")
    plt.close(fig)


def scatter_plot(ax, all_pred, all_exp, rstats, scatter_s, 
                   LINE_WIDTH, STATS_FONT, AXIS_FONT, TICK_FONT, subplot = True):

    ax.scatter(all_pred, all_exp, alpha=0.4, s=scatter_s, edgecolors='k',
               linewidth=0.2, color='#0072B2', zorder=3)

    lo = min(all_exp.min(), all_pred.min())
    hi = max(all_exp.max(), all_pred.max())
    pad = (hi - lo) * 0.03
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
              'r--', linewidth=LINE_WIDTH, alpha=0.7, zorder=2)

    ax.text(
        0.05, 0.95,
        f"R$^{{2}}$ = {rstats['r2']:.2f}\n"
        f"MAE = {rstats['mae']:.2f} eV\n"
        f"STD = {rstats['std']:.2f} eV",
        ha='left', va='top', transform=ax.transAxes,
        fontsize=STATS_FONT, fontweight='bold',
        bbox=dict(boxstyle='round', edgecolor='grey',
                  facecolor='white', alpha=0.85, pad=0.5),
        zorder=5,
    )

    ax.set_xlabel('GNN Predicted CEBE (eV)', fontsize=AXIS_FONT,
                  fontweight='bold')
    ax.set_ylabel('Experimental CEBE (eV)', fontsize=AXIS_FONT,
                  fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_FONT)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.grid(True, alpha=0.3, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    if subplot:
        ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes,
              fontsize=16, fontweight='bold', va='top')
    return ax

# ─────────────────────────────────────────────────────────────────────────────
#  One-panel figure: scatter
# ─────────────────────────────────────────────────────────────────────────────

def scatter(model_type='EQ', result_file=None):

    if result_file is None:
        result_file = f'cebe_035_random_{model_type}4_h32_fold1'

    results_path = f'train_results/outputs/{result_file}_results.txt'
    rdata, rstats = load_results_data_and_compute_stats(results_path)

    all_exp  = rdata[:, 0]
    all_pred = rdata[:, 1]

    AXIS_FONT  = 10
    TICK_FONT  = 10
    STATS_FONT = 10
    LINE_WIDTH = 1.6
    scatter_s  = 20

    fig, ax = plt.subplots(figsize=(5, 4))

    ax = scatter_plot(ax, all_pred, all_exp, rstats, scatter_s,  
                    LINE_WIDTH, STATS_FONT, AXIS_FONT, TICK_FONT, subplot=False)

    # ── Save ─────────────────────────────────────────────────────────────
    png_path = f'pngs/{result_file}_scatter.png'
    pdf_path = f'pngs/{result_file}_scatter.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  ✓ {png_path}")
    print(f"  ✓ {pdf_path}")
    plt.close(fig)
# ─────────────────────────────────────────────────────────────────────────────
#  Two-panel figure: (a) scatter  (b) Spearman ρ histogram
# ─────────────────────────────────────────────────────────────────────────────

def scatter_and_rank(model_type='EQ'):
    """Combined two-panel figure: predicted-vs-experimental scatter (left)
    and per-molecule Spearman rank-correlation histogram (right)."""

    result = f'cebe_035_random_{model_type}4_h32_fold1'

    results_path = f'train_results/outputs/{result}_results.txt'
    labels_path  = f'train_results/outputs/{result}_labels.txt'

    # ── Load data ────────────────────────────────────────────────────────
    rdata, rstats = load_results_data_and_compute_stats(results_path)
    molecules     = load_label_data(labels_path)
    rank_stats    = compute_ranking_stats(molecules)

    all_exp  = rdata[:, 0]
    all_pred = rdata[:, 1]
    rhos     = np.array([r['rho'] for r in rank_stats])
    n_perfect = sum(r['perfect'] for r in rank_stats)
    n_mols    = len(rank_stats)
    frac_perfect = n_perfect / n_mols if n_mols else 0

    print(f"  Scatter:  R²={rstats['r2']:.4f}  MAE={rstats['mae']:.4f} eV  "
          f"STD={rstats['std']:.4f} eV")
    print(f"  Ranking:  {n_mols} molecules with ≥2 unique C CEBEs")
    print(f"            mean ρ = {rhos.mean():.4f}  median ρ = {np.median(rhos):.4f}")
    print(f"            perfect ranking: {n_perfect}/{n_mols} "
          f"({frac_perfect:.0%})")

    # ── Style ────────────────────────────────────────────────────────────
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 10, 'axes.linewidth': 1.2,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
        'xtick.major.size': 5,    'ytick.major.size': 5,
        'legend.frameon': True,
    })
    AXIS_FONT  = 16
    TICK_FONT  = 17
    STATS_FONT = 16
    LINE_WIDTH = 2.2
    scatter_s  = 40

    fig = plt.figure(figsize=(16, 6), constrained_layout=True)
    gs  = GridSpec(1, 2, figure=fig, wspace=0.02)

    # ══════════════════════════════════════════════════════════════════════
    # Panel (a): Predicted vs Experimental scatter
#     # ══════════════════════════════════════════════════════════════════════
    ax_s = fig.add_subplot(gs[0, 0])

    ax_s = scatter_plot(ax_s, all_pred, all_exp, rstats, scatter_s, 
                           LINE_WIDTH, STATS_FONT, AXIS_FONT, TICK_FONT)

    ax_r = fig.add_subplot(gs[0, 1])

    bins = np.arange(-0.025, 1.075, 0.02)
    ax_r.hist(rhos, bins=bins, color='#0072B2', edgecolor='k',
              linewidth=0.6, alpha=0.8, zorder=3)

    # Dashed line at median
    med = np.median(rhos)
#     ax_r.axvline(med, color='#D55E00', linestyle='--',
#                  linewidth=LINE_WIDTH, zorder=4,
#                  label=f'median ρ = {med:.2f}')

    ax_r.text(
        0.05, 0.95,
        #f"N = {n_mols} molecules\n"
        f"mean = {rhos.mean():.2f}\n"
        f"median = {med:.2f}\n"
        f"perfect order: {n_perfect}/{n_mols} ({frac_perfect:.0%})",
        ha='left', va='top', transform=ax_r.transAxes,
        fontsize=STATS_FONT, fontweight='bold',
        bbox=dict(boxstyle='round', edgecolor='grey',
                  facecolor='white', alpha=0.85, pad=0.5),
        zorder=5,
    )

    ax_r.set_xlabel(f'Spearman $\\rho$ (per molecule)', fontsize=AXIS_FONT,
                    fontweight='bold')
    ax_r.set_ylabel('Number of Molecules', fontsize=AXIS_FONT,
                    fontweight='bold')
    ax_r.tick_params(axis='both', labelsize=TICK_FONT)
    ax_r.set_xlim(0.38, 1.025)
    #ax_r.legend(fontsize=14, loc='upper left', bbox_to_anchor=(0.35, 0.95))

    ax_r.grid(True, alpha=0.3, linewidth=1.0, zorder=0)
    ax_r.set_axisbelow(True)
    ax_r.text(-0.12, 1.05, '(b)', transform=ax_r.transAxes,
              fontsize=16, fontweight='bold', va='top')

    # ── Save ─────────────────────────────────────────────────────────────
    png_path = f'pngs/{result}_scatter_rank.png'
    pdf_path = f'pngs/{result}_scatter_rank.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  ✓ {png_path}")
    print(f"  ✓ {pdf_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Locality analysis: residual CEBE std vs bond radius
# ─────────────────────────────────────────────────────────────────────────────

def locality_plot(ax, loc_data, LINE_WIDTH, STATS_FONT, AXIS_FONT, TICK_FONT,
                  fig=None, subplot=True, panel_label='(b)',
                  xlim=(0.5, 5.5), ylim=(-0.02, 1.0)):
    """Draw residual intra-group CEBE std vs topological radius on *ax*.

    Each multi-member Morgan-FP group is shown as a scatter point
    (colour = number of carbons in the group).  The mean intra-group
    σ line is overlaid.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    loc_data : dict
        Output of ``locality_analysis.compute_locality_data()``.
    fig : matplotlib.figure.Figure or None
        If provided, a colorbar is attached to *fig*.  Pass ``None``
        to skip the colorbar (useful in tight multi-panel layouts).
    xlim, ylim : tuple
        Axis limits.  Set to ``None`` for automatic.
    """
    radii         = loc_data['radii']
    mean_std      = loc_data['mean_intra_std']
    group_details = loc_data['group_details']   # list-of-lists of (n, std)

    # ── Scatter: per-group std, coloured by group size ───────────────────
    scatter_r, scatter_std, scatter_n = [], [], []
    for r, gd in zip(radii, group_details):
        for n_g, std_g in gd:
            scatter_r.append(r)
            scatter_std.append(std_g)
            scatter_n.append(n_g)

    sc = None
    jitter = None
    if scatter_n:
        vmin, vmax = min(scatter_n), max(scatter_n)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.viridis
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12,
                                                    size=len(scatter_r))
        sc = ax.scatter(np.array(scatter_r) + jitter, scatter_std,
                        c=scatter_n, cmap=cmap, norm=norm,
                        s=28, alpha=0.7, edgecolors='white',
                        linewidths=0.4, zorder=2)
        if fig is not None:
            cbar = fig.colorbar(sc, ax=ax, pad=0.02, aspect=25, shrink=0.85)
            cbar.set_label('Carbons in group', fontsize=AXIS_FONT)
            cbar.ax.tick_params(labelsize=TICK_FONT - 1)

    # ── Mean intra-group std line ────────────────────────────────────────
    ax.plot(radii, mean_std, '-o', color='grey', markersize=7,
            linewidth=LINE_WIDTH, label='Mean Std', zorder=4)

    ax.set_xlabel('Topological (Bond) Radius', fontsize=AXIS_FONT,
                  fontweight='bold')
    ax.set_ylabel('Exp. CEBE Std (eV) (Intra Env.)', fontsize=AXIS_FONT,
                  fontweight='bold')
    ax.set_xticks(radii)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis='both', labelsize=TICK_FONT)
    ax.legend(fontsize=STATS_FONT - 1, loc='upper right', framealpha=0.85)
    ax.grid(True, alpha=0.25, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
 
    # ── Annotate aryl-fluoride outlier groups ────────────────────────────
    # Radius 2 group 2 (N=6, std≈0.498) and radius 3 group 3 (N=5, std≈0.514)
    _annotate_targets = [
        (2, 6, 0.498),   # radius 2, group 2
        (3, 5, 0.514),   # radius 3, group 3
    ]
    for _tgt_r, _tgt_n, _tgt_std in _annotate_targets:
        # Find the matching scatter point (closest std at that radius/size)
        best_idx, best_dist = None, 1e9
        for i, (sr, ss, sn) in enumerate(zip(scatter_r, scatter_std,
                                              scatter_n)):
            if sr == _tgt_r and sn == _tgt_n:
                dist = abs(ss - _tgt_std)
                if dist < best_dist:
                    best_dist, best_idx = dist, i
        if best_idx is not None:
            ax.annotate(
                '*', xy=(scatter_r[best_idx] + jitter[best_idx]+0.1,
                         scatter_std[best_idx]-0.04),
                fontsize=16, fontweight='bold', color='orange',
                ha='center', va='bottom', zorder=5)

    # Text box explaining the star annotations
    ax.text(3.8, 0.6, '* Para-di-substituted fluorobenzenes\n  C-F spread = 1.22 eV',
              fontsize=10, color='orange', fontweight='bold',
              ha='center', va='center', zorder=5,
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='none', alpha=0.9))


    if subplot:
        ax.text(-0.12, 1.05, panel_label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')

    return ax


# ─────────────────────────────────────────────────────────────────────────────
#  One-panel figure: locality
# ─────────────────────────────────────────────────────────────────────────────

def locality():
    """Single-panel: residual CEBE std vs bond radius."""
    from locality_analysis import compute_locality_data

    loc_data = compute_locality_data()

    AXIS_FONT  = 10
    TICK_FONT  = 10
    STATS_FONT = 9
    LINE_WIDTH = 1.6

    fig, ax = plt.subplots(figsize=(6, 4.5))
    locality_plot(ax, loc_data, LINE_WIDTH, STATS_FONT,
                  AXIS_FONT, TICK_FONT, fig=fig, subplot=False)

    fig.tight_layout()
    png_path = 'pngs/locality_variance.png'
    pdf_path = 'pngs/locality_variance.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  ✓ {png_path}")
    print(f"  ✓ {pdf_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Two-panel figure: (a) layer sweep  (b) locality variance
# ─────────────────────────────────────────────────────────────────────────────

def layer_sweep_and_locality(model_type='h32'):
    """Single-column two-row figure: layer sweep (top) + locality (bottom)."""
    from locality_analysis import compute_locality_data

    sweep    = load_layer_sweep_data()
    loc_data = compute_locality_data()

    # ── Style ────────────────────────────────────────────────────────────
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 10, 'axes.linewidth': 1.2,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
        'xtick.major.size': 5,    'ytick.major.size': 5,
        'legend.frameon': True,
    })
    AXIS_FONT  = 14
    TICK_FONT  = 13
    STATS_FONT = 12
    LINE_WIDTH = 2.0

    fig = plt.figure(figsize=(8, 10), constrained_layout=True)
    gs  = GridSpec(2, 1, figure=fig, hspace=0.05)

    # Panel (a): layer sweep
    ax_a = fig.add_subplot(gs[0, 0])
    layer_sweep_plot(ax_a, sweep, model_type, LINE_WIDTH, STATS_FONT,
                     AXIS_FONT, TICK_FONT, subplot=True, panel_label='(a)',
                     log_scale=True)

    # Panel (b): locality variance
    ax_b = fig.add_subplot(gs[1, 0])
    locality_plot(ax_b, loc_data, LINE_WIDTH, STATS_FONT,
                  AXIS_FONT, TICK_FONT, fig=fig, subplot=True,
                  panel_label='(b)')

    # ── Save ─────────────────────────────────────────────────────────────
    tag = model_type
    png_path = f'pngs/layer_sweep_locality_{tag}.png'
    pdf_path = f'pngs/layer_sweep_locality_{tag}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  ✓ {png_path}")
    print(f"  ✓ {pdf_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Aryl-fluoride C–F spread + MAE vs GNN layers
# ─────────────────────────────────────────────────────────────────────────────

CF_GROUP_FILE = 'locality_analysis_groups/radius_2/group_002.txt'


def _parse_aryl_fluoride_targets(group_file=CF_GROUP_FILE):
    """Parse the aryl-fluoride group file into targets and experimental data.

    Returns
    -------
    targets : list of (mol_name, atom_idx, exp_cebe)
    exp_cebes : np.ndarray
    exp_spread : float
    """
    targets = []
    with open(group_file) as f:
        in_table = False
        for line in f:
            if line.startswith('---'):
                in_table = True
                continue
            if not in_table:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            targets.append((parts[0], int(parts[1]), float(parts[2])))
    exp_cebes = np.array([c for _, _, c in targets])
    exp_spread = float(exp_cebes.max() - exp_cebes.min())
    return targets, exp_cebes, exp_spread


def _parse_atom_predictions(labels_path):
    """Parse a labels file and return per-atom predictions by molecule.

    Returns
    -------
    dict : {mol_name: list of (symbol, true_BE, pred_BE, error) or None}
        One entry per atom line.  Non-predicted atoms (``—``) yield None.
    """
    results = {}
    current_mol = None
    atoms = []

    with open(labels_path) as fh:
        for line in fh:
            m = re.match(r'^# --- (.+?) ---', line)
            if m:
                if current_mol is not None:
                    results[current_mol] = atoms
                current_mol = m.group(1).strip()
                atoms = []
                continue
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            parts = stripped.split()
            if len(parts) >= 4 and parts[1] != '—':
                try:
                    atoms.append((parts[0], float(parts[1]),
                                  float(parts[2]), float(parts[3])))
                except ValueError:
                    atoms.append(None)
            else:
                atoms.append(None)
    if current_mol is not None:
        results[current_mol] = atoms
    return results


def _load_cf_spread_data(model_type='h32',
                         json_path=PARAM_SUMMARY_PATH,
                         group_file=CF_GROUP_FILE):
    """Compute C–F predicted spread and MAE at each layer count.

    Reads the param-search summary JSON to locate per-layer labels
    files, then extracts the predicted CEBE for each C–F target atom.

    Returns
    -------
    layer_counts : list of int
    pred_spreads : list of float
    pred_maes : list of float
    exp_spread : float
    """
    targets, exp_cebes, exp_spread = _parse_aryl_fluoride_targets(group_file)

    with open(json_path) as f:
        summary = json.load(f)

    search_id = summary['search_id']
    output_dir = 'param_results/outputs'

    # Filter runs matching the requested hidden dim, sorted by n_layers
    hid = int(model_type.replace('h', ''))
    runs = sorted(
        [r for r in summary['runs'] if r['hidden_channels'] == hid],
        key=lambda r: r['n_layers'],
    )

    layer_counts = []
    pred_spreads = []
    pred_maes = []

    for run in runs:
        labels_path = os.path.join(
            output_dir,
            f"{search_id}_{run['model_id']}_fold1_{run['config_id']}_labels.txt",
        )
        atom_data = _parse_atom_predictions(labels_path)

        preds = []
        for mol, aidx, _ in targets:
            atoms = atom_data.get(mol, [])
            if aidx < len(atoms) and atoms[aidx] is not None:
                preds.append(atoms[aidx][2])  # pred_BE
            else:
                preds.append(np.nan)

        p = np.array(preds)
        valid_mask = ~np.isnan(p)
        valid = p[valid_mask]
        exp_valid = exp_cebes[valid_mask]

        layer_counts.append(run['n_layers'])
        if len(valid) >= 2:
            pred_spreads.append(float(valid.max() - valid.min()))
            pred_maes.append(float(np.mean(np.abs(valid - exp_valid))))
        else:
            pred_spreads.append(np.nan)
            pred_maes.append(np.nan)

    return layer_counts, pred_spreads, pred_maes, exp_spread


def aryl_fluoride_spread_mae_plot(ax, layer_counts, pred_spreads, pred_maes,
                                  exp_spread, LINE_WIDTH, STATS_FONT,
                                  AXIS_FONT, TICK_FONT,
                                  subplot=True, panel_label='(a)'):
    """Dual-axis panel: C–F predicted spread (left) + MAE (right) vs layers.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes (left y-axis = spread).
    """
    c_spread = '#0072B2'
    c_mae = '#D55E00'

    ax2 = ax.twinx()

    # Left axis: spread
    l1, = ax.plot(layer_counts, pred_spreads, '-o', color=c_spread,
                  markersize=7, linewidth=LINE_WIDTH,
                  label='Predicted spread', zorder=3)
    l_exp = ax.axhline(exp_spread, ls=':', color=c_spread,
                       lw=1.5, alpha=0.5,
                       label=f'Exp. spread = {exp_spread:.2f} eV',
                       zorder=1)

    ax.set_ylabel('C–F CEBE Spread (eV)', fontsize=AXIS_FONT,
                  fontweight='bold', color=c_spread)
    ax.tick_params(axis='y', labelcolor=c_spread, labelsize=TICK_FONT)
    ax.tick_params(axis='x', labelsize=TICK_FONT)
    ax.set_xlabel('Number of GNN Layers', fontsize=AXIS_FONT,
                  fontweight='bold')
    ax.set_xticks(layer_counts)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    # Right axis: MAE
    l2, = ax2.plot(layer_counts, pred_maes, '--^', color=c_mae,
                   markersize=7, linewidth=LINE_WIDTH,
                   label='C–F MAE', zorder=3)

    ax2.set_ylabel('C–F CEBE MAE (eV)', fontsize=AXIS_FONT,
                   fontweight='bold', color=c_mae)
    ax2.tick_params(axis='y', labelcolor=c_mae, labelsize=TICK_FONT)
    ax2.set_ylim(bottom=0)

    # Combined legend
    lines = [l1, l_exp, l2]
    labs = [l.get_label() for l in lines]
    legend = ax.legend(lines, labs, fontsize=STATS_FONT, loc='lower right',
                       framealpha=0.85)
    legend.set_zorder(10)

    if subplot:
        ax.text(-0.12, 1.05, panel_label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')

    return ax, ax2


def aryl_fluoride_spread_mae(model_type='h32'):
    """Single-panel figure: C–F spread + MAE vs GNN layers."""
    layer_counts, spreads, maes, exp_spread = _load_cf_spread_data(model_type)

    # Print table
    print(f"\n  Aryl-fluoride C–F group ({model_type}) — spread + MAE vs layers")
    print(f"  Experimental spread = {exp_spread:.3f} eV")
    print(f"\n  {'Layers':>6s}  {'Spread':>8s}  {'MAE':>8s}")
    print("  " + "-" * 28)
    for n, ps, pm in zip(layer_counts, spreads, maes):
        print(f"  {n:6d}  {ps:8.3f}  {pm:8.3f}")

    AXIS_FONT  = 10
    TICK_FONT  = 10
    STATS_FONT = 9
    LINE_WIDTH = 1.6

    fig, ax = plt.subplots(figsize=(6, 4.5))
    aryl_fluoride_spread_mae_plot(ax, layer_counts, spreads, maes,
                                  exp_spread, LINE_WIDTH, STATS_FONT,
                                  AXIS_FONT, TICK_FONT, subplot=False)

    fig.tight_layout()
    png_path = f'pngs/aryl_fluoride_spread_mae_{model_type}.png'
    pdf_path = f'pngs/aryl_fluoride_spread_mae_{model_type}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  ✓ {png_path}")
    print(f"  ✓ {pdf_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Aryl-fluoride C–F : predicted CEBE per molecule vs layers + Spearman ρ
# ─────────────────────────────────────────────────────────────────────────────

def _load_cf_predicted_cebes(model_type='h32',
                             json_path=PARAM_SUMMARY_PATH,
                             group_file=CF_GROUP_FILE):
    """Return per-molecule predicted C–F CEBEs at each GNN layer count.

    Returns
    -------
    layer_counts : list of int
    mol_pred : dict  {mol_name: list of float (one per layer)}
    targets : list of (mol_name, atom_idx, exp_cebe)
    exp_cebes : np.ndarray
    """
    targets, exp_cebes, _ = _parse_aryl_fluoride_targets(group_file)

    with open(json_path) as f:
        summary = json.load(f)

    search_id = summary['search_id']
    output_dir = 'param_results/outputs'

    hid = int(model_type.replace('h', ''))
    runs = sorted(
        [r for r in summary['runs'] if r['hidden_channels'] == hid],
        key=lambda r: r['n_layers'],
    )

    layer_counts = [r['n_layers'] for r in runs]
    mol_pred = {mol: [] for mol, _, _ in targets}

    for run in runs:
        labels_path = os.path.join(
            output_dir,
            f"{search_id}_{run['model_id']}_fold1_{run['config_id']}_labels.txt",
        )
        atom_data = _parse_atom_predictions(labels_path)
        for mol, aidx, _ in targets:
            atoms = atom_data.get(mol, [])
            if aidx < len(atoms) and atoms[aidx] is not None:
                mol_pred[mol].append(atoms[aidx][2])  # pred_BE
            else:
                mol_pred[mol].append(np.nan)

    return layer_counts, mol_pred, targets, exp_cebes


def _load_mean_spearman_rho(model_type='h32',
                            json_path=PARAM_SUMMARY_PATH):
    """Return layer_counts and mean Spearman ρ arrays for model_type.

    Reuses :func:`load_layer_sweep_rank_data`.
    """
    rank_sweep = load_layer_sweep_rank_data(json_path)
    gk = model_type  # e.g. 'h32'
    if gk not in rank_sweep:
        raise KeyError(f"model_type '{gk}' not found in rank sweep data. "
                       f"Available: {list(rank_sweep.keys())}")
    return rank_sweep[gk]['layers'], rank_sweep[gk]['mean_rho']


def aryl_fluoride_spread_vs_layers_plot(ax, layer_counts, mol_pred,
                                        targets, exp_cebes,
                                        rho_layers, mean_rho,
                                        LINE_WIDTH, STATS_FONT,
                                        AXIS_FONT, TICK_FONT,
                                        subplot=True, panel_label='(a)'):
    """Per-molecule predicted C–F CEBE vs layers, with Spearman ρ on right.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    layer_counts : list of int
    mol_pred : dict  {mol_name: list of float}
    targets : list of (mol_name, atom_idx, exp_cebe)
    exp_cebes : np.ndarray
    rho_layers : array-like   (layer counts matching mean_rho)
    mean_rho : array-like     (mean Spearman ρ per layer)
    """
    colors = plt.cm.Set2(np.linspace(0, 1, len(targets)))
    markers_list = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

    for i, (mol, aidx, exp_c) in enumerate(targets):
        label = mol.replace('-', '-')  # keep hyphens intact
        preds = mol_pred[mol]
        ax.plot(layer_counts, preds,
                '-', marker=markers_list[i % len(markers_list)],
                color=colors[i], markersize=7, linewidth=LINE_WIDTH,
                label=label, zorder=3)
        # Horizontal dotted line for the experimental CEBE
        ax.axhline(exp_c, ls=':', color=colors[i], lw=2.0, alpha=0.4,
                   zorder=1)

    ax.set_xlabel('Number of GNN Layers', fontsize=AXIS_FONT,
                  fontweight='bold')
    ax.set_ylabel('Predicted C–F CEBE (eV)', fontsize=AXIS_FONT,
                  fontweight='bold')
    ax.set_xticks(layer_counts)
    ax.tick_params(labelsize=TICK_FONT)
    ax.grid(True, alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    # Legend for molecule lines (left side)
    leg = ax.legend(fontsize=STATS_FONT - 1, loc='upper left',
                    framealpha=0.85, title='Molecule (dotted = exp.)',
                    title_fontsize=STATS_FONT - 1)
    leg.set_zorder(10)

    # Right y-axis: mean Spearman ρ
    c_rho = '#555555'
    ax2 = ax.twinx()
    l_rho, = ax2.plot(rho_layers, mean_rho, '--s', color=c_rho,
                      markersize=6, linewidth=LINE_WIDTH,
                      label='Mean Spearman ρ', zorder=4)
    ax2.set_ylabel('Mean Spearman ρ', fontsize=AXIS_FONT,
                   fontweight='bold', color=c_rho)
    ax2.tick_params(axis='y', labelcolor=c_rho, labelsize=TICK_FONT)
    ax2.set_ylim(0, 1.05)

    # Small legend for the ρ line (right side)
    leg2 = ax2.legend(handles=[l_rho], fontsize=STATS_FONT - 1,
                      loc='lower right', framealpha=0.85)
    leg2.set_zorder(10)

    if subplot:
        ax.text(-0.12, 1.05, panel_label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')

    return ax, ax2


def aryl_fluoride_spread_vs_layers(model_type='h32'):
    """Single-panel: per-molecule predicted C–F CEBE + Spearman ρ vs layers."""
    layer_counts, mol_pred, targets, exp_cebes = \
        _load_cf_predicted_cebes(model_type)
    rho_layers, mean_rho = _load_mean_spearman_rho(model_type)

    # Print table
    print(f"\n  Aryl-fluoride C–F group ({model_type}) — predicted CEBE per molecule")
    print(f"  {'Layers':>6s}", end='')
    for mol, _, _ in targets:
        print(f"  {mol:>22s}", end='')
    print(f"  {'Mean ρ':>8s}")
    print("  " + "-" * (6 + 22 * len(targets) + 10))
    for j, n in enumerate(layer_counts):
        print(f"  {n:6d}", end='')
        for mol, _, _ in targets:
            print(f"  {mol_pred[mol][j]:22.3f}", end='')
        rho_idx = np.where(rho_layers == n)[0]
        rho_val = mean_rho[rho_idx[0]] if len(rho_idx) else np.nan
        print(f"  {rho_val:8.3f}")

    AXIS_FONT  = 10
    TICK_FONT  = 10
    STATS_FONT = 9
    LINE_WIDTH = 1.5

    fig, ax = plt.subplots(figsize=(7, 5))
    aryl_fluoride_spread_vs_layers_plot(
        ax, layer_counts, mol_pred, targets, exp_cebes,
        rho_layers, mean_rho,
        LINE_WIDTH, STATS_FONT, AXIS_FONT, TICK_FONT, subplot=False,
    )

    fig.tight_layout()
    png_path = f'pngs/aryl_fluoride_spread_vs_layers_{model_type}.png'
    pdf_path = f'pngs/aryl_fluoride_spread_vs_layers_{model_type}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  ✓ {png_path}")
    print(f"  ✓ {pdf_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    #PARAM_SUMMARY_PATH = 'param_results/search_layer_type2_n_layers8_param_summary.json'
    scatter_and_rank(model_type='EQ')
    scatter_and_rank(model_type='IN')
    scatter(model_type='EQ')
    scatter(model_type='IN')
    layer_sweep(model_type='h32')
    layer_sweep(model_type='h16')
    layer_sweep_rank(model_type='h32')
    layer_sweep_rank(model_type='h16')
    locality()
    layer_sweep_and_locality(model_type='h32')
    aryl_fluoride_spread_mae(model_type='h32')
    aryl_fluoride_spread_vs_layers(model_type='h32')