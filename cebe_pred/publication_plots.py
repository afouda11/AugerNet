import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import r2_score

os.makedirs('pngs', exist_ok=True)


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

# ─────────────────────────────────────────────────────────────────────────────
#  Layer-scan data loader + reusable axes-level plot
# ─────────────────────────────────────────────────────────────────────────────


def _grouping_key(run):
    """Return the scan grouping key for a param-search run.

    If the run dict contains ``'layer_type'`` (layer-type scan), use
    that (e.g. ``'EQ'``).  Otherwise fall back to
    ``'h{hidden_channels}'`` (hidden-dim scan, e.g. ``'h32'``).
    """
    if 'layer_type' in run:
        return run['layer_type']
    return f"h{run['hidden_channels']}"


def load_layer_scan_data(param_json):
    """Load layer-scan param-search JSON and return per-group arrays.

    The grouping key is auto-detected: ``layer_type`` when present,
    otherwise ``h{hidden_channels}``.

    Returns
    -------
    scan : dict
        ``{group_key: {'layers': np.array, 'train_loss': …,
        'val_loss': …, 'mae': …}}`` sorted by n_layers.
    """
    with open(param_json) as f:
        summary = json.load(f)

    scan = {}
    for run in summary['runs']:
        gk = _grouping_key(run)
        scan.setdefault(gk, {'layers': [], 'train_loss': [], 'val_loss': [], 'mae': []})
        scan[gk]['layers'].append(run['n_layers'])
        scan[gk]['train_loss'].append(run['best_train_loss'])
        scan[gk]['val_loss'].append(run['best_val_loss'])
        scan[gk]['mae'].append(run['eval_mae'])

    for gk in scan:
        order = np.argsort(scan[gk]['layers'])
        for k in ('layers', 'train_loss', 'val_loss', 'mae'):
            scan[gk][k] = np.array(scan[gk][k])[order]

    return scan


def layer_scan_plot(ax, scan, model_type, LINE_WIDTH, STATS_FONT,
                     AXIS_FONT, TICK_FONT, subplot=True, panel_label='(b)',
                     log_scale=False):
    """Draw a dual-axis layer-scan on *ax* for a single model type.

    Left y-axis: train loss + val loss.  Right y-axis: eval MAE.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes (left y-axis = loss).
    scan : dict
        Output of :func:`load_layer_scan_data`.
    model_type : str
        Grouping key — ``'EQ'``, ``'IN'`` (layer-type scan) or
        ``'h16'``, ``'h32'`` (hidden-dim scan).
        If True, use log₁₀ scale on the left (loss) axis.  Makes the
        growing train/val gap (overfitting) much more visible.
    """
    d = scan[model_type]
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
    ax.set_ylabel('Loss', fontsize=AXIS_FONT, fontweight='bold')
    #ax.set_ylabel('Loss (MSE)', fontsize=AXIS_FONT, fontweight='bold')
    #ax.set_ylabel('Best Loss', fontsize=AXIS_FONT, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_FONT)
    ax.set_xticks(d['layers'])
    ax.grid(True, alpha=0.1, linewidth=1.0, zorder=0)
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
    if panel_label == '(a)':
        legend = ax.legend(h1 + h2, l1 + l2, fontsize=STATS_FONT+2, loc='lower left',
                           framealpha=0.85)
        legend.set_zorder(15)

    if subplot:
        ax.text(-0.12, 1.05, panel_label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')

    return ax, ax2


# ─────────────────────────────────────────────────────────────────────────────
#  One-panel figure: layer scan
# ─────────────────────────────────────────────────────────────────────────────

def layer_scan(param_json, model_type='EQ', name=None):
    """Single-panel dual-axis layer scan (train/val loss + MAE vs n_layers)."""
    scan = load_layer_scan_data(param_json)

    AXIS_FONT  = 10
    TICK_FONT  = 10
    STATS_FONT = 9
    LINE_WIDTH = 1.6

    fig, ax = plt.subplots(figsize=(6, 4))
    layer_scan_plot(ax, scan, model_type, LINE_WIDTH, STATS_FONT,
                     AXIS_FONT, TICK_FONT, subplot=False, log_scale=True)

    fig.tight_layout()
    if name is not None:
        png_path = f'pngs/layer_scan_{model_type}_{name}.png'
        pdf_path = f'pngs/layer_scan_{model_type}_{name}.pdf'
    else:
        png_path = f'pngs/layer_scan_{model_type}.png'
        pdf_path = f'pngs/layer_scan_{model_type}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  ✓ {png_path}")
    print(f"  ✓ {pdf_path}")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
#  Two-panel figure: layer scan
# ─────────────────────────────────────────────────────────────────────────────

def layer_scan_2_panel(param_json_1, param_json_2, model_type='h32', name=None):
    """Combined two-panel figure (1-col, 2-row): layer scan from two JSONs."""

    scan_1 = load_layer_scan_data(param_json_1)
    scan_2 = load_layer_scan_data(param_json_2)

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
    STATS_FONT = 12
    LINE_WIDTH = 2.2

    fig = plt.figure(figsize=(8, 10), constrained_layout=True)
    gs  = GridSpec(2, 1, figure=fig, hspace=0.05)

    # Panel (a) — top row
    ax_top = fig.add_subplot(gs[0, 0])
    layer_scan_plot(ax_top, scan_1, model_type, LINE_WIDTH, STATS_FONT,
                    AXIS_FONT, TICK_FONT, subplot=True, panel_label='(a)',
                    log_scale=True)

    # Add custom text box to bottom panel
    ax_top.text(
        0.95, 0.86,
        r"Skipatom-200 + At-BE + E-neg.",
        #r"Skipatom-200 + At-BE + E-neg." + "\n"
        #r"$d$ = 32",
        ha='right', va='bottom', transform=ax_top.transAxes,
        fontsize=STATS_FONT+5, fontweight='bold',
#         bbox=dict(boxstyle='round', edgecolor='grey',
#                   facecolor='white', alpha=0.85, pad=0.5),
        zorder=15,
    )

    # Hide x-axis title and tick labels on the top panel
    ax_top.set_xlabel('')
    ax_top.tick_params(axis='x', labelbottom=False)

    # Panel (b) — bottom row
    ax_bot = fig.add_subplot(gs[1, 0])
    layer_scan_plot(ax_bot, scan_2, model_type, LINE_WIDTH, STATS_FONT,
                    AXIS_FONT, TICK_FONT, subplot=True, panel_label='(b)',
                    log_scale=True)

    ax_bot.text(
        0.95, 0.85,
        r"Skipatom-200",
        #r"Skipatom-200" + "\n"
        #r"$d$ = 32",
        ha='right', va='bottom', transform=ax_bot.transAxes,
        fontsize=STATS_FONT+5, fontweight='bold',
#         bbox=dict(boxstyle='round', edgecolor='grey',
#                   facecolor='white', alpha=0.85, pad=0.5),
        zorder=15,
    )
    # ── Save ─────────────────────────────────────────────────────────────
    tag = f'_{name}' if name else ''
    png_path = f'pngs/2_panel_layer_scan_{model_type}{tag}.png'
    pdf_path = f'pngs/2_panel_layer_scan_{model_type}{tag}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  ✓ {png_path}")
    print(f"  ✓ {pdf_path}")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
#  scatter function
# ─────────────────────────────────────────────────────────────────────────────

def scatter_plot(ax, all_pred, all_exp, rstats, scatter_s, 
                   LINE_WIDTH, STATS_FONT, AXIS_FONT, TICK_FONT, subplot=None):

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

    if subplot == None or subplot == '(a)':
        ax.set_ylabel('Experimental CEBE (eV)', fontsize=AXIS_FONT,
                  fontweight='bold')
    else:
        ax.set_ylabel('')

    ax.tick_params(axis='both', labelsize=TICK_FONT)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.grid(True, alpha=0.3, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    if subplot:
        ax.text(-0.12, 1.05, subplot, transform=ax.transAxes,
              fontsize=AXIS_FONT+2, fontweight='bold', va='top')
    return ax

# ─────────────────────────────────────────────────────────────────────────────
#  One-panel figure: scatter
# ─────────────────────────────────────────────────────────────────────────────

def scatter(model):

    results_path = f'train_results/outputs/{model}_results.txt'
    rdata, rstats = load_results_data_and_compute_stats(results_path)

    all_exp  = rdata[:, 0]
    all_pred = rdata[:, 1]

    AXIS_FONT  = 11
    TICK_FONT  = 12
    STATS_FONT = 11
    LINE_WIDTH = 1.6
    scatter_s  = 20

    fig, ax = plt.subplots(figsize=(5, 4))

    ax = scatter_plot(ax, all_pred, all_exp, rstats, scatter_s,  
                    LINE_WIDTH, STATS_FONT, AXIS_FONT, TICK_FONT)

    # ── Save ─────────────────────────────────────────────────────────────
    png_path = f'pngs/{model}_scatter.png'
    pdf_path = f'pngs/{model}_scatter.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  ✓ {png_path}")
    print(f"  ✓ {pdf_path}")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
#  Two-panel figure: scatter
# ─────────────────────────────────────────────────────────────────────────────

def scatter_2_panel(model_1, model_2, name=None):
    """Combined two-panel figure: predicted-vs-experimental scatter (left)
    and per-molecule Spearman rank-correlation histogram (right)."""

    results_path_1 = f'train_results/outputs/{model_1}_results.txt'
    results_path_2 = f'train_results/outputs/{model_2}_results.txt'

    # ── Load data ────────────────────────────────────────────────────────
    rdata_1, rstats_1 = load_results_data_and_compute_stats(results_path_1)
    rdata_2, rstats_2 = load_results_data_and_compute_stats(results_path_2)

    all_exp_1  = rdata_1[:, 0]
    all_pred_1 = rdata_1[:, 1]
    all_exp_2  = rdata_2[:, 0]
    all_pred_2 = rdata_2[:, 1]

    print(f"  Scatter 1:  R²={rstats_1['r2']:.4f}  MAE={rstats_1['mae']:.4f} eV  "
          f"STD={rstats_1['std']:.4f} eV")
    print(f"  Scatter 2:  R²={rstats_2['r2']:.4f}  MAE={rstats_2['mae']:.4f} eV  "
          f"STD={rstats_2['std']:.4f} eV")

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
    ax_l = fig.add_subplot(gs[0, 0])

    ax_l = scatter_plot(ax_l, all_pred_1, all_exp_1, rstats_1, scatter_s, 
                           LINE_WIDTH, STATS_FONT, AXIS_FONT, TICK_FONT, subplot='(a)')

    ax_r = fig.add_subplot(gs[0, 1])

    ax_r = scatter_plot(ax_r, all_pred_2, all_exp_2, rstats_2, scatter_s,
                             LINE_WIDTH, STATS_FONT, AXIS_FONT, TICK_FONT, subplot='(b)')

    ax_l.text(
        0.95, 0.05,
        r"$\ell$ = 3   " + "\n"
        r"$d$ = 64",
        ha='right', va='bottom', transform=ax_l.transAxes,
        fontsize=STATS_FONT, fontweight='bold',
        bbox=dict(boxstyle='round', edgecolor='grey',
                facecolor='white', alpha=0.85, pad=0.5),
        zorder=5,
    )
    ax_r.text(
        0.95, 0.05,
#         f"$l$ = 4\n"
#         f"R$^{{32}}$\n$",
        r"$\ell$ = 4   " + "\n"
        r"$d$ = 32",
        ha='right', va='bottom', transform=ax_r.transAxes,
        fontsize=STATS_FONT, fontweight='bold',
        bbox=dict(boxstyle='round', edgecolor='grey',
                  facecolor='white', alpha=0.85, pad=0.5),
        zorder=5,
    )
    # ── Save ─────────────────────────────────────────────────────────────
    if name is not None:
        png_path = f'pngs/2_panel_scatter_{name}.png'
        pdf_path = f'pngs/2_panel_scatter_{name}.pdf'
    else:
        png_path = f'pngs/2_panel_scatter.png'
        pdf_path = f'pngs/2_panel_scatter.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  ✓ {png_path}")
    print(f"  ✓ {pdf_path}")
    plt.close(fig)



# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    param_json_1 = 'param_results/search_layer_type2_n_layers8_param_summary.json'
    param_json_2 = 'param_results/search_hidden_channels2_n_layers8_param_summary.json'
    param_json_3 = 'param_results/search_hidden_channels2_n_layers8_cebe_0_random_EQ3_h64_param_summary.json'

    model_EQ_1 = f'cebe_035_random_EQ3_h64_fold1'
    model_IN_1 = f'cebe_035_random_IN3_h64_fold1'

    model_EQ_2 = f'cebe_035_random_EQ4_h32_fold1'
    model_IN_2 = f'cebe_035_random_IN4_h32_fold1'

#     scatter(model_EQ_1)
#     scatter(model_IN_1)
#     scatter(model_EQ_2)
#     scatter(model_IN_2)
    scatter_2_panel(model_EQ_1, model_EQ_2, name='EQ')
    scatter_2_panel(model_IN_1, model_IN_2, name='IN')
#     layer_scan(param_json_1, model_type='EQ',  name='035')
#     layer_scan(param_json_2, model_type='h32', name='035')
#     layer_scan(param_json_3, model_type='h64', name='0')
#     layer_scan(param_json_3, model_type='h32', name='0')
    layer_scan_2_panel(param_json_2, param_json_3, model_type='h32')