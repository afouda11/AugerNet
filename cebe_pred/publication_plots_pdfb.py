import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import r2_score

from publication_plots import layer_scan_plot, load_layer_scan_data 
from publication_plots_rank import load_layer_scan_rank_data

os.makedirs('locality_analysis', exist_ok=True)

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
                        s=35, alpha=0.7, edgecolors='white',
                        linewidths=0.4, zorder=2)
        if fig is not None:
            cax = ax.inset_axes([1.02, 0.0, 0.03, 0.85])
            cbar = fig.colorbar(sc, cax=cax)
            cbar.set_label('Carbons in group', fontsize=AXIS_FONT)
            cbar.ax.tick_params(labelsize=TICK_FONT - 1)

    # ── Mean intra-group std line ────────────────────────────────────────
#     ax.plot(radii, mean_std, '-o', color='grey', markersize=7,
#             linewidth=LINE_WIDTH, label='Mean Std', zorder=4)

    ax.set_xlabel('Topological (Bond) Radius', fontsize=AXIS_FONT,
                  fontweight='bold')
    ax.set_ylabel('Exp. Intra-Env. Std (eV) ', fontsize=AXIS_FONT,
                  fontweight='bold')
    ax.set_xticks(radii)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis='both', labelsize=TICK_FONT)
    #ax.legend(fontsize=STATS_FONT - 1, loc='upper right', framealpha=0.85)
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
                fontsize=20, fontweight='bold', color='orange',
                ha='center', va='bottom', zorder=5)

    # Text box explaining the star annotations
    if subplot:
        fontsize=14
    else:
        fontsize=10

    ax.text(3.8, 0.6, '* Para-di-substituted fluorobenzenes\n  C-F spread = 1.22 eV',
              fontsize=fontsize, color='orange', fontweight='bold',
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
    png_path = 'locality_analysis/locality_variance.png'
    pdf_path = 'locality_analysis/locality_variance.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  ✓ {png_path}")
    print(f"  ✓ {pdf_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Two-panel figure: (a) layer scan  (b) locality variance
# ─────────────────────────────────────────────────────────────────────────────

def layer_scan_and_locality(param_json, model_type='h32'):
    """Single-column two-row figure: layer scan (top) + locality (bottom)."""
    from locality_analysis import compute_locality_data

    scan    = load_layer_scan_data(param_json)
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

    # Panel (a): layer scan
    ax_a = fig.add_subplot(gs[0, 0])
    layer_scan_plot(ax_a, scan, model_type, LINE_WIDTH, STATS_FONT,
                     AXIS_FONT, TICK_FONT, subplot=True, panel_label='(a)',
                     log_scale=True)

    # Panel (b): locality variance
    ax_b = fig.add_subplot(gs[1, 0])
    locality_plot(ax_b, loc_data, LINE_WIDTH, STATS_FONT,
                  AXIS_FONT, TICK_FONT, fig=fig, subplot=True,
                  panel_label='(b)')

    # ── Save ─────────────────────────────────────────────────────────────
    tag = model_type
    png_path = f'locality_analysis/layer_scan_locality_{tag}.png'
    pdf_path = f'locality_analysis/layer_scan_locality_{tag}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  ✓ {png_path}")
    print(f"  ✓ {pdf_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Aryl-fluoride C–F spread + MAE vs GNN layers
# ─────────────────────────────────────────────────────────────────────────────

CF_GROUP_FILE = 'locality_analysis/radius_2/group_002.txt'


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


def _load_cf_spread_data(param_json, model_type='h32', param='feature_keys',
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

    with open(param_json) as f:
        summary = json.load(f)

    search_id = summary['search_id']
    output_dir = 'param_results/outputs'

    # Filter runs matching the requested hidden dim, sorted by n_layers
    if param == 'hidden_channels':
        model_type = int(model_type.replace('h', ''))
    runs = sorted(
            [r for r in summary['runs'] if r[param] == model_type],
            key=lambda r: r['n_layers'],
        )

    layer_counts = []
    pred_spreads = []
    pred_maes = []

    for run in runs:
        labels_path = os.path.join(
            output_dir,
            f"{search_id}_{run['model_id']}_fold5_{run['config_id']}_labels.txt",
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


def aryl_fluoride_spread_plot(ax, layer_counts, pred_spreads,
                              exp_spread, LINE_WIDTH, STATS_FONT,
                              AXIS_FONT, TICK_FONT,
                              subplot=True, panel_label='(b)',
                              color='#0072B2', marker='o',
                              label='Predicted spread',
                              show_exp_line=True,
                              draw_axes_labels=True,
                              pred_maes=None,
                              mae_color=None, mae_marker=None,
                              mae_label=None,
                              ax_mae=None):
    """Single-axis panel: C–F predicted spread vs layers (no MAE axis).

    Call once per series to overlay multiple models on the same axes.
    Set *show_exp_line* and *draw_axes_labels* to ``False`` for the
    second (and later) calls so decorations are drawn only once.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    color, marker, label : styling for this series.
    show_exp_line : bool – draw the horizontal experimental-spread line.
    draw_axes_labels : bool – set axis labels, ticks, grid.
    pred_maes : array-like or None
        If given, plot C–F MAE on a right-hand twin axis.
    mae_color, mae_marker, mae_label : styling for the MAE series.
    ax_mae : matplotlib.axes.Axes or None
        Pre-existing twin axis for MAE (reuse across calls).
        If *pred_maes* is given but *ax_mae* is None, a new twin
        axis is created and returned.
    """
    if show_exp_line:
        ax.axhline(exp_spread, ls=':', color='grey',
                   lw=3.0, alpha=0.6,
                   #label=f'Exp. spread = {exp_spread:.2f} eV',
                   label=f'Exp. spread (1.22 eV)',
                   zorder=1)
        
    l1, = ax.plot(layer_counts, pred_spreads, f'-{marker}', color=color,
                  markersize=7, linewidth=LINE_WIDTH,
                  label=label, zorder=3)

    if draw_axes_labels:
        #ax.set_ylabel('C–F CEBE Spread (eV)', fontsize=AXIS_FONT,
        ax.set_ylabel('C–F Spread (eV)', fontsize=AXIS_FONT,
                      fontweight='bold')
        ax.tick_params(axis='both', labelsize=TICK_FONT)
        ax.set_xlabel('Number of GNN Layers', fontsize=AXIS_FONT,
                      fontweight='bold')
        ax.set_xticks(layer_counts)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25, zorder=0)
        ax.set_axisbelow(True)

    # ── Optional MAE on right-hand twin axis ─────────────────────────────
    if pred_maes is not None:
        if ax_mae is None:
            ax_mae = ax.twinx()
        _mc = mae_color or '#D55E00'
        _mm = mae_marker or '^'
        _ml = mae_label or 'C–F MAE'
        ax_mae.plot(layer_counts, pred_maes, f'--{_mm}', color=_mc,
                    markersize=7, linewidth=LINE_WIDTH,
                    label=_ml, zorder=3)
        if draw_axes_labels:
            #ax_mae.set_ylabel('C–F CEBE MAE (eV)', fontsize=AXIS_FONT,
            ax_mae.set_ylabel('C–F MAE (eV)', fontsize=AXIS_FONT,
                              fontweight='bold', color=_mc)
            ax_mae.tick_params(axis='y', labelcolor=_mc, labelsize=TICK_FONT)
            ax_mae.set_ylim(bottom=0)

    # ── Combined legend (gather handles from ax + ax_mae) ────────────────
    handles, labels_ = ax.get_legend_handles_labels()
    if ax_mae is not None:
        h2, l2 = ax_mae.get_legend_handles_labels()
        handles += h2
        labels_ += l2

    legend = ax.legend(handles, labels_, fontsize=STATS_FONT,
                       loc='lower right', framealpha=0.85)
    legend.set_zorder(10)

    if subplot:
        ax.text(-0.12, 1.05, panel_label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')

    return ax, ax_mae


def aryl_fluoride_spread_mae(param_json, model_type='h32'):
    """Single-panel figure: C–F spread + MAE vs GNN layers."""
    layer_counts, spreads, maes, exp_spread = _load_cf_spread_data(param_json, model_type)

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
    png_path = f'locality_analysis/aryl_fluoride_spread_mae_{model_type}.png'
    pdf_path = f'locality_analysis/aryl_fluoride_spread_mae_{model_type}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  ✓ {png_path}")
    print(f"  ✓ {pdf_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Two-panel figure: (a) locality  (b) aryl-fluoride spread (single y-axis)
# ─────────────────────────────────────────────────────────────────────────────

def locality_and_spread(param_json, model_type='h64',
                        param_json_2=None, model_type_2=None,
                        name=None):
    """1-col 2-row figure: locality variance (top) + C–F spread (bottom).

    If *param_json_2* / *model_type_2* are given, a second spread
    series is overlaid on the bottom panel so the two models can be
    compared.
    """
    from locality_analysis import compute_locality_data

    loc_data = compute_locality_data()

    # First model
    lc1, sp1, mae1, exp_spread = _load_cf_spread_data(param_json, model_type)

    # Optional second model
    lc2, sp2, mae2 = None, None, None
    if param_json_2 is not None and model_type_2 is not None:
        lc2, sp2, mae2, _ = _load_cf_spread_data(param_json_2, model_type_2)

    # ── Style (matches layer_scan_2_panel) ───────────────────────────────
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

    # Panel (a) — locality variance
    ax_top = fig.add_subplot(gs[0, 0])
    locality_plot(ax_top, loc_data, LINE_WIDTH, STATS_FONT,
                  AXIS_FONT, TICK_FONT, fig=fig, subplot=True,
                  panel_label='(a)')

    # Hide x-axis title and tick labels on the top panel
    ax_top.set_xlabel('')
    ax_top.tick_params(axis='x', labelbottom=False)

    # Panel (b) — C–F spread (left axis) + MAE (right axis)
    ax_bot = fig.add_subplot(gs[1, 0])

    # First series (always drawn)
    _, ax_mae = aryl_fluoride_spread_plot(
        ax_bot, lc1, sp1, exp_spread,
        LINE_WIDTH, STATS_FONT, AXIS_FONT, TICK_FONT,
        subplot=True, panel_label='(b)',
        color='#0072B2', marker='o',
        label=f'Spread: Skipatom-200 + At-BE + E-neg.',
        show_exp_line=True, draw_axes_labels=True,
        pred_maes=mae1,
        mae_color='#0072B2', mae_marker='^',
        mae_label=f'MAE: Skipatom-200 + At-BE + E-neg.',
    )

    # Second series (optional)
    if lc2 is not None:
        _, ax_mae = aryl_fluoride_spread_plot(
            ax_bot, lc2, sp2, exp_spread,
            LINE_WIDTH, STATS_FONT, AXIS_FONT, TICK_FONT,
            subplot=False,          # panel label already drawn
            color='#D55E00', marker='s',
            label=f'Spread: Skipatom-200',
            show_exp_line=False,    # already drawn by first call
            draw_axes_labels=False, # already drawn by first call
            pred_maes=mae2,
            mae_color='#D55E00', mae_marker='v',
            mae_label=f'MAE: Skipatom-200',
            ax_mae=ax_mae,          # reuse same twin axis
        )

    # ── Save ─────────────────────────────────────────────────────────────
    if name:
        tag = f'_{name}'
    elif model_type_2:
        tag = f'_{model_type}_{model_type_2}'
    else:
        tag = f'_{model_type}'
    png_path = f'locality_analysis/locality_and_spread{tag}.png'
    pdf_path = f'locality_analysis/locality_and_spread{tag}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  ✓ {png_path}")
    print(f"  ✓ {pdf_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Two-panel figure: (a) C–F spread vs layers  (b) C–F MAE vs layers
# ─────────────────────────────────────────────────────────────────────────────

def _mae_plot(ax, layer_counts, pred_maes, LINE_WIDTH, STATS_FONT,
              AXIS_FONT, TICK_FONT,
              subplot=True, panel_label='(b)',
              color='#0072B2', marker='^',
              label='C–F MAE',
              draw_axes_labels=True):
    """Plot C–F MAE vs GNN layers on *ax*.

    Call once per model series.  Set *draw_axes_labels* to ``False``
    for the second call so axis decorations are drawn only once.
    """
    ax.plot(layer_counts, pred_maes, f'-{marker}', color=color,
            markersize=7, linewidth=LINE_WIDTH,
            label=label, zorder=3)

    if draw_axes_labels:
        ax.set_ylabel('C–F MAE (eV)', fontsize=AXIS_FONT,
                      fontweight='bold')
        ax.tick_params(axis='both', labelsize=TICK_FONT)
        ax.set_xlabel('Number of GNN Layers', fontsize=AXIS_FONT,
                      fontweight='bold')
        ax.set_xticks(layer_counts)
        ax.set_ylim(0.2, 0.7)
        ax.grid(True, alpha=0.25, zorder=0)
        ax.set_axisbelow(True)

    if panel_label == '(a)':
        legend = ax.legend(fontsize=STATS_FONT, loc='upper right',
                           framealpha=0.85)
        legend.set_zorder(10)

    if subplot:
        ax.text(-0.12, 1.05, panel_label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')

    return ax


def spread_and_mae(param_json, model_type='h64',
                   param_json_2=None, model_type_2=None,
                   param=None,
                   name=None):
    """1-col 2-row figure: C–F spread (top) + C–F MAE (bottom).

    If *param_json_2* / *model_type_2* are given, a second series
    is overlaid on both panels for comparison.
    """
    # First model
    lc1, sp1, mae1, exp_spread = _load_cf_spread_data(param_json, model_type, param)

    # Optional second model
    lc2, sp2, mae2 = None, None, None
    if param_json_2 is not None and model_type_2 is not None:
        lc2, sp2, mae2, _ = _load_cf_spread_data(param_json_2, model_type_2, param)

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

    # Panel (a) — C–F spread vs layers
    ax_top = fig.add_subplot(gs[0, 0])
    aryl_fluoride_spread_plot(
        ax_top, lc1, sp1, exp_spread,
        LINE_WIDTH, STATS_FONT, AXIS_FONT, TICK_FONT,
        subplot=True, panel_label='(a)',
        color='#0072B2', marker='o',
        label='Skipatom-200 + At-BE + E-neg.',
        show_exp_line=True, draw_axes_labels=True,
    )
    if lc2 is not None:
        aryl_fluoride_spread_plot(
            ax_top, lc2, sp2, exp_spread,
            LINE_WIDTH, STATS_FONT, AXIS_FONT, TICK_FONT,
            subplot=False,
            color='#D55E00', marker='s',
            label='Skipatom-200',
            show_exp_line=False, draw_axes_labels=False,
        )

    # Hide x-axis label on top panel (shared x-axis)
    ax_top.set_xlabel('')
    ax_top.tick_params(axis='x', labelbottom=False)

    # Panel (b) — C–F MAE vs layers
    ax_bot = fig.add_subplot(gs[1, 0])
    _mae_plot(
        ax_bot, lc1, mae1,
        LINE_WIDTH, STATS_FONT, AXIS_FONT, TICK_FONT,
        subplot=True, panel_label='(c)',
        color='#0072B2', marker='^',
        label='Skipatom-200 + At-BE + E-neg.',
        draw_axes_labels=True,
    )
    if lc2 is not None:
        _mae_plot(
            ax_bot, lc2, mae2,
            LINE_WIDTH, STATS_FONT, AXIS_FONT, TICK_FONT,
            subplot=False,
            color='#D55E00', marker='v',
            label='Skipatom-200',
            draw_axes_labels=False,
        )

    # ── Save ─────────────────────────────────────────────────────────────
    if name:
        tag = f'_{name}'
    elif model_type_2:
        tag = f'_{model_type}_{model_type_2}'
    else:
        tag = f'_{model_type}'
    png_path = f'locality_analysis/spread_and_mae{tag}.png'
    pdf_path = f'locality_analysis/spread_and_mae{tag}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  ✓ {png_path}")
    print(f"  ✓ {pdf_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Aryl-fluoride C–F : predicted CEBE per molecule vs layers + Spearman ρ
# ─────────────────────────────────────────────────────────────────────────────

def _load_cf_predicted_cebes(param_json, model_type='h32',
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

    with open(param_json) as f:
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


def _load_mean_spearman_rho(param_json, model_type='h32'):
    """Return layer_counts and mean Spearman ρ arrays for model_type.

    Reuses :func:`load_layer_scan_rank_data`.
    """
    rank_scan = load_layer_scan_rank_data(param_json)
    gk = model_type  # e.g. 'h32'
    if gk not in rank_scan:
        raise KeyError(f"model_type '{gk}' not found in rank scan data. "
                       f"Available: {list(rank_scan.keys())}")
    return rank_scan[gk]['layers'], rank_scan[gk]['mean_rho']


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


def aryl_fluoride_spread_vs_layers(param_json, model_type='h32'):
    """Single-panel: per-molecule predicted C–F CEBE + Spearman ρ vs layers."""
    layer_counts, mol_pred, targets, exp_cebes = \
        _load_cf_predicted_cebes(param_json, model_type)
    rho_layers, mean_rho = _load_mean_spearman_rho(param_json, model_type)

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
    png_path = f'locality_analysis/aryl_fluoride_spread_vs_layers_{model_type}.png'
    pdf_path = f'locality_analysis/aryl_fluoride_spread_vs_layers_{model_type}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  ✓ {png_path}")
    print(f"  ✓ {pdf_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

#     param_json_1 = 'param_results/search_layer_type2_n_layers8_param_summary.json'
#     param_json_2 = 'param_results/search_hidden_channels2_n_layers8_param_summary.json'
#     param_json_3 = 'param_results/search_hidden_channels2_n_layers8_cebe_0_random_EQ3_h64_param_summary.json'

#     model_EQ_1 = f'cebe_035_random_EQ3_h64_fold1'
#     model_IN_1 = f'cebe_035_random_IN3_h64_fold1'
# 
#     model_EQ_2 = f'cebe_035_random_EQ4_h32_fold1'
#     model_IN_2 = f'cebe_035_random_IN4_h32_fold1'

    param_json = 'param_results/search_feature_keys2_n_layers8_cebe_035_butina_EQ3_h64_param_summary.json'

    locality()
#     layer_scan_and_locality(param_json_3, model_type='h64')
#     locality_and_spread(param_json_3, model_type='h32')
#     aryl_fluoride_spread_mae(param_json_3, model_type='h64')
#     aryl_fluoride_spread_vs_layers(param_json_3, model_type='h64')
#     locality_and_spread(param_json, model_type='035',
#                         param_json_2=param_json, model_type_2='0')
    spread_and_mae(param_json, model_type='035',
                   param_json_2=param_json, model_type_2='0', param="feature_keys")