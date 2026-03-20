import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.colorbar import ColorbarBase
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


def load_mol_sizes_per_carbon(labels_path):
    """Load a labels file and return per-carbon molecule size (N_atoms).

    Returns an array of length N_carbons (matching the rows in the
    corresponding ``_results.txt``), where each entry is the total
    number of atoms in the molecule that carbon belongs to.
    """
    sizes = []
    n_atoms = 0
    n_carbons_in_mol = 0
    in_mol = False

    with open(labels_path) as f:
        for line in f:
            line = line.rstrip('\n')
            # Molecule header
            if re.match(r'^# --- .+ ---$', line):
                # Flush previous molecule
                if in_mol and n_carbons_in_mol > 0:
                    sizes.extend([n_atoms] * n_carbons_in_mol)
                n_atoms = 0
                n_carbons_in_mol = 0
                in_mol = True
                continue
            # Skip comment / blank lines
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            # Count every atom line
            n_atoms += 1
            # Count carbons with valid data (not sentinel)
            if parts[0] == 'C' and parts[1] != '—':
                n_carbons_in_mol += 1

    # Flush last molecule
    if in_mol and n_carbons_in_mol > 0:
        sizes.extend([n_atoms] * n_carbons_in_mol)

    return np.array(sizes)


# ─────────────────────────────────────────────────────────────────────────────
#  Layer-scan data loader + reusable axes-level plot
# ─────────────────────────────────────────────────────────────────────────────

def _compute_val_eval_mae(labels_path, val_set, eval_set):
    """Compute MAE separately for val and eval molecules from a labels file.

    Parameters
    ----------
    labels_path : str
        Path to a ``*_labels.txt`` file.
    val_set : set
        Molecule names in the validation split.
    eval_set : set
        Molecule names in the evaluation split.

    Returns
    -------
    val_mae, eval_mae : float
        Mean absolute errors for the two splits.
    """
    molecules = load_label_data(labels_path)

    val_errors, eval_errors = [], []
    for mol in molecules:
        errors = np.abs(mol['pred'] - mol['true'])
        if mol['name'] in val_set:
            val_errors.extend(errors.tolist())
        elif mol['name'] in eval_set:
            eval_errors.extend(errors.tolist())

    val_mae  = np.mean(val_errors)  if val_errors  else np.nan
    eval_mae = np.mean(eval_errors) if eval_errors else np.nan
    return val_mae, eval_mae


def load_layer_scan_data(param_json, param):
    """Load layer-scan param-search JSON and return per-group arrays.

    The grouping key is auto-detected: ``layer_type`` when present,
    otherwise ``h{hidden_channels}``.

    Returns
    -------
    scan : dict
        ``{group_key: {'layers': np.array, 'train_loss': …,
        'val_loss': …, 'mae': …, 'val_mae': …, 'eval_mae': …}}``
        sorted by n_layers.
    """
    import glob as _glob

    with open(param_json) as f:
        summary = json.load(f)

    search_id = summary['search_id']

    # Load val / eval molecule name sets
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(os.path.dirname(script_dir), 'data', 'raw', 'exp_cebe')
    val_set  = set(open(os.path.join(data_root, 'mol_list_val.txt')).read().split())
    eval_set = set(open(os.path.join(data_root, 'mol_list_eval.txt')).read().split())

    scan = {}
    for run in summary['runs']:
        gk = run[param]
        scan.setdefault(gk, {'layers': [], 'train_loss': [], 'val_loss': [],
                              'mae': [], 'val_mae': [], 'eval_mae': []})
        scan[gk]['layers'].append(run['n_layers'])
        scan[gk]['train_loss'].append(run['best_train_loss'])
        scan[gk]['val_loss'].append(run['best_val_loss'])
        scan[gk]['mae'].append(run['eval_mae'])

        # Locate the labels file for this run (auto-detect fold number)
        outputs_dir = os.path.join(os.path.dirname(param_json), 'outputs')
        pattern = os.path.join(
            outputs_dir,
            f"{search_id}_{run['model_id']}_*_{run['config_id']}_labels.txt",
        )
        matches = _glob.glob(pattern)
        if matches:
            v_mae, e_mae = _compute_val_eval_mae(matches[0], val_set, eval_set)
        else:
            v_mae, e_mae = np.nan, np.nan
        scan[gk]['val_mae'].append(v_mae)
        scan[gk]['eval_mae'].append(e_mae)

    for gk in scan:
        order = np.argsort(scan[gk]['layers'])
        for k in ('layers', 'train_loss', 'val_loss', 'mae', 'val_mae', 'eval_mae'):
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
    color_train    = '#0072B2'   # blue
    color_val      = '#E69F00'   # orange
    color_mae_val  = '#4daf4a'   # green  — exp val MAE
    color_mae_eval = '#e41a1c'   # red    — exp eval MAE

    # Left axis — train & val loss
    ax.plot(d['layers'], d['train_loss'], 'o-', color=color_train,
            linewidth=LINE_WIDTH, markersize=5, label='Train loss', zorder=3)
    ax.plot(d['layers'], d['val_loss'], 's-', color=color_val,
            linewidth=LINE_WIDTH, markersize=5, label='Val loss', zorder=3)

    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel('Number of GNN Layers', fontsize=AXIS_FONT, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=AXIS_FONT, fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_FONT)
    ax.set_xticks(d['layers'])
    ax.grid(True, alpha=0.1, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)

    # Right axis — val MAE + eval MAE (split)
    ax2 = ax.twinx()
    ax2.plot(d['layers'], d['val_mae'], '^--', color=color_mae_val,
             linewidth=LINE_WIDTH, markersize=5, label='Exp. Val MAE', zorder=3)
    ax2.plot(d['layers'], d['eval_mae'], 'v--', color=color_mae_eval,
             linewidth=LINE_WIDTH, markersize=5, label='Exp. Eval MAE', zorder=3)

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

def layer_scan_2_panel(param_json_1, param_json_2, param="feature_tag", model_1=None, model_2=None, name=None):
    """Combined two-panel figure (1-col, 2-row): layer scan from two JSONs."""

    scan_1 = load_layer_scan_data(param_json_1, param)
    scan_2 = load_layer_scan_data(param_json_2, param)

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
    layer_scan_plot(ax_top, scan_1, model_1, LINE_WIDTH, STATS_FONT,
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
    layer_scan_plot(ax_bot, scan_2, model_2, LINE_WIDTH, STATS_FONT,
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
    png_path = f'pngs/2_panel_layer_scan{tag}.png'
    pdf_path = f'pngs/2_panel_layer_scan{tag}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  ✓ {png_path}")
    print(f"  ✓ {pdf_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Combined two-panel: (a) Exp MAE val/eval  (b) C–F MAE + C–F spread
# ─────────────────────────────────────────────────────────────────────────────

def mae_and_cf_2_panel(param_json, param='feature_keys',
                       model_1='035', model_2='0',
                       label_1='Skip-200 + At-BE + E-neg.',
                       label_2='Skip-200',
                       name=None):
    """Combined 3-row figure comparing two feature sets across GNN depth.

    Panel (a): Exp. Val MAE and Exp. Eval MAE for both feature sets.
    Panel (b): C–F MAE for both feature sets.
    Panel (c): C–F spread for both feature sets.

    Parameters
    ----------
    param_json : str
        Path to param-search summary JSON.
    param : str
        Grouping key in the JSON (e.g. ``'feature_keys'``).
    model_1, model_2 : str
        Group keys for the two feature sets.
    label_1, label_2 : str
        Human-readable names for the legend.
    name : str or None
        Suffix for the output filenames.
    """
    from publication_plots_pdfb import _load_cf_spread_data

    # ── Load data ────────────────────────────────────────────────────────
    scan = load_layer_scan_data(param_json, param)

    d1 = scan[model_1]
    d2 = scan[model_2]

    lc1, sp1, cf_mae1, exp_spread = _load_cf_spread_data(
        param_json, model_type=model_1, param=param)
    lc2, sp2, cf_mae2, _ = _load_cf_spread_data(
        param_json, model_type=model_2, param=param)

    # ── Style ────────────────────────────────────────────────────────────
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 10, 'axes.linewidth': 1.2,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
        'xtick.major.size': 5,    'ytick.major.size': 5,
        'legend.frameon': True,
    })
    AXIS_FONT  = 24
    TICK_FONT  = 24
    STATS_FONT = 22
    LABEL_FONT = 22
    LINE_WIDTH = 2.2

    # Colours: model 1 = blue family, model 2 = orange family
    c1_val  = '#0072B2'   # blue  — model 1 val
    c1_eval = '#56B4E9'   # light blue — model 1 eval
    c2_val  = '#D55E00'   # red-orange — model 2 val
    c2_eval = '#E69F00'   # amber — model 2 eval

    fig = plt.figure(figsize=(8, 18), constrained_layout=True)
    gs  = GridSpec(3, 1, figure=fig, hspace=0.025)

    # ── Panel (a): Exp. Val MAE + Eval MAE ───────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])

    ax_a.plot(d1['layers'], d1['val_mae'], 'o-', color=c1_val,
              linewidth=LINE_WIDTH, markersize=6,
              label=f'Val.:   {label_1}', zorder=3)
    ax_a.plot(d1['layers'], d1['eval_mae'], 'o--', color=c1_val,
              linewidth=LINE_WIDTH, markersize=6,
              label=f'Eval.: {label_1}', zorder=3)
    ax_a.plot(d2['layers'], d2['val_mae'], 's-', color=c2_val,
              linewidth=LINE_WIDTH, markersize=6,
              label=f'Val.:   {label_2}', zorder=3)
    ax_a.plot(d2['layers'], d2['eval_mae'], 's--', color=c2_val,
              linewidth=LINE_WIDTH, markersize=6,
              label=f'Eval.: {label_2}', zorder=3)

    ax_a.set_ylabel('Exp. MAE (eV)', fontsize=AXIS_FONT, fontweight='bold')
    ax_a.tick_params(axis='both', labelsize=TICK_FONT)
    ax_a.set_xticks(d1['layers'])
    ax_a.grid(True, alpha=0.25, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.legend(fontsize=STATS_FONT, loc='upper right', framealpha=0.85)

    # Hide x-axis label (shared x)
    ax_a.set_xlabel('')
    ax_a.tick_params(axis='x', labelbottom='')

    ax_a.text(-0.10, 1.05, '(a)', transform=ax_a.transAxes,
              fontsize=LABEL_FONT, fontweight='bold', va='top')

    # ── Panel (b): C–F MAE ───────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[1, 0])

    ax_b.plot(lc1, cf_mae1, 'o-', color=c1_val,
              linewidth=LINE_WIDTH, markersize=6,
              label=label_1, zorder=3)
    ax_b.plot(lc2, cf_mae2, 's-', color=c2_val,
              linewidth=LINE_WIDTH, markersize=6,
              label=label_2, zorder=3)

    ax_b.set_ylabel('C–F MAE (eV)', fontsize=AXIS_FONT, fontweight='bold')
    ax_b.tick_params(axis='both', labelsize=TICK_FONT)
    ax_b.set_xticks(d1['layers'])
    ax_b.grid(True, alpha=0.25, zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.legend(fontsize=STATS_FONT, loc='lower right', framealpha=0.85)

    # Hide x-axis label (shared x)
    ax_b.set_xlabel('')
    ax_b.tick_params(axis='x', labelbottom=False)

    ax_b.text(-0.10, 1.05, '(b)', transform=ax_b.transAxes,
              fontsize=LABEL_FONT, fontweight='bold', va='top')

    # ── Panel (c): C–F Spread ────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[2, 0])

    ax_c.plot(lc1, sp1, 'o-', color=c1_val,
              linewidth=LINE_WIDTH, markersize=6,
              zorder=3)
              #label=label_1, zorder=3)
    ax_c.plot(lc2, sp2, 's-', color=c2_val,
              linewidth=LINE_WIDTH, markersize=6,
              zorder=3)
              #label=label_2, zorder=3)
    ax_c.axhline(exp_spread, ls=':', color='grey', lw=2.5, alpha=0.6,
                 label=f'Exp. spread ({exp_spread:.2f} eV)', zorder=1)

    ax_c.set_xlabel(r'Number of EGNN Layers ($l$)', fontsize=AXIS_FONT,
                    fontweight='bold')
    ax_c.set_ylabel('C–F Spread (eV)', fontsize=AXIS_FONT, fontweight='bold')
    ax_c.tick_params(axis='both', labelsize=TICK_FONT)
    ax_c.set_xticks(d1['layers'])
    ax_c.set_ylim(bottom=0)
    ax_c.grid(True, alpha=0.25, zorder=0)
    ax_c.set_axisbelow(True)
    ax_c.legend(fontsize=STATS_FONT, loc='lower right', framealpha=0.85)

    ax_c.text(-0.10, 1.05, '(c)', transform=ax_c.transAxes,
              fontsize=LABEL_FONT, fontweight='bold', va='top')

    # ── Save ─────────────────────────────────────────────────────────────
    tag = f'_{name}' if name else ''
    png_path = f'pngs/3_panel_mae_cf{tag}.png'
    pdf_path = f'pngs/3_panel_mae_cf{tag}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  ✓ {png_path}")
    print(f"  ✓ {pdf_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  scatter function
# ─────────────────────────────────────────────────────────────────────────────

def scatter_plot(ax, all_pred, all_exp, rstats, scatter_s, 
                   LINE_WIDTH, STATS_FONT, AXIS_FONT, TICK_FONT,
                   subplot=None, mol_sizes=None, vmin=None, vmax=None,
                   cmap='YlOrRd', size_threshold=16,
                   below_color='#0072B2'):
    """Axes-level scatter of predicted vs experimental CEBE.

    If *mol_sizes* is provided (array same length as all_pred), points
    for molecules with size <= *size_threshold* are drawn in a fixed
    *below_color*, while molecules above the threshold are coloured by
    molecule size using *cmap*.  Returns ``(ax, sc)`` where *sc* is the
    colormapped scatter ``PathCollection`` (needed for colorbars).
    """

    if mol_sizes is not None:
        mask_below = mol_sizes <= size_threshold
        mask_above = ~mask_below

        # Plot molecules at or below threshold in fixed colour
        ax.scatter(all_pred[mask_below], all_exp[mask_below],
                   alpha=0.6, s=scatter_s,
                   color=below_color, edgecolors=None,
                   linewidth=0.2, zorder=3,
                   label=f'≤{size_threshold} atoms')

        # Plot molecules above threshold with colour gradient
        cmap_vmin = vmin if vmin is not None else mol_sizes[mask_above].min()
        cmap_vmax = vmax if vmax is not None else mol_sizes[mask_above].max()
        norm = mcolors.Normalize(vmin=cmap_vmin, vmax=cmap_vmax)
        sc = ax.scatter(all_pred[mask_above], all_exp[mask_above],
                        alpha=0.8, s=scatter_s,
                        c=mol_sizes[mask_above], cmap=cmap, norm=norm,
                        edgecolors='black', linewidth=0.4, zorder=4)
    else:
        sc = ax.scatter(all_pred, all_exp, alpha=0.4, s=scatter_s,
                        edgecolors=None, linewidth=0.2, color=below_color,
                        zorder=3)

    lo = min(all_exp.min(), all_pred.min())
    hi = max(all_exp.max(), all_pred.max())
    pad = (hi - lo) * 0.03
    diag = np.array([lo - pad, hi + pad])
    ax.fill_between(diag, diag - 1.0, diag + 1.0,
                    color='#0072B2', alpha=0.08, zorder=1,
                    label='±1 eV')
    ax.plot(diag, diag,
              'k:', linewidth=LINE_WIDTH, alpha=0.7, zorder=2)

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
    return ax, sc

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

    ax, _ = scatter_plot(ax, all_pred, all_exp, rstats, scatter_s,  
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
    """Two-panel exp-eval scatter: EQ (left) vs IN (right),
    coloured by molecule size (N_atoms)."""

    results_path_1 = f'train_results/outputs/{model_1}_results.txt'
    results_path_2 = f'train_results/outputs/{model_2}_results.txt'
    labels_path_1  = f'train_results/outputs/{model_1}_labels.txt'
    labels_path_2  = f'train_results/outputs/{model_2}_labels.txt'

    # ── Load data ────────────────────────────────────────────────────────
    rdata_1, rstats_1 = load_results_data_and_compute_stats(results_path_1)
    rdata_2, rstats_2 = load_results_data_and_compute_stats(results_path_2)
    sizes_1 = load_mol_sizes_per_carbon(labels_path_1)
    sizes_2 = load_mol_sizes_per_carbon(labels_path_2)

    all_exp_1  = rdata_1[:, 0]
    all_pred_1 = rdata_1[:, 1]
    all_exp_2  = rdata_2[:, 0]
    all_pred_2 = rdata_2[:, 1]

    print(f"  Scatter 1:  R²={rstats_1['r2']:.4f}  MAE={rstats_1['mae']:.4f} eV  "
          f"STD={rstats_1['std']:.4f} eV")
    print(f"  Scatter 2:  R²={rstats_2['r2']:.4f}  MAE={rstats_2['mae']:.4f} eV  "
          f"STD={rstats_2['std']:.4f} eV")

    # Common colormap range across both panels
    vmin = min(sizes_1.min(), sizes_2.min())
    vmax = max(sizes_1.max(), sizes_2.max())

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
    gs  = GridSpec(1, 2, figure=fig, wspace=0.01)

    # Panel (a) — EQ
    ax_l = fig.add_subplot(gs[0, 0])
    ax_l, sc = scatter_plot(ax_l, all_pred_1, all_exp_1, rstats_1, scatter_s,
                            LINE_WIDTH, STATS_FONT, AXIS_FONT, TICK_FONT,
                            subplot='(a)', mol_sizes=sizes_1,
                            vmin=vmin, vmax=vmax, cmap='viridis')
    ax_l.text(0.95, 0.05, 'EGNN',
              ha='right', va='bottom', transform=ax_l.transAxes,
              fontsize=STATS_FONT, fontweight='bold',
              bbox=dict(boxstyle='round', edgecolor='grey',
                        facecolor='white', alpha=0.85, pad=0.5),
              zorder=5)

    # Panel (b) — IN
    ax_r = fig.add_subplot(gs[0, 1])
    ax_r, _ = scatter_plot(ax_r, all_pred_2, all_exp_2, rstats_2, scatter_s,
                           LINE_WIDTH, STATS_FONT, AXIS_FONT, TICK_FONT,
                           subplot='(b)', mol_sizes=sizes_2,
                           vmin=vmin, vmax=vmax, cmap='viridis')
    ax_r.text(0.95, 0.05, 'IGNN',
              ha='right', va='bottom', transform=ax_r.transAxes,
              fontsize=STATS_FONT, fontweight='bold',
              bbox=dict(boxstyle='round', edgecolor='grey',
                        facecolor='white', alpha=0.85, pad=0.5),
              zorder=5)

    # Shared colorbar
    cbar = fig.colorbar(sc, ax=[ax_l, ax_r], shrink=0.85, pad=0.02)
    cbar.set_label('Molecule Size (atoms)', fontsize=AXIS_FONT,
                   fontweight='bold')
    cbar.ax.tick_params(labelsize=TICK_FONT)

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
#  Two-panel figure: Val (left) vs Eval (right) for a single model type
# ─────────────────────────────────────────────────────────────────────────────

def scatter_2_panel_val_eval(model_type='EQ', name=None):
    """Two-panel scatter: exp-val (left) vs exp-eval (right) for one model.

    Parameters
    ----------
    model_type : str
        ``'EQ'`` or ``'IN'`` — selects the GNN layer type.
    name : str, optional
        Custom suffix for the output filenames.
    """
    layer_tag = f'{model_type}3'
    model_base = f'cebe_035_butina_{layer_tag}_h64_fold5'
    model_val  = f'expval_{model_base}'
    model_eval = f'expeval_{model_base}'

    results_path_val  = f'train_results/outputs/{model_val}_results.txt'
    results_path_eval = f'train_results/outputs/{model_eval}_results.txt'
    labels_path_val   = f'train_results/outputs/{model_val}_labels.txt'
    labels_path_eval  = f'train_results/outputs/{model_eval}_labels.txt'

    # ── Load data ────────────────────────────────────────────────────────
    rdata_val,  rstats_val  = load_results_data_and_compute_stats(results_path_val)
    rdata_eval, rstats_eval = load_results_data_and_compute_stats(results_path_eval)
    sizes_val  = load_mol_sizes_per_carbon(labels_path_val)
    sizes_eval = load_mol_sizes_per_carbon(labels_path_eval)

    all_exp_val   = rdata_val[:, 0]
    all_pred_val  = rdata_val[:, 1]
    all_exp_eval  = rdata_eval[:, 0]
    all_pred_eval = rdata_eval[:, 1]

    long_name = 'EGNN' if model_type == 'EQ' else 'IGNN'
    print(f"\n  {long_name} Val:   R²={rstats_val['r2']:.4f}  "
          f"MAE={rstats_val['mae']:.4f} eV  STD={rstats_val['std']:.4f} eV")
    print(f"  {long_name} Eval:  R²={rstats_eval['r2']:.4f}  "
          f"MAE={rstats_eval['mae']:.4f} eV  STD={rstats_eval['std']:.4f} eV")

    # Common colormap range across both panels (only molecules > 16 atoms)
    above_val  = sizes_val[sizes_val > 16]
    above_eval = sizes_eval[sizes_eval > 16]
    all_above  = np.concatenate([above_val, above_eval])
    vmin = all_above.min() if len(all_above) else 17
    vmax = all_above.max() if len(all_above) else 45

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
    scatter_s  = 60

    fig = plt.figure(figsize=(16, 6), constrained_layout=True)
    gs  = GridSpec(1, 2, figure=fig, wspace=0.01)

    # Panel (a) — Exp Validation
    ax_l = fig.add_subplot(gs[0, 0])
    ax_l, sc = scatter_plot(ax_l, all_pred_val, all_exp_val, rstats_val,
                            scatter_s, LINE_WIDTH, STATS_FONT, AXIS_FONT,
                            TICK_FONT, subplot='(a)', mol_sizes=sizes_val,
                            vmin=vmin, vmax=vmax, cmap='YlOrRd')
    #ax_l.text(0.95, 0.05, f'{long_name} — Val (N={len(all_exp_val)})',
    ax_l.text(0.95, 0.05, f'Exp. Validation',
              ha='right', va='bottom', transform=ax_l.transAxes,
              fontsize=STATS_FONT, fontweight='bold',
              bbox=dict(boxstyle='round', edgecolor='grey',
                        facecolor='white', alpha=0.85, pad=0.5),
              zorder=5)

    # Panel (b) — Exp Evaluation
    ax_r = fig.add_subplot(gs[0, 1])
    ax_r, sc_eval = scatter_plot(ax_r, all_pred_eval, all_exp_eval, rstats_eval,
                           scatter_s, LINE_WIDTH, STATS_FONT, AXIS_FONT,
                           TICK_FONT, subplot='(b)', mol_sizes=sizes_eval,
                           vmin=vmin, vmax=vmax, cmap='YlOrRd')
    #ax_r.text(0.95, 0.05, f'{long_name} — Eval (N={len(all_exp_eval)})',
    ax_r.text(0.95, 0.05, f'Exp. Evaluation',
              ha='right', va='bottom', transform=ax_r.transAxes,
              fontsize=STATS_FONT, fontweight='bold',
              bbox=dict(boxstyle='round', edgecolor='grey',
                        facecolor='white', alpha=0.85, pad=0.5),
              zorder=5)

    # Shared colorbar (use eval scatter which has the larger molecules)
    cbar = fig.colorbar(sc_eval, ax=[ax_l, ax_r], shrink=0.85, pad=0.02)
    cbar.set_label('Molecule Size (atoms)', fontsize=AXIS_FONT,
                   fontweight='bold')
    cbar.ax.tick_params(labelsize=TICK_FONT)

    # ── Save ─────────────────────────────────────────────────────────────
    tag = name if name else f'val_eval_{model_type}'
    png_path = f'pngs/2_panel_scatter_{tag}.png'
    pdf_path = f'pngs/2_panel_scatter_{tag}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  ✓ {png_path}")
    print(f"  ✓ {pdf_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    param_json = 'param_results/search_feature_keys2_n_layers8_cebe_035_butina_EQ3_h64_param_summary.json'

    # Butina fold-5 exp-eval: EQ vs IN
    model_EQ = 'expeval_cebe_035_butina_EQ3_h64_fold5'
    model_IN = 'expeval_cebe_035_butina_IN3_h64_fold5'

    scatter_2_panel(model_EQ, model_IN, name='EQ_vs_IN')
    scatter_2_panel_val_eval(model_type='EQ')
    scatter_2_panel_val_eval(model_type='IN')
    layer_scan_2_panel(param_json, param_json, param = "feature_keys", model_1='035', model_2='0', name='feature_keys')
    mae_and_cf_2_panel(param_json, param='feature_keys',
                       model_1='035', model_2='0', name='feature_keys')
