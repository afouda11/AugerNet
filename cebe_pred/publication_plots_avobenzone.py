import os
import re
import json
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import seaborn as sns
from collections import OrderedDict, Counter

from torch_geometric.data import Data
from augernet.carbon_environment import IDX_TO_CARBON_ENV

from publication_plots import load_label_data

os.makedirs('avobenzone_analysis', exist_ok=True)

PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
ANALYSIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'avobenzone_analysis')


# ─────────────────────────────────────────────────────────────────────────────
#  Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_molecule_graphs():
    """Load processed CEBE eval data and return per-molecule Data objects.

    Each returned object carries ``carbon_env_labels``, ``atom_symbols``,
    ``true_cebe``, ``edge_index``, and ``mol_name`` (str).
    """
    data_path = os.path.join(PROJECT_ROOT, 'data', 'processed',
                             'gnn_exp_cebe_data.pt')
    collated, slices = torch.load(data_path, weights_only=False)
    n_mols = len(slices[list(slices.keys())[0]]) - 1

    molecules = []
    for i in range(n_mols):
        d = Data()
        for key in slices:
            s_idx = slices[key][i].item()
            e_idx = slices[key][i + 1].item()
            attr = getattr(collated, key)
            if torch.is_tensor(attr):
                if attr.dim() == 0:
                    d[key] = attr
                elif attr.dim() == 2 and key == 'edge_index':
                    d[key] = attr[:, s_idx:e_idx]
                else:
                    d[key] = attr[s_idx:e_idx]
            else:
                d[key] = attr[s_idx:e_idx] if isinstance(attr, list) else attr
        # Attach per-molecule list fields that are indexed by molecule, not atom
        d._mol_name = collated.mol_name[i]
        d._atom_symbols = collated.atom_symbols[i]
        molecules.append(d)
    return molecules


def _load_spectrum(filename):
    """Load a comma-separated spectrum file, sort by BE, normalise to [0,1]."""
    fpath = os.path.join(ANALYSIS_DIR, filename)
    if not os.path.exists(fpath):
        print(f"  ⚠  Spectrum file not found: {fpath}")
        return None, None
    d = np.genfromtxt(fpath, delimiter=',', dtype=float)
    d = d[d[:, 0].argsort()]
    be = d[:, 0]
    intensity = d[:, 1]
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    return be, intensity


def _load_avobenzone_data(labels_path):
    """Load GNN predictions and carbon-env labels for keto / enol avobenzone.

    Returns
    -------
    mol_data : dict
        ``{mol_name: list of (env_name, env_idx, true_BE, pred_BE, atom_idx)}``
    """
    molecules = _load_molecule_graphs()
    label_mols = load_label_data(labels_path)

    target_names = {'ketoavobenzone', 'enolavobenzone'}

    # Build a lookup from mol_name → Data for the two target molecules
    graph_by_name = {}
    for mol in molecules:
        name = mol._mol_name
        if name in target_names:
            graph_by_name[name] = mol

    # Build a lookup from mol_name → label data
    label_by_name = {}
    for entry in label_mols:
        if entry['name'] in target_names:
            label_by_name[entry['name']] = entry

    mol_data = {}
    for name in ['ketoavobenzone', 'enolavobenzone']:
        if name not in graph_by_name:
            print(f"  ⚠  {name} not found in graph data")
            continue
        if name not in label_by_name:
            print(f"  ⚠  {name} not found in label results")
            continue

        g = graph_by_name[name]
        lab = label_by_name[name]
        env_labels = g.carbon_env_labels.tolist()
        true_cebes = g.true_cebe.squeeze().tolist()
        atom_syms = g._atom_symbols

        # label data: arrays of true / pred  (carbon-only)
        pred_arr = lab['pred']
        true_arr = lab['true']

        entries = []
        c_idx = 0   # counter for carbon atoms (label arrays are C-only)
        for j, sym in enumerate(atom_syms):
            if sym == 'C':
                eidx = env_labels[j] if j < len(env_labels) else -1
                env_name = IDX_TO_CARBON_ENV.get(eidx, 'C_unknown')
                t_be = true_cebes[j] if j < len(true_cebes) and true_cebes[j] > 0 else np.nan
                p_be = pred_arr[c_idx] if c_idx < len(pred_arr) else np.nan
                entries.append((env_name, eidx, float(t_be), float(p_be), j))
                c_idx += 1
        mol_data[name] = entries

    return mol_data


# ─────────────────────────────────────────────────────────────────────────────
#  Axes-level plot: XPS overlay
# ─────────────────────────────────────────────────────────────────────────────

def xps_overlay_plot(ax, mol_data,
                     exp_be, exp_int,
                     keto_calc_be, keto_calc_int,
                     enol_calc_be, enol_calc_int,
                     LINE_WIDTH, STATS_FONT, AXIS_FONT, TICK_FONT,
                     subplot=True, panel_label='(a)'):
    """Draw the XPS overlay on *ax*: spectra (top) + GNN sticks (keto, enol).

    Layout (bottom → top inside the axes):
      * enol sticks  y ∈ [0.00, 0.22]
      * keto sticks  y ∈ [0.28, 0.50]
      * spectra      y ∈ [0.56, ~1.01]
    """
    # ── Layout constants ─────────────────────────────────────────────────
    ENOL_BASE  = 0.00;  ENOL_HEIGHT = 0.22
    KETO_BASE  = 0.28;  KETO_HEIGHT = 0.22
    EXP_OFFSET = 0.56;  EXP_SCALE   = 0.45
    Y_TOP      = EXP_OFFSET + EXP_SCALE + 0.08

    row_params = {
        'ketoavobenzone':  (KETO_BASE, KETO_HEIGHT),
        'enolavobenzone':  (ENOL_BASE, ENOL_HEIGHT),
    }

    STICK_WIDTH = 2.0
    ANNOT_FONT  = 12
    LEGEND_FONT = 11

    # ── Colour / style dicts (keyed by env name) ────────────────────────
    label_prefix = {
        'C_aromatic':    'a: ', 'C_ketone':       'b: ',
        'C_methylene':   'c: ', 'C_aryl_ether':   'd: ',
        'C_ether':       'e: ', 'C_quaternary':   'f: ',
        'C_methyl':      'g: ', 'C_enol':         'h: ',
        'C_vinyl':       'i: ',
    }
    env_color = {
        'C_aromatic':    '#0072B2', 'C_ketone':     '#CC79A7',
        'C_methylene':   '#009E73', 'C_aryl_ether': '#999933',
        'C_ether':       '#56B4E9', 'C_quaternary': '#000000',
        'C_methyl':      '#F0E442', 'C_enol':       '#D55E00',
        'C_vinyl':       '#E91E8C',
    }
    env_ls = {
        'C_aromatic':    '--', 'C_ketone':       ':',
        'C_methylene':   '--', 'C_aryl_ether':   ':',
        'C_ether':       ':', 'C_quaternary':    '--',
        'C_methyl':      '--', 'C_enol':         ':',
        'C_vinyl':       '--',
    }

    # ── Experimental spectrum (top zone) ─────────────────────────────────
    exp_raised = exp_int * EXP_SCALE + EXP_OFFSET
    ax.plot(exp_be, exp_raised, 'k-', linewidth=LINE_WIDTH, alpha=0.7,
            label='Exp.', zorder=5)

    # ── Calculated (DFT) spectra ─────────────────────────────────────────
    if keto_calc_be is not None:
        keto_raised = keto_calc_int * EXP_SCALE + EXP_OFFSET
        ax.plot(keto_calc_be, keto_raised, '--', color='g',
                linewidth=LINE_WIDTH, alpha=0.7, label='Calc. Keto', zorder=4)
    if enol_calc_be is not None:
        enol_raised = enol_calc_int * EXP_SCALE + EXP_OFFSET
        ax.plot(enol_calc_be, enol_raised, ':', color='m',
                linewidth=LINE_WIDTH, alpha=0.7, label='Calc. Enol', zorder=4)

    # ── Separator lines between zones ────────────────────────────────────
    for y_sep in [KETO_BASE + KETO_HEIGHT + 0.03, EXP_OFFSET - 0.03]:
        ax.axhline(y_sep, color='#999999', linewidth=0.5, linestyle='-',
                   alpha=0.4, zorder=1)

    # ── GNN sticks for each molecule ─────────────────────────────────────
    target_order = ['ketoavobenzone', 'enolavobenzone']
    plotted_env_labels = set()

    for mol_name in target_order:
        entries = mol_data.get(mol_name, [])
        if not entries:
            continue
        base, height = row_params[mol_name]

        env_groups = OrderedDict()
        for env_name, _eidx, _t, p_be, _aidx in entries:
            env_groups.setdefault(env_name, []).append(p_be)

        for env_name, preds in env_groups.items():
            for p_be in preds:
                lbl = None
                if env_name not in plotted_env_labels:
                    lbl = env_name.replace(
                        'C_', label_prefix.get(env_name, ''))
                    lbl = lbl.replace('_', ' ')
                    plotted_env_labels.add(env_name)

                ax.plot([p_be, p_be], [base, base + height],
                        color=env_color.get(env_name, '#888888'),
                        linewidth=STICK_WIDTH,
                        linestyle=env_ls.get(env_name, '-'),
                        alpha=0.85, label=lbl, zorder=3)

    # ── Row annotations ──────────────────────────────────────────────────
    ax.annotate('Keto', xy=(0.02, KETO_BASE + KETO_HEIGHT / 2),
                xycoords=('axes fraction', 'data'),
                fontsize=ANNOT_FONT, fontweight='bold',
                va='center', ha='left', color='g')
    ax.annotate('Enol', xy=(0.02, ENOL_BASE + ENOL_HEIGHT / 2),
                xycoords=('axes fraction', 'data'),
                fontsize=ANNOT_FONT, fontweight='bold',
                va='center', ha='left', color='m')

    # ── Split legend: spectra (top-left) + sticks (right) ───────────────
    handles, labels = ax.get_legend_handles_labels()
    spec_h, spec_l, stick_h, stick_l = [], [], [], []
    spectra_keys = {'Exp.', 'Calc. Keto', 'Calc. Enol'}
    for h, l in zip(handles, labels):
        (spec_h if l in spectra_keys else stick_h).append(h)
        (spec_l if l in spectra_keys else stick_l).append(l)

    leg_top = ax.legend(spec_h, spec_l, fontsize=LEGEND_FONT,
                        loc='upper left', ncol=1, framealpha=0.85,
                        fancybox='round', bbox_to_anchor=(0.0, 1.0))
    ax.add_artist(leg_top)
    ax.legend(stick_h, stick_l, fontsize=LEGEND_FONT,
              loc='upper right', ncol=1, framealpha=0.85,
              fancybox='round', bbox_to_anchor=(1.0, 0.51))

    # ── Axes formatting ──────────────────────────────────────────────────
    ax.set_xlabel('Binding Energy (eV)', fontsize=AXIS_FONT,
                  fontweight='bold')
    ax.set_ylabel('Intensity (arb. units)', fontsize=AXIS_FONT,
                  fontweight='bold')
    ax.tick_params(axis='both', labelsize=TICK_FONT)
    ax.set_yticks([])
    ax.set_ylim(-0.05, Y_TOP)
    ax.grid(True, alpha=0.3, linewidth=1.0, axis='both', zorder=0)
    ax.set_axisbelow(True)
    ax.invert_xaxis()

    if subplot:
        ax.text(-0.12, 1.05, panel_label, transform=ax.transAxes,
                fontsize=AXIS_FONT + 2, fontweight='bold', va='top')

    return ax


# ─────────────────────────────────────────────────────────────────────────────
#  One-panel figure: XPS overlay
# ─────────────────────────────────────────────────────────────────────────────

def xps_overlay(labels_path, name=None):
    """Single-panel XPS overlay: experimental + DFT spectra + GNN sticks."""

    mol_data = _load_avobenzone_data(labels_path)
    if not mol_data:
        print("  ⚠  No avobenzone data — skipping XPS overlay")
        return

    # Load spectra
    exp_be, exp_int = _load_spectrum('enol_keto_avobenzene_exp_spec.txt')
    keto_be, keto_int = _load_spectrum('keto_avobenzene_calc_spec.txt')
    enol_be, enol_int = _load_spectrum('enol_avobenzene_calc_spec.txt')

    if exp_be is None:
        print("  ⚠  Experimental spectrum not found — skipping")
        return

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
    TICK_FONT  = 12
    STATS_FONT = 11
    LINE_WIDTH = 2.5

    fig, ax = plt.subplots(figsize=(8, 5))
    xps_overlay_plot(ax, mol_data, exp_be, exp_int,
                     keto_be, keto_int, enol_be, enol_int,
                     LINE_WIDTH, STATS_FONT, AXIS_FONT, TICK_FONT,
                     subplot=False)
    fig.tight_layout()

    tag = f'_{name}' if name else ''
    png_path = f'avobenzone_analysis/xps_overlay_avobenzone{tag}.png'
    pdf_path = f'avobenzone_analysis/xps_overlay_avobenzone{tag}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  ✓ {png_path}")
    print(f"  ✓ {pdf_path}")
    plt.close(fig)

    # ── Print companion tables ───────────────────────────────────────────
    _print_avobenzone_tables(mol_data, tag)


# ─────────────────────────────────────────────────────────────────────────────
#  Companion tables (CSV + stdout)
# ─────────────────────────────────────────────────────────────────────────────

def _print_avobenzone_tables(mol_data, tag=''):
    """Write per-atom and averaged CSV tables + print to stdout."""

    target_order = ['ketoavobenzone', 'enolavobenzone']

    # Shared letter mapping across both molecules
    all_env_names = list(OrderedDict.fromkeys(
        env_name
        for mol_name in target_order
        for env_name, *_ in mol_data.get(mol_name, [])
    ))
    env_to_letter = {env: chr(ord('a') + i) for i, env in enumerate(all_env_names)}

    # ── Per-atom CSV ─────────────────────────────────────────────────────
    table_rows = []
    for mol_name in target_order:
        for env_name, eidx, t_be, p_be, atom_idx in mol_data.get(mol_name, []):
            table_rows.append({
                'molecule': mol_name,
                'label': env_to_letter[env_name],
                'atom_index': atom_idx + 1,
                'carbon_env': env_name.replace('C_', ''),
                'true_BE_eV': f'{t_be:.1f}',
                'pred_BE_eV': f'{p_be:.4f}',
                'error_eV': f'{p_be - t_be:.4f}',
            })

    csv_path = f'avobenzone_analysis/avobenzone_table{tag}.csv'
    fieldnames = ['molecule', 'label', 'atom_index', 'carbon_env',
                  'true_BE_eV', 'pred_BE_eV', 'error_eV']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(table_rows)
    print(f"  ✓ {csv_path}")

    # ── Averaged CSV ─────────────────────────────────────────────────────
    avg_rows = []
    for mol_name in target_order:
        env_groups = OrderedDict()
        for env_name, eidx, t_be, p_be, aidx in mol_data.get(mol_name, []):
            env_groups.setdefault(env_name, []).append((t_be, p_be))
        for env_name, group in env_groups.items():
            tv = np.array([t for t, p in group])
            pv = np.array([p for t, p in group])
            avg_rows.append({
                'molecule': mol_name,
                'label': env_to_letter[env_name],
                'carbon_env': env_name.replace('C_', ''),
                'n_atoms': len(group),
                'avg_true_BE_eV': f'{np.mean(tv):.4f}',
                'avg_pred_BE_eV': f'{np.mean(pv):.4f}',
                'avg_error_eV': f'{np.mean(pv - tv):.4f}',
                'std_pred_eV': f'{np.std(pv):.4f}' if len(group) > 1 else '—',
            })

    avg_path = f'avobenzone_analysis/avobenzone_table_avg{tag}.csv'
    avg_fields = ['molecule', 'label', 'carbon_env', 'n_atoms',
                  'avg_true_BE_eV', 'avg_pred_BE_eV', 'avg_error_eV',
                  'std_pred_eV']
    with open(avg_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=avg_fields)
        w.writeheader()
        w.writerows(avg_rows)
    print(f"  ✓ {avg_path}")

    # ── Stdout: per-atom table ───────────────────────────────────────────
    print(f"\n  {'Molecule':<20s} {'Lbl':>3s} {'Idx':>4s} {'Carbon Env':<18s} "
          f"{'True (eV)':>9s} {'Pred (eV)':>10s} {'Err (eV)':>9s}")
    print("  " + "-" * 80)
    for r in table_rows:
        print(f"  {r['molecule']:<20s} {r['label']:>3s} {r['atom_index']:>4d} "
              f"{r['carbon_env']:<18s} "
              f"{r['true_BE_eV']:>9s} {r['pred_BE_eV']:>10s} "
              f"{r['error_eV']:>9s}")

    # ── Stdout: averaged table ───────────────────────────────────────────
    print(f"\n  {'Molecule':<20s} {'Lbl':>3s} {'Carbon Env':<18s} {'N':>3s} "
          f"{'⟨True⟩':>9s} {'⟨Pred⟩':>10s} {'⟨Err⟩':>9s} {'σ(Pred)':>9s}")
    print("  " + "-" * 85)
    for r in avg_rows:
        print(f"  {r['molecule']:<20s} {r['label']:>3s} "
              f"{r['carbon_env']:<18s} {r['n_atoms']:>3d} "
              f"{r['avg_true_BE_eV']:>9s} {r['avg_pred_BE_eV']:>10s} "
              f"{r['avg_error_eV']:>9s} {r['std_pred_eV']:>9s}")


# ─────────────────────────────────────────────────────────────────────────────
#  LaTeX table: side-by-side keto / enol with atom labels from key files
# ─────────────────────────────────────────────────────────────────────────────

KEY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'avobenzone_analysis')


def _parse_key_file(path):
    """Parse a keto_key.txt / enol_key.txt file.

    Returns
    -------
    entries : list of (label, atom_index_1based)
        In file order.  ``atom_index`` is 1-based (XYZ convention).
    """
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('label'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                entries.append((parts[0], int(parts[1])))
    return entries


def _build_table_rows(mol_data, key_entries, mol_name):
    """Build ordered table rows for one molecule using its key-file entries.

    For labels with a numeric suffix (e.g. ``a1``, ``b2``) → single atom.
    For labels without a suffix (e.g. ``c``, ``g``) → average all atoms
    that share that environment.

    Returns
    -------
    rows : list of dict
        Keys: ``label``, ``env``, ``exp``, ``pred``, ``error``
    """
    entries = mol_data.get(mol_name, [])
    # Index by 1-based atom index → (env_name, true_BE, pred_BE)
    by_aidx = {}
    for env_name, _eidx, t_be, p_be, atom_idx_0 in entries:
        by_aidx[atom_idx_0 + 1] = (env_name, t_be, p_be)

    # Group all atoms by their env name (for averaged rows)
    from collections import defaultdict
    env_atoms = defaultdict(list)
    for env_name, _eidx, t_be, p_be, _aidx in entries:
        env_atoms[env_name].append((t_be, p_be))

    # Determine which labels are per-atom vs averaged
    # Labels with a digit → per-atom;  letters only → averaged
    import re as _re

    rows = []
    seen_env_for_avg = set()

    for label, aidx_1 in key_entries:
        has_digit = bool(_re.search(r'\d', label))

        if has_digit:
            # Per-atom row
            if aidx_1 not in by_aidx:
                continue
            env_name, t_be, p_be = by_aidx[aidx_1]
            rows.append({
                'label': label,
                'env': env_name.replace('C_', '').replace('_', ' '),
                'exp': t_be,
                'pred': p_be,
                'error': p_be - t_be,
            })
        else:
            # Averaged row — average all atoms of this env
            if aidx_1 not in by_aidx:
                continue
            env_name = by_aidx[aidx_1][0]
            if env_name in seen_env_for_avg:
                continue
            seen_env_for_avg.add(env_name)

            group = env_atoms[env_name]
            t_arr = np.array([t for t, p in group])
            p_arr = np.array([p for t, p in group])
            rows.append({
                'label': label,
                'env': env_name.replace('C_', '').replace('_', ' '),
                'exp': float(np.mean(t_arr)),
                'pred': float(np.mean(p_arr)),
                'error': float(np.mean(p_arr - t_arr)),
            })

    return rows


def _fmt_error(val):
    """Format error for LaTeX: negative values get $-$prefix."""
    if val < 0:
        return f'$-${abs(val):.2f}'
    return f'{val:.2f}'


def _is_first_of_env(rows, idx):
    """True if row *idx* is the first row with its environment name."""
    env = rows[idx]['env']
    for i in range(idx):
        if rows[i]['env'] == env:
            return False
    return True


def _env_block_changed(rows, idx):
    """True if the env letter-prefix changed from the previous row."""
    if idx == 0:
        return False
    cur_letter = rows[idx]['label'].rstrip('0123456789')
    prev_letter = rows[idx - 1]['label'].rstrip('0123456789')
    return cur_letter != prev_letter


def _exp_str(row, rows, idx):
    """Return the exp column string (shown only for the middle row of each env block)."""
    env = rows[idx]['env']
    # Collect all row indices that share this env
    block = [i for i, r in enumerate(rows) if r['env'] == env]
    mid = block[len(block) // 2]
    if idx == mid:
        return f'{row["exp"]:.1f}'
    return ''


def write_avobenzone_latex(mol_data, tag=''):
    """Write the side-by-side keto/enol LaTeX table matching the paper format."""

    keto_key = _parse_key_file(os.path.join(KEY_DIR, 'keto_key.txt'))
    enol_key = _parse_key_file(os.path.join(KEY_DIR, 'enol_key.txt'))

    keto_rows = _build_table_rows(mol_data, keto_key, 'ketoavobenzone')
    enol_rows = _build_table_rows(mol_data, enol_key, 'enolavobenzone')

    n_rows = max(len(keto_rows), len(enol_rows))

    lines = []
    lines.append(r'\begin{table*}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{EGNN predicted and approximate experimental carbon 1$s$ '
                 r"CEBE's of ketoavobenzone and enolavobenzone with the approximate "
                 r'mean average errors (MAE) of the predictions with respect to the '
                 r'experimental values, all energies are given in (eV). The C column '
                 r'has the label for each carbon atom given in Figure \ref{fig:avobenzone} '
                 r'b) and the Env.\ column contains the carbons environment class, '
                 r'determined by the SMARTS pattern matching procedure given in the SI.}')
    lines.append(r'\label{tab:avobenzone}')
    lines.append(r'\footnotesize')
    lines.append(r'\setlength{\tabcolsep}{9.5pt}')
    lines.append(r'\begin{tabular}{llcrr@{\hskip 2em}llcrr}')
    lines.append(r'\hline')
    lines.append(r'\multicolumn{5}{c}{Ketoavobenzone} & '
                 r'\multicolumn{5}{c}{Enolavobenzone} \\')
    lines.append(r'C   & Env.        & $\approx$ Exp.\ (eV) & GNN  (eV)  '
                 r'& $\approx$ Error (eV) & C   & Env.        '
                 r'& $\approx$ Exp.\ (eV) & GNN (eV) & $\approx$ Error (eV) \\')
    lines.append(r'\hline')

    for i in range(n_rows):
        # ── Left (keto) ─────────────────────────────────────────────────
        if i < len(keto_rows):
            kr = keto_rows[i]
            k_label = kr['label']
            k_env = kr['env'] if _is_first_of_env(keto_rows, i) else ''
            k_exp = _exp_str(kr, keto_rows, i)
            k_pred = f'{kr["pred"]:.2f}'
            k_err = _fmt_error(kr['error'])
        else:
            k_label = k_env = k_exp = k_pred = k_err = ''

        # ── Right (enol) ────────────────────────────────────────────────
        if i < len(enol_rows):
            er = enol_rows[i]
            e_label = er['label']
            e_env = er['env'] if _is_first_of_env(enol_rows, i) else ''
            e_exp = _exp_str(er, enol_rows, i)
            e_pred = f'{er["pred"]:.2f}'
            e_err = _fmt_error(er['error'])
        else:
            e_label = e_env = e_exp = e_pred = e_err = ''

        row_str = (f'{k_label:<4s} & {k_env:<12s} & {k_exp:>10s} & '
                   f'{k_pred:>8s} & {k_err:>12s} & '
                   f'{e_label:<4s} & {e_env:<12s} & {e_exp:>10s} & '
                   f'{e_pred:>8s} & {e_err:>12s} \\\\')
        lines.append(row_str)

        # ── Env-block separators ────────────────────────────────────────
        # Insert \cmidrule when the env block changes on either side
        need_keto_rule = (i + 1 < len(keto_rows)
                          and _env_block_changed(keto_rows, i + 1))
        need_enol_rule = (i + 1 < len(enol_rows)
                          and _env_block_changed(enol_rows, i + 1))

        # Also add rules at the very end of one side if the other continues
        if i + 1 == len(keto_rows) and i + 1 < len(enol_rows):
            need_keto_rule = True
        if i + 1 == len(enol_rows) and i + 1 < len(keto_rows):
            need_enol_rule = True

        if need_keto_rule and need_enol_rule:
            lines.append(r'\cmidrule(r){1-5} \cmidrule{6-10}')
        elif need_keto_rule:
            lines.append(r'\cmidrule(r){1-5}')
        elif need_enol_rule:
            lines.append(r'\cmidrule{6-10}')

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table*}')

    tex_path = f'avobenzone_analysis/avobenzone_table{tag}.tex'
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  ✓ {tex_path}")

    return tex_path


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    model = f'cebe_035_random_EQ4_h32_fold1'

    # Labels file is in the param_results/outputs directory
    labels_path = f'train_results/outputs/{model}_labels.txt'

    xps_overlay(labels_path, name=model)

    # Generate the LaTeX table
    mol_data = _load_avobenzone_data(labels_path)
    write_avobenzone_latex(mol_data, tag=f'_{model}')
