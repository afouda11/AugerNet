#!/usr/bin/env python3
"""
Carbon-environment class distribution figure for the CEBE GNN paper.

Produces a 2-panel figure with a shared legend/key table below:

  (e)  Train vs Calc-val.   — grouped bar chart (log scale)
  (f)  Exp-val. vs Eval.    — grouped bar chart (log scale)

Environment classes are assigned compact labels (A-i, A-ii, …) grouped
by chemical similarity.  A key table below the panels maps every label
to its full SMARTS-based class name.

Two width variants are saved:
  * ``\\columnwidth`` (~3.5 in):  ``env_distribution_col.{png,pdf}``
  * ``\\textwidth``   (~7.2 in):  ``env_distribution_text.{png,pdf}``

Usage::

    cd cebe_pred && python publication_plots_env.py
"""

import os, sys, collections
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import GroupKFold

# ── Project imports ──────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from augernet.build_molecular_graphs import get_butina_clusters
from augernet.carbon_environment import IDX_TO_CARBON_ENV

PNG_DIR = os.path.join(SCRIPT_DIR, 'pngs')
os.makedirs(PNG_DIR, exist_ok=True)

# ── Parameters ───────────────────────────────────────────────────────────────
N_FOLDS = 10
FOLD    = 5          # 1-indexed

# ═════════════════════════════════════════════════════════════════════════════
#  Environment grouping
# ═════════════════════════════════════════════════════════════════════════════
#
# Each group is (group_letter, group_display_name, [(env_name, sub_label), …])
# The order within each group defines the sub-label suffix (i, ii, iii, …).
# The order of groups defines the left→right bar ordering on the x-axis.

_ROMAN = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii']

ENV_GROUPS = [
    ('A', 'Aromatic', [
        ('C_aromatic',   'generic'),
        ('C_arom_N',     'adj. N'),
        ('C_arom_O',     'adj. O'),
        ('C_arom_O_N',   'adj. O,N'),
    ]),
    ('B', 'Aryl-subst.', [
        ('C_aryl_amine',    'Ar-NH$_x$'),
        ('C_phenol',        'Ar-OH'),
        ('C_aryl_ether',    'Ar-O-R'),
        ('C_aryl_fluoride', 'Ar-F'),
        ('C_aryl_nitro',    'Ar-NO$_2$'),
    ]),
    ('C', 'Carbonyl', [
        ('C_ketone',          'ketone'),
        ('C_aldehyde',        'aldehyde'),
        ('C_amide_carbonyl',  'amide'),
        ('C_ester_carbonyl',  'ester'),
        ('C_carboxylic_acid', 'COOH'),
        ('C_carboxylate',     'COO$^-$'),
        ('C_acyl_fluoride',   'COF'),
    ]),
    ('D', 'N-bonded', [
        ('C_nitrile', 'nitrile'),
        ('C_imine',   'imine'),
        ('C_amine',   'amine'),
    ]),
    ('E', 'Aliphatic', [
        ('C_methyl',      'CH$_3$'),
        ('C_methylene',   'CH$_2$'),
        ('C_methine',     'CH'),
        ('C_quaternary',  'C(quat.)'),
    ]),
    ('F', 'Unsaturated', [
        ('C_alkyne',     'alkyne'),
        ('C_vinyl',      'vinyl'),
        ('C_enol',       'enol'),
        ('C_allene',     'allene'),
        ('C_ketene',     'ketene'),
        ('C_ketenimine', 'ketenimine'),
    ]),
    ('G', 'O-bonded (sp$^3$)', [
        ('C_alcohol',    'alcohol'),
        ('C_ether',      'ether'),
        ('C_ester_alkyl','ester alkyl'),
    ]),
    ('H', 'Fluorinated', [
        ('C_fluorinated', 'C-F'),
    ]),
    ('I', 'Cumulene / misc.', [
        ('C_isocyanate',    'isocyanate'),
        ('C_carbodiimide',  'carbodiimide'),
        ('C_CO2',           'CO$_2$'),
    ]),
]

# Build flat ordered lists: short_label, env_index, full display name
_ENV_NAME_TO_IDX = {v: k for k, v in IDX_TO_CARBON_ENV.items()}

ENV_ORDER     = []   # env indices in display order
ENV_TICK      = []   # short x-tick labels  e.g. "A-i"
ENV_KEY_ROWS  = []   # (short_label, full_name) for the key table

for grp_letter, grp_name, members in ENV_GROUPS:
    for sub_idx, (env_name, human_name) in enumerate(members):
        env_idx = _ENV_NAME_TO_IDX[env_name]
        short = f'{grp_letter}-{_ROMAN[sub_idx]}'
        ENV_ORDER.append(env_idx)
        ENV_TICK.append(short)
        ENV_KEY_ROWS.append((short, grp_name, human_name))


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _load_exp_split_names():
    exp_dir = os.path.join(PROJECT_ROOT, 'data', 'raw', 'exp_cebe')
    def _read(fname):
        with open(os.path.join(exp_dir, fname)) as f:
            return {line.strip() for line in f if line.strip()}
    return _read('mol_list_val.txt'), _read('mol_list_eval.txt')


def _exp_split_indices(data_exp, slices_exp):
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


def _count_carbon_envs(data, slices, mol_indices):
    """Counter mapping env_idx → count (non-carbon atoms excluded)."""
    env_sl = slices['carbon_env_labels']
    cts = collections.Counter()
    for mi in mol_indices:
        s = env_sl[mi].item()
        e = env_sl[mi + 1].item()
        for lab in data.carbon_env_labels[s:e].tolist():
            if lab >= 0:
                cts[lab] += 1
    return cts


# ═════════════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ═════════════════════════════════════════════════════════════════════════════

def _env_panel(ax, cts_a, cts_b, label_a, label_b, color_a, color_b,
               panel_label='(e)', y_title='Count', x_title=None,
               AXIS_FONT=12, TICK_FONT=9, LEG_FONT=10):
    """Grouped bar chart for two splits on one Axes, using ENV_ORDER."""
    n = len(ENV_ORDER)
    x = np.arange(n)
    w = 0.38

    vals_a = np.array([cts_a.get(e, 0) for e in ENV_ORDER], dtype=float)
    vals_b = np.array([cts_b.get(e, 0) for e in ENV_ORDER], dtype=float)

    ax.bar(x - w/2, vals_a, width=w, color=color_a, alpha=0.85,
           edgecolor='white', linewidth=0.3, label=label_a, zorder=3)
    ax.bar(x + w/2, vals_b, width=w, color=color_b, alpha=0.85,
           edgecolor='white', linewidth=0.3, label=label_b, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(ENV_TICK, fontsize=TICK_FONT, rotation=60, ha='right')

    ax.set_yscale('log')
    ax.set_ylim(0.8, None)

    if y_title:
        ax.set_ylabel(y_title, fontsize=AXIS_FONT, fontweight='bold')
    if x_title:
        ax.set_xlabel(x_title, fontsize=AXIS_FONT, fontweight='bold')

    # Add group separators (vertical lines between groups)
    pos = 0
    for _, _, members in ENV_GROUPS:
        pos += len(members)
        if pos < n:
            ax.axvline(pos - 0.5, color='grey', linewidth=0.6,
                       linestyle='--', alpha=0.5, zorder=1)

    ax.text(-0.12, 1.06, panel_label, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')


def _draw_key_panel(ax_key, FONT=8):
    """Draw the environment-class key as text inside a dedicated Axes.

    Groups are listed top-to-bottom, one after another, fitting into the
    right-hand side panel that spans the full height of both bar panels.
    """
    ax_key.axis('off')

    # Total lines needed: each group has 1 header + len(members) entries
    total_lines = sum(1 + len(m) for _, _, m in ENV_GROUPS)
    # add small gaps between groups
    total_lines += len(ENV_GROUPS) - 1  # 1-line gap between groups

    line_height = 1.0 / (total_lines + 0.5)
    y = 1.0   # start at top

    for gi, (grp_letter, grp_name, members) in enumerate(ENV_GROUPS):
        # Group header
        ax_key.text(0.0, y, f'{grp_letter} -- {grp_name}',
                    fontsize=FONT + 0.5, fontweight='bold',
                    family='sans-serif', va='top',
                    transform=ax_key.transAxes)
        y -= line_height
        for sub_idx, (env_name, human_name) in enumerate(members):
            short = f'{grp_letter}-{_ROMAN[sub_idx]}'
            ax_key.text(0.04, y, f'{short}:  {human_name}',
                        fontsize=FONT, family='sans-serif', va='top',
                        transform=ax_key.transAxes)
            y -= line_height
        # gap between groups
        if gi < len(ENV_GROUPS) - 1:
            y -= line_height * 0.3


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def _load_splits():
    """Return (data_calc, sl_calc, train_idx, val_idx,
               data_exp, sl_exp, exp_val_idx, exp_eval_idx)."""
    CALC_PT = os.path.join(PROJECT_ROOT, 'data', 'processed',
                           'gnn_calc_cebe_data.pt')
    EXP_PT  = os.path.join(PROJECT_ROOT, 'data', 'processed',
                           'gnn_exp_cebe_data.pt')

    data_calc, sl_calc = torch.load(CALC_PT, weights_only=False)
    data_exp,  sl_exp  = torch.load(EXP_PT,  weights_only=False)

    n_calc = len(sl_calc['x']) - 1

    smiles_list = [data_calc.smiles[i] for i in range(n_calc)]
    cluster_ids = get_butina_clusters(smiles_list, cutoff=0.65)
    print(f"  Butina clustering -> {len(set(cluster_ids))} clusters")

    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = list(gkf.split(np.arange(n_calc), groups=cluster_ids))
    train_idx, val_idx = folds[FOLD - 1]
    print(f"  Fold {FOLD}: {len(train_idx)} train / {len(val_idx)} val (calc)")

    exp_val_idx, exp_eval_idx = _exp_split_indices(data_exp, sl_exp)
    print(f"  Exp: {len(exp_val_idx)} val / {len(exp_eval_idx)} eval")

    return (data_calc, sl_calc, train_idx, val_idx,
            data_exp, sl_exp, exp_val_idx, exp_eval_idx)


def _make_figure(figwidth, suffix, key_font=8):
    """Build and save the env-distribution figure at a given width.

    Layout: 2 rows × 1 col (bar panels) on the left,
            key/legend panel on the right spanning both rows.
    """

    (data_calc, sl_calc, train_idx, val_idx,
     data_exp, sl_exp, exp_val_idx, exp_eval_idx) = _load_splits()

    env_train = _count_carbon_envs(data_calc, sl_calc, train_idx)
    env_cval  = _count_carbon_envs(data_calc, sl_calc, val_idx)
    env_expv  = _count_carbon_envs(data_exp,  sl_exp,  exp_val_idx)
    env_expe  = _count_carbon_envs(data_exp,  sl_exp,  exp_eval_idx)

    # ── Colours (same as data_analysis) ───────────────────────────────────
    c_train = '#0072B2'
    c_val   = '#E69F00'
    c_expv  = '#4daf4a'
    c_expe  = '#984ea3'

    # ── Style ─────────────────────────────────────────────────────────────
    sns.set_style('whitegrid')
    plt.rcParams.update({
        'font.size':        9,
        'axes.linewidth':   1.0,
        'xtick.direction':  'in',
        'ytick.direction':  'in',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size':  4,
        'ytick.major.size':  4,
        'legend.frameon':   True,
    })

    # Adapt font sizes to figure width
    is_wide = (figwidth > 5)
    AXIS_FONT  = 13 if is_wide else 10
    TICK_FONT  = 10 if is_wide else 7
    LEG_FONT   = 10 if is_wide else 8
    XTICK_FONT = 9  if is_wide else 6.5

    # ── Layout: bars (left, ~70%) | key (right, ~30%) ─────────────────────
    fig_height = 6.5 if is_wide else 5.8
    bar_width_frac = 0.68
    key_width_frac = 1.0 - bar_width_frac

    fig = plt.figure(figsize=(figwidth, fig_height))

    # Outer grid: 1 row, 2 cols  (bars | key)
    outer = gridspec.GridSpec(1, 2, figure=fig,
                              width_ratios=[bar_width_frac, key_width_frac],
                              left=0.09, right=0.9,
                              top=0.96, bottom=0.08,
                              wspace=0.02)

    # Inner grid for bars: 2 rows, 1 col
    inner_bars = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0, 0], hspace=0.30)

    ax_e = fig.add_subplot(inner_bars[0])
    ax_f = fig.add_subplot(inner_bars[1])

    # Key panel spans both rows on the right
    ax_key = fig.add_subplot(outer[0, 1])

    # ── Panel (e): Train vs Calc-val ──────────────────────────────────────
    _env_panel(ax_e, env_train, env_cval,
               f'Train ({len(train_idx)})',
               f'Calc-val. ({len(val_idx)})',
               c_train, c_val,
               panel_label='(e)', y_title='Count',
               AXIS_FONT=AXIS_FONT, TICK_FONT=XTICK_FONT, LEG_FONT=LEG_FONT)
    ax_e.legend(fontsize=LEG_FONT, framealpha=0.85, fancybox=True,
                handlelength=1.0, handletextpad=0.4, borderpad=0.3,
                loc='upper right')
    ax_e.tick_params(axis='y', labelsize=TICK_FONT)
    ax_e.grid(True, axis='y', alpha=0.3, linewidth=0.8, zorder=0)
    ax_e.set_axisbelow(True)

    # ── Panel (f): Exp-val vs Eval ────────────────────────────────────────
    _env_panel(ax_f, env_expv, env_expe,
               f'Exp-val. ({len(exp_val_idx)})',
               f'Eval. ({len(exp_eval_idx)})',
               c_expv, c_expe,
               panel_label='(f)', y_title='Count',
               AXIS_FONT=AXIS_FONT, TICK_FONT=XTICK_FONT, LEG_FONT=LEG_FONT)
    ax_f.legend(fontsize=LEG_FONT, framealpha=0.85, fancybox=True,
                handlelength=1.0, handletextpad=0.4, borderpad=0.3,
                loc='upper right')
    ax_f.tick_params(axis='y', labelsize=TICK_FONT)
    ax_f.grid(True, axis='y', alpha=0.3, linewidth=0.8, zorder=0)
    ax_f.set_axisbelow(True)

    # ── Key panel ─────────────────────────────────────────────────────────
    _draw_key_panel(ax_key, FONT=key_font)

    # ── Save ──────────────────────────────────────────────────────────────
    for ext in ('png', 'pdf'):
        path = os.path.join(PNG_DIR, f'env_distribution_{suffix}.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'  -> {path}')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 60)
    print('  Carbon-Environment Distribution Figure')
    print('=' * 60)

    print('\n  --- \\textwidth  (7.2 in) ---')
    _make_figure(figwidth=7.2, suffix='text', key_font=8)

    print('\n  --- \\columnwidth (3.5 in) ---')
    _make_figure(figwidth=3.5, suffix='col', key_font=6.5)

    print('\n  Done.')
