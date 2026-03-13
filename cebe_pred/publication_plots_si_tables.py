#!/usr/bin/env python3
"""
Generate SI tables for the CEBE GNN paper.

Produces (all written to ``si_tables/``):

1. ``cebe_results_SI{tag}.csv``
     – CSV with unique (carbon_env, true_BE) rows per molecule,
       alphabetically sorted, with experimental references.

2. ``cebe_results_SI_table{tag}.tex``
     – LaTeX ``longtable`` version of the same data (7 columns:
       Name, Formula, Environment, Neighbors, Exp., GNN, Error).

3. ``carbon_env_summary.csv``
     – Carbon-environment classification summary: environment name,
       SMARTS, priority, calc-training count, exp-eval count.

4. ``carbon_env_summary_table.tex``
     – LaTeX ``table*`` with the same information, formatted for SI.

Usage:
    cd cebe_pred && python publication_plots_si_tables.py
"""

import os
import re
import csv
import numpy as np
import torch
from collections import Counter

from torch_geometric.data import Data
from augernet.carbon_environment import (
    IDX_TO_CARBON_ENV,
    CARBON_ENVIRONMENT_PATTERNS,
    CARBON_ENV_PRIORITY,
)

from publication_plots import load_label_data

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SI_DIR       = os.path.join(SCRIPT_DIR, 'si_tables')
os.makedirs(SI_DIR, exist_ok=True)

# BibTeX key mapping (Ref-key → citation key)
BIBTEX_KEYS = {
    'Jolly':  'jolly_core-electron_1984',
    'xps_1':  'ganguly_coincidence_2022',
    'xps_2':  'hitchcock_carbon_1986',
    'xps_3':  'plekan_experimental_2020',
    'xps_4':  'travnikova_esca_2012',
    'xps_5':  'naves_de_brito_experimental_1991',
    'xps_6':  'vall-llosera_c_2008',
    'xps_7':  'abid_electronion_2020',
}

# Human-readable environment names (strip ``C_`` prefix, replace ``_`` → space)
def _pretty_env(name):
    """``'C_aryl_ether'`` → ``'aryl ether'``."""
    return name.replace('C_', '', 1).replace('_', ' ')


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loaders
# ═══════════════════════════════════════════════════════════════════════════════

def _load_molecule_graphs():
    """Return per-molecule ``Data`` objects from the processed eval dataset."""
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
        d._mol_name  = collated.mol_name[i]
        d._atom_symbols = collated.atom_symbols[i]
        molecules.append(d)
    return molecules

def _load_refs():
    r"""Parse ``si_tables/refs_per_carbon.txt`` → nested dict.

    The file uses **slug** names (column 0) that match the ``.pt`` data
    directly, plus a human-readable **pretty** name (column 1).

    Returns
    -------
    refs : dict
        ``refs[slug][carbon_env]`` = raw ref string.
    display_names : dict
        ``display_names[slug]`` = pretty name (commas, parentheses, spaces).
    """
    refs_file = os.path.join(SI_DIR, 'refs_per_carbon.txt')
    refs = {}
    display_names = {}
    if not os.path.exists(refs_file):
        print(f'  ⚠  {refs_file} not found – references will be empty')
        return refs, display_names

    cur_slug = None
    with open(refs_file, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('ID'):          # header
                continue
            parts = line.split('\t')
            if len(parts) < 6:
                continue
            slug     = parts[0].strip()
            pretty   = parts[1].strip()
            env      = parts[3].strip()
            ref      = parts[5].strip()

            if slug:                           # first row of a molecule
                cur_slug = slug
                display_names[cur_slug] = pretty
            if cur_slug is None:
                continue
            refs.setdefault(cur_slug, {})
            if env and env not in refs[cur_slug]:
                refs[cur_slug][env] = ref
    return refs, display_names


def _ref_to_cite(ref_str):
    r"""Convert a raw ref string to a ``\cite{…}`` fragment (or ``''``)."""
    if not ref_str:
        return ''
    low = ref_str.strip().lower()
    if low.startswith('jolly'):
        return r'\cite{jolly_core-electron_1984}'
    if low.startswith('xps'):
        m = re.search(r'(\d+)', low)
        if m:
            key = f'xps_{m.group(1)}'
            bib = BIBTEX_KEYS.get(key)
            if bib:
                return rf'\cite{{{bib}}}'
    return ''


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper: neighbor string & molecular formula
# ═══════════════════════════════════════════════════════════════════════════════

def _neighbor_string(mol, atom_idx, atom_symbols):
    """Compact, sorted neighbor-symbol string for *atom_idx*.

    E.g. ``'C2H'`` for an aromatic carbon in benzene.
    """
    ei = mol.edge_index
    nbr_ids = sorted(set(ei[1][ei[0] == atom_idx].tolist()))
    nbr_syms = [atom_symbols[n] for n in nbr_ids if n < len(atom_symbols)]
    counts = Counter(nbr_syms)
    return ''.join(f'{s}{c}' if c > 1 else s
                   for s, c in sorted(counts.items()))


def _molecular_formula(atom_symbols):
    """Hill-order molecular formula from a list of atom symbols."""
    counts = Counter(atom_symbols)
    parts = []
    for elem in ('C', 'H'):
        if elem in counts:
            cnt = counts.pop(elem)
            parts.append(f'{elem}{cnt}' if cnt > 1 else elem)
    for elem in sorted(counts):
        cnt = counts[elem]
        parts.append(f'{elem}{cnt}' if cnt > 1 else elem)
    return ''.join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  TABLE 1 — CEBE results: CSV + LaTeX longtable
# ═══════════════════════════════════════════════════════════════════════════════

def build_cebe_results(labels_path, tag=''):
    """Build alphabetically-sorted CSV and LaTeX longtable of CEBE results.

    One row per unique (molecule, carbon_env, true_BE) combination.
    """
    print('\n' + '=' * 72)
    print('  SI Table 1: CEBE Results (CSV + LaTeX longtable)')
    print('=' * 72)

    molecules  = _load_molecule_graphs()
    label_mols = load_label_data(labels_path)
    refs, display_names = _load_refs()

    # Index label data by molecule name (hyphenated, as stored in labels file)
    label_by_name = {}
    for entry in label_mols:
        label_by_name[entry['name']] = entry

    # ── Collect rows ───────────────────────────────────────────────────────
    raw_blocks = []          # list of (name, pretty_name, formula, [row_dicts])
    for mol in molecules:
        name = mol._mol_name              # hyphenated key
        pretty = display_names.get(name, name)  # original pretty name
        syms = mol._atom_symbols
        formula = _molecular_formula(syms)
        lbl = label_by_name.get(name)
        if lbl is None:
            continue

        env_labels = mol.carbon_env_labels
        true_cebe  = mol.true_cebe

        seen = set()         # (env_name, true_BE_rounded)
        rows = []
        c_idx = 0            # carbon counter into label arrays
        for a_idx, sym in enumerate(syms):
            if sym != 'C':
                continue
            env_idx  = env_labels[a_idx].item()
            env_name = _pretty_env(IDX_TO_CARBON_ENV[env_idx])
            true_be  = round(true_cebe[a_idx].item(), 2)

            # Skip sentinel values (no experimental reference for this C)
            if true_be <= 0:
                c_idx += 1
                continue

            nbr      = _neighbor_string(mol, a_idx, syms)

            key = (env_name, true_be)
            if key in seen:
                c_idx += 1
                continue
            seen.add(key)

            pred_be = round(lbl['pred'][c_idx], 2)
            err     = round(pred_be - true_be, 2)

            # Lookup ref
            mol_refs = refs.get(name, {})
            env_key_underscore = IDX_TO_CARBON_ENV[env_idx].replace('C_', '', 1)
            ref_raw = mol_refs.get(env_key_underscore, '')
            cite    = _ref_to_cite(ref_raw)

            rows.append({
                'name':     pretty,
                'formula':  formula,
                'env':      env_name,
                'neighbors': nbr,
                'true_be':  true_be,
                'pred_be':  pred_be,
                'error':    err,
                'cite':     cite,
            })
            c_idx += 1

        if rows:
            raw_blocks.append((pretty, formula, rows))

    # ── Sort blocks alphabetically by molecule name ────────────────────────
    def _sort_key(name):
        """Strip leading numeric / positional prefixes for sorting."""
        s = re.sub(r'^[\d,]+-?', '', name)
        s = re.sub(r'^[mop]-', '', s)
        return s.lower()

    raw_blocks.sort(key=lambda b: _sort_key(b[0]))

    # Flatten: first row of each block gets Name/Formula; rest are blank
    flat_rows = []
    for name, formula, rows in raw_blocks:
        for i, r in enumerate(rows):
            flat_rows.append({
                'Name':      name if i == 0 else '',
                'formula':   formula if i == 0 else '',
                'carbon_env': r['env'],
                'neighbors':  r['neighbors'],
                'true_BE_eV': r['true_be'],
                'pred_BE_eV': r['pred_be'],
                'error_eV':   r['error'],
                'cite':       r['cite'],
            })

    # ── Write CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(SI_DIR, f'cebe_results_SI{tag}.csv')
    fields = ['Name', 'formula', 'carbon_env', 'neighbors',
              'true_BE_eV', 'pred_BE_eV', 'error_eV', 'cite']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(flat_rows)
    print(f'  ✓ CSV  ({len(flat_rows)} rows) → {csv_path}')

    # ── Write LaTeX longtable ──────────────────────────────────────────────
    _write_cebe_longtable(flat_rows, tag)

    return flat_rows


def _write_cebe_longtable(rows, tag=''):
    r"""Write a ``longtable`` for the SI from flattened row dicts."""
    tex_path = os.path.join(SI_DIR, f'cebe_results_SI_table{tag}.tex')

    def _esc(s):
        """Escape LaTeX special characters in plain text."""
        for ch in ('&', '%', '#', '_'):
            s = s.replace(ch, f'\\{ch}')
        return s

    def _subscript(formula):
        """``'C8H12N2'`` → ``'C$_{8}$H$_{12}$N$_{2}$'``."""
        return re.sub(r'([A-Z][a-z]?)(\d+)',
                      lambda m: f'{m.group(1)}$_{{{m.group(2)}}}$',
                      formula)

    n_cols = 7
    col_spec = 'llllrrr'
    header = (r'Name & Formula & Environment & Neighbors '
              r'& Exp. (eV) & GNN (eV) & Error (eV)')

    lines = []
    lines.append(r'% Requires: \usepackage{longtable, booktabs}')
    lines.append(r'\footnotesize')
    lines.append(r'\setlength{\tabcolsep}{4pt}')
    lines.append(r'\renewcommand{\arraystretch}{0.3}')
    lines.append(r'\begin{longtable}{' + col_spec + r'}')
    lines.append(r'\caption{GNN-predicted C\,1s core-electron binding energies '
                 r'compared with experimental values, grouped by molecule.}')
    lines.append(r'\label{tab:cebe_results_SI} \\')
    lines.append(r'\toprule')
    lines.append(header + r' \\')
    lines.append(r'\midrule')
    lines.append(r'\endfirsthead')
    lines.append('')
    lines.append(rf'\multicolumn{{{n_cols}}}{{l}}'
                 r'{\tablename\ \thetable\ -- \textit{Continued from previous page}} \\')
    lines.append(r'\toprule')
    lines.append(header + r' \\')
    lines.append(r'\midrule')
    lines.append(r'\endhead')
    lines.append('')
    lines.append(rf'\multicolumn{{{n_cols}}}{{r}}'
                 r'{\textit{Continued on next page}}')
    lines.append(r'\endfoot')
    lines.append('')
    lines.append(r'\bottomrule')
    lines.append(r'\endlastfoot')
    lines.append('')

    for i, row in enumerate(rows):
        name = _esc(row['Name'])
        form = _subscript(row['formula']) if row['formula'] else ''
        env  = _esc(row['carbon_env'])
        nbr  = _subscript(row['neighbors'])

        true_str = f"{row['true_BE_eV']:.2f}"
        if row['cite']:
            true_str += f" {row['cite']}"

        pred_str = f"{row['pred_BE_eV']:.2f}"
        err_str  = f"{row['error_eV']:.2f}"

        line = f'{name} & {form} & {env} & {nbr} & {true_str} & {pred_str} & {err_str} \\\\'

        # Insert \midrule between molecule blocks
        if i > 0 and row['Name']:
            lines.append(r'\midrule')
        lines.append(line)

    lines.append(r'\end{longtable}')

    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  ✓ LaTeX longtable → {tex_path}')


# ═══════════════════════════════════════════════════════════════════════════════
#  TABLE 2 — Carbon-environment classification summary: CSV + LaTeX table*
# ═══════════════════════════════════════════════════════════════════════════════

def build_env_summary():
    """Build carbon-environment summary CSV and LaTeX ``table*``.

    Calc counts come from the DFT training+validation dataset
    (``calc_cebe_data.pt``); Exp counts come from the experimental
    evaluation dataset (``gnn_exp_cebe_data.pt``).
    """
    print('\n' + '=' * 72)
    print('  SI Table 2: Carbon Environment Summary (CSV + LaTeX table*)')
    print('=' * 72)

    # ── Count environments in processed datasets ───────────────────────────
    calc_counts = _count_envs('gnn_calc_cebe_data.pt')
    exp_counts  = _count_envs('gnn_exp_cebe_data.pt')

    # ── Build rows in priority order (highest first) ───────────────────────
    env_items = sorted(CARBON_ENVIRONMENT_PATTERNS.items(),
                       key=lambda kv: CARBON_ENV_PRIORITY.get(kv[0], 0),
                       reverse=True)

    csv_rows = []
    for env_name, smarts in env_items:
        priority  = CARBON_ENV_PRIORITY.get(env_name, 0)
        pretty    = _pretty_env(env_name)
        c_count   = calc_counts.get(env_name, 0)
        e_count   = exp_counts.get(env_name, 0)
        csv_rows.append({
            'environment': pretty,
            'SMARTS':      smarts,
            'priority':    priority,
            'calc_count':  c_count,
            'exp_count':   e_count,
        })

    # ── Write CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(SI_DIR, 'carbon_env_summary.csv')
    fields = ['environment', 'SMARTS', 'priority', 'calc_count', 'exp_count']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(csv_rows)
    print(f'  ✓ CSV  ({len(csv_rows)} rows) → {csv_path}')

    # ── Write LaTeX table* ─────────────────────────────────────────────────
    _write_env_latex(csv_rows)

    return csv_rows


def _count_envs(pt_filename):
    """Count carbon-environment occurrences in a processed ``.pt`` file."""
    path = os.path.join(PROJECT_ROOT, 'data', 'processed', pt_filename)
    if not os.path.exists(path):
        print(f'  ⚠  {path} not found – counts will be zero')
        return {}
    collated, slices = torch.load(path, weights_only=False)
    env_labels = collated.carbon_env_labels
    atom_symbols_flat = []
    n_mols = len(slices[list(slices.keys())[0]]) - 1
    for i in range(n_mols):
        s = slices['x'][i].item()
        e = slices['x'][i + 1].item()
        atom_symbols_flat.extend(collated.atom_symbols[i])

    counts = {}
    for idx, sym in enumerate(atom_symbols_flat):
        if sym != 'C':
            continue
        env_idx = env_labels[idx].item()
        env_name = IDX_TO_CARBON_ENV.get(env_idx, 'unknown')
        counts[env_name] = counts.get(env_name, 0) + 1
    return counts


def _write_env_latex(rows):
    r"""Write the carbon-environment summary as a LaTeX ``table*``."""
    tex_path = os.path.join(SI_DIR, 'carbon_env_summary_table.tex')

    def _esc_smarts(s):
        r"""Escape SMARTS for LaTeX ``\ttfamily`` column."""
        s = s.replace('#', r'\#')
        s = s.replace('$', r'\$')
        # Wrap very long SMARTS with a \newline hint if > 40 chars
        if len(s) > 45:
            # Try to break at a semicolon or closing bracket
            for brk in (';!', ';', '])'):
                pos = s.find(brk, 20)
                if pos > 0:
                    pos += len(brk)
                    s = s[:pos] + r'\newline ' + s[pos:]
                    break
        return s

    def _esc_name(name):
        """Format environment name for LaTeX (handle subscripts like CO₂)."""
        # CO2 → CO\textsubscript{2}
        name = re.sub(r'CO2', r'CO\\textsubscript{2}', name)
        return name

    lines = []
    lines.append(r'\begin{table*}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Carbon environment classes used in this work and '
                 r'associated SMARTS patterns. Each environment is assigned a '
                 r'priority for the pattern matching algorithm, which assigns '
                 r'a higher priority to more specific environments. The counts '
                 r'of each environment class in the training (including '
                 r'validation) and experimental evaluation datasets are given.}')
    lines.append(r'\label{tab:carbon_envs}')
    lines.append(r'\footnotesize')
    lines.append(r'\setlength{\tabcolsep}{4pt}')
    lines.append(r'\begin{tabular}{l>{\ttfamily\raggedright\arraybackslash}'
                 r'p{7.0cm}crr}')
    lines.append(r'\toprule')
    lines.append(r'Environment & \textnormal{SMARTS} & Priority '
                 r'& Calc.\ Count & Exp.\ Count \\')
    lines.append(r'\midrule')

    for r in rows:
        name   = _esc_name(r['environment'])
        smarts = _esc_smarts(r['SMARTS'])
        pri    = r['priority']
        cc     = r['calc_count']
        ec     = r['exp_count']
        lines.append(f'{name} & {smarts} & {pri} & {cc:>4d} & {ec:>3d} \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table*}')

    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  ✓ LaTeX table* → {tex_path}')


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    model = 'cebe_035_random_EQ4_h32_fold1'
    labels_path = f'train_results/outputs/{model}_labels.txt'
    tag = f'_{model}'

    build_cebe_results(labels_path, tag=tag)
    build_env_summary()

    print('\n  ✅  All SI tables generated.')

