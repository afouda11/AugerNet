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

5. ``cv_butina_10fold.tex``
     – LaTeX ``table`` with 10-fold Butina CV results, parsed from the
       JS    lines.append(r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccccc@{}}')
    lines.append(r'\toprule')
    lines.append(r'Atom Type & At-BE '
                 r'& E-neg.\ & $d_x$ '
                 r'& Val.\ $\mathcal{L}$ '
                 r'& $\Delta_{\mathrm{G}} \mathcal{L}$ '
                 r'& Exp.\ Val.\ MAE'
                 r'& $\Delta_{\mathrm{G}}$ MAE \\')y produced by the training driver.

6. ``node_feature_ablation.tex``
     – LaTeX ``table*`` with node-feature ablation results (14 configs),
       parsed from the param-search JSON summary.

Usage:
    cd cebe_pred && python publication_plots_si_tables.py
"""

import json
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

# Exp-split molecule lists (hyphenated names, one per line)
_VAL_LIST  = os.path.join(PROJECT_ROOT, 'data', 'raw', 'exp_cebe',
                          'mol_list_val.txt')
_EVAL_LIST = os.path.join(PROJECT_ROOT, 'data', 'raw', 'exp_cebe',
                          'mol_list_eval.txt')


def _load_split_sets():
    """Return ``(val_set, eval_set)`` of hyphenated molecule names."""
    def _read(path):
        if not os.path.exists(path):
            print(f'  ⚠  {path} not found')
            return set()
        with open(path) as f:
            return {line.strip() for line in f if line.strip()}
    return _read(_VAL_LIST), _read(_EVAL_LIST)


def build_cebe_results(labels_paths, tag=''):
    """Build alphabetically-sorted CSV and LaTeX longtable of CEBE results.

    Parameters
    ----------
    labels_paths : str or list[str]
        One or more ``_labels.txt`` paths.  When two are given (val +
        eval), molecules from both are merged.
    tag : str
        Filename suffix for the output files.

    One row per unique (molecule, carbon_env, true_BE) combination.
    A *Split* column marks each molecule as **Val** or **Eval**.
    """
    print('\n' + '=' * 72)
    print('  SI Table 1: CEBE Results (CSV + LaTeX longtable)')
    print('=' * 72)

    if isinstance(labels_paths, str):
        labels_paths = [labels_paths]

    molecules  = _load_molecule_graphs()
    refs, display_names = _load_refs()
    val_set, eval_set = _load_split_sets()

    # Merge label data from all provided labels files
    label_by_name = {}
    for lp in labels_paths:
        for entry in load_label_data(lp):
            label_by_name[entry['name']] = entry

    # ── Collect rows ───────────────────────────────────────────────────────
    raw_blocks = []          # list of (name, pretty_name, formula, split, [row_dicts])
    for mol in molecules:
        name = mol._mol_name              # hyphenated key
        pretty = display_names.get(name, name)  # original pretty name
        syms = mol._atom_symbols
        formula = _molecular_formula(syms)
        lbl = label_by_name.get(name)
        if lbl is None:
            continue

        # Determine split group
        if name in val_set:
            split = 'Val'
        elif name in eval_set:
            split = 'Eval'
        else:
            split = '—'

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
                'split':    split,
            })
            c_idx += 1

        if rows:
            raw_blocks.append((pretty, formula, split, rows))

    # ── Sort blocks alphabetically by molecule name ────────────────────────
    def _sort_key(name):
        """Strip leading numeric / positional prefixes for sorting."""
        s = re.sub(r'^[\d,]+-?', '', name)
        s = re.sub(r'^[mop]-', '', s)
        return s.lower()

    raw_blocks.sort(key=lambda b: _sort_key(b[0]))

    # Flatten: first row of each block gets Name/Formula/Split; rest blank
    flat_rows = []
    for name, formula, split, rows in raw_blocks:
        for i, r in enumerate(rows):
            flat_rows.append({
                'Name':      name if i == 0 else '',
                'formula':   formula if i == 0 else '',
                'split':     split if i == 0 else '',
                'carbon_env': r['env'],
                'neighbors':  r['neighbors'],
                'true_BE_eV': r['true_be'],
                'pred_BE_eV': r['pred_be'],
                'error_eV':   r['error'],
                'cite':       r['cite'],
            })

    # ── Write CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(SI_DIR, f'cebe_results_SI{tag}.csv')
    fields = ['Name', 'formula', 'split', 'carbon_env', 'neighbors',
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

	n_cols = 8
	col_spec = 'lllllrrr'
	header = (r'Name & Formula & Split & Environment & Neighbors '
			  r'& Exp. (eV) & GNN (eV) & Error (eV)')

	lines = []
	lines.append(r'% Requires: \usepackage{longtable, booktabs}')
	lines.append(r'\footnotesize')
	lines.append(r'\setlength{\tabcolsep}{4pt}')
	lines.append(r'\renewcommand{\arraystretch}{0.3}')
	lines.append(r'\begin{longtable}{' + col_spec + r'}')
	lines.append(r'\caption{Predicted C\,1s CEBEs with 3 layer EGNN model using a (Skipatom-200, At-BE, E-neg) node')
	lines.append(r'feature specification and comapred to experimental values. The molecule names, molecular formula, carbon enviroment class,')
	lines.append(r'and nearest neighbor atoms are provided for each CEBE prediction.}')
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
		split = row.get('split', '')
		env  = _esc(row['carbon_env'])
		nbr  = _subscript(row['neighbors'])

		true_str = f"{row['true_BE_eV']:.2f}"
		if row['cite']:
			true_str += f" {row['cite']}"

		pred_str = f"{row['pred_BE_eV']:.2f}"
		err_str  = f"{row['error_eV']:.2f}"

		line = f'{name} & {form} & {split} & {env} & {nbr} & {true_str} & {pred_str} & {err_str} \\\\'

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

    Counts are broken down into four splits:

    * **Train** / **DFT Val** — DFT-calculated data, Butina fold 5
    * **Exp. Val** / **Exp. Eval** — experimental data, per molecule lists
    """
    print('\n' + '=' * 72)
    print('  SI Table 2: Carbon Environment Summary (CSV + LaTeX table*)')
    print('=' * 72)

    # ── DFT calc data: Train / Val via Butina fold 5 ──────────────────────
    calc_train_counts, calc_val_counts = _count_calc_envs_by_fold(
        'gnn_calc_cebe_data.pt', fold=5, n_folds=10, cutoff=0.65,
    )

    # ── Exp data: Val / Eval via molecule lists ───────────────────────────
    val_set, eval_set = _load_split_sets()
    exp_val_counts, exp_eval_counts = _count_exp_envs_by_split(
        'gnn_exp_cebe_data.pt', val_set, eval_set,
    )

    # ── Build rows in priority order (highest first) ───────────────────────
    env_items = sorted(CARBON_ENVIRONMENT_PATTERNS.items(),
                       key=lambda kv: CARBON_ENV_PRIORITY.get(kv[0], 0),
                       reverse=True)

    csv_rows = []
    for env_name, smarts in env_items:
        priority  = CARBON_ENV_PRIORITY.get(env_name, 0)
        pretty    = _pretty_env(env_name)
        csv_rows.append({
            'environment':    pretty,
            'SMARTS':         smarts,
            'priority':       priority,
            'train_count':    calc_train_counts.get(env_name, 0),
            'val_count':      calc_val_counts.get(env_name, 0),
            'exp_val_count':  exp_val_counts.get(env_name, 0),
            'exp_eval_count': exp_eval_counts.get(env_name, 0),
        })

    # ── Write CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(SI_DIR, 'carbon_env_summary.csv')
    fields = ['environment', 'SMARTS', 'priority',
              'train_count', 'val_count', 'exp_val_count', 'exp_eval_count']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(csv_rows)
    print(f'  ✓ CSV  ({len(csv_rows)} rows) → {csv_path}')

    # ── Write LaTeX table* ─────────────────────────────────────────────────
    _write_env_latex(csv_rows)

    return csv_rows


def _count_calc_envs_by_fold(pt_filename, fold=5, n_folds=10, cutoff=0.65):
    """Count carbon-env occurrences in the DFT calc dataset, split by Butina fold.

    Returns
    -------
    train_counts, val_counts : dict, dict
        ``{env_name: int}`` for the training and validation splits.
    """
    from augernet.build_molecular_graphs import get_butina_clusters
    from sklearn.model_selection import GroupKFold

    path = os.path.join(PROJECT_ROOT, 'data', 'processed', pt_filename)
    if not os.path.exists(path):
        print(f'  ⚠  {path} not found – counts will be zero')
        return {}, {}

    collated, slices = torch.load(path, weights_only=False)
    env_labels = collated.carbon_env_labels
    n_mols = len(slices[list(slices.keys())[0]]) - 1

    # Reproduce the Butina fold split
    smiles_list = [collated.smiles[i] for i in range(n_mols)]
    cluster_ids = get_butina_clusters(smiles_list, cutoff=cutoff)
    gkf = GroupKFold(n_splits=n_folds)
    folds = list(gkf.split(np.arange(n_mols), groups=cluster_ids))
    train_idx, val_idx = folds[fold - 1]
    train_set = set(train_idx.tolist() if hasattr(train_idx, 'tolist')
                    else list(train_idx))
    val_set   = set(val_idx.tolist() if hasattr(val_idx, 'tolist')
                    else list(val_idx))

    train_counts, val_counts = {}, {}
    for i in range(n_mols):
        s = slices['x'][i].item()
        e = slices['x'][i + 1].item()
        syms = collated.atom_symbols[i]
        target = train_counts if i in train_set else val_counts
        for local_j, sym in enumerate(syms):
            if sym != 'C':
                continue
            global_idx = s + local_j
            env_idx = env_labels[global_idx].item()
            env_name = IDX_TO_CARBON_ENV.get(env_idx, 'unknown')
            target[env_name] = target.get(env_name, 0) + 1

    print(f'  DFT calc: {sum(train_counts.values())} train C atoms, '
          f'{sum(val_counts.values())} val C atoms  (fold {fold})')
    return train_counts, val_counts


def _count_exp_envs_by_split(pt_filename, val_names, eval_names):
    """Count carbon-env occurrences in the experimental dataset by split.

    Parameters
    ----------
    val_names, eval_names : set of str
        Molecule names for the experimental validation / evaluation splits.

    Returns
    -------
    val_counts, eval_counts : dict, dict
        ``{env_name: int}`` for the exp-val and exp-eval splits.
    """
    path = os.path.join(PROJECT_ROOT, 'data', 'processed', pt_filename)
    if not os.path.exists(path):
        print(f'  ⚠  {path} not found – counts will be zero')
        return {}, {}

    collated, slices = torch.load(path, weights_only=False)
    env_labels = collated.carbon_env_labels
    n_mols = len(slices[list(slices.keys())[0]]) - 1

    val_counts, eval_counts = {}, {}
    for i in range(n_mols):
        mol_name = collated.mol_name[i]
        s = slices['x'][i].item()
        e = slices['x'][i + 1].item()
        syms = collated.atom_symbols[i]

        if mol_name in val_names:
            target = val_counts
        elif mol_name in eval_names:
            target = eval_counts
        else:
            continue  # skip molecules not in either split

        for local_j, sym in enumerate(syms):
            if sym != 'C':
                continue
            global_idx = s + local_j
            env_idx = env_labels[global_idx].item()
            env_name = IDX_TO_CARBON_ENV.get(env_idx, 'unknown')
            target[env_name] = target.get(env_name, 0) + 1

    print(f'  Exp data: {sum(val_counts.values())} val C atoms, '
          f'{sum(eval_counts.values())} eval C atoms')
    return val_counts, eval_counts


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
	lines.append(r'\caption{Carbon environment classes used in this work and ')
	lines.append(r'associated SMARTS patterns. Each environment is assigned a ')
	lines.append(r'priority for the pattern-matching algorithm, which assigns ')
	lines.append(r'a higher priority to more constrained environments.')
	lines.append(r'The counts of carbon atoms in each environment class are ')
	lines.append(r'reported for the calculated training (Train), the calculated validation set (Calc-val), the')
	lines.append(r'experiemntal validation set (Exp-val) and the experiemntal evalution set (Eval)}')
	lines.append(r'\label{tab:carbon_envs}')
	lines.append(r'\footnotesize')
	lines.append(r'\setlength{\tabcolsep}{4pt}')
	lines.append(r'\begin{tabular}{l>{\ttfamily\raggedright\arraybackslash}'
				 r'p{6.5cm}crrrr}')
	lines.append(r'\toprule')
	lines.append(r'Environment & \textnormal{SMARTS} & Priority '
				 r'& Train & Calc-val & Exp-val & Eval \\')
	lines.append(r'\midrule')

	for r in rows:
		name   = _esc_name(r['environment'])
		smarts = _esc_smarts(r['SMARTS'])
		pri    = r['priority']
		tc     = r['train_count']
		vc     = r['val_count']
		evc    = r['exp_val_count']
		eec    = r['exp_eval_count']
		lines.append(
			f'{name} & {smarts} & {pri} '
			f'& {tc:>5d} & {vc:>4d} & {evc:>3d} & {eec:>3d} \\\\'
		)

	lines.append(r'\bottomrule')
	lines.append(r'\end{tabular}')
	lines.append(r'\end{table*}')

	with open(tex_path, 'w') as f:
		f.write('\n'.join(lines) + '\n')
	print(f'  ✓ LaTeX table* → {tex_path}')


# ═══════════════════════════════════════════════════════════════════════════════
#  TABLE 3 — 10-fold Butina CV results: LaTeX table (from JSON)
# ═══════════════════════════════════════════════════════════════════════════════

# Fold to highlight (bold) in the CV table
BEST_FOLD = 5

def build_cv_table(cv_json_path):
    r"""Build ``cv_butina_10fold.tex`` from the CV summary JSON.

    Parameters
    ----------
    cv_json_path : str
        Path to the ``*_cv_summary.json`` produced by the training driver.
    """
    print('\n' + '=' * 72)
    print('  SI Table 3: 10-fold Butina CV (LaTeX table)')
    print('=' * 72)

    with open(cv_json_path) as f:
        summary = json.load(f)

    runs = sorted(summary['runs'], key=lambda r: r['fold'])
    n_folds = summary['n_folds']

    # ── Collect per-fold data ──────────────────────────────────────────────
    folds = []
    train_losses, val_losses, exp_maes = [], [], []
    for r in runs:
        fold = r['fold']
        epochs = r['n_epochs']
        trn = r['best_train_loss']
        val = r['best_val_loss']
        mae = r['eval_mae']
        folds.append((fold, epochs, trn, val, mae))
        train_losses.append(trn)
        val_losses.append(val)
        exp_maes.append(mae)

    mean_trn = np.mean(train_losses)
    mean_val = np.mean(val_losses)
    mean_mae = np.mean(exp_maes)
    std_trn  = np.std(train_losses)
    std_val  = np.std(val_losses)
    std_mae  = np.std(exp_maes)

    # ── Build LaTeX ────────────────────────────────────────────────────────
    lines = []
    lines.append(r'% ' + '─' * 78)
    lines.append(r'%  SI Table: 10-fold Butina cross-validation results')
    lines.append(r'%  Generated by publication_plots_si_tables.py from JSON')
    lines.append(r'% ' + '─' * 78)
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\small')
    lines.append(r'\caption{Ten-fold Butina-clustered cross-validation results for the EGNN')
    lines.append(r'model (3 layers, $d=64$, feature key \texttt{'
                 + summary['feature_tag'] + r'}).  Splitting is based on')
    lines.append(r'Butina clustering of Tanimoto-distance Morgan fingerprints')
    lines.append(r'(cutoff\,=\,0.65, 664 clusters) to ensure that structurally similar')
    lines.append(r'molecules appear in the same fold.  Train and validation losses are MSE;')
    lines.append(r'Exp.\ Val.\ MAE is the mean absolute error on the 63 experimental')
    lines.append(r'validation molecules.  Fold\,' + str(BEST_FOLD)
                 + r' (bold) is selected for final evaluation.}')
    lines.append(r'\label{tab:cv_butina}')
    lines.append(r'\setlength{\tabcolsep}{6pt}')
    lines.append(r'\begin{tabular}{ccccc}')
    lines.append(r'\toprule')
    lines.append(r'Fold & Epochs & Train Loss & Val.\ Loss & Exp.\ Val.\ MAE (eV) \\')
    lines.append(r'\midrule')

    for i, (fold, epochs, trn, val, mae) in enumerate(folds):
        trn_s = f'{trn:.5f}'
        val_s = f'{val:.5f}'
        mae_s = f'{mae:.3f}'
        if fold == BEST_FOLD:
            row = (rf' \textbf{{{fold}}} & \textbf{{{epochs}}} '
                   rf'& \textbf{{{trn_s}}} & \textbf{{{val_s}}} '
                   rf'& \textbf{{{mae_s}}}')
        else:
            row = f'{fold:2d} & {epochs} & {trn_s} & {val_s} & {mae_s}'
        # Add [2pt] spacing except on the last fold row
        suffix = r' \\[2pt]' if i < len(folds) - 1 else r' \\'
        lines.append(row + suffix)

    lines.append(r'\midrule')
    lines.append(f'Mean &     & {mean_trn:.5f} & {mean_val:.5f} '
                 f'& {mean_mae:.3f} \\\\[2pt]')
    lines.append(f'$\\pm$\\,STD &  & $\\pm${std_trn:.4f} '
                 f'& $\\pm${std_val:.4f} & $\\pm${std_mae:.3f} \\\\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    tex_path = os.path.join(SI_DIR, 'cv_butina_10fold.tex')
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  ✓ LaTeX table ({n_folds} folds) → {tex_path}')


# ═══════════════════════════════════════════════════════════════════════════════
#  TABLE 4 — Node-feature ablation: LaTeX table* (from JSON)
# ═══════════════════════════════════════════════════════════════════════════════

# Feature-key → dimension mapping  (must match FEATURE_CATALOG)
_FEAT_DIM = {
    '0': 200,   # skipatom_200
    '1': 30,    # skipatom_30
    '2': 5,     # onehot
    '3': 1,     # atomic_be
    '4': 1,     # mol_be
    '5': 1,     # e_score
    '6': 8,     # env_onehot (approx)
    '7': 256,   # morgan_fp
}

# Which key serves as the "atom-type" embedding
_ATOM_TYPE_KEYS = {
    '0': r'Skip-200',
    '1': r'Skip-30',
    '2': r'One-hot',
}

# Grouping order: which atom-type key group the feature_keys string belongs to
_GROUP_ORDER = ['0', '1', '2', None]   # None = no atom-type embedding


def _classify_config(feature_keys_str):
    """Classify a feature_keys string into (atom_type_key, has_be, has_eneg).

    Returns
    -------
    atom_type_key : str or None
        '0', '1', '2', or None.
    has_be : bool
        Whether key '3' (atomic_be) is present.
    has_eneg : bool
        Whether key '5' (e_score) is present.
    d_x : int
        Total node-feature dimension.
    """
    keys = list(feature_keys_str)  # e.g. '035' → ['0', '3', '5']
    atom_type_key = None
    for k in ('0', '1', '2'):
        if k in keys:
            atom_type_key = k
            break
    has_be   = '3' in keys
    has_eneg = '5' in keys
    d_x = sum(_FEAT_DIM.get(k, 0) for k in keys)
    return atom_type_key, has_be, has_eneg, d_x


def build_feature_ablation_table(param_json_path):
	r"""Build ``node_feature_ablation.tex`` from the param-search JSON.

	Parameters
	----------
	param_json_path : str
		Path to the ``*_param_summary.json`` produced by the training driver.
	"""
	print('\n' + '=' * 72)
	print('  SI Table 4: Node-feature ablation (LaTeX table*)')
	print('=' * 72)

	with open(param_json_path) as f:
		summary = json.load(f)

	runs = summary['runs']   # already sorted by rank in JSON

# ── Classify each run ──────────────────────────────────────────────────
	classified = []  # (atom_type_key, has_be, has_eneg, d_x, val_loss, mae)
	for r in runs:
		fk = r['feature_keys']
		atk, has_be, has_eneg, d_x = _classify_config(fk)
		classified.append((atk, has_be, has_eneg, d_x,
						   r['best_val_loss'], r['eval_mae']))

# ── Compute global best values ───────────────────────────────────────
	all_val_losses = [c[4] for c in classified]
	all_maes       = [c[5] for c in classified]
	global_best_loss = min(all_val_losses)
	global_best_mae  = min(all_maes)

# ── Group by atom-type key, with consistent sub-ordering ───────────────
#    Within each group: both → be-only → eneg-only → neither
	def _sub_sort_key(item):
		_, has_be, has_eneg, *_ = item
		return (not (has_be and has_eneg),     # both first
				not (has_be and not has_eneg),  # be-only second
				not (not has_be and has_eneg),  # eneg-only third
				)

	groups = []
	for gk in _GROUP_ORDER:
		members = [c for c in classified if c[0] == gk]
		if not members:
			continue
		members.sort(key=_sub_sort_key)
		groups.append((gk, members))

# ── Build LaTeX ────────────────────────────────────────────────────────
	lines = []
	lines.append(r'% ' + '─' * 78)
	lines.append(r'%  SI Table: Node-feature ablation study')
	lines.append(r'%  Generated by publication_plots_si_tables.py from JSON')
	lines.append(r'% ' + '─' * 78)
	lines.append(r'\begin{table*}[htbp]')
	lines.append(r'\centering')
	lines.append(r'\caption{Effect of node-feature specifications on a 3 layer EGNN model.')
	lines.append(r'The atom-type column indicates the atom-type representation;')
	lines.append(r'At-BE is the graph-normalized atomic 1s binding energy ($\in \mathbb{R}^{1}$)')
	lines.append(r'and E-neg is the graph-normalized environment electronegativity')
	lines.append(r'($\in \mathbb{R}^{1}$).')
	lines.append(r'$d_x$ denotes the total node-feature dimension.')
	lines.append(r'Calc-val $\mathcal{L}$ is the best MSE reached with patience of 30 epochs and the calculated validation dataset;')
	lines.append(r'Exp-val MAE is the mean absolute error on the 63 experimental')
	lines.append(r'validation molecules in eV.')
	lines.append(r'The $\Delta$ columns give the relative difference of the Calc-val $\mathcal{L}$')
	lines.append(r'and Exp-val MAE metrics.}')
	lines.append(r'\label{tab:nodefeat}')
	lines.append(r'\small')
	lines.append(r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccccc@{}}')
	lines.append(r'\toprule')
	lines.append(r'Atom Type & At-BE '
				 r'& E-neg & $d_n$ '
				 r'& Calc-val $\mathcal{L}$ '
				 r'& $\Delta \mathcal{L}$ '
				 r'& Exp-val MAE (eV) '
				 r'& $\Delta $ MAE (eV) \\')
	lines.append(r'\midrule')

	for g_idx, (gk, members) in enumerate(groups):
		group_label = _ATOM_TYPE_KEYS.get(gk, 'None') if gk else 'None'
		lines.append(f'%% --- {group_label} group ---')
		for m_idx, (atk, has_be, has_eneg, d_x, val_loss, mae) in enumerate(members):
			atom_str = _ATOM_TYPE_KEYS.get(atk, 'None') if atk else 'None'
			be_str   = r'\checkmark' if has_be else ''
			en_str   = r'\checkmark' if has_eneg else ''
			val_s    = f'{val_loss:.3f}'
			mae_s    = f'{mae:.3f}'

			# Global relative (difference from global best)
			dg_loss = val_loss - global_best_loss
			dg_mae  = mae - global_best_mae

			# Format: show 0 for the best, +X.XXX otherwise
			def _fmt_delta(v):
				if abs(v) < 5e-4:
					return '0'
				return f'{v:.3f}'

			dg_loss_s   = _fmt_delta(dg_loss)
			dg_mae_s    = _fmt_delta(dg_mae)

			# [2pt] spacing except last row of group
			suffix = r' \\[2pt]' if m_idx < len(members) - 1 else r' \\'
			lines.append(
				f'{atom_str} & {be_str} & {en_str} & {d_x:>3d} '
				f'& {val_s} & {dg_loss_s} '
				f'& {mae_s} & {dg_mae_s}{suffix}'
			)
		# \midrule between groups (not after the last one)
		if g_idx < len(groups) - 1:
			lines.append(r'\midrule')

	lines.append(r'\bottomrule')
	lines.append(r'\end{tabular*}')
	lines.append(r'\end{table*}')

	tex_path = os.path.join(SI_DIR, 'node_features.tex')
	with open(tex_path, 'w') as f:
		f.write('\n'.join(lines) + '\n')
	n_configs = sum(len(m) for _, m in groups)
	print(f'  ✓ LaTeX table* ({n_configs} configs) → {tex_path}')


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    model = 'cebe_035_butina_EQ3_h64_fold5'
    labels_val  = os.path.join(SCRIPT_DIR, 'train_results', 'outputs',
                               f'expval_{model}_labels.txt')
    labels_eval = os.path.join(SCRIPT_DIR, 'train_results', 'outputs',
                               f'expeval_{model}_labels.txt')
    tag = f'_{model}'

    build_cebe_results([labels_val, labels_eval], tag=tag)
    build_env_summary()

    # ── CV & feature-ablation tables (from JSON summaries) ─────────────
    cv_json = os.path.join(SCRIPT_DIR, 'cv_results',
                           'cebe_035_butina_EQ3_h64_cv_summary.json')
    param_json = os.path.join(
        SCRIPT_DIR, 'param_results',
        'search_feature_keys14_layer_type1_cebe_035_butina_EQ3_h64'
        '_param_summary.json')

    if os.path.exists(cv_json):
        build_cv_table(cv_json)
    else:
        print(f'  ⚠  CV JSON not found: {cv_json}')

    if os.path.exists(param_json):
        build_feature_ablation_table(param_json)
    else:
        print(f'  ⚠  Param JSON not found: {param_json}')

    print('\n  ✅  All SI tables generated.')

