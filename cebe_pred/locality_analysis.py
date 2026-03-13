#!/usr/bin/env python3
"""
Lightweight locality analysis for CEBE variance decomposition.

Computes the intra-group CEBE spread at each topological (bond) radius
using Morgan fingerprints.  This is the minimal analysis needed to
produce the "residual CEBE std vs bond radius" panel in
``publication_plots.py``.

The heavy lifting:
  1. Load the 113 experimental molecules from the ``.pt`` data file.
  2. For each carbon with an experimental CEBE, compute Morgan FPs at
     radii 1–7.
  3. Deduplicate symmetry-equivalent carbons (same CEBE in same mol).
  4. At each radius, group carbons by fingerprint and compute ANOVA
     variance decomposition (SS_between / SS_total = % explained).

Usage
-----
    from locality_analysis import compute_locality_data
    loc = compute_locality_data()       # dict ready for plotting
"""

import os
import sys
import numpy as np
import torch
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import rdFingerprintGenerator

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH    = os.path.join(PROJECT_ROOT, 'data', 'processed',
                            'gnn_exp_cebe_data.pt')

sys.path.insert(0, PROJECT_ROOT)
from augernet.carbon_environment import IDX_TO_CARBON_ENV

# Morgan fingerprint size (must match build_molecular_graphs.py)
MORGAN_N_BITS = 2048


# ═════════════════════════════════════════════════════════════════════════
#  Build RDKit Mol from .pt graph data
# ═════════════════════════════════════════════════════════════════════════

def _mol_from_pt(collated, slices, mol_idx):
    """Build an RDKit Mol (with explicit H) from the collated .pt data.

    Uses ``atom_symbols`` for element types and ``pos`` for 3-D
    coordinates, then infers bonds via ``DetermineBonds``.

    Returns
    -------
    mol : rdkit.Chem.Mol   (explicit-H, sanitised)
    symbols : list[str]     element symbols in XYZ atom order
    """
    xs = slices['x'][mol_idx].item()
    xe = slices['x'][mol_idx + 1].item()
    ps = slices['pos'][mol_idx].item()
    pe = slices['pos'][mol_idx + 1].item()

    n_atoms = xe - xs
    symbols = collated.atom_symbols[mol_idx]
    pos = collated.pos[ps:pe].numpy()

    # Build bare molecule
    mol = Chem.RWMol()
    for sym in symbols:
        mol.AddAtom(Chem.Atom(sym))
    mol = mol.GetMol()

    # Attach 3-D coordinates
    conf = Chem.Conformer(n_atoms)
    for i in range(n_atoms):
        conf.SetAtomPosition(i, pos[i].tolist())
    mol.RemoveAllConformers()
    mol.AddConformer(conf)

    # Infer bonds from geometry
    mol.ClearComputedProps()
    mol.UpdatePropertyCache(strict=False)
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
    except Exception:
        rdDetermineBonds.DetermineConnectivity(mol)

    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol)
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)

    return mol, symbols


def _morgan_fp_for_atom(mol, atom_idx, radius, n_bits=MORGAN_N_BITS):
    """Hashable Morgan fingerprint (frozenset of bit indices) for one atom."""
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    gen.GetFingerprintAsNumPy(mol, additionalOutput=ao)
    bit_info = ao.GetBitInfoMap()
    atom_bits = set()
    for bit_idx, contributors in bit_info.items():
        for contributing_atom, _r in contributors:
            if contributing_atom == atom_idx:
                atom_bits.add(bit_idx)
    return frozenset(atom_bits)


# ═════════════════════════════════════════════════════════════════════════
#  Load carbon sites
# ═════════════════════════════════════════════════════════════════════════

def _load_carbon_sites(max_radius=7):
    """Load exp. molecules and return one dict per measured carbon site.

    Each dict contains ``mol_name``, ``cebe``, ``env_name``, and
    ``morgan_1`` … ``morgan_{max_radius}`` fingerprint keys.
    """
    collated, slices = torch.load(DATA_PATH, weights_only=False)
    n_mols = len(slices['x']) - 1

    sites = []
    skipped = 0

    for i in range(n_mols):
        xs = slices['x'][i].item()
        xe = slices['x'][i + 1].item()
        name = collated.mol_name[i]

        node_mask = collated.node_mask[xs:xe].tolist()
        true_cebe = collated.true_cebe[xs:xe].tolist()

        cs = slices['carbon_env_labels'][i].item()
        ce = slices['carbon_env_labels'][i + 1].item()
        env_labels = collated.carbon_env_labels[cs:ce].tolist()

        try:
            mol_h, _symbols = _mol_from_pt(collated, slices, i)
        except Exception:
            skipped += 1
            continue

        n_atoms_pt = xe - xs
        if n_atoms_pt != mol_h.GetNumAtoms():
            skipped += 1
            continue

        for j in range(n_atoms_pt):
            if node_mask[j] < 0.5:          # not a carbon
                continue
            cebe = true_cebe[j]
            if cebe < 0:                     # no experimental value
                continue

            eidx = env_labels[j]
            env_name = IDX_TO_CARBON_ENV.get(eidx, f'unknown({eidx})')

            fps = {}
            for r in range(1, max_radius + 1):
                fps[f'morgan_{r}'] = _morgan_fp_for_atom(mol_h, j, r)

            sites.append({
                'mol_name': name,
                'atom_idx': j,
                'cebe':     cebe,
                'env_name': env_name,
                **fps,
            })

    if skipped:
        print(f"  ⚠  Skipped {skipped} molecules (atom-count mismatch)")
    print(f"  Loaded {len(sites)} carbon sites with experimental CEBE values")
    return sites


# ═════════════════════════════════════════════════════════════════════════
#  Dedup + ANOVA
# ═════════════════════════════════════════════════════════════════════════

def _dedup_sites(sites):
    """One representative per unique (molecule, CEBE) pair."""
    seen = set()
    dedup = []
    for s in sites:
        k = (s['mol_name'], round(s['cebe'], 6))
        if k not in seen:
            seen.add(k)
            dedup.append(s)
    return dedup


def _anova_one_radius(dedup_sites, key, grand_mean, SS_total):
    """One-way ANOVA for a single Morgan-FP grouping key."""
    N = len(dedup_sites)
    groups = defaultdict(list)
    for s in dedup_sites:
        groups[s[key]].append(s)

    n_groups     = len(groups)
    n_singletons = sum(1 for v in groups.values() if len(v) == 1)
    n_multi      = n_groups - n_singletons
    n_in_multi   = sum(len(v) for v in groups.values() if len(v) >= 2)

    ss_between = 0.0
    ss_within  = 0.0
    for gsites in groups.values():
        cebes = np.array([s['cebe'] for s in gsites])
        mu_g  = cebes.mean()
        n_g   = len(cebes)
        ss_between += n_g * (mu_g - grand_mean) ** 2
        ss_within  += np.sum((cebes - mu_g) ** 2)

    pct = 100.0 * ss_between / SS_total if SS_total > 0 else 0.0

    # Per multi-member group details (for scatter in plot)
    group_details = []
    intra_stds = []
    group_sites = []            # list of (fp_bits, site_list) per multi-member group
    for fp, gsites in groups.items():
        if len(gsites) < 2:
            continue
        cebes = np.array([s['cebe'] for s in gsites])
        std_g = float(np.std(cebes))
        intra_stds.append(std_g)
        group_details.append((len(gsites), std_g))
        group_sites.append((fp, gsites))

    mean_intra_std    = float(np.mean(intra_stds)) if intra_stds else 0.0
    pooled_residual   = float(np.sqrt(ss_within / N)) if N > 0 else 0.0

    return {
        'pct_explained':      pct,
        'n_groups':           n_groups,
        'n_singletons':       n_singletons,
        'n_multi':            n_multi,
        'n_sites_in_multi':   n_in_multi,
        'mean_intra_std':     mean_intra_std,
        'pooled_residual_std': pooled_residual,
        'group_details':      group_details,
        'group_sites':        group_sites,
    }


# ═════════════════════════════════════════════════════════════════════════
#  Public API
# ═════════════════════════════════════════════════════════════════════════

def compute_locality_data(max_radius=7):
    """Run the full locality analysis and return a dict for plotting.

    Returns
    -------
    dict with keys:
        radii, mean_intra_std, pooled_std, total_std, pct_explained,
        group_details (list-of-lists of (n_carbons, std) per radius),
        n_dedup, n_raw
    """
    print("\n  Computing locality analysis (Morgan FP, radii 1–%d) …"
          % max_radius)
    sites = _load_carbon_sites(max_radius)
    dedup = _dedup_sites(sites)

    all_cebes  = np.array([s['cebe'] for s in dedup])
    grand_mean = float(np.mean(all_cebes))
    SS_total   = float(np.sum((all_cebes - grand_mean) ** 2))
    total_std  = float(np.std(all_cebes))

    radii = list(range(1, max_radius + 1))
    results = [_anova_one_radius(dedup, f'morgan_{r}', grand_mean, SS_total)
               for r in radii]

    # Print summary
    print(f"  {len(sites)} raw → {len(dedup)} dedup carbon sites")
    print(f"  Total CEBE std: {total_std:.3f} eV")
    print(f"\n  {'Radius':>6s}  {'Groups':>6s}  {'Multi':>5s}  "
          f"{'⟨σ_g⟩':>7s}  {'Pooled σ':>8s}  {'% Expl':>8s}")
    print("  " + "-" * 50)
    for r, d in zip(radii, results):
        print(f"  {r:6d}  {d['n_groups']:6d}  {d['n_multi']:5d}  "
              f"{d['mean_intra_std']:7.3f}  "
              f"{d['pooled_residual_std']:8.3f}  "
              f"{d['pct_explained']:7.1f}%")

    return {
        'radii':          radii,
        'mean_intra_std': [d['mean_intra_std']      for d in results],
        'pooled_std':     [d['pooled_residual_std'] for d in results],
        'pct_explained':  [d['pct_explained']       for d in results],
        'group_details':  [d['group_details']       for d in results],
        'group_sites':    [d['group_sites']         for d in results],
        'total_std':      total_std,
        'n_dedup':        len(dedup),
        'n_raw':          len(sites),
    }


# ═════════════════════════════════════════════════════════════════════════
#  Write per-radius group files
# ═════════════════════════════════════════════════════════════════════════

def write_group_files(loc_data, out_dir='locality_analysis'):
    """Write per-radius summary + per-group member lists.

    Creates ``out_dir/radius_{r}/_summary.txt`` and
    ``out_dir/radius_{r}/group_{NNN}.txt`` for every multi-member group.
    """
    os.makedirs(out_dir, exist_ok=True)

    for r, gs_list in zip(loc_data['radii'], loc_data['group_sites']):
        rdir = os.path.join(out_dir, f'radius_{r}')
        os.makedirs(rdir, exist_ok=True)

        # Sort groups by size descending, then by spread descending
        sorted_groups = sorted(
            gs_list,
            key=lambda x: (-len(x[1]),
                           -(max(s['cebe'] for s in x[1])
                             - min(s['cebe'] for s in x[1]))),
        )

        # ── _summary.txt ────────────────────────────────────────────────
        with open(os.path.join(rdir, '_summary.txt'), 'w') as f:
            f.write(f"Topological radius {r} (Morgan FP): "
                    f"{len(sorted_groups)} groups (deduplicated)\n")
            f.write(f" {'Group':>5s}  {'N':>4s}  {'Mols':>4s}  "
                    f"{'Mean CEBE':>10s}  {'Spread':>7s}  {'Std':>7s}  "
                    f"{'#bits':>5s}  Bits (first 10)\n")
            f.write("-" * 90 + "\n")

            for idx, (fp, gsites) in enumerate(sorted_groups, 1):
                cebes = np.array([s['cebe'] for s in gsites])
                n_mols = len(set(s['mol_name'] for s in gsites))
                bits_str = '{' + ','.join(str(b) for b in sorted(fp)[:10]) + '}'
                f.write(f" {idx:5d}  {len(gsites):4d}  {n_mols:4d}  "
                        f"{cebes.mean():10.3f}  {cebes.max()-cebes.min():7.3f}  "
                        f"{np.std(cebes):7.3f}  {len(fp):5d}  {bits_str}\n")

        # ── group_NNN.txt ───────────────────────────────────────────────
        for idx, (fp, gsites) in enumerate(sorted_groups, 1):
            cebes = np.array([s['cebe'] for s in gsites])
            bits_str = '{' + ','.join(str(b) for b in sorted(fp)) + '}'

            with open(os.path.join(rdir, f'group_{idx:03d}.txt'), 'w') as f:
                f.write(f"Topological radius: {r}\n")
                f.write(f"Group {idx}: {len(gsites)} carbons from "
                        f"{len(set(s['mol_name'] for s in gsites))} molecules\n")
                f.write(f"Morgan FP ({len(fp)} bits on): {bits_str}\n")
                f.write(f"Mean CEBE:   {cebes.mean():.3f} eV\n")
                f.write(f"Std CEBE:    {np.std(cebes):.3f} eV\n")
                f.write(f"Spread:      {cebes.max()-cebes.min():.3f} eV\n\n")
                f.write(f"{'Molecule':<30s}  {'Atom':>4s}  {'CEBE (eV)':>10s}  "
                        f"{'Env label':<25s}\n")
                f.write("-" * 75 + "\n")
                for s in sorted(gsites, key=lambda s: s['cebe']):
                    f.write(f"{s['mol_name']:<30s}  {s.get('atom_idx','-'):>4}  "
                            f"{s['cebe']:10.3f}  {s['env_name']:<25s}\n")

    print(f"\n  ✓ Group files written to {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    loc = compute_locality_data()
    write_group_files(loc)
