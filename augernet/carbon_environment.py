"""
Carbon Environment Classification 

This module provides:
SMARTS-based categorical carbon environment classification

"""

import numpy as np

from rdkit import Chem
from typing import List, Tuple
from collections import OrderedDict

# =============================================================================
# CARBON ENVIRONMENT CLASSIFICATION
# =============================================================================

# Define carbon environment categories with SMARTS patterns
# Ordered from most specific to least specific for hierarchical matching
CARBON_ENVIRONMENT_PATTERNS = OrderedDict([
    # Carbonyl-containing groups (most specific first)
    ('C_carboxylic_acid', '[CX3](=O)[OX2H1]'),           # -COOH
    ('C_carboxylate', '[CX3](=O)[O-]'),                   # -COO-
    ('C_ester_carbonyl', '[CX3](=O)[OX2][#6]'),          # -COO-R (carbonyl C)
    ('C_amide_carbonyl', '[CX3](=O)[#7]'),                # -CO-N (carbonyl C bonded to any N)
    ('C_acyl_fluoride', '[CX3](=O)[F]'),           # -COF
    ('C_ketone', '[CX3](=O)([#6])[#6]'),                 # R-CO-R (must have 2 C neighbors)
    # Aldehyde: carbonyl C with =O and at least one H, NOT bonded to C on both sides
    # This matches R-CHO and also H2C=O (formaldehyde)
    ('C_aldehyde', '[CH1,CH2;X3](=O)'),                  # CHO or CH2O carbonyl (aldehyde incl. formaldehyde)
    
    # Nitrogen-containing carbons
    ('C_nitrile', '[CX2]#N'),                            # -C≡N
    ('C_imine', '[CX3]=[#7]'),                            # C=N (any N, incl. charged N+)
    
    # Oxygen-containing carbons (non-carbonyl)
    ('C_ether', '[CX4][OX2][#6]'),                       # C-O-C
    ('C_alcohol', '[CX4][OX2H]'),                        # C-OH
    ('C_ester_alkyl', '[#6][OX2][CX3]=O'),               # R-O-CO- (alkyl C next to O)
    
    # Halogen-containing carbons (check before generic sp3/sp2)
    # NOTE: Only fluorine kept for H,C,N,O,F dataset (removed Cl, Br, I)
    ('C_fluorinated', '[#6][F]'),                        # Any C-F (including CF4)
    
    # Nitrogen-bonded carbons (sp3)
    ('C_amine', '[CX4][NX3;H2,H1,H0;!$(NC=O)]'),        # C-N (not amide)
    
    # Unsaturated carbons
    ('C_alkyne', '[CX2;$([CX2]#[CX2])]'),                 # Either C in C≡C
    ('C_CO2', '[CX2](=[OX1])=[OX1]'),                     # O=C=O  (carbon dioxide)
    ('C_isocyanate', '[CX2](=[OX1])=[NX2]'),              # O=C=N  (isocyanate / isocyanic acid)
    ('C_carbodiimide', '[CX2](=[NX2])=[NX2]'),            # N=C=N  (carbodiimide central C)
    ('C_ketene', '[CX2](=[#6])=[OX1]'),                   # C=C=O  (ketene sp C, also carbon suboxide outer)
    ('C_ketenimine', '[CX2](=[#6])=[NX2]'),               # C=C=N  (ketenimine sp C)
    ('C_allene', '[CX2](=[#6])=[#6]'),                    # C=C=C  (central sp C of allene / cumulene)
    ('C_enol', '[CX3;$([CX3]=[CX3])][OX2H]'),            # Vinyl alcohol C=C-OH (enol)
    ('C_vinyl', '[CX3;$([CX3]=[CX3,CX2])]'),             # Either C in C=C (vinyl/alkene, incl. cumulated terminal)
    
    # Aromatic carbons - use both aromatic (c) and Kekulé ([#6]) patterns
    # to handle cases where aromaticity isn't perceived
    # Order: most specific heteroaromatic patterns first
    ('C_phenol', '[c,C;R][OX2H]'),                       # Ar-OH (ring C with -OH substituent)
    ('C_aryl_ether', '[c,C;R][OX2H0][#6]'),              # Ar-O-R (ring C with ether linkage, not phenol)
    ('C_aryl_amine', '[c,C;R][NX3;H2,H1]'),              # Ar-NH2 or Ar-NHR
    #('C_aryl_halide', '[c,C;R][F,Cl,Br,I]'),             # Ar-X
    ('C_aryl_fluoride', '[c,C;R][F]'),             # Ar-X
    ('C_aryl_nitro', '[c,C;R][NX3+](=O)[O-]'),           # Ar-NO2
    
    # Heteroaromatic carbons (C in ring with O/N as ring members)
    # NOTE: Removed aromatic with S patterns (C_arom_S, C_arom_O_S, C_arom_N_S)
    # Only keeping O and N heteroaromatic patterns for H,C,N,O,F dataset
    ('C_arom_O_N', '[c,C;R;$([#6](~[#8;R])(~[#7;R]))]'), # Aromatic C bonded to both O and N in ring (e.g., oxazole C2)
    
    # C adjacent to single heteroatom in ring
    ('C_arom_O', '[c,C;R;$([#6]~[#8;R]);!$([#6](~[#8;R])(~[#7;R]))]'),  # Aromatic C next to ring O only
    ('C_arom_N', '[c,C;R;$([#6]~[#7;R]);!$([#6](~[#7;R])(~[#8;R]))]'),  # Aromatic C next to ring N only
    
    # Generic aromatic (no heteroatom neighbors in ring, or fallback)
    ('C_aromatic', '[c,$(C1=CC=CC=C1),$(C1=CC=CC=1)]'),  # Aromatic C (carbocyclic or unmatched)
    
    # Aliphatic sp3 carbons (by hydrogen count)
    ('C_methyl', '[CH3,CH4]'),                           # -CH3 or CH4 (methane)
    ('C_methylene', '[CH2X4]'),                          # -CH2-
    ('C_methine', '[CHX4]'),                             # -CH<
    ('C_quaternary', '[CX4H0]'),                         # >C< (no H)
    
    # Fallback categories
   # ('C_sp3', '[CX4]'),                                  # Any sp3 carbon
    #('C_sp2', '[CX3]'),                                  # Any sp2 carbon
    #('C_sp', '[CX2]'),                                   # Any sp carbon
   # ('C_unknown', '[#6]'),                               # Any carbon (catch-all)
])

# Create category name to index mapping
CARBON_ENV_TO_IDX = {name: idx for idx, name in enumerate(CARBON_ENVIRONMENT_PATTERNS.keys())}
IDX_TO_CARBON_ENV = {idx: name for name, idx in CARBON_ENV_TO_IDX.items()}
NUM_CARBON_CATEGORIES = len(CARBON_ENVIRONMENT_PATTERNS)

# Explicit specificity scores for resolving multi-match conflicts.
# Higher score = more specific = wins when a carbon matches multiple patterns.
# This replaces the fragile order-dependent first-match-wins approach.
#
# Design principles:
#   - Carbonyl + substituent patterns are most specific (100-106)
#   - Aromatic + substituent patterns beat generic substituent (90-93)
#   - Heteroaromatic patterns (80-82)
#   - Generic substituent patterns: halogen, amine, O-linked (70-72)
#   - Triple/double bond patterns (60-62)
#   - Generic aromatic fallback (50)
#   - Specific aliphatic by H-count (40-43)
#   - Generic hybridization fallbacks (10-12)
#   - Catch-all (0)
CARBON_ENV_PRIORITY = {
    # Carbonyl-containing: very specific (multiple heavy atoms in pattern)
    'C_carboxylic_acid':  106,
    'C_carboxylate':      105,
    'C_ester_carbonyl':   104,
    'C_amide_carbonyl':   103,
    'C_acyl_fluoride':      102,
    'C_ketone':           101,
    'C_aldehyde':         100,
    
    # Nitrogen sp-bonded: very specific
    'C_nitrile':          95,
    'C_imine':            93,
    
    # Aromatic + substituent: more specific than bare substituent
    'C_phenol':           92,
    'C_aryl_ether':       91,
    'C_aryl_amine':       90,
    'C_aryl_fluoride':      89,
    'C_aryl_nitro':       88,
    
    # Heteroaromatic (C in ring with heteroatom)
    'C_arom_O_N':         82,
    'C_arom_O':           81,
    'C_arom_N':           80,
    
    # Oxygen-linked sp3 (non-carbonyl)
    'C_ester_alkyl':      73,
    'C_ether':            72,
    'C_alcohol':          71,
    
    # Generic halogen / amine on sp3 carbon
    'C_fluorinated':      70,
    'C_amine':            69,
    
    # Unsaturated non-aromatic
    'C_alkyne':           62,
    'C_CO2':              67,
    'C_isocyanate':       66,
    'C_carbodiimide':     66,
    'C_ketene':           65,
    'C_ketenimine':       65,
    'C_allene':           64,
    'C_enol':             63,
    'C_vinyl':            60,
    
    # Generic aromatic fallback (no substituent matched)
    'C_aromatic':         50,
    
    # Aliphatic sp3 by H-count
    'C_methyl':           43,
    'C_methylene':        42,
    'C_methine':          41,
    'C_quaternary':       40,
    
    # Generic hybridization fallbacks
    #'C_sp3':              12,
    #'C_sp2':              11,
    #'C_sp':               10,
    
    # Catch-all
    #'C_unknown':           0,
}


# Pre-compile SMARTS patterns for performance
_COMPILED_PATTERNS = []
for _name, _smarts in CARBON_ENVIRONMENT_PATTERNS.items():
    _pat = Chem.MolFromSmarts(_smarts)
    _COMPILED_PATTERNS.append((_name, _pat, CARBON_ENV_PRIORITY[_name]))


def _get_carbon_environment_label(mol: Chem.Mol, atom_idx: int) -> Tuple[str, int]:
    """
    Determine the carbon environment category for a specific atom.
    
    Uses priority-scored multi-match resolution: all SMARTS patterns are tested
    against the atom, and the most specific matching pattern (highest priority
    score) wins. This avoids order-dependent classification errors where a
    generic pattern (e.g., C_fluorinated) would incorrectly shadow a more
    specific one (e.g., C_aryl_fluoride for aromatic C-F bonds).

    Returns ('non_carbon', -1) for non-carbon atoms.
    
    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object
    atom_idx : int
        Index of the atom to classify
        
    Returns
    -------
    category_name : str
        Name of the carbon environment category
    category_idx : int
        Integer index of the category (-1 for non-carbons)
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    
    # Check if carbon
    if atom.GetAtomicNum() != 6:
        return ('non_carbon', -1)
    
    # Find ALL matching patterns and pick the one with highest priority
    best_name = None
    best_priority = -1
    
    for category_name, pattern, priority in _COMPILED_PATTERNS:
        if pattern is None:
            continue
        
        # Skip patterns with lower priority than what we already found
        if priority <= best_priority:
            continue
            
        matches = mol.GetSubstructMatches(pattern)
        for match in matches:
            # Only match if the atom is at position 0 in the SMARTS match.
            # This ensures we classify the carbon that the pattern is designed
            # to identify, not neighboring atoms that happen to be part of
            # the pattern.
            if len(match) > 0 and match[0] == atom_idx:
                best_name = category_name
                best_priority = priority
                break  # Found a match for this pattern, check next pattern
    
    if best_name is not None:
        return (best_name, CARBON_ENV_TO_IDX[best_name])
    
    # Fallback: should never be reached since C_sp2/C_sp patterns catch all
    # remaining carbons, but guard against edge cases.
    # Return C_sp2 (idx 29) as a safe fallback since it's the broadest
    # remaining pattern.
    return ('C_sp2', CARBON_ENV_TO_IDX['C_sp2'])


def get_all_carbon_environment_labels(mol: Chem.Mol) -> Tuple[List[str], List[int], np.ndarray]:
    """
    Get single-label carbon environment classification for all atoms in a molecule.
    
    Each carbon is classified into exactly ONE category based on hierarchical SMARTS matching.
    Non-carbon atoms receive label index -1.
    
    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object
        
    Returns
    -------
    category_names : List[str]
        Category name for each atom
    category_indices : List[int]
        Category index for each atom (-1 for non-carbons)
    onehot_labels : np.ndarray
        One-hot encoded labels, shape (n_atoms, NUM_CARBON_CATEGORIES)
        All zeros for non-carbon atoms
    """
    n_atoms = mol.GetNumAtoms()
    category_names = []
    category_indices = []
    onehot_labels = np.zeros((n_atoms, NUM_CARBON_CATEGORIES), dtype=np.float32)
    
    for atom_idx in range(n_atoms):
        name, idx = _get_carbon_environment_label(mol, atom_idx)
        category_names.append(name)
        category_indices.append(idx)
        
        if idx >= 0:  # Carbon atom - set single label
            onehot_labels[atom_idx, idx] = 1.0
    
    return category_names, category_indices, onehot_labels


def analyze_carbon_environments(data_list, verbose=False):
    """
    Analyze and print carbon environment distribution across dataset.
    
    Parameters
    ----------
    data_list : list
        List of Data objects
    verbose : bool
        If True, print detailed diversity metrics
    """
    from collections import Counter
    
    if not data_list or len(data_list) == 0:
        print("No data to analyze")
        return
    
    # Collect all carbon environment labels (excluding non-carbons)
    # Use node_mask to distinguish carbons with target values from
    # structurally identified carbons that lack measurements.
    all_labels = []
    n_unmeasured = 0
    for data in data_list:
        labels = data.carbon_env_labels.tolist()
        mask   = data.node_mask.tolist() if hasattr(data, 'node_mask') else None
        for j, l in enumerate(labels):
            if l < 0:
                continue          # not a carbon
            if mask is not None and mask[j] < 0.5:
                n_unmeasured += 1
                continue          # carbon exists but has no target value
            all_labels.append(l)
    
    if not all_labels:
        print("No carbon atoms found in dataset")
        return
    
    # Count occurrences
    label_counts = Counter(all_labels)
    
    # Get category names
    idx_to_name = {v: k for k, v in CARBON_ENV_TO_IDX.items()}
    
    # Sort by count descending
    sorted_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    
    total_carbons = len(all_labels)
    unique_envs = len(label_counts)
    
    unmeasured_note = (f" ({n_unmeasured} additional carbons without target values)"
                       if n_unmeasured > 0 else "")
    print(f"\n  Carbon environments: {total_carbons} carbons, {unique_envs} types, "
          f"{len(data_list)} molecules{unmeasured_note}")
    
    # Compact table (top 10 + others)
    top_n = 10 if not verbose else len(sorted_counts)
    for idx, count in sorted_counts[:top_n]:
        category_name = idx_to_name[idx]
        percentage = 100.0 * count / total_carbons
        print(f"    {category_name:<25} {count:>5} ({percentage:>5.1f}%)")
    
    if len(sorted_counts) > top_n:
        other_count = sum(c for _, c in sorted_counts[top_n:])
        print(f"    {'... others':<25} {other_count:>5} ({100.0*other_count/total_carbons:>5.1f}%)")
    
    if verbose:
        # Diversity metrics
        import math
        max_count = sorted_counts[0][1]
        min_count = sorted_counts[-1][1]
        entropy = -sum((count / total_carbons) * math.log(count / total_carbons) 
                       for _, count in sorted_counts)
        max_entropy = math.log(unique_envs)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        print(f"  Entropy: {normalized_entropy:.3f} ({'balanced' if normalized_entropy > 0.7 else 'skewed'})")
