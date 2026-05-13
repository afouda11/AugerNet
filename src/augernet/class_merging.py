"""
Carbon Environment Class Merging Schemes
=========================================

Provides configurable class merging for the 36-class carbon environment
taxonomy defined in carbon_environment.py.  Merging reduces the number of
classes to improve CNN classification accuracy by grouping spectrally and
chemically similar environments.

Three schemes are available:

  'none'          -- No merging (original 36 classes, ~33 active)
  'chemical'  -- 36 -> 16 classes, chosen for chemical similarity and well defined shapes

Usage:
    from augernet.class_merging import apply_label_merging, get_merged_class_names

    # In training script, after loading the DataFrame:
    df = apply_label_merging(df, scheme='practical')
    num_classes = df['carbon_env_label'].nunique()  # or max+1

    # For display:
    merged_names = get_merged_class_names(scheme='practical')
    # merged_names[new_label_idx] -> human-readable name
"""

from collections import OrderedDict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from augernet.carbon_environment import (
    CARBON_ENVIRONMENT_PATTERNS,
    CARBON_ENV_TO_IDX,
    IDX_TO_CARBON_ENV,
)

# Original class name -> original index (for reference)
_ORIG_NAMES = list(CARBON_ENVIRONMENT_PATTERNS.keys())
  # e.g. 'C_methyl' -> 32


# =============================================================================
#  SCHEME DEFINITIONS
# =============================================================================
# Each scheme maps: merged_class_name -> list of original C_ class names.
# The order of keys defines the new label indices (0, 1, 2, ...).

MERGING_SCHEMES: Dict[str, OrderedDict] = {}

MERGING_SCHEMES['heteroaromatic_only'] = OrderedDict([
    ('heteroaromatic',      ['C_arom_N', 'C_arom_O', 'C_arom_O_N']),
    # everything else: identity map, one-to-one
    ('carboxylic_acid',     ['C_carboxylic_acid']),
    ('carboxylate',         ['C_carboxylate']),
    ('ester_carbonyl',      ['C_ester_carbonyl']),
    ('amide_carbonyl',      ['C_amide_carbonyl']),
    ('acyl_fluoride',       ['C_acyl_fluoride']),
    ('ketone',              ['C_ketone']),
    ('aldehyde',            ['C_aldehyde']),
    ('nitrile',             ['C_nitrile']),
    ('imine',               ['C_imine']),
    ('ether',               ['C_ether']),
    ('alcohol',             ['C_alcohol']),
    ('ester_alkyl',         ['C_ester_alkyl']),
    ('fluorinated',         ['C_fluorinated']),
    ('amine',               ['C_amine']),
    ('alkyne',              ['C_alkyne']),
    ('CO2',                 ['C_CO2']),
    ('isocyanate',          ['C_isocyanate']),
    ('carbodiimide',        ['C_carbodiimide']),
    ('ketene',              ['C_ketene']),
    ('ketenimine',          ['C_ketenimine']),
    ('allene',              ['C_allene']),
    ('enol',                ['C_enol']),
    ('vinyl',               ['C_vinyl']),
    ('phenol',              ['C_phenol']),
    ('aryl_ether',          ['C_aryl_ether']),
    ('aryl_amine',          ['C_aryl_amine']),
    ('aryl_fluoride',       ['C_aryl_fluoride']),
    ('aryl_nitro',          ['C_aryl_nitro']),
    ('aromatic',            ['C_aromatic']),
    ('methyl',              ['C_methyl']),
    ('methylene',           ['C_methylene']),
    ('methine',             ['C_methine']),
    ('quaternary',          ['C_quaternary']),
])


# ------------------------------------------------------------------------------
#  CHEMICAL (36 -> 16)  
# ------------------------------------------------------------------------------
MERGING_SCHEMES['chemical'] = OrderedDict([
    ('heteroaromatic',      ['C_arom_N', 'C_arom_O', 'C_arom_O_N']),
    ('aryl_N',              ['C_aryl_amine', 'C_aryl_nitro']),
    ('aryl_O',              ['C_phenol', 'C_aryl_ether']),
    ('aryl_F',              ['C_aryl_fluoride']),
    ('hydrocarbon',         ['C_methyl', 'C_methylene', 'C_methine', 'C_quaternary',
                                'C_alkyne', 'C_vinyl', 'C_allene', 'C_aromatic']),
    ('carbonyl',            ['C_ketone', 'C_aldehyde', 'C_ester_carbonyl', 'C_ester_alkyl', 
                                'C_carboxylic_acid', 'C_carboxylate']),
    ('amide_carbonyl',      ['C_amide_carbonyl']),
    ('nitrile',             ['C_nitrile']),
    ('imine',               ['C_imine']),
    ('C_O_single',          ['C_ether', 'C_alcohol', 'C_enol']),
    ('C_N_single',          ['C_amine']),
    ('alkyl_fluorinated',   ['C_fluorinated', 'C_acyl_fluoride']),
    ('cumulated_N',         ['C_carbodiimide', 'C_ketenimine']),
    ('cumulated_O',         ['C_ketene', 'C_CO2']),
    ('isocyanate',          ['C_isocyanate']),
])

MERGING_SCHEMES['heteroatom'] = OrderedDict([
    # C=O containing (8 classes)
    ('carbonyl',        ['C_ketone', 'C_aldehyde', 'C_ester_carbonyl', 'C_amide_carbonyl',
                         'C_carboxylic_acid', 'C_carboxylate', 'C_CO2', 'C_ketene']),
    # C-O single bond / O-substituted (6 classes)
    ('oxygen_single',   ['C_ether', 'C_alcohol', 'C_ester_alkyl', 'C_phenol',
                         'C_enol', 'C_aryl_ether']),
    # N-containing: all kinds (10 classes)
    ('nitrogen',        ['C_nitrile', 'C_imine', 'C_amine', 'C_aryl_amine',
                         'C_aryl_nitro', 'C_arom_N', 'C_arom_O_N',
                         'C_isocyanate', 'C_carbodiimide', 'C_ketenimine']),
    # Fluorinated (3 classes)
    ('halogen',         ['C_fluorinated', 'C_aryl_fluoride', 'C_acyl_fluoride']),
    # Pure aromatic ring carbons (no N, no heteroatom in ring) (2 classes)
    ('aromatic',        ['C_aromatic', 'C_arom_O']),
    # Pure hydrocarbon: saturated (4 classes)
    ('aliphatic',       ['C_methyl', 'C_methylene', 'C_methine', 'C_quaternary']),
    # Pure hydrocarbon: unsaturated non-aromatic (3 classes)
    ('unsaturated',     ['C_alkyne', 'C_allene', 'C_vinyl']),
])

# =============================================================================
#  PUBLIC API
# =============================================================================

def get_available_schemes() -> List[str]:
    """Return list of available merging scheme names."""
    return ['none'] + list(MERGING_SCHEMES.keys())


def get_scheme(name: str) -> OrderedDict:
    """Return the merging scheme definition (merged_name -> [original_names])."""
    if name == 'none':
        # Identity: each original class maps to itself
        return OrderedDict(
            (n.removeprefix('C_'), [n]) for n in _ORIG_NAMES
        )
    if name not in MERGING_SCHEMES:
        raise ValueError(
            f"Unknown merging scheme '{name}'. "
            f"Available: {get_available_schemes()}"
        )
    return MERGING_SCHEMES[name]


def build_label_map(scheme_name: str) -> Dict[int, int]:
    """
    Build a mapping from original label index -> new (merged) label index.

    Parameters
    ----------
    scheme_name : str
        Name of the merging scheme ('none', 'chemical').

    Returns
    -------
    label_map : dict
        {original_index: new_index}.
        
    Raises
    ------
    ValueError
        If the merging scheme does not cover all original classes.
    """
    scheme = get_scheme(scheme_name)
    label_map = {}
    for new_idx, (merged_name, orig_names) in enumerate(scheme.items()):
        for orig_name in orig_names:
            label_map[CARBON_ENV_TO_IDX[orig_name]] = new_idx
    
    # Validate that all original classes are mapped
    original_range = set(range(len(IDX_TO_CARBON_ENV)))
    mapped_indices = set(label_map.keys())
    if mapped_indices != original_range:
        missing = original_range - mapped_indices
        missing_names = [IDX_TO_CARBON_ENV.get(i, f"Unknown_{i}") for i in sorted(missing)]
        raise ValueError(
            f"Merging scheme '{scheme_name}' does not cover all {len(original_range)} original classes. "
            f"Missing {len(missing)}: {missing_names}"
        )
    
    return label_map


def get_merged_class_names(scheme_name: str) -> List[str]:
    """
    Return ordered list of merged class names for a scheme.

    Index i in the returned list is the display name for merged label i.
    """
    scheme = get_scheme(scheme_name)
    return list(scheme.keys())


def get_num_classes(scheme_name: str) -> int:
    """Return the number of classes after merging."""
    return len(get_scheme(scheme_name))


def get_merged_idx_to_name(scheme_name: str) -> Dict[int, str]:
    """Return {merged_index: merged_name} mapping."""
    return {i: n for i, n in enumerate(get_merged_class_names(scheme_name))}


def get_merged_name_to_idx(scheme_name: str) -> Dict[str, int]:
    """Return {merged_name: merged_index} mapping."""
    return {n: i for i, n in enumerate(get_merged_class_names(scheme_name))}


def apply_label_merging(
    df: pd.DataFrame,
    scheme_name: str,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Remap labels in a carbon DataFrame according to a merging scheme.

    This modifies the ``carbon_env_label`` column (or a copy) so that
    spectrally similar classes share the same integer label.

    Parameters
    ----------
    df : pd.DataFrame
        Carbon-centric DataFrame with a ``carbon_env_label`` column.
    scheme_name : str
        Merging scheme name.
    inplace : bool
        If True, modify ``df`` directly; otherwise return a copy.

    Returns
    -------
    pd.DataFrame
        DataFrame with remapped labels.
    """
    if scheme_name == 'none':
        return df if inplace else df.copy()

    if not inplace:
        df = df.copy()

    label_map = build_label_map(scheme_name)
    merged_names = get_merged_class_names(scheme_name)
    # get all 'carbon_env_index' from original DataFrame
    original_index = df['carbon_env_index'].values
    original_labels = df['carbon_env_label'].values
    new_index = np.full_like(original_index, fill_value=-1)
    new_labels = np.full_like(original_labels, fill_value=-1)

    for i, orig in enumerate(original_index):
        new_index[i] = label_map.get(int(orig), -1)
        new_labels[i] = merged_names[new_index[i]]

    df['carbon_env_index'] = new_index
    df['carbon_env_label'] = new_labels

    # Also store the original label for reference / reverse lookup
    if 'carbon_env_label_original' not in df.columns:
        df['carbon_env_label_original'] = original_index

    from collections import Counter
    counts = Counter(new_index[new_index >= 0])
    n_classes = len(counts)
    total = (new_index >= 0).sum()
    print(f"\n  Class merging breakdown: scheme='{scheme_name}'  "
            f"({len(IDX_TO_CARBON_ENV)} to {n_classes} classes)")
    
    # Get the scheme to show original classes
    scheme = get_scheme(scheme_name)
    for new_idx in sorted(counts.keys()):
        merged_name = merged_names[new_idx]
        pct = 100 * counts[new_idx] / total
        print(f"{new_idx:>3}: {merged_name:<25} {counts[new_idx]:>5} ({pct:>5.1f}%)")
        
        # Show the original classes that compose this merged class
        orig_names = list(scheme.values())[new_idx]
        orig_counts = {}
        for orig_name in orig_names:
            orig_idx = CARBON_ENV_TO_IDX[orig_name]
            # Count how many times this original index appears in remapped data
            orig_mask = original_index == orig_idx
            orig_counts[orig_name.removeprefix('C_')] = orig_mask.sum()
        
        # Print original classes and their counts (indented)
        for orig_name, orig_count in orig_counts.items():
            if orig_count > 0:
                print(f"       -- {orig_name:<22} {orig_count:>5}")
        print()

    return df


def original_label_to_merged_name(
    original_idx: int,
    scheme_name: str,
) -> str:
    """Map an original label index to its merged class display name."""
    label_map = build_label_map(scheme_name)
    merged_names = get_merged_class_names(scheme_name)
    new_idx = label_map.get(original_idx, -1)
    if new_idx < 0:
        return f"unmapped_{original_idx}"
    return merged_names[new_idx]


def print_scheme_summary(scheme_name: str) -> None:
    """Print a detailed summary of a merging scheme."""
    scheme = get_scheme(scheme_name)
    print(f"\n{'=' * 70}")
    print(f"  Merging Scheme: '{scheme_name}'  ({len(scheme)} classes)")
    print(f"{'=' * 70}")
    for new_idx, (merged_name, orig_names) in enumerate(scheme.items()):
        orig_str = ', '.join(n.removeprefix('C_') for n in orig_names)
        print(f"  {new_idx:>3}: {merged_name:<25} <- {orig_str}")
    print()
