"""
Carbon Environment Class Merging Schemes
=========================================

Provides configurable class merging for the 36-class carbon environment
taxonomy defined in carbon_environment.py.  Merging reduces the number of
classes to improve CNN classification accuracy by grouping spectrally and
chemically similar environments.

Three schemes are available:

  'none'          — No merging (original 36 classes, ~33 active)
  'conservative'  — 36 → 16 classes.  Merges only spectrally indistinguishable
                     pairs while preserving maximum chemical resolution.
  'practical'     — 36 → 11 classes.  Guided by what the CNN can actually
                     distinguish based on spectral separability analysis.
  'aggressive'    — 36 → 6 classes.  Maximum CNN accuracy, minimum chemical
                     resolution.

Usage:
    from augernet.class_merging import apply_label_merging, get_merged_class_names

    # In training script, after loading the DataFrame:
    df = apply_label_merging(df, scheme='practical')
    num_classes = df['carbon_env_label'].nunique()  # or max+1

    # For display:
    merged_names = get_merged_class_names(scheme='practical')
    # merged_names[new_label_idx] → human-readable name
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

# Original class name → original index (for reference)
_ORIG_NAMES = list(CARBON_ENVIRONMENT_PATTERNS.keys())
_NAME2IDX = CARBON_ENV_TO_IDX  # e.g. 'C_methyl' → 32


# =============================================================================
#  SCHEME DEFINITIONS
# =============================================================================
# Each scheme maps: merged_class_name → list of original C_ class names.
# The order of keys defines the new label indices (0, 1, 2, ...).

MERGING_SCHEMES: Dict[str, OrderedDict] = {}

# ─────────────────────────────────────────────────────────────────────────────
#  CHEMICAL (36 → 17)  ★ Best experimental generalization
#  Groups by heteroatom neighbors.  Aromatic merged into hydrocarbon —
#  critical for bridging the calc→exp domain gap (exp benzene spectra are
#  shifted ~5 eV and 3-10× broader, making "aromatic" indistinguishable
#  from other C─C environments at experimental resolution).
# ─────────────────────────────────────────────────────────────────────────────
MERGING_SCHEMES['chemical'] = OrderedDict([
    #('heteroaromatic',      ['C_arom_N', 'C_arom_O', 'C_arom_O_N']),
    ('arom_N',              ['C_arom_N']),
    ('arom_O',              ['C_arom_O']),
    ('arom_O_N',            ['C_arom_O_N']),
    ('aryl_N',              ['C_aryl_amine', 'C_aryl_nitro']),
    ('aryl_O',              ['C_phenol', 'C_aryl_ether']),
    ('aryl_F',              ['C_aryl_fluoride']),
    ('hydrocarbon',         ['C_methyl', 'C_methylene', 'C_methine', 'C_quaternary',
                                'C_alkyne', 'C_vinyl', 'C_allene', 'C_aromatic']),
    ('carbonyl',            ['C_ketone', 'C_aldehyde', 'C_ester_carbonyl', 'C_ester_alkyl', 
                                'C_carboxylic_acid', 'C_carboxylate']),
    ('amide_carbonyl',      ['C_amide_carbonyl']),
    ('acyl_fluoride',       ['C_acyl_fluoride']),
    ('nitrile',             ['C_nitrile']),
    ('imine',               ['C_imine']),
    ('enol',                ['C_enol']),
    ('C_O_single',          ['C_ether', 'C_alcohol']),
    ('C_N_single',          ['C_amine']),
    ('fluorinated',         ['C_fluorinated']),
    ('cumulated_N',         ['C_carbodiimide', 'C_ketenimine']),
    ('cumulated_O',         ['C_ketene', 'C_CO2']),
    ('isocyanate',        ['C_isocyanate']),
])
# MERGING_SCHEMES['chemical'] = OrderedDict([
#     ('heteroaromatic',      ['C_arom_N', 'C_arom_O', 'C_arom_O_N']),
#     ('aryl_N',              ['C_aryl_amine', 'C_aryl_nitro']),
#     ('aryl_O',              ['C_phenol', 'C_aryl_ether']),
#     ('aryl_F',              ['C_aryl_fluoride']),
#     ('hydrocarbon',       ['C_methyl', 'C_methylene', 'C_methine', 'C_quaternary',
#                                 'C_alkyne', 'C_vinyl', 'C_allene', 'C_aromatic']),
#     ('carbonyl',            ['C_ketone', 'C_aldehyde', 'C_ester_carbonyl', 'C_ester_alkyl', 
#                                 'C_carboxylic_acid', 'C_carboxylate']),
#     ('amide_carbonyl',      ['C_amide_carbonyl']),
#     ('acyl_fluoride',       ['C_acyl_fluoride']),
#     ('nitrile',             ['C_nitrile']),
#     ('imine',               ['C_imine']),
#     ('enol',                ['C_enol']),
#     ('C_O_single',          ['C_ether', 'C_alcohol']),
#     ('C_N_single',          ['C_amine']),
#     ('fluorinated',         ['C_fluorinated']),
#     ('heterocumulated',     ['C_carbodiimide', 'C_ketenimine',
#                              'C_ketene', 'C_CO2', 'C_isocyanate']),
# ])

# ─────────────────────────────────────────────────────────────────────────────
#  CONSERVATIVE (36 → 16)
#  Merges only the most spectrally overlapping groups.
# ─────────────────────────────────────────────────────────────────────────────
MERGING_SCHEMES['conservative'] = OrderedDict([
    ('aromatic_ring',     ['C_aromatic', 'C_arom_N', 'C_arom_O', 'C_arom_O_N']),
    ('aryl_substituted',  ['C_phenol', 'C_aryl_amine', 'C_aryl_ether',
                           'C_aryl_fluoride', 'C_aryl_nitro']),
    ('sp3_saturated',     ['C_methyl', 'C_methylene', 'C_methine', 'C_quaternary']),
    ('carbonyl',          ['C_ketone', 'C_aldehyde', 'C_amide_carbonyl',
                           'C_ester_carbonyl', 'C_carboxylic_acid',
                           'C_carboxylate', 'C_acyl_fluoride']),
    ('alkyne',            ['C_alkyne']),
    ('nitrile',           ['C_nitrile']),
    ('imine',             ['C_imine']),
    ('vinyl',             ['C_vinyl', 'C_enol']),
    ('C_O_single',        ['C_ether', 'C_alcohol', 'C_ester_alkyl']),
    ('C_N_single',        ['C_amine']),
    ('fluorinated',       ['C_fluorinated']),
    ('cumulated',         ['C_isocyanate', 'C_carbodiimide', 'C_ketene',
                           'C_ketenimine', 'C_allene', 'C_CO2']),
])

# ─────────────────────────────────────────────────────────────────────────────
#  PRACTICAL (36 → 11)  ★ Recommended
#  Guided by spectral separability + CNN confusion analysis.
# ─────────────────────────────────────────────────────────────────────────────
MERGING_SCHEMES['practical'] = OrderedDict([
    ('aromatic_ring',     ['C_aromatic', 'C_arom_N', 'C_arom_O', 'C_arom_O_N']),
    ('aryl_substituted',  ['C_phenol', 'C_aryl_amine', 'C_aryl_ether',
                           'C_aryl_fluoride', 'C_aryl_nitro']),
    ('sp3_chain',         ['C_methyl', 'C_methylene', 'C_methine', 'C_quaternary']),
    ('carbonyl',          ['C_ketone', 'C_aldehyde', 'C_amide_carbonyl',
                           'C_ester_carbonyl', 'C_carboxylic_acid',
                           'C_carboxylate', 'C_acyl_fluoride']),
    ('alkyne',            ['C_alkyne']),
    ('nitrile',           ['C_nitrile']),
    ('imine',             ['C_imine']),
    ('vinyl',             ['C_vinyl', 'C_enol']),
    ('heteroatom_sp3',    ['C_ether', 'C_alcohol', 'C_ester_alkyl', 'C_amine']),
    ('fluorinated',       ['C_fluorinated']),
    ('cumulated',         ['C_isocyanate', 'C_carbodiimide', 'C_ketene',
                           'C_ketenimine', 'C_allene', 'C_CO2']),
])

# ─────────────────────────────────────────────────────────────────────────────
#  AGGRESSIVE (36 → 6)
#  Maximum discriminability, minimum resolution.
# ─────────────────────────────────────────────────────────────────────────────
MERGING_SCHEMES['aggressive'] = OrderedDict([
    ('aromatic_pi',       ['C_aromatic', 'C_arom_N', 'C_arom_O', 'C_arom_O_N',
                           'C_phenol', 'C_aryl_amine', 'C_aryl_ether',
                           'C_aryl_fluoride', 'C_aryl_nitro',
                           'C_vinyl', 'C_enol', 'C_imine', 'C_methine']),
    ('sp3_aliphatic',     ['C_methyl', 'C_methylene', 'C_quaternary',
                           'C_ether', 'C_alcohol', 'C_ester_alkyl', 'C_amine']),
    ('carbonyl',          ['C_ketone', 'C_aldehyde', 'C_amide_carbonyl',
                           'C_ester_carbonyl', 'C_carboxylic_acid',
                           'C_carboxylate', 'C_acyl_fluoride']),
    ('sp_linear',         ['C_alkyne', 'C_nitrile', 'C_isocyanate',
                           'C_carbodiimide', 'C_ketene', 'C_ketenimine',
                           'C_CO2', 'C_allene']),
    ('fluorinated',       ['C_fluorinated']),
])


# =============================================================================
#  PUBLIC API
# =============================================================================

def get_available_schemes() -> List[str]:
    """Return list of available merging scheme names."""
    return ['none'] + list(MERGING_SCHEMES.keys())


def get_scheme(name: str) -> OrderedDict:
    """Return the merging scheme definition (merged_name → [original_names])."""
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
    Build a mapping from original label index → new (merged) label index.

    Parameters
    ----------
    scheme_name : str
        Name of the merging scheme ('none', 'conservative', 'practical', 'aggressive').

    Returns
    -------
    label_map : dict
        {original_index: new_index}.  Unmapped original indices are mapped to -1.
    """
    scheme = get_scheme(scheme_name)
    label_map = {}
    for new_idx, (merged_name, orig_names) in enumerate(scheme.items()):
        for orig_name in orig_names:
            if orig_name in _NAME2IDX:
                label_map[_NAME2IDX[orig_name]] = new_idx
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
    label_col: str = 'carbon_env_label',
    inplace: bool = False,
    verbose: bool = True,
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
    label_col : str
        Column containing original integer labels.
    inplace : bool
        If True, modify ``df`` directly; otherwise return a copy.
    verbose : bool
        Print a summary of the remapping.

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

    original_labels = df[label_col].values
    new_labels = np.full_like(original_labels, fill_value=-1)

    for i, orig in enumerate(original_labels):
        new_labels[i] = label_map.get(int(orig), -1)

    # Check for unmapped labels
    unmapped_mask = new_labels == -1
    n_unmapped = unmapped_mask.sum()

    df[label_col] = new_labels

    # Also store the original label for reference / reverse lookup
    if 'carbon_env_label_original' not in df.columns:
        df['carbon_env_label_original'] = original_labels

    if verbose:
        from collections import Counter
        counts = Counter(new_labels[new_labels >= 0])
        n_classes = len(counts)
        total = (new_labels >= 0).sum()
        print(f"\n  Label merging: scheme='{scheme_name}'  "
              f"({len(IDX_TO_CARBON_ENV)} → {n_classes} classes)")
        for new_idx in sorted(counts.keys()):
            pct = 100 * counts[new_idx] / total
            print(f"    {new_idx:>3}: {merged_names[new_idx]:<25} "
                  f"{counts[new_idx]:>5} ({pct:>5.1f}%)")
        if n_unmapped > 0:
            print(f"    ⚠ {n_unmapped} atoms had unmapped labels (set to -1)")

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
        print(f"  {new_idx:>3}: {merged_name:<25} ← {orig_str}")
    print()
