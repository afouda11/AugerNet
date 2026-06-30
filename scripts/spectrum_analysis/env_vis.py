"""
Environment Visualization Helpers
=================================

Semantic colour and marker palettes for carbon-environment scatter plots.

*   Original 36-class labels  -> ``IDX_TO_CARBON_ENV`` (from ``carbon_environment.py``)
*   Merged-class labels       -> ``get_merged_class_names`` (from ``class_merging.py``)

Public API
----------
get_environment_colors(df, merged_scheme=None)   -> {label_idx: RGBA}
get_environment_markers(df, merged_scheme=None)   -> {label_idx: marker_str}
get_ordered_unique_envs(df, merged_scheme=None)   -> (ordered_indices, {idx: label})
get_group_ordered_envs(active_envs, merged_scheme=None) -> [idx, ...]
format_env_label(name)                            -> str
fit_spectra_and_compute_scalars(df, fwhm, ...)  -> dict of arrays
"""

from __future__ import annotations

import colorsys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from augernet.carbon_environment import CARBON_ENV_TO_IDX, IDX_TO_CARBON_ENV
from augernet.class_merging import (
    MERGING_SCHEMES,
    get_merged_class_names,
    get_merged_idx_to_name,
)

# ------------------------------------------------------------------------------
#  Visual aesthetics per group (hue families, colorblind-friendly).
#
#  Class membership is NOT duplicated here -- it is pulled from MERGING_SCHEMES
#  in class_merging.py so both stay in sync automatically.
#
#  For ENV_GROUPS  : keys match 'heteroatom' scheme group names.
#  For MERGED_GROUPS: keys match 'chemical' scheme group names, grouped into
#                    the same hue families.
# ------------------------------------------------------------------------------

# heteroatom group name -> visual style
_HETEROATOM_AESTHETICS: Dict[str, dict] = {
    'carbonyl':      {'base_hue': 270, 'sat': 0.70, 'marker': 's'},   # purple
    'oxygen_single': {'base_hue': 210, 'sat': 0.65, 'marker': '^'},   # blue
    'nitrogen':      {'base_hue': 150, 'sat': 0.65, 'marker': 'D'},   # teal
    'halogen':       {'base_hue':  20, 'sat': 0.80, 'marker': 'P'},   # orange
    'aromatic':      {'base_hue': 330, 'sat': 0.60, 'marker': 'o'},   # pink
    'unsaturated':   {'base_hue':  50, 'sat': 0.75, 'marker': 'v'},   # gold
    'aliphatic':     {'base_hue':   0, 'sat': 0.00, 'marker': 'X'},   # grey
}

# chemical group name -> which heteroatom hue family it belongs to
_CHEMICAL_GROUP_FAMILY: Dict[str, str] = {
    'carbonyl':         'carbonyl',
    'amide_carbonyl':   'carbonyl',
    'cumulated_O':      'carbonyl',
    'C_O_single':       'oxygen_single',
    'aryl_O':           'oxygen_single',
    'nitrile':          'nitrogen',
    'imine':            'nitrogen',
    'C_N_single':       'nitrogen',
    'cumulated_N':      'nitrogen',
    'isocyanate':       'nitrogen',
    'aryl_N':           'nitrogen',
    'alkyl_fluorinated':'halogen',
    'aryl_F':           'halogen',
    'heteroaromatic':   'aromatic',
    'hydrocarbon':      'aliphatic',
}


def _build_env_groups_from_scheme(
    scheme_name: str,
    aesthetics: Dict[str, dict],
    family_map: Optional[Dict[str, str]] = None,
) -> Dict[str, dict]:
    """
    Build an ENV_GROUPS-style dict by pulling class lists from a merging scheme
    and attaching visual metadata from *aesthetics*.

    Parameters
    ----------
    scheme_name : str
        Key into MERGING_SCHEMES ('heteroatom' or 'chemical').
    aesthetics : dict
        {group_name: {base_hue, sat, marker}} for the canonical hue families.
    family_map : dict, optional
        {merged_group_name: hue_family_name}.  Required when scheme group names
        differ from aesthetics keys (i.e. for 'chemical').
    """
    scheme = MERGING_SCHEMES[scheme_name]
    groups: Dict[str, dict] = {}
    for group_name, orig_names in scheme.items():
        family = family_map[group_name] if family_map else group_name
        style = aesthetics.get(family, {'base_hue': 0, 'sat': 0.5, 'marker': 'o'})
        groups[group_name] = {
            'envs': list(orig_names),
            **style,
        }
    return groups


# Derived at module load -- single source of truth for class membership.
ENV_GROUPS: Dict[str, dict] = _build_env_groups_from_scheme(
    'heteroatom', _HETEROATOM_AESTHETICS
)
MERGED_GROUPS: Dict[str, dict] = _build_env_groups_from_scheme(
    'chemical', _HETEROATOM_AESTHETICS, family_map=_CHEMICAL_GROUP_FAMILY
)


# ------------------------------------------------------------------------------
#  Internal colour-palette builders
# ------------------------------------------------------------------------------

def _hsl_to_rgb(h: float, s: float, l: float) -> Tuple[float, float, float]:
    """HSL -> RGB  (h in degrees, s/l in 0-1)."""
    return colorsys.hls_to_rgb(h / 360.0, l, s)


def _build_palette(
    groups: Dict[str, dict],
) -> Tuple[Dict[str, tuple], Dict[str, str]]:
    """Build name -> RGBA and name -> marker dicts from a group definition.

    Keys are the individual env names listed in each group's 'envs'.
    Used for the original 36-class (no merging) case.
    """
    colors: Dict[str, tuple] = {}
    markers: Dict[str, str] = {}
    for grp in groups.values():
        envs = grp['envs']
        n = len(envs)
        if n == 0:
            continue
        hue, sat, marker = grp['base_hue'], grp['sat'], grp['marker']
        lightnesses = [0.55] if n == 1 else np.linspace(0.75, 0.35, n).tolist()
        for name, light in zip(envs, lightnesses):
            r, g, b = _hsl_to_rgb(hue, sat, light)
            colors[name] = (r, g, b, 1.0)
            markers[name] = marker
    return colors, markers


def _build_merged_palette(
    groups: Dict[str, dict],
) -> Tuple[Dict[str, tuple], Dict[str, str]]:
    """Build group_name -> RGBA and group_name -> marker dicts.

    Keys are the group names themselves (i.e. the merged class labels).
    Used when a merging scheme is active.
    """
    colors: Dict[str, tuple] = {}
    markers: Dict[str, str] = {}
    for group_name, grp in groups.items():
        hue, sat, marker = grp['base_hue'], grp['sat'], grp['marker']
        r, g, b = _hsl_to_rgb(hue, sat, 0.55)
        colors[group_name] = (r, g, b, 1.0)
        markers[group_name] = marker
    return colors, markers


# Pre-compute palettes at module load -- one entry per scheme.
# 'none' / None  -> original 36-class palette (keys are C_* env names)
# any scheme     -> palette keyed by merged group names for that scheme

_ORIG_COLORS, _ORIG_MARKERS = _build_palette(ENV_GROUPS)

# Build one palette per named scheme so any merged_scheme value works.
_SCHEME_COLORS: Dict[str, Dict[str, tuple]] = {}
_SCHEME_MARKERS: Dict[str, Dict[str, str]] = {}

for _scheme_name, _scheme_groups in [
    ('heteroatom', ENV_GROUPS),
    ('chemical',   MERGED_GROUPS),
]:
    _c, _m = _build_merged_palette(_scheme_groups)
    _SCHEME_COLORS[_scheme_name] = _c
    _SCHEME_MARKERS[_scheme_name] = _m

_FALLBACK_COLOR = (0.5, 0.5, 0.5, 1.0)
_FALLBACK_MARKER = 'o'


# ------------------------------------------------------------------------------
#  Public helpers
# ------------------------------------------------------------------------------

def format_env_label(env_name: str) -> str:
    """Remove ``C_`` prefix and replace underscores with spaces."""
    return env_name.replace('C_', '').replace('_', ' ')


def _idx_to_name(idx: int, merged_scheme: Optional[str]) -> str:
    """Resolve an integer label to its display name."""
    if merged_scheme is None or merged_scheme == 'none':
        return IDX_TO_CARBON_ENV.get(int(idx), f'Unknown_{idx}')
    return get_merged_idx_to_name(merged_scheme).get(int(idx), f'merged_{idx}')


def _name_to_color(name: str, merged_scheme: Optional[str]) -> tuple:
    if merged_scheme and merged_scheme != 'none':
        palette = _SCHEME_COLORS.get(merged_scheme, {})
        return palette.get(name, _FALLBACK_COLOR)
    return _ORIG_COLORS.get(name, _FALLBACK_COLOR)


def _name_to_marker(name: str, merged_scheme: Optional[str]) -> str:
    if merged_scheme and merged_scheme != 'none':
        palette = _SCHEME_MARKERS.get(merged_scheme, {})
        return palette.get(name, _FALLBACK_MARKER)
    return _ORIG_MARKERS.get(name, _FALLBACK_MARKER)


# --- Public API ---

def get_environment_colors(
    df: pd.DataFrame,
    merged_scheme: Optional[str] = None,
    label_col: str = 'carbon_env_label',
) -> Dict[int, tuple]:
    """Return ``{label_idx: RGBA}`` for every active label in *df*."""
    result: Dict[int, tuple] = {}
    for idx in sorted(df[label_col].unique()):
        if int(idx) < 0:
            continue
        name = _idx_to_name(idx, merged_scheme)
        result[idx] = _name_to_color(name, merged_scheme)
    return result


def get_environment_markers(
    df: pd.DataFrame,
    merged_scheme: Optional[str] = None,
    label_col: str = 'carbon_env_label',
) -> Dict[int, str]:
    """Return ``{label_idx: marker_str}`` for every active label in *df*."""
    result: Dict[int, str] = {}
    for idx in sorted(df[label_col].unique()):
        if int(idx) < 0:
            continue
        name = _idx_to_name(idx, merged_scheme)
        result[idx] = _name_to_marker(name, merged_scheme)
    return result


def get_ordered_unique_envs(
    df: pd.DataFrame,
    merged_scheme: Optional[str] = None,
    label_col: str = 'carbon_env_label',
) -> Tuple[List[int], Dict[int, str]]:
    """
    Unique environments ordered by descending count.

    Returns (ordered_indices, {idx: formatted_label}).
    """
    counts = df[label_col].value_counts()
    ordered = [i for i in counts.index.tolist() if int(i) >= 0]
    labels = {
        idx: format_env_label(_idx_to_name(idx, merged_scheme))
        for idx in ordered
    }
    return ordered, labels


def get_env_family(env_name: str, merged_scheme: Optional[str] = None) -> str:
    """Return the hue-family group name for an env name.

    For 'chemical': multiple merged classes map to a shared hue family.
    For 'heteroatom': each merged label is already its own family.
    For None/'none': look up which ENV_GROUPS group the C_* name belongs to.
    """
    if merged_scheme == 'chemical':
        return _CHEMICAL_GROUP_FAMILY.get(env_name, env_name)
    if merged_scheme == 'heteroatom':
        return env_name
    for grp_name, grp in ENV_GROUPS.items():
        if env_name in grp['envs']:
            return grp_name
    return env_name


def get_group_ordered_envs(
    active_envs,
    merged_scheme: Optional[str] = None,
) -> List[int]:
    """
    Return environment indices ordered by group so similar hues are adjacent
    in legends.
    """
    active = set(int(e) for e in active_envs)
    groups = MERGED_GROUPS if (merged_scheme and merged_scheme != 'none') else ENV_GROUPS

    # Resolve name -> idx depending on scheme
    if merged_scheme and merged_scheme != 'none':
        name2idx = {n: i for i, n in get_merged_idx_to_name(merged_scheme).items()}
    else:
        name2idx = CARBON_ENV_TO_IDX

    ordered: List[int] = []
    for grp in groups.values():
        for env_name in grp['envs']:
            idx = name2idx.get(env_name)
            if idx is not None and idx in active:
                ordered.append(idx)
    # Append any remaining not in groups
    for idx in sorted(active):
        if idx not in ordered:
            ordered.append(idx)
    return ordered


def get_group_ordered_envs_str(
    active_envs,
    merged_scheme: Optional[str] = None,
) -> List[str]:
    """
    Return environment names ordered by group so similar hues are adjacent
    in legends. Works directly with string labels.
    """
    active = set(active_envs)
    groups = MERGED_GROUPS if (merged_scheme and merged_scheme != 'none') else ENV_GROUPS

    ordered: List[str] = []

    if merged_scheme and merged_scheme != 'none':
        # Labels are the group names themselves -- iterate group keys directly.
        for group_name in groups:
            if group_name in active and group_name not in ordered:
                ordered.append(group_name)
    else:
        # Labels are original C_* names -- iterate envs within each group.
        for grp in groups.values():
            for env_name in grp['envs']:
                if env_name in active and env_name not in ordered:
                    ordered.append(env_name)

    # Append any remaining not covered by groups
    for env_name in sorted(active):
        if env_name not in ordered:
            ordered.append(env_name)
    return ordered


# ------------------------------------------------------------------------------
#  Spectral-scalar computation
# ------------------------------------------------------------------------------

def fit_spectra_and_compute_scalars(
    df: pd.DataFrame,
    fwhm: float = 1.6,
    energy_min: float = 200.0,
    energy_max: float = 273.0,
    n_points: int = 731,
) -> Dict[str, np.ndarray]:
    """
    Compute scalar spectral features from stick data.

    Works with the new data format (``sing_stick_*`` + ``trip_stick_*``
    columns) or the old ``stick_energies`` / ``stick_intensities`` columns.

    Returns dict with keys: ``centroid``, ``width``, ``skewness``, ``entropy``.
    Also adds broadened ``spectrum_intensity_only`` back into *df* (in-place).
    """
    from augernet.spec_utils import fit_spectrum_to_grid

    energy_grid = np.linspace(energy_min, energy_max, n_points)
    dE = (energy_max - energy_min) / (n_points - 1)

    has_sing = 'sing_stick_energies' in df.columns
    has_trip = 'trip_stick_energies' in df.columns
    has_combined = 'stick_energies' in df.columns

    centroids, widths, skews, entropies = [], [], [], []
    all_spectra = []

    for i in range(len(df)):
        row = df.iloc[i]

        # Collect stick data
        if has_sing and has_trip:
            se = np.asarray(row['sing_stick_energies'], dtype=np.float64)
            si = np.asarray(row['sing_stick_intensities'], dtype=np.float64)
            te = np.asarray(row['trip_stick_energies'], dtype=np.float64)
            ti = np.asarray(row['trip_stick_intensities'], dtype=np.float64)
            energies = np.concatenate([se, te])
            intensities = np.concatenate([si, ti])
        elif has_combined:
            energies = np.asarray(row['stick_energies'], dtype=np.float64)
            intensities = np.asarray(row['stick_intensities'], dtype=np.float64)
        else:
            spec = np.zeros(n_points, dtype=np.float64)
            all_spectra.append(spec)
            centroids.append(0.0)
            widths.append(0.0)
            skews.append(0.0)
            entropies.append(0.0)
            continue

        # Broaden
        if len(energies) > 0:
            _, spec = fit_spectrum_to_grid(
                energies, intensities,
                fwhm=fwhm,
                energy_min=energy_min,
                energy_max=energy_max,
                n_points=n_points,
            )
        else:
            spec = np.zeros(n_points, dtype=np.float64)

        all_spectra.append(spec)

        total = spec.sum() + 1e-8
        mu = np.sum(spec * energy_grid) / total
        sigma = np.sqrt(np.sum(spec * (energy_grid - mu) ** 2) / total)
        if sigma > 1e-10:
            skew = np.sum(spec * ((energy_grid - mu) / sigma) ** 3) / total
        else:
            skew = 0.0
        spec_norm = spec / total
        ent = -np.sum(spec_norm * np.log(spec_norm + 1e-10))

        centroids.append(mu)
        widths.append(sigma)
        skews.append(skew)
        entropies.append(ent)

    # Store broadened spectra in-place for downstream use (PCA, etc.)
    df['spectrum_intensity_only'] = all_spectra

    return {
        'centroid': np.asarray(centroids),
        'width': np.asarray(widths),
        'skewness': np.asarray(skews),
        'entropy': np.asarray(entropies),
    }
