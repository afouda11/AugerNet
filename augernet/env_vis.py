"""
Environment Visualization Helpers
=================================

Semantic colour and marker palettes for carbon-environment scatter plots.

*   Original 36-class labels  → ``IDX_TO_CARBON_ENV`` (from ``carbon_environment.py``)
*   Merged-class labels       → ``get_merged_class_names`` (from ``class_merging.py``)

Public API
----------
get_environment_colors(df, merged_scheme=None)   → {label_idx: RGBA}
get_environment_markers(df, merged_scheme=None)   → {label_idx: marker_str}
get_ordered_unique_envs(df, merged_scheme=None)   → (ordered_indices, {idx: label})
get_group_ordered_envs(active_envs, merged_scheme=None) → [idx, ...]
format_env_label(name)                            → str
compute_spectral_scalars(energies, intensities, ...)  → dict of arrays
"""

from __future__ import annotations

import colorsys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from augernet.carbon_environment import CARBON_ENV_TO_IDX, IDX_TO_CARBON_ENV
from augernet.class_merging import (
    get_merged_class_names,
    get_merged_idx_to_name,
)

# ──────────────────────────────────────────────────────────────────────────────
#  Group definitions (heteroatom-based hue families, colorblind-friendly)
# ──────────────────────────────────────────────────────────────────────────────

ENV_GROUPS: Dict[str, dict] = {
    'carbonyl': {
        'envs': [
            'C_carboxylic_acid', 'C_carboxylate', 'C_ester_carbonyl',
            'C_amide_carbonyl', 'C_ketone', 'C_aldehyde',
            'C_CO2', 'C_ketene',
        ],
        'base_hue': 270,   # purple
        'sat': 0.70,
        'marker': 's',
    },
    'oxygen_single': {
        'envs': [
            'C_ether', 'C_alcohol', 'C_ester_alkyl', 'C_phenol',
            'C_enol', 'C_aryl_ether',
        ],
        'base_hue': 210,   # blue
        'sat': 0.65,
        'marker': '^',
    },
    'nitrogen': {
        'envs': [
            'C_nitrile', 'C_imine', 'C_amine', 'C_aryl_amine',
            'C_aryl_nitro', 'C_arom_N', 'C_arom_O_N',
            'C_isocyanate', 'C_carbodiimide', 'C_ketenimine',
        ],
        'base_hue': 150,   # teal / green
        'sat': 0.65,
        'marker': 'D',
    },
    'halogen': {
        'envs': ['C_fluorinated', 'C_aryl_halide', 'C_acyl_halide'],
        'base_hue': 20,    # orange
        'sat': 0.80,
        'marker': 'P',
    },
    'aromatic': {
        'envs': ['C_aromatic', 'C_arom_O'],
        'base_hue': 330,   # pink / magenta
        'sat': 0.60,
        'marker': 'o',
    },
    'unsaturated': {
        'envs': ['C_alkyne', 'C_allene', 'C_vinyl'],
        'base_hue': 50,    # yellow / gold
        'sat': 0.75,
        'marker': 'v',
    },
    'aliphatic': {
        'envs': ['C_methyl', 'C_methylene', 'C_methine', 'C_quaternary'],
        'base_hue': 0,     # grey (sat=0)
        'sat': 0.0,
        'marker': 'X',
    },
}

# Merged-class → group mapping (for the 'chemical' scheme)
# Keys are merged class names (as returned by get_merged_class_names).
# If a merged class doesn't appear here it falls back to generic colour.

MERGED_GROUPS: Dict[str, dict] = {
    'carbonyl': {
        'envs': ['carbonyl', 'amide_carbonyl', 'acyl_fluoride'],
        'base_hue': 270, 'sat': 0.70, 'marker': 's',
    },
    'oxygen_single': {
        'envs': ['C_O_single', 'enol'],
        'base_hue': 210, 'sat': 0.65, 'marker': '^',
    },
    'nitrogen': {
        'envs': [
            'nitrile', 'imine', 'C_N_single', 'cumulated_N',
            'isocyanate', 'cumulated_O',
        ],
        'base_hue': 150, 'sat': 0.65, 'marker': 'D',
    },
    'halogen': {
        'envs': ['fluorinated', 'aryl_F'],
        'base_hue': 20, 'sat': 0.80, 'marker': 'P',
    },
    'aromatic': {
        'envs': ['arom_N', 'arom_O', 'arom_O_N', 'aryl_N', 'aryl_O'],
        'base_hue': 330, 'sat': 0.60, 'marker': 'o',
    },
    'unsaturated': {
        'envs': [],
        'base_hue': 50, 'sat': 0.75, 'marker': 'v',
    },
    'aliphatic': {
        'envs': ['hydrocarbon'],
        'base_hue': 0, 'sat': 0.0, 'marker': 'X',
    },
}


# ──────────────────────────────────────────────────────────────────────────────
#  Internal colour-palette builders
# ──────────────────────────────────────────────────────────────────────────────

def _hsl_to_rgb(h: float, s: float, l: float) -> Tuple[float, float, float]:
    """HSL → RGB  (h in degrees, s/l in 0-1)."""
    return colorsys.hls_to_rgb(h / 360.0, l, s)


def _build_palette(
    groups: Dict[str, dict],
) -> Tuple[Dict[str, tuple], Dict[str, str]]:
    """Build name → RGBA and name → marker dicts from a group definition."""
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


# Pre-compute at module load
_ORIG_COLORS, _ORIG_MARKERS = _build_palette(ENV_GROUPS)
_MERGED_COLORS, _MERGED_MARKERS = _build_palette(MERGED_GROUPS)

_FALLBACK_COLOR = (0.5, 0.5, 0.5, 1.0)
_FALLBACK_MARKER = 'o'


# ──────────────────────────────────────────────────────────────────────────────
#  Public helpers
# ──────────────────────────────────────────────────────────────────────────────

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
        return _MERGED_COLORS.get(name, _FALLBACK_COLOR)
    return _ORIG_COLORS.get(name, _FALLBACK_COLOR)


def _name_to_marker(name: str, merged_scheme: Optional[str]) -> str:
    if merged_scheme and merged_scheme != 'none':
        return _MERGED_MARKERS.get(name, _FALLBACK_MARKER)
    return _ORIG_MARKERS.get(name, _FALLBACK_MARKER)


# ─────────────── Public API ───────────────────────────────────────────────────

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

    # Resolve name → idx depending on scheme
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


# ──────────────────────────────────────────────────────────────────────────────
#  Spectral-scalar computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_spectral_scalars_from_sticks(
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


def compute_spectral_scalars(
    df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """
    Compute spectral scalars from pre-computed ``spectrum_intensity_only``
    and ``cdf`` columns (legacy format).

    If those columns are absent, falls back to
    :func:`compute_spectral_scalars_from_sticks`.
    """
    if 'spectrum_intensity_only' not in df.columns:
        return compute_spectral_scalars_from_sticks(df)

    E_MIN, E_MAX = 200.0, 270.0
    centroids, widths, skews, entropies = [], [], [], []

    for _, row in df.iterrows():
        spec = np.asarray(row['spectrum_intensity_only'])
        n_points = len(spec)
        energy_axis = np.linspace(E_MIN, E_MAX, n_points)
        total = spec.sum() + 1e-8

        mu = np.sum(spec * energy_axis) / total
        sigma = np.sqrt(np.sum(spec * (energy_axis - mu) ** 2) / total)
        skew = (np.sum(spec * ((energy_axis - mu) / max(sigma, 1e-10)) ** 3)
                / total) if sigma > 1e-10 else 0.0
        spec_norm = spec / total
        ent = -np.sum(spec_norm * np.log(spec_norm + 1e-10))

        centroids.append(mu)
        widths.append(sigma)
        skews.append(skew)
        entropies.append(ent)

    return {
        'centroid': np.asarray(centroids),
        'width': np.asarray(widths),
        'skewness': np.asarray(skews),
        'entropy': np.asarray(entropies),
    }
