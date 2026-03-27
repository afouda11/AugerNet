"""
Auger CNN Data — Carbon-centric DataFrame utilities
=====================================================

Loads the per-carbon DataFrames produced by ``prepare_data.py`` and wraps
them in a PyTorch ``Dataset`` ready for the 1-D CNN pipeline.

The new data format has **separate singlet / triplet stick columns**::

    sing_stick_energies, sing_stick_intensities,
    trip_stick_energies, trip_stick_intensities

These are combined and Gaussian-broadened into a single spectrum inside
``CarbonDataset.__init__`` (cached once, not re-computed every epoch).

Public API (consumed by ``backend_cnn.py``):
    load_carbon_dataframe(path) → pd.DataFrame
    CarbonDataset(df, ...)      → torch Dataset
    diagnose_dataframe(df)      → prints column summary
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import time as _time
from pathlib import Path
from torch.utils.data import Dataset
from typing import Any, Dict

from augernet.spec_utils import fit_spectrum_to_grid


# ─────────────────────────────────────────────────────────────────────────────
#  I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_carbon_dataframe(filepath: str) -> pd.DataFrame:
    """Load a carbon DataFrame from ``.pkl`` or ``.parquet``."""
    filepath = str(filepath)
    if filepath.endswith(('.pkl', '.pickle')):
        df = pd.read_pickle(filepath)
    elif filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        # Try pickle first, then parquet
        try:
            df = pd.read_pickle(filepath)
        except Exception:
            df = pd.read_parquet(filepath)
    print(f"✓ Loaded {len(df)} carbon atoms from: {filepath}")
    return df


def diagnose_dataframe(df: pd.DataFrame) -> None:
    """Print a one-line summary of a carbon DataFrame."""
    n_mols = df['mol_name'].nunique()
    n_envs = df['carbon_env_label'].nunique()
    print(f"  DataFrame: {len(df)} carbons, {n_mols} molecules, "
          f"{n_envs} environments")


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CarbonDataset(Dataset):
    """PyTorch Dataset — one sample per carbon atom.

    Handles the new data format with separate singlet/triplet sticks.
    Singlet + triplet peaks are **combined** and Gaussian-broadened once
    in ``__init__`` (no per-epoch overhead).

    Parameters
    ----------
    df : pd.DataFrame
        Carbon-centric DataFrame (``cnn_auger_calc.pkl`` or similar).
    include_augmentation : bool
        Prepend ``delta_be`` as an extra input channel.
    augmentation_type : str
        ``'normalized'`` (z-score) or ``'scaled'`` (raw / scale_factor).
    delta_be_scale : float
        Divisor when ``augmentation_type='scaled'``.
    normalize_delta_be : bool
        Z-score normalise delta_be across the dataset.
    normalize_spectrum : bool
        Normalise broadened spectrum to [0, 1] per atom.
    broadening_fwhm : float
        FWHM for Gaussian broadening (eV).
    energy_min, energy_max : float
        Energy grid limits (eV).
    n_points : int
        Number of grid points.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        include_augmentation: bool = False,
        augmentation_type: str = 'normalized',
        delta_be_scale: float = 100.0,
        normalize_delta_be: bool = True,
        normalize_spectrum: bool = True,
        broadening_fwhm: float = 1.6,
        energy_min: float = 200.0,
        energy_max: float = 273.0,
        n_points: int = 731,
    ):
        self.df = df.reset_index(drop=True)
        self.include_augmentation = include_augmentation
        self.augmentation_type = augmentation_type
        self.delta_be_scale = delta_be_scale
        self.normalize_spectrum = normalize_spectrum
        self.broadening_fwhm = broadening_fwhm

        n_atoms = len(self.df)

        # ── Combine sing+trip sticks and broaden once ────────────────────
        t0 = _time.time()
        energy_grid = np.linspace(energy_min, energy_max, n_points)
        self._spectra = np.zeros((n_atoms, n_points), dtype=np.float32)

        has_sing = 'sing_stick_energies' in df.columns
        has_trip = 'trip_stick_energies' in df.columns
        # Fall back to old combined format if present
        has_combined = 'stick_energies' in df.columns

        for i in range(n_atoms):
            row = self.df.iloc[i]

            if has_sing and has_trip:
                # New format — separate singlet / triplet
                se = np.asarray(row['sing_stick_energies'], dtype=np.float64)
                si = np.asarray(row['sing_stick_intensities'], dtype=np.float64)
                te = np.asarray(row['trip_stick_energies'], dtype=np.float64)
                ti = np.asarray(row['trip_stick_intensities'], dtype=np.float64)
                energies = np.concatenate([se, te])
                intensities = np.concatenate([si, ti])
            elif has_combined:
                # Old format — already combined
                energies = np.asarray(row['stick_energies'], dtype=np.float64)
                intensities = np.asarray(row['stick_intensities'], dtype=np.float64)
            else:
                continue

            if len(energies) > 0:
                _, intensity_grid = fit_spectrum_to_grid(
                    energies, intensities,
                    fwhm=broadening_fwhm,
                    energy_min=energy_min,
                    energy_max=energy_max,
                    n_points=n_points,
                )
                self._spectra[i] = intensity_grid

        elapsed = _time.time() - t0
        print(f"  Pre-broadened {n_atoms} spectra "
              f"(FWHM={broadening_fwhm} eV) in {elapsed:.1f}s")

        # ── delta_be normalisation ───────────────────────────────────────
        if normalize_delta_be and 'delta_be' in df.columns:
            mu = df['delta_be'].mean()
            std = df['delta_be'].std() + 1e-8
            self._delta_be_norm = ((df['delta_be'] - mu) / std).values
        else:
            self._delta_be_norm = np.zeros(n_atoms, dtype=np.float32)

        print(f"✓ CarbonDataset: {n_atoms} atoms, "
              f"augment={include_augmentation} ({augmentation_type})")

    # ─────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        spectrum = torch.from_numpy(self._spectra[idx].copy())

        # Per-atom [0, 1] normalisation
        if self.normalize_spectrum:
            smax = spectrum.max()
            smin = spectrum.min()
            if smax - smin > 1e-8:
                spectrum = (spectrum - smin) / (smax - smin)
            else:
                spectrum = torch.zeros_like(spectrum)

        # Optional delta_be augmentation (prepend to spectrum)
        if self.include_augmentation:
            if self.augmentation_type == 'normalized':
                dbe = float(self._delta_be_norm[idx])
            else:  # scaled
                dbe = float(row['delta_be'] / self.delta_be_scale)
            spectrum = torch.cat([torch.tensor([dbe]), spectrum])

        label = torch.tensor(row['carbon_env_label'], dtype=torch.long)

        return {
            'spectrum': spectrum,
            'label': label,
            'mol_name': row['mol_name'],
            'atom_idx': int(row['atom_idx']),
        }
