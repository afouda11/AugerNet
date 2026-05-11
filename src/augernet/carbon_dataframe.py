"""
Auger CNN Data - Carbon-centric DataFrame utilities
=====================================================

Loads the per-carbon DataFrames produced by ``prepare_data.py`` and wraps
them in a PyTorch ``Dataset`` ready for the 1-D CNN pipeline.

The new data format has **separate singlet / triplet stick columns**::

    sing_stick_energies, sing_stick_intensities,
    trip_stick_energies, trip_stick_intensities

These are combined and Gaussian-broadened into a single spectrum inside
``CarbonDataset.__init__`` (cached once, not re-computed every epoch).

Public API (consumed by ``backend_cnn.py``):
    CarbonDataset(df, ...)      -> torch Dataset
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import time as _time
from pathlib import Path
from typing import Any, Dict, Tuple
from torch.utils.data import Dataset

from augernet.spec_utils import fit_spectrum_to_grid

# ---------------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------------

class CarbonDataset(Dataset):
    """PyTorch Dataset - one sample per carbon atom with labels.

    Handles the new data format with separate singlet/triplet sticks.
    Singlet + triplet peaks are **combined** and Gaussian-broadened once
    in ``__init__`` (no per-epoch overhead). Returns spectrum + label in
    a standardized format.

    Parameters
    ----------
    df : pd.DataFrame
        Carbon-centric DataFrame (``cnn_auger_calc.pkl`` or similar).
        Must include ``carbon_env_label`` column.
    include_augmentation : bool
        Prepend z-score normalised ``delta_be`` as an extra input channel.
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
        normalize_intensity: bool = True,
        broadening_fwhm: float = 1.6,
        energy_min: float = 200.0,
        energy_max: float = 273.0,
        
        n_points: int = 731,
    ):
        self.df = df.reset_index(drop=True)
        self.include_augmentation = include_augmentation
        self.normalize_intensity = normalize_intensity
        self.broadening_fwhm = broadening_fwhm

        n_atoms = len(self.df)

        # -- Combine sing+trip sticks and broaden once --
        t0 = _time.time()
        self._spectra = np.zeros((n_atoms, n_points), dtype=np.float32)

        for i in range(n_atoms):
            row = self.df.iloc[i]

            # New format - separate singlet / triplet
            se = np.asarray(row['sing_stick_energies'], dtype=np.float64)
            si = np.asarray(row['sing_stick_intensities'], dtype=np.float64)
            te = np.asarray(row['trip_stick_energies'], dtype=np.float64)
            ti = np.asarray(row['trip_stick_intensities'], dtype=np.float64)
            energies = np.concatenate([se, te])
            intensities = np.concatenate([si, ti])

            if len(energies) > 0:
                _, intensity_grid = fit_spectrum_to_grid(
                    energies, intensities,
                    fwhm=broadening_fwhm,
                    energy_min=energy_min,
                    energy_max=energy_max,
                    n_points=n_points,
                    normalize=self.normalize_intensity,
                )
                self._spectra[i] = intensity_grid

        elapsed = _time.time() - t0
        print(f"  Pre-broadened {n_atoms} spectra "
              f"(FWHM={broadening_fwhm} eV) in {elapsed:.1f}s")

        # -- delta_be z-score normalisation --
        if 'delta_be' in df.columns:
            mu = df['delta_be'].mean()
            std = df['delta_be'].std() + 1e-8
            self._delta_be_norm = ((df['delta_be'] - mu) / std).values

        print(f"CarbonDataset: {n_atoms} atoms, "
              f"augment={include_augmentation}")

    # -------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (spectrum, label) tuple for DataLoader compatibility."""
        row = self.df.iloc[idx]

        spectrum = torch.from_numpy(self._spectra[idx].copy())

        dbe = float(self._delta_be_norm[idx])
        # Optional delta_be augmentation (prepend to spectrum)
        if self.include_augmentation:
            spectrum = torch.cat([torch.tensor([dbe]), spectrum])

        label = torch.tensor(row['carbon_env_index'], dtype=torch.long)

        return spectrum, dbe, label

    # -------------------------------------------------------------------
    def get_class_weights_and_counts(self, num_classes: int) -> Tuple[torch.Tensor, Dict[int, int]]:
        """Compute inverse-frequency class weights and per-class counts.

        Parameters
        ----------
        num_classes : int
            Total number of classes.

        Returns
        -------
        weights : torch.Tensor
            Shape ``(num_classes,)``.  Inverse-frequency weights normalised so that
            the active (non-zero) entries have unit mean.
        counts : Dict[int, int]
            ``{class_idx: sample_count}`` for every class index in
            ``range(num_classes)``, with ``0`` for classes absent from dataset.
        """

        raw_counts = self.df['carbon_env_index'].value_counts().to_dict()
        counts = {i: raw_counts.get(i, 0) for i in range(num_classes)}

        weights = torch.zeros(num_classes, dtype=torch.float32)
        total_samples = len(self.df)
        for class_idx, count in raw_counts.items():
            if 0 <= class_idx < num_classes:
                weights[class_idx] = total_samples / (num_classes * count)
        active_mask = weights > 0
        if active_mask.sum() > 0:
            weights[active_mask] = weights[active_mask] / weights[active_mask].mean()

        return weights, counts
