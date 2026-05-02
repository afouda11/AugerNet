#!/usr/bin/env python3

import os, argparse, shutil, string, pathlib
import numpy as np
import warnings

# Suppress RDKit deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


# =============================================================================
# GAUSSIAN FITTING FOR SPECTRA (combined singlet + triplet)
# =============================================================================

def gaussian1D(yo, xo, x, d):
    """
    Gaussian function (unnormalized).
    
    Parameters
    ----------
    yo : float
        Peak height (intensity)
    xo : float
        Peak center (energy)
    x : np.ndarray
        Energy grid
    d : float
        Gaussian sigma (FWHM-related parameter)
        
    Returns
    -------
    np.ndarray
        Gaussian curve values
    """
    return yo * np.exp(-1.0 * ((x - xo) ** 2) / (2.0 * (d ** 2)))


def fit_spectrum_to_grid(energy_peaks, intensity_peaks, fwhm=1.5, 
                         energy_min=200.0, energy_max=270.0, n_points=1401):
    """
    Fit discrete spectrum peaks (energy-intensity pairs) to a continuous grid
    using Gaussian broadening.
    
    This mimics experimental spectra where each transition is broadened by
    instrumental resolution (typically ~1.5 eV FWHM).
    
    Parameters
    ----------
    energy_peaks : np.ndarray
        Energy values of spectral peaks, shape (n_peaks,)
    intensity_peaks : np.ndarray
        Intensity values of spectral peaks, shape (n_peaks,)
    fwhm : float, default=1.5
        Full-width-at-half-maximum of Gaussian broadening in eV
    energy_min : float, default=200.0
        Minimum energy for the output grid (eV)
    energy_max : float, default=270.0
        Maximum energy for the output grid (eV)
    n_points : int, default=1401
        Number of points in the output grid (1 point per 0.05 eV)
        
    Returns
    -------
    energy_grid : np.ndarray
        Energy values of the fitted spectrum, shape (n_points,)
    intensity_grid : np.ndarray
        Fitted (smoothed) intensity values, shape (n_points,)
    """
    # Create energy grid
    energy_grid = np.linspace(energy_min, energy_max, n_points)
    
    # Convert FWHM to sigma: FWHM = 2.355 * sigma
    sigma = fwhm / 2.355
    
    # Initialize fitted spectrum
    intensity_grid = np.zeros(n_points, dtype=np.float32)
    
    # Convolve each peak with Gaussian
    for energy_peak, intensity_peak in zip(energy_peaks, intensity_peaks):
        intensity_grid += gaussian1D(intensity_peak, energy_peak, energy_grid, sigma)
    
    # Normalize to [0, 1]
#     max_intensity = intensity_grid.max()
#     if max_intensity > 0:
#         intensity_grid = intensity_grid / max_intensity
#     else:
#         print(f"Warning: max_intensity is {max_intensity} (<=0), spectrum may be invalid")
    
    return energy_grid, intensity_grid


def get_maxI_maxE(
    data_type: str,
    mol_dir: str,
    mol_name: str,
    max_spec_len: int
    ):
    """
    Load singlet and triplet spectra for every carbon in *mol_id*,
    using _out_map.txt file to correctly map spectrum indices
    to atom positions in the XYZ file.
    
    The mapping file contains:
    - Column 1: Carbon index (c1, c2, c3, ...) or 0 for non-carbon atoms
    - Column 2: Binding energy or -1.0 for non-carbon atoms
    - Row order: Same as atoms in XYZ file (and thus same as node features order)
    
    Returns
    -------
    spec_out  : list[np.ndarray]
        Per-atom node labels (zero-padded spectra for carbons, zeros for non-carbons).
    spec_len : int
        Actual spectrum lengths before padding.
    """
    
    sing_spec_out = []
    trip_spec_out = []
    
    # ---- Load mapping from node_features_mapped.txt or cebe_mapped.txt ----
    # Try node_features_mapped.txt first (calc data naming)
    mapped_file = os.path.join(mol_dir, f"{mol_name}_out_map.txt")
    mapped_data = np.loadtxt(mapped_file)
    
    # mapped_data[:, 0] contains the carbon indices (c_idx+1) or 0 for non-carbons
    # Each row corresponds to an atom in XYZ order
    carbon_idx_mapping = mapped_data[:, 0].astype(int)  # Column 1: carbon index
    
    maxI = []
    maxE = []
    # Determine max carbon index in this molecule
    for c_idx in carbon_idx_mapping:
        
        if c_idx == 0.0:
            empty_spec = np.zeros((max_spec_len, 2))
            sing_spec_out.append(empty_spec)
            trip_spec_out.append(empty_spec)
        else:
            #print(f"[{mol_id}] loading spectra for carbon c{c_idx}...")
            if data_type == 'calc_auger':
                sing_spec_path = os.path.join(
                    mol_dir, f"{mol_name}_auger_singlet_c{c_idx}.auger.spectrum.out"
                )
                trip_spec_path = os.path.join(
                    mol_dir, f"{mol_name}_auger_triplet_c{c_idx}.auger.spectrum.out"
                )
            if data_type == 'eval_auger':
                sing_spec_path = os.path.join(
                    mol_dir, f"{mol_name}_mcpdft_hybrid_rcc_singlet_c{c_idx}.auger.spectrum.out"
                )
                trip_spec_path = os.path.join(
                    mol_dir, f"{mol_name}_mcpdft_hybrid_rcc_triplet_c{c_idx}.auger.spectrum.out"
                )
            
            # ---- read both spectra (skip on any error) -----------------------------
            sing_spec_arr = np.loadtxt(sing_spec_path)
            if sing_spec_arr.size == 0 :
                raise ValueError("empty singlet spectrum")

            trip_spec_arr = np.loadtxt(trip_spec_path)
            if trip_spec_arr.size == 0 :
                raise ValueError("empty triplet spectrum")

            sing_spec_arr = sing_spec_arr[sing_spec_arr[:, 0].argsort()]
            trip_spec_arr = trip_spec_arr[trip_spec_arr[:, 0].argsort()]

            maxE.append([sing_spec_arr[:, 0].max(), trip_spec_arr[:, 0].max()])
            maxI.append([sing_spec_arr[:, 1].max(), trip_spec_arr[:, 1].max()])

    return maxE, maxI


def extract_spectra(
    data_type: str,
    mol_dir: str,
    mol_name: str,
    maxE: float,
    maxI: float,
    max_spec_len: int
    ):
    """
    Load singlet and triplet spectra for every carbon in *mol_id*,
    using _out_map.txt file to correctly map spectrum indices
    to atom positions in the XYZ file.
    
    The mapping file contains:
    - Column 1: Carbon index (c1, c2, c3, ...) or 0 for non-carbon atoms
    - Column 2: Binding energy or -1.0 for non-carbon atoms
    - Row order: Same as atoms in XYZ file (and thus same as node features order)
    
    Returns
    -------
    spec_out  : list[np.ndarray]
        Per-atom node labels (zero-padded spectra for carbons, zeros for non-carbons).
    spec_len : int
        Actual spectrum lengths before padding.
    """
    
    sing_spec_out = []
    trip_spec_out = []
    
    # ---- Load mapping from node_features_mapped.txt or cebe_mapped.txt ----
    # Try node_features_mapped.txt first (calc data naming)
    mapped_file = os.path.join(mol_dir, f"{mol_name}_out_map.txt")
    mapped_data = np.loadtxt(mapped_file)
    
    # mapped_data[:, 0] contains the carbon indices (c_idx+1) or 0 for non-carbons
    # Each row corresponds to an atom in XYZ order
    carbon_idx_mapping = mapped_data[:, 0].astype(int)  # Column 1: carbon index
    
    sing_spec_len = 0
    trip_spec_len = 0

    # Determine max carbon index in this molecule
    for c_idx in carbon_idx_mapping:
        
        if c_idx == 0.0:
            empty_spec = np.zeros((max_spec_len, 2))
            sing_spec_out.append(empty_spec)
            trip_spec_out.append(empty_spec)
        else:
            #print(f"[{mol_id}] loading spectra for carbon c{c_idx}...")
            if data_type == 'calc_auger':
                sing_spec_path = os.path.join(
                    mol_dir, f"{mol_name}_auger_singlet_c{c_idx}.auger.spectrum.out"
                )
                trip_spec_path = os.path.join(
                    mol_dir, f"{mol_name}_auger_triplet_c{c_idx}.auger.spectrum.out"
                )
            if data_type == 'eval_auger':
                sing_spec_path = os.path.join(
                    mol_dir, f"{mol_name}_mcpdft_hybrid_rcc_singlet_c{c_idx}.auger.spectrum.out"
                )
                trip_spec_path = os.path.join(
                    mol_dir, f"{mol_name}_mcpdft_hybrid_rcc_triplet_c{c_idx}.auger.spectrum.out"
                )
            
            # ---- read both spectra (skip on any error) -----------------------------
            sing_spec_arr = np.loadtxt(sing_spec_path)
            if sing_spec_arr.size == 0 :
                raise ValueError("empty singlet spectrum")

            trip_spec_arr = np.loadtxt(trip_spec_path)
            if trip_spec_arr.size == 0 :
                raise ValueError("empty triplet spectrum")

            # ---- sort by increasing energy before normalization ----
            # Raw QC output is NOT energy-ordered.  Sorting creates a
            # canonical ordering so the same energy region always maps to
            # the same index range in the flattened target vector.
            # this suprisingly made it worse
            sing_spec_arr = sing_spec_arr[sing_spec_arr[:, 0].argsort()]
            trip_spec_arr = trip_spec_arr[trip_spec_arr[:, 0].argsort()]

            # ---- normalize spectra ----
            sing_spec_arr[:, 0] /= maxE             # norm KE
            trip_spec_arr[:, 0] /= maxE             # norm KE
            sing_spec_arr[:, 1] /= maxI             # norm I
            trip_spec_arr[:, 1] /= maxI             # norm I

            sing_spec_len = sing_spec_arr.shape[0]
            trip_spec_len = trip_spec_arr.shape[0]

            #print(f"[{mol_id}]  singlet length: {len_sing}, triplet length: {len_trip}") 
            # ---- zero-pad to fixed length ----
            sing_spec_pad = np.zeros((max_spec_len, 2))
            trip_spec_pad = np.zeros((max_spec_len, 2))

            # fill len_sing/len_trip rows and both colums with spec data
            sing_spec_pad[: sing_spec_len, :] = sing_spec_arr
            trip_spec_pad[: trip_spec_len, :] = trip_spec_arr

            # This is a carbon atom with a spectrum
            sing_spec_out.append(sing_spec_pad)
            trip_spec_out.append(trip_spec_pad)

    return sing_spec_out, trip_spec_out, sing_spec_len, trip_spec_len
