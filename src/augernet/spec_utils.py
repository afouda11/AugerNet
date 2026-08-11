#!/usr/bin/env python3

import os
import numpy as np
import warnings


# Suppress RDKit deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

def gaussian1D(yo, xo, x, sigma):
    return yo * np.exp(-1.0 * ((x - xo) ** 2) / (2.0 * (sigma ** 2)))


def fit_spectrum_to_grid(energy_peaks, intensity_peaks, fwhm=1.5,
                         energy_min=200.0, energy_max=270.0, n_points=1401, normalize=False):
                         
    energy_grid = np.linspace(energy_min, energy_max, n_points)
    sigma = fwhm / 2.355
    intensity_grid = np.zeros(n_points, dtype=np.float64)

    for energy_peak, intensity_peak in zip(energy_peaks, intensity_peaks):
        if intensity_peak == 0.0:
            continue                      # zero-padded row, contributes nothing
        intensity_grid += gaussian1D(intensity_peak, energy_peak, energy_grid, sigma)

    if normalize:
        max_intensity = intensity_grid.max()
        if max_intensity > 0:             # guard: all-zero spectrum stays zero
            intensity_grid = intensity_grid / max_intensity

    return energy_grid, intensity_grid.astype(np.float32)

def extract_spectra(
    data_type: str,
    mol_dir: str,
    mol_name: str,
    max_spec_len: int
    ):

    # returns uniftted spectra of shape [n_atoms, max_spec_len, 2]
    # for raw data from both singlet and triplet spectra
    # assigns the indexing from openmolcas files to the right atoms in xyz via the
    # _out_map.txt files

    sing_spec_out = []
    trip_spec_out = []
    
    # ---- Load mapping from node_features_mapped.txt or cebe_mapped.txt ----
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
            # append empty spec for non carbon atoms
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
            sing_spec_arr = sing_spec_arr[sing_spec_arr[:, 0].argsort()]
            trip_spec_arr = trip_spec_arr[trip_spec_arr[:, 0].argsort()]

            sing_spec_len = sing_spec_arr.shape[0]
            trip_spec_len = trip_spec_arr.shape[0]

            #print(f"[{mol_id}]  singlet length: {len_sing}, triplet length: {len_trip}") 
            # ---- zero-pad to fixed length ----
            sing_spec_pad = np.zeros((max_spec_len, 2))
            trip_spec_pad = np.zeros((max_spec_len, 2))

            # fill len_sing/len_trip rows and both colums with spec data
            sing_spec_pad[: sing_spec_len, :] = sing_spec_arr
            trip_spec_pad[: trip_spec_len, :] = trip_spec_arr

            # Append carbon atom spec 
            sing_spec_out.append(sing_spec_pad)
            trip_spec_out.append(trip_spec_pad)

    return sing_spec_out, trip_spec_out, carbon_idx_mapping