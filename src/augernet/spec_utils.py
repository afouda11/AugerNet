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
    max_intensity = intensity_grid.max()
    if max_intensity > 0:
        intensity_grid = intensity_grid / max_intensity
    else:
        print(f"Warning: max_intensity is {max_intensity} (<=0), spectrum may be invalid")
    
    return energy_grid, intensity_grid


def extract_spectra(
    data_type: str,
    mol_dir: str,
    mol_name: str,
    spin: str,
    max_ke: int,
    max_spec_len: int
    ):
    """
    Load singlet / triplet spectra for every carbon in *mol_id*,
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
    
    spec_out = []
    
    # ---- Load mapping from node_features_mapped.txt or cebe_mapped.txt ----
    # Try node_features_mapped.txt first (calc data naming)
    mapped_file = os.path.join(mol_dir, f"{mol_name}_out_map.txt")
    mapped_data = np.loadtxt(mapped_file)
    
    # mapped_data[:, 0] contains the carbon indices (c_idx+1) or 0 for non-carbons
    # Each row corresponds to an atom in XYZ order
    carbon_idx_mapping = mapped_data[:, 0].astype(int)  # Column 1: carbon index
    
    spec_len = 0

    # Determine max carbon index in this molecule
    for c_idx in carbon_idx_mapping:
        
        if c_idx == 0.0:
            empty_spec = np.zeros((max_spec_len, 2))
            spec_out.append(empty_spec)
        else:
            #print(f"[{mol_id}] loading spectra for carbon c{c_idx}...")
            if data_type == 'calc_auger':
                spec_path = os.path.join(
                    mol_dir, f"{mol_name}_auger_{spin}_c{c_idx}.auger.spectrum.out"
                )
            if data_type == 'eval_auger':
                spec_path = os.path.join(
                    mol_dir, f"{mol_name}_mcpdft_hybrid_rcc_{spin}_c{c_idx}.auger.spectrum.out"
                )
            
            # ---- read both spectra (skip on any error) -----------------------------
            spec_arr = np.loadtxt(spec_path)
            if spec_arr.size == 0 :
                raise ValueError("empty spectrum")
            
            # ---- sort by increasing energy before normalization ----
            # Raw QC output is NOT energy-ordered.  Sorting creates a
            # canonical ordering so the same energy region always maps to
            # the same index range in the flattened target vector.
            # this suprisingly made it worse
            spec_arr = spec_arr[spec_arr[:, 0].argsort()]

            # ---- normalize spectra ----
            spec_arr[:, 0] /= max_ke             # norm KE
            spec_arr[:, 1] /= spec_arr[:, 1].max()    # norm I

            spec_len = spec_arr.shape[0]

            #print(f"[{mol_id}]  singlet length: {len_sing}, triplet length: {len_trip}") 
            # ---- zero-pad to fixed length ----
            spec_pad = np.zeros((max_spec_len, 2))

            # fill len_sing/len_trip rows and both colums with spec data
            spec_pad[: spec_len, :] = spec_arr

            # This is a carbon atom with a spectrum
            spec_out.append(spec_pad)

    return spec_out, spec_len

def extract_spectra_fitted(
    mol_id: str,
    atoms: dict,
    atom_list: list[str],
    *,
    auger_dir: str,
    cebe_dir: str,
    fwhm: float = 1.5,
    energy_min: float = 200.0,
    energy_max: float = 270.0,
    n_points: int = 1401
) -> tuple[
    list[np.ndarray], np.ndarray
]:
    """
    Load singlet/triplet spectra for every carbon and fit to a common energy grid.
    
    Combines singlet and triplet spectra for each carbon into a single fitted spectrum
    that resembles experimental Auger spectra more closely (Gaussian broadened).
    
    Uses the node_features_mapped.txt or cebe_mapped.txt file to correctly map spectrum indices
    to atom positions in the XYZ file.
    
    The mapping file contains:
    - Column 1: Carbon index (c1, c2, c3, ...) or 0 for non-carbon atoms
    - Column 2: Binding energy or -1.0 for non-carbon atoms
    - Row order: Same as atoms in XYZ file (and thus same as node features order)
    
    Tries to load node_features_mapped.txt first (calc data), then cebe_mapped.txt (eval data).

    Parameters
    ----------
    mol_id : str
        Molecule ID
    atoms : dict
        Atom information
    atom_list : list[str]
        List of atom symbols
    auger_dir : str
        Directory containing Auger spectrum files
    cebe_dir : str
        Directory containing CEBE mapping files
    fwhm : float, default=1.5
        Full-width-at-half-maximum for Gaussian broadening (eV)
    energy_min : float, default=200.0
        Minimum energy for fitted grid (eV)
    energy_max : float, default=270.0
        Maximum energy for fitted grid (eV)
    n_points : int, default=1401
        Number of points in the fitted grid (0.05 eV spacing for 200-270 eV)

    Returns
    -------
    combined_nl : list[np.ndarray]
        Per-atom fitted combined spectra, shape (n_atoms, n_points, 2)
        Each spectrum is [energy_grid, intensity_grid] as columns
        No mask needed since spectra are fitted to a common grid
    cebe_feat : np.ndarray
        Binding energy features from the mapping file
    """
    
    combined_nl = []
    
    # ---- Load mapping from node_features_mapped.txt or cebe_mapped.txt ----
    # Try node_features_mapped.txt first (calc data naming)
    mapped_file = os.path.join(cebe_dir, f"{mol_id}_node_features_mapped.txt")
    if not os.path.exists(mapped_file):
        # Fall back to cebe_mapped.txt (eval data naming)
        mapped_file = os.path.join(cebe_dir, f"{mol_id}_cebe_mapped.txt")
    
    try:
        mapped_data = np.loadtxt(mapped_file)
        if mapped_data.ndim == 1:  # Single row
            mapped_data = mapped_data.reshape(1, -1)
    except OSError as exc:
        raise RuntimeError(f"[{mol_id}] missing node_features_mapped.txt or cebe_mapped.txt file: {exc}") from exc
    
    # mapped_data[:, 0] contains the carbon indices (c_idx+1) or 0 for non-carbons
    # Each row corresponds to an atom in XYZ order
    carbon_idx_mapping = mapped_data[:, 0].astype(int)  # Column 1: carbon index
    cebe_feat = mapped_data[:, 1]  # Column 2: binding energy or -1.0
    
    # ---- Load all singlet and triplet spectra for this molecule ----
    sing_spectra = {}  # keyed by carbon index (1-indexed: c1, c2, c3, ...)
    trip_spectra = {}
    
    # Determine max carbon index in this molecule
    max_c_idx = int(np.max(carbon_idx_mapping))
    
    for c_idx in range(1, max_c_idx + 1):
        sing_path = os.path.join(
            auger_dir, f"{mol_id}_auger_singlet_c{c_idx}.auger.spectrum.out"
        )
        trip_path = os.path.join(
            auger_dir, f"{mol_id}_auger_triplet_c{c_idx}.auger.spectrum.out"
        )
        
        # For eval data, try alternative naming convention (mcpdft_hybrid_rcc)
        if not os.path.exists(sing_path):
            sing_path = os.path.join(
                auger_dir, f"{mol_id}_mcpdft_hybrid_rcc_singlet_c{c_idx}.auger.spectrum.out"
            )
        if not os.path.exists(trip_path):
            trip_path = os.path.join(
                auger_dir, f"{mol_id}_mcpdft_hybrid_rcc_triplet_c{c_idx}.auger.spectrum.out"
            )

        # ---- read both spectra (skip on any error) ----
        sing_arr = None
        trip_arr = None
        
        try:
            sing_arr = np.loadtxt(sing_path)
            if sing_arr.size == 0:
                raise ValueError("empty spectrum")
            if sing_arr.ndim == 1:
                sing_arr = sing_arr.reshape(1, -1)
        except (OSError, ValueError):
            pass
        
        try:
            trip_arr = np.loadtxt(trip_path)
            if trip_arr.size == 0:
                raise ValueError("empty spectrum")
            if trip_arr.ndim == 1:
                trip_arr = trip_arr.reshape(1, -1)
        except (OSError, ValueError):
            pass
        
        # Skip if neither singlet nor triplet available
        if sing_arr is None and trip_arr is None:
            continue
        
        # Skip if only one of singlet or triplet is available (need both for combined spectrum)
        if sing_arr is None:
            print(f"[{mol_id}] Warning: skipping carbon c{c_idx} (missing singlet spectrum, only triplet available)")
            continue
        elif trip_arr is None:
            print(f"[{mol_id}] Warning: skipping carbon c{c_idx} (missing triplet spectrum, only singlet available)")
            continue
        
        sing_spectra[c_idx] = sing_arr
        trip_spectra[c_idx] = trip_arr
    
    # ---- build per-atom fitted spectra using the mapping ----
    # For each atom in XYZ order, create a fitted combined spectrum
    for atom_idx, c_idx in enumerate(carbon_idx_mapping):
        if c_idx > 0 and c_idx in sing_spectra:
            # This is a carbon atom with spectra
            sing_arr = sing_spectra[c_idx]
            trip_arr = trip_spectra[c_idx]
            
            # Extract energy and intensity from both spectra
            sing_energy = sing_arr[:, 0]
            sing_intensity = sing_arr[:, 1]
            trip_energy = trip_arr[:, 0]
            trip_intensity = trip_arr[:, 1]
            
            # Combine singlet and triplet peaks
            combined_energy = np.concatenate([sing_energy, trip_energy])
            combined_intensity = np.concatenate([sing_intensity, trip_intensity])
            
            # Fit combined spectrum to grid
            energy_grid, intensity_grid = fit_spectrum_to_grid(
                combined_energy, combined_intensity,
                fwhm=fwhm,
                energy_min=energy_min,
                energy_max=energy_max,
                n_points=n_points
            )
            
            # Stack energy and intensity as columns
            fitted_spectrum = np.column_stack((energy_grid, intensity_grid)).astype(np.float32)
            combined_nl.append(fitted_spectrum)
        else:
            # Non-carbon atom or missing spectrum - use empty fitted spectrum
            # Only warn if this is a carbon atom with missing spectrum
            if carbon_idx_mapping[atom_idx] > 0:
                print(f"[{mol_id}] Warning: carbon c{carbon_idx_mapping[atom_idx]} (atom index {atom_idx}) has no spectrum, using empty spectrum")
            energy_grid = np.linspace(energy_min, energy_max, n_points)
            empty_spectrum = np.column_stack((
                energy_grid, 
                np.zeros(n_points, dtype=np.float32)
            )).astype(np.float32)
            combined_nl.append(empty_spectrum)

    return (
        combined_nl,
        cebe_feat
    )


def extract_spectra_stick(
    mol_id: str,
    atoms: dict,
    atom_list: list[str],
    *,
    auger_dir: str,
    cebe_dir: str,
) -> tuple[
    list[np.ndarray], np.ndarray
]:
    """
    Load raw stick spectra (energy, intensity pairs) for every carbon — NO broadening.

    Returns the combined singlet+triplet stick peaks for each atom as
    variable-length arrays.  Non-carbon atoms get an empty (0, 2) array.

    The returned data can later be broadened on-the-fly at training or
    analysis time via ``fit_spectrum_to_grid()``.

    Uses the same mapped-file logic as ``extract_spectra_fitted()``.

    Parameters
    ----------
    mol_id : str
        Molecule ID.
    atoms : dict
        Atom information.
    atom_list : list[str]
        List of atom symbols.
    auger_dir : str
        Directory containing Auger spectrum files.
    cebe_dir : str
        Directory containing CEBE mapping files.

    Returns
    -------
    stick_spectra : list[np.ndarray]
        Per-atom stick spectra.  Each element is an (N_peaks, 2) float32
        array with columns [energy, intensity].  N_peaks varies per atom;
        non-carbon atoms get shape (0, 2).
    cebe_feat : np.ndarray
        Binding energy features from the mapping file.
    """

    stick_spectra: list[np.ndarray] = []

    # ---- Load mapping ----
    mapped_file = os.path.join(cebe_dir, f"{mol_id}_node_features_mapped.txt")
    if not os.path.exists(mapped_file):
        mapped_file = os.path.join(cebe_dir, f"{mol_id}_cebe_mapped.txt")

    try:
        mapped_data = np.loadtxt(mapped_file)
        if mapped_data.ndim == 1:
            mapped_data = mapped_data.reshape(1, -1)
    except OSError as exc:
        raise RuntimeError(
            f"[{mol_id}] missing node_features_mapped.txt or cebe_mapped.txt: {exc}"
        ) from exc

    carbon_idx_mapping = mapped_data[:, 0].astype(int)
    cebe_feat = mapped_data[:, 1]

    # ---- Load all singlet/triplet spectra ----
    sing_spectra: dict[int, np.ndarray] = {}
    trip_spectra: dict[int, np.ndarray] = {}
    max_c_idx = int(np.max(carbon_idx_mapping))

    for c_idx in range(1, max_c_idx + 1):
        sing_path = os.path.join(
            auger_dir, f"{mol_id}_auger_singlet_c{c_idx}.auger.spectrum.out"
        )
        trip_path = os.path.join(
            auger_dir, f"{mol_id}_auger_triplet_c{c_idx}.auger.spectrum.out"
        )

        # Eval naming fallback
        if not os.path.exists(sing_path):
            sing_path = os.path.join(
                auger_dir,
                f"{mol_id}_mcpdft_hybrid_rcc_singlet_c{c_idx}.auger.spectrum.out",
            )
        if not os.path.exists(trip_path):
            trip_path = os.path.join(
                auger_dir,
                f"{mol_id}_mcpdft_hybrid_rcc_triplet_c{c_idx}.auger.spectrum.out",
            )

        sing_arr = None
        trip_arr = None

        try:
            sing_arr = np.loadtxt(sing_path)
            if sing_arr.size == 0:
                raise ValueError("empty spectrum")
            if sing_arr.ndim == 1:
                sing_arr = sing_arr.reshape(1, -1)
        except (OSError, ValueError):
            pass

        try:
            trip_arr = np.loadtxt(trip_path)
            if trip_arr.size == 0:
                raise ValueError("empty spectrum")
            if trip_arr.ndim == 1:
                trip_arr = trip_arr.reshape(1, -1)
        except (OSError, ValueError):
            pass

        if sing_arr is None and trip_arr is None:
            continue
        if sing_arr is None:
            print(f"[{mol_id}] Warning: skipping c{c_idx} (missing singlet)")
            continue
        if trip_arr is None:
            print(f"[{mol_id}] Warning: skipping c{c_idx} (missing triplet)")
            continue

        sing_spectra[c_idx] = sing_arr
        trip_spectra[c_idx] = trip_arr

    # ---- Build per-atom stick spectra ----
    for atom_idx, c_idx in enumerate(carbon_idx_mapping):
        if c_idx > 0 and c_idx in sing_spectra:
            combined = np.vstack([
                sing_spectra[c_idx],
                trip_spectra[c_idx],
            ]).astype(np.float32)
            stick_spectra.append(combined)
        else:
            if carbon_idx_mapping[atom_idx] > 0:
                print(
                    f"[{mol_id}] Warning: carbon c{carbon_idx_mapping[atom_idx]} "
                    f"(atom {atom_idx}) has no spectrum, using empty stick"
                )
            stick_spectra.append(np.empty((0, 2), dtype=np.float32))

    return stick_spectra, cebe_feat


def extract_spectra_experimental(
    mol_id: str,
    atoms: dict,
    atom_list: list[str],
    *,
    exp_dir: str,
    cebe_dir: str,
    energy_min: float = 200.0,
    energy_max: float = 270.0,
    n_points: int = 1401,
    smooth_fwhm: float = 0.0
) -> tuple[
    list[np.ndarray], np.ndarray
]:
    """
    Load experimental Auger spectra for molecules with equivalent carbon environments.
    
    For molecules where all carbons share the same environment (e.g., benzene, ethane,
    methane), loads the single experimental spectrum from {mol_id}_exp.txt and 
    duplicates it to all carbon atom positions.
    
    Resample experimental spectra to the standard energy grid. Optionally apply
    Gaussian smoothing to match the broadening of calculated spectra.
    
    Parameters
    ----------
    mol_id : str
        Molecule ID
    atoms : dict
        Atom information
    atom_list : list[str]
        List of atom symbols
    exp_dir : str
        Directory containing experimental spectra files ({mol_id}_exp.txt)
    cebe_dir : str
        Directory containing CEBE mapping files
    energy_min : float, default=200.0
        Minimum energy for fitted grid (eV)
    energy_max : float, default=270.0
        Maximum energy for fitted grid (eV)
    n_points : int, default=1401
        Number of points in the fitted grid (0.05 eV spacing for 200-270 eV)
    smooth_fwhm : float, default=0.0
        If > 0, apply Gaussian smoothing with this FWHM (in eV) to experimental spectra.
        Set to match the FWHM used for calculated spectra (e.g., 3.768 eV) to reduce
        domain gap between calculated and experimental data.

    Returns
    -------
    exp_nl : list[np.ndarray]
        Per-atom experimental spectra, shape (n_atoms, n_points, 2)
        Each spectrum is [energy_grid, intensity_grid] as columns
        Carbon atoms get the experimental spectrum, non-carbons get zeros
        No mask needed since spectra are on a common grid
    cebe_feat : np.ndarray
        Binding energy features from the mapping file
    """
    
    exp_nl = []
    
    # ---- Load mapping from cebe_mapped.txt ----
    mapped_file = os.path.join(cebe_dir, f"{mol_id}_cebe_mapped.txt")
    
    try:
        mapped_data = np.loadtxt(mapped_file)
        if mapped_data.ndim == 1:  # Single row
            mapped_data = mapped_data.reshape(1, -1)
    except OSError as exc:
        raise RuntimeError(f"[{mol_id}] missing cebe_mapped.txt file: {exc}") from exc
    
    # mapped_data[:, 0] contains the carbon indices (1 for carbon, 0 for non-carbon)
    # mapped_data[:, 1] contains binding energies
    carbon_idx_mapping = mapped_data[:, 0].astype(int)  # Column 1: carbon index
    cebe_feat = mapped_data[:, 1]  # Column 2: binding energy or -1.0
    
    # Count carbons (atoms with carbon_idx > 0)
    n_carbons = int(np.sum(carbon_idx_mapping > 0))
    
    # ---- Load experimental spectrum ----
    exp_path = os.path.join(exp_dir, f"{mol_id}_exp.txt")
    
    try:
        exp_data = np.loadtxt(exp_path)
        if exp_data.ndim == 1:  # Single data point
            exp_data = exp_data.reshape(1, -1)
        
        # exp_data should be (n_exp_points, 2): [energy, intensity]
        if exp_data.shape[1] != 2:
            raise ValueError(f"Expected shape (n_points, 2), got {exp_data.shape}")
        
        exp_energies = exp_data[:, 0]
        exp_intensities = exp_data[:, 1]
        
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"[{mol_id}] failed to load experimental spectrum: {exc}") from exc
    
    # ---- Resample to standard energy grid via linear interpolation ----
    energy_grid = np.linspace(energy_min, energy_max, n_points)
    
    # Clip experimental data to grid range (avoid extrapolation)
    valid_idx = (exp_energies >= energy_min) & (exp_energies <= energy_max)
    exp_e_clipped = exp_energies[valid_idx]
    exp_i_clipped = exp_intensities[valid_idx]
    
    if len(exp_e_clipped) < 2:
        raise ValueError(f"[{mol_id}] insufficient experimental data points in energy range {energy_min}-{energy_max} eV")
    
    # Linear interpolation to standard grid
    resampled_intensity = np.interp(
        energy_grid, 
        exp_e_clipped, 
        exp_i_clipped,
        left=0.0,  # Pad with zeros outside data range
        right=0.0
    ).astype(np.float32)
    
    # ---- Optional Gaussian smoothing to match calculated spectra ----
    if smooth_fwhm > 0:
        from scipy.ndimage import gaussian_filter1d
        
        # Convert FWHM to sigma: FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.355 * sigma
        # Also need to convert from eV to grid points
        energy_spacing = (energy_max - energy_min) / (n_points - 1)
        sigma_eV = smooth_fwhm / 2.355
        sigma_points = sigma_eV / energy_spacing
        
        # Apply Gaussian smoothing
        resampled_intensity = gaussian_filter1d(
            resampled_intensity, 
            sigma=sigma_points, 
            mode='nearest'
        ).astype(np.float32)
        
        # Re-normalize to [0, 1] after smoothing
        if resampled_intensity.max() > 0:
            resampled_intensity = resampled_intensity / resampled_intensity.max()
    
    # ---- Build per-atom spectra ----
    # For molecules with equivalent carbons, all carbons get the same spectrum
    n_atoms = len(atom_list)
    
    for atom_idx in range(n_atoms):
        if carbon_idx_mapping[atom_idx] > 0:  # This is a carbon atom
            # Assemble experimental spectrum: [energy_grid, intensity_grid]
            exp_spectrum = np.column_stack((energy_grid, resampled_intensity)).astype(np.float32)
            exp_nl.append(exp_spectrum)
        else:
            # Non-carbon atom: use empty spectrum
            empty_spectrum = np.column_stack((
                energy_grid,
                np.zeros(n_points, dtype=np.float32)
            )).astype(np.float32)
            exp_nl.append(empty_spectrum)
    
    return (
        exp_nl,
        cebe_feat
    )


