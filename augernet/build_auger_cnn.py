"""
Auger CNN Data Processing Module

Unified data processing for:
- Calculated training data (auger_db)
- Evaluation data (auger_eval - calculated spectra)
- Experimental data (auger_eval - experimental spectra)

Minimal PyTorch Data objects containing only essential components:
- y: Spectra (n_atoms, 2*n_points) flattened [E1,...,En_points, I1,...,In_points]
- carbon_env_labels: Carbon type indices (-1 for non-carbons)
- carbon_env_onehot: One-hot carbon environment encoding
- morgan_fp: Per-atom Morgan fingerprints
- e_neg_scores: Electronegativity difference scores, for environment analysis only
- atomic_be: Atomic 1s binding energies
- mask_bin: Binary mask for valid spectral points
- node_mask: Mask indicating which atoms are carbons
- name: Molecule name/identifier
"""

import os
import json
import re
import numpy as np
import torch
from typing import Tuple
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdmolops, AddHs, rdDetermineBonds
from rdkit.Chem.rdmolops import RenumberAtoms
from skipatom import SkipAtomInducedModel

from augernet import utils, spec_utils, eneg_diff as ed
from augernet import build_molecular_graphs
from augernet.carbon_environment import (
    get_all_carbon_environment_labels,
    get_all_atom_morgan_fingerprints,
    CARBON_ENV_TO_IDX,
    NUM_CARBON_CATEGORIES,
    print_category_distribution
)

_EN_MAT = ed.get_eleneg_diff_mat(num_elements=100)
au2eV = 27.21139

# ============================================================================
# ESSENTIAL HELPER FUNCTIONS
# ============================================================================

def _e_neg_scores(smiles: str, add_bonds=True):
    """Return {atom_idx: Pauling-difference score} for every heavy atom.

    .. deprecated::
        Use :func:`_e_neg_scores_from_mol` instead to avoid atom-ordering
        mismatches between SMILES-canonical and XYZ orderings.
    """
    vecs = ed.get_full_neighbor_vectors(smiles, add_bonds=True)
    scores = {}
    for idx, symbol, vec in vecs:
        z = Chem.Atom(symbol).GetAtomicNum() - 1
        lmat = np.zeros((1, 100))
        lmat[0, z] = 1
        scores[idx] = float(np.einsum('ij,jk,ki->i', lmat, _EN_MAT, vec)[0])
    return scores


def _e_neg_scores_from_mol(mol, add_bonds=True, num_elements=100):
    """Compute Pauling electronegativity-difference scores directly from an RDKit Mol.

    Corrected version that uses the mol object's own atom ordering instead of
    re-parsing a SMILES string (which creates canonical ordering that disagrees
    with XYZ ordering).

    Parameters
    ----------
    mol : RDKit Mol
        Molecule with hydrogens (``Chem.AddHs`` already applied).
    add_bonds : bool
        Weight neighbour counts by bond order (default: True).
    num_elements : int
        Size of the neighbour vector (default: 100).

    Returns
    -------
    dict[int, float]
        ``{atom_idx: e_neg_score}`` for every atom in ``mol``.
    """
    scores = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        z = atom.GetAtomicNum() - 1

        vec = np.zeros((num_elements, 1), dtype=float)
        for nbr in atom.GetNeighbors():
            nbr_z = nbr.GetAtomicNum()
            if nbr_z <= num_elements:
                if add_bonds:
                    bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
                    bo = bond.GetBondTypeAsDouble()
                else:
                    bo = 1.0
                vec[nbr_z - 1, 0] += bo

        lmat = np.zeros((1, num_elements))
        lmat[0, z] = 1
        scores[idx] = float(np.einsum('ij,jk,ki->i', lmat, _EN_MAT, vec)[0])

    return scores


def giveorbitalenergy(ele, orb, orbital_energy_file='orbitalenergy.json'):
    """Get 1s orbital energy for an element in eV."""
    with open(orbital_energy_file, 'r') as f:
        data = json.load(f)
    try:
        orbenegele = data[ele]
    except KeyError:
        raise KeyError(f"Element {ele} not found in orbital energy file")
    
    # Get n and l from orbital string (e.g., "1s" -> n=1, l=0)
    n = int(orb[0])
    l = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4}.get(orb[1], 0)
    
    cbenergy = orbenegele[str(l)][n - l - 1]
    return cbenergy * au2eV


def extract_carbon_features(mol, morgan_radius=2, morgan_bits=1024):
    """
    Extract single-label carbon environment classification and Morgan fingerprints.
    
    Each carbon is assigned to exactly ONE category based on hierarchical SMARTS matching.
    Returns: carbon environment labels (one-hot encoded) and per-atom Morgan fingerprints.
    """
    carbon_env_names, carbon_env_labels, carbon_env_onehot = get_all_carbon_environment_labels(mol)
    morgan_fps = get_all_atom_morgan_fingerprints(mol, radius=morgan_radius, n_bits=morgan_bits)
    
    return (
        carbon_env_names,
        np.array(carbon_env_labels, dtype=np.int64),
        carbon_env_onehot.astype(np.float32),
        morgan_fps.astype(np.float32)
    )


def build_node_features_auger(mol, xyz_path, orbital_energy_file, mol_smiles=None):
    """Build simplified node features: electronegativity scores and atomic binding energies."""
    # Get electronegativity scores directly from mol object (correct atom ordering)
    e_score_dict = _e_neg_scores_from_mol(mol)
    
    # Extract features for each atom
    e_neg_scores = []
    atomic_be_values = []
    
    for iatom, atom in enumerate(mol.GetAtoms()):
        atom_symbol = atom.GetSymbol()
        
        # Electronegativity score
        e_neg_scores.append(e_score_dict.get(iatom, 0.0))
        
        # Atomic binding energy (1s, in eV)
        atom_be = -giveorbitalenergy(atom_symbol, "1s", orbital_energy_file)
        atomic_be_values.append(atom_be)
    
    return np.array(e_neg_scores, dtype=np.float32), np.array(atomic_be_values, dtype=np.float32)


def load_auger_eval_mol(fname):
    """Load Auger evaluation XYZ file with labeled atoms (C1, O2, etc.).
    
    .. deprecated::
        Use ``build_molecular_graphs.mol_from_xyz_order(fname, labeled_atoms=True)`` instead.
    """
    if not os.path.exists(fname):
        raise FileNotFoundError(f"XYZ file not found: {fname}")
    
    lines = open(fname).read().splitlines()
    n_atoms = int(lines[0])
    
    # Extract atoms and coordinates
    xyz_lines = lines[2:2 + n_atoms]
    xyz_symbols = []
    xyz_coords = []
    
    for line in xyz_lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        label = parts[0]
        element = ''.join([c for c in label if c.isalpha()])
        xyz_symbols.append(element)
        xyz_coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    xyz_coords = np.array(xyz_coords)
    
    # Create molecule from element list and coordinates
    mol = Chem.RWMol()
    for symbol in xyz_symbols:
        atom = Chem.Atom(symbol)
        mol.AddAtom(atom)
    
    mol = mol.GetMol()
    conf = Chem.Conformer(len(xyz_symbols))
    for i in range(len(xyz_symbols)):
        conf.SetAtomPosition(i, xyz_coords[i])
    
    mol.RemoveAllConformers()
    mol.AddConformer(conf)
    
    # Infer bonds from 3D geometry
    mol.ClearComputedProps()
    mol.UpdatePropertyCache(strict=False)

    try:
        # Use DetermineBonds which determines both connectivity and bond orders
        # This properly identifies C=O double bonds, C=C double bonds, etc.
        rdDetermineBonds.DetermineBonds(mol, charge=0)
    except Exception as e:
        # Fall back to just connectivity if bond order determination fails
        print(f"  Warning: DetermineBonds failed ({e}), falling back to DetermineConnectivity")
        rdDetermineBonds.DetermineConnectivity(mol)
    
    # Sanitize and perceive aromaticity
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        # Try again with less strict sanitization
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ 
                         Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    
    # Explicitly set aromaticity for ring systems
    # This ensures aromatic patterns 'c' work in SMARTS
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
    
    mol = Chem.AddHs(mol)
    
    # Generate SMILES
    smiles = Chem.MolToSmiles(mol)
    
    return mol, xyz_symbols, xyz_coords, smiles


def load_qm9_ordered_mol(xyz_path):
    """Load QM9 XYZ file and create RDKit molecule with correct atom ordering.
    
    .. deprecated::
        Use ``build_molecular_graphs.mol_from_xyz_order(xyz_path, labeled_atoms=False)``
        instead, which uses DetermineBonds + MDL aromaticity for consistent
        environment classification across GNN and CNN pipelines.
    """
    import warnings
    warnings.warn(
        "load_qm9_ordered_mol is deprecated. Use "
        "build_molecular_graphs.mol_from_xyz_order(xyz_path, labeled_atoms=False) instead.",
        DeprecationWarning, stacklevel=2
    )
    ind, gdb_smi, relax_smi = build_molecular_graphs.cleanup_qm9_xyz(xyz_path)
    xyz_lines = ind.splitlines()
    
    n_atoms = int(xyz_lines[0])
    
    # Parse coordinates and symbols
    xyz_symbols = []
    xyz_coords = []
    for i in range(2, 2 + n_atoms):
        parts = xyz_lines[i].split()
        if len(parts) >= 4:
            xyz_symbols.append(parts[0])
            xyz_coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    xyz_coords = np.array(xyz_coords)
    
    # Create molecule from SMILES
    mol = Chem.MolFromSmiles(gdb_smi)
    if mol is None:
        mol = Chem.MolFromSmiles(relax_smi)
    if mol is None:
        raise ValueError(f"Could not parse SMILES for {xyz_path}")
    
    mol = Chem.AddHs(mol)
    
    # Get SMILES atom symbols
    smiles_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    
    # Create permutation map
    permutation = []
    used = set()
    for sym in xyz_symbols:
        for i, s in enumerate(smiles_symbols):
            if i not in used and s == sym:
                permutation.append(i)
                used.add(i)
                break
    
    mol = RenumberAtoms(mol, permutation)
    
    # Assign QM9 coordinates
    conf = Chem.Conformer(n_atoms)
    for i in range(n_atoms):
        conf.SetAtomPosition(i, xyz_coords[i])
    
    mol.RemoveAllConformers()
    mol.AddConformer(conf)
    Chem.SanitizeMol(mol)
    
    return mol, gdb_smi, relax_smi, xyz_symbols, xyz_coords


# ============================================================================
# HELPER: CREATE DATA OBJECT FROM SPECTRA AND MOLECULE
# ============================================================================

def _compute_cdf_from_spectra(nodelabel) -> np.ndarray:
    """
    Compute cumulative distribution function from intensity spectra.
    
    Parameters
    ----------
    nodelabel : list[np.ndarray] or np.ndarray
        If list: List of spectrum arrays, each shape (n_spec_pts, 2)
        If array: Shape (n_atoms, n_spec_pts, 2) where last dimension is [energy, intensity]
    
    Returns
    -------
    np.ndarray
        CDF values, shape (n_atoms, n_spec_pts)
        Values range from 0 to 1, representing cumulative sum of normalized intensity
    """
    # Convert list to array if needed
    if isinstance(nodelabel, list):
        nodelabel = np.array(nodelabel, dtype=np.float32)  # Shape: (n_atoms, n_spec_pts, 2)
    else:
        nodelabel = np.array(nodelabel, dtype=np.float32)
    
    n_atoms = nodelabel.shape[0]
    n_spec_pts = nodelabel.shape[1]
    cdf = np.zeros((n_atoms, n_spec_pts), dtype=np.float32)
    
    for atom_idx in range(n_atoms):
        # Extract intensity channel (second channel)
        intensity = nodelabel[atom_idx, :, 1]  # Shape: (n_spec_pts,)
        
        # Normalize intensity to [0, 1]
        intensity_min = intensity.min()
        intensity_max = intensity.max()
        
        if intensity_max > intensity_min:
            intensity_norm = (intensity - intensity_min) / (intensity_max - intensity_min)
        else:
            intensity_norm = np.zeros_like(intensity)
        
        # Compute cumulative sum
        cumsum = np.cumsum(intensity_norm)
        
        # Normalize to [0, 1] range
        if cumsum[-1] > 0:
            cdf[atom_idx] = cumsum / cumsum[-1]
        else:
            cdf[atom_idx] = np.zeros_like(cumsum)
    
    return cdf


def _create_data_object(mol, nodelabel, mol_name, orbital_energy_file, 
                       cebe, morgan_radius, morgan_bits, name_suffix='',
                       delta_be=None, delta_be_norm=None, 
                       e_neg_scores=None, atomic_be_values=None,
                       smiles=None, spectrum_type='fitted'):
    """
    Create a minimal Data object from molecule and spectra.
    
    This consolidates the common pipeline: extract features → convert to tensor
    → create Data object.
    
    Includes:
    - y: Flattened spectra (n_atoms, 2*n_spec_pts) with [energy, intensity] interleaved
         (only for spectrum_type='fitted')
    - stick_spectra: list of (N_peaks, 2) arrays with raw stick peaks per atom
         (only for spectrum_type='stick')
    - cdf: Cumulative distribution function of intensity (n_atoms, n_spec_pts)
         (only for spectrum_type='fitted')
    - delta_be: Raw binding energy difference (n_atoms,)
    - delta_be_norm: Normalized delta_be (z-score) using global statistics (n_atoms,)
    
    Note: Augmentation (prepending delta_be to spectra) is handled in the
    AugerSpectralDataset class during training/inference, not during data generation.
    
    Parameters
    ----------
    mol : RDKit Mol
        Molecule
    nodelabel : array-like
        For spectrum_type='fitted': Spectra array (n_atoms, n_spec_pts, 2)
        For spectrum_type='stick': list of (N_peaks, 2) arrays per atom
    mol_name : str
        Molecule name
    orbital_energy_file : str
        Path to orbital energy JSON file
    cebe : float or array-like
        CEBE value(s)
    morgan_radius : int
        Morgan fingerprint radius
    morgan_bits : int
        Morgan fingerprint bits
    name_suffix : str
        Suffix to append to molecule name
    delta_be : np.ndarray, optional
        Precomputed raw delta_be. If None, computed from atomic_be - cebe
    delta_be_norm : np.ndarray, optional
        Precomputed normalized delta_be (z-score normalized). If None, not included
    e_neg_scores : np.ndarray, optional
        Precomputed electronegativity scores. If None, computed
    atomic_be_values : np.ndarray, optional
        Precomputed atomic binding energies. If None, computed
    spectrum_type : str, default='fitted'
        'fitted' or 'stick'. Controls how spectra are stored.
    """
    # Build node features if not provided
    if e_neg_scores is None or atomic_be_values is None:
        e_neg_scores, atomic_be_values = build_node_features_auger(
            mol, None, orbital_energy_file,
            mol_smiles=Chem.MolToSmiles(mol)
        )
    
    # Compute delta_be if not provided
    if delta_be is None:
        delta_be = atomic_be_values - cebe
    
    # Extract carbon environment features
    carbon_env_names, carbon_env_labels, carbon_env_onehot, morgan_fps = extract_carbon_features(
        mol, morgan_radius, morgan_bits
    )
    
    # Node mask (carbon atoms)
    node_mask = torch.FloatTensor([1.0 if c != -1 else 0.0 for c in carbon_env_labels])
    
    # Generate SMILES if not provided
    if smiles is None:
        smiles = Chem.MolToSmiles(mol)

    # Create minimal Data object
    data_dict = {
        'carbon_env_labels': torch.LongTensor(carbon_env_labels),
        'carbon_env_onehot': torch.FloatTensor(carbon_env_onehot),
        'morgan_fp': torch.FloatTensor(morgan_fps),
        'e_neg_scores': torch.FloatTensor(e_neg_scores),
        'atomic_be': torch.FloatTensor(atomic_be_values),
        'delta_be': torch.FloatTensor(delta_be),
        'node_mask': node_mask,
        'name': mol_name + name_suffix,
        'smiles': smiles,
        'spectrum_type': spectrum_type,
    }

    if spectrum_type == 'stick':
        # Store variable-length stick spectra as a Python list
        # Each element is an (N_peaks, 2) float32 array
        data_dict['stick_spectra'] = [np.array(s, dtype=np.float32) for s in nodelabel]
    else:
        # Legacy fitted path — rectangular tensors
        y = np.array(nodelabel, dtype=np.float32)
        y = torch.FloatTensor(y)
        
        n_atoms_mol, n_spec_pts, _ = y.shape
        mask_rows = torch.ones((n_atoms_mol, n_spec_pts), dtype=torch.float32)
        mask_flat = mask_rows.repeat(1, 2)
        y_flat = y.reshape(n_atoms_mol, 2 * n_spec_pts)
        
        # Compute CDF from spectra
        y_np = nodelabel  # Original numpy array (n_atoms, n_spec_pts, 2)
        cdf_np = _compute_cdf_from_spectra(y_np)
        cdf = torch.FloatTensor(cdf_np)
        
        data_dict['y'] = y_flat
        data_dict['cdf'] = cdf
        data_dict['mask_bin'] = mask_flat
    
    # Add normalized delta_be if provided
    if delta_be_norm is not None:
        data_dict['delta_be_norm'] = torch.FloatTensor(delta_be_norm)
    
    data = Data(**data_dict)
    return data


# ============================================================================
# UNIFIED DATA PROCESSING FUNCTION
# ============================================================================

def process_auger_data(
    raw_dir,
    data_type='calc',
    n_points=731,
    e_max=273,
    e_min=200,
    morgan_radius=2,
    morgan_bits=1024,
    gaussian_fwhm=3.768,
    DEBUG=False,
    delta_be_stats=None,
    exp_smooth_fwhm=0.0,
    spectrum_type='fitted',
):
    """
    Unified Auger data processing for all data types.
    
    Creates minimal PyTorch Data objects with only essential components.
    
    Parameters
    ----------
    raw_dir : str
        Root directory for raw data
    data_type : str
        Type of data to process: 'calc', 'eval', or 'exp'
        - 'calc': Calculated training data (auger_db with fitted spectra)
        - 'eval': Evaluation data (auger_eval with calculated fitted spectra)
        - 'exp': Experimental data (auger_eval with experimental spectra)
    n_points : int
        Number of spectral points
    morgan_radius : int
        Radius for Morgan fingerprints
    morgan_bits : int
        Number of bits for Morgan fingerprints
    gaussian_fwhm : float
        Full-width at half-maximum (FWHM) of Gaussian broadening in eV (default: 3.768)
    DEBUG : bool
        Debug mode (process one molecule)
    delta_be_stats : tuple, optional
        Pre-computed (mean, std) for delta_be normalization.
        If provided, use these instead of computing from current data.
        IMPORTANT: For eval/exp data, pass the stats from calc (training) data
        to ensure consistent normalization across datasets.
    exp_smooth_fwhm : float, optional
        FWHM for Gaussian smoothing of experimental spectra (in eV).
        Set to match gaussian_fwhm (e.g., 3.768 eV) to reduce domain gap
        between calculated and experimental spectra. Default: 0.0 (no smoothing).
    spectrum_type : str, default='fitted'
        How to store spectra in the Data objects:
        - 'fitted': Apply Gaussian broadening during preparation (legacy).
        - 'stick':  Store raw stick peaks (variable-length) for each atom.
                    Broadening is deferred to training / analysis time.
                    For 'exp' data_type, the experimental spectrum is resampled
                    to the standard grid (broadening is inherent in exp data).
    
    Returns
    -------
    tuple : (data_list, delta_be_stats)
        - data_list: List of minimal PyTorch Geometric Data objects
        - delta_be_stats: Tuple of (mean, std) used for normalization
          (pass this to eval/exp processing for consistent normalization)
    """
    
    orbital_energy_file = os.path.join(raw_dir, 'orbitalenergy.json')
    
    data_list = []
    
    # ========================================================================
    # CONFIGURE DATA TYPE SPECIFIC PARAMETERS
    # ========================================================================
    
    if data_type == 'calc':
        # Calculated training data from QM9
        auger_dir = os.path.join(raw_dir, "auger_db")
        xyz_dir = os.path.join(raw_dir, "xyz_db")
        cebe_dir = os.path.join(raw_dir, "cebe_db")
        
        calc_list_path = os.path.join(auger_dir, "debug.txt" if DEBUG else "auger_success_calcs.txt")
        with open(calc_list_path) as fh:
            mol_list = [ln.strip() for ln in fh if ln.strip()]
        if DEBUG:
            mol_list = mol_list[:5]
        
        # Loader function for calc data
        # Use mol_from_xyz_order (DetermineBonds + MDL aromaticity) for consistency
        # with GNN pipelines — gives identical atom ordering and environment labels
        def load_mol(mol_name):
            xyz_path = os.path.join(xyz_dir, f"{mol_name}.xyz")
            mol, _, _, smiles = build_molecular_graphs.mol_from_xyz_order(xyz_path, labeled_atoms=False)
            return mol, None, smiles
        
        # Spectrum extractor for calc data
        def extract_spectrum(mol_name, mol):
            atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
            atoms, atom_list, _ = utils.data_from_geom(
                build_molecular_graphs.cleanup_qm9_xyz(
                    os.path.join(xyz_dir, f"{mol_name}.xyz")
                )[0].splitlines()
            )[1:4]
            
            if spectrum_type == 'stick':
                nodelabel, cebe = spec_utils.extract_spectra_stick(
                    mol_name, atoms, atom_list,
                    auger_dir=auger_dir,
                    cebe_dir=cebe_dir,
                )
            else:
                nodelabel, cebe = spec_utils.extract_spectra_fitted(
                    mol_name, atoms, atom_list,
                    auger_dir=auger_dir,
                    cebe_dir=cebe_dir,
                    fwhm=gaussian_fwhm,
                    energy_min=e_min,
                    energy_max=e_max,
                    n_points=n_points
                )
            return nodelabel, cebe
        
        name_suffix = ''
        
    elif data_type == 'eval':
        # Evaluation data with fitted spectra
        eval_dir = os.path.join(raw_dir, "auger_eval")
        
        eval_list_path = os.path.join(eval_dir, "list_small.txt")
        with open(eval_list_path) as f:
            mol_list = [line.strip() for line in f]
        if DEBUG:
            mol_list = mol_list[:2]
        
        # Loader function for eval data
        def load_mol(mol_name):
            mol_path = os.path.join(eval_dir, f"{mol_name}.xyz")
            mol, _, _, smiles = build_molecular_graphs.mol_from_xyz_order(mol_path, labeled_atoms=True)
            return mol, mol_path, smiles
        
        # Spectrum extractor for eval data
        def extract_spectrum(mol_name, mol):
            atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
            atoms_dict = {}
            for symbol in atom_symbols:
                atoms_dict[symbol] = atoms_dict.get(symbol, 0) + 1
            
            if spectrum_type == 'stick':
                nodelabel, cebe = spec_utils.extract_spectra_stick(
                    mol_name, atoms_dict, atom_symbols,
                    auger_dir=eval_dir,
                    cebe_dir=eval_dir,
                )
            else:
                nodelabel, cebe = spec_utils.extract_spectra_fitted(
                    mol_name, atoms_dict, atom_symbols,
                    auger_dir=eval_dir,
                    cebe_dir=eval_dir,
                    fwhm=gaussian_fwhm,
                    energy_min=e_min,
                    energy_max=e_max,
                    n_points=n_points
                )
            return nodelabel, cebe
        
        name_suffix = ''
        
    elif data_type == 'exp':
        # Experimental spectra (optionally smoothed)
        eval_dir = os.path.join(raw_dir, "auger_eval")
        
        mol_list = [
            'acetylene',
            'benzene',
            'cyclobutane',
            'cyclopropane',
            'ethane',
            'ethylene',
            'formaldehyde',
            'formamide',
            'methane',
            'tetrafluoromethane'
        ]
        
        # Loader function for exp data
        def load_mol(mol_name):
            mol_path = os.path.join(eval_dir, f"{mol_name}.xyz")
            mol, _, _, smiles = build_molecular_graphs.mol_from_xyz_order(mol_path, labeled_atoms=True)
            return mol, mol_path, smiles
        
        # Spectrum extractor for exp data (with optional smoothing)
        def extract_spectrum(mol_name, mol):
            atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
            atoms_dict = {}
            for symbol in atom_symbols:
                atoms_dict[symbol] = atoms_dict.get(symbol, 0) + 1
            
            nodelabel, cebe = spec_utils.extract_spectra_experimental(
                mol_name, atoms_dict, atom_symbols,
                exp_dir=eval_dir,
                cebe_dir=eval_dir,
                energy_min=e_min,
                energy_max=e_max,
                n_points=n_points,
                smooth_fwhm=exp_smooth_fwhm  # Apply smoothing to match calc spectra
            )
            return nodelabel, cebe
        
        if exp_smooth_fwhm > 0:
            print(f"  Applying Gaussian smoothing (FWHM={exp_smooth_fwhm:.2f} eV) to experimental spectra")
        
        # Experimental spectra are already continuous — always store as 'fitted'
        spectrum_type = 'fitted'
        
        name_suffix = '_exp'
    
    else:
        raise ValueError(f"Unknown data_type: {data_type}. Must be 'calc', 'eval', or 'exp'")
    
    # ========================================================================
    # PASS 1: COLLECT ALL DELTA_BE VALUES TO COMPUTE GLOBAL STATISTICS
    # ========================================================================
    print(f"  Processing {len(mol_list)} molecules...", end="", flush=True)
    all_delta_be = []  # Only carbon atoms
    mol_data_cache = {}  # Cache extracted data for pass 2
    errors = []
    
    for mol_idx, mol_name in enumerate(mol_list):
        try:
            mol, mol_path, mol_smiles = load_mol(mol_name)
            nodelabel, cebe = extract_spectrum(mol_name, mol)
            
            # Extract atomic binding energies
            e_neg_scores, atomic_be_values = build_node_features_auger(
                mol, None, orbital_energy_file,
                mol_smiles=mol_smiles
            )
            
            # Compute delta_be for this molecule (all atoms)
            delta_be = atomic_be_values - cebe
            
            # Collect delta_be only for CARBON atoms (for statistics)
            atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
            carbon_delta_be = [delta_be[i] for i, sym in enumerate(atom_symbols) if sym == 'C']
            all_delta_be.extend(carbon_delta_be)
            
            # Cache for pass 2
            mol_data_cache[mol_name] = {
                'mol': mol,
                'nodelabel': nodelabel,
                'cebe': cebe,
                'e_neg_scores': e_neg_scores,
                'atomic_be_values': atomic_be_values,
                'delta_be': delta_be,
                'smiles': mol_smiles,
            }
            
            # Progress indicator (dots every 10%)
            if (mol_idx + 1) % max(1, len(mol_list) // 10) == 0:
                print(".", end="", flush=True)
            
        except Exception as e:
            errors.append((mol_name, str(e)))
            if DEBUG:
                raise
            continue
    
    print(" done")
    if errors:
        print(f"  ⚠ {len(errors)} molecules failed (first: {errors[0][0]})")
    
    # Compute or use provided statistics (carbon atoms only)
    all_delta_be = np.array(all_delta_be, dtype=np.float32)
    
    if delta_be_stats is not None:
        # Use provided statistics (from training data)
        delta_be_mean, delta_be_std = delta_be_stats
        print(f"  delta_be (C only): using training stats: mean={delta_be_mean:.2f}, std={delta_be_std:.2f}")
        print(f"  delta_be (C only): local range=[{all_delta_be.min():.2f}, {all_delta_be.max():.2f}]")
    else:
        # Compute statistics from current data (only for 'calc' training data)
        delta_be_mean = all_delta_be.mean()
        delta_be_std = all_delta_be.std()
        
        if delta_be_std == 0:
            delta_be_std = 1.0  # Avoid division by zero
        
        print(f"  delta_be (C only): mean={delta_be_mean:.2f}, std={delta_be_std:.2f}, range=[{all_delta_be.min():.2f}, {all_delta_be.max():.2f}]")
    
    # ========================================================================
    # PASS 2: CREATE DATA OBJECTS WITH NORMALIZED DELTA_BE
    # ========================================================================
    # (Using cached data from pass 1 - no repeated computation)
    
    for mol_name in mol_data_cache:
        try:
            cached_data = mol_data_cache[mol_name]
            mol = cached_data['mol']
            nodelabel = cached_data['nodelabel']
            cebe = cached_data['cebe']
            e_neg_scores = cached_data['e_neg_scores']
            atomic_be_values = cached_data['atomic_be_values']
            delta_be = cached_data['delta_be']
            mol_smiles = cached_data['smiles']
            
            # Normalize delta_be using global statistics
            delta_be_norm = (delta_be - delta_be_mean) / delta_be_std
            
            # Create Data object with normalized delta_be
            data = _create_data_object(
                mol, nodelabel, mol_name, orbital_energy_file,
                cebe, morgan_radius, morgan_bits, name_suffix,
                delta_be=delta_be,
                delta_be_norm=delta_be_norm,
                e_neg_scores=e_neg_scores,
                atomic_be_values=atomic_be_values,
                smiles=mol_smiles,
                spectrum_type=spectrum_type,
            )
            data_list.append(data)
            
        except Exception as e:
            if DEBUG:
                raise
            continue
    
    print(f"  ✓ {len(data_list)} molecules processed successfully")
    
    # Return both data and normalization statistics
    # The stats should be saved and reused for eval/exp data
    return data_list, (delta_be_mean, delta_be_std)


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def print_eval_molecule_details(mol_name, mol, data_obj):
    """
    Print detailed SMARTS assignment for each carbon in an evaluation molecule.
    
    Shows per-carbon information in a table format.
    """
    idx_to_name = {v: k for k, v in CARBON_ENV_TO_IDX.items()}
    
    # Extract data from molecule and data object
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    carbon_indices = [i for i, sym in enumerate(atom_symbols) if sym == 'C']
    
    if not carbon_indices:
        print(f"\n{mol_name}: No carbons found")
        return
    
    labels = data_obj.carbon_env_labels.tolist()
    onehot = data_obj.carbon_env_onehot.numpy()
    morgan_fps = data_obj.morgan_fp.numpy()
    
    print(f"\n{'-'*120}")
    print(f"Molecule: {mol_name}")
    print(f"Total atoms: {mol.GetNumAtoms()}, Carbons: {len(carbon_indices)}")
    print(f"{'-'*120}")
    print(f"{'XYZ Idx':<10} {'Symbol':<8} {'Label Idx':<12} {'SMARTS Category':<35} {'OnHot Idx':<12} {'Morgan FP Stats':<20}")
    print(f"{'-'*120}")
    
    for xyz_idx, atom_idx in enumerate(carbon_indices):
        symbol = atom_symbols[atom_idx]
        label_idx = labels[atom_idx]
        
        if label_idx >= 0:
            category_name = idx_to_name[label_idx]
        else:
            category_name = "non_carbon"
            label_idx = -1
        
        # Get which onehot index is set
        onehot_indices = [i for i, val in enumerate(onehot[atom_idx]) if val > 0]
        onehot_str = str(onehot_indices[0]) if onehot_indices else "N/A"
        
        # Morgan fingerprint stats
        morgan_fp = morgan_fps[atom_idx]
        fp_count = int(np.sum(morgan_fp))
        fp_density = fp_count / len(morgan_fp) if len(morgan_fp) > 0 else 0
        
        print(f"{xyz_idx:<10} {symbol:<8} {label_idx:<12} {category_name:<35} {onehot_str:<12} {fp_count} bits ({fp_density*100:.1f}%)")
    
    print(f"{'-'*120}")


def print_eval_analysis(eval_data, mol_list, raw_dir, verbose=False):
    """
    Print analysis of evaluation molecules with SMARTS assignments.
    Always prints per-molecule carbon environment summary.
    If verbose=True, prints detailed per-atom table.
    """
    eval_dir = os.path.join(raw_dir, "auger_eval")
    idx_to_name = {v: k for k, v in CARBON_ENV_TO_IDX.items()}
    
    print(f"\n  Evaluation molecule carbon environments:")
    
    for mol_idx, (mol_name, data_obj) in enumerate(zip(mol_list, eval_data)):
        mol_path = os.path.join(eval_dir, f"{mol_name}.xyz")
        try:
            mol, _, _, _ = build_molecular_graphs.mol_from_xyz_order(mol_path, labeled_atoms=True)
            
            # Get carbon environment labels
            labels = data_obj.carbon_env_labels.tolist()
            carbon_labels = [l for l in labels if l >= 0]
            
            # Count environments for this molecule
            from collections import Counter
            env_counts = Counter(carbon_labels)
            env_summary = ", ".join([f"{idx_to_name.get(idx, 'unknown')}:{cnt}" 
                                     for idx, cnt in sorted(env_counts.items())])
            
            print(f"    {mol_name}: {len(carbon_labels)} C - {env_summary}")
            
            if verbose:
                print_eval_molecule_details(mol_name, mol, data_obj)
                
        except Exception as e:
            print(f"    {mol_name}: ⚠ {e}")

