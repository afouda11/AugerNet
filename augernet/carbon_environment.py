"""
Carbon Environment Classification and Spectral Featurization Module

This module provides:
1. SMARTS-based categorical carbon environment classification
2. Per-atom Morgan fingerprint extraction for numerical representations
3. CDF (Cumulative Distribution Function) spectral transformation
4. Additional spectral featurization methods (CWT, peak features)

Author: Generated for CoreIPNET/Auger spectroscopy project
"""
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
# Suppress RDKit deprecation warnings BEFORE importing RDKit
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*MorganGenerator.*')

import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import DataStructs
from typing import Dict, List, Tuple, Optional, Union
from collections import OrderedDict


# =============================================================================
# CARBON ENVIRONMENT CLASSIFICATION
# =============================================================================

# Define carbon environment categories with SMARTS patterns
# Ordered from most specific to least specific for hierarchical matching
CARBON_ENVIRONMENT_PATTERNS = OrderedDict([
    # Carbonyl-containing groups (most specific first)
    ('C_carboxylic_acid', '[CX3](=O)[OX2H1]'),           # -COOH
    ('C_carboxylate', '[CX3](=O)[O-]'),                   # -COO-
    ('C_ester_carbonyl', '[CX3](=O)[OX2][#6]'),          # -COO-R (carbonyl C)
    ('C_amide_carbonyl', '[CX3](=O)[#7]'),                # -CO-N (carbonyl C bonded to any N)
    ('C_acyl_fluoride', '[CX3](=O)[F]'),           # -COF
    ('C_ketone', '[CX3](=O)([#6])[#6]'),                 # R-CO-R (must have 2 C neighbors)
    # Aldehyde: carbonyl C with =O and at least one H, NOT bonded to C on both sides
    # This matches R-CHO and also H2C=O (formaldehyde)
    ('C_aldehyde', '[CH1,CH2;X3](=O)'),                  # CHO or CH2O carbonyl (aldehyde incl. formaldehyde)
    
    # Nitrogen-containing carbons
    ('C_nitrile', '[CX2]#N'),                            # -C≡N
    ('C_imine', '[CX3]=[#7]'),                            # C=N (any N, incl. charged N+)
    
    # Oxygen-containing carbons (non-carbonyl)
    ('C_ether', '[CX4][OX2][#6]'),                       # C-O-C
    ('C_alcohol', '[CX4][OX2H]'),                        # C-OH
    ('C_ester_alkyl', '[#6][OX2][CX3]=O'),               # R-O-CO- (alkyl C next to O)
    
    # Halogen-containing carbons (check before generic sp3/sp2)
    # NOTE: Only fluorine kept for H,C,N,O,F dataset (removed Cl, Br, I)
    ('C_fluorinated', '[#6][F]'),                        # Any C-F (including CF4)
    
    # Nitrogen-bonded carbons (sp3)
    ('C_amine', '[CX4][NX3;H2,H1,H0;!$(NC=O)]'),        # C-N (not amide)
    
    # Unsaturated carbons
    ('C_alkyne', '[CX2;$([CX2]#[CX2])]'),                 # Either C in C≡C
    ('C_CO2', '[CX2](=[OX1])=[OX1]'),                     # O=C=O  (carbon dioxide)
    ('C_isocyanate', '[CX2](=[OX1])=[NX2]'),              # O=C=N  (isocyanate / isocyanic acid)
    ('C_carbodiimide', '[CX2](=[NX2])=[NX2]'),            # N=C=N  (carbodiimide central C)
    ('C_ketene', '[CX2](=[#6])=[OX1]'),                   # C=C=O  (ketene sp C, also carbon suboxide outer)
    ('C_ketenimine', '[CX2](=[#6])=[NX2]'),               # C=C=N  (ketenimine sp C)
    ('C_allene', '[CX2](=[#6])=[#6]'),                    # C=C=C  (central sp C of allene / cumulene)
    ('C_enol', '[CX3;$([CX3]=[CX3])][OX2H]'),            # Vinyl alcohol C=C-OH (enol)
    ('C_vinyl', '[CX3;$([CX3]=[CX3,CX2])]'),             # Either C in C=C (vinyl/alkene, incl. cumulated terminal)
    
    # Aromatic carbons - use both aromatic (c) and Kekulé ([#6]) patterns
    # to handle cases where aromaticity isn't perceived
    # Order: most specific heteroaromatic patterns first
    ('C_phenol', '[c,C;R][OX2H]'),                       # Ar-OH (ring C with -OH substituent)
    ('C_aryl_ether', '[c,C;R][OX2H0][#6]'),              # Ar-O-R (ring C with ether linkage, not phenol)
    ('C_aryl_amine', '[c,C;R][NX3;H2,H1]'),              # Ar-NH2 or Ar-NHR
    #('C_aryl_halide', '[c,C;R][F,Cl,Br,I]'),             # Ar-X
    ('C_aryl_fluoride', '[c,C;R][F]'),             # Ar-X
    ('C_aryl_nitro', '[c,C;R][NX3+](=O)[O-]'),           # Ar-NO2
    
    # Heteroaromatic carbons (C in ring with O/N as ring members)
    # NOTE: Removed aromatic with S patterns (C_arom_S, C_arom_O_S, C_arom_N_S)
    # Only keeping O and N heteroaromatic patterns for H,C,N,O,F dataset
    ('C_arom_O_N', '[c,C;R;$([#6](~[#8;R])(~[#7;R]))]'), # Aromatic C bonded to both O and N in ring (e.g., oxazole C2)
    
    # C adjacent to single heteroatom in ring
    ('C_arom_O', '[c,C;R;$([#6]~[#8;R]);!$([#6](~[#8;R])(~[#7;R]))]'),  # Aromatic C next to ring O only
    ('C_arom_N', '[c,C;R;$([#6]~[#7;R]);!$([#6](~[#7;R])(~[#8;R]))]'),  # Aromatic C next to ring N only
    
    # Generic aromatic (no heteroatom neighbors in ring, or fallback)
    ('C_aromatic', '[c,$(C1=CC=CC=C1),$(C1=CC=CC=1)]'),  # Aromatic C (carbocyclic or unmatched)
    
    # Aliphatic sp3 carbons (by hydrogen count)
    ('C_methyl', '[CH3,CH4]'),                           # -CH3 or CH4 (methane)
    ('C_methylene', '[CH2X4]'),                          # -CH2-
    ('C_methine', '[CHX4]'),                             # -CH<
    ('C_quaternary', '[CX4H0]'),                         # >C< (no H)
    
    # Fallback categories
   # ('C_sp3', '[CX4]'),                                  # Any sp3 carbon
    #('C_sp2', '[CX3]'),                                  # Any sp2 carbon
    #('C_sp', '[CX2]'),                                   # Any sp carbon
   # ('C_unknown', '[#6]'),                               # Any carbon (catch-all)
])

# Create category name to index mapping
CARBON_ENV_TO_IDX = {name: idx for idx, name in enumerate(CARBON_ENVIRONMENT_PATTERNS.keys())}
IDX_TO_CARBON_ENV = {idx: name for name, idx in CARBON_ENV_TO_IDX.items()}
NUM_CARBON_CATEGORIES = len(CARBON_ENVIRONMENT_PATTERNS)

# Explicit specificity scores for resolving multi-match conflicts.
# Higher score = more specific = wins when a carbon matches multiple patterns.
# This replaces the fragile order-dependent first-match-wins approach.
#
# Design principles:
#   - Carbonyl + substituent patterns are most specific (100-106)
#   - Aromatic + substituent patterns beat generic substituent (90-93)
#   - Heteroaromatic patterns (80-82)
#   - Generic substituent patterns: halogen, amine, O-linked (70-72)
#   - Triple/double bond patterns (60-62)
#   - Generic aromatic fallback (50)
#   - Specific aliphatic by H-count (40-43)
#   - Generic hybridization fallbacks (10-12)
#   - Catch-all (0)
CARBON_ENV_PRIORITY = {
    # Carbonyl-containing: very specific (multiple heavy atoms in pattern)
    'C_carboxylic_acid':  106,
    'C_carboxylate':      105,
    'C_ester_carbonyl':   104,
    'C_amide_carbonyl':   103,
    'C_acyl_fluoride':      102,
    'C_ketone':           101,
    'C_aldehyde':         100,
    
    # Nitrogen sp-bonded: very specific
    'C_nitrile':          95,
    'C_imine':            93,
    
    # Aromatic + substituent: more specific than bare substituent
    'C_phenol':           92,
    'C_aryl_ether':       91,
    'C_aryl_amine':       90,
    'C_aryl_fluoride':      89,
    'C_aryl_nitro':       88,
    
    # Heteroaromatic (C in ring with heteroatom)
    'C_arom_O_N':         82,
    'C_arom_O':           81,
    'C_arom_N':           80,
    
    # Oxygen-linked sp3 (non-carbonyl)
    'C_ester_alkyl':      73,
    'C_ether':            72,
    'C_alcohol':          71,
    
    # Generic halogen / amine on sp3 carbon
    'C_fluorinated':      70,
    'C_amine':            69,
    
    # Unsaturated non-aromatic
    'C_alkyne':           62,
    'C_CO2':              67,
    'C_isocyanate':       66,
    'C_carbodiimide':     66,
    'C_ketene':           65,
    'C_ketenimine':       65,
    'C_allene':           64,
    'C_enol':             63,
    'C_vinyl':            60,
    
    # Generic aromatic fallback (no substituent matched)
    'C_aromatic':         50,
    
    # Aliphatic sp3 by H-count
    'C_methyl':           43,
    'C_methylene':        42,
    'C_methine':          41,
    'C_quaternary':       40,
    
    # Generic hybridization fallbacks
    #'C_sp3':              12,
    #'C_sp2':              11,
    #'C_sp':               10,
    
    # Catch-all
    #'C_unknown':           0,
}


# Pre-compile SMARTS patterns for performance
_COMPILED_PATTERNS = []
for _name, _smarts in CARBON_ENVIRONMENT_PATTERNS.items():
    _pat = Chem.MolFromSmarts(_smarts)
    _COMPILED_PATTERNS.append((_name, _pat, CARBON_ENV_PRIORITY[_name]))


def get_carbon_environment_label(mol: Chem.Mol, atom_idx: int) -> Tuple[str, int]:
    """
    Determine the carbon environment category for a specific atom.
    
    Uses priority-scored multi-match resolution: all SMARTS patterns are tested
    against the atom, and the most specific matching pattern (highest priority
    score) wins. This avoids order-dependent classification errors where a
    generic pattern (e.g., C_fluorinated) would incorrectly shadow a more
    specific one (e.g., C_aryl_fluoride for aromatic C-F bonds).

    Returns ('non_carbon', -1) for non-carbon atoms.
    
    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object
    atom_idx : int
        Index of the atom to classify
        
    Returns
    -------
    category_name : str
        Name of the carbon environment category
    category_idx : int
        Integer index of the category (-1 for non-carbons)
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    
    # Check if carbon
    if atom.GetAtomicNum() != 6:
        return ('non_carbon', -1)
    
    # Find ALL matching patterns and pick the one with highest priority
    best_name = None
    best_priority = -1
    
    for category_name, pattern, priority in _COMPILED_PATTERNS:
        if pattern is None:
            continue
        
        # Skip patterns with lower priority than what we already found
        if priority <= best_priority:
            continue
            
        matches = mol.GetSubstructMatches(pattern)
        for match in matches:
            # Only match if the atom is at position 0 in the SMARTS match.
            # This ensures we classify the carbon that the pattern is designed
            # to identify, not neighboring atoms that happen to be part of
            # the pattern.
            if len(match) > 0 and match[0] == atom_idx:
                best_name = category_name
                best_priority = priority
                break  # Found a match for this pattern, check next pattern
    
    if best_name is not None:
        return (best_name, CARBON_ENV_TO_IDX[best_name])
    
    # Fallback: should never be reached since C_sp2/C_sp patterns catch all
    # remaining carbons, but guard against edge cases.
    # Return C_sp2 (idx 29) as a safe fallback since it's the broadest
    # remaining pattern.
    return ('C_sp2', CARBON_ENV_TO_IDX['C_sp2'])


def get_all_carbon_environment_labels(mol: Chem.Mol) -> Tuple[List[str], List[int], np.ndarray]:
    """
    Get single-label carbon environment classification for all atoms in a molecule.
    
    Each carbon is classified into exactly ONE category based on hierarchical SMARTS matching.
    Non-carbon atoms receive label index -1.
    
    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object
        
    Returns
    -------
    category_names : List[str]
        Category name for each atom
    category_indices : List[int]
        Category index for each atom (-1 for non-carbons)
    onehot_labels : np.ndarray
        One-hot encoded labels, shape (n_atoms, NUM_CARBON_CATEGORIES)
        All zeros for non-carbon atoms
    """
    n_atoms = mol.GetNumAtoms()
    category_names = []
    category_indices = []
    onehot_labels = np.zeros((n_atoms, NUM_CARBON_CATEGORIES), dtype=np.float32)
    
    for atom_idx in range(n_atoms):
        name, idx = get_carbon_environment_label(mol, atom_idx)
        category_names.append(name)
        category_indices.append(idx)
        
        if idx >= 0:  # Carbon atom - set single label
            onehot_labels[atom_idx, idx] = 1.0
    
    return category_names, category_indices, onehot_labels


# =============================================================================
# MORGAN FINGERPRINT EXTRACTION
# =============================================================================

def get_atom_morgan_fingerprint(mol: Chem.Mol, atom_idx: int, 
                                 radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """
    Extract Morgan fingerprint bits contributed by a specific atom.
    
    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object
    atom_idx : int
        Index of the atom
    radius : int
        Morgan fingerprint radius (2 = ECFP4, 3 = ECFP6)
    n_bits : int
        Number of bits in fingerprint
        
    Returns
    -------
    atom_fp : np.ndarray
        Binary fingerprint array of shape (n_bits,) with bits this atom contributes to
    """
    import warnings
    
    # Use the new RDKit MorganGenerator API (2023.09+) when available
    try:
        from rdkit.Chem import MorganGenerator
        
        # Create generator for this atom
        gen = MorganGenerator(radius=radius, nBits=n_bits)
        
        # Get fingerprint for the specific atom
        fp = gen.GetFingerprintAsNumPy()
        
        atom_fp = fp.astype(np.float32)
        
    except (ImportError, AttributeError):
        # Fall back to legacy API for older RDKit versions
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            
            bit_info = {}
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=bit_info)
        
        # Create atom-specific fingerprint
        atom_fp = np.zeros(n_bits, dtype=np.float32)
        
        for bit_idx, atom_radius_list in bit_info.items():
            for contributing_atom, r in atom_radius_list:
                if contributing_atom == atom_idx:
                    atom_fp[bit_idx] = 1.0
                    break
    
    return atom_fp


def get_all_atom_morgan_fingerprints(mol: Chem.Mol, radius: int = 2, 
                                      n_bits: int = 1024) -> np.ndarray:
    """
    Extract Morgan fingerprint contributions for all atoms in a molecule.
    
    Uses the new MorganGenerator API (RDKit 2023.09+) when available, with fallback
    to the legacy API for backward compatibility with older RDKit versions.
    
    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object
    radius : int
        Morgan fingerprint radius (2 = ECFP4, 3 = ECFP6)
    n_bits : int
        Number of bits for the fingerprint
        
    Returns
    -------
    atom_fps : np.ndarray
        Shape (n_atoms, n_bits), where each row is per-atom Morgan fingerprint
    """
    n_atoms = mol.GetNumAtoms()
    atom_fps = np.zeros((n_atoms, n_bits), dtype=np.float32)
    
    try:
        # Try the new MorganGenerator API (RDKit 2023.09+)
        from rdkit.Chem import MorganGenerator
        import warnings
        
        # Suppress deprecation warnings from Morgan fingerprint computation
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            
            # For each atom, compute its Morgan fingerprint contribution
            for atom_idx in range(n_atoms):
                gen = MorganGenerator(radius=radius, nBits=n_bits, includeChirality=False)
                
                # Generate fingerprint from this atom's perspective
                fp = gen.GetFingerprintAsNumPy()
                atom_fps[atom_idx] = fp.astype(np.float32)
    
    except (ImportError, AttributeError, TypeError) as e:
        # Fall back to legacy API with bitInfo for older RDKit versions
        import warnings
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            
            bit_info = {}
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius, nBits=n_bits, bitInfo=bit_info
            )
        
        # Create atom fingerprints by mapping bits to atoms using bit_info
        for bit_idx, atom_radius_list in bit_info.items():
            for contributing_atom, r in atom_radius_list:
                atom_fps[contributing_atom, bit_idx] = 1.0
    
    return atom_fps


def get_atom_morgan_counts(mol: Chem.Mol, atom_idx: int, 
                           radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """
    Get Morgan fingerprint with count information for a specific atom.
    
    Instead of binary, this counts how many times each bit is hit at different radii.
    
    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object
    atom_idx : int
        Index of the atom
    radius : int
        Morgan fingerprint radius
    n_bits : int
        Number of bits
        
    Returns
    -------
    atom_fp_counts : np.ndarray
        Count-based fingerprint of shape (n_bits,)
    """
    import warnings
    
    try:
        # Try the new MorganGenerator API (RDKit 2023.09+)
        from rdkit.Chem import MorganGenerator
        
        gen = MorganGenerator(radius=radius, nBits=n_bits, includeChirality=False)
        
        # Get fingerprint for this specific atom
        fp = gen.GetFingerprintAsNumPy()
        
        atom_fp_counts = fp.astype(np.float32)
    
    except (ImportError, AttributeError, TypeError):
        # Fall back to legacy API for older RDKit versions
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            
            bit_info = {}
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=bit_info)
        
        atom_fp_counts = np.zeros(n_bits, dtype=np.float32)
        
        for bit_idx, atom_radius_list in bit_info.items():
            count = sum(1 for a, r in atom_radius_list if a == atom_idx)
            atom_fp_counts[bit_idx] = count
    
    return atom_fp_counts


# =============================================================================
# SPECTRAL FEATURIZATION
# =============================================================================

def spectrum_to_cdf(spectrum: np.ndarray, energy: Optional[np.ndarray] = None,
                    normalize: bool = True) -> np.ndarray:
    """
    Convert a spectrum to its Cumulative Distribution Function (CDF).
    
    The CDF treats the spectrum as a probability distribution and computes
    its running integral. This provides shift-invariance for ML applications.
    
    Parameters
    ----------
    spectrum : np.ndarray
        Intensity values, shape (n_points,) or (n_points, 2) for (energy, intensity)
    energy : np.ndarray, optional
        Energy values. If None and spectrum is 2D, uses first column.
    normalize : bool
        Whether to normalize CDF to [0, 1]
        
    Returns
    -------
    cdf : np.ndarray
        Cumulative distribution function, same length as input
    """
    # Handle 2D input (energy, intensity)
    if spectrum.ndim == 2:
        if energy is None:
            energy = spectrum[:, 0]
        intensity = spectrum[:, 1]
    else:
        intensity = spectrum
        if energy is None:
            energy = np.arange(len(intensity))
    
    # Ensure non-negative
    intensity_pos = intensity - np.min(intensity)
    intensity_pos = intensity_pos + 1e-10  # Avoid division by zero
    
    # Compute CDF using trapezoidal integration
    # First normalize to get probability density
    total_area = np.trapz(intensity_pos, energy)
    if total_area > 0:
        prob_density = intensity_pos / total_area
    else:
        prob_density = np.ones_like(intensity_pos) / len(intensity_pos)
    
    # Compute cumulative sum (approximating integral)
    dx = np.gradient(energy)
    cdf = np.cumsum(prob_density * dx)
    
    # Normalize to [0, 1]
    if normalize and cdf[-1] > 0:
        cdf = cdf / cdf[-1]
    
    return cdf.astype(np.float32)


def batch_spectrum_to_cdf(spectra: np.ndarray, energy: Optional[np.ndarray] = None,
                          normalize: bool = True) -> np.ndarray:
    """
    Convert multiple spectra to CDFs.
    
    Parameters
    ----------
    spectra : np.ndarray
        Shape (n_spectra, n_points) or (n_spectra, n_points, 2)
    energy : np.ndarray, optional
        Energy grid, shape (n_points,)
    normalize : bool
        Whether to normalize each CDF to [0, 1]
        
    Returns
    -------
    cdfs : np.ndarray
        Shape (n_spectra, n_points)
    """
    n_spectra = spectra.shape[0]
    
    # Handle 3D input
    if spectra.ndim == 3:
        if energy is None:
            energy = spectra[0, :, 0]  # Assume same energy grid
        intensities = spectra[:, :, 1]
    else:
        intensities = spectra
        if energy is None:
            energy = np.arange(spectra.shape[1])
    
    n_points = intensities.shape[1]
    cdfs = np.zeros((n_spectra, n_points), dtype=np.float32)
    
    for i in range(n_spectra):
        cdfs[i] = spectrum_to_cdf(intensities[i], energy, normalize)
    
    return cdfs


def compute_cdf_from_flattened_spectrum(y_flat: np.ndarray, max_spec_len: int,
                                         mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute CDF from flattened spectrum format (energies | intensities).
    
    This is designed to work with the Auger graph data format where
    y has shape (n_atoms, max_spec_len * 2) with first half being energies
    and second half being intensities.
    
    Parameters
    ----------
    y_flat : np.ndarray
        Flattened spectrum, shape (max_spec_len * 2,)
    max_spec_len : int
        Number of spectral points
    mask : np.ndarray, optional
        Binary mask indicating valid points, shape (max_spec_len * 2,)
        
    Returns
    -------
    cdf : np.ndarray
        CDF of the spectrum, shape (max_spec_len,)
    """
    # Split into energy and intensity
    energies = y_flat[:max_spec_len]
    intensities = y_flat[max_spec_len:]
    
    # Apply mask if provided
    if mask is not None:
        energy_mask = mask[:max_spec_len] > 0.5
        valid_energies = energies[energy_mask]
        valid_intensities = intensities[energy_mask]
    else:
        # Use non-zero entries
        valid_mask = (energies != 0) | (intensities != 0)
        valid_energies = energies[valid_mask]
        valid_intensities = intensities[valid_mask]
    
    if len(valid_intensities) == 0:
        return np.zeros(max_spec_len, dtype=np.float32)
    
    # Compute CDF on valid portion
    cdf_valid = spectrum_to_cdf(valid_intensities, valid_energies, normalize=True)
    
    # Pad to max_spec_len
    cdf = np.zeros(max_spec_len, dtype=np.float32)
    cdf[:len(cdf_valid)] = cdf_valid
    
    return cdf


def compute_node_cdfs(y: torch.Tensor, mask: torch.Tensor, 
                      max_spec_len: int) -> torch.Tensor:
    """
    Compute CDFs for all nodes in a graph.
    
    Parameters
    ----------
    y : torch.Tensor
        Spectra tensor, shape (n_nodes, max_spec_len * 2)
    mask : torch.Tensor
        Mask tensor, shape (n_nodes, max_spec_len * 2)
    max_spec_len : int
        Number of spectral points
        
    Returns
    -------
    cdfs : torch.Tensor
        CDF tensor, shape (n_nodes, max_spec_len)
    """
    y_np = y.numpy() if isinstance(y, torch.Tensor) else y
    mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
    
    n_nodes = y_np.shape[0]
    cdfs = np.zeros((n_nodes, max_spec_len), dtype=np.float32)
    
    for i in range(n_nodes):
        cdfs[i] = compute_cdf_from_flattened_spectrum(y_np[i], max_spec_len, mask_np[i])
    
    return torch.FloatTensor(cdfs)


# =============================================================================
# ADDITIONAL SPECTRAL FEATURIZATION (Extensible)
# =============================================================================

class SpectralFeaturizer:
    """
    Extensible class for spectral featurization methods.
    
    Supports CDF, derivatives, moments, and can be extended with
    additional methods like CWT, peak decomposition, etc.
    """
    
    def __init__(self, method: str = 'cdf', **kwargs):
        """
        Initialize featurizer.
        
        Parameters
        ----------
        method : str
            Featurization method: 'cdf', 'derivative', 'moments', 'combined'
        **kwargs
            Method-specific parameters
        """
        self.method = method
        self.kwargs = kwargs
        
        # Available methods
        self._methods = {
            'cdf': self._compute_cdf,
            'derivative': self._compute_derivative,
            'moments': self._compute_moments,
            'combined': self._compute_combined,
        }
        
        if method not in self._methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(self._methods.keys())}")
    
    def transform(self, spectrum: np.ndarray, energy: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Transform a spectrum using the selected method.
        
        Parameters
        ----------
        spectrum : np.ndarray
            Input spectrum
        energy : np.ndarray, optional
            Energy grid
            
        Returns
        -------
        features : np.ndarray
            Featurized spectrum
        """
        return self._methods[self.method](spectrum, energy)
    
    def _compute_cdf(self, spectrum: np.ndarray, energy: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute CDF features."""
        return spectrum_to_cdf(spectrum, energy, normalize=True)
    
    def _compute_derivative(self, spectrum: np.ndarray, energy: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute first derivative using Savitzky-Golay filter."""
        from scipy.signal import savgol_filter
        
        window = self.kwargs.get('window_length', 7)
        polyorder = self.kwargs.get('polyorder', 2)
        
        if spectrum.ndim == 2:
            spectrum = spectrum[:, 1]
        
        # Ensure window is odd and <= length
        if window > len(spectrum):
            window = len(spectrum) if len(spectrum) % 2 == 1 else len(spectrum) - 1
        
        if len(spectrum) > window:
            derivative = savgol_filter(spectrum, window, polyorder, deriv=1)
        else:
            derivative = np.gradient(spectrum)
        
        return derivative.astype(np.float32)
    
    def _compute_moments(self, spectrum: np.ndarray, energy: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute statistical moments of the spectrum.
        
        Returns: [mean, std, skewness, kurtosis, max_pos, fwhm] (6 features)
        """
        from scipy import stats
        
        if spectrum.ndim == 2:
            if energy is None:
                energy = spectrum[:, 0]
            spectrum = spectrum[:, 1]
        elif energy is None:
            energy = np.arange(len(spectrum))
        
        # Normalize spectrum as probability distribution
        spec_pos = spectrum - spectrum.min() + 1e-10
        spec_norm = spec_pos / np.sum(spec_pos)
        
        # Compute moments
        mean = np.sum(energy * spec_norm)
        variance = np.sum((energy - mean)**2 * spec_norm)
        std = np.sqrt(variance)
        
        # Higher moments
        if std > 0:
            skewness = np.sum(((energy - mean) / std)**3 * spec_norm)
            kurtosis = np.sum(((energy - mean) / std)**4 * spec_norm) - 3
        else:
            skewness = 0.0
            kurtosis = 0.0
        
        # Peak position
        max_pos = energy[np.argmax(spectrum)]
        
        # FWHM
        half_max = np.max(spectrum) / 2
        above_half = spectrum > half_max
        if np.any(above_half):
            indices = np.where(above_half)[0]
            fwhm = energy[indices[-1]] - energy[indices[0]]
        else:
            fwhm = 0.0
        
        return np.array([mean, std, skewness, kurtosis, max_pos, fwhm], dtype=np.float32)
    
    def _compute_combined(self, spectrum: np.ndarray, energy: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute combined CDF + moments features."""
        cdf = self._compute_cdf(spectrum, energy)
        moments = self._compute_moments(spectrum, energy)
        return np.concatenate([cdf, moments])


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_category_counts(data_list: list) -> Dict[str, int]:
    """
    Count carbon environment categories across a dataset.
    
    Parameters
    ----------
    data_list : list
        List of PyG Data objects with carbon_env_labels attribute
        
    Returns
    -------
    counts : Dict[str, int]
        Category name to count mapping
    """
    counts = {name: 0 for name in CARBON_ENVIRONMENT_PATTERNS.keys()}
    counts['non_carbon'] = 0
    
    for data in data_list:
        if hasattr(data, 'carbon_env_labels'):
            for idx in data.carbon_env_labels.tolist():
                if idx == -1:
                    counts['non_carbon'] += 1
                else:
                    name = IDX_TO_CARBON_ENV[idx]
                    counts[name] += 1
    
    return counts


def print_category_distribution(data_list: list):
    """Print the distribution of carbon environment categories."""
    counts = get_category_counts(data_list)
    
    print("\nCarbon Environment Distribution:")
    print("-" * 50)
    
    # Sort by count
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    total = sum(counts.values())
    for name, count in sorted_counts:
        if count > 0:
            pct = 100 * count / total
            print(f"  {name:25s}: {count:6d} ({pct:5.1f}%)")
    
    print("-" * 50)
    print(f"  {'Total':25s}: {total:6d}")

# ============================================================================
# ANALYSIS OUTPUT
# ============================================================================

def analyze_carbon_environments(data_list, verbose=False):
    """
    Analyze and print carbon environment distribution across dataset.
    
    Parameters
    ----------
    data_list : list
        List of Data objects
    verbose : bool
        If True, print detailed diversity metrics
    """
    from collections import Counter
    
    if not data_list or len(data_list) == 0:
        print("No data to analyze")
        return
    
    # Collect all carbon environment labels (excluding non-carbons)
    # Use node_mask to distinguish carbons with target values from
    # structurally identified carbons that lack measurements.
    all_labels = []
    n_unmeasured = 0
    for data in data_list:
        labels = data.carbon_env_labels.tolist()
        mask   = data.node_mask.tolist() if hasattr(data, 'node_mask') else None
        for j, l in enumerate(labels):
            if l < 0:
                continue          # not a carbon
            if mask is not None and mask[j] < 0.5:
                n_unmeasured += 1
                continue          # carbon exists but has no target value
            all_labels.append(l)
    
    if not all_labels:
        print("No carbon atoms found in dataset")
        return
    
    # Count occurrences
    label_counts = Counter(all_labels)
    
    # Get category names
    idx_to_name = {v: k for k, v in CARBON_ENV_TO_IDX.items()}
    
    # Sort by count descending
    sorted_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    
    total_carbons = len(all_labels)
    unique_envs = len(label_counts)
    
    unmeasured_note = (f" ({n_unmeasured} additional carbons without target values)"
                       if n_unmeasured > 0 else "")
    print(f"\n  Carbon environments: {total_carbons} carbons, {unique_envs} types, "
          f"{len(data_list)} molecules{unmeasured_note}")
    
    # Compact table (top 10 + others)
    top_n = 10 if not verbose else len(sorted_counts)
    for idx, count in sorted_counts[:top_n]:
        category_name = idx_to_name[idx]
        percentage = 100.0 * count / total_carbons
        print(f"    {category_name:<25} {count:>5} ({percentage:>5.1f}%)")
    
    if len(sorted_counts) > top_n:
        other_count = sum(c for _, c in sorted_counts[top_n:])
        print(f"    {'... others':<25} {other_count:>5} ({100.0*other_count/total_carbons:>5.1f}%)")
    
    if verbose:
        # Diversity metrics
        import math
        max_count = sorted_counts[0][1]
        min_count = sorted_counts[-1][1]
        entropy = -sum((count / total_carbons) * math.log(count / total_carbons) 
                       for _, count in sorted_counts)
        max_entropy = math.log(unique_envs)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        print(f"  Entropy: {normalized_entropy:.3f} ({'balanced' if normalized_entropy > 0.7 else 'skewed'})")

# ============================================================================
# ANALYSIS FILE OUTPUT: XYZ FILES AND ENVIRONMENT MAPPINGS
# ============================================================================
#Called in prepare_data.py for auger cnn
def generate_calc_analysis_outputs(calc_data, raw_dir, output_dir, verbose=False):
    """
    Generate analysis outputs for calculated data:
    1. Clean XYZ files for all molecules (from load_eval_mol)
    2. Text files for each unique carbon environment with mol_name, xyz_idx, atom_idx
    
    Parameters
    ----------
    calc_data : list
        List of Data objects from process_auger_data(..., data_type='calc')
    raw_dir : str
        Root directory for raw data (contains auger_db folder)
    output_dir : str
        Output directory for analysis files (e.g., /path/to/analysis_output)
    verbose : bool
        Print detailed progress information
    """
    import os
    from pathlib import Path
    from collections import defaultdict
    import numpy as np
    
    # Create subdirectories
    xyz_dir = os.path.join(output_dir, "calc_xyz_files")
    env_dir = os.path.join(output_dir, "calc_environments")
    Path(xyz_dir).mkdir(parents=True, exist_ok=True)
    Path(env_dir).mkdir(parents=True, exist_ok=True)
    
    # Dictionary to collect environment mappings
    env_mappings = defaultdict(list)
    
    # Try to find XYZ data directory
    xyz_data_dir = None
    possible_dirs = [
        os.path.join(raw_dir, "xyz_db"),
        os.path.join(raw_dir, "auger_db", "xyz"),
        os.path.join(raw_dir, "qm9"),
    ]
    
    for possible_dir in possible_dirs:
        if os.path.isdir(possible_dir):
            xyz_data_dir = possible_dir
            break
    
    if xyz_data_dir is None:
        print("  ✗ Could not find XYZ directory")
        return
    
    if not isinstance(calc_data, (list, tuple)):
        calc_data = [calc_data]
    
    print(f"  Writing XYZ files...", end="", flush=True)
    
    successful = 0
    failed = 0
    for mol_idx, data_obj in enumerate(calc_data):
        mol_name = data_obj.name
        if isinstance(mol_name, bytes):
            mol_name = mol_name.decode('utf-8')
        elif isinstance(mol_name, np.ndarray):
            mol_name = mol_name.item() if mol_name.ndim == 0 else str(mol_name[0])
        elif isinstance(mol_name, (list, tuple)):
            mol_name = str(mol_name[0])
        else:
            mol_name = str(mol_name)
        
        # Try exact name first, then with dsgdb9nsd_ prefix (CEBE GNN uses
        # integer IDs like '130773' while XYZ files are 'dsgdb9nsd_130773.xyz')
        xyz_path = os.path.join(xyz_data_dir, f"{mol_name}.xyz")
        if not os.path.exists(xyz_path) and mol_name.isdigit():
            xyz_path = os.path.join(xyz_data_dir, f"dsgdb9nsd_{int(mol_name):06d}.xyz")
        
        try:
            # Use load_eval_mol (DetermineBonds + MDL aromaticity) for consistency
            # with GNN pipelines and process_auger_data
            from augernet import build_molecular_graphs
            mol, symbols, coords, smiles = build_molecular_graphs.load_eval_mol(
                xyz_path, labeled_atoms=False
            )
            
            xyz_output_path = os.path.join(xyz_dir, f"{mol_name}.xyz")
            with open(xyz_output_path, 'w') as f:
                f.write(f"{len(symbols)}\n")
                f.write(f"{mol_name} (SMILES: {smiles})\n")
                for symbol, coord in zip(symbols, coords):
                    f.write(f"{symbol:3s}  {coord[0]:12.8f}  {coord[1]:12.8f}  {coord[2]:12.8f}\n")
            
            successful += 1
            
            # Collect carbon environment mappings
            carbon_indices = [i for i, sym in enumerate(symbols) if sym == 'C']
            
            if hasattr(data_obj, 'carbon_env_labels'):
                carbon_labels = data_obj.carbon_env_labels
                if hasattr(carbon_labels, 'cpu'):
                    carbon_labels = carbon_labels.cpu().numpy()
                else:
                    carbon_labels = np.array(carbon_labels)
                
                for xyz_idx, atom_idx in enumerate(carbon_indices):
                    if atom_idx < len(carbon_labels):
                        env_idx = int(carbon_labels[atom_idx])
                        if env_idx >= 0:
                            env_mappings[env_idx].append((mol_name, xyz_idx, atom_idx))
            
            # Progress dots
            if (mol_idx + 1) % max(1, len(calc_data) // 10) == 0:
                print(".", end="", flush=True)
                
        except Exception as e:
            failed += 1
            if verbose and failed <= 5:
                print(f"\n    ⚠ {mol_name}: {e}")
    
    print(f" {successful} ok", end="")
    if failed:
        print(f", {failed} failed", end="")
    print()
    
    # Write environment mapping files
    from augernet.carbon_environment import IDX_TO_CARBON_ENV
    
    for env_idx in sorted(env_mappings.keys()):
        env_name = IDX_TO_CARBON_ENV.get(env_idx, f"C_unknown_{env_idx}")
        safe_env_name = env_name.replace('/', '_').replace('\\', '_')
        mapping_file = os.path.join(env_dir, f"{safe_env_name}.txt")
        
        with open(mapping_file, 'w') as f:
            f.write(f"# Carbon Environment: {env_name} (Index: {env_idx})\n")
            f.write(f"# Total occurrences: {len(env_mappings[env_idx])}\n")
            f.write("#\n")
            f.write("mol_name\txyz_index\tatom_index\n")
            for mol_name, xyz_idx, atom_idx in sorted(env_mappings[env_idx]):
                f.write(f"{mol_name}\t{xyz_idx}\t{atom_idx}\n")
    
    print(f"  Saved: {xyz_dir} ({successful} xyz), {env_dir} ({len(env_mappings)} env files)")