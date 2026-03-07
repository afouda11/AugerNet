"""
Unified Molecular Graph Building Module
=======================================

Builds PyTorch Geometric graphs for molecular property prediction.
Supports both CEBE (Core Electron Binding Energy) and Auger spectroscopy outputs.

Usage:
    data_list = build_molecular_graphs(
        data_type='cebe',       # 'cebe' or 'auger'
        source_type='calc',     # 'calc', 'eval', or 'exp'
        ATOM_REP='SKIPATOM',
        raw_dir='/path/to/data',
        ...
    )

Key differences between graph types:
    - CEBE graphs: y = normalized (delta_be - mean) / std for binding energies
    - Auger graphs: y = flattened spectra [n_atoms, max_spec_len * 2]
    - Auger be_feature uses either molecular CEBE for carbons and atomic for others (be_feat = 'mol')
        or uses atomic reference values for all atoms (be_feat = 'atom')
    - CEBE be_feature uses atomic reference values for all atoms (be_feat = 'atom')
"""

import os
import re
import json
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
import statistics
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdmolops, rdchem, rdDetermineBonds, AllChem
from skipatom import SkipAtomInducedModel # Alt. options: OneHotVectors, RandomVectors, AtomVectors 

from augernet import utils
from augernet import spec_utils 
from augernet import eneg_diff as ed
from augernet import carbon_environment as ce

from augernet import DATA_RAW_DIR, DATA_PROCESSED_DIR

# Global electronegativity matrix
_EN_MAT = ed.get_eleneg_diff_mat(num_elements=100)

# Constants
au2eV = 27.21139

# Morgan fingerprint defaults (per-atom structural environment encoding)
MORGAN_RADIUS = 1       # ECFP2 — immediate neighbors (matches Auger lineshape locality)
MORGAN_N_BITS = 256     # compact representation (trade-off: 1024 is standard but large for GNN)

# Permitted bond types for edge encoding
# Default: AUGER-NET (4 types)
permitted_list_of_bond_types = [
    rdchem.BondType.SINGLE, 
    rdchem.BondType.DOUBLE, 
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC
]

# Extended bond types for CoreIPNET finetune compatibility (6 types)
coreipnet_permitted_bond_types = [
    rdchem.BondType.SINGLE, 
    rdchem.BondType.DOUBLE, 
    rdchem.BondType.TRIPLE,
    rdchem.BondType.QUADRUPLE,
    rdchem.BondType.AROMATIC,
    rdchem.BondType.DATIVE
]

# Permitted atom types for one-hot encoding (only elements in this work)
PERMITTED_ATOM_TYPES = ['H', 'C', 'N', 'O', 'F']


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _e_neg_scores_from_mol(mol, add_bonds=True, num_elements=100):
    """Compute Pauling electronegativity-difference scores directly from an RDKit Mol.

    This is the corrected version of :func:`_e_neg_scores`.  Instead of parsing
    a SMILES string (which creates a **new** mol with canonical atom ordering),
    this function operates on the *existing* ``mol`` object so that atom indices
    are guaranteed to match the XYZ / graph ordering already stored in the Data
    object.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule with hydrogens (``Chem.AddHs`` already applied).
        Must use the same atom ordering as the graph nodes.
    add_bonds : bool
        If True, weight neighbour counts by bond order (default: True).
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
        z = atom.GetAtomicNum() - 1          # 0-based index into EN matrix

        # Build neighbour vector (same logic as eneg_diff.get_full_neighbor_vectors)
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


def get_l(l):
    """Convert orbital letter to angular momentum quantum number."""
    return {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4}.get(l, "l value is not valid")


def get_n_l(orb):
    """Parse orbital string (e.g., '1s') into (n, l) quantum numbers."""
    n = int(orb[0])
    l = get_l(str(orb[1]))
    return n, l


def giveorbitalenergy(ele, orb, orbital_energy_file='orbitalenergy.json'):
    """
    For a given element and orbital, return the orbital energy in eV.
    
    Parameters
    ----------
    ele : str
        Element symbol (e.g., 'C', 'O')
    orb : str
        Orbital string (e.g., '1s', '2p')
    orbital_energy_file : str
        Path to JSON file with orbital energies
    
    Returns
    -------
    float
        Core binding energy in eV
    """
    with open(orbital_energy_file, 'r') as f:
        data = json.load(f)
    try:
        orbenegele = data[ele]
        del data
    except KeyError:
        raise KeyError("Element symbol not found")
    
    n, l = get_n_l(orb)
    cbenergy = orbenegele[str(l)][n-l-1]
    cbenergy *= au2eV
    return cbenergy

def initialize_all_atom_encoders(skipatom_dir, max_atomic_num=118):
    """
    Initialize ALL atom encoders for the feature-store approach.

    Returns a dict of ``{name: (encoder_fn, dim)}``:
      - ``skipatom_200``: 200-dim SkipAtom embedding
      - ``skipatom_30``:  30-dim  SkipAtom embedding
      - ``onehot``:       5-dim element one-hot (H, C, N, O, F)

    Parameters
    ----------
    skipatom_dir : str
        Directory containing SkipAtom model files
    max_atomic_num : int
        Unused (kept for backward compat). One-hot uses PERMITTED_ATOM_TYPES.

    Returns
    -------
    dict[str, tuple[callable, int]]
    """
    encoders = {}

    # --- one-hot (compact: H, C, N, O, F) ---
    atom_types = PERMITTED_ATOM_TYPES
    def one_hot_encoder(symbol):
        vec = np.zeros(len(atom_types))
        if symbol in atom_types:
            vec[atom_types.index(symbol)] = 1
        else:
            print(f"Warning: Atom type {symbol} not in PERMITTED_ATOM_TYPES, using zeros")
        return vec
    encoders['onehot'] = (one_hot_encoder, len(atom_types))

    # --- SkipAtom 200-dim ---
    data_file = os.path.join(skipatom_dir, "mp_2020_10_09.training.data")
    model_200_file = os.path.join(skipatom_dir, "mp_2020_10_09.dim200.model")
    if os.path.exists(model_200_file) and os.path.exists(data_file):
        model_200 = SkipAtomInducedModel.load(model_200_file, data_file, min_count=2e7, top_n=5)
        def skipatom_200_encoder(symbol, _m=model_200):
            if symbol in _m.dictionary:
                return _m.vectors[_m.dictionary[symbol]]
            else:
                print(f"Warning: Atom type {symbol} not in SkipAtom-200 dictionary")
                return np.zeros(_m.vectors.shape[1])
        encoders['skipatom_200'] = (skipatom_200_encoder, model_200.vectors.shape[1])
    else:
        print(f"Warning: SkipAtom 200-dim model not found in {skipatom_dir}")

    # --- SkipAtom 30-dim ---
    model_30_file = os.path.join(skipatom_dir, "mp_2020_10_09.dim30.model")
    if os.path.exists(model_30_file) and os.path.exists(data_file):
        model_30 = SkipAtomInducedModel.load(model_30_file, data_file, min_count=2e7, top_n=5)
        def skipatom_30_encoder(symbol, _m=model_30):
            if symbol in _m.dictionary:
                return _m.vectors[_m.dictionary[symbol]]
            else:
                print(f"Warning: Atom type {symbol} not in SkipAtom-30 dictionary")
                return np.zeros(_m.vectors.shape[1])
        encoders['skipatom_30'] = (skipatom_30_encoder, model_30.vectors.shape[1])
    else:
        print(f"Warning: SkipAtom 30-dim model not found in {skipatom_dir}")

    return encoders


def one_hot_encoding(x, permitted_list):
    """One-hot encode x based on permitted list. Unknown values map to last element."""
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding


def cleanup_qm9_xyz(fname):
    """
    Clean up QM9 XYZ format (removes malformed header, keeps valid atom data).
    
    Returns
    -------
    ind : str
        Cleaned XYZ content
    gdb_smi : str
        GDB SMILES string
    relax_smi : str
        Relaxed SMILES string
    """
    ind = open(fname).readlines()
    nAts = int(ind[0])
    gdb_smi, relax_smi = ind[-2].split()[:2]
    ind[1] = '\n'
    ind = ind[:nAts+2]
    for i in range(2, nAts+2):
        l = ind[i]
        l = l.split('\t')
        l.pop(-1)
        ind[i] = '\t'.join(l)+'\n'
    ind = ''.join(ind)
    return ind, gdb_smi, relax_smi

def extract_edge_attributes(mol, edge_index_order):
    """
    Extract edge attributes (bond types) from an RDKit molecule.
    
    Parameters
    ----------
    mol : RDKit Mol
        Molecule with correct atom ordering (from mol_from_xyz_order)
    edge_index_order : list of tuples
        List of (i, j) edge indices
    
    Returns
    -------
    bond_types : np.ndarray
        One-hot encoded bond types for each edge, shape (num_edges, num_bond_types)
    """
    bond_types = []
    for i, j in edge_index_order:
        bond = mol.GetBondBetweenAtoms(int(i), int(j))
        if bond is None:
            raise ValueError(f"No bond found between atoms {i} and {j}")
        
        bond_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
        if len(bond_types) == 0:
            bond_types = bond_enc
        else:
            bond_types = np.row_stack((bond_types, bond_enc))
    
    return bond_types



# =============================================================================
# MORGAN FINGERPRINT (PER-ATOM)
# =============================================================================

def compute_per_atom_morgan_fp(mol, radius=MORGAN_RADIUS, n_bits=MORGAN_N_BITS):
    """
    Compute per-atom Morgan fingerprint vectors for ALL atoms in a molecule.

    Uses the ``bitInfo`` output of ``GetMorganFingerprintAsBitVect`` to assign
    each set bit to the atom(s) that contributed to it.  This gives every atom
    a binary vector describing its local structural environment.

    Parameters
    ----------
    mol : RDKit Mol
        Must already have hydrogens (``Chem.AddHs``).
    radius : int
        Morgan FP radius.  1 = ECFP2 (nearest neighbors only).
    n_bits : int
        Number of bits in the hashed fingerprint.

    Returns
    -------
    np.ndarray, shape (n_atoms, n_bits), dtype float32
        Binary fingerprint vectors for every atom.
    """
    n_atoms = mol.GetNumAtoms()
    bit_info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=bit_info)

    atom_fps = np.zeros((n_atoms, n_bits), dtype=np.float32)
    for bit_idx, atom_radius_list in bit_info.items():
        for contributing_atom, _r in atom_radius_list:
            atom_fps[contributing_atom, bit_idx] = 1.0

    return atom_fps


# =============================================================================
# NODE AND EDGE FEATURE BUILDING
# =============================================================================

def build_node_and_edge_features(mol, all_encoders, xyz_path, orbital_energy_file, 
                                     category_feature=None, mol_smiles=None, 
                                     mol_name=None, cebe_dir=None):
    """
    Build node and edge features using the **feature-store** approach.

    Computes ALL possible node features as separate tensors (raw, unscaled).
    Scaling is deferred to training time via ``feature_assembly.assemble_node_features``.

    Parameters
    ----------
    mol : RDKit Mol
        RDKit molecule (with hydrogens added)
    all_encoders : dict
        Output of ``initialize_all_atom_encoders()`` — maps encoder name
        to ``(encoder_fn, dim)`` tuple.
    xyz_path : str
        Path to XYZ file
    orbital_energy_file : str
        Path to orbitalenergy.json
    category_feature : array-like, optional
        Category feature (e.g. [1,0,0] for CEBE, [0,1,0] for singlet Auger).
        Stored directly in ``data.x``.
    mol_smiles : str, optional
        SMILES string for electronegativity score computation.
    mol_name : str, optional
        Molecule name for loading CEBE values.
    cebe_dir : str, optional
        Directory containing ``{mol_name}_node_features.txt`` CEBE files.

    Returns
    -------
    node_features : dict[str, torch.Tensor]
        Separate feature tensors, keyed by FEATURE_CATALOG attribute names:
        ``feat_skipatom_200``, ``feat_skipatom_30``, ``feat_onehot``,
        ``feat_atomic_be``, ``feat_mol_be``, ``feat_e_score``, ``feat_env_onehot``,
        ``feat_morgan_fp``
    x : torch.Tensor
        Category feature only, shape (N, 3), or empty (N, 0) if no category.
    edge_index : torch.Tensor
    edge_attr : torch.Tensor
    pos : torch.Tensor
    atomic_be_tensor : torch.Tensor
        Atomic 1s BEs in eV (for output denormalisation / evaluation).
    carbon_env_indices : list[int]
        Per-atom carbon environment index (>=0 for carbons, -1 for others).
    """
    n_atoms = mol.GetNumAtoms()

    # ── electronegativity scores ──
    # Compute directly from the mol object to guarantee atom-index consistency
    # with the XYZ/graph ordering.  (The old SMILES-based path used canonical
    # ordering which disagrees with XYZ ordering for ~84% of molecules.)
    e_score = _e_neg_scores_from_mol(mol)

    # ── CEBE values for mol_be (if available) ──
    cebe_values = None
    if cebe_dir is not None and mol_name is not None:
        cebe_path = os.path.join(cebe_dir, f"{mol_name}_node_features.txt")
        if not os.path.exists(cebe_path):
            cebe_path = os.path.join(cebe_dir, f"{mol_name}_cebe.txt")
        if os.path.exists(cebe_path):
            cebe_values = np.loadtxt(cebe_path)

    # ── per-atom loop ──
    skipatom_200_list = []
    skipatom_30_list = []
    onehot_list = []
    atomic_be_list = []       # eV (evaluation reference)
    atomic_be_feat_list = []  # Hartree (feature: isolated atom BE)
    mol_be_feat_list = []     # Hartree (feature: molecular CEBE for C, atomic for others)
    e_score_list = []
    atom_symbols = []

    for iatom, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        atom_symbols.append(symbol)

        # SkipAtom-200
        if 'skipatom_200' in all_encoders:
            enc, _ = all_encoders['skipatom_200']
            skipatom_200_list.append(enc(symbol))

        # SkipAtom-30
        if 'skipatom_30' in all_encoders:
            enc, _ = all_encoders['skipatom_30']
            skipatom_30_list.append(enc(symbol))

        # One-hot
        if 'onehot' in all_encoders:
            enc, _ = all_encoders['onehot']
            onehot_list.append(enc(symbol))

        # Atomic 1s BE (eV, positive)
        atom_be_eV = -giveorbitalenergy(symbol, "1s", orbital_energy_file)
        atomic_be_list.append(atom_be_eV)

        # Atomic BE feature (Hartree, raw)
        atomic_be_feat_list.append(atom_be_eV / au2eV)

        # Molecular BE feature: CEBE for carbons, atomic for others (Hartree, raw)
        if cebe_values is not None and symbol == 'C' and cebe_values[iatom] != -1.:
            mol_be_feat_list.append(cebe_values[iatom] / au2eV)
        else:
            mol_be_feat_list.append(atom_be_eV / au2eV)

        # Electronegativity score (raw)
        if iatom < len(e_score):
            e_score_list.append(e_score[iatom])
        else:
            print(f"Warning: atom index {iatom} >= e_score length {len(e_score)}")
            e_score_list.append(0.0)

    # ── carbon environment one-hot ──
    _, carbon_env_indices, env_onehot_np = ce.get_all_carbon_environment_labels(mol)

    # ── assemble node_features dict ──
    node_features = {}

    if skipatom_200_list:
        node_features['feat_skipatom_200'] = torch.tensor(
            np.array(skipatom_200_list), dtype=torch.float)

    if skipatom_30_list:
        node_features['feat_skipatom_30'] = torch.tensor(
            np.array(skipatom_30_list), dtype=torch.float)

    if onehot_list:
        node_features['feat_onehot'] = torch.tensor(
            np.array(onehot_list), dtype=torch.float)

    node_features['feat_atomic_be'] = torch.tensor(
        atomic_be_feat_list, dtype=torch.float)          # (N,)

    node_features['feat_mol_be'] = torch.tensor(
        mol_be_feat_list, dtype=torch.float)              # (N,)

    node_features['feat_e_score'] = torch.tensor(
        e_score_list, dtype=torch.float)                  # (N,)

    node_features['feat_env_onehot'] = torch.tensor(
        env_onehot_np, dtype=torch.float)                 # (N, NUM_CARBON_CATEGORIES)

    # ── per-atom Morgan fingerprint ──
    morgan_fp = compute_per_atom_morgan_fp(mol)
    node_features['feat_morgan_fp'] = torch.tensor(
        morgan_fp, dtype=torch.float)                     # (N, MORGAN_N_BITS)

    # ── category feature → data.x ──
    if category_feature is not None:
        cat_feat = np.tile(category_feature, (n_atoms, 1))
        x = torch.tensor(cat_feat, dtype=torch.float)
    else:
        x = torch.zeros(n_atoms, 0, dtype=torch.float)

    # ── edge features ──
    adj_mat = rdmolops.GetAdjacencyMatrix(mol)
    edge_index_order = []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if adj_mat[i, j] != 0.:
                edge_index_order.append((i, j))

    edge_index = torch.tensor(edge_index_order, dtype=torch.long).t().contiguous()

    bond_types = extract_edge_attributes(mol, edge_index_order)

    edge_attr = torch.tensor(bond_types, dtype=torch.float)

    atomic_be_tensor = torch.tensor(atomic_be_list, dtype=torch.float)

    return node_features, x, edge_index, edge_attr, atomic_be_tensor, carbon_env_indices

# =============================================================================
# LOAD MOLECULE FROM XYZ 2 MOL WITH PRECISE ATOM ORDERING
# =============================================================================

def mol_from_xyz_order(fname, labeled_atoms=False):
    """
    Load an evaluation XYZ file and return an RDKit molecule with consistent ordering.
    
    Parameters
    ----------
    fname : str
        Path to XYZ file
    labeled_atoms : bool
        If True, atoms have labels like C1, O2, etc. (Auger eval format)
        If False, atoms are simple element symbols (CEBE exp format)
    
    Returns
    -------
    mol : RDKit.Mol
        Molecule with correct bonding, atom ordering, and coordinates
    xyz_symbols : list
        List of atomic symbols from XYZ file
    xyz_coords : np.ndarray
        Atomic coordinates from XYZ file (N, 3) (pos)
    smiles : str
        SMILES string generated from the molecule
    """
    if not os.path.exists(fname):
        raise FileNotFoundError(f"XYZ file not found: {fname}")
    
    lines = open(fname).read().splitlines()
    n_atoms = int(lines[0])
    
    xyz_lines = lines[2:2 + n_atoms]
    xyz_symbols = []
    xyz_coords = []
    
    for line in xyz_lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        
        if labeled_atoms:
            # Extract element from label (e.g., "C1" -> "C", "O3" -> "O")
            label = parts[0]
            element = ''.join([c for c in label if c.isalpha()])
        else:
            # Simple element symbol, may have trailing numbers from coordinate artifacts
            element = parts[0]
            # Handle potential malformed entries
            if not element.isalpha():
                element = ''.join([c for c in element if c.isalpha()])
        
        xyz_symbols.append(element)
        xyz_coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    xyz_coords = np.array(xyz_coords)
    
    if len(xyz_symbols) != n_atoms:
        raise ValueError(
            f"Atom count mismatch in {fname}: "
            f"Header says {n_atoms}, but found {len(xyz_symbols)} atoms"
        )
    
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
        rdDetermineBonds.DetermineBonds(mol, charge=0)
    except Exception as e:
        print(f"  Warning: DetermineBonds failed for {fname} ({e}), falling back to DetermineConnectivity")
        rdDetermineBonds.DetermineConnectivity(mol)
    
    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol)
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
    
    # Verify atom count matches (DetermineBonds preserves XYZ atom ordering)
    mol_symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    if len(mol_symbols) != len(xyz_symbols):
        raise ValueError(
            f"Atom count mismatch for {fname}:\n"
            f"  XYZ has {len(xyz_symbols)} atoms: {xyz_symbols}\n"
            f"  After AddHs() mol has {len(mol_symbols)} atoms: {mol_symbols}"
        )
    
    # Note: No permutation needed - DetermineBonds preserves XYZ atom ordering
    # (verified by benchmarking: 100% identical edge indices, 0% reordering needed)
    
    smiles = Chem.MolToSmiles(mol)
    
    return mol, xyz_symbols, xyz_coords, smiles


# =============================================================================
# NORMALIZATION STATISTICS
# =============================================================================

def compute_cebe_normalization_stats(cebe_dir, mol_list):
    """
    Load all CEBE data to compute normalization statistics.
    Returns: mean and std of (C_1s_BE - mol_cebe)
    """

    # Carbon 1s binding energy (atomic reference)
    atom_cebe = 308.23974136400005
    
    all_rel_cebe = []

    for mol_name in mol_list:

        cebe_path = f"{cebe_dir}/{mol_name}_node_features.txt"
        
        cebe = np.loadtxt(cebe_path)
        
        for i in cebe:
            if i != -1.:
                all_rel_cebe.append(atom_cebe - i)
    
    mean = np.mean(all_rel_cebe)
    std = np.std(all_rel_cebe, ddof=1)
    
    print(f"CEBE normalization stats:")
    print(f"  Mean: {mean}")
    print(f"  Std: {std}")
    print(f"  Total atoms: {len(all_rel_cebe)}")
    
    return mean, std

# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def build_cebe_gnn(data_dir, data_type, mol_file="mol_list.txt", OUT_SCALE="MEANSTD", scale_value=10, DEBUG=False):
    """
    Process calculated CEBE data using the feature-store approach.
    
    All node features are stored as separate ``data.feat_*`` attributes.
    ``data.x`` contains only the category_feature.
    """
    mol_dir = os.path.join(DATA_RAW_DIR, data_dir)

    skipatom_dir = os.path.join(DATA_RAW_DIR, "skipatom")

    mol_list_path = os.path.join(mol_dir, mol_file)
    with open(mol_list_path, 'r') as f:
        mol_list = [line.strip() for line in f]

    all_encoders = initialize_all_atom_encoders(skipatom_dir)
    orbital_energy_file = os.path.join(DATA_RAW_DIR, 'orbitalenergy.json')

    # Compute or load stats before loop over mol_list
    norm_stats_path = os.path.join(DATA_PROCESSED_DIR, 'cebe_norm_stats.pt')

    if data_type == 'calc':
        mean, std = compute_cebe_normalization_stats(mol_dir, mol_list)
        norm_stats = {'mean': mean, 'std': std, 'scale_value': scale_value}
        print("Normalization statistics:", norm_stats)
        torch.save(norm_stats, norm_stats_path)

    if data_type in ['exp', 'pes']:
        norm_stats = torch.load(norm_stats_path)
        mean = norm_stats['mean']
        std = norm_stats['std']

    data_list = []

    for mol_name in mol_list:

        mol_xyz_path = os.path.join(mol_dir, f"{mol_name}.xyz")
        mol, xyz_symbols, pos, smiles = mol_from_xyz_order(mol_xyz_path, labeled_atoms=False)

        #Use to differentiate between cebe, auger sing, auger trip
        category_feature=np.array([1, 0, 0])
        node_features, x, edge_index, edge_attr, atomic_be, carbon_env_indices = \
            build_node_and_edge_features(
                mol, all_encoders, orbital_energy_file,
                category_feature=category_feature,
                mol_smiles=smiles, mol_name=mol_name, cebe_dir=mol_dir,
            )

        cebe_path = f"{mol_dir}/{mol_name}_out.txt"
        cebe = np.loadtxt(cebe_path)

        # Build targets (same logic as v1)
        out = []
        for n, val in enumerate(cebe):
            if val == -1:
                out.append(-1)
            else:
                ref_e = atomic_be[n].item()
                dum = ref_e - val
                if OUT_SCALE == "MEANSTD":
                    out.append((dum - mean) / std)
                elif OUT_SCALE == "SCALAR":
                    out.append(dum / scale_value)
                else:
                    out.append(dum)

        y = torch.FloatTensor(out)
        node_mask = [0. if n == -1 else 1. for n in out]

        # Store original CEBE values (eV) so evaluation can display them
        # without round-trip precision loss through normalize/denormalize.
        true_cebe = torch.tensor(
            [float(v) if v != -1 else -1.0 for v in cebe],
            dtype=torch.float64,
        )

        data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            node_mask=torch.FloatTensor(node_mask),
            y=y.view(-1, 1), pos=pos, atomic_be=atomic_be,
            atom_symbols=xyz_symbols, true_cebe=true_cebe,
            smiles=smiles, mol_name=mol_name,
            carbon_env_labels=torch.tensor(carbon_env_indices, dtype=torch.long),
        )

        # Store all features as separate attributes
        for attr_name, tensor in node_features.items():
            setattr(data, attr_name, tensor)

        data_list.append(data)

    print("Total molecules processed:", len(data_list))

    return data_list

def _process_cebe_exp(raw_dir, category_feature=np.array([1, 0, 0]), OUT_SCALE="MEANSTD", scale_value=10, pes=False, 
                    DEBUG=False):

    """
    Process experimental CEBE data using the feature-store approach.
    """
    if pes == False:
        data_dir = "cebe_eval"
    if pes == True:
        data_dir = "cebe_pes_eval"

    exp_dir = os.path.join(raw_dir, data_dir)
    skipatom_dir = os.path.join(raw_dir, "skipatom")

    exp_list_path = os.path.join(exp_dir, "list_all.txt")
    with open(exp_list_path) as f:
        exp_list = [line.strip() for line in f]
    if DEBUG:
        exp_list = exp_list[:1]

    if os.path.exists('cebe_normalization_stats.pt'):
        norm_stats = torch.load('cebe_normalization_stats.pt')
        mean = norm_stats['mean']
        std = norm_stats['std']
    else:
        print("Warning: No normalization stats found, using defaults")
        mean, std = 0, 1

    all_encoders = initialize_all_atom_encoders(skipatom_dir)
    orbital_energy_file = os.path.join(raw_dir, 'orbitalenergy.json')

    exp_data_list = []

    for i in exp_list:
        mol_path = os.path.join(exp_dir, f"{i}.xyz")
        mol, xyz_symbols, xyz_coords, smiles = mol_from_xyz_order(mol_path, labeled_atoms=False)

        node_features, x, edge_index, edge_attr, pos, atomic_be, carbon_env_indices = \
            build_node_and_edge_features(
                mol, all_encoders, mol_path, orbital_energy_file,
                category_feature=category_feature,
                mol_smiles=smiles, mol_name=i, cebe_dir=exp_dir,
            )

        atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

        out_path = os.path.join(exp_dir, f"{i}_out.txt")
        cebe = np.loadtxt(out_path)
        out = []
        for n, val in enumerate(cebe):
            if val == -1:
                out.append(-1)
            else:
                ref_e = -giveorbitalenergy(atom_symbols[n], "1s", orbital_energy_file)
                dum = ref_e - val
                if OUT_SCALE == "MEANSTD":
                    out.append((dum - mean) / std)
                elif OUT_SCALE == "FEATURE_SCALE":
                    out.append(dum / scale_value)
                else:
                    out.append(dum)

        y = torch.FloatTensor(out)
        node_mask = [0. if n == -1 else 1. for n in out]

        # Store original CEBE values (eV) so evaluation can display them
        # without round-trip precision loss through normalize/denormalize.
        true_cebe = torch.tensor(
            [float(v) if v != -1 else -1.0 for v in cebe],
            dtype=torch.float64,
        )

        data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            node_mask=torch.FloatTensor(node_mask),
            y=y.view(-1, 1), pos=pos, atomic_be=atomic_be,
            true_cebe=true_cebe,
            smiles=smiles, name=i,
            carbon_env_labels=torch.tensor(carbon_env_indices, dtype=torch.long),
        )

        for attr_name, tensor in node_features.items():
            setattr(data, attr_name, tensor)

        exp_data_list.append(data)

    return exp_data_list
