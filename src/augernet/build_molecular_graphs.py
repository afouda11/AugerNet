"""
Molecular Graph Building
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
import json
import numpy as np
import torch
from typing import List
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdmolops, rdchem, rdDetermineBonds, AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from skipatom import SkipAtomInducedModel # Alt. options: OneHotVectors, RandomVectors, AtomVectors 

from . import eneg_diff as ed
from . import carbon_environment as ce
from . import spec_utils

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*MorganGenerator.*')
from augernet import DATA_RAW_DIR, DATA_PROCESSED_DIR

# Global electronegativity matrix
_EN_MAT = ed.get_eleneg_diff_mat(num_elements=100)

# Constants
au2eV = 27.21139



# Permitted bond types for edge encoding
# Default: AUGER-NET (4 types)
permitted_list_of_bond_types = [
    rdchem.BondType.SINGLE, 
    rdchem.BondType.DOUBLE, 
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC
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


def _get_l(l):
    """Convert orbital letter to angular momentum quantum number."""
    return {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4}.get(l, "l value is not valid")


def _get_n_l(orb):
    """Parse orbital string (e.g., '1s') into (n, l) quantum numbers."""
    n = int(orb[0])
    l = _get_l(str(orb[1]))
    return n, l


def _giveorbitalenergy(ele, orb, orbital_energy_file='orbitalenergy.json'):
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
    
    n, l = _get_n_l(orb)
    cbenergy = orbenegele[str(l)][n-l-1]
    cbenergy *= au2eV
    return cbenergy

def _initialize_all_atom_encoders(skipatom_dir, max_atomic_num=118):

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


def _one_hot_encoding(x, permitted_list):
    """One-hot encode x based on permitted list. Unknown values map to last element."""
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding

def _extract_edge_attributes(mol, edge_index_order):
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
        
        bond_enc = _one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
        if len(bond_types) == 0:
            bond_types = bond_enc
        else:
            bond_types = np.vstack((bond_types, bond_enc))
    
    return bond_types




# =============================================================================
# NODE AND EDGE FEATURE BUILDING
# =============================================================================

def _build_node_and_edge_features(mol, all_encoders, category_feature, cebe_values):
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
    category_feature : array-like, optional
        Category feature (e.g. [1,0,0] for CEBE, [0,1,0] for singlet Auger).
        Stored directly in ``data.x``.
    cebe : np.ndarray
        CEBE values for the molecule, for mol_be feature for Auger spec only

    Returns
    -------
    node_features : dict[str, torch.Tensor]
        Separate feature tensors, keyed by FEATURE_NAMES values:
        ``skipatom_200``, ``skipatom_30``, ``onehot``,
        ``atomic_be``, ``mol_be``, ``e_score``, ``env_onehot``,
        ``morgan_fp``
    x : torch.Tensor
        Category feature only, shape (N, 3), or empty (N, 0) if no category.
    edge_index : torch.Tensor
    edge_attr : torch.Tensor
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

    orbital_energy_file = os.path.join(DATA_RAW_DIR, 'orbitalenergy.json')

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
        atom_be_eV = -_giveorbitalenergy(symbol, "1s", orbital_energy_file)
        atomic_be_list.append(atom_be_eV)

        # Atomic BE feature (Hartree, raw)
        atomic_be_feat_list.append(atom_be_eV / au2eV)
        
        # Molecular BE feature: CEBE for carbons, atomic for others (Hartree, raw)
        if symbol == 'C' and cebe_values[iatom] != -1.:
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
        node_features['skipatom_200'] = torch.tensor(
            np.array(skipatom_200_list), dtype=torch.float)

    if skipatom_30_list:
        node_features['skipatom_30'] = torch.tensor(
            np.array(skipatom_30_list), dtype=torch.float)

    if onehot_list:
        node_features['onehot'] = torch.tensor(
            np.array(onehot_list), dtype=torch.float)

    node_features['atomic_be'] = torch.tensor(
        atomic_be_feat_list, dtype=torch.float)          # (N,)

    node_features['mol_be'] = torch.tensor(
        mol_be_feat_list, dtype=torch.float)              # (N,)

    node_features['e_score'] = torch.tensor(
        e_score_list, dtype=torch.float)                  # (N,)

    node_features['env_onehot'] = torch.tensor(
        env_onehot_np, dtype=torch.float)                 # (N, NUM_CARBON_CATEGORIES)

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

    bond_types = _extract_edge_attributes(mol, edge_index_order)

    edge_attr = torch.tensor(bond_types, dtype=torch.float)

    atomic_be_tensor = torch.tensor(atomic_be_list, dtype=torch.float)

    return node_features, x, edge_index, edge_attr, atomic_be_tensor, carbon_env_indices

# =============================================================================
# LOAD MOLECULE FROM XYZ 2 MOL WITH PRECISE ATOM ORDERING
# =============================================================================

def _mol_from_xyz_order(fname, labeled_atoms=False):
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

    # AddHs ensures all hydrogens are explicit.  Since XYZ files in this
    # dataset already contain all H atoms, AddHs should be a no-op.  If it
    # *does* append atoms the ordering would silently break, so we check.
    n_before = mol.GetNumAtoms()
    mol = Chem.AddHs(mol)
    if mol.GetNumAtoms() != n_before:
        raise RuntimeError(
            f"Chem.AddHs() added {mol.GetNumAtoms() - n_before} atom(s) to "
            f"{fname} — the XYZ file is missing explicit hydrogens.  "
            f"All H atoms must be present in the XYZ to guarantee ordering."
        )

    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
    
    # Verify atom ordering is preserved (not just count)
    mol_symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    if mol_symbols != xyz_symbols:
        # Find the first mismatch for a useful error message
        for k, (ms, xs) in enumerate(zip(mol_symbols, xyz_symbols)):
            if ms != xs:
                raise ValueError(
                    f"Atom ordering mismatch for {fname} at index {k}: "
                    f"mol has '{ms}' but XYZ has '{xs}'.\n"
                    f"  XYZ symbols: {xyz_symbols}\n"
                    f"  Mol symbols: {mol_symbols}"
                )
        # Length mismatch (shouldn't reach here given AddHs guard, but just in case)
        raise ValueError(
            f"Atom count mismatch for {fname}:\n"
            f"  XYZ has {len(xyz_symbols)} atoms, mol has {len(mol_symbols)}"
        )
    
    # Note: No permutation needed - DetermineBonds preserves XYZ atom ordering
    # (verified by benchmarking: 100% identical edge indices, 0% reordering needed)
    
    smiles = Chem.MolToSmiles(mol)
    
    return mol, xyz_symbols, xyz_coords, smiles


# =============================================================================
# NORMALIZATION STATISTICS
# =============================================================================

def _compute_cebe_normalization_stats(cebe_dir, mol_list):
    """
    Load all CEBE data to compute normalization statistics.
    Returns: mean and std of (C_1s_BE - mol_cebe)
    """

    # Carbon 1s binding energy (atomic reference)
    atom_cebe = 308.23974136400005
    
    all_rel_cebe = []

    for mol_name in mol_list:

        cebe_path = f"{cebe_dir}/{mol_name}_out.txt"
        
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

def build_graphs(data_type, mol_file="mol_list.txt", 
                 auger_spin=None, auger_max_ke=273,
                 auger_max_spec_len = 300,
                 DEBUG=False):
    """
    Process calculated CEBE data using the feature-store approach.
    
    All node features are stored as separate ``data.feat_*`` attributes.
    ``data.x`` contains only the category_feature.
    """
    mol_dir = os.path.join(DATA_RAW_DIR, data_type)

    skipatom_dir = os.path.join(DATA_RAW_DIR, "skipatom")

    mol_list_path = os.path.join(mol_dir, mol_file)
    with open(mol_list_path, 'r') as f:
        mol_list = [line.strip() for line in f]

    all_encoders = _initialize_all_atom_encoders(skipatom_dir)

    # Compute or load stats before loop over mol_list:

    # Calculate and save norm stats for calc data
    if data_type in ['calc_cebe', 'exp_cebe']:
        norm_stats_path = os.path.join(DATA_PROCESSED_DIR, 'cebe_norm_stats.pt')
    elif data_type in ['calc_auger', 'eval_auger']:
        norm_stats_path = os.path.join(DATA_PROCESSED_DIR, 'auger_cebe_norm_stats.pt')

    if data_type in ['calc_cebe', 'calc_auger']:
        mean, std = _compute_cebe_normalization_stats(mol_dir, mol_list)
        norm_stats = {'mean': float(mean), 'std': float(std)}
        print("Normalization statistics:", norm_stats)
        torch.save(norm_stats, norm_stats_path)

    # Load norm stats  from calc data for exp eval and other predictions
    if data_type in ['exp_cebe', 'eval_auger']:
        norm_stats = torch.load(norm_stats_path, weights_only=False)
        mean = norm_stats['mean']
        std = norm_stats['std']

    data_list = []

    if DEBUG:
        mol_list = mol_list[:5]

    for mol_name in mol_list:

        mol_xyz_path = os.path.join(mol_dir, f"{mol_name}.xyz")
        mol, xyz_symbols, pos, smiles = _mol_from_xyz_order(mol_xyz_path, labeled_atoms=False)

        cebe_path = f"{mol_dir}/{mol_name}_out.txt"
        cebe = np.loadtxt(cebe_path)

        if data_type in ['calc_auger', 'eval_auger']:
            spec_out, spec_len = spec_utils.extract_spectra(
                                        data_type, mol_dir, mol_name, 
                                        auger_spin, auger_max_ke, 
                                        auger_max_spec_len) 

        #Use to differentiate between cebe, auger sing, auger trip
        if data_type in ['calc_cebe', 'exp_cebe']: 
            category_feature=np.array([1, 0, 0])
        if data_type in ['calc_auger', 'eval_auger']:
            if auger_spin == 'singlet': 
                category_feature=np.array([0, 1, 0])
            if auger_spin == 'triplet': 
                category_feature=np.array([0, 0, 1])
        #print("mol_name:", mol_name)
        node_features, x, edge_index, edge_attr, atomic_be, carbon_env_indices = \
            _build_node_and_edge_features(mol, all_encoders, category_feature, cebe)
        #print(xyz_symbols)
        #print(np.column_stack((node_features['atomic_be'], node_features['mol_be'])))
        #print(" ")

        if data_type in ['calc_cebe', 'exp_cebe']:
            # Build targets (same logic as v1)
            out = []
            for n, val in enumerate(cebe):
                if val == -1:
                    out.append(-1)
                else:
                    ref_e = atomic_be[n].item()
                    dum = ref_e - val
                    #Mean std normalized output, across the full calc dataset
                    out.append((dum - mean) / std)

            y = torch.FloatTensor(out)

        if data_type in ['calc_auger', 'eval_auger']:
            spec_out_array = np.array(spec_out)
            y = torch.from_numpy(spec_out_array).float()
            mask_rows = ~(y.abs().sum(dim=2) == 0)
            mask_flat = mask_rows.repeat(1, 2).float()
            y = y.transpose(1, 2).reshape(len(xyz_symbols), auger_max_spec_len * 2)


        node_mask = [0. if n == -1 else 1. for n in cebe]

        # Store original CEBE values (eV) so evaluation can display them
        # without round-trip precision loss through normalize/denormalize.
        true_cebe = torch.tensor(
            [float(v) if v != -1 else -1.0 for v in cebe],
            dtype=torch.float64,
        )

        if data_type in ['calc_cebe', 'exp_cebe']: 
            data = Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr,
                node_mask=torch.FloatTensor(node_mask),
                y=y.view(-1, 1), 
                pos=torch.tensor(pos, dtype=torch.float), 
                atomic_be_eV=atomic_be,
                atom_symbols=xyz_symbols, 
                true_cebe=true_cebe,
                smiles=smiles, 
                mol_name=mol_name,
                carbon_env_labels=torch.tensor(carbon_env_indices, dtype=torch.long),
            )
        if data_type in ['calc_auger', 'eval_auger']:
            data = Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr,
                node_mask=torch.FloatTensor(node_mask),
                y=y.view(-1, 1),
                mask_bin=mask_flat,
                spec_len=spec_len,
                pos=torch.tensor(pos, dtype=torch.float), 
                atomic_be_eV=atomic_be,
                true_cebe=true_cebe,
                atom_symbols=xyz_symbols, 
                smiles=smiles, 
                mol_name=mol_name,
                carbon_env_labels=torch.tensor(carbon_env_indices, dtype=torch.long),
            )

        # Store all features as separate attributes
        for attr_name, tensor in node_features.items():
            setattr(data, attr_name, tensor)

        data_list.append(data)

    print("Total molecules processed:", len(data_list))

    return data_list

# =============================================================================
# BUTINA CLUSTERING (for scaffold-aware train/val splits)
# =============================================================================

# Butina clustering uses whole-molecule ECFP4 (radius 2, 1024 bits — standard)
BUTINA_RADIUS = 2
BUTINA_N_BITS = 1024

def _taylor_butina_clustering(fp_list, cutoff=0.65):
    """Cluster fingerprints using the RDKit Taylor-Butina algorithm.

    Parameters
    ----------
    fp_list : list of DataStructs.ExplicitBitVect
        Molecular fingerprints.
    cutoff : float
        Distance cutoff (1 - Tanimoto similarity).  Molecules within
        this distance are placed in the same cluster.

    Returns
    -------
    list of int
        Cluster ID for each molecule (0-indexed, ordered by decreasing
        cluster size — cluster 0 is the largest).
    """
    nfps = len(fp_list)
    dists = []
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        dists.extend([1.0 - x for x in sims])

    cluster_res = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)

    cluster_ids = np.zeros(nfps, dtype=int)
    for cluster_num, members in enumerate(cluster_res):
        for member in members:
            cluster_ids[member] = cluster_num
    return cluster_ids.tolist()

def get_butina_clusters(smiles_list, cutoff=0.65):
    """Assign Butina cluster IDs from a list of SMILES strings.

    Uses Morgan radius-2 / 1024-bit fingerprints (ECFP4) for the
    Tanimoto distance matrix, then Taylor-Butina clustering.

    Parameters
    ----------
    smiles_list : list of str
        SMILES for every molecule in the dataset.
    cutoff : float
        Distance cutoff passed to :func:`_taylor_butina_clustering`.

    Returns
    -------
    list of int
        One cluster ID per molecule.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=BUTINA_RADIUS, fpSize=BUTINA_N_BITS)
    fp_list = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {smi}")
        fp_list.append(gen.GetFingerprint(mol))
    return _taylor_butina_clustering(fp_list, cutoff=cutoff)

# =============================================================================
# MORGAN FINGERPRINT (PER-ATOM)
# =============================================================================

def get_per_atom_morgan_bits(mol, radius=1, n_bits=2048):
    """Compute per-atom Morgan fingerprint bit sets for every atom.

    This is the canonical low-level function used by all Morgan-FP
    consumers in this project (node features, locality analysis, etc.).

    Parameters
    ----------
    mol : RDKit Mol
        Must already have explicit hydrogens (``Chem.AddHs``).
    radius : int
        Morgan FP radius.  1 = ECFP2, 2 = ECFP4, …
    n_bits : int
        Number of bits in the hashed fingerprint.

    Returns
    -------
    list[frozenset[int]]
        One ``frozenset`` of active bit indices per atom (length = n_atoms).
    """
    n_atoms = mol.GetNumAtoms()

    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits,
    )

    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateAtomToBits()

    # Side-effect: populates ao with atom→bit mapping
    gen.GetFingerprintAsNumPy(mol, additionalOutput=ao)

    atom_to_bits = ao.GetAtomToBits()
    return [frozenset(atom_to_bits[i]) for i in range(n_atoms)]
