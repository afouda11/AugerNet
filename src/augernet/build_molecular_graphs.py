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
    - CEBE graphs:  y = normalized (delta_be - mean) / std for binding energies
    - Auger graphs: y = flattened spectra [n_atoms, max_spec_len * 2]
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

def _build_node_and_edge_features(mol, all_encoders, cebe_values):
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
    cebe : np.ndarray
        CEBE values for the molecule, for mol_be feature for Auger spec only

    Returns
    -------
    node_features : dict[str, torch.Tensor]
        Separate feature tensors, keyed by FEATURE_NAMES values:
        ``skipatom_200``, ``skipatom_30``, ``onehot``,
        ``atomic_be``, ``mol_be``, ``e_score``, ``env_onehot``,
        ``morgan_fp``
    edge_index : torch.Tensor
    edge_attr : torch.Tensor
    atomic_be_tensor : torch.Tensor
        Atomic 1s BEs in eV (for output denormalisation / evaluation).
    carbon_env_indices : list[int]
        Per-atom carbon environment index (>=0 for carbons, -1 for others).
    """
    n_atoms = mol.GetNumAtoms()

    # ── electronegativity scores ──
    # Compute from the mol object to guarantee atom-index consistency
    # with the XYZ/graph ordering.
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
    carbon_env_labels, carbon_env_indices, env_onehot_np = ce.get_all_carbon_environment_labels(mol)

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
        atomic_be_feat_list, dtype=torch.float)          

    node_features['mol_be'] = torch.tensor(
        mol_be_feat_list, dtype=torch.float)             

    node_features['e_score'] = torch.tensor(
        e_score_list, dtype=torch.float)                 

    node_features['env_onehot'] = torch.tensor(
        env_onehot_np, dtype=torch.float)                

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

    x = torch.zeros(n_atoms, 0, dtype=torch.float)

    return node_features, x, edge_index, edge_attr, atomic_be_tensor, carbon_env_indices, carbon_env_labels

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
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def build_graphs(data_type, 
                 mol_file="mol_list.txt", 
                 auger_max_spec_len = 300,
                 DEBUG=False
                 ):
    """
    Process calculated CEBE data using the feature-store approach.
    
    All node features are stored as separate ``data.feat_*`` attributes.
    """
    mol_dir = os.path.join(DATA_RAW_DIR, data_type)

    skipatom_dir = os.path.join(DATA_RAW_DIR, "skipatom")

    mol_list_path = os.path.join(mol_dir, mol_file)
    with open(mol_list_path, 'r') as f:
        mol_list = [line.strip() for line in f]

    all_encoders = _initialize_all_atom_encoders(skipatom_dir)

    data_list = []

    if DEBUG:
        mol_list = mol_list[:10]

    # QM9 molecules with dissociated N2 groups identified 
    # https://figshare.com/ndownloader/files/3195404 is the orginal file of mols from KCGNN
    # The molecules exculded from AugerNet calculated database are in data/raw/excluded_molecules.txt
    EXCLUDED_MOLECULES_FILE = os.path.join(DATA_RAW_DIR, "excluded_molecules.txt")
    with open(EXCLUDED_MOLECULES_FILE, 'r') as f:
        excluded_mol_list = {line.strip() for line in f} # a set {} gets a harsh lookup with 'in' 

    for mol_name in mol_list:

        if mol_name in excluded_mol_list:
            print(f"{mol_name} in exclusion list due to dissociated N2 group, skipping")
            continue

        mol_xyz_path = os.path.join(mol_dir, f"{mol_name}.xyz")
        mol, xyz_symbols, pos, smiles = _mol_from_xyz_order(mol_xyz_path, labeled_atoms=False)

        cebe_path = f"{mol_dir}/{mol_name}_out.txt"
        cebe = np.loadtxt(cebe_path)

        #print("mol_name:", mol_name)
        node_features, x, edge_index, edge_attr, atomic_be, carbon_env_indices, carbon_env_labels = \
            _build_node_and_edge_features(mol, all_encoders, cebe)
        
        ###### cat feature debug check
        #if data_type in ['calc_cebe', 'exp_cebe']:
        #    n_atoms = mol.GetNumAtoms() 
        #    category_feature=np.array([1, 0, 0])
        #    cat_feat = np.tile(category_feature, (n_atoms, 1))
        #    x = torch.tensor(cat_feat, dtype=torch.float)
        ######

        # Build targets.  RAW (eV) — not normalised.
        # Normalisation constants are fitted per fold from the training
        # molecules at train time (backend_gnn._fit_fold_norm), which
        # recomputes this same quantity as atomic_be_eV - true_cebe.  Storing
        # it raw here means the graphs carry no dataset-wide statistic.
        cebe_out = []
        for n, val in enumerate(cebe):
            if val == -1:
                cebe_out.append(-1)
            else:
                ref_e = atomic_be[n].item()
                cebe_out.append(ref_e - val)

        cebe_y = torch.FloatTensor(cebe_out)

        node_mask = [0. if n == -1 else 1. for n in cebe]

        # Store original CEBE values (eV) so evaluation can display them
        # without round-trip precision loss through normalize/denormalize.
        true_cebe = torch.tensor(
            [float(v) if v != -1 else -1.0 for v in cebe],
            dtype=torch.float32,
        )

        if data_type in ['calc_cebe', 'exp_cebe']: 
            data = Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr,
                node_mask=torch.FloatTensor(node_mask),
                cebe_y=cebe_y.view(-1, 1), 
                pos=torch.tensor(pos, dtype=torch.float), 
                atomic_be_eV=atomic_be,
                atom_symbols=xyz_symbols, 
                true_cebe=true_cebe,
                smiles=smiles, 
                mol_name=mol_name,
                carbon_env_labels=carbon_env_labels,
                carbon_env_indices=torch.tensor(carbon_env_indices, dtype=torch.long),
            )

        if data_type in ['calc_auger', 'eval_auger']:

            sing_spec_out, trip_spec_out, carbon_idx_mapping = \
                                        spec_utils.extract_spectra(
                                            data_type, mol_dir, mol_name,
                                            auger_max_spec_len
                                        )
            # pass openmolcas to xyz index map to data object for evalution
            carbon_spec_idx = torch.tensor(np.asarray(carbon_idx_mapping), dtype=torch.long)

            #singlet
            sing_spec_out_array = np.array(sing_spec_out)
            sing_y = torch.from_numpy(sing_spec_out_array).float()
            sing_mask_rows = (sing_y.abs().sum(dim=-1) > 0).float()
            #triplet
            trip_spec_out_array = np.array(trip_spec_out)
            trip_y = torch.from_numpy(trip_spec_out_array).float()
            trip_mask_rows = (trip_y.abs().sum(dim=-1) > 0).float()

            data = Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr,
                node_mask=torch.FloatTensor(node_mask),
                cebe_y=cebe_y.view(-1, 1),
                sing_y=sing_y,
                trip_y=trip_y,
                sing_mask_bin=sing_mask_rows,
                trip_mask_bin=trip_mask_rows,
                pos=torch.tensor(pos, dtype=torch.float), 
                atomic_be_eV=atomic_be,
                true_cebe=true_cebe,
                atom_symbols=xyz_symbols, 
                smiles=smiles, 
                mol_name=mol_name,
                carbon_env_labels=carbon_env_labels,
                carbon_env_indices=torch.tensor(carbon_env_indices, dtype=torch.long),
                carbon_spec_idx=carbon_spec_idx,
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
