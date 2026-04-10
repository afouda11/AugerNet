"""
CEBE-GNN graph building tests.

Uses the real molecule dsgdb9nsd_133427 (C4H3F2N3, 12 atoms) from
tests/test_mol/ to test XYZ parsing, bond detection, node/edge
feature construction, and carbon environment labelling.

Tests marked 'full' exercise the heavier data-preparation path.
Tests marked 'essential' are fast and suitable for CI.
"""

import os
import pytest
import numpy as np

pytest.importorskip("rdkit")

from rdkit import Chem

from conftest import TEST_MOL_DIR, MOL_NAME, XYZ_PATH, CEBE_PATH

# ---------------------------------------------------------------------------
# Test molecule reference data (dsgdb9nsd_133427):
#   Atoms: F C C C F N N C N H H H  (12 atoms)
#   SMILES: FC1=CC(F)=NNC1=N  (aromatic triazine derivative with 2 F)
#   CEBE (eV): -1, 293.58, 291.70, 294.38, -1, -1, -1, 293.81, -1, -1, -1, -1
#   Carbons at indices: 1, 2, 3, 7  (4 carbon atoms with CEBE values)
# ---------------------------------------------------------------------------

EXPECTED_SYMBOLS = ["F", "C", "C", "C", "F", "N", "N", "C", "N", "H", "H", "H"]
N_ATOMS = 12
N_CARBONS = 4
CARBON_INDICES = [1, 2, 3, 7]


# -- XYZ parsing and RDKit molecule ------------------------------------------

@pytest.mark.essential
class TestMolFromXyz:

    def test_parses_correct_atom_count(self):
        from augernet.build_molecular_graphs import _mol_from_xyz_order
        mol, syms, pos, smiles = _mol_from_xyz_order(XYZ_PATH, labeled_atoms=False)
        assert mol.GetNumAtoms() == N_ATOMS

    def test_atom_symbols_match(self):
        from augernet.build_molecular_graphs import _mol_from_xyz_order
        _, syms, _, _ = _mol_from_xyz_order(XYZ_PATH, labeled_atoms=False)
        assert syms == EXPECTED_SYMBOLS

    def test_coordinates_shape(self):
        from augernet.build_molecular_graphs import _mol_from_xyz_order
        _, _, pos, _ = _mol_from_xyz_order(XYZ_PATH, labeled_atoms=False)
        assert pos.shape == (N_ATOMS, 3)

    def test_returns_valid_smiles(self):
        from augernet.build_molecular_graphs import _mol_from_xyz_order
        _, _, _, smiles = _mol_from_xyz_order(XYZ_PATH, labeled_atoms=False)
        assert Chem.MolFromSmiles(smiles) is not None

    def test_missing_file_raises(self):
        from augernet.build_molecular_graphs import _mol_from_xyz_order
        with pytest.raises(FileNotFoundError):
            _mol_from_xyz_order("/nonexistent/file.xyz", labeled_atoms=False)

    def test_bonds_detected(self):
        from augernet.build_molecular_graphs import _mol_from_xyz_order
        from rdkit.Chem import rdmolops
        mol, _, _, _ = _mol_from_xyz_order(XYZ_PATH, labeled_atoms=False)
        n_bonds = rdmolops.GetAdjacencyMatrix(mol).sum() // 2
        assert n_bonds >= N_ATOMS - 1  # at least a spanning tree


# -- node and edge feature construction --------------------------------------

@pytest.mark.essential
class TestBuildNodeEdgeFeatures:

    def test_edge_index_shape(self, real_mol_graph):
        assert real_mol_graph.edge_index.shape[0] == 2
        assert real_mol_graph.edge_index.shape[1] > 0

    def test_edge_attr_bond_types(self, real_mol_graph):
        # 4 bond types: single, double, triple, aromatic
        assert real_mol_graph.edge_attr.shape[1] == 4
        # each row is one-hot
        row_sums = real_mol_graph.edge_attr.sum(dim=1)
        assert (row_sums == 1.0).all()

    def test_category_feature_shape(self, real_mol_graph):
        assert real_mol_graph.x.shape == (N_ATOMS, 3)

    def test_pos_shape(self, real_mol_graph):
        assert real_mol_graph.pos.shape == (N_ATOMS, 3)

    def test_skipatom_200_shape(self, real_mol_graph):
        assert real_mol_graph.skipatom_200.shape == (N_ATOMS, 200)

    def test_atomic_be_shape(self, real_mol_graph):
        assert real_mol_graph.atomic_be.shape == (N_ATOMS,)

    def test_e_score_shape(self, real_mol_graph):
        assert real_mol_graph.e_score.shape == (N_ATOMS,)

    def test_node_mask_has_4_carbons(self, real_mol_graph):
        assert real_mol_graph.node_mask.sum().item() == N_CARBONS

    def test_carbon_env_labels_shape(self, real_mol_graph):
        labels = real_mol_graph.carbon_env_labels
        assert labels.shape == (N_ATOMS,)
        # non-carbons should be -1
        non_c = [i for i in range(N_ATOMS) if i not in CARBON_INDICES]
        assert all(labels[i].item() == -1 for i in non_c)
        # carbons should have valid env indices (>= 0)
        assert all(labels[i].item() >= 0 for i in CARBON_INDICES)


# -- carbon environment classification ----------------------------------------

@pytest.mark.essential
class TestCarbonEnvironment:

    def test_non_carbon_label(self):
        from augernet.carbon_environment import _get_carbon_environment_label
        mol = Chem.MolFromSmiles("CC(=O)O")
        o_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
        name, idx = _get_carbon_environment_label(mol, o_idx)
        assert name == "non_carbon" and idx == -1

    def test_methyl_in_ethane(self):
        from augernet.carbon_environment import _get_carbon_environment_label
        mol = Chem.MolFromSmiles("CC")
        for c in [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6]:
            name, _ = _get_carbon_environment_label(mol, c)
            assert name == "C_methyl"

    def test_benzene_aromatic(self):
        from augernet.carbon_environment import _get_carbon_environment_label
        mol = Chem.MolFromSmiles("c1ccccc1")
        for c in [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6]:
            name, _ = _get_carbon_environment_label(mol, c)
            assert name == "C_aromatic"

    def test_vinyl(self):
        from augernet.carbon_environment import _get_carbon_environment_label
        mol = Chem.MolFromSmiles("C=C")
        for c in [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6]:
            name, _ = _get_carbon_environment_label(mol, c)
            assert name == "C_vinyl"

    def test_whole_molecule_output_shape(self):
        from augernet.carbon_environment import (
            get_all_carbon_environment_labels,
            NUM_CARBON_CATEGORIES,
        )
        mol = Chem.MolFromSmiles("CC(=O)O")
        n = mol.GetNumAtoms()
        names, indices, onehot = get_all_carbon_environment_labels(mol)
        assert len(names) == n
        assert len(indices) == n
        assert onehot.shape == (n, NUM_CARBON_CATEGORIES)

    def test_carbon_onehot_is_one_hot(self):
        from augernet.carbon_environment import get_all_carbon_environment_labels
        mol = Chem.MolFromSmiles("CC(=O)O")
        _, _, onehot = get_all_carbon_environment_labels(mol)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:
                row = onehot[atom.GetIdx()]
                assert row.sum() == pytest.approx(1.0)


# -- electronegativity scores ------------------------------------------------

@pytest.mark.essential
class TestElectronegativity:

    def test_pauling_data(self):
        from augernet.eneg_diff import give_paulingdata, pdata
        pd = give_paulingdata(pdata)
        assert pd[1] == pytest.approx(2.20)  # H
        assert pd[6] == pytest.approx(2.55)  # C
        assert pd[8] == pytest.approx(3.44)  # O

    def test_diff_matrix_antisymmetric(self):
        from augernet.eneg_diff import get_eleneg_diff_mat
        mat = get_eleneg_diff_mat(num_elements=100)
        assert mat.shape == (100, 100)
        assert np.allclose(mat, -mat.T)
        assert np.allclose(np.diag(mat), 0.0)

    def test_ch4_carbon_positive(self):
        from augernet.eneg_diff import get_e_neg_score
        scores = get_e_neg_score("C")
        from rdkit import Chem
        mol_h = Chem.AddHs(Chem.MolFromSmiles("C"))
        c_idx = next(a.GetIdx() for a in mol_h.GetAtoms() if a.GetAtomicNum() == 6)
        assert scores[c_idx] > 0.0

    def test_ch4_score_value(self):
        from augernet.eneg_diff import get_e_neg_score
        from rdkit import Chem
        mol_h = Chem.AddHs(Chem.MolFromSmiles("C"))
        c_idx = next(a.GetIdx() for a in mol_h.GetAtoms() if a.GetAtomicNum() == 6)
        scores = get_e_neg_score("C")
        # C has 4 H neighbors: (EN[C]-EN[H]) * 4 = 0.35 * 4 = 1.40
        assert scores[c_idx] == pytest.approx(1.40, abs=1e-6)


# -- full data preparation (slower, exercises build_graphs) -------------------

@pytest.mark.full
class TestDataPreparation:

    def test_assemble_features_on_real_graph(self, real_mol_graph):
        """Assemble feature_keys='035' on the real molecule graph."""
        from augernet.feature_assembly import assemble_node_features
        data = assemble_node_features(
            real_mol_graph, feature_keys=[0, 3, 5], inplace=False
        )
        # 3 (category) + 200 (skipatom) + 1 (atomic_be) + 1 (e_score) = 205
        assert data.x.shape == (N_ATOMS, 205)

    def test_cebe_values_loaded_correctly(self):
        """CEBE output file has correct carbon values and -1 for non-carbons."""
        cebe = np.loadtxt(CEBE_PATH)
        assert len(cebe) == N_ATOMS
        # 4 carbon atoms have CEBE > 200 eV
        n_valid = sum(1 for v in cebe if v > 200)
        assert n_valid == N_CARBONS
        # non-carbon entries are -1
        n_neg = sum(1 for v in cebe if v == -1)
        assert n_neg == N_ATOMS - N_CARBONS
