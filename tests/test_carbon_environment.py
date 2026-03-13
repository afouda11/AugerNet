"""
Tests for augernet.carbon_environment

Tier B — numpy + rdkit (no torch required)

Covered:
  _get_carbon_environment_label     — single atom classification
  get_all_carbon_environment_labels — whole-molecule classification
  SMARTS priority resolution        — specific patterns beat generic ones
"""

import pytest
import numpy as np

pytest.importorskip("rdkit")

from rdkit import Chem

from augernet.carbon_environment import (
    _get_carbon_environment_label,
    get_all_carbon_environment_labels,
    NUM_CARBON_CATEGORIES,
    CARBON_ENV_TO_IDX,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def mol_from(smiles: str) -> Chem.Mol:
    """Return an RDKit Mol from SMILES (raises if invalid)."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    return mol


def carbon_indices(mol: Chem.Mol):
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6]


# ─────────────────────────────────────────────────────────────────────────────
# _get_carbon_environment_label — known molecule → expected category
# ─────────────────────────────────────────────────────────────────────────────

class TestGetCarbonEnvironmentLabel:
    # ── non-carbon atoms ────────────────────────────────────────────────────

    def test_non_carbon_name(self):
        mol = mol_from("CC(=O)O")  # acetic acid; atoms 2,3 are O
        o_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
        name, idx = _get_carbon_environment_label(mol, o_idx)
        assert name == "non_carbon"

    def test_non_carbon_index(self):
        mol = mol_from("CC(=O)O")
        o_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
        _, idx = _get_carbon_environment_label(mol, o_idx)
        assert idx == -1

    def test_non_carbon_nitrogen(self):
        mol = mol_from("CC#N")  # acetonitrile; N is at index 2
        n_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
        name, idx = _get_carbon_environment_label(mol, n_idx)
        assert name == "non_carbon"
        assert idx == -1

    # ── methyl / methane ────────────────────────────────────────────────────

    def test_methane_is_methyl(self):
        mol = mol_from("C")   # CH4
        name, idx = _get_carbon_environment_label(mol, 0)
        assert name == "C_methyl"
        assert idx == CARBON_ENV_TO_IDX["C_methyl"]

    def test_methyl_in_ethane(self):
        mol = mol_from("CC")  # both Cs are methyls
        for c_idx in carbon_indices(mol):
            name, _ = _get_carbon_environment_label(mol, c_idx)
            assert name == "C_methyl"

    # ── aromatic ────────────────────────────────────────────────────────────

    def test_benzene_aromatic(self):
        mol = mol_from("c1ccccc1")
        for c_idx in carbon_indices(mol):
            name, _ = _get_carbon_environment_label(mol, c_idx)
            assert name == "C_aromatic"

    # ── carbonyl / carboxylic acid ───────────────────────────────────────────

    def test_acetic_acid_carbonyl_C(self):
        """C(=O)O carbon in acetic acid → C_carboxylic_acid."""
        mol = mol_from("CC(=O)O")
        # atom 1 is the carbonyl C
        name, _ = _get_carbon_environment_label(mol, 1)
        assert name == "C_carboxylic_acid"

    def test_acetic_acid_methyl_C(self):
        mol = mol_from("CC(=O)O")
        # atom 0 is the methyl C
        name, _ = _get_carbon_environment_label(mol, 0)
        assert name == "C_methyl"

    # ── nitrile ─────────────────────────────────────────────────────────────

    def test_nitrile_C(self):
        """C#N carbon in acetonitrile → C_nitrile."""
        mol = mol_from("CC#N")
        # atom 1 is the nitrile C
        name, _ = _get_carbon_environment_label(mol, 1)
        assert name == "C_nitrile"

    # ── ketone ──────────────────────────────────────────────────────────────

    def test_ketone_C(self):
        """C=O carbon flanked by two carbons → C_ketone."""
        mol = mol_from("CC(=O)C")  # acetone
        # atom 1 is the ketone carbonyl
        name, _ = _get_carbon_environment_label(mol, 1)
        assert name == "C_ketone"

    # ── vinyl / alkene ───────────────────────────────────────────────────────

    def test_vinyl_C(self):
        """Terminal alkene carbon → C_vinyl."""
        mol = mol_from("C=C")  # ethylene
        for c_idx in carbon_indices(mol):
            name, _ = _get_carbon_environment_label(mol, c_idx)
            assert name == "C_vinyl"

    # ── return types ────────────────────────────────────────────────────────

    def test_returns_tuple(self):
        mol = mol_from("C")
        result = _get_carbon_environment_label(mol, 0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_index_is_valid_category(self):
        mol = mol_from("c1ccccc1")
        for c_idx in carbon_indices(mol):
            _, idx = _get_carbon_environment_label(mol, c_idx)
            assert 0 <= idx < NUM_CARBON_CATEGORIES


# ─────────────────────────────────────────────────────────────────────────────
# get_all_carbon_environment_labels — whole-molecule output shapes
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAllCarbonEnvironmentLabels:
    def test_output_lengths_match_num_atoms(self):
        mol = mol_from("CC(=O)O")
        n_atoms = mol.GetNumAtoms()
        names, indices, onehot = get_all_carbon_environment_labels(mol)
        assert len(names) == n_atoms
        assert len(indices) == n_atoms

    def test_onehot_shape(self):
        mol = mol_from("CC(=O)O")
        n_atoms = mol.GetNumAtoms()
        _, _, onehot = get_all_carbon_environment_labels(mol)
        assert onehot.shape == (n_atoms, NUM_CARBON_CATEGORIES)

    def test_onehot_dtype(self):
        mol = mol_from("CC(=O)O")
        _, _, onehot = get_all_carbon_environment_labels(mol)
        assert onehot.dtype == np.float32

    def test_non_carbon_index_is_minus_one(self):
        mol = mol_from("CC(=O)O")
        _, indices, _ = get_all_carbon_environment_labels(mol)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                assert indices[atom.GetIdx()] == -1

    def test_non_carbon_onehot_all_zeros(self):
        mol = mol_from("CC(=O)O")
        _, _, onehot = get_all_carbon_environment_labels(mol)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                assert np.all(onehot[atom.GetIdx()] == 0.0)

    def test_carbon_onehot_is_one_hot(self):
        """Each carbon row should have exactly one 1.0 entry."""
        mol = mol_from("CC(=O)O")
        _, _, onehot = get_all_carbon_environment_labels(mol)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:
                row = onehot[atom.GetIdx()]
                assert row.sum() == pytest.approx(1.0)
                assert (row == 1.0).sum() == 1

    def test_single_atom_methane(self):
        mol = mol_from("C")
        names, indices, onehot = get_all_carbon_environment_labels(mol)
        assert len(names) == 1
        assert names[0] == "C_methyl"
        assert onehot.shape == (1, NUM_CARBON_CATEGORIES)

    def test_benzene_all_aromatic(self):
        mol = mol_from("c1ccccc1")
        names, _, _ = get_all_carbon_environment_labels(mol)
        assert all(n == "C_aromatic" for n in names)

    def test_consistency_label_and_onehot(self):
        """The 1-hot position should match the integer category index."""
        mol = mol_from("CC#N")
        names, indices, onehot = get_all_carbon_environment_labels(mol)
        for atom in mol.GetAtoms():
            i = atom.GetIdx()
            if indices[i] >= 0:  # carbon
                assert onehot[i, indices[i]] == pytest.approx(1.0)
