"""
Tests for augernet.eneg_diff

Tier B — numpy + rdkit (no torch required)

Covered:
  give_paulingdata   — parses the embedded periodic-table string
  get_eleneg_diff_mat — constructs the (100, 100) pairwise EN difference matrix
  get_e_neg_score     — computes per-atom electronegativity-difference scores
"""

import pytest
import numpy as np

# Skip the whole module if rdkit is unavailable
pytest.importorskip("rdkit")

from augernet.eneg_diff import (
    give_paulingdata,
    get_eleneg_diff_mat,
    get_e_neg_score,
    pdata,
)


# ─────────────────────────────────────────────────────────────────────────────
# give_paulingdata
# ─────────────────────────────────────────────────────────────────────────────

class TestGivePaulingData:
    @pytest.fixture(autouse=True)
    def _data(self):
        self.pd = give_paulingdata(pdata)

    def test_returns_dict(self):
        assert isinstance(self.pd, dict)

    def test_hydrogen_electronegativity(self):
        assert self.pd[1] == pytest.approx(2.20)

    def test_carbon_electronegativity(self):
        assert self.pd[6] == pytest.approx(2.55)

    def test_nitrogen_electronegativity(self):
        assert self.pd[7] == pytest.approx(3.04)

    def test_oxygen_electronegativity(self):
        assert self.pd[8] == pytest.approx(3.44)

    def test_fluorine_electronegativity(self):
        assert self.pd[9] == pytest.approx(3.98)

    def test_noble_gas_zero(self):
        # He and Ar are explicitly set to 0 in the data
        assert self.pd[2] == pytest.approx(0.0)
        assert self.pd[18] == pytest.approx(0.0)

    def test_all_values_finite_or_zero(self):
        for v in self.pd.values():
            assert np.isfinite(v) or v == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# get_eleneg_diff_mat
# ─────────────────────────────────────────────────────────────────────────────

class TestGetElenegDiffMat:
    @pytest.fixture(autouse=True)
    def _mat(self):
        self.mat = get_eleneg_diff_mat(num_elements=100)

    def test_shape(self):
        assert self.mat.shape == (100, 100)

    def test_diagonal_zero(self):
        """EN difference of an element with itself must be zero."""
        diag = np.diag(self.mat)
        assert np.allclose(diag, 0.0)

    def test_antisymmetric(self):
        """mat[i, j] == -mat[j, i]  (EN_i - EN_j = -(EN_j - EN_i))"""
        assert np.allclose(self.mat, -self.mat.T)

    def test_known_value_C_minus_H(self):
        """EN[C] - EN[H] ≈ 2.55 - 2.20 = 0.35"""
        c_idx = 6 - 1   # atomic num 6 → index 5
        h_idx = 1 - 1   # atomic num 1 → index 0
        assert self.mat[c_idx, h_idx] == pytest.approx(2.55 - 2.20, abs=1e-9)

    def test_known_value_O_minus_C(self):
        """EN[O] - EN[C] ≈ 3.44 - 2.55 = 0.89"""
        o_idx = 8 - 1
        c_idx = 6 - 1
        assert self.mat[o_idx, c_idx] == pytest.approx(3.44 - 2.55, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# get_e_neg_score
# ─────────────────────────────────────────────────────────────────────────────

class TestGetEScoreNeg:
    def test_returns_list(self):
        scores = get_e_neg_score("C")
        assert isinstance(scores, list)

    def test_length_matches_atoms_with_H(self):
        """get_full_neighbor_vectors operates on mol with AddHs → all atoms."""
        from augernet.eneg_diff import get_full_neighbor_vectors
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CO")
        mol_h = Chem.AddHs(mol)
        n_atoms = mol_h.GetNumAtoms()
        scores = get_e_neg_score("CO")
        assert len(scores) == n_atoms

    def test_N2_score_is_zero(self):
        """
        In molecular nitrogen (N≡N) every atom has only N neighbours.
        EN[N] - EN[N] = 0, so the score is 0 for all atoms.
        """
        scores = get_e_neg_score("N#N")
        assert all(s == pytest.approx(0.0) for s in scores)

    def test_carbon_in_CH4_positive(self):
        """
        C is more electronegative than H, so the C in methane has a
        positive e-score (EN[C] > EN[H]).
        """
        scores = get_e_neg_score("C")  # methane
        # First atom in RDKit mol from "C" SMILES is the carbon
        from rdkit import Chem
        mol = Chem.MolFromSmiles("C")
        mol_h = Chem.AddHs(mol)
        c_idx = next(
            a.GetIdx() for a in mol_h.GetAtoms() if a.GetAtomicNum() == 6
        )
        assert scores[c_idx] > 0.0

    def test_h_in_CH4_negative(self):
        """H neighbours of C have negative score (EN[H] < EN[C])."""
        from rdkit import Chem
        mol = Chem.MolFromSmiles("C")
        mol_h = Chem.AddHs(mol)
        scores = get_e_neg_score("C")
        h_indices = [a.GetIdx() for a in mol_h.GetAtoms() if a.GetAtomicNum() == 1]
        for idx in h_indices:
            assert scores[idx] < 0.0

    def test_CH4_carbon_score_value(self):
        """
        Methane: C has 4 H neighbours (single bonds).
        score = (EN[C] - EN[H]) * 4 = (2.55 - 2.20) * 4 = 1.40
        """
        from rdkit import Chem
        mol_h = Chem.AddHs(Chem.MolFromSmiles("C"))
        c_idx = next(a.GetIdx() for a in mol_h.GetAtoms() if a.GetAtomicNum() == 6)
        scores = get_e_neg_score("C")
        assert scores[c_idx] == pytest.approx(1.40, abs=1e-6)
