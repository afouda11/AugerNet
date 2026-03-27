"""
Tests for augernet.gnn_train_utils — symmetry properties (Tier C)

Wraps the existing runtime symmetry checks in gnn_train_utils as proper
pytest test functions.  All tests are skipped when torch or torch_geometric
is not installed.

Tests
-----
  MPNN model (EQ)   — permutation equivariance
  MPNN model (IN)   — permutation equivariance
  MPNN model (EQ)   — rotation + translation invariance
  MPNN model (IN)   — rotation + translation invariance
  EQ layer          — permutation equivariance
  IN layer          — permutation equivariance
  EQ layer          — rotation + translation equivariance
  IN layer          — rotation + translation invariance
"""

import copy
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.loader import DataLoader

from augernet.gnn_train_utils import (
    MPNN,
    EquivariantMPNNLayer,
    InvariantMPNNLayer,
    permutation_equivariance_unit_test_model,
    permutation_equivariance_unit_test_layer,
    rot_trans_invariance_unit_test,
    rot_trans_equivariance_unit_test,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared model/loader builders
# ─────────────────────────────────────────────────────────────────────────────

IN_DIM   = 3    # must match mini_pyg_data.x.shape[1]
EDGE_DIM = 4    # must match mini_pyg_data.edge_attr.shape[1]
EMB_DIM  = 16   # small for speed


def _make_loader(mini_pyg_data):
    return DataLoader(
        [copy.deepcopy(mini_pyg_data)],
        batch_size=1,
        shuffle=False,
    )


def _make_mpnn(layer_type: str) -> MPNN:
    return MPNN(
        num_layers=2,
        emb_dim=EMB_DIM,
        in_dim=IN_DIM,
        edge_dim=EDGE_DIM,
        out_dim=1,
        layer_type=layer_type,
        pred_type="CEBE",
    ).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Model-level: permutation equivariance
# ─────────────────────────────────────────────────────────────────────────────

class TestModelPermutationEquivariance:
    def test_eq_model(self, mini_pyg_data):
        model = _make_mpnn("EQ")
        loader = _make_loader(mini_pyg_data)
        assert permutation_equivariance_unit_test_model(model, loader)

    def test_in_model(self, mini_pyg_data):
        model = _make_mpnn("IN")
        loader = _make_loader(mini_pyg_data)
        assert permutation_equivariance_unit_test_model(model, loader)


# ─────────────────────────────────────────────────────────────────────────────
# Model-level: rotation + translation invariance
# ─────────────────────────────────────────────────────────────────────────────

class TestModelRotTransInvariance:
    def test_eq_model(self, mini_pyg_data):
        model = _make_mpnn("EQ")
        loader = _make_loader(mini_pyg_data)
        assert rot_trans_invariance_unit_test(model, loader)

    def test_in_model(self, mini_pyg_data):
        model = _make_mpnn("IN")
        loader = _make_loader(mini_pyg_data)
        assert rot_trans_invariance_unit_test(model, loader)


# ─────────────────────────────────────────────────────────────────────────────
# Layer-level: permutation equivariance
# ─────────────────────────────────────────────────────────────────────────────

class TestLayerPermutationEquivariance:
    def test_eq_layer(self, mini_pyg_data):
        model = _make_mpnn("EQ")
        layer = model.convs[0]
        loader = _make_loader(mini_pyg_data)
        assert permutation_equivariance_unit_test_layer(
            layer, loader, lin_in=model.lin_in
        )

    def test_in_layer(self, mini_pyg_data):
        model = _make_mpnn("IN")
        layer = model.convs[0]
        loader = _make_loader(mini_pyg_data)
        assert permutation_equivariance_unit_test_layer(
            layer, loader, lin_in=model.lin_in
        )


# ─────────────────────────────────────────────────────────────────────────────
# Layer-level: rotation + translation symmetry
# ─────────────────────────────────────────────────────────────────────────────

class TestLayerRotTransSymmetry:
    def test_eq_layer_equivariance(self, mini_pyg_data):
        """EQ layer output features are invariant; coordinates are equivariant."""
        model = _make_mpnn("EQ")
        layer = model.convs[0]
        loader = _make_loader(mini_pyg_data)
        assert rot_trans_equivariance_unit_test(
            layer, loader, lin_in=model.lin_in
        )

    def test_in_layer_invariance(self, mini_pyg_data):
        """IN layer output features are invariant to rotations/translations."""
        model = _make_mpnn("IN")
        layer = model.convs[0]
        loader = _make_loader(mini_pyg_data)
        assert rot_trans_invariance_unit_test(
            layer, loader, lin_in=model.lin_in
        )


# ─────────────────────────────────────────────────────────────────────────────
# run_unit_tests convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TestRunUnitTests:
    """Smoke-test the high-level run_unit_tests() wrapper used at training time."""

    def test_eq_all_pass(self, mini_pyg_data):
        from augernet.gnn_train_utils import run_unit_tests
        model = _make_mpnn("EQ")
        results = run_unit_tests(model, [mini_pyg_data], layer_type="EQ")
        assert all(results.values()), f"Some tests failed: {results}"

    def test_in_all_pass(self, mini_pyg_data):
        from augernet.gnn_train_utils import run_unit_tests
        model = _make_mpnn("IN")
        results = run_unit_tests(model, [mini_pyg_data], layer_type="IN")
        assert all(results.values()), f"Some tests failed: {results}"
