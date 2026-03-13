"""
Shared fixtures and pytest configuration for the AugerNet test suite.
"""

import types
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_torch: mark test as requiring PyTorch (skipped if torch is not installed)",
    )


# ---------------------------------------------------------------------------
# Lightweight mock Data object (no torch_geometric import required)
# ---------------------------------------------------------------------------

def make_mock_data(n_atoms: int = 4, n_carbon_categories: int = 8):
    """
    Build a SimpleNamespace mimicking a PyG Data object with all ``feat_*``
    attributes pre-populated for use in feature-assembly tests.

    Uses torch tensors but does NOT import torch_geometric.
    """
    torch = pytest.importorskip("torch")

    data = types.SimpleNamespace()
    # category_feature (3 columns: H / C / other one-hot, used as base of data.x)
    data.x = torch.zeros(n_atoms, 3)

    data.feat_skipatom_200 = torch.randn(n_atoms, 200)
    data.feat_skipatom_30  = torch.randn(n_atoms, 30)
    data.feat_onehot       = torch.randn(n_atoms, 5)
    data.feat_atomic_be    = torch.randn(n_atoms, 1)
    data.feat_mol_be       = torch.randn(n_atoms, 1)
    data.feat_e_score      = torch.randn(n_atoms, 1)
    data.feat_env_onehot   = torch.randn(n_atoms, n_carbon_categories)
    data.feat_morgan_fp    = torch.randn(n_atoms, 256)
    return data


@pytest.fixture
def mock_data():
    """Minimal mock Data object with all feat_* attributes."""
    return make_mock_data()


# ---------------------------------------------------------------------------
# Minimal PyG Data fixture for GNN symmetry tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mini_pyg_data():
    """
    A tiny synthetic molecular graph (5 nodes, ring topology) with all
    attributes required by the MPNN forward pass under pred_type='CEBE'.

    Attributes
    ----------
    x          : (5, 3) node features
    pos        : (5, 3) 3-D coordinates
    edge_index : (2, 10) bidirectional edges (ring)
    edge_attr  : (10, 4) edge features (zeros are fine for symmetry tests)
    node_mask  : (5,) all ones (every node is a target carbon)
    y          : (5, 1) dummy target values
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data

    N, in_dim, edge_dim = 5, 3, 4

    # Bidirectional ring: 0-1-2-3-4-0
    src = [0, 1, 1, 2, 2, 3, 3, 4, 4, 0]
    dst = [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    data = Data(
        x          = torch.randn(N, in_dim),
        pos        = torch.randn(N, 3),
        edge_index = edge_index,
        edge_attr  = torch.zeros(edge_index.size(1), edge_dim),
        node_mask  = torch.ones(N),
        y          = torch.randn(N, 1),
    )
    return data
