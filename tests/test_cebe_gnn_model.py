"""
CEBE-GNN model (MPNN) tests.

Tests model construction, forward pass shapes, and symmetry properties
for both Invariant and Equivariant MPNN layers.

Essential tests use the fast mini_pyg_data fixture.
Full tests repeat with the real molecule graph.
"""

import pytest
import numpy as np

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from augernet.gnn_train_utils import (
    MPNN,
    InvariantMPNNLayer,
    EquivariantMPNNLayer,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _random_rotation_matrix():
    """3x3 orthogonal matrix with det +1 (proper rotation)."""
    from scipy.stats import ortho_group
    Q = ortho_group.rvs(3)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return torch.tensor(Q, dtype=torch.float)


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------

@pytest.mark.essential
class TestCebeGnnModelBuild:

    @pytest.mark.parametrize("layer_type", ["IN", "EQ"])
    def test_mpnn_constructs(self, layer_type):
        model = MPNN(
            num_layers=2, emb_dim=16, in_dim=3, edge_dim=4,
            out_dim=1, layer_type=layer_type, pred_type="CEBE",
        )
        assert isinstance(model, torch.nn.Module)

    @pytest.mark.parametrize("layer_type", ["IN", "EQ"])
    def test_forward_output_shape(self, mini_pyg_data, layer_type):
        model = MPNN(
            num_layers=2, emb_dim=16, in_dim=3, edge_dim=4,
            out_dim=1, layer_type=layer_type, pred_type="CEBE",
        )
        model.eval()
        with torch.no_grad():
            out = model(mini_pyg_data)
        assert out.shape == (mini_pyg_data.x.size(0), 1)

    def test_lin_pred_exists_for_cebe(self):
        model = MPNN(
            num_layers=2, emb_dim=16, in_dim=3, edge_dim=4,
            out_dim=1, layer_type="IN", pred_type="CEBE",
        )
        assert hasattr(model, "lin_pred")


# ---------------------------------------------------------------------------
# invariant layer symmetry tests (essential - uses mini graph)
# ---------------------------------------------------------------------------

@pytest.mark.essential
class TestInvariantLayerSymmetry:

    def _make_layer(self, emb_dim=16, edge_dim=4):
        layer = InvariantMPNNLayer(emb_dim=emb_dim, edge_dim=edge_dim)
        layer.eval()
        return layer

    def test_translation_invariance(self, mini_pyg_data):
        """Shifting all positions by a constant should not change output."""
        layer = self._make_layer(emb_dim=16, edge_dim=4)
        h = torch.randn(5, 16)
        t = torch.randn(1, 3) * 10
        with torch.no_grad():
            out1 = layer(h, mini_pyg_data.pos,
                         mini_pyg_data.edge_index, mini_pyg_data.edge_attr)
            out2 = layer(h, mini_pyg_data.pos + t,
                         mini_pyg_data.edge_index, mini_pyg_data.edge_attr)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_rotation_invariance(self, mini_pyg_data):
        """Rotating all positions should not change output (invariant layer)."""
        layer = self._make_layer(emb_dim=16, edge_dim=4)
        h = torch.randn(5, 16)
        R = _random_rotation_matrix()
        pos_rot = mini_pyg_data.pos @ R.T
        with torch.no_grad():
            out1 = layer(h, mini_pyg_data.pos,
                         mini_pyg_data.edge_index, mini_pyg_data.edge_attr)
            out2 = layer(h, pos_rot,
                         mini_pyg_data.edge_index, mini_pyg_data.edge_attr)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_permutation_equivariance(self, mini_pyg_data):
        """Permuting nodes should permute outputs the same way."""
        layer = self._make_layer(emb_dim=16, edge_dim=4)
        n = mini_pyg_data.x.size(0)
        h = torch.randn(n, 16)
        perm = torch.randperm(n)
        inv = torch.argsort(perm)

        with torch.no_grad():
            out1 = layer(h, mini_pyg_data.pos,
                         mini_pyg_data.edge_index, mini_pyg_data.edge_attr)
            out2 = layer(h[perm], mini_pyg_data.pos[perm],
                         inv[mini_pyg_data.edge_index], mini_pyg_data.edge_attr)
        assert torch.allclose(out1[perm], out2, atol=1e-5)


# ---------------------------------------------------------------------------
# equivariant layer symmetry tests (essential - uses mini graph)
# ---------------------------------------------------------------------------

@pytest.mark.essential
class TestEquivariantLayerSymmetry:

    def _make_layer(self, emb_dim=16, edge_dim=4):
        layer = EquivariantMPNNLayer(emb_dim=emb_dim, edge_dim=edge_dim)
        layer.eval()
        return layer

    def test_translation_equivariance(self, mini_pyg_data):
        """Translated positions -> h unchanged, pos shifted by same amount."""
        layer = self._make_layer(emb_dim=16, edge_dim=4)
        h = torch.randn(5, 16)
        t = torch.randn(1, 3) * 10
        with torch.no_grad():
            h1, pos1 = layer(h, mini_pyg_data.pos,
                             mini_pyg_data.edge_index, mini_pyg_data.edge_attr)
            h2, pos2 = layer(h, mini_pyg_data.pos + t,
                             mini_pyg_data.edge_index, mini_pyg_data.edge_attr)
        assert torch.allclose(h1, h2, atol=1e-5)
        assert torch.allclose(pos1 + t, pos2, atol=1e-5)

    def test_rotation_equivariance(self, mini_pyg_data):
        """Rotated positions -> h unchanged, pos rotated."""
        layer = self._make_layer(emb_dim=16, edge_dim=4)
        h = torch.randn(5, 16)
        R = _random_rotation_matrix()
        pos_rot = mini_pyg_data.pos @ R.T
        with torch.no_grad():
            h1, pos1 = layer(h, mini_pyg_data.pos,
                             mini_pyg_data.edge_index, mini_pyg_data.edge_attr)
            h2, pos2 = layer(h, pos_rot,
                             mini_pyg_data.edge_index, mini_pyg_data.edge_attr)
        assert torch.allclose(h1, h2, atol=1e-5)
        assert torch.allclose(pos1 @ R.T, pos2, atol=1e-4)

    def test_permutation_equivariance(self, mini_pyg_data):
        """Permuting nodes -> permuted outputs."""
        layer = self._make_layer(emb_dim=16, edge_dim=4)
        n = mini_pyg_data.x.size(0)
        h = torch.randn(n, 16)
        perm = torch.randperm(n)
        inv = torch.argsort(perm)

        with torch.no_grad():
            h1, pos1 = layer(h, mini_pyg_data.pos,
                             mini_pyg_data.edge_index, mini_pyg_data.edge_attr)
            h2, pos2 = layer(h[perm], mini_pyg_data.pos[perm],
                             inv[mini_pyg_data.edge_index], mini_pyg_data.edge_attr)
        assert torch.allclose(h1[perm], h2, atol=1e-5)
        assert torch.allclose(pos1[perm], pos2, atol=1e-5)


# ---------------------------------------------------------------------------
# full-model invariance tests (essential - uses mini graph)
# ---------------------------------------------------------------------------

@pytest.mark.essential
class TestCebeGnnModelInvariance:

    @pytest.mark.parametrize("layer_type", ["IN", "EQ"])
    def test_translation_invariance(self, mini_pyg_data, layer_type):
        """CEBE predictions should not change under global translation."""
        from torch_geometric.data import Data
        model = MPNN(
            num_layers=2, emb_dim=16, in_dim=3, edge_dim=4,
            out_dim=1, layer_type=layer_type, pred_type="CEBE",
        )
        model.eval()
        t = torch.randn(1, 3) * 10
        shifted = Data(
            x=mini_pyg_data.x.clone(),
            pos=mini_pyg_data.pos + t,
            edge_index=mini_pyg_data.edge_index.clone(),
            edge_attr=mini_pyg_data.edge_attr.clone(),
        )
        with torch.no_grad():
            out1 = model(mini_pyg_data)
            out2 = model(shifted)
        assert torch.allclose(out1, out2, atol=1e-5)

    @pytest.mark.parametrize("layer_type", ["IN", "EQ"])
    def test_rotation_invariance(self, mini_pyg_data, layer_type):
        """CEBE predictions should not change under rotation."""
        from torch_geometric.data import Data
        model = MPNN(
            num_layers=2, emb_dim=16, in_dim=3, edge_dim=4,
            out_dim=1, layer_type=layer_type, pred_type="CEBE",
        )
        model.eval()
        R = _random_rotation_matrix()
        rotated = Data(
            x=mini_pyg_data.x.clone(),
            pos=mini_pyg_data.pos @ R.T,
            edge_index=mini_pyg_data.edge_index.clone(),
            edge_attr=mini_pyg_data.edge_attr.clone(),
        )
        with torch.no_grad():
            out1 = model(mini_pyg_data)
            out2 = model(rotated)
        assert torch.allclose(out1, out2, atol=1e-4)


# ---------------------------------------------------------------------------
# full tests with real molecule graph
# ---------------------------------------------------------------------------

@pytest.mark.full
class TestCebeGnnModelRealMol:

    def test_forward_shape(self, real_mol_graph):
        in_dim = real_mol_graph.x.size(1)
        edge_dim = real_mol_graph.edge_attr.size(1)
        model = MPNN(
            num_layers=3, emb_dim=64, in_dim=in_dim, edge_dim=edge_dim,
            out_dim=1, layer_type="IN", pred_type="CEBE",
        )
        model.eval()
        with torch.no_grad():
            out = model(real_mol_graph)
        assert out.shape == (real_mol_graph.x.size(0), 1)

    def test_translation_invariance_real(self, real_mol_graph):
        from torch_geometric.data import Data
        in_dim = real_mol_graph.x.size(1)
        edge_dim = real_mol_graph.edge_attr.size(1)
        model = MPNN(
            num_layers=3, emb_dim=64, in_dim=in_dim, edge_dim=edge_dim,
            out_dim=1, layer_type="IN", pred_type="CEBE",
        )
        model.eval()
        t = torch.randn(1, 3) * 10
        shifted = Data(
            x=real_mol_graph.x.clone(),
            pos=real_mol_graph.pos + t,
            edge_index=real_mol_graph.edge_index.clone(),
            edge_attr=real_mol_graph.edge_attr.clone(),
        )
        with torch.no_grad():
            out1 = model(real_mol_graph)
            out2 = model(shifted)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_rotation_invariance_real(self, real_mol_graph):
        from torch_geometric.data import Data
        in_dim = real_mol_graph.x.size(1)
        edge_dim = real_mol_graph.edge_attr.size(1)
        model = MPNN(
            num_layers=3, emb_dim=64, in_dim=in_dim, edge_dim=edge_dim,
            out_dim=1, layer_type="EQ", pred_type="CEBE",
        )
        model.eval()
        R = _random_rotation_matrix()
        rotated = Data(
            x=real_mol_graph.x.clone(),
            pos=real_mol_graph.pos @ R.T,
            edge_index=real_mol_graph.edge_index.clone(),
            edge_attr=real_mol_graph.edge_attr.clone(),
        )
        with torch.no_grad():
            out1 = model(real_mol_graph)
            out2 = model(rotated)
        assert torch.allclose(out1, out2, atol=1e-4)
