"""
Shared fixtures for the AugerNet test suite.

Provides two fixture scopes:
  - mock_data:       lightweight SimpleNamespace (no torch_geometric needed)
  - real_mol_graph:  PyG Data built from tests/test_mol/dsgdb9nsd_133427.xyz

Markers
-------
  essential : fast tests for CI (GitHub Actions)
  full      : longer tests (data preparation, training loops, etc.)
"""

import os
import shutil
import tempfile
import types
import pytest
import numpy as np

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_MOL_DIR = os.path.join(TEST_DIR, "test_mol")
MOL_NAME = "dsgdb9nsd_133427"
XYZ_PATH = os.path.join(TEST_MOL_DIR, f"{MOL_NAME}.xyz")
CEBE_PATH = os.path.join(TEST_MOL_DIR, f"{MOL_NAME}_out.txt")

# Feature keys assembled onto real_mol_graph: skipatom_200 + atomic_be + e_score,
# matching the '035' default used by the shipped configs.  Gives in_dim = 202.
REAL_MOL_FEATURE_KEYS = [0, 3, 5]


# -- pytest markers ----------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "essential: fast CI tests")
    config.addinivalue_line("markers", "full: longer tests (data prep, training)")


# -- sandbox cwd (avoid polluting repo root with *_results dirs) --------------

@pytest.fixture(autouse=True, scope="session")
def _sandbox_cwd():
    """chdir into a temp directory under tests/ for the entire test session.

    config.resolve() calls os.makedirs(result_dir) using os.getcwd(),
    so without this creates temp dir created inside tests/ and removed 
    automatically when the session finishes.
    """
    orig = os.getcwd()
    tmpdir = tempfile.mkdtemp(prefix="augernet_tests_", dir=TEST_DIR)
    os.chdir(tmpdir)
    yield tmpdir
    os.chdir(orig)
    shutil.rmtree(tmpdir, ignore_errors=True)


# -- mock Data (no torch_geometric) ------------------------------------------

def make_mock_data(n_atoms=4, n_carbon_categories=8):
    """SimpleNamespace mimicking a PyG Data with all feature attributes."""
    torch = pytest.importorskip("torch")
    data = types.SimpleNamespace()
    # Zero-width placeholder, matching what build_molecular_graphs emits:
    # assemble_node_features REPLACES data.x with the selected features, it does
    # not append to it.  This used to be a 3-column category_feature, which is
    # now retired (the block is commented out in _build_node_and_edge_features),
    # so a 3-column mock made every assembled width look 3 too wide.
    data.x = torch.zeros(n_atoms, 0)
    data.skipatom_200 = torch.randn(n_atoms, 200)
    data.skipatom_30 = torch.randn(n_atoms, 30)
    data.onehot = torch.randn(n_atoms, 5)
    data.atomic_be = torch.randn(n_atoms, 1)
    data.mol_be = torch.randn(n_atoms, 1)
    data.e_score = torch.randn(n_atoms, 1)
    data.env_onehot = torch.randn(n_atoms, n_carbon_categories)
    return data


@pytest.fixture
def mock_data():
    return make_mock_data()


# -- real molecule graph (session-scoped, built once) -------------------------

@pytest.fixture(scope="session")
def real_mol_graph_raw():
    """Real PyG Data graph from dsgdb9nsd_133427 (C4H3F2N3), UNASSEMBLED.

    Exactly what ``_build_node_and_edge_features`` produces: ``x`` is a
    zero-width placeholder and the node features live as separate attributes.
    Use this to assert the graph-builder's contract.  For anything that feeds a
    model, use ``real_mol_graph`` instead — a zero-width ``x`` makes
    ``MPNN.lin_in`` a ``Linear(0, emb_dim)`` whose output is the bias alone, so
    symmetry tests would pass without the output depending on the molecule.

    The molecule has 12 atoms: 4C, 3H, 2F, 3N.
    CEBE values are available for 4 carbon atoms.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("rdkit")

    from torch_geometric.data import Data
    from augernet.build_molecular_graphs import (
        _mol_from_xyz_order,
        _build_node_and_edge_features,
        _initialize_all_atom_encoders,
    )
    from augernet import DATA_RAW_DIR

    skipatom_dir = os.path.join(DATA_RAW_DIR, "skipatom")
    all_encoders = _initialize_all_atom_encoders(skipatom_dir)

    mol, xyz_symbols, pos, smiles = _mol_from_xyz_order(XYZ_PATH, labeled_atoms=False)
    cebe = np.loadtxt(CEBE_PATH)

    node_features, x, edge_index, edge_attr, atomic_be, carbon_env_indices, carbon_env_labels = \
        _build_node_and_edge_features(mol, all_encoders, cebe)

    node_mask = [0.0 if v == -1 else 1.0 for v in cebe]

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=torch.tensor(pos, dtype=torch.float),
        atomic_be_eV=atomic_be,
        node_mask=torch.FloatTensor(node_mask),
        y=torch.randn(len(xyz_symbols), 1),  # dummy targets
        atom_symbols=xyz_symbols,
        smiles=smiles,
        mol_name=MOL_NAME,
        carbon_env_labels=carbon_env_labels,
        carbon_env_indices=torch.tensor(carbon_env_indices, dtype=torch.long),
    )
    for attr_name, tensor in node_features.items():
        setattr(data, attr_name, tensor)

    assert data.x.shape[1] == 0, (
        "real_mol_graph_raw is meant to be unassembled — the builder should "
        "return a zero-width x placeholder."
    )
    return data


@pytest.fixture(scope="session")
def real_mol_graph(real_mol_graph_raw):
    """``real_mol_graph_raw`` with node features assembled, ready for a model.

    Mirrors what the training pipeline does before any graph reaches the
    network: feature_assembly concatenates the selected features into ``x``.
    Deep-copied so the raw fixture stays unassembled for the builder tests.
    """
    import copy

    from augernet.feature_assembly import assemble_dataset

    data = copy.deepcopy(real_mol_graph_raw)
    assemble_dataset([data], REAL_MOL_FEATURE_KEYS, scale_mode="graph")
    assert data.x.size(1) > 0, "real_mol_graph: feature assembly produced no columns"
    return data


# -- mini synthetic graph (for fast layer-only tests) -------------------------

@pytest.fixture
def mini_pyg_data():
    """Tiny synthetic graph (5 nodes, ring) for fast symmetry tests."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data

    N, in_dim, edge_dim = 5, 3, 4
    src = [0, 1, 1, 2, 2, 3, 3, 4, 4, 0]
    dst = [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    return Data(
        x=torch.randn(N, in_dim),
        pos=torch.randn(N, 3),
        edge_index=edge_index,
        edge_attr=torch.zeros(edge_index.size(1), edge_dim),
        node_mask=torch.ones(N),
        y=torch.randn(N, 1),
    )
