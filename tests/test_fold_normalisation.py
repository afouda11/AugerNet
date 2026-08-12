"""Guard tests for per-fold normalisation (B11).

The invariant under test: every constant fitted across molecules — the CEBE
target shift/scale, the Auger intensity scale, the node-feature statistics — is
fitted on the TRAINING molecules of a fold only, and never on the validation
fold or the calculated hold-out.

These are the regression tests for the change; if anyone reintroduces a
dataset-wide fit, `test_constants_differ_between_folds` and
`test_holdout_does_not_influence_fit` fail.

Run:  pytest tests/test_fold_normalisation.py -v
"""

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from augernet import DATA_PROCESSED_DIR
from augernet.backend_gnn import (
    _apply_cebe_norm,
    _cebe_delta,
    _fit_cebe_norm,
    _deserialise_feature_stats,
    _serialise_feature_stats,
    load_norm_sidecar,
    norm_sidecar_path,
    save_norm_sidecar,
)

CALC_PT = os.path.join(DATA_PROCESSED_DIR, "gnn_calc_cebe_data.pt")
LEGACY_STATS = os.path.join(DATA_PROCESSED_DIR, "cebe_norm_stats.pt")


# ── synthetic fixtures (no data files needed) ────────────────────────────────

def _mol(n_carbon, n_atoms, shift=0.0, seed=0):
    """Minimal stand-in carrying only what the norm helpers touch."""
    from torch_geometric.data import Data
    g = torch.Generator().manual_seed(seed)
    node_mask = torch.zeros(n_atoms)
    node_mask[:n_carbon] = 1.0
    true_cebe = torch.where(
        node_mask > 0.5,
        290.0 + shift + torch.randn(n_atoms, generator=g) * 2.0,
        torch.full((n_atoms,), -1.0),
    )
    return Data(
        node_mask=node_mask,
        atomic_be_eV=torch.full((n_atoms,), 308.23974136400005),
        true_cebe=true_cebe,
        cebe_y=torch.zeros(n_atoms, 1),
    )


@pytest.fixture
def calc_data():
    return [_mol(3, 8, shift=i * 0.1, seed=i) for i in range(40)]


@pytest.fixture
def holdout():
    # deliberately off-distribution: if it leaks into a fit, the fit moves
    return [_mol(3, 8, shift=50.0, seed=900 + i) for i in range(10)]


# ── the invariant ────────────────────────────────────────────────────────────

@pytest.mark.essential
def test_constants_differ_between_folds(calc_data):
    a = _fit_cebe_norm(calc_data, list(range(0, 32)))
    b = _fit_cebe_norm(calc_data, list(range(8, 40)))
    assert a["mean"] != b["mean"] or a["std"] != b["std"], (
        "Fold constants are identical — normalisation is being fitted "
        "dataset-wide again rather than per fold."
    )


@pytest.mark.essential
def test_holdout_does_not_influence_fit(calc_data, holdout):
    train_idx = list(range(0, 32))
    before = _fit_cebe_norm(calc_data, train_idx)
    after = _fit_cebe_norm(calc_data + holdout, train_idx)
    assert before == after, (
        "Appending hold-out molecules changed the fitted constants — the fit "
        "is not restricted to train_idx."
    )


@pytest.mark.essential
def test_apply_reaches_every_split_and_round_trips(calc_data, holdout):
    norm = _fit_cebe_norm(calc_data, list(range(0, 32)))
    _apply_cebe_norm([calc_data, holdout], norm)

    for split in (calc_data, holdout):
        d = split[0]
        mask = d.node_mask.view(-1) > 0.5
        recon = d.cebe_y.view(-1)[mask] * norm["std"] + norm["mean"]
        assert torch.allclose(recon, _cebe_delta(d)[mask], atol=1e-4)
        assert (d.cebe_y.view(-1)[~mask] == -1.0).all(), "sentinel overwritten"


@pytest.mark.essential
def test_fit_requires_training_molecules(calc_data):
    with pytest.raises(ValueError):
        _fit_cebe_norm(calc_data, [])


# ── sidecar ──────────────────────────────────────────────────────────────────

@pytest.mark.essential
def test_sidecar_round_trip(tmp_path):
    model_path = str(tmp_path / "cebe_gnn_demo_fold3.pth")
    fs = {"atomic_be": (torch.tensor([[0.5]]), torch.tensor([[0.1]]))}
    norm = {
        "model": "cebe-gnn", "feature_keys": "035", "node_feature_norm": "data",
        "n_train_molecules": 32, "fold": 3,
        "cebe": {"mean": 18.34, "std": 2.22, "n_carbons": 96},
        "feature_stats": _serialise_feature_stats(fs),
    }
    save_norm_sidecar(model_path, norm)
    assert os.path.isfile(norm_sidecar_path(model_path))

    back = load_norm_sidecar(model_path)
    assert back["cebe"] == norm["cebe"]
    rehydrated = _deserialise_feature_stats(back["feature_stats"])
    assert torch.allclose(rehydrated["atomic_be"][0], fs["atomic_be"][0])
    assert torch.allclose(rehydrated["atomic_be"][1], fs["atomic_be"][1])


@pytest.mark.essential
def test_missing_sidecar_raises_and_does_not_fall_back(tmp_path):
    """A missing sidecar must be fatal — never a silent dataset-wide substitute."""
    model_path = str(tmp_path / "no_sidecar_fold1.pth")
    with pytest.raises(FileNotFoundError, match="not available"):
        load_norm_sidecar(model_path)


# ── against the real dataset, when present ───────────────────────────────────

@pytest.mark.full
@pytest.mark.skipif(not os.path.isfile(CALC_PT), reason="processed data not built")
def test_raw_target_is_reconstructable_from_stored_graphs():
    """cebe_y must be derivable as atomic_be_eV - true_cebe.

    This is what lets the target be re-normalised per fold without
    regenerating any .pt file.  Graphs built before B11 store a NORMALISED
    cebe_y, so only the reconstruction itself is asserted here — the runtime
    never trusts the stored value.
    """
    from augernet import gnn_train_utils as gtu
    from augernet import DATA_DIR

    ds = gtu.LoadDataset(DATA_DIR, file_name="gnn_calc_cebe_data.pt")
    for i in range(min(25, len(ds))):
        d = ds[i]
        mask = d.node_mask.view(-1) > 0.5
        delta = _cebe_delta(d)[mask]
        assert torch.isfinite(delta).all()
        # physical sanity: C 1s chemical shifts sit well inside this window
        assert (delta > 0).all() and (delta < 60).all()


@pytest.mark.full
@pytest.mark.skipif(
    not (os.path.isfile(CALC_PT) and os.path.isfile(LEGACY_STATS)),
    reason="needs processed data + the retired cebe_norm_stats.pt",
)
def test_full_set_fit_reproduces_retired_global_stats():
    """Fitting on ALL molecules must reproduce the old dataset-wide constants.

    Proves the new estimator is the same estimator, just restricted to the
    training fold.  Delete this test once cebe_norm_stats.pt is gone from
    data/processed.
    """
    from augernet import gnn_train_utils as gtu
    from augernet import DATA_DIR

    legacy = torch.load(LEGACY_STATS, weights_only=False)
    ds = gtu.LoadDataset(DATA_DIR, file_name="gnn_calc_cebe_data.pt")
    calc = [ds[i] for i in range(len(ds))]
    fitted = _fit_cebe_norm(calc, list(range(len(calc))))

    assert fitted["mean"] == pytest.approx(legacy["mean"], abs=1e-3)
    assert fitted["std"] == pytest.approx(legacy["std"], abs=1e-3)
