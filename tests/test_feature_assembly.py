"""
Tests for augernet.feature_assembly

Tier A  — compute_feature_tag, parse_feature_keys, describe_features
          (pure Python logic, but the module-level ``import torch`` means
          the entire module is skipped when torch is absent)
Tier C  — _scale_tensor, assemble_node_features (torch required)
"""

import pytest

# Skip this entire module if torch is not installed.
torch = pytest.importorskip("torch")

from augernet.feature_assembly import (
    FEATURE_CATALOG,
    compute_feature_tag,
    parse_feature_keys,
    describe_features,
    assemble_node_features,
    assemble_dataset,
    _scale_tensor,
)


# ─────────────────────────────────────────────────────────────────────────────
# compute_feature_tag
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeFeatureTag:
    def test_sorted_output(self):
        assert compute_feature_tag([3, 0, 5]) == "035"

    def test_already_sorted(self):
        assert compute_feature_tag([0, 3, 5]) == "035"

    def test_single_key(self):
        assert compute_feature_tag([7]) == "7"

    def test_all_keys(self):
        keys = sorted(FEATURE_CATALOG.keys())
        tag = compute_feature_tag(keys)
        assert tag == "".join(str(k) for k in keys)

    def test_empty_list(self):
        assert compute_feature_tag([]) == ""

    def test_round_trip(self):
        keys = [0, 3, 5]
        assert compute_feature_tag(keys) == compute_feature_tag(sorted(keys))


# ─────────────────────────────────────────────────────────────────────────────
# parse_feature_keys
# ─────────────────────────────────────────────────────────────────────────────

class TestParseFeatureKeys:
    def test_string_form(self):
        assert parse_feature_keys("035") == [0, 3, 5]

    def test_list_form(self):
        assert parse_feature_keys([0, 3, 5]) == [0, 3, 5]

    def test_tuple_form(self):
        assert parse_feature_keys((5, 0, 3)) == [0, 3, 5]

    def test_unsorted_string_is_sorted(self):
        assert parse_feature_keys("530") == [0, 3, 5]

    def test_single_key_string(self):
        assert parse_feature_keys("0") == [0]

    def test_invalid_key_raises(self):
        with pytest.raises(ValueError, match="Unknown feature key"):
            parse_feature_keys("9")

    def test_invalid_key_in_list_no_validation(self):
        # NOTE: parse_feature_keys silently accepts unknown keys when given a
        # list (validation only runs on the string path).  This test documents
        # that current behaviour; if the source is ever tightened to also
        # validate list inputs the assertion below should become a
        # pytest.raises(ValueError) instead.
        result = parse_feature_keys([0, 9])
        assert result == [0, 9]

    def test_round_trip_with_compute_tag(self):
        original_keys = [0, 3, 5]
        tag = compute_feature_tag(original_keys)
        assert parse_feature_keys(tag) == original_keys


# ─────────────────────────────────────────────────────────────────────────────
# describe_features
# ─────────────────────────────────────────────────────────────────────────────

class TestDescribeFeatures:
    def test_known_output(self):
        desc = describe_features([0, 3, 5])
        assert "skipatom_200" in desc
        assert "atomic_be" in desc
        assert "e_score" in desc

    def test_separator(self):
        desc = describe_features([2, 3])
        assert " + " in desc

    def test_single_key(self):
        desc = describe_features([2])
        assert "onehot" in desc
        assert "+" not in desc

    def test_order_sorted(self):
        desc_fwd = describe_features([0, 3])
        desc_rev = describe_features([3, 0])
        assert desc_fwd == desc_rev


# ─────────────────────────────────────────────────────────────────────────────
# _scale_tensor  (Tier C — torch required)
# ─────────────────────────────────────────────────────────────────────────────

class TestScaleTensor:
    def test_output_shape_2d(self):
        t = torch.randn(6, 3)
        out = _scale_tensor(t)
        assert out.shape == t.shape

    def test_output_shape_1d_becomes_2d(self):
        t = torch.randn(6)
        out = _scale_tensor(t)
        assert out.shape == (6, 1)

    def test_near_zero_mean_per_col(self):
        t = torch.randn(100, 4)
        out = _scale_tensor(t)
        assert torch.allclose(out.mean(dim=0), torch.zeros(4), atol=1e-5)

    def test_near_unit_std_per_col(self):
        t = torch.randn(100, 4)
        out = _scale_tensor(t)
        assert torch.allclose(out.std(dim=0), torch.ones(4), atol=1e-4)

    def test_constant_column_does_not_raise(self):
        """A column of all-identical values should not produce NaN/Inf."""
        t = torch.ones(5, 2)
        out = _scale_tensor(t)
        assert torch.isfinite(out).all()


# ─────────────────────────────────────────────────────────────────────────────
# assemble_node_features  (Tier C — torch required)
# ─────────────────────────────────────────────────────────────────────────────

class TestAssembleNodeFeatures:
    def test_output_shape_single_key(self, mock_data):
        """Selecting key 3 (atomic_be, dim=1) appends 1 col to the 3 base cols."""
        data = assemble_node_features(mock_data, feature_keys=[3], inplace=False)
        n_atoms = mock_data.x.shape[0]
        assert data.x.shape == (n_atoms, 3 + 1)

    def test_output_shape_multiple_keys(self, mock_data):
        """Keys [0, 3, 5] → 200 + 1 + 1 = 202 additional cols."""
        data = assemble_node_features(mock_data, feature_keys=[0, 3, 5], inplace=False)
        n_atoms = mock_data.x.shape[0]
        assert data.x.shape == (n_atoms, 3 + 200 + 1 + 1)

    def test_inplace_false_does_not_modify_original(self, mock_data):
        original_shape = mock_data.x.shape
        _ = assemble_node_features(mock_data, feature_keys=[3], inplace=False)
        assert mock_data.x.shape == original_shape

    def test_inplace_true_modifies_data(self, mock_data):
        assemble_node_features(mock_data, feature_keys=[3], inplace=True)
        n_atoms = 4  # from conftest make_mock_data default
        assert mock_data.x.shape == (n_atoms, 3 + 1)

    def test_idempotent_reassembly(self, mock_data):
        """Calling assemble twice with different keys uses the same base cols."""
        assemble_node_features(mock_data, feature_keys=[3], inplace=True)
        shape_after_first = mock_data.x.shape

        assemble_node_features(mock_data, feature_keys=[3, 5], inplace=True)
        # Should be base(3) + feat_3(1) + feat_5(1) = 5, not base + feat accumulated
        assert mock_data.x.shape == (shape_after_first[0], 3 + 1 + 1)

    def test_no_scaling_for_key_0(self, mock_data):
        """SkipAtom embeddings (key 0) are passed through without scaling."""
        import torch as _torch
        orig_feat = mock_data.feat_skipatom_200.clone()
        data = assemble_node_features(mock_data, feature_keys=[0], inplace=False)
        assembled_cols = data.x[:, 3:]  # drop base 3 cols
        assert _torch.allclose(assembled_cols.float(), orig_feat.float())

    def test_missing_feature_raises(self, mock_data):
        delattr(mock_data, "feat_atomic_be")
        with pytest.raises(ValueError, match="not found on Data object"):
            assemble_node_features(mock_data, feature_keys=[3])


# ─────────────────────────────────────────────────────────────────────────────
# assemble_dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestAssembleDataset:
    def test_returns_same_list(self, mock_data):
        from conftest import make_mock_data
        data_list = [make_mock_data(), make_mock_data()]
        result = assemble_dataset(data_list, feature_keys=[3])
        assert result is data_list

    def test_all_items_modified(self, mock_data):
        from conftest import make_mock_data
        data_list = [make_mock_data() for _ in range(3)]
        assemble_dataset(data_list, feature_keys=[3])
        for d in data_list:
            assert d.x.shape[1] == 3 + 1
