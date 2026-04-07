"""
CEBE-GNN feature assembly tests.

Tests feature catalog utilities (parse, tag, describe) and the
assemble_node_features / assemble_dataset pipeline.
"""

import pytest

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


# -- compute_feature_tag ------------------------------------------------------

class TestComputeFeatureTag:

    def test_sorted_output(self):
        assert compute_feature_tag([3, 0, 5]) == "035"

    def test_single_key(self):
        assert compute_feature_tag([7]) == "7"

    def test_empty(self):
        assert compute_feature_tag([]) == ""

    def test_round_trip(self):
        keys = [0, 3, 5]
        assert parse_feature_keys(compute_feature_tag(keys)) == keys


# -- parse_feature_keys -------------------------------------------------------

class TestParseFeatureKeys:

    def test_basic(self):
        assert parse_feature_keys("035") == [0, 3, 5]

    def test_unsorted_is_sorted(self):
        assert parse_feature_keys("530") == [0, 3, 5]

    def test_invalid_key_raises(self):
        with pytest.raises(ValueError, match="Unknown feature key"):
            parse_feature_keys("9")

    def test_empty_string(self):
        assert parse_feature_keys("") == []


# -- describe_features --------------------------------------------------------

class TestDescribeFeatures:

    def test_035(self):
        desc = describe_features([0, 3, 5])
        assert "skipatom_200" in desc
        assert "atomic_be" in desc
        assert "e_score" in desc

    def test_separator(self):
        assert " + " in describe_features([2, 3])

    def test_order_independent(self):
        assert describe_features([0, 3]) == describe_features([3, 0])


# -- _scale_tensor ------------------------------------------------------------

class TestScaleTensor:

    def test_2d_shape_preserved(self):
        t = torch.randn(6, 3)
        assert _scale_tensor(t).shape == t.shape

    def test_1d_becomes_2d(self):
        assert _scale_tensor(torch.randn(6)).shape == (6, 1)

    def test_near_zero_mean(self):
        out = _scale_tensor(torch.randn(100, 4))
        assert torch.allclose(out.mean(dim=0), torch.zeros(4), atol=1e-5)

    def test_constant_column_no_nan(self):
        out = _scale_tensor(torch.ones(5, 2))
        assert torch.isfinite(out).all()


# -- assemble_node_features ---------------------------------------------------

class TestAssembleNodeFeatures:

    def test_single_key_shape(self, mock_data):
        data = assemble_node_features(mock_data, feature_keys=[3], inplace=False)
        assert data.x.shape == (4, 3 + 1)  # base 3 + atomic_be 1

    def test_035_shape(self, mock_data):
        data = assemble_node_features(mock_data, feature_keys=[0, 3, 5], inplace=False)
        assert data.x.shape == (4, 3 + 200 + 1 + 1)

    def test_inplace_false_preserves_original(self, mock_data):
        original_shape = mock_data.x.shape
        assemble_node_features(mock_data, feature_keys=[3], inplace=False)
        assert mock_data.x.shape == original_shape

    def test_inplace_true_modifies(self, mock_data):
        assemble_node_features(mock_data, feature_keys=[3], inplace=True)
        assert mock_data.x.shape == (4, 3 + 1)

    def test_missing_feature_raises(self, mock_data):
        delattr(mock_data, "feat_atomic_be")
        with pytest.raises(ValueError, match="not found on Data object"):
            assemble_node_features(mock_data, feature_keys=[3])

    def test_skipatom_not_scaled(self, mock_data):
        orig = mock_data.feat_skipatom_200.clone()
        data = assemble_node_features(mock_data, feature_keys=[0], inplace=False)
        assembled = data.x[:, 3:]
        assert torch.allclose(assembled.float(), orig.float())


# -- assemble_dataset ---------------------------------------------------------

class TestAssembleDataset:

    def test_returns_same_list(self):
        from conftest import make_mock_data
        data_list = [make_mock_data(), make_mock_data()]
        result = assemble_dataset(data_list, feature_keys=[3])
        assert result is data_list

    def test_all_items_modified(self):
        from conftest import make_mock_data
        data_list = [make_mock_data() for _ in range(3)]
        assemble_dataset(data_list, feature_keys=[3])
        for d in data_list:
            assert d.x.shape[1] == 3 + 1
