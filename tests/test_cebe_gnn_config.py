"""
CEBE-GNN config tests.

Tests AugerNetConfig defaults, resolve(), load_config(), and model_id
generation specific to the cebe-gnn model type.
"""

import os
import textwrap
import pytest

from augernet.config import AugerNetConfig, load_config


# -- Defaults (no torch required) --------------------------------------------

class TestCebeGnnConfigDefaults:

    def test_override_fields(self):
        cfg = AugerNetConfig(hidden_channels=128, n_layers=4, layer_type="IN")
        assert cfg.hidden_channels == 128
        assert cfg.n_layers == 4
        assert cfg.layer_type == "IN"

    def test_to_dict(self):
        d = AugerNetConfig().to_dict()
        assert isinstance(d, dict)
        assert "model" in d and "hidden_channels" in d


# -- resolve() (requires torch) ----------------------------------------------

class TestCebeGnnConfigResolve:

    @pytest.fixture(autouse=True)
    def _require_torch(self):
        pytest.importorskip("torch")

    def test_model_id_default(self):
        cfg = AugerNetConfig()
        cfg.resolve()
        assert cfg.model_id == "cebe_gnn_035_random_EQ3_h64"

    def test_model_id_custom(self):
        cfg = AugerNetConfig(
            feature_keys="035", split_method="butina",
            layer_type="IN", n_layers=4, hidden_channels=128,
        )
        cfg.resolve()
        assert cfg.model_id == "cebe_gnn_035_butina_IN4_h128"

    def test_feature_keys_normalised(self):
        cfg = AugerNetConfig(feature_keys="530")
        cfg.resolve()
        assert cfg.feature_keys == "035"

    def test_feature_keys_parsed(self):
        cfg = AugerNetConfig(feature_keys="035")
        cfg.resolve()
        assert cfg.feature_keys_parsed == [0, 3, 5]

    def test_norm_stats_file_set(self):
        cfg = AugerNetConfig(model="cebe-gnn")
        cfg.resolve()
        assert "cebe_norm_stats.pt" in cfg.norm_stats_file


# -- load_config (YAML validation) -------------------------------------------

class TestCebeGnnLoadConfig:

    def _write_yaml(self, tmp_path, content):
        p = tmp_path / "cfg.yml"
        p.write_text(textwrap.dedent(content))
        return str(p)

    def test_unknown_key_raises(self, tmp_path):
        path = self._write_yaml(tmp_path, """
            mode: train
            totally_unknown_field: 99
        """)
        with pytest.raises(ValueError, match="Unknown config fields"):
            load_config(path)

    def test_multiple_unknown_keys_listed(self, tmp_path):
        path = self._write_yaml(tmp_path, """
            bad_key_a: 1
            bad_key_b: 2
        """)
        with pytest.raises(ValueError) as exc_info:
            load_config(path)
        msg = str(exc_info.value)
        assert "bad_key_a" in msg and "bad_key_b" in msg

    def test_valid_train_config(self, tmp_path):
        pytest.importorskip("torch")
        path = self._write_yaml(tmp_path, """
            model: cebe-gnn
            mode: train
            feature_keys: '035'
            layer_type: EQ
            hidden_channels: 64
            n_layers: 3
        """)
        cfg = load_config(path)
        assert cfg.model == "cebe-gnn"
        assert cfg.layer_type == "EQ"
        assert cfg.feature_keys_parsed == [0, 3, 5]

    def test_predict_config_fields(self, tmp_path):
        pytest.importorskip("torch")
        path = self._write_yaml(tmp_path, """
            model: cebe-gnn
            mode: predict
            model_path: models/my_model.pth
            predict_dir: my_xyz/
            feature_keys: '035'
            layer_type: EQ
            hidden_channels: 64
            n_layers: 3
        """)
        cfg = load_config(path)
        assert cfg.mode == "predict"
        assert cfg.predict_dir == "my_xyz/"
