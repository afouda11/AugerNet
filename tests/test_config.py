"""
Tests for augernet.config

Tier A — AugerNetConfig dataclass defaults and field validation
          (config.py imports only stdlib + yaml; no torch required here)

Tier C — AugerNetConfig.resolve() and load_config() call into
          feature_assembly which imports torch; those tests are skipped
          when torch is absent.
"""

import os
import textwrap
import pytest

from augernet.config import AugerNetConfig, load_config


# ─────────────────────────────────────────────────────────────────────────────
# AugerNetConfig — dataclass defaults (no torch required)
# ─────────────────────────────────────────────────────────────────────────────

class TestAugerNetConfigDefaults:
    def test_default_mode(self):
        cfg = AugerNetConfig()
        assert cfg.mode == "train"

    def test_default_model(self):
        cfg = AugerNetConfig()
        assert cfg.model == "cebe-gnn"

    def test_default_layer_type(self):
        cfg = AugerNetConfig()
        assert cfg.layer_type == "EQ"

    def test_default_hidden_channels(self):
        cfg = AugerNetConfig()
        assert cfg.hidden_channels == 64

    def test_default_n_layers(self):
        cfg = AugerNetConfig()
        assert cfg.n_layers == 3

    def test_default_n_folds(self):
        cfg = AugerNetConfig()
        assert cfg.n_folds == 5

    def test_default_feature_keys(self):
        cfg = AugerNetConfig()
        assert cfg.feature_keys == "035"

    def test_default_random_seed(self):
        cfg = AugerNetConfig()
        assert cfg.random_seed == 42

    def test_default_split_method(self):
        cfg = AugerNetConfig()
        assert cfg.split_method == "random"

    def test_override_fields(self):
        cfg = AugerNetConfig(hidden_channels=128, n_layers=4, layer_type="IN")
        assert cfg.hidden_channels == 128
        assert cfg.n_layers == 4
        assert cfg.layer_type == "IN"

    def test_to_dict_returns_dict(self):
        cfg = AugerNetConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "mode" in d
        assert "hidden_channels" in d


# ─────────────────────────────────────────────────────────────────────────────
# AugerNetConfig.resolve() — requires torch (deferred import inside method)
# ─────────────────────────────────────────────────────────────────────────────

class TestAugerNetConfigResolve:
    @pytest.fixture(autouse=True)
    def _require_torch(self):
        pytest.importorskip("torch")

    def test_model_id_default(self):
        """Default config → cebe_035_random_EQ3_h64"""
        cfg = AugerNetConfig()
        cfg.resolve()
        assert cfg.model_id == "cebe_035_random_EQ3_h64"

    def test_model_id_custom(self):
        cfg = AugerNetConfig(
            feature_keys="035",
            split_method="stratified",
            layer_type="IN",
            n_layers=4,
            hidden_channels=128,
        )
        cfg.resolve()
        assert cfg.model_id == "cebe_035_stratified_IN4_h128"

    def test_feature_keys_normalised(self):
        """'530' (unsorted) should be normalised to '035'."""
        cfg = AugerNetConfig(feature_keys="530")
        cfg.resolve()
        assert cfg.feature_keys == "035"

    def test_feature_tag_set(self):
        cfg = AugerNetConfig(feature_keys="035")
        cfg.resolve()
        assert cfg.feature_tag == "035"

    def test_feature_keys_parsed(self):
        cfg = AugerNetConfig(feature_keys="035")
        cfg.resolve()
        assert cfg.feature_keys_parsed == [0, 3, 5]

    def test_output_dirs_are_strings(self):
        cfg = AugerNetConfig()
        cfg.resolve()
        assert isinstance(cfg.cv_dir, str)
        assert isinstance(cfg.train_dir, str)
        assert isinstance(cfg.evaluate_dir, str)

    def test_norm_stats_file_set_for_cebe(self):
        cfg = AugerNetConfig(model="cebe-gnn")
        cfg.resolve()
        assert cfg.norm_stats_file != ""
        assert "cebe_norm_stats.pt" in cfg.norm_stats_file


# ─────────────────────────────────────────────────────────────────────────────
# load_config — YAML parsing and validation
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadConfig:
    def _write_yaml(self, tmp_path, content: str) -> str:
        """Write a YAML string to a temp file and return its path."""
        p = tmp_path / "cfg.yml"
        p.write_text(textwrap.dedent(content))
        return str(p)

    def test_unknown_key_raises_before_torch(self, tmp_path):
        """
        The unknown-key check fires before resolve(), so this test does NOT
        require torch.  It verifies strict config validation.
        """
        yaml_path = self._write_yaml(tmp_path, """
            mode: train
            totally_unknown_field: 99
        """)
        with pytest.raises(ValueError, match="Unknown config fields"):
            load_config(yaml_path)

    def test_multiple_unknown_keys_listed(self, tmp_path):
        yaml_path = self._write_yaml(tmp_path, """
            bad_key_a: 1
            bad_key_b: 2
        """)
        with pytest.raises(ValueError) as exc_info:
            load_config(yaml_path)
        msg = str(exc_info.value)
        assert "bad_key_a" in msg
        assert "bad_key_b" in msg

    @pytest.mark.requires_torch
    def test_valid_config_loads(self, tmp_path):
        pytest.importorskip("torch")
        yaml_path = self._write_yaml(tmp_path, """
            mode: train
            layer_type: IN
            hidden_channels: 32
            n_layers: 2
            feature_keys: '035'
        """)
        cfg = load_config(yaml_path)
        assert cfg.mode == "train"
        assert cfg.hidden_channels == 32
        assert cfg.layer_type == "IN"

    @pytest.mark.requires_torch
    def test_empty_yaml_gives_defaults(self, tmp_path):
        pytest.importorskip("torch")
        yaml_path = self._write_yaml(tmp_path, "")
        cfg = load_config(yaml_path)
        assert cfg.mode == "train"
        assert cfg.hidden_channels == 64
