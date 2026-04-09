"""
CEBE-GNN config tests.

Tests AugerNetConfig defaults, resolve(), load_config(), model_id
generation, and filename construction for all run modes (train, cv,
param, evaluate, predict).
"""

import os
import textwrap
import pytest

from augernet.config import AugerNetConfig, load_config
from augernet.train_driver import (
    _build_save_paths,
    _param_search_id,
    _infer_fold_from_path,
)


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

    def test_model_id_default(self):
        cfg = AugerNetConfig()
        cfg.resolve()
        assert cfg.model_id == "cebe_gnn_035_random5_EQ3_h64"

    def test_model_id_custom(self):
        cfg = AugerNetConfig(
            feature_keys="035", split_method="butina",
            layer_type="IN", n_layers=4, hidden_channels=128,
        )
        cfg.resolve()
        assert cfg.model_id == "cebe_gnn_035_butina5_IN4_h128"

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


# -- resolve() param mode (requires torch) ------------------------------------

class TestCebeGnnParamModeResolve:
    """Tests for mode='param' — verifies result_dir, model_id, and
    sub-directory creation match the param search convention."""

    def test_param_result_dir_created(self):
        cfg = AugerNetConfig(model="cebe-gnn", mode="param")
        cfg.resolve()
        assert cfg.result_dir.endswith("cebe_gnn_param_results")
        assert os.path.isdir(cfg.result_dir)

    def test_param_models_dir_created(self):
        cfg = AugerNetConfig(model="cebe-gnn", mode="param")
        cfg.resolve()
        assert os.path.isdir(cfg.models_dir)
        assert cfg.models_dir.endswith("models")

    def test_param_outputs_dir_created(self):
        cfg = AugerNetConfig(model="cebe-gnn", mode="param")
        cfg.resolve()
        assert os.path.isdir(cfg.outputs_dir)
        assert cfg.outputs_dir.endswith("outputs")

    def test_param_pngs_dir_created(self):
        cfg = AugerNetConfig(model="cebe-gnn", mode="param")
        cfg.resolve()
        assert os.path.isdir(cfg.pngs_dir)
        assert cfg.pngs_dir.endswith("pngs")

    def test_param_model_id_same_as_train(self):
        """mode='param' builds the same model_id as train/cv —
        the prefix and config_id are added at runtime by train_driver."""
        cfg_param = AugerNetConfig(model="cebe-gnn", mode="param")
        cfg_param.resolve()
        cfg_train = AugerNetConfig(model="cebe-gnn", mode="train")
        cfg_train.resolve()
        assert cfg_param.model_id == cfg_train.model_id

    def test_param_save_paths_with_grid(self):
        """End-to-end: resolve param config, build prefix from grid,
        then verify _build_save_paths produces the correct filename."""
        cfg = AugerNetConfig(model="cebe-gnn", mode="param")
        cfg.resolve()
        grid = {'layer_type': ['EQ', 'IN'], 'hidden_channels': [32, 64, 128]}
        prefix = _param_search_id(grid)
        paths = _build_save_paths(
            cfg, fold=2, save_dir=cfg.models_dir,
            prefix=prefix, config_id="cfg003",
        )
        fname = os.path.basename(paths['model'])
        assert fname.startswith("search_hidden_channels3_layer_type2_")
        assert "_fold2_" in fname
        assert fname.endswith("_cfg003.pth")
        assert cfg.model_id in fname


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

    def test_param_grid_valid_keys_accepted(self, tmp_path):
        path = self._write_yaml(tmp_path, """
            model: cebe-gnn
            mode: param
            param_grid:
              learning_rate: [0.001, 0.0001]
              n_layers: [2, 3, 4]
        """)
        cfg = load_config(path)
        assert cfg.param_grid == {
            'learning_rate': [0.001, 0.0001],
            'n_layers': [2, 3, 4],
        }

    def test_param_grid_invalid_key_raises(self, tmp_path):
        path = self._write_yaml(tmp_path, """
            model: cebe-gnn
            mode: param
            param_grid:
              learning_rate: [0.001]
              nonexistent_param: [1, 2]
        """)
        with pytest.raises(ValueError, match="non-overridable"):
            load_config(path)

    def test_param_grid_all_overridable_fields(self, tmp_path):
        """Every key in OVERRIDABLE_FIELDS should be accepted."""
        from augernet.config import OVERRIDABLE_FIELDS

        # Build a grid with every overridable key (single-value lists)
        grid_lines = []
        for key in sorted(OVERRIDABLE_FIELDS):
            cfg_default = AugerNetConfig()
            val = getattr(cfg_default, key, None)
            if val is None or isinstance(val, dict):
                continue  # skip dict fields for simplicity
            grid_lines.append(f"  {key}: [{val!r}]")
        yaml_str = (
            "model: cebe-gnn\n"
            "mode: param\n"
            "param_grid:\n"
            + "\n".join(grid_lines)
        )
        path = tmp_path / "cfg.yml"
        path.write_text(yaml_str)
        cfg = load_config(str(path))
        assert cfg.param_grid


# ─────────────────────────────────────────────────────────────────────────────
#  OVERRIDABLE_FIELDS consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestOverridableFields:
    """Ensure OVERRIDABLE_FIELDS is consistent with the dataclass."""

    def test_all_overridable_are_dataclass_fields(self):
        from augernet.config import OVERRIDABLE_FIELDS
        dc_fields = {f.name for f in AugerNetConfig.__dataclass_fields__.values()}
        bad = OVERRIDABLE_FIELDS - dc_fields
        assert bad == set(), f"OVERRIDABLE_FIELDS contains non-dataclass names: {bad}"

    def test_non_overridable_excluded(self):
        """Fields that should never be in param_grid."""
        from augernet.config import OVERRIDABLE_FIELDS
        forbidden = {'model', 'mode', 'result_dir', 'models_dir',
                     'outputs_dir', 'pngs_dir', 'model_id',
                     'feature_keys_parsed', 'model_path', 'predict_dir'}
        overlap = OVERRIDABLE_FIELDS & forbidden
        assert overlap == set(), f"Should not be overridable: {overlap}"


# ─────────────────────────────────────────────────────────────────────────────
#  Filename construction — _build_save_paths
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildSavePaths:
    """Test that _build_save_paths produces correct .pth filenames
    for train, cv, and param search modes with various folds and configs."""

    @pytest.fixture()
    def cebe_cfg(self):
        cfg = AugerNetConfig(model="cebe-gnn", mode="train")
        cfg.resolve()
        return cfg

    # ── Train mode (single fold) ─────────────────────────────────────────

    def test_train_fold1(self, cebe_cfg):
        paths = _build_save_paths(cebe_cfg, fold=1, save_dir="/tmp/models")
        assert paths == {
            'model': "/tmp/models/cebe_gnn_035_random5_EQ3_h64_fold1.pth",
        }

    def test_train_fold3(self, cebe_cfg):
        paths = _build_save_paths(cebe_cfg, fold=3, save_dir="/tmp/models")
        assert paths == {
            'model': "/tmp/models/cebe_gnn_035_random5_EQ3_h64_fold3.pth",
        }

    def test_train_fold5(self, cebe_cfg):
        paths = _build_save_paths(cebe_cfg, fold=5, save_dir="/tmp/models")
        assert paths == {
            'model': "/tmp/models/cebe_gnn_035_random5_EQ3_h64_fold5.pth",
        }

    # ── CV mode (multiple folds, same convention) ────────────────────────

    def test_cv_all_folds(self, cebe_cfg):
        """CV calls _build_save_paths once per fold — verify all 5."""
        for fold in range(1, 6):
            paths = _build_save_paths(cebe_cfg, fold=fold,
                                      save_dir="/tmp/cv_models")
            expected = (
                f"/tmp/cv_models/cebe_gnn_035_random5_EQ3_h64_fold{fold}.pth"
            )
            assert paths == {'model': expected}

    def test_cv_custom_architecture(self):
        cfg = AugerNetConfig(
            model="cebe-gnn", mode="cv",
            feature_keys="035", split_method="butina",
            layer_type="IN", n_layers=4, hidden_channels=128,
        )
        cfg.resolve()
        for fold in range(1, 6):
            paths = _build_save_paths(cfg, fold=fold, save_dir="/out/m")
            expected = f"/out/m/cebe_gnn_035_butina5_IN4_h128_fold{fold}.pth"
            assert paths == {'model': expected}

    # ── Param search (prefix + config_id) ────────────────────────────────

    def test_param_search_single_config(self, cebe_cfg):
        paths = _build_save_paths(
            cebe_cfg, fold=3, save_dir="/tmp/models",
            prefix="search_layer_type2", config_id="cfg000",
        )
        assert paths == {
            'model': (
                "/tmp/models/"
                "search_layer_type2_cebe_gnn_035_random5_EQ3_h64_fold3_cfg000.pth"
            ),
        }

    def test_param_search_multiple_configs(self, cebe_cfg):
        """Each config_id yields a unique filename — no overwrite."""
        prefix = "search_hidden_channels3_n_layers2"
        seen = set()
        for i in range(6):
            paths = _build_save_paths(
                cebe_cfg, fold=1, save_dir="/m",
                prefix=prefix, config_id=f"cfg{i:03d}",
            )
            fname = os.path.basename(paths['model'])
            assert fname not in seen, f"Duplicate filename: {fname}"
            seen.add(fname)
            assert f"cfg{i:03d}" in fname
            assert fname.startswith("search_hidden_channels3_n_layers2_")

    def test_param_search_different_folds(self, cebe_cfg):
        """Same config on different folds to different filenames."""
        p1 = _build_save_paths(cebe_cfg, fold=1, save_dir="/m",
                               prefix="s", config_id="cfg001")
        p3 = _build_save_paths(cebe_cfg, fold=3, save_dir="/m",
                               prefix="s", config_id="cfg001")
        assert p1['model'] != p3['model']
        assert "_fold1_" in p1['model']
        assert "_fold3_" in p3['model']

    # ── Only prefix or only config_id ────────────────────────────────────

    def test_prefix_only_no_config_id(self, cebe_cfg):
        paths = _build_save_paths(cebe_cfg, fold=2, save_dir="/m",
                                  prefix="search_x2")
        fname = os.path.basename(paths['model'])
        assert fname.startswith("search_x2_")
        assert "cfg" not in fname
        assert fname.endswith("_fold2.pth")

    def test_config_id_only_no_prefix(self, cebe_cfg):
        paths = _build_save_paths(cebe_cfg, fold=2, save_dir="/m",
                                  config_id="cfg007")
        fname = os.path.basename(paths['model'])
        assert fname.endswith("_fold2_cfg007.pth")
        assert not fname.startswith("search")

    # ── Keys returned ────────────────────────────────────────────────────

    def test_cebe_returns_model_key(self, cebe_cfg):
        paths = _build_save_paths(cebe_cfg, fold=1, save_dir="/m")
        assert set(paths.keys()) == {'model'}

    # ── save_dir is part of the path ─────────────────────────────────────

    def test_save_dir_is_parent(self, cebe_cfg):
        paths = _build_save_paths(cebe_cfg, fold=1,
                                  save_dir="/a/b/c/models")
        assert os.path.dirname(paths['model']) == "/a/b/c/models"


# ─────────────────────────────────────────────────────────────────────────────
#  Param search ID builder
# ─────────────────────────────────────────────────────────────────────────────

class TestParamSearchId:

    def test_single_param(self):
        grid = {'layer_type': ['EQ', 'IN']}
        assert _param_search_id(grid) == "search_layer_type2"

    def test_two_params_sorted(self):
        grid = {'n_layers': [2, 3, 4], 'layer_type': ['EQ', 'IN']}
        assert _param_search_id(grid) == "search_layer_type2_n_layers3"

    def test_three_params(self):
        grid = {
            'hidden_channels': [32, 64, 128],
            'n_layers': [2, 3],
            'learning_rate': [1e-3, 1e-4],
        }
        sid = _param_search_id(grid)
        assert sid == "search_hidden_channels3_learning_rate2_n_layers2"

    def test_single_value_param(self):
        grid = {'dropout': [0.1]}
        assert _param_search_id(grid) == "search_dropout1"


# ─────────────────────────────────────────────────────────────────────────────
#  Fold inference from user-supplied model paths (evaluate / predict)
# ─────────────────────────────────────────────────────────────────────────────

class TestInferFoldFromPath:

    def test_standard_train_path(self):
        assert _infer_fold_from_path(
            "models/cebe_gnn_035_random5_EQ3_h64_fold3.pth"
        ) == 3

    def test_fold1(self):
        assert _infer_fold_from_path("some_model_fold1.pth") == 1

    def test_fold10(self):
        assert _infer_fold_from_path("model_fold10.pth") == 10

    def test_param_search_path(self):
        path = (
            "models/search_layer_type2_"
            "cebe_gnn_035_random5_EQ3_h64_fold3_cfg002.pth"
        )
        assert _infer_fold_from_path(path) == 3

    def test_no_fold_returns_none(self):
        assert _infer_fold_from_path("my_custom_model.pth") is None

    def test_user_custom_name_with_fold(self):
        assert _infer_fold_from_path(
            "/abs/path/to/my_experiment_fold7.pth"
        ) == 7

    def test_absolute_path(self):
        assert _infer_fold_from_path(
            "/home/user/results/models/cebe_gnn_035_random5_EQ3_h64_fold5.pth"
        ) == 5


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluate & predict mode — model_id derived from user model_path
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluatePredictModelId:
    """When mode is evaluate or predict the user supplies model_path in
    the YAML, and config.resolve() derives model_id from the filename stem."""

    def test_evaluate_model_id_from_train(self):
        cfg = AugerNetConfig(
            model="cebe-gnn", mode="evaluate",
            model_path="models/cebe_gnn_035_random5_EQ3_h64_fold3.pth",
        )
        cfg.resolve()
        assert cfg.model_id == "cebe_gnn_035_random5_EQ3_h64_fold3"

    def test_evaluate_model_id_from_param_search(self):
        cfg = AugerNetConfig(
            model="cebe-gnn", mode="evaluate",
            model_path="models/search_layer_type2_cebe_gnn_035_random5_EQ3_h64_fold1_cfg005.pth",
        )
        cfg.resolve()
        assert cfg.model_id == (
            "search_layer_type2_cebe_gnn_035_random5_EQ3_h64_fold1_cfg005"
        )

    def test_predict_model_id_from_custom_name(self):
        cfg = AugerNetConfig(
            model="cebe-gnn", mode="predict",
            model_path="my_models/best_cebe_fold2.pth",
        )
        cfg.resolve()
        assert cfg.model_id == "best_cebe_fold2"

    def test_evaluate_model_id_from_butina(self):
        cfg = AugerNetConfig(
            model="cebe-gnn", mode="evaluate",
            model_path="results/cebe_gnn_035_butina5_IN4_h128_fold1.pth",
        )
        cfg.resolve()
        assert cfg.model_id == "cebe_gnn_035_butina5_IN4_h128_fold1"

    def test_evaluate_model_id_absolute_path(self):
        cfg = AugerNetConfig(
            model="cebe-gnn", mode="evaluate",
            model_path="/home/user/saved/my_great_model_fold5.pth",
        )
        cfg.resolve()
        assert cfg.model_id == "my_great_model_fold5"

    def test_predict_model_id_no_fold(self):
        """User may supply a model file with no fold in the name."""
        cfg = AugerNetConfig(
            model="cebe-gnn", mode="predict",
            model_path="final_model.pth",
        )
        cfg.resolve()
        assert cfg.model_id == "final_model"


# ─────────────────────────────────────────────────────────────────────────────
#  Save and load path consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveLoadConsistency:
    """Verify that the filenames produced by _build_save_paths (used at save
    time) match the expected naming convention so that loading after training
    always finds the file."""

    @pytest.fixture()
    def cebe_cfg(self):
        cfg = AugerNetConfig(model="cebe-gnn", mode="cv")
        cfg.resolve()
        return cfg

    def test_save_load_match_all_folds(self, cebe_cfg):
        """For every fold, the save path must match the expected naming
        convention so loading after training always finds the file."""

        save_dir = "/tmp/models"

        for fold in range(1, 6):
            # Path that train_driver would pass to train_single_run
            save_paths = _build_save_paths(cebe_cfg, fold=fold,
                                           save_dir=save_dir)

            # Expected path based on naming convention
            model_id = cebe_cfg.model_id
            load_path = os.path.join(
                save_dir, f"{model_id}_fold{fold}.pth"
            )

            assert save_paths['model'] == load_path, (
                f"Fold {fold}: save={save_paths['model']} != load={load_path}"
            )

    def test_save_load_custom_architecture(self):
        cfg = AugerNetConfig(
            model="cebe-gnn", mode="cv",
            split_method="butina", layer_type="IN",
            n_layers=4, hidden_channels=128,
        )
        cfg.resolve()
        save_dir = "/results/models"

        for fold in range(1, 6):
            save_paths = _build_save_paths(cfg, fold=fold,
                                           save_dir=save_dir)
            expected = os.path.join(
                save_dir, f"{cfg.model_id}_fold{fold}.pth"
            )
            assert save_paths['model'] == expected
