"""
AugerNet Configuration System
=============================

Loads training / evaluation configuration from YAML files and provides
a single ``AugerNetConfig`` dataclass consumed by ``train_driver.py``.

Usage
-----
    from augernet.config import load_config

    cfg = load_config('configs/cebe_default.yml')
"""

from __future__ import annotations

import importlib.resources
import sys
import os
import copy
import yaml
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AugerNetConfig:
    """Complete configuration for a single AugerNet run."""

    # Model type and run mode 
    model: str = 'cebe-gnn'          # 'cebe-gnn' only atm
    mode: str = 'train'              # cv | train | param | evaluate_cv | evaluate_train | evaluate_param

    # Evaluation on exp.evaluation data 
    run_evaluation: bool = True
    # Sanity check permutation invariance & rotational invariance/equivariance
    run_unit_tests: bool = True

    # k-fold 
    n_folds: int = 5
    train_fold: int = 3
    split_method: str = 'random'     # stratified | random | umap | size

    # data paths 
    data_path: str = ''              # base data directory (resolved at runtime)
    exp_dir: str = ''                # experimental data directory

    # node features 
    feature_keys: List[int] = field(default_factory=lambda: [0, 3, 5])
    feature_scale: str = 'MEANSTD'  # MEANSTD | NORM | NONE

    # output scaling
    out_scale: str = 'MEANSTD'       # NONE | FEATURE_SCALE | MEANSTD
    norm_stats_file: str = ''

    # GNN hyper-parameters
    layer_type: str = 'EQ'           # EQ (equivariant) | IN (invariant)
    hidden_channels: int = 64
    n_layers: int = 3
    num_epochs: int = 500
    patience: int = 50
    batch_size: int = 24
    learning_rate: float = 0.001
    random_seed: int = 42

    # optimizer
    optimizer_type: str = 'adamw'
    weight_decay: float = 5e-4
    gradient_clip_norm: float = 0.5
    warmup_epochs: int = 10
    min_lr: float = 1e-7

    # scheduler
    scheduler_type: str = 'cosine'   # cosine | onecycle
    pct_start: float = 0.3           # OneCycleLR only

    # param search
    param_grid: Dict[str, List[Any]] = field(default_factory=dict)

    # directories (auto-computed)
    script_dir: str = ''             # directory of the backend script
    project_root: str = ''           # repository root
    cv_dir: str = ''
    train_dir: str = ''
    param_dir: str = ''
    split_dir: str = ''
    split_file: str = ''

    # ── computed (populated by resolve()) ───────────────────────────────
    feature_tag: str = ''            # pure feature identity: e.g. '035'
    model_tag: str = ''              # full filename label: e.g. '035_random_EQ_3'

    # ─────────────────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def resolve(self, project_root: str | None = None) -> 'AugerNetConfig':
        """Fill in computed / derived fields after loading."""
        from augernet.feature_assembly import compute_feature_tag

        if project_root:
            self.project_root = project_root

        # ── script_dir  (where the backend lives) ───────────────────────
        _backend_dirs = {
            'cebe-gnn':  os.path.join(self.project_root, 'cebe_pred'),
        }
        self.script_dir = _backend_dirs.get(self.model, '')

        # ── data_path default ───────────────────────────────────────────
        if not self.data_path:
            self.data_path = os.path.join(self.project_root, 'data')

        # ── exp_dir (CEBE) ──────────────────────────────────────────────
        if self.model == 'cebe-gnn' and not self.exp_dir:
            self.exp_dir = os.path.join(self.data_path, 'raw', 'cebe_eval')

        # ── CNN data paths ──────────────────────────────────────────────
        if self.model == 'auger-cnn':
            proc = os.path.join(self.data_path, 'processed')
            if not self.train_data:
                self.train_data = os.path.join(proc, 'cnn_auger_calc_carbon.pkl')
            if not self.eval_data:
                self.eval_data = os.path.join(proc, 'cnn_auger_eval_carbon.pkl')
            if not self.exp_data:
                self.exp_data = os.path.join(proc, 'cnn_auger_exp_carbon.pkl')

        # ── norm_stats_file (CEBE) ──────────────────────────────────────
        if self.model == 'cebe-gnn' and not self.norm_stats_file:
            self.norm_stats_file = os.path.join(
                self.script_dir, 'cebe_normalization_stats.pt'
            )

        # ── output dirs ─────────────────────────────────────────────────
        # By default results are written relative to the working directory so that
        # the user controls where outputs land by cd-ing into the right
        # place before running the CLI.  Evaluation modes will then find
        # the saved models in the same directory.
        # However this can be overwritten by specifying output directories in the CLI
        # By including --cv_dir '/path/to/cv_results' etc. in python augernet
        cwd = os.getcwd()
        if not self.cv_dir:
            self.cv_dir = os.path.join(cwd, 'cv_results')
        if not self.train_dir:
            self.train_dir = os.path.join(cwd, 'train_results')
        if not self.param_dir:
            self.param_dir = os.path.join(cwd, 'param_results')

        # ── feature_tag + model_tag (GNN models) ──────────────────────────
        # feature_tag  = pure feature identity, e.g. '035'
        # model_tag    = full filename label used for save/load, including
        #                split_method (for cv/evaluate_cv),
        #                layer_type, n_layers, fwhm, etc.
        #
        # Both training and evaluation modes produce the SAME model_tag so
        # that load_saved_model finds exactly the file train/cv saved.
        _cv_modes = ('cv', 'evaluate_cv')

        if self.model in ('cebe-gnn', 'auger-gnn'):
            self.feature_tag = compute_feature_tag(self.feature_keys)

            # Start building model_tag from pure feature_tag
            parts = [self.feature_tag]

            # Split method is part of the tag for CV-related modes
            if self.mode in _cv_modes:
                parts.append(self.split_method)

            # Auger GNN: always append FWHM
            if self.model == 'auger-gnn':
                parts.append(f"fwhm{str(self.eval_fwhm).replace('.', 'pt')}")

            self.model_tag = '_'.join(parts)

        # ── cv_suffix (CNN) ─────────────────────────────────────────────
        if self.model == 'auger-cnn':
            parts = []
            if self.mode in _cv_modes:
                parts.append(self.split_method)
            _fwhm_str = f"fwhm{str(self.broadening_fwhm).replace('.', 'pt')}"
            if self.merge_scheme != 'none':
                parts.append(self.merge_scheme)
            parts.append(_fwhm_str)
            self.cv_suffix = '_' + '_'.join(parts) if parts else ''
            self.model_tag = self.cv_suffix  # CNN uses cv_suffix as its tag

        # ── split file ──────────────────────────────────────────────────
        if self.model == 'auger-cnn':
            if not self.split_dir:
                self.split_dir = os.path.join(self.data_path, 'processed', 'splits')
            if not self.split_file:
                self.split_file = os.path.join(
                    self.split_dir,
                    f'auger_cnn_splits_{self.split_method}.json',
                )
        elif self.model in ('cebe-gnn', 'auger-gnn'):
            if not self.split_dir:
                self.split_dir = os.path.join(self.data_path, 'splits')

        return self


# ─────────────────────────────────────────────────────────────────────────────
#  Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> AugerNetConfig:

    """
    Load an ``AugerNetConfig`` from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to a YAML file.

    Returns
    -------
    AugerNetConfig  (already resolved)
    """
    config_path = os.path.abspath(config_path)

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    # Known dataclass field names
    known = {f.name for f in AugerNetConfig.__dataclass_fields__.values()}

    # Smart flattening: only flatten nested dicts whose *key* is NOT a known
    # dataclass field (e.g. grouping headers in YAML).  Known dict-typed fields
    # like 'architecture' and 'param_grid' are kept as-is.
    flat: Dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, dict) and k not in known:
            # This is a grouping header — flatten its children
            flat.update(v)
        else:
            flat[k] = v

    # Build dataclass — ignore unknown keys gracefully
    known = {f.name for f in AugerNetConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in flat.items() if k in known}

    cfg = AugerNetConfig(**filtered)

    # Resolve project root from the config file's location
    # Walk up until we find setup.py or augernet/
    project_root = _find_project_root()
    cfg.resolve(project_root)

    return cfg

def _find_project_root() -> str:
    """Get the augernet package root directory."""
    if sys.version_info >= (3, 9):
        # Python 3.9+: use importlib.resources
        try:
            return str(importlib.resources.files('augernet').parent)
        except (ImportError, AttributeError):
            pass
    
    # Fallback: use __file__
    import augernet
    return os.path.dirname(os.path.dirname(os.path.abspath(augernet.__file__)))