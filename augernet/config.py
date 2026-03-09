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

from augernet import PROJECT_ROOT, DATA_RAW_DIR, DATA_PROCESSED_DIR

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

    def resolve(self) -> 'AugerNetConfig':

        """Fill in computed / derived fields after loading."""
        from augernet.feature_assembly import compute_feature_tag

        # ── exp_dir (CEBE) ──────────────────────────────────────────────
        if self.model == 'cebe-gnn' and not self.exp_dir:
            self.exp_dir = os.path.join(DATA_RAW_DIR, 'exp_cebe')

        # ── norm_stats_file (CEBE) ──────────────────────────────────────
        if self.model == 'cebe-gnn' and not self.norm_stats_file:
            self.norm_stats_file = os.path.join(
                DATA_PROCESSED_DIR, 'cebe_norm_stats.pt'
            )

        # ── output dirs ─────────────────────────────────────────────────
        # By default results are written relative to the working directory so that
        # the user controls where outputs land by cd-ing into the right
        # place before running the CLI.  Evaluation modes will then find
        # the saved models in the same directory.
        cwd = os.getcwd()
        self.cv_dir = os.path.join(cwd, 'cv_results')
        self.train_dir = os.path.join(cwd, 'train_results')
        self.param_dir = os.path.join(cwd, 'param_results')

        # ── feature_tag + model_tag (GNN models) ──────────────────────────
        # feature_tag  = pure feature identity, e.g. '035'
        # model_tag    = full filename label used for save/load, including
        #                split_method (for cv/evaluate_cv),
        #                layer_type, n_layers, fwhm, etc.
        #
        # Both training and evaluation modes produce the SAME model_tag so
        # that load_saved_model finds exactly the file train/cv saved.

        if self.model == 'cebe-gnn':
            self.feature_tag = compute_feature_tag(self.feature_keys)

            # Start building model_tag from pure feature_tag
            parts = [self.feature_tag]

            # Split method is part of the tag for CV-related modes
            parts.append(self.split_method)

            self.model_tag = '_'.join(parts)

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

     # Strict mode: reject unknown keys
    unknown = set(raw.keys()) - known
    if unknown:
        raise ValueError(
            f"Unknown config fields in {config_path}:\n"
            f"  {', '.join(sorted(unknown))}\n"
            f"Allowed fields: {', '.join(sorted(known))}"
        )

    cfg = AugerNetConfig(**raw)

    # Resolve project root from the config file's location
    # Walk up until we find setup.py or augernet/
    cfg.resolve()

    return cfg