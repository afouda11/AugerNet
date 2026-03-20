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

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Union

from augernet import PROJECT_ROOT, DATA_PROCESSED_DIR

# ─────────────────────────────────────────────────────────────────────────────
#  Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AugerNetConfig:
    """Complete configuration for a single AugerNet run."""

    # Model type and run mode 
    model: str = 'cebe-gnn'          # 'cebe-gnn' only atm
    mode: str = 'train'              # cv | train | param | evaluate | predict

    # Evaluation on exp.evaluation data 
    run_evaluation: bool = True
    # experimental data split
    exp_split: str = 'both'           # all | val | eval | both
    # Sanity check permutation invariance & rotational invariance/equivariance
    run_unit_tests: bool = False

    # k-fold 
    n_folds: int = 5
    train_fold: int = 3
    split_method: str = 'random'     # random | butina

    # data paths 
    data_path: str = ''              # base data directory (resolved at runtime)

    # node features 
    feature_keys: str = '035'        # compact string: '035' → keys [0,3,5]
    norm_stats_file: str = ''

    # GNN hyper-parameters
    layer_type: str = 'EQ'           # EQ (equivariant) | IN (invariant)
    hidden_channels: int = 64
    n_layers: int = 3
    num_epochs: int = 300
    patience: int = 30
    batch_size: int = 24
    learning_rate: float = 0.001
    random_seed: int = 42

    # regularisation
    dropout: float = 0.1              # dropout between message-passing layers

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


    # evaluate / predict modes
    model_path: str = ''             # relative path to a saved .pth model file
    predict_dir: str = ''            # directory of .xyz files for predict mode

    # directories (auto-computed)
    script_dir: str = ''             # directory of the backend script
    project_root: str = ''           # repository root
    cv_dir: str = ''
    train_dir: str = ''
    param_dir: str = ''
    evaluate_dir: str = ''
    predict_output_dir: str = ''
    split_dir: str = ''
    split_file: str = ''

    # ── computed (populated by resolve()) ───────────────────────────────
    feature_tag: str = ''            # pure feature identity: e.g. '035'
    feature_keys_parsed: List[int] = field(default_factory=list)  # [0, 3, 5]
    model_tag: str = ''              # full filename label: e.g. '035_random_EQ_3'
    model_id: str = ''               # unified filename stem: e.g. 'cebe_035_random_EQ3_h64'

    # ─────────────────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def resolve(self) -> 'AugerNetConfig':

        """Fill in computed / derived fields after loading."""
        from augernet.feature_assembly import compute_feature_tag, parse_feature_keys

        # ── Parse feature_keys string → list ────────────────────────────
        # Accepts '035' (new) or [0, 3, 5] (legacy YAML list).
        self.feature_keys_parsed = parse_feature_keys(self.feature_keys)
        # Normalise the string form so it's always canonical
        self.feature_keys = compute_feature_tag(self.feature_keys_parsed)

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
        self.evaluate_dir = os.path.join(cwd, 'evaluate_results')
        self.predict_output_dir = os.path.join(cwd, 'predict_results')

        # ── feature_tag + model_id (GNN models) ─────────────────────────
        # feature_tag  = pure feature identity, e.g. '035'
        # model_tag    = legacy label (feature_tag + split_method), kept for
        #                backward compatibility but not used in filenames.
        # model_id     = THE single unified filename stem used everywhere:
        #                cebe_{feature_tag}_{split}_{layer}{n_layers}_h{hidden}
        #
        # Example: cebe_035_random_EQ3_h64
        # All output files are then:
        #   {model_id}_fold{fold}.pth      (model)
        #   {model_id}_fold{fold}_loss.png (loss curves)
        #   {model_id}_fold{fold}_scatter.png
        #   {model_id}_cv_summary.json     (CV summary)
        #   etc.

        if self.model == 'cebe-gnn':
            self.feature_tag = compute_feature_tag(self.feature_keys_parsed)

            # Start building model_tag from pure feature_tag
            parts = [self.feature_tag]

            # Split method is part of the tag for CV-related modes
            parts.append(self.split_method)

            self.model_tag = '_'.join(parts)

            # model_id — single unified stem for ALL output filenames.
            # Format: cebe_{feature_tag}_{split}_{layer_type}{n_layers}_h{hidden}
            # Example: cebe_035_random_EQ3_h64
            self.model_id = (
                f"cebe_{self.feature_tag}_{self.split_method}"
                f"_{self.layer_type}{self.n_layers}_h{self.hidden_channels}"
            )

            # In predict / evaluate mode, if model_path is given, infer
            # model_id from the filename so output files match the loaded
            # model rather than the (possibly default) split_method.
            if self.mode in ('predict', 'evaluate') and self.model_path:
                import re as _re
                stem = os.path.splitext(os.path.basename(self.model_path))[0]
                # Strip trailing _foldN if present
                stem = _re.sub(r'_fold\d+$', '', stem)
                if stem.startswith('cebe_'):
                    self.model_id = stem

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