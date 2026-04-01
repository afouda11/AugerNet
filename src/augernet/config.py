"""
AugerNet Configuration System
=============================

Loads configuration from YAML files and provides a single
``AugerNetConfig`` dataclass consumed by ``train_driver.py``.

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
    model: str = 'cebe-gnn'          # cebe-gnn | auger-gnn | cnn
    mode: str = 'train'              # cv | train | param | evaluate | predict

    # Evaluation on exp.evaluation data
    run_evaluation: bool = True
    # 113 mols in exp cebe data split into: 
    #   validation (to assist fold and param search) 
    #   final evaluation sets
    exp_split: str = 'both'           # all | val | eval | both
    # Sanity check permutation invariance & rotational invariance/equivariance
    run_unit_tests: bool = False

    # k-fold
    n_folds: int = 5
    train_fold: int = 3
    split_method: str = 'random'     # random | butina

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


    # evaluate + predict modes
    model_path: str = ''             # relative path to a saved .pth model file
    # predict mode
    predict_dir: str = ''            # directory of .xyz files for predict mode

    # directories (auto-computed)
    result_dir: str = ''
    models_dir: str = ''
    outputs_dir: str = ''
    pngs_dir: str = ''

    # ── computed (populated by resolve()) ───────────────────────────────
    feature_tag: str = ''            # pure feature identity: e.g. '035'
    feature_keys_parsed: List[int] = field(default_factory=list)  # [0, 3, 5]
    model_id: str = ''               # unified filename stem: e.g. 'cebe_gnn_035_random_EQ3_h64'

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

        cwd = os.getcwd()

        # ── feature_tag + model_id (GNN models) ─────────────────────────
        # feature_tag  = pure feature identity, e.g. '035'
        # model_id     = Filename stem:
        #                cebe_gnn_{feature_tag}_{split}_{layer}{n_layers}_h{hidden}
        #
        # Example model_id: cebe_gnn_035_random_EQ3_h64
        # All output files are then:
        #   {model_id}_fold{fold}.pth      (model)
        #   {model_id}_fold{fold}_loss.png (loss curves)
        #   {model_id}_fold{fold}_scatter.png
        #   {model_id}_cv_summary.json     (CV summary)
        #   etc.

        if self.model == 'cebe-gnn':

            # ── results dir ──────────────────────────────────────────────
            # By default the results and models are written to current working directory
            # named by "<model>_<mode>_results"
            self.result_dir  = os.path.join(cwd, f'cebe_gnn_{self.mode}_results')

            self.feature_tag = compute_feature_tag(self.feature_keys_parsed)

            # model_id — stem for output filenames.
            # Format: cebe_gnn_{feature_tag}_{split}_{layer_type}{n_layers}_h{hidden}
            # Example: cebe_gnn_035_random_EQ3_h64
            #
            # For cv/train/param modes this is the default stem used for
            # summary filenames (e.g. *_cv_summary.json).  The backend's
            # train_single_run() builds its own per-config model_id that
            # may differ when param search overrides hyperparameters.
            #
            # For predict/evaluate modes the user supplies model_path in
            # the YAML and model_id is derived from that filename.
            if self.mode in ('predict', 'evaluate') and self.model_path:
                # User-supplied model file → derive model_id from filename
                stem = os.path.splitext(os.path.basename(self.model_path))[0]
                self.model_id = stem
            else:
                # cv / train / param → build from config fields
                self.model_id = (
                    f"cebe_gnn_{self.feature_tag}_{self.split_method}"
                    f"_{self.layer_type}{self.n_layers}_h{self.hidden_channels}"
                )

        self.models_dir  = os.path.join(self.result_dir, 'models')
        self.outputs_dir = os.path.join(self.result_dir, 'outputs')
        self.pngs_dir    = os.path.join(self.result_dir, 'pngs')

        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)
        os.makedirs(self.pngs_dir, exist_ok=True)
        if self.mode in ('train', 'cv', 'param'):
            os.makedirs(self.models_dir, exist_ok=True)

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