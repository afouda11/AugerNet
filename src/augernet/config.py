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
#  Overridable fields  –  the canonical set that param_grid may override
# ─────────────────────────────────────────────────────────────────────────────

OVERRIDABLE_FIELDS: frozenset[str] = frozenset({
    # node features
    'feature_keys',
    # GNN hyper-parameters
    'layer_type', 'hidden_channels', 'n_layers',
    'num_epochs', 'patience', 'batch_size', 'learning_rate', 'random_seed',
    # regularisation
    'dropout',
    # optimizer
    'optimizer_type', 'weight_decay', 'gradient_clip_norm',
    'warmup_epochs', 'min_lr',
    # scheduler
    'scheduler_type', 'pct_start',
    # spectrum
    'spectrum_type', 'max_spec_len', 'max_ke', 'min_ke',
    'n_points', 'fwhm', 'ke_shift_calc',
    # CNN-specific
    'architecture', 'use_augmented', 'merge_scheme',
    # splitting
    'n_folds', 'split_method',
})

# ─────────────────────────────────────────────────────────────────────────────
#  Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AugerNetConfig:
    """Complete configuration for a single AugerNet run."""

    # Model type and run mode
    model: str = 'cebe-gnn'          # cebe-gnn | auger-gnn | cnn
    mode:  str = 'train'              # cv | train | param | evaluate | predict

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

    norm_stats_file: str = 'cebe_norm_stats.pt'

    # ── Spectrum ─────────────────────────────────────────────────────────────────
    spectrum_type: str = 'stick'                # stick | fitted
    max_spec_len: int = 300
    max_ke: int = 273
    min_ke: int = 200
    n_points: int = 731
    fwhm: float = 3.768
    ke_shift_calc: float = -2.0

    # ── CNN-specific (auger-cnn) ─────────────────────────────────────────
    architecture: Dict[str, Any] = field(default_factory=dict)  # CNN arch dict
    use_augmented: bool = True       # prepend z-score normalised delta_be to spectrum
    merge_scheme: str = 'none'       # class merging scheme

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
    feature_keys_parsed: List[int] = field(default_factory=list)  # [0, 3, 5]
    model_id: str = ''               # unified filename stem: e.g. 'cebe_gnn_035_random_EQ3_h64'


    # ─────────────────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def resolve(self) -> 'AugerNetConfig':

        """Fill in computed / derived fields after loading."""
        from augernet.feature_assembly import compute_feature_tag, parse_feature_keys

        # ── norm_stats_file  ──────────────────────────────────────
        # assign filename into processed dir
        self.norm_stats_file = os.path.join(
                DATA_PROCESSED_DIR, self.norm_stats_file
            )

        cwd = os.getcwd()

        # ── results dir ──────────────────────────────────────────────
        # By default the results and models are written to current working directory
        # named by "<model>_<mode>_results"
        if self.model == 'cebe-gnn':
            self.result_dir  = os.path.join(cwd, f'cebe_gnn_{self.mode}_results')
        if self.model == 'auger-gnn':
            self.result_dir  = os.path.join(cwd, f'auger_gnn_{self.mode}_results')
        if self.model == 'auger-cnn':
            self.result_dir  = os.path.join(cwd, f'auger_cnn_{self.mode}_results')

        os.makedirs(self.result_dir, exist_ok=True)

        # ── Parse and canonicalize feature_keys for GNN ────────────────────────────
        if self.model in ('cebe-gnn', 'auger-gnn'):
            self.feature_keys_parsed = parse_feature_keys(self.feature_keys)
            self.feature_keys = compute_feature_tag(self.feature_keys_parsed)

        # ── model_id's for output file names  ────────────────────────

        # cebe-gnn: 
        # model_id = cebe_gnn_{feature_keys}_{split_method}{n_folds}_{layer}{n_layers}_h{hidden}

        # auger-gnn spectrum_type: fitted
        # model_id = auger_gnn_{spectrum_type}{fwhm}_{feature_keys}_{split_method}{n_folds}_{layer}{n_layers}_h{hidden}

        # auger-gnn spectrum_type: stick (no fwhm in model_id)
        # model_id = auger_gnn_{spectrum_type}_{feature_keys}_{split_method}{n_folds}_{layer}{n_layers}_h{hidden}

        # auger-cnn
        # model_id = auger_cnn_{fwhm}_{split_method}{n_folds}_{merge_scheme}_BE{use_augmented}_f{filters}_k{kernels}_p{pool}_h{hidden}
        
        # For all run mode and model types, the specific train_fold is appeneded to model_id at runtime
        # For parameter search (param) mode
        # In train_driver.py: 
        #   Parameter types in param_grid prefixed to model_id (file names get very long for many params!)
        #   Config id of the parameter grid appended to model_id

        # For predict/evaluate modes the user supplies model_path in
        # the YAML and model_id is derived from that filename.
        if self.mode in ('predict', 'evaluate') and self.model_path:
            stem = os.path.splitext(os.path.basename(self.model_path))[0]
            self.model_id = stem
        else:
            if self.model == 'cebe-gnn':
                self.model_id = (
                    f"cebe_gnn_{self.feature_keys}_{self.split_method}{self.n_folds}"
                    f"_{self.layer_type}{self.n_layers}_h{self.hidden_channels}"
                )
            if self.model == 'auger-gnn':
                if self.spectrum_type == 'fitted':
                    fwhm_str = str(self.fwhm).replace('.', 'pt')
                    self.model_id = (
                        f"auger_gnn_{self.spectrum_type}{fwhm_str}_{self.feature_keys}_{self.split_method}{self.n_folds}"
                        f"_{self.layer_type}{self.n_layers}_h{self.hidden_channels}"
                    )
                if self.spectrum_type == 'stick':
                    self.model_id = (
                        f"auger_gnn_{self.spectrum_type}_{self.feature_keys}_{self.split_method}{self.n_folds}"
                        f"_{self.layer_type}{self.n_layers}_h{self.hidden_channels}"
                    )
            if self.model == 'auger-cnn':
                fwhm_str = str(self.fwhm).replace('.', 'pt')
                # Build filter and kernel strings from architecture
                filters_str = 'f' + '_'.join(str(f) for f in self.architecture.get('conv_filters', []))
                kernels_str = 'k' + '_'.join(str(k) for k in self.architecture.get('conv_kernels', []))
                p_str =  f"p{self.architecture.get('pool_size', [])}"
                h_str =  f"h{self.architecture.get('fc_hidden', [])}"
                self.model_id = (
                    f"auger_cnn_{fwhm_str}_{self.split_method}{self.n_folds}_{self.merge_scheme}"
                    f"BE{self.use_augmented}_{filters_str}_{kernels_str}_{p_str}_{h_str}"
                )

        # results sub dirs: outputs files, train loss and eval pngs, and models 
        self.outputs_dir = os.path.join(self.result_dir, 'outputs')
        os.makedirs(self.outputs_dir, exist_ok=True)

        #no scatter or training loss pngs for predict, just raw values to output
        if self.mode != 'predict':
            self.pngs_dir    = os.path.join(self.result_dir, 'pngs')
            os.makedirs(self.pngs_dir, exist_ok=True)

        if self.mode in ('train', 'cv', 'param'):
            self.models_dir  = os.path.join(self.result_dir, 'models')
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

    # Validate param_grid keys against the canonical overridable set
    if cfg.param_grid:
        bad_keys = set(cfg.param_grid.keys()) - OVERRIDABLE_FIELDS
        if bad_keys:
            raise ValueError(
                f"param_grid contains non-overridable keys:\n"
                f"  {', '.join(sorted(bad_keys))}\n"
                f"Allowed param_grid keys:\n"
                f"  {', '.join(sorted(OVERRIDABLE_FIELDS))}"
            )

    # Resolve project root from the config file's location
    # Walk up until we find setup.py or augernet/
    cfg.resolve()

    return cfg