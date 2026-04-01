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
    feature_keys_parsed: List[int] = field(default_factory=list)  # [0, 3, 5]
    model_id: str = ''               # unified filename stem: e.g. 'cebe_gnn_035_random_EQ3_h64'

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
    train_data: str = ''             # path to CNN training pickle
    eval_data: str = ''              # path to CNN eval pickle
    use_augmented: bool = True       # normalised delta_be augmentation
    augmented_scaled: bool = False   # scaled delta_be augmentation
    delta_be_scale: float = 100.0    # scale factor for delta_be
    use_cosine_schedule: bool = True # cosine-annealing LR schedule
    broadening_fwhm: float = 1.6     # FWHM for Gaussian broadening (eV)
    energy_min: float = 200.0        # energy grid minimum (eV)
    energy_max: float = 273.0        # energy grid maximum (eV)
    n_spectrum_points: int = 731     # number of spectrum grid points
    merge_scheme: str = 'none'       # class merging scheme

    # ─────────────────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def resolve(self) -> 'AugerNetConfig':

        """Fill in computed / derived fields after loading."""
        from augernet.feature_assembly import compute_feature_tag, parse_feature_keys


        # ── norm_stats_file (CEBE) ──────────────────────────────────────
        if self.model == 'cebe-gnn' and not self.norm_stats_file:
            self.norm_stats_file = os.path.join(
                DATA_PROCESSED_DIR, 'cebe_norm_stats.pt'
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

        if self.model in ('cebe-gnn', 'auger-gnn'):

            # ── feature_keys + model_id (GNN models) ────────────────────────
            # feature_keys = compact string tag, e.g. '035'
            # model_id     = Filename stem:
            #                cebe_gnn_{feature_keys}_{split}_{layer}{n_layers}_h{hidden}
            #
            # Example model_id: cebe_gnn_035_random_EQ3_h64
            # All output files are then:
            #   {model_id}_fold{fold}.pth      (model)
            #   {model_id}_fold{fold}_loss.png (loss curves)
            #   {model_id}_fold{fold}_scatter.png
            #   {model_id}_cv_summary.json     (CV summary)
            #   etc.
            #
            # For cv/train/param modes this is the default stem used for
            # summary filenames (e.g. *_cv_summary.json).  The backend's
            # train_single_run() builds its own per-config model_id that
            # may differ when param search overrides hyperparameters.
            #
            # For predict/evaluate modes the user supplies model_path in
            # the YAML and model_id is derived from that filename.

            # ── Parse feature_keys string to list ────────────────────────────
            self.feature_keys_parsed = parse_feature_keys(self.feature_keys)
            # Normalise the string form so it's always canonical
            self.feature_keys = compute_feature_tag(self.feature_keys_parsed)

            if self.mode in ('predict', 'evaluate') and self.model_path:
                # User-supplied model file → derive model_id from filename
                stem = os.path.splitext(os.path.basename(self.model_path))[0]
                self.model_id = stem
            else:
                # cv / train / param → build from config fields
                if self.model == 'cebe-gnn':
                    self.model_id = (
                        f"cebe_gnn_{self.feature_keys}_{self.split_method}"
                        f"_{self.layer_type}{self.n_layers}_h{self.hidden_channels}"
                    )
                if self.model == 'auger-gnn':
                    self.model_id = (
                        f"auger_gnn_{self.feature_keys}_{self.split_method}"
                        f"_{self.layer_type}{self.n_layers}_h{self.hidden_channels}"
                    )
        
        elif self.model == 'auger-cnn':
            # CNN model — model_id uses split method + CNN suffix
            # The CNN doesn't use GNN feature_keys / layer_type.
            _split = getattr(self, 'split_method', 'size')
            _ms = getattr(self, 'merge_scheme', 'none')
            _fwhm = getattr(self, 'broadening_fwhm', 1.6)
            _fwhm_str = f"fwhm{str(_fwhm).replace('.', 'pt')}"
            parts = ['auger_cnn', _split]
            if _ms != 'none':
                parts.append(_ms)
            parts.append(_fwhm_str)
            self.model_id = '_'.join(parts)

        self.outputs_dir = os.path.join(self.result_dir, 'outputs')
        os.makedirs(self.outputs_dir, exist_ok=True)

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

    # Resolve project root from the config file's location
    # Walk up until we find setup.py or augernet/
    cfg.resolve()

    return cfg