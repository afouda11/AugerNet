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
    'node_feature_norm',
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
    # auger spectrum
    'max_spec_len', 'max_ke', 'min_ke',
    'n_points', 'fwhm', 'ke_shift_calc',
    # CNN-specific
    'architecture', 'cebe_augment', 'merge_scheme',
    'label_smoothing', 'augment_noise_std', 'film_inputs',
    # splitting
    'n_folds', 'split_method',
    # train data size
    'train_frac', 'train_subsample_seed',
    # multi-task
    'mt_warmup_epochs', 'mt_finetune_auger', 'mt_finetune_epochs',
})

# ─────────────────────────────────────────────────────────────────────────────
#  CNN architecture -> model_id tag
# ─────────────────────────────────────────────────────────────────────────────

# (architecture key, filename prefix) for the fields that distinguish one
# auger-cnn run from another.  conv_dropout and film_hidden are deliberately
# absent: they are swept rarely and the id is long enough already.
#
# These keys must exist on AugerCNN1D_FiLMd.__init__.  Two renames have already
# slipped past a `.get(key, '')` here and silently emptied part of the id --
# conv_filters/conv_kernels/pool_size/fc_hidden first, then pool_kernel ->
# pool_output -- producing runs whose names claimed an architecture they were
# not trained with.  _arch_tag therefore raises on a key the architecture dict
# does not carry rather than degrading quietly.
_CNN_ARCH_TAGS: tuple = (
    ('parallel_filters',       'pf'),
    ('parallel_kernel_sizes',  'pk'),
    ('sequential_filters',     'sf'),
    ('sequential_kernel_size', 'sk'),
    ('stride',                 'st'),
    ('pool_output',            'pool'),
)


def _arch_tag(architecture: Dict[str, Any]) -> str:
    """Filename tag for a resolved auger-cnn ``architecture`` dict.

    Raises
    ------
    ValueError
        If any key in ``_CNN_ARCH_TAGS`` is missing.  That means the spec here
        and AugerCNN1D_FiLMd's signature have drifted apart, which must be a
        hard failure -- an id that silently drops a swept parameter collides
        two different models onto one set of output filenames.
    """
    missing = [k for k, _ in _CNN_ARCH_TAGS if k not in (architecture or {})]
    if missing:
        raise ValueError(
            f"architecture is missing key(s) required for model_id: {missing}. "
            f"Add them to the yml's architecture: block, or update "
            f"_CNN_ARCH_TAGS in config.py if AugerCNN1D_FiLMd's signature "
            f"changed."
        )
    parts = []
    for key, prefix in _CNN_ARCH_TAGS:
        val = architecture[key]
        if isinstance(val, (list, tuple)):
            parts.append(prefix + '_'.join(str(v) for v in val))
        else:
            parts.append(f'{prefix}{val}')
    return '_'.join(parts)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AugerNetConfig:
    """Complete configuration for a single AugerNet run."""

    # Model type and run mode
    model: str = 'cebe-gnn'          # cebe-gnn | auger-gnn | cnn
    mode:  str = 'train'              # cv | train | param | evaluate | predict

    train_data_file:      str = '' # GNN train data file, no default to ensure user selects correct file in yml
    # Separate cebe and auger eval options for multi-task gnn evaluation
    # Default eval data are the ones used in the GNN publications
    cebe_eval_data_file:  str = 'gnn_exp_cebe_data.pt'
    auger_eval_data_file: str = 'gnn_eval_auger_data.pt'

    cnn_calc_data_file:   str = 'cnn_auger_calc.pkl'
    cnn_eval_data_file:   str = 'cnn_auger_eval.pkl'

    # NOTE: there is no dataset-wide normalisation-stats file.  The CEBE target
    # shift/scale, the Auger intensity scale and the node-feature statistics are
    # all fitted on the training molecules of each fold (backend_gnn._fit_fold_norm)
    # and written to '{model}_norm.json' beside the checkpoint.  evaluate and
    # predict read that sidecar; a missing one is an error, not a fallback.

    # Run model on selected evaluation data
    run_evaluation: bool = True
    # 113 mols in exp cebe data split into: 
    #   val  (validation exp set to assist fold and param search) 
    #   eval (final evaluation sets)
    #   all  (run model on eval + val together)
    #   both (run model on both eval and val separately)
    cebe_exp_split: str = 'both'           # all | val | eval | both

    # Sanity check permutation invariance & rotational invariance/equivariance
    run_unit_tests: bool = False

    # split conformal prediction confidence level
    cp_alpha: float = 0.1              

    # k-fold
    n_folds: int = 5
    train_fold: int = 3
    split_method: str = 'random'     # random | butina 
    butina_cutoff: float = 0.65

    # reduce training data size
    train_frac: float = 1.0 
    train_subsample_seed: int = 0

    # node features
    feature_keys: str = '035'        # compact string: '035' keys [0,3,5]
    node_feature_norm: str = 'graph' # 'graph' (per molecule norm) or 'data' (all calc data norm)

    # GNN hyper-parameters
    layer_type: str = 'EQ'           # EQ (equivariant) | IN (invariant)
    hidden_channels: int = 64
    n_layers: int = 3
    num_epochs: int = 500
    patience: int = 30
    batch_size: int = 24
    learning_rate: float = 0.001
    random_seed: int = 0

    # regularisation
    dropout: float = 0.1              # dropout between message-passing layers

    # gnn loss
    auger_loss: str = 'mae' # mae or mse
    cebe_loss: str = 'mse'

    # optimizer
    optimizer_type: str = 'adamw'
    weight_decay: float = 5e-4
    gradient_clip_norm: float = 0.5
    warmup_epochs: int = 10
    min_lr: float = 1e-7

    # scheduler
    scheduler_type: str = 'cosine'   # cosine | onecycle
    pct_start: float = 0.3           # OneCycleLR only

    # ── Spectrum ─────────────────────────────────────────────────────────────────
    max_spec_len: int = 300
    max_ke: int = 273
    min_ke: int = 200
    n_points: int = 731
    fwhm: float = 3.768
    ke_shift_calc: float = -2.0


    # ── Auger GNN specific (auger-gnn) ─────────────────────────────────────────
    task_type: str = 'single'                    # single (just auger) | multi (auger + cebe)
    # multi-task hyper-params (only used when task_type == 'multi')
    mt_warmup_epochs: int = 10                   # epochs of CEBE-only warmup before joint training
    mt_finetune_auger: bool = False              # after joint training, fine-tune on Auger loss only
    mt_finetune_epochs: int = 50                 # epochs of Auger-only fine-tune (if mt_finetune_auger)

    # ── CNN specific (auger-cnn) ─────────────────────────────────────────
    architecture: Dict[str, Any] = field(default_factory=dict)  # CNN arch dict
    merge_scheme: str = 'none'       # class merging scheme
    label_smoothing: float = 0.0     # CrossEntropyLoss label smoothing (0 = off)
    augment_noise_std: float = 0.0   # online Gaussian noise std added during training (0 = off)
    normalize_intensity: bool = True
    cebe_augment: bool = True        # prepend z-score normalised delta_be to spectrum
    film_inputs: str = 'none'        # FiLM conditioning: 'none' | 'be'

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
        # model_id = cebe_gnn_{cebe_loss}_{feature_keys}_{split_method}{n_folds}_{layer}{n_layers}_h{hidden}{de_tag} 

        # auger-gnn:
        # model_id = auger_gnn_{fwhm}{task_tag}_{feature_keys}_{split_method}{n_folds}_{layer}{n_layers}_h{hidden}{de_tag}
        #   task_tag (single) = _{auger_loss}
        #   task_tag (multi)  = _multi_w{mt_warmup_epochs}[_ft{mt_finetune_epochs}]{phys_tag}_l_a{auger_loss}_c{cebe_loss}

        # auger-cnn:
        # model_id = auger_cnn_{fwhm}_{split_method}{n_folds}_{merge_scheme}BE{cebe_augment}{film_tag}_{arch_tag}{de_tag}
        #   film_tag = '' when film_inputs is 'none', else _film{film_inputs}
        #   arch_tag = pf{...}_pk{...}_sf{...}_sk{...}_st{...}_pool{...}, built
        #              from _CNN_ARCH_TAGS above

        # de_tag (all model types, data-efficiency sweep):
        #   ''                                     when train_frac == 1.0 (full data)
        #   _tf{train_frac*100:03d}_s{train_subsample_seed}   when train_frac < 1.0

        # For all run mode and model types, the specific train_fold is appeneded to model_id at runtime

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
            # Data-efficiency sweep tag
            # Only used when train_frac < 1.0
            # Encodes fraction + subsample seed
            if self.train_frac < 1.0:
                de_tag = (f'_tf{int(round(self.train_frac * 100)):03d}'
                          f'_s{self.train_subsample_seed}')
            else:
                de_tag = ''

            nn_tag = '' if self.node_feature_norm == 'graph' else '_ndata'

            if self.model == 'cebe-gnn':
                self.model_id = (
                    f"cebe_gnn_{self.cebe_loss}_{self.feature_keys}_{self.split_method}{self.n_folds}"
                    f"_{self.layer_type}{self.n_layers}_h{self.hidden_channels}{nn_tag}{de_tag}"
                )
            if self.model == 'auger-gnn':
                if self.task_type == 'multi':
                    loss_tag = f'_a{self.auger_loss}_c{self.cebe_loss}'
                    ft_tag = f'_ft{self.mt_finetune_epochs}' if self.mt_finetune_auger else ''

                    task_tag = f'_multi_w{self.mt_warmup_epochs}{ft_tag}_l{loss_tag}'
                else:
                    task_tag = f'_{self.auger_loss}'
                fwhm_str = str(self.fwhm).replace('.', 'pt')
                self.model_id = (
                    f"auger_gnn_{fwhm_str}{task_tag}_{self.feature_keys}_{self.split_method}{self.n_folds}"
                    f"_{self.layer_type}{self.n_layers}_h{self.hidden_channels}{nn_tag}{de_tag}"
                )
            if self.model == 'auger-cnn':
                fwhm_str = str(self.fwhm).replace('.', 'pt')
                film = str(getattr(self, 'film_inputs', 'none') or 'none')
                #film_tag = '' if film in ('none', '') else '_film' + film.replace(',', '')
                # Tag built from the real AugerCNN1D_FiLMd architecture keys;
                # _arch_tag raises if the spec and the yml have drifted apart.
                if film == 'none' and self.cebe_augment == False: 
                    be_tag = 'nobe'
                elif film == 'be' and self.cebe_augment == False:
                    be_tag = 'filmbe'
                elif film == 'none' and self.cebe_augment == True:
                    be_tag = 'augbe'
                elif film == 'be' and self.cebe_augment == True: 
                    raise ValueError(
                        "Only augmented binding energies or FiLM binding energies "
                        "can be used, not both. Set either:\n"
                        "  cebe_augment: false (to use FiLM)\n"
                        "  film_inputs: 'none' (to use augmented BE)\n"
                    )
                arch_str = _arch_tag(self.architecture)
                self.model_id = (
                    f"auger_cnn_{fwhm_str}_{self.split_method}{self.n_folds}_{self.merge_scheme}"
                    f"_{be_tag}_{arch_str}{de_tag}"
                )

        # results sub dirs: outputs files, train loss and eval pngs, and models 
        self.outputs_dir = os.path.join(self.result_dir, 'outputs', self.model_id)
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
    if cfg.mode == 'predict' and cfg.node_feature_norm == 'data':
        raise ValueError(
            "  Predict mode is not compatible with node_feature_norm: 'data'.\n"
            "  Dataset-wide feature statistics are computed from the calculated\n"
            "  training set in load_data(), which predict mode never loads, so the\n"
            "  features would be assembled with per-graph scaling instead and would\n"
            "  not match the model's training distribution.\n")

    # ── Required data files ──────────────────────────────────────────────
    # The GNN backends need train_data_file for every mode that loads data
    # (predict builds its graphs from predict_dir instead).  Left empty,
    # LoadDataset resolves to the processed directory itself and torch.load
    # fails with an IsADirectoryError that says nothing about the real cause.
    if cfg.model in ('cebe-gnn', 'auger-gnn') and cfg.mode != 'predict' \
            and not cfg.train_data_file:
        raise ValueError(
            f"'{cfg.model}' in mode '{cfg.mode}' requires 'train_data_file' — "
            f"the processed graph file in data/processed to train on.\n"
            f"  cebe-gnn:   train_data_file: gnn_calc_cebe_data.pt\n"
            f"  auger-gnn:  train_data_file: gnn_calc_auger_data.pt\n"
            f"(auger-cnn does not use this field; it reads cnn_calc_data_file "
            f"and cnn_eval_data_file.)"
        )

    # Resolve project root from the config file's location
    # Walk up until we find setup.py or augernet/
    cfg.resolve()

    return cfg