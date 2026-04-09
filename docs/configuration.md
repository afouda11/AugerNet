# Configuration Reference

All options are set in a YAML config file passed to the CLI via `--config`.
See `examples/` for complete examples for each model type and mode.

```bash
python -m augernet --config examples/gnn_cebe_configs/cv.yml
```

---

## Identity

| Field   | Default    | Description                                         |
|---------|------------|-----------------------------------------------------|
| `model` | `cebe-gnn` | Model type: `cebe-gnn` / `auger-gnn` / `auger-cnn` |
| `mode`  | `train`    | Run mode: `cv` / `train` / `param` / `evaluate` / `predict` |

---

## Node Features (GNN models only)

Node features are selected by a compact key string where each digit maps to a feature
from the catalog below. For example, `'035'` selects SkipAtom-200 + atomic BE + e-score.

| Key | Name           | Dim | Description                             |
|-----|----------------|-----|-----------------------------------------|
| `0` | `skipatom_200` | 200 | SkipAtom atom-type embedding            |
| `1` | `skipatom_30`  | 30  | SkipAtom atom-type embedding (compact)  |
| `2` | `onehot`       | 5   | Element one-hot (H, C, N, O, F)         |
| `3` | `atomic_be`    | 1   | Isolated-atom 1s binding energy         |
| `4` | `mol_be`       | 1   | Molecular CEBE for C, atomic for others |
| `5` | `e_score`      | 1   | Electronegativity-difference score      |
| `6` | `env_onehot`   | ~8  | Carbon-environment one-hot              |
| `7` | `morgan_fp`    | 256 | Per-atom Morgan fingerprint (ECFP2)     |

```yaml
feature_keys: '035'   # SkipAtom-200 + atomic_be + e_score
```

---

## GNN Architecture

These fields apply to both `cebe-gnn` and `auger-gnn`.

| Field             | Default | Description                            |
|-------------------|---------|----------------------------------------|
| `layer_type`      | `EQ`    | `EQ` (equivariant) or `IN` (invariant) |
| `hidden_channels` | `64`    | Hidden channel width                   |
| `n_layers`        | `3`     | Number of message-passing layers       |

---

## GNN Training

| Field                | Default  | Description                                |
|----------------------|----------|--------------------------------------------|
| `num_epochs`         | `300`    | Maximum training epochs                    |
| `patience`           | `30`     | Early-stopping patience (epochs)           |
| `batch_size`         | `24`     | Mini-batch size                            |
| `learning_rate`      | `0.001`  | Peak learning rate                         |
| `optimizer_type`     | `adamw`  | Optimizer                                  |
| `weight_decay`       | `5e-4`   | L2 regularisation                          |
| `gradient_clip_norm` | `0.5`    | Max gradient norm for clipping             |
| `warmup_epochs`      | `10`     | Linear LR warmup epochs                    |
| `min_lr`             | `1e-7`   | Minimum learning rate for scheduler        |
| `scheduler_type`     | `cosine` | LR scheduler: `cosine` / `onecycle`        |
| `dropout`            | `0.1`    | Dropout between message-passing layers     |
| `random_seed`        | `42`     | Random seed for reproducibility            |

---

## Auger GNN — Spectrum Settings

These fields apply to `auger-gnn` only.

| Field           | Default  | Description                                  |
|-----------------|----------|----------------------------------------------|
| `spectrum_type` | `stick`  | `stick` (singlet+triplet) or `fitted`        |
| `max_spec_len`  | `300`    | Maximum number of stick lines per spectrum   |
| `max_ke`        | `273`    | Maximum kinetic energy (eV)                  |
| `min_ke`        | `200`    | Minimum kinetic energy (eV)                  |
| `n_points`      | `731`    | Number of grid points for fitted spectra     |
| `fwhm`          | `3.768`  | Broadening FWHM for fitted spectra (eV)      |
| `ke_shift_calc` | `-2.0`   | Kinetic energy shift for calculated data     |

---

## Auger CNN — Specific Settings

These fields apply to `auger-cnn` only.

### Architecture

The CNN architecture is specified as a dict. If omitted, the built-in
`recommended` preset from `cnn_train_utils.py` is used.

```yaml
architecture:
  conv_filters: [32, 64, 128, 128]
  conv_kernels: [41, 21, 11, 7]
  pool_size: 3
  fc_hidden: [256, 128]
  use_batch_norm: true
  dropout: 0.3
  dropout_conv: 0.1
```

Architecture dict keys:

| Key             | Type       | Description                              |
|-----------------|------------|------------------------------------------|
| `conv_filters`  | list[int]  | Number of filters per conv block         |
| `conv_kernels`  | list[int]  | Kernel size per conv block               |
| `pool_size`     | int        | Max-pool kernel size                     |
| `fc_hidden`     | list[int]  | Hidden layer sizes in the FC head        |
| `use_batch_norm`| bool       | BatchNorm after each conv block          |
| `dropout`       | float      | Dropout rate for FC layers               |
| `dropout_conv`  | float      | Dropout after each conv block            |

### Data and augmentation

| Field               | Default  | Description                               |
|---------------------|----------|-------------------------------------------|
| `merge_scheme`      | `none`   | Carbon-class merging scheme               |
| `use_augmented`     | `true`   | Prepend z-score normalised delta_be       |

### Training

| Field                 | Default  | Description                              |
|-----------------------|----------|------------------------------------------|
| `num_epochs`          | `500`    | Maximum training epochs                  |
| `patience`            | `40`     | Early-stopping patience                  |
| `batch_size`          | `64`     | Mini-batch size                          |
| `learning_rate`       | `3e-4`   | Peak learning rate                       |
| `weight_decay`        | `1e-4`   | L2 regularisation                        |
| `random_seed`         | `42`     | Random seed for reproducibility          |

### Splitting

The CNN uses **random molecule-level splitting** only. All carbon atoms
from the same molecule are kept in the same fold to prevent data leakage.
The GNN `split_method` options (`butina`, etc.) do not apply to the CNN.

---

## Cross-Validation

| Field            | Default   | Description                                       |
|------------------|-----------|---------------------------------------------------|
| `n_folds`        | `5`       | Number of CV folds                                |
| `train_fold`     | `3`       | Which fold to use for `train` / `param` modes     |
| `split_method`   | `random`  | `random` / `butina` (GNN only; CNN always random) |
| `run_evaluation` | `true`    | Evaluate on experimental data after each fold     |
| `exp_split`      | `both`    | Experimental data subset: `all` / `val` / `eval` / `both` |
| `run_unit_tests` | `false`   | Check permutation/rotation invariance (GNN only)  |

---

## Hyperparameter Search

| Field        | Default | Description                                   |
|--------------|---------|-----------------------------------------------|
| `param_grid` | `{}`    | Dict of `field: [value, ...]` lists to search |

### GNN param grid example

```yaml
param_grid:
  feature_keys:    ['035', '03']
  learning_rate:   [0.0001, 0.001]
  hidden_channels: [48, 64]
```

### CNN param grid example

```yaml
param_grid:
  merge_scheme:    [none, heteroatom]
  broadening_fwhm: [1.2, 1.6, 2.0]
  learning_rate:   [0.0001, 0.0003]
  use_augmented:   [true, false]
```

---

## Evaluate / Predict

| Field         | Default | Description                                                |
|---------------|---------|------------------------------------------------------------|
| `model_path`  | `''`    | Path to a saved `.pth` model (relative to cwd or absolute) |
| `predict_dir` | `''`    | Directory of `.xyz` files for `predict` mode (GNN only)    |

For evaluate and predict modes, the `model_id` used in output filenames is
derived from the `model_path` filename (minus the `.pth` extension). The
architecture fields must match the values used during training.

> **Note:** Predict mode is not yet implemented for `auger-cnn`.

---

## Output File Naming

### model_id

Each model type constructs its `model_id` differently:

| Model        | Format                                                                     | Example                      |
|--------------|---------------------------------------------------------------------------|------------------------------|
| `cebe-gnn`   | `cebe_gnn_{feature_keys}_{split}_{layer_type}{n_layers}_h{hidden}`        | `cebe_gnn_035_random_EQ3_h64`|
| `auger-gnn`  | `auger_gnn_{feature_keys}_{split}_{layer_type}{n_layers}_h{hidden}`       | `auger_gnn_035_random_EQ3_h64`|
| `auger-cnn`  | `auger_cnn_{merge_scheme}`                                                | `auger_cnn_none`             |

For evaluate and predict modes, `model_id` is the `model_path` filename
without the `.pth` extension.

### GNN output files (per fold)

| File | Description |
|------|-------------|
| `{model_id}_fold{fold}.pth` | Saved model weights |
| `{model_id}_fold{fold}_loss.png` | Training/validation loss curves |
| `{model_id}_fold{fold}_scatter.png` | Predicted vs experimental scatter |
| `{model_id}_fold{fold}_results.txt` | Numeric predicted vs true (carbon only) |
| `{model_id}_cv_summary.json` | Cross-validation summary (`cv` mode) |

### CNN output files (per fold)

| File | Description |
|------|-------------|
| `{model_id}_fold{fold}.pth` | Saved model weights |
| `training_history_fold{fold}.csv` | Per-epoch loss and accuracy |
| `training_plots_fold{fold}.png` | Training curve plots |

### Param search files (per config)

| File | Description |
|------|-------------|
| `{search_id}_{model_id}_fold{fold}_{config_id}.pth` | Saved model weights |
| `{search_id}_{model_id}_param_summary.json` | Ranked leaderboard JSON |

### Output directory naming

Each model type writes to its own results directory:

| Model      | Directory pattern                |
|------------|----------------------------------|
| `cebe-gnn` | `cebe_gnn_{mode}_results/`       |
| `auger-gnn`| `auger_gnn_{mode}_results/`      |
| `auger-cnn`| `auger_cnn_{mode}_results/`      |

Each contains `outputs/` and `pngs/` subdirectories. Train, cv, and param
modes also create a `models/` subdirectory.
