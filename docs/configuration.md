# Configuration Reference

All options are set in a YAML config file passed to the CLI via `--config`.
See `examples/gnn_cebe_configs/` for complete examples for each mode.

```bash
python -m augernet --config examples/gnn_cebe_configs/cv.yml
```

---

## Identity

| Field   | Default    | Description                            |
|---------|------------|----------------------------------------|
| `model` | `cebe-gnn` | Model type (`cebe-gnn` only currently) |
| `mode`  | `train`    | Run mode: `cv` / `train` / `param` / `evaluate` / `predict` |

---

## Node Features

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

| Field             | Default | Description                            |
|-------------------|---------|----------------------------------------|
| `layer_type`      | `EQ`    | `EQ` (equivariant) or `IN` (invariant) |
| `hidden_channels` | `64`    | Hidden channel width                   |
| `n_layers`        | `3`     | Number of message-passing layers       |

---

## Training

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

## Cross-Validation

| Field            | Default   | Description                                       |
|------------------|-----------|---------------------------------------------------|
| `n_folds`        | `5`       | Number of CV folds                                |
| `train_fold`     | `3`       | Which fold to use for `train` / `param` modes     |
| `split_method`   | `random`  | `random` / `butina`                               |
| `run_evaluation` | `true`    | Evaluate on experimental data after each fold     |
| `exp_split`      | `both`    | Experimental data subset: `all` / `val` / `eval` / `both` |
| `run_unit_tests` | `false`   | Check permutation/rotation invariance             |

---

## Hyperparameter Search

| Field        | Default | Description                                   |
|--------------|---------|-----------------------------------------------|
| `param_grid` | `{}`    | Dict of `field: [value, ...]` lists to search |

```yaml
param_grid:
  feature_keys:    ['035', '03']
  learning_rate:   [0.0001, 0.001]
  hidden_channels: [48, 64]
```

---

## Evaluate / Predict

| Field         | Default | Description                                                |
|---------------|---------|------------------------------------------------------------|
| `model_path`  | `''`    | Path to a saved `.pth` model (relative to cwd or absolute) |
| `predict_dir` | `''`    | Directory of `.xyz` files for `predict` mode               |

For evaluate and predict modes, the `model_id` used in output filenames is
derived from the `model_path` filename (minus the `.pth` extension). The
architecture fields (`feature_keys`, `layer_type`, `hidden_channels`,
`n_layers`) must match the values used during training.

---

## Output File Naming

### model_id

For train, cv, and param modes, `model_id` is constructed from the config:

```
cebe_gnn_{feature_keys}_{split_method}_{layer_type}{n_layers}_h{hidden_channels}
```

Example: `cebe_gnn_035_random_EQ3_h64`

For evaluate and predict modes, `model_id` is the `model_path` filename
without the `.pth` extension.

In param search mode, each configuration in the grid may produce a different
`model_id` when hyperparameters like `feature_keys` or `hidden_channels` are
varied.

### Train / CV files (per fold)

| File | Description |
|------|-------------|
| `{model_id}_fold{fold}.pth` | Saved model weights |
| `{model_id}_fold{fold}_loss.png` | Training/validation loss curves |
| `{model_id}_fold{fold}_scatter.png` | Predicted vs experimental scatter |
| `{model_id}_fold{fold}_results.txt` | Numeric predicted vs true (carbon only) |
| `{model_id}_cv_summary.json` | Cross-validation summary (`cv` mode) |

### Param search files (per config)

| File | Description |
|------|-------------|
| `{search_id}_{model_id}_fold{fold}_{config_id}.pth` | Saved model weights |
| `{search_id}_{model_id}_fold{fold}_{config_id}_loss.png` | Loss curves |
| `{search_id}_{model_id}_fold{fold}_{config_id}_scatter.png` | Scatter plot |
| `{search_id}_{model_id}_param_summary.json` | Ranked leaderboard JSON |

### Output directory naming

Each mode writes to a directory named `cebe_gnn_{mode}_results/` in the
working directory:

| Mode     | Directory                     |
|----------|-------------------------------|
| train    | `cebe_gnn_train_results/`     |
| cv       | `cebe_gnn_cv_results/`        |
| param    | `cebe_gnn_param_results/`     |
| evaluate | `cebe_gnn_evaluate_results/`  |
| predict  | `cebe_gnn_predict_results/`   |

Each contains `outputs/` and `pngs/` subdirectories. Train, cv, and param
modes also create a `models/` subdirectory.
