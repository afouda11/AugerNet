# Configuration Reference

All options are set in a YAML config file passed to the CLI via `--config`.
See `config_examples/` for complete examples for each mode.

```bash
uv run augernet --config config_examples/cv.yml
```

---

## Identity

| Field   | Default    | Description                            |
|---------|------------|----------------------------------------|
| `model` | `cebe-gnn` | Model type (`cebe-gnn` only currently) |
| `mode`  | `train`    | Run mode: `cv` \| `train` \| `param` \| `evaluate` \| `predict` |

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
| `num_epochs`         | `500`    | Maximum training epochs                    |
| `patience`           | `50`     | Early-stopping patience (epochs)           |
| `batch_size`         | `24`     | Mini-batch size                            |
| `learning_rate`      | `0.001`  | Peak learning rate                         |
| `optimizer_type`     | `adamw`  | Optimizer                                  |
| `weight_decay`       | `5e-4`   | L2 regularisation                          |
| `gradient_clip_norm` | `0.5`    | Max gradient norm for clipping             |
| `warmup_epochs`      | `10`     | Linear LR warmup epochs                    |
| `min_lr`             | `1e-7`   | Minimum learning rate for scheduler        |
| `scheduler_type`     | `cosine` | LR scheduler: `cosine` \| `onecycle`      |
| `dropout`            | `0.0`    | Dropout between message-passing layers     |
| `random_seed`        | `42`     | Random seed for reproducibility            |

---

## Cross-Validation

| Field            | Default   | Description                                       |
|------------------|-----------|---------------------------------------------------|
| `n_folds`        | `5`       | Number of CV folds                                |
| `train_fold`     | `3`       | Which fold to use for `train` / `param` modes     |
| `split_method`   | `random`  | `random` \| `stratified` \| `umap` \| `size`     |
| `run_evaluation` | `true`    | Evaluate on experimental data after each fold     |
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

---

## Output File Naming

All output files share a unified `model_id` stem derived from the config:

```
cebe_{feature_keys}_{split_method}_{layer_type}{n_layers}_h{hidden_channels}
```

Example: `cebe_035_random_EQ3_h64`

### Train / CV files (per fold)

| File | Description |
|---|---|
| `{model_id}_fold{fold}.pth` | Saved model weights |
| `{model_id}_fold{fold}_loss.png` | Training/validation loss curves |
| `{model_id}_fold{fold}_scatter.png` | Predicted vs experimental scatter |
| `{model_id}_fold{fold}_results.txt` | Numeric predicted vs true (carbon only) |
| `{model_id}_cv_summary.json` | Cross-validation summary (`cv` mode) |

### Param search files (per config)

| File | Description |
|---|---|
| `{search_id}_{model_id}_fold{fold}_{config_id}.pth` | Saved model weights |
| `{search_id}_{model_id}_fold{fold}_{config_id}_loss.png` | Loss curves |
| `{search_id}_{model_id}_fold{fold}_{config_id}_scatter.png` | Scatter plot |
| `{search_id}_param_summary.json` | Ranked leaderboard JSON |
