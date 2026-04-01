# Run Modes

AugerNet supports five run modes, set via `mode:` in the YAML config.

| Mode | Description |
|---|---|
| `train` | Train one GNN on a single k-fold split |
| `cv` | K-fold cross-validation across all folds |
| `param` | Cartesian-product hyperparameter search |
| `evaluate` | Load a saved model and evaluate on experimental data |
| `predict` | Run inference on new `.xyz` files |

---

## `train` — Train a single model

Train one GNN on a single k-fold split, with optional post-training evaluation on
experimental data.

```yaml
mode: train
train_fold: 3          # which fold to train (1-indexed)
n_folds: 5
run_evaluation: true   # evaluate on experimental data after training
run_unit_tests: true   # check permutation/rotation invariance
```

**Output:** `results/train/models/`, `results/train/outputs/`, `results/train/pngs/`

---

## `cv` — K-fold cross-validation

Train one GNN per fold, evaluate each fold on experimental data, and write a JSON summary
identifying the best fold by validation loss.

```yaml
mode: cv
n_folds: 5
split_method: random   # random | stratified | umap | size
run_evaluation: true
```

**Output:**

```
results/cv/
├── models/   {model_id}_fold{N}.pth
├── outputs/  per-fold metrics
├── pngs/     loss curves + scatter plots
└── {model_id}_cv_summary.json
```

---

## `param` — Hyperparameter search

Trains one fold per configuration from a Cartesian-product grid. Evaluates every
configuration and writes a ranked leaderboard to JSON.

```yaml
mode: param
param_grid:
  feature_keys:    ['035', '03', '0356']
  learning_rate:   [0.0001, 0.0003, 0.001]
  hidden_channels: [48, 64]
  n_layers:        [3, 4, 5]
  batch_size:      [24, 32]
```

A unique `search_id` is derived from the searched dimensions so that different grid
searches never overwrite each other. For example, searching `feature_keys` (3 values)
and `layer_type` (2 values) yields:

```
search_id = search_feature_keys3_layer_type2
```

**Output:**

```
results/param/
├── models/   {search_id}_{model_id}_fold{fold}_{config_id}.pth
├── pngs/     per-config plots
└── {search_id}_{model_id}_param_summary.json
```

---

## `evaluate` — Evaluate a saved model

Load a previously trained `.pth` model and evaluate it on the built-in experimental
CEBE dataset (`data/processed/gnn_exp_cebe_data.pt`). The fold number is inferred
automatically from the filename.

```yaml
mode: evaluate
model_path: results/train/models/cebe_gnn_035_random_EQ3_h64_fold3.pth
feature_keys: '035'
layer_type: EQ
hidden_channels: 64
n_layers: 3
```

**Output:** `results/evaluate/outputs/`, `results/evaluate/pngs/`

---

## `predict` — Predict on new molecules

Run inference on a user-specified directory of `.xyz` files using a saved model.
No pre-processing is needed — molecular graphs are built on the fly.

```yaml
mode: predict
model_path: results/train/models/cebe_gnn_035_random_EQ3_h64_fold3.pth
predict_dir: my_molecules/
```

Each file in `predict_dir/` should be a standard XYZ file named `molecule_name.xyz`.

!!! note
    The model is trained exclusively on carbon 1s CEBEs. Predictions for non-carbon
    atoms (H, N, O, F) are not meaningful and are marked with `*` in the output labels
    file. The numeric `_results.txt` file contains carbon predictions only.

**Output:**

| File | Description |
|---|---|
| `results/predict/outputs/{model_id}_fold{fold}_labels.txt` | Per-atom predictions (all elements) |
| `results/predict/outputs/{model_id}_fold{fold}_results.txt` | Numeric carbon predictions only |
