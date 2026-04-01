# Run Modes

AugerNet supports five run modes, set via `mode:` in the YAML config.

| Mode       | Description                                      |
|------------|--------------------------------------------------|
| `train`    | Train one GNN on a single k-fold split           |
| `cv`       | K-fold cross-validation across all folds         |
| `param`    | Cartesian-product hyperparameter search          |
| `evaluate` | Load a saved model and evaluate on experimental data |
| `predict`  | Run inference on new `.xyz` files                |

---

## train -- Train a single model

Train one GNN on a single k-fold split, with optional post-training evaluation on
experimental data.

```yaml
mode: train
train_fold: 3          # which fold to train (1-indexed)
n_folds: 5
run_evaluation: true   # evaluate on experimental data after training
run_unit_tests: true   # check permutation/rotation invariance
```

Output is written to `cebe_gnn_train_results/` in the working directory:

```
cebe_gnn_train_results/
  models/   {model_id}_fold{N}.pth
  outputs/  per-fold metrics
  pngs/     loss curves + scatter plots
```

---

## cv -- K-fold cross-validation

Train one GNN per fold, evaluate each fold on experimental data, and write a JSON summary
identifying the best fold by validation loss.

```yaml
mode: cv
n_folds: 5
split_method: random   # random | butina
run_evaluation: true
```

Output:

```
cebe_gnn_cv_results/
  models/   {model_id}_fold{N}.pth
  outputs/  per-fold metrics
  pngs/     loss curves + scatter plots
  {model_id}_cv_summary.json
```

---

## param -- Hyperparameter search

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

Each configuration in the grid may produce a different `model_id` when hyperparameters
like `feature_keys` or `hidden_channels` are varied.

Output:

```
cebe_gnn_param_results/
  models/   {search_id}_{model_id}_fold{fold}_{config_id}.pth
  outputs/  per-config metrics
  pngs/     per-config plots
  {search_id}_{model_id}_param_summary.json
```

---

## evaluate -- Evaluate a saved model

Load a previously trained `.pth` model and evaluate it on the built-in experimental
CEBE dataset (`data/processed/gnn_exp_cebe_data.pt`). The fold number is inferred
automatically from the filename.

The `model_id` used for output filenames is derived from the `model_path` filename
(minus the `.pth` extension). The architecture fields (`feature_keys`, `layer_type`,
`hidden_channels`, `n_layers`) must match the values used during training.

```yaml
mode: evaluate
model_path: cebe_gnn_train_results/models/cebe_gnn_035_random_EQ3_h64_fold3.pth
feature_keys: '035'
layer_type: EQ
hidden_channels: 64
n_layers: 3
```

Output: `cebe_gnn_evaluate_results/outputs/` and `cebe_gnn_evaluate_results/pngs/`

---

## predict -- Predict on new molecules

Run inference on a user-specified directory of `.xyz` files using a saved model.
No pre-processing is needed -- molecular graphs are built on the fly.

The `model_id` is derived from the `model_path` filename, same as in evaluate mode.
Architecture fields must match training.

```yaml
mode: predict
model_path: cebe_gnn_train_results/models/cebe_gnn_035_random_EQ3_h64_fold3.pth
predict_dir: my_molecules/
feature_keys: '035'
layer_type: EQ
hidden_channels: 64
n_layers: 3
```

Each file in `predict_dir/` should be a standard XYZ file named `molecule_name.xyz`.

> **Note:** The model is trained exclusively on carbon 1s CEBEs. Predictions for
> non-carbon atoms (H, N, O, F) are not meaningful and are marked with `*` in the
> output labels file. The numeric `_results.txt` file contains carbon predictions only.

Output:

| File | Description |
|------|-------------|
| `cebe_gnn_predict_results/outputs/{model_id}_fold{fold}_labels.txt` | Per-atom predictions (all elements) |
| `cebe_gnn_predict_results/outputs/{model_id}_fold{fold}_results.txt` | Numeric carbon predictions only |
