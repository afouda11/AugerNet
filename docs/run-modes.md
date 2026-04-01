# Run Modes

AugerNet supports five run modes, set via `mode:` in the YAML config.
All three model types (`cebe-gnn`, `auger-gnn`, `auger-cnn`) share the
same mode system.

| Mode       | Description                                        |
|------------|----------------------------------------------------|
| `train`    | Train one model on a single k-fold split           |
| `cv`       | K-fold cross-validation across all folds           |
| `param`    | Cartesian-product hyperparameter search            |
| `evaluate` | Load a saved model and evaluate on experimental data |
| `predict`  | Run inference on new `.xyz` files (GNN only)       |

---

## train — Train a single model

Train one model on a single k-fold split, with optional post-training
evaluation on experimental data.

```yaml
mode: train
model: cebe-gnn        # or auger-gnn or auger-cnn
train_fold: 3          # which fold to train (1-indexed)
n_folds: 5
run_evaluation: true   # evaluate on experimental data after training
run_unit_tests: true   # check permutation/rotation invariance (GNN only)
```

Output is written to `{model_type}_{mode}_results/` in the working directory:

```
cebe_gnn_train_results/
  models/   {model_id}_fold{N}.pth
  outputs/  per-fold metrics
  pngs/     loss curves + scatter plots
```

For CNN:

```
auger_cnn_train_results/
  models/   {model_id}_fold{N}.pth
  outputs/  training history CSVs
  pngs/     training plots
```

---

## cv — K-fold cross-validation

Train one model per fold, evaluate each fold, and write a JSON summary
identifying the best fold by validation loss.

```yaml
mode: cv
model: cebe-gnn
n_folds: 5
split_method: random   # random | butina (GNN only; CNN always uses random)
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

## param — Hyperparameter search

Trains one fold per configuration from a Cartesian-product grid. Evaluates
every configuration and writes a ranked leaderboard to JSON.

### GNN example

```yaml
mode: param
model: cebe-gnn
param_grid:
  feature_keys:    ['035', '03', '0356']
  learning_rate:   [0.0001, 0.0003, 0.001]
  hidden_channels: [48, 64]
  n_layers:        [3, 4, 5]
  batch_size:      [24, 32]
```

### CNN example

```yaml
mode: param
model: auger-cnn
param_grid:
  merge_scheme:    [none, heteroatom]
  broadening_fwhm: [1.2, 1.6, 2.0]
  learning_rate:   [0.0001, 0.0003]
  use_augmented:   [true, false]
```

A unique `search_id` is derived from the searched dimensions so that
different grid searches never overwrite each other. For example,
searching `feature_keys` (3 values) and `layer_type` (2 values) yields:

```
search_id = search_feature_keys3_layer_type2
```

Each configuration in the grid may produce a different `model_id` when
hyperparameters like `feature_keys` or `hidden_channels` are varied.

Output:

```
cebe_gnn_param_results/
  models/   {search_id}_{model_id}_fold{fold}_{config_id}.pth
  outputs/  per-config metrics
  pngs/     per-config plots
  {search_id}_{model_id}_param_summary.json
```

---

## evaluate — Evaluate a saved model

Load a previously trained `.pth` model and evaluate it on experimental data.
The fold number is inferred automatically from the filename.

The `model_id` used for output filenames is derived from the `model_path`
filename (minus the `.pth` extension). Architecture fields must match the
values used during training.

### GNN example

```yaml
mode: evaluate
model: cebe-gnn
model_path: cebe_gnn_train_results/models/cebe_gnn_035_random_EQ3_h64_fold3.pth
feature_keys: '035'
layer_type: EQ
hidden_channels: 64
n_layers: 3
```

### CNN example

```yaml
mode: evaluate
model: auger-cnn
model_path: auger_cnn_train_results/models/auger_cnn_none_fold3.pth
merge_scheme: none
broadening_fwhm: 1.6
```

Output: `{model_type}_evaluate_results/outputs/` and
`{model_type}_evaluate_results/pngs/`

---

## predict — Predict on new molecules

Run inference on a user-specified directory of `.xyz` files using a saved
GNN model. No pre-processing is needed — molecular graphs are built on
the fly.

The `model_id` is derived from the `model_path` filename, same as in
evaluate mode. Architecture fields must match training.

```yaml
mode: predict
model: cebe-gnn
model_path: cebe_gnn_train_results/models/cebe_gnn_035_random_EQ3_h64_fold3.pth
predict_dir: my_molecules/
feature_keys: '035'
layer_type: EQ
hidden_channels: 64
n_layers: 3
```

Each file in `predict_dir/` should be a standard XYZ file named
`molecule_name.xyz`.

> **Note:** The GNN models are trained on carbon 1s properties. Predictions
> for non-carbon atoms (H, N, O, F) are not meaningful and are marked with
> `*` in the output labels file. The numeric `_results.txt` file contains
> carbon predictions only.

> **Note:** Predict mode is not yet implemented for `auger-cnn`.

Output:

| File | Description |
|------|-------------|
| `{model_type}_predict_results/outputs/{model_id}_fold{fold}_labels.txt` | Per-atom predictions (all elements) |
| `{model_type}_predict_results/outputs/{model_id}_fold{fold}_results.txt` | Numeric carbon predictions only |
