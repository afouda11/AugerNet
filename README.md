# AugerNet

GNN predictions of carbon 1s core-electron binding energies (CEBEs) from molecular geometry.

Given a set of `.xyz` files, AugerNet builds molecular graphs, encodes atomic environments using a configurable set of node features (SkipAtom embeddings, atomic binding energies, electronegativity scores, etc.), and predicts per-atom C 1s CEBEs using an equivariant or invariant message-passing neural network.

A future release will include GNN predictions of Auger-electron spectroscopy (AES) and CNN classifications of local bond environments from AES spectra augmented with CEBEs.

## Installation

Requires Python >= 3.10 and [conda](https://docs.conda.io/en/latest/miniconda.html).

```bash
# Clone the repository
git clone https://github.com/afouda11/AugerNet.git
cd AugerNet

# Create the conda environment (installs all dependencies + the package)
conda env create -f environment.yml
conda activate augernet
```

The `environment.yml` installs all required dependencies (PyTorch, PyTorch Geometric,
RDKit, scikit-learn, SkipAtom, etc.) and the `augernet` package itself in editable mode.

> **Note:** The provided `environment.yml` targets macOS (Apple Silicon / MPS).
> For Linux with CUDA, replace the PyTorch pip lines with the appropriate versions
> from the [PyTorch install guide](https://pytorch.org/get-started/locally/) and
> the [PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### uv alternative

`uv` is a fast Python package manager; see their documentation [here](https://docs.astral.sh/uv/).

```bash
# Install uv - https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install AugerNet Python package dependencies
uv sync
```

> You can use `uv` to run the Python commands by prefixing them with `uv run`.
> `uv run` executes commands in an isolated virtual environment with all required
> dependencies. See the [documentation](https://docs.astral.sh/uv/reference/cli/#uv-run).

## Quick Start

All runs are controlled by a single YAML config file.
Example configs are provided in `examples/gnn_cebe_configs/`.

```bash
# With conda
python -m augernet --config examples/gnn_cebe_configs/train.yml

# With uv
uv run python -m augernet --config examples/gnn_cebe_configs/train.yml
```

## Run Modes

AugerNet supports five run modes, set via `mode:` in the YAML config.

### train -- Train a single model

Train one GNN on a single k-fold split with optional evaluation on
experimental data.

```yaml
mode: train
train_fold: 3          # which fold to train (1-indexed)
n_folds: 5
run_evaluation: true   # evaluate on experimental data after training
run_unit_tests: true   # check permutation/rotation invariance
```

Output is written to `cebe_gnn_train_results/` in the working directory,
with subdirectories `models/`, `outputs/`, and `pngs/`.

### cv -- K-fold cross-validation

Train one GNN per fold, evaluate each, and write a JSON summary with
per-fold metrics.

```yaml
mode: cv
n_folds: 5
split_method: random   # random | butina
run_evaluation: true
```

Output is written to `cebe_gnn_cv_results/` with the same subdirectory
layout plus a `{model_id}_cv_summary.json` in the top-level results
directory.

### param -- Hyperparameter search

Train one fold per configuration from a Cartesian-product grid.
Evaluate every configuration and write a ranked leaderboard to JSON.

```yaml
mode: param
param_grid:
  feature_keys: ['035', '03', '0356']
  learning_rate: [0.0001, 0.0003, 0.001]
  hidden_channels: [48, 64]
  n_layers: [3, 4, 5]
  batch_size: [24, 32]
```

Output is written to `cebe_gnn_param_results/`. A unique `search_id` is
derived from the searched dimensions so that different grid searches
never overwrite each other. For example, searching `feature_keys`
(3 values) and `layer_type` (2 values) yields:

```
search_id = search_feature_keys3_layer_type2
```

Every model and evaluation file is prefixed with `{search_id}_` and
suffixed with `_{config_id}` (e.g. `cfg000`, `cfg001`).

### evaluate -- Evaluate a saved model

Load a previously trained `.pth` model and evaluate it on the built-in
experimental dataset in `data/processed/gnn_exp_cebe_data.pt`.
The fold number is inferred automatically from the filename.

The `model_id` used for output filenames is derived from the `model_path`
filename (minus the `.pth` extension). The architecture fields
(`feature_keys`, `layer_type`, `hidden_channels`, `n_layers`) must match
the values used during training.

```yaml
mode: evaluate
model_path: cebe_gnn_train_results/models/cebe_gnn_035_random_EQ3_h64_fold3.pth
feature_keys: '035'
layer_type: EQ
hidden_channels: 64
n_layers: 3
```

Output is written to `cebe_gnn_evaluate_results/outputs/` and
`cebe_gnn_evaluate_results/pngs/`.

### predict -- Predict on new molecules

Run inference on a user-specified directory of `.xyz` files using a saved
model. No pre-processing is needed -- molecular graphs are built on the
fly.

The `model_id` is derived from the `model_path` filename, same as in
evaluate mode. Architecture fields must match training.

```yaml
mode: predict
model_path: cebe_gnn_train_results/models/cebe_gnn_035_random_EQ3_h64_fold3.pth
predict_dir: my_molecules/
feature_keys: '035'
layer_type: EQ
hidden_channels: 64
n_layers: 3
```

Each file in `predict_dir/` should be a standard XYZ file named
`molecule_name.xyz`.

> **Note:** The model is trained exclusively on carbon 1s CEBEs. Predictions
> for non-carbon atoms (H, N, O, F) are not meaningful and are marked with
> `*` in the output labels file. The numeric `_results.txt` file contains
> carbon predictions only.

Output is written to `cebe_gnn_predict_results/outputs/`.

## Configuration Reference

All options are set in the YAML config file. See `examples/gnn_cebe_configs/`
for complete examples.

### Identity

| Field   | Default    | Description                            |
| ------- | ---------- | -------------------------------------- |
| `model` | `cebe-gnn` | Model type (only `cebe-gnn` currently) |
| `mode`  | `train`    | Run mode: `cv` / `train` / `param` / `evaluate` / `predict` |

### Features

Node features are selected by a compact key string. Each digit refers to a
feature from the catalog below. For example, `'035'` selects SkipAtom-200 +
atomic BE + e-score.

| Key | Name           | Dim | Description                             |
| --- | -------------- | --- | --------------------------------------- |
| 0   | `skipatom_200` | 200 | SkipAtom atom-type embedding            |
| 1   | `skipatom_30`  | 30  | SkipAtom atom-type embedding (compact)  |
| 2   | `onehot`       | 5   | Element one-hot (H, C, N, O, F)         |
| 3   | `atomic_be`    | 1   | Isolated-atom 1s binding energy         |
| 4   | `mol_be`       | 1   | Molecular CEBE for C, atomic for others |
| 5   | `e_score`      | 1   | Electronegativity-difference score      |
| 6   | `env_onehot`   | ~8  | Carbon-environment one-hot              |
| 7   | `morgan_fp`    | 256 | Per-atom Morgan fingerprint (ECFP2)     |

```yaml
feature_keys: '035'        # SkipAtom-200 + atomic_be + e_score
```

### GNN Architecture

| Field             | Default | Description                            |
| ----------------- | ------- | -------------------------------------- |
| `layer_type`      | `EQ`    | `EQ` (equivariant) or `IN` (invariant) |
| `hidden_channels` | `64`    | Hidden channel width                   |
| `n_layers`        | `3`     | Number of message-passing layers       |

### Training

| Field                | Default  | Description                        |
| -------------------- | -------- | ---------------------------------- |
| `num_epochs`         | `300`    | Maximum training epochs            |
| `patience`           | `30`     | Early-stopping patience            |
| `batch_size`         | `24`     | Mini-batch size                    |
| `learning_rate`      | `0.001`  | Peak learning rate                 |
| `optimizer_type`     | `adamw`  | Optimizer                          |
| `weight_decay`       | `5e-4`   | L2 regularisation                  |
| `gradient_clip_norm` | `0.5`    | Max gradient norm for clipping     |
| `warmup_epochs`      | `10`     | Linear LR warmup epochs           |
| `min_lr`             | `1e-7`   | Minimum learning rate              |
| `scheduler_type`     | `cosine` | LR scheduler: `cosine` / `onecycle` |
| `dropout`            | `0.1`    | Dropout between message-passing layers |
| `random_seed`        | `42`     | Random seed for reproducibility    |

### Cross-Validation

| Field            | Default  | Description                                   |
| ---------------- | -------- | --------------------------------------------- |
| `n_folds`        | `5`      | Number of CV folds                            |
| `train_fold`     | `3`      | Which fold to use for `train` / `param` modes |
| `split_method`   | `random` | `random` / `butina`                           |
| `run_evaluation` | `true`   | Evaluate on experimental data after each fold |
| `exp_split`      | `both`   | Experimental data subset: `all` / `val` / `eval` / `both` |
| `run_unit_tests` | `false`  | Check permutation/rotation invariance         |

### Evaluate / Predict

| Field         | Default | Description                                                |
| ------------- | ------- | ---------------------------------------------------------- |
| `model_path`  | `''`    | Path to a saved `.pth` model (relative to cwd or absolute) |
| `predict_dir` | `''`    | Directory of `.xyz` files for `predict` mode               |

## Output File Naming

All output files use a `model_id` stem. For train, cv, and param modes the
stem is constructed from the config fields:

```
cebe_gnn_{feature_keys}_{split_method}_{layer_type}{n_layers}_h{hidden_channels}
```

For example: `cebe_gnn_035_random_EQ3_h64`

For evaluate and predict modes the `model_id` is derived from the
`model_path` filename (minus the `.pth` extension).

In param search mode, each configuration in the grid may produce a
different `model_id` when hyperparameters like `feature_keys` or
`hidden_channels` are varied.

### Train / CV mode

Files produced per fold:

| File                                | Description                             |
| ----------------------------------- | --------------------------------------- |
| `{model_id}_fold{fold}.pth`         | Saved model weights                     |
| `{model_id}_fold{fold}_loss.png`    | Training/validation loss curves         |
| `{model_id}_fold{fold}_loss.txt`    | Epoch-level loss data (tab-separated)   |
| `{model_id}_fold{fold}_scatter.png` | Predicted vs experimental scatter plot  |
| `{model_id}_fold{fold}_labels.txt`  | Per-atom predicted and true CEBEs       |
| `{model_id}_fold{fold}_results.txt` | Numeric predicted vs true (carbon only) |
| `{model_id}_cv_summary.json`        | Cross-validation summary (cv mode)      |

### Param search mode

A unique `search_id` is built from the searched dimensions
(e.g. `search_feature_keys3_layer_type2`). Each configuration receives a
`config_id` (`cfg000`, `cfg001`, ...). All files are prefixed with
`{search_id}_` and model files are suffixed with `_{config_id}` so that
different searches never overwrite each other.

| File                                                        | Description             |
| ----------------------------------------------------------- | ----------------------- |
| `{search_id}_{model_id}_fold{fold}_{config_id}.pth`         | Saved model weights     |
| `{search_id}_{model_id}_fold{fold}_{config_id}_loss.png`    | Loss curves             |
| `{search_id}_{model_id}_fold{fold}_{config_id}_scatter.png` | Scatter plot            |
| `{search_id}_{model_id}_param_summary.json`                 | Ranked leaderboard JSON |

## Data Preparation

Pre-processed data files are provided in `data/processed/`. To regenerate
them from the raw XYZ and CEBE files in `data/raw/`:

```bash
python scripts/prepare_data.py
```

## Artifact Generation

After running cross-validation, use `scripts/export_best_model.py` to
identify the best fold and copy its weights, plots, and config into the
tracked `artifacts/` directory for release.

```bash
# From the repo root (uses default CV results):
uv run python scripts/export_best_model.py

# Point at a specific results directory:
uv run python scripts/export_best_model.py --results-dir cebe_gnn_param_results

# Overwrite previously exported artifacts:
uv run python scripts/export_best_model.py --overwrite
```

This produces:

```
artifacts/
  config/
    cv.yml                                    # training config used
  model_weights/
    {model_id}_fold{best_fold}.pth            # best-fold model weights
  plots/
    {model_id}_fold{best_fold}_loss.pdf       # loss curve (print-quality)
    {model_id}_fold{best_fold}_loss.png       # loss curve
    {model_id}_fold{best_fold}_scatter.png    # predicted vs experimental
```

The script exits non-zero if any source file is missing. The `artifacts/`
directory is tracked in git and can be committed and attached to a GitHub
release.

`artifacts/data_manifest.yml` records the Zenodo DOI and SHA-256 checksums
for all data files. Update the `doi` and `url` fields before publishing a
release. To verify data integrity after downloading:

```bash
shasum -a 256 data/processed/*.pt data/raw/*.tar.gz
```

## Project Structure

```
AugerNet/
  src/
    augernet/                       # Python package (src layout)
      __init__.py                   # Sets data paths
      __main__.py                   # CLI entry point
      config.py                     # YAML -> AugerNetConfig dataclass
      train_driver.py               # Mode dispatch, CV, param search
      backend_cebe.py               # CEBE model hooks (data, train, eval, predict)
      feature_assembly.py           # Runtime feature selection and scaling
      gnn_train_utils.py            # MPNN model, train loop, unit tests
      build_molecular_graphs.py     # XYZ -> PyG graphs
      eneg_diff.py                  # Electronegativity scoring
      evaluation_scripts/
        evaluate_cebe_model.py      # Evaluation plots and metrics
  scripts/
    prepare_data.py                 # Regenerate processed datasets from raw
    export_best_model.py            # Export best CV fold to artifacts/
  examples/
    gnn_cebe_configs/               # Example YAML configs for each run mode
      train.yml
      cv.yml
      param.yml
      evaluate.yml
      predict.yml
  data/
    raw/                            # Raw XYZ + CEBE files
    processed/                      # Pre-built PyG datasets
  artifacts/                        # Release artifacts (tracked in git)
    data_manifest.yml               # Zenodo DOI + SHA-256 checksums
    config/                         # Training config used for release model
    model_weights/                  # Best-fold model weights (.pth)
    plots/                          # Diagnostic plots (loss curves, scatter)
  environment.yml                   # Conda environment specification
  pyproject.toml
  README.md
```

