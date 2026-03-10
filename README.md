# AugerNet

GNN predictions of carbon 1s core-electron binding energies (CEBEs) from molecular geometry.

Given a set of `.xyz` files, AugerNet builds molecular graphs, encodes atomic environments using a configurable set of node features (SkipAtom embeddings, atomic binding energies, electronegativity scores, etc.), and predicts per-atom C 1s CEBEs using an equivariant or invariant message-passing neural network.

## Installation

Requires Python ≥ 3.9 and [conda](https://docs.conda.io/en/latest/miniconda.html).

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

## Quick Start

All runs are controlled by a single YAML config file.
Example configs are provided in `config_examples/`.

```bash
python -m augernet --config config_examples/cebe_train_default.yml
```

## Run Modes

AugerNet supports five run modes, set via `mode:` in the YAML config.

### `train` — Train a single model

Trains one GNN on a single k-fold split. Optionally evaluates on experimental data afterwards.

```yaml
mode: train
train_fold: 3          # which fold to train (1-indexed)
n_folds: 5
run_evaluation: true   # evaluate on experimental data after training
run_unit_tests: true   # check permutation/rotation invariance
```

**Output:** `train_results/models/`, `train_results/outputs/`, `train_results/pngs/`

### `cv` — K-fold cross-validation

Trains one model per fold, evaluates each, and writes a JSON summary with per-fold metrics.

```yaml
mode: cv
n_folds: 5
split_method: random   # random | stratified | umap | size
```

**Output:** `cv_results/models/`, `cv_results/outputs/`, `cv_results/pngs/`,
`cv_results/{model_id}_cv_summary.json`

### `param` — Hyperparameter search

Trains one fold per configuration from a Cartesian-product grid.
Evaluates every configuration (not just the best) and writes a ranked leaderboard.

```yaml
mode: param
param_grid:
  feature_keys: ['035', '03', '0356']
  learning_rate: [0.0001, 0.0003, 0.001]
  hidden_channels: [48, 64]
  n_layers: [3, 4, 5]
  batch_size: [24, 32]
```

**Output:** `param_results/models/`, `param_results/outputs/`, `param_results/pngs/`,
`param_results/{model_id}_param_summary.json`

### `evaluate` — Evaluate a saved model

Loads a previously trained `.pth` model and evaluates it on the built-in experimental dataset.
Requires `model_path` in the config.

```yaml
mode: evaluate
model_path: train_results/models/cebe_035_random_EQ3_h64_fold3.pth
```

The fold number is inferred automatically from the filename.

**Output:** `evaluate_results/outputs/`, `evaluate_results/pngs/`

### `predict` — Predict on new molecules

Runs inference on a directory of `.xyz` files using a saved model.
No pre-processing is needed — molecular graphs are built on the fly.

```yaml
mode: predict
model_path: train_results/models/cebe_035_random_EQ3_h64_fold3.pth
predict_dir: my_molecules/
```

Each file in `predict_dir/` should be a standard XYZ file named `molecule_name.xyz`.

> **Note:** The model is trained exclusively on carbon 1s CEBEs. Predictions for
> non-carbon atoms (H, N, O, F) are not meaningful and are marked with `*` in the
> output labels file. The numeric `_results.txt` file contains carbon predictions only.

**Output:** `predict_results/outputs/{model_id}_fold{fold}_labels.txt` (per-atom predictions)
and `predict_results/outputs/{model_id}_fold{fold}_results.txt` (numeric carbon values).

## Configuration Reference

All options are set in the YAML config file. See `config_examples/` for complete examples.

### Identity

| Field | Default | Description |
|-------|---------|-------------|
| `model` | `cebe-gnn` | Model type (only `cebe-gnn` currently) |
| `mode` | `train` | Run mode: `cv` \| `train` \| `param` \| `evaluate` \| `predict` |

### Features

Node features are selected by a compact key string. Each digit refers to a feature
from the catalog below. For example, `'035'` selects SkipAtom-200 + atomic BE + e-score.

| Key | Name | Dim | Description |
|-----|------|-----|-------------|
| 0 | `skipatom_200` | 200 | SkipAtom atom-type embedding |
| 1 | `skipatom_30` | 30 | SkipAtom atom-type embedding (compact) |
| 2 | `onehot` | 5 | Element one-hot (H, C, N, O, F) |
| 3 | `atomic_be` | 1 | Isolated-atom 1s binding energy |
| 4 | `mol_be` | 1 | Molecular CEBE for C, atomic for others |
| 5 | `e_score` | 1 | Electronegativity-difference score |
| 6 | `env_onehot` | ~8 | Carbon-environment one-hot |
| 7 | `morgan_fp` | 256 | Per-atom Morgan fingerprint (ECFP2) |

```yaml
feature_keys: '035'        # SkipAtom-200 + atomic_be + e_score
feature_scale: MEANSTD     # MEANSTD | NORM | NONE
```

### GNN Architecture

| Field | Default | Description |
|-------|---------|-------------|
| `layer_type` | `EQ` | `EQ` (equivariant) or `IN` (invariant) |
| `hidden_channels` | `64` | Hidden channel width |
| `n_layers` | `3` | Number of message-passing layers |

### Training

| Field | Default | Description |
|-------|---------|-------------|
| `num_epochs` | `500` | Maximum training epochs |
| `patience` | `50` | Early-stopping patience |
| `batch_size` | `24` | Mini-batch size |
| `learning_rate` | `0.001` | Peak learning rate |
| `optimizer_type` | `adamw` | Optimizer |
| `weight_decay` | `5e-4` | L2 regularisation |
| `scheduler_type` | `cosine` | LR scheduler: `cosine` \| `onecycle` |
| `random_seed` | `42` | Random seed for reproducibility |
| `out_scale` | `MEANSTD` | Output scaling: `MEANSTD` \| `NONE` |

### Evaluate / Predict

| Field | Default | Description |
|-------|---------|-------------|
| `model_path` | `''` | Path to a saved `.pth` model (relative to cwd or absolute) |
| `predict_dir` | `''` | Directory of `.xyz` files for `predict` mode |

## Output File Naming

All output files use a unified `model_id` stem derived from the config:

```
cebe_{feature_keys}_{split_method}_{layer_type}{n_layers}_h{hidden_channels}
```

For example: `cebe_035_random_EQ3_h64`

Files produced per fold:

| File | Description |
|------|-------------|
| `{model_id}_fold{fold}.pth` | Saved model weights |
| `{model_id}_fold{fold}_loss.png` | Training/validation loss curves |
| `{model_id}_fold{fold}_scatter.png` | Predicted vs experimental scatter plot |
| `{model_id}_fold{fold}_labels.txt` | Per-atom predicted and true CEBEs |
| `{model_id}_fold{fold}_results.txt` | Numeric predicted vs true (carbon only) |
| `{model_id}_cv_summary.json` | Cross-validation summary (cv mode) |
| `{model_id}_param_summary.json` | Parameter search leaderboard (param mode) |

## Data Preparation

Pre-processed data files are provided in `data/processed/`. To regenerate
them from the raw XYZ and CEBE files in `data/raw/`:

```bash
cd prepare_data
python prepare_data.py
```

## Project Structure

```
AugerNet/
├── augernet/                    # Python package
│   ├── __init__.py
│   ├── __main__.py              # CLI entry point
│   ├── config.py                # YAML → AugerNetConfig dataclass
│   ├── train_driver.py          # Mode dispatch, CV, param search
│   ├── backend_cebe.py          # CEBE model hooks (data, train, eval, predict)
│   ├── feature_assembly.py      # Runtime feature selection & scaling
│   ├── gnn_train_utils.py       # MPNN model, train loop, unit tests
│   ├── build_molecular_graphs.py  # XYZ → PyG graphs
│   ├── eneg_diff.py             # Electronegativity scoring
│   └── evaluation_scripts/
│       └── evaluate_cebe_model.py  # Evaluation plots & metrics
├── config_examples/             # Example YAML configs
│   ├── cebe_train_default.yml
│   ├── cebe_train_cv_random.yml
│   ├── cebe_evaluate.yml
│   └── cebe_predict.yml
├── data/
│   ├── raw/                     # Raw XYZ + CEBE files
│   └── processed/               # Pre-built PyG datasets
├── prepare_data/
│   └── prepare_data.py          # Data preparation script
├── environment.yml              # Conda environment specification
├── setup.py
└── README.md
```




