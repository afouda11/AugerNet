# AugerNet

Machine-learning for Auger-electron spectroscopy (AES) and x-ray photoelectron spectroscopy (XPS)

Includes:
1) Equivariant GNN predictions of: 
  a) core-electron binding energies (CEBE) 
  b) Auger-Electron spectra (AES) 

2) CNN classifications of local bond environments (functional groups) from AES spectra augmented with CEBEs

Currently the data for training is not available and will be released when the associated papers come online.
A paper for the GNN CEBE predictions will be released soon and the full GNN CEBE pipeline will become availble.
This will soon be followed by a manuscript on GNN Auger predictions and CNN bond env classification.
Once the manuscripts are online, the software will be fully operational.
The present release contains the routines for data preparation, model training, evaluating and predicting.

AugerNet currently provides **three model types**:

| Model        | Config name  | Task                                               |
|--------------|--------------|-----------------------------------------------------|
| **CEBE GNN** | `cebe-gnn`   | C 1s CEBE prediction from molecular graphs |
| **Auger GNN**| `auger-gnn`  | Auger spectrum prediction (stick or fitted) from molecular graphs |
| **Auger CNN**| `auger-cnn`  | Carbon-environment classification from broadened Auger spectra |

Doc site template undergoing updates can be found at https://afouda11.github.io/AugerNet/
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
Example configs are provided in `examples/`.

```bash
# CEBE GNN — cross-validation
python -m augernet --config examples/gnn_cebe_configs/cv.yml

```

## Run Modes

AugerNet supports five run modes, set via `mode:` in the YAML config.
All three model types share the same mode system.

### train — Train a single model

Train one model on a single k-fold split with optional evaluation.

```yaml
mode: train
model: cebe-gnn        # or auger-gnn or auger-cnn
train_fold: 3
n_folds: 5
run_evaluation: true
```

Output is written to `{model_type}_{mode}_results/` (e.g.
`cebe_gnn_train_results/`, `auger_cnn_train_results/`) with
subdirectories `models/`, `outputs/`, and `pngs/`.

### cv — K-fold cross-validation

Train one model per fold, evaluate each, and write a JSON summary.

```yaml
mode: cv
model: cebe-gnn
n_folds: 5
split_method: random   # random | butina (GNN only)
run_evaluation: true
```

### param — Hyperparameter search

Train one fold per configuration from a Cartesian-product grid.

```yaml
mode: param
model: cebe-gnn
param_grid:
  feature_keys: ['035', '03', '0356']
  learning_rate: [0.0001, 0.0003, 0.001]
  hidden_channels: [48, 64]
  n_layers: [3, 4, 5]
```

A unique `search_id` is derived from the searched dimensions so that
different grid searches never overwrite each other.

### evaluate — Evaluate a saved model

Load a previously trained `.pth` model and evaluate it on experimental data.
Architecture fields must match the values used during training.

```yaml
mode: evaluate
model: cebe-gnn
model_path: cebe_gnn_train_results/models/cebe_gnn_035_random_EQ3_h64_fold3.pth
feature_keys: '035'
layer_type: EQ
hidden_channels: 64
n_layers: 3
```

### predict — Predict on new molecules

Run inference on a directory of `.xyz` files using a saved GNN model.
No pre-processing is needed — molecular graphs are built on the fly.

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

> **Note:** The GNN models are trained on carbon 1s properties. Predictions
> for non-carbon atoms are not meaningful and are marked with `*` in the
> output labels file.

## Model Types

### CEBE GNN (`cebe-gnn`)

Predicts per-atom carbon 1s core-electron binding energies using an
equivariant or invariant message-passing neural network. Input is a set
of `.xyz` molecular geometries converted to PyG graphs with configurable
node features.

```yaml
model: cebe-gnn
feature_keys: '035'
layer_type: EQ
hidden_channels: 64
n_layers: 3
```

**model_id format:** `cebe_gnn_{feature_keys}_{split_method}_{layer_type}{n_layers}_h{hidden_channels}`
Example: `cebe_gnn_035_random_EQ3_h64`

### Auger GNN (`auger-gnn`)

Predicts Auger-electron spectra from molecular graphs. Supports two
spectrum types:

- **`stick`** — predicts separate singlet and triplet stick spectra
  (2 models per fold)
- **`fitted`** — predicts a single combined broadened spectrum
  (1 model per fold)

```yaml
model: auger-gnn
feature_keys: '035'
spectrum_type: stick     # stick | fitted
layer_type: EQ
hidden_channels: 64
n_layers: 3
```

**model_id format:** `auger_gnn_{feature_keys}_{split_method}_{layer_type}{n_layers}_h{hidden_channels}`
Example: `auger_gnn_035_random_EQ3_h64`

### Auger CNN (`auger-cnn`)

Classifies carbon environments from broadened Auger spectra using a 1D
CNN. The input is a per-carbon DataFrame with Gaussian-broadened spectra,
optionally augmented with CEBE shift information.

```yaml
model: auger-cnn
merge_scheme: none       # class merging (none | heteroatom | ...)
broadening_fwhm: 1.6     # Gaussian broadening FWHM in eV
use_augmented: true       # include delta_be augmentation
n_spectrum_points: 731
num_epochs: 500
patience: 40
batch_size: 64
learning_rate: 0.0003
```

The CNN architecture is specified as a dict in the config file:

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

**model_id format:** `auger_cnn_{merge_scheme}`
Example: `auger_cnn_none`

## Configuration Reference

See [docs/configuration.md](docs/configuration.md) for the full reference,
or see the summary tables below.

### Identity

| Field   | Default    | Description                                         |
|---------|------------|-----------------------------------------------------|
| `model` | `cebe-gnn` | Model type: `cebe-gnn` / `auger-gnn` / `auger-cnn` |
| `mode`  | `train`    | Run mode: `cv` / `train` / `param` / `evaluate` / `predict` |

### Node Features (GNN models)

| Key | Name           | Dim | Description                             |
|-----|----------------|-----|-----------------------------------------|
| 0   | `skipatom_200` | 200 | SkipAtom atom-type embedding            |
| 1   | `skipatom_30`  | 30  | SkipAtom atom-type embedding (compact)  |
| 2   | `onehot`       | 5   | Element one-hot (H, C, N, O, F)         |
| 3   | `atomic_be`    | 1   | Isolated-atom 1s binding energy         |
| 4   | `mol_be`       | 1   | Molecular CEBE for C, atomic for others |
| 5   | `e_score`      | 1   | Electronegativity-difference score      |
| 6   | `env_onehot`   | ~8  | Carbon-environment one-hot              |
| 7   | `morgan_fp`    | 256 | Per-atom Morgan fingerprint (ECFP2)     |

### GNN Architecture

| Field             | Default | Description                            |
|-------------------|---------|----------------------------------------|
| `layer_type`      | `EQ`    | `EQ` (equivariant) or `IN` (invariant) |
| `hidden_channels` | `64`    | Hidden channel width                   |
| `n_layers`        | `3`     | Number of message-passing layers       |

### Auger GNN — Spectrum

| Field           | Default  | Description                              |
|-----------------|----------|------------------------------------------|
| `spectrum_type` | `stick`  | `stick` (singlet+triplet) or `fitted`    |
| `max_spec_len`  | `300`    | Maximum number of stick lines            |
| `n_points`      | `731`    | Number of grid points for fitted spectra |
| `fwhm`          | `3.768`  | Broadening FWHM for fitted spectra (eV)  |
| `ke_shift_calc` | `-2.0`   | Kinetic energy shift for calculated data |

### Auger CNN — Specific

| Field               | Default | Description                               |
|---------------------|---------|-------------------------------------------|
| `architecture`      | `{}`    | CNN architecture dict (see above)         |
| `merge_scheme`      | `none`  | Carbon-class merging scheme               |
| `broadening_fwhm`   | `1.6`   | Gaussian broadening FWHM (eV)            |
| `use_augmented`     | `true`  | Include normalised delta_be augmentation  |
| `augmented_scaled`  | `false` | Use scaled delta_be instead               |
| `delta_be_scale`    | `100.0` | Scale factor for delta_be                 |

## Output File Naming

### Output directory naming

Each model type writes to its own results directory:

| Model      | Directory pattern                |
|------------|----------------------------------|
| `cebe-gnn` | `cebe_gnn_{mode}_results/`       |
| `auger-gnn`| `auger_gnn_{mode}_results/`      |
| `auger-cnn`| `auger_cnn_{mode}_results/`      |

Each contains `outputs/` and `pngs/` subdirectories. Train, cv, and param
modes also create a `models/` subdirectory.

### GNN output files (per fold)

| File                                | Description                             |
|-------------------------------------|-----------------------------------------|
| `{model_id}_fold{fold}.pth`         | Saved model weights                     |
| `{model_id}_fold{fold}_loss.png`    | Training/validation loss curves         |
| `{model_id}_fold{fold}_scatter.png` | Predicted vs experimental scatter plot  |
| `{model_id}_fold{fold}_results.txt` | Numeric predicted vs true (carbon only) |
| `{model_id}_cv_summary.json`        | Cross-validation summary (cv mode)      |

### CNN output files (per fold)

| File                                          | Description                     |
|-----------------------------------------------|---------------------------------|
| `{model_id}_fold{fold}.pth`                   | Saved model weights             |
| `training_history_fold{fold}.csv`             | Per-epoch loss and accuracy     |
| `training_plots_fold{fold}.png`               | Training curve plots            |

## Data Preparation

Pre-processed data files are provided in `data/processed/`. To regenerate
them from the raw files in `data/raw/`:

```bash
python scripts/prepare_data.py
```

## Tests

Tests are split into two tiers using pytest markers:

| Tier | Marker | Count | Description |
|------|--------|-------|-------------|
| **Essential** | `@pytest.mark.essential` | ~40 | Fast tests (config, features, parsing)|
| **Full** | `@pytest.mark.full` | ~40 | Slower tests (real molecule graphs, model symmetry) |

Currently only `test_cebe_gnn_config.py` is ran in the CI workflow to reduce run-time

### Running tests

```bash
# Essential tests only 
uv run pytest tests/ -m essential -v --tb=short

# Full suite (all tests)
uv run pytest tests/ -v --tb=short

# Single file
uv run pytest tests/test_cebe_gnn_model.py -v
```

### Test files

| File | What it tests |
|------|---------------|
| `test_cebe_gnn_config.py` | Dataclass defaults, `resolve()` derived fields, YAML loading and validation |
| `test_cebe_gnn_features.py` | Feature key parsing, z-score scaling, node feature assembly, dataset assembly |
| `test_cebe_gnn_graph.py` | XYZ-to-graph pipeline, bond detection, edge attributes, electronegativity scores, carbon environments |
| `test_cebe_gnn_model.py` | MPNN forward pass shapes, translation/rotation invariance, permutation equivariance for both EQ and IN layers |

Graph and model tests use a real molecule (`dsgdb9nsd_133427`) from `tests/test_mol/`
rather than synthetic data. Model symmetry tests verify that CEBE predictions
are invariant to rotation and translation and equivariant to atom reordering
properties required by the physics of the problem.


## Artifact Generation

After running cross-validation, use `scripts/export_best_model.py` to
identify the best fold and copy its weights, plots, and config into the
tracked `artifacts/` directory for release.

```bash
uv run python scripts/export_best_model.py
uv run python scripts/export_best_model.py --results-dir auger_gnn_param_results
uv run python scripts/export_best_model.py --overwrite
```

`artifacts/data_manifest.yml` records the Zenodo DOI and SHA-256 checksums
for all data files. To verify integrity:

```bash
shasum -a 256 data/processed/*.pt data/raw/*.tar.gz
```

## Project Structure

```
AugerNet/
  src/
    augernet/                           # Python package (src layout)
      __init__.py                       # Sets data paths
      __main__.py                       # CLI entry point
      config.py                         # YAML to AugerNetConfig dataclass
      train_driver.py                   # Mode dispatch, CV, param search
      backend_gnn.py                    # Unified GNN backend (cebe-gnn + auger-gnn)
      backend_cnn.py                    # CNN backend (auger-cnn)
      feature_assembly.py               # Runtime feature selection and scaling
      gnn_train_utils.py                # MPNN model, train loop, unit tests
      cnn_train_utils.py                # AugerCNN1D model, CNN trainer
      carbon_dataframe.py               # CarbonDataset for CNN spectra
      carbon_environment.py             # Carbon environment patterns and labels
      class_merging.py                  # Carbon-class merging schemes
      spec_utils.py                     # Spectrum broadening and processing
      build_molecular_graphs.py         # XYZ to PyG graphs
      eneg_diff.py                      # Electronegativity scoring
      evaluation_scripts/
        evaluate_cebe_model.py          # CEBE evaluation plots and metrics
  scripts/
    prepare_data.py                     # Regenerate processed datasets from raw
    export_best_model.py                # Export best CV fold to artifacts/
  examples/
    gnn_cebe_configs/                   # Example YAML configs for CEBE GNN
  tests/
    conftest.py                         # Shared fixtures and markers
    test_mol/                           # Real molecule XYZ data for tests
    test_cebe_gnn_config.py             # Config defaults, resolution, YAML loading
    test_cebe_gnn_features.py           # Feature tags, parsing, scaling, assembly
    test_cebe_gnn_graph.py              # XYZ parsing, graph building, node/edge features
    test_cebe_gnn_model.py              # MPNN construction, forward pass, symmetry tests
  data/
    raw/                                # Raw XYZ + CEBE files
    processed/                          # Pre-built PyG datasets + CNN pickles
  artifacts/                            # Release artifacts (tracked in git)
    data_manifest.yml
    config/
    model_weights/
    plots/
  environment.yml
  pyproject.toml
  README.md
```

