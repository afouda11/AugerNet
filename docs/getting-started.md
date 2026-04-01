# Getting Started

## Installation

Requires Python >= 3.10. Install with [conda](https://docs.conda.io/en/latest/miniconda.html)
or [uv](https://docs.astral.sh/uv/).

```bash
# Clone the repository
git clone https://github.com/afouda11/AugerNet.git
cd AugerNet

# Create the conda environment (installs all dependencies + the package)
conda env create -f environment.yml
conda activate augernet
```

### uv alternative

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync
```

!!! note
    The provided `environment.yml` targets macOS (Apple Silicon / MPS).
    For Linux with CUDA, replace the PyTorch pip lines with the appropriate versions
    from the [PyTorch install guide](https://pytorch.org/get-started/locally/) and the
    [PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Quick Start

All runs are controlled by a single YAML config file. Example configs are provided in
`examples/gnn_cebe_configs/`.

```bash
# With conda
python -m augernet --config examples/gnn_cebe_configs/train.yml

# With uv
uv run python -m augernet --config examples/gnn_cebe_configs/train.yml
```

## Data Preparation

Pre-processed PyG datasets are provided in `data/processed/` (downloaded separately from
Zenodo — see `artifacts/data_manifest.yml`). To regenerate them from the raw XYZ and
CEBE files in `data/raw/`:

```bash
python scripts/prepare_data.py
```

## Project Structure

```
AugerNet/
├── src/
│   └── augernet/                # Python package (src layout)
│       ├── __init__.py
│       ├── __main__.py          # CLI entry point
│       ├── config.py            # YAML -> AugerNetConfig dataclass
│       ├── train_driver.py      # Mode dispatch, CV, param search
│       ├── backend_cebe.py      # CEBE model hooks
│       ├── feature_assembly.py  # Runtime feature selection and scaling
│       ├── gnn_train_utils.py   # MPNN model, train loop, unit tests
│       ├── build_molecular_graphs.py
│       └── evaluation_scripts/
│           └── evaluate_cebe_model.py
├── scripts/
│   ├── prepare_data.py
│   └── export_best_model.py
├── examples/
│   └── gnn_cebe_configs/        # Example YAML configs for each run mode
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
└── pyproject.toml
```
