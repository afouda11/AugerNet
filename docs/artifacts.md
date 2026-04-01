# Artifact Generation

After a cross-validation or training run, use `scripts/export_best_model.py` to identify
the best model and copy its weights, plots, and config into the tracked `artifacts/`
directory for release.

## Export script

```bash
# Auto-select best fold from CV results (default):
uv run python scripts/export_best_model.py

# Point at any results directory -- type is auto-detected:
uv run python scripts/export_best_model.py --results-dir cebe_gnn_param_results
uv run python scripts/export_best_model.py --results-dir auger_gnn_train_results
uv run python scripts/export_best_model.py --results-dir auger_cnn_cv_results

# Overwrite previously exported artifacts:
uv run python scripts/export_best_model.py --overwrite
```

The script auto-detects the results type from the summary JSON present:


| Summary file found     | Mode  | Selection                  |
| ---------------------- | ----- | -------------------------- |
| `*_cv_summary.json`    | CV    | best fold by val loss      |
| `*_param_summary.json` | Param | best config by val loss    |
| no summary             | Train | single `.pth` in `models/` |


### Output structure

```
artifacts/
├── config/
│   └── cv.yml                                 # training config used
├── model_weights/
│   └── {model_id}_fold{best_fold}.pth         # best-fold weights
└── plots/
    ├── {model_id}_fold{best_fold}_loss.pdf
    ├── {model_id}_fold{best_fold}_loss.png
    └── {model_id}_fold{best_fold}_scatter.png
```

The `artifacts/` directory is tracked in git and attached to GitHub releases automatically
by the release workflow.

---

## Data manifest

`artifacts/data_manifest.yml` records the Zenodo DOI, per-file download URLs, sizes, and
SHA-256 checksums for all data files. Update the `doi`, `record_id`, and `url` fields
after publishing the Zenodo record.

```yaml
zenodo:
  doi: "10.5281/zenodo.XXXXXXX"
  record_id: "XXXXXXX"
  url: "https://doi.org/10.5281/zenodo.XXXXXXX"
  version: "1.0.0"
  title: "AugerNet: CEBE Training Data"
  license: "CC-BY-4.0"
```

### Verifying data integrity

After downloading the data files, verify their checksums against the manifest:

```bash
shasum -a 256 data/processed/*.pt data/raw/*.tar.gz
```

Compare the output against the `sha256` fields in `artifacts/data_manifest.yml`.

---

## GitHub release workflow

The `.github/workflows/release.yml` workflow automates the full release process:

1. **Version bump** — edits `pyproject.toml`, commits, and pushes a `vX.Y.Z` tag
2. **GitHub release** — creates a release with `artifacts/` contents as downloadable assets

Trigger a release manually from the Actions tab, or automatically on push to `main`.