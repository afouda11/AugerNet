# AugerNet

This doc site is currently acting as template and undergoing updates

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

## Quick Links

- [Project Repository](https://github.com/afouda11/AugerNet)
- [Getting Started](./getting-started.md)
- [Run Modes](./run-modes.md)
- [Configuration Reference](./configuration.md)
- [Artifact Generation](./artifacts.md)
- [API Reference](./api/index.md)

## Stay in Touch

If you have questions or want to contribute, please open an issue or pull request on GitHub.
