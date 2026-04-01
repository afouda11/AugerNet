# AugerNet

Machine-learning predictions of carbon 1s core-electron binding energies (CEBEs)
and Auger-electron spectra from molecular geometry.

## Overview

AugerNet provides three model types:

| Model        | Config name  | Task                                               |
|--------------|--------------|-----------------------------------------------------|
| **CEBE GNN** | `cebe-gnn`   | Per-atom C 1s CEBE regression from molecular graphs |
| **Auger GNN**| `auger-gnn`  | Auger spectrum prediction (stick or fitted) from molecular graphs |
| **Auger CNN**| `auger-cnn`  | Carbon-environment classification from broadened Auger spectra |

Given a set of `.xyz` files, the GNN models build molecular graphs, encode atomic
environments using a configurable set of node features (SkipAtom embeddings, atomic
binding energies, electronegativity scores, etc.), and predict per-atom properties
using equivariant or invariant message-passing neural networks.

The CNN model classifies carbon environments from 1D Gaussian-broadened Auger spectra,
optionally augmented with CEBE shift information.

## Quick Links

- [Project Repository](https://github.com/afouda11/AugerNet)
- [Getting Started](./getting-started.md)
- [Run Modes](./run-modes.md)
- [Configuration Reference](./configuration.md)
- [Artifact Generation](./artifacts.md)
- [API Reference](./api/index.md)

## Stay in Touch

If you have questions or want to contribute, please open an issue or pull request on GitHub.
