"""
Unified Data Preparation for ALL Model Pipelines
=================================================

Replaces the 3 separate preparation scripts:
  - cebe_pred/prepare_cebe_data.py         → CEBE GNN
  - auger_pred/prepare_auger_data.py       → Auger GNN (singlet + triplet)
  - auger_cnn/prepare_auger_data_cnn.py    → Auger CNN

All outputs are written to  data/processed/  under the project root.
Carbon environments are assigned ONCE per molecule and used consistently
across all pipelines.

Feature Store Approach
------------------------------------
ALL possible node features are computed and stored as separate ``data.feat_*``
attributes during preparation.  ``data.x`` contains only the category_feature.
Feature selection and scaling are deferred to training time via
``feature_assembly.assemble_node_features()``.

File naming:
  gnn_calc_cebe_data.pt           (CEBE calculated training)
  gnn_exp_cebe_data.pt            (CEBE experimental evaluation)
  gnn_sing_calc_auger_data.pt     (Auger singlet calculated)
  gnn_trip_calc_auger_data.pt     (Auger triplet calculated)
  gnn_sing_eval_auger_data.pt     (Auger singlet evaluation)
  gnn_trip_eval_auger_data.pt     (Auger triplet evaluation)

CLI Reference
-------------
  python prepare_data.py [OPTIONS]

Pipeline selection:
  --skip-cebe          Skip CEBE GNN data preparation
  --skip-auger-gnn     Skip Auger GNN data preparation
  --skip-auger-cnn     Skip Auger CNN data preparation

General:
  --debug              Use small data subset for testing
  --verbose, -v        Print detailed per-molecule environment tables

Examples
--------
    # Prepare everything with feature-store approach (default)
    python prepare_data.py

    # Debug mode (small subset)
    python prepare_data.py --debug

Run from:
    /Users/foudaae/AUGER-NET/prepare_data/
"""

import os
import warnings
import argparse
import numpy as np
from collections import Counter

# Suppress RDKit deprecation warnings BEFORE importing RDKit
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*MorganGenerator.*')

import torch
from torch_geometric.data import InMemoryDataset

from augernet import build_molecular_graphs as bmg
from augernet import carbon_environment as ce
from augernet import PROJECT_ROOT, DATA_RAW_DIR, DATA_PROCESSED_DIR

# =============================================================================
# HELPERS
# =============================================================================

def _save_collated(data_list, output_path):
    """Collate and save a list of PyG Data objects."""
    collated, slices = InMemoryDataset.collate(data_list)
    torch.save((collated, slices), output_path)
    print(f"    Saved: {output_path}  ({len(data_list)} molecules)")


def _print_graph_stats(data_list, label):
    """Print compact graph dimension summary."""
    node_dims = set(g.x.shape[1] for g in data_list)
    edge_dims = set(g.edge_attr.shape[1] if g.edge_attr is not None else 0 for g in data_list)
    n_nodes = [g.x.shape[0] for g in data_list]
    print(f"    {label}: {len(data_list)} mols, "
          f"node_dim={node_dims}, edge_dim={edge_dims}, "
          f"nodes/mol={np.mean(n_nodes):.1f}±{np.std(n_nodes):.1f} "
          f"({min(n_nodes)}-{max(n_nodes)})")
    
    # Show stored feature attributes (feature-store)
    g0 = data_list[0]
    feat_attrs = [a for a in sorted(dir(g0)) if a.startswith('feat_')]
    if feat_attrs:
        feat_dims = {a: getattr(g0, a).shape for a in feat_attrs}
        print(f"    Feature store: {', '.join(f'{k}={v}' for k, v in feat_dims.items())}")


# =============================================================================
# PIPELINE 1: CEBE GNN
# =============================================================================

def prepare_cebe_gnn(args):
    """Prepare CEBE GNN data (calc + exp)."""
    print("\n" + "=" * 80)
    print("PIPELINE 1: CEBE GNN")
    print("=" * 80)

    # --- Calculated training data ---
    print("\n  [1a] Calculated CEBE training data ...")
    calc_data = bmg.build_cebe_graphs('calc', 'calc_cebe', DEBUG=args.debug)

    _print_graph_stats(calc_data, "CEBE calc ")

    ce.analyze_carbon_environments(calc_data, verbose=args.verbose)

    _save_collated(calc_data, os.path.join(DATA_PROCESSED_DIR, "gnn_calc_cebe_data.pt"))

    # --- Experimental evaluation data ---
    print("\n  [1b] Experimental CEBE evaluation data ...")
    exp_data = bmg.build_cebe_graphs('exp', 'exp_cebe', DEBUG=args.debug)

    _print_graph_stats(exp_data, "CEBE exp ")

    ce.analyze_carbon_environments(exp_data, verbose=args.verbose)

    _save_collated(exp_data, os.path.join(DATA_PROCESSED_DIR, "gnn_exp_cebe_data.pt"))

    # --- Experimental evaluation data ---
    print("\n  [1b] Experimental CEBE evaluation data ...")
    pes_data = bmg.build_cebe_graphs('pes', 'pes_cebe', DEBUG=args.debug)

    _print_graph_stats(pes_data, "CEBE pes ")

    ce.analyze_carbon_environments(pes_data, verbose=args.verbose)

    _save_collated(pes_data, os.path.join(DATA_PROCESSED_DIR, "gnn_pes_cebe_data.pt"))

# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Unified data preparation for CEBE GNN, Auger GNN, and Auger CNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--debug', action='store_true',
                        help='Use small data subset for testing')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed per-molecule environment tables')
    args = parser.parse_args()

    print("╔" + "═" * 78 + "╗")
    print("║" + "  DATA PREPARATION — AUGER-NET".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Raw data     : {DATA_RAW_DIR}")
    print(f"  Output       : {DATA_PROCESSED_DIR}")
    print(f"  Debug mode   : {args.debug}")
    print(f"  Verbose      : {args.verbose}")

    prepare_cebe_gnn(args)

    # ---- Final summary ----
    print("\n" + "=" * 80)
    print("ALL DATA PREPARATION COMPLETE")
    print("=" * 80)

    print("\nReady for training!")

if __name__ == "__main__":
    main()
