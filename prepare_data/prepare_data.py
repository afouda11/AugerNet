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

import sys
import os
import warnings
import argparse
import traceback
import numpy as np
from pathlib import Path
from collections import Counter

# Suppress RDKit deprecation warnings BEFORE importing RDKit
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*MorganGenerator.*')

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch_geometric.data import InMemoryDataset

from augernet import build_molecular_graphs as bmg
from augernet import build_auger_cnn as bac
from augernet import carbon_dataframe as cdf
from augernet import carbon_environment as ce
#only for cnn, need to check if still needed
from augernet.stratified_split import print_aromatic_sanity_check_from_names


# =============================================================================
# CONFIGURATION
# =============================================================================

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
# ONLY APPLIED IN AUGER CNN GENERATION: for env assignment analysis/debug with xyz files
ANALYSIS_DIR = PROJECT_ROOT / "prepare_data" / "env_analysis_output"

# GNN settings (shared by CEBE + Auger GNN)
ATOM_REP = "SKIPATOM"
FEATURE_SCALE = "MEANSTD"
SCALE_VALUE = 10.0


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
    calc_data = bmg.build_molecular_graphs(
        'cebe', 'calc', str(RAW_DIR), DEBUG=args.debug
    )
    _print_graph_stats(calc_data, "CEBE calc ")
#     ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
#     ce.generate_calc_analysis_outputs(calc_data, str(RAW_DIR), str(ANALYSIS_DIR),
#                                         verbose=args.verbose)
    ce.analyze_carbon_environments(calc_data, verbose=args.verbose)
    _save_collated(calc_data, PROCESSED_DIR / "gnn_calc_cebe_data.pt")

    # --- Experimental evaluation data ---
    print("\n  [1b] Experimental CEBE evaluation data ...")
    exp_data = bmg.build_molecular_graphs(
        'cebe', 'exp', str(RAW_DIR), DEBUG=args.debug
    )
    _print_graph_stats(exp_data, "CEBE exp ")
    ce.analyze_carbon_environments(exp_data, verbose=args.verbose)
    _save_collated(exp_data, PROCESSED_DIR / "gnn_exp_cebe_data.pt")

    # --- Experimental evaluation data ---
    print("\n  [1b] Experimental CEBE evaluation data ...")
    pes_data = bmg.build_molecular_graphs(
        'cebe', 'pes', str(RAW_DIR), DEBUG=args.debug
    )
    _print_graph_stats(pes_data, "CEBE pes ")
    ce.analyze_carbon_environments(pes_data, verbose=args.verbose)
    _save_collated(pes_data, PROCESSED_DIR / "gnn_pes_cebe_data.pt")
    return calc_data, pes_data

# =============================================================================
# ENVIRONMENT INSPECTION TABLE
# =============================================================================

def print_environment_inspection(data_list, pipeline_name, verbose=False):
    """
    Print a per-molecule, per-carbon table of environment assignments.
    Always prints a summary; if verbose, prints full per-atom detail.
    """
    if not data_list:
        return

    print(f"\n  {'─'*70}")
    print(f"  Environment inspection: {pipeline_name}")
    print(f"  {'─'*70}")

    idx_to_name = ce.IDX_TO_CARBON_ENV
    all_labels = []
    n_unmeasured = 0

    for data in data_list:
        name = getattr(data, 'mol_name', getattr(data, 'name', '?'))
        labels = data.carbon_env_labels.tolist()
        mask   = data.node_mask.tolist() if hasattr(data, 'node_mask') else None
        carbon_labels = []
        for j, l in enumerate(labels):
            if l < 0:
                continue
            if mask is not None and mask[j] < 0.5:
                n_unmeasured += 1
                continue
            carbon_labels.append(l)
        all_labels.extend(carbon_labels)

        if verbose:
            env_counts = Counter(carbon_labels)
            env_str = ", ".join(
                f"{idx_to_name.get(idx, '?')}:{cnt}"
                for idx, cnt in sorted(env_counts.items())
            )
            print(f"    {str(name):<20} {len(carbon_labels):>3} C  │ {env_str}")

    # Overall summary
    total = len(all_labels)
    unique = len(set(all_labels))
    counter = Counter(all_labels)
    unmeasured_note = (f" ({n_unmeasured} additional carbons without target values)"
                       if n_unmeasured > 0 else "")
    print(f"\n  Summary: {total} carbons, {unique} unique environments, "
          f"{len(data_list)} molecules{unmeasured_note}")
    top5 = counter.most_common(5)
    for idx, cnt in top5:
        print(f"    {idx_to_name.get(idx, '?'):<30} {cnt:>5} ({100*cnt/total:>5.1f}%)")
    if len(counter) > 5:
        other = total - sum(c for _, c in top5)
        print(f"    {'... others':<30} {other:>5} ({100*other/total:>5.1f}%)")


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
    print("║" + "  UNIFIED DATA PREPARATION — AUGER-NET".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Raw data     : {RAW_DIR}")
    print(f"  Output       : {PROCESSED_DIR}")
    print(f"  Analysis     : {ANALYSIS_DIR}")
    print(f"  Debug mode   : {args.debug}")
    print(f"  Verbose      : {args.verbose}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    cebe_calc_data, cebe_exp_data = prepare_cebe_gnn(args)

    print_environment_inspection(cebe_calc_data, "CEBE GNN — calc")
    print_environment_inspection(cebe_exp_data, "CEBE GNN — exp")

    # ---- Final summary ----
    print("\n" + "=" * 80)
    print("ALL DATA PREPARATION COMPLETE")
    print("=" * 80)

    print("\nReady for training!")


if __name__ == "__main__":
    main()
