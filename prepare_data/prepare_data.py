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
import pandas as pd
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
# CEBE GNN
# =============================================================================

def prepare_cebe_gnn(args):
    """Prepare CEBE GNN data (calc + exp)."""
    print("\n" + "=" * 80)
    print("Preparing CEBE GNN data (calc + exp)")
    print("=" * 80)

    # --- Calculated training data ---
    print("\n Calculated CEBE training data ...")
    calc_data = bmg.build_graphs('calc_cebe', DEBUG=args.debug)

    _print_graph_stats(calc_data, "CEBE calc ")

    ce.analyze_carbon_environments(calc_data, verbose=args.verbose)

    _save_collated(calc_data, os.path.join(DATA_PROCESSED_DIR, "gnn_calc_cebe_data.pt"))

    # --- Experimental evaluation data ---
    print("\n Experimental CEBE evaluation data ...")
    exp_data = bmg.build_graphs('exp_cebe', DEBUG=args.debug)

    _print_graph_stats(exp_data, "CEBE exp ")

    ce.analyze_carbon_environments(exp_data, verbose=args.verbose)

    _save_collated(exp_data, os.path.join(DATA_PROCESSED_DIR, "gnn_exp_cebe_data.pt"))

# =============================================================================
# AUGER GNN
# =============================================================================

def prepare_auger_gnn(args):
    """Prepare Singlet and Triplet Auger GNN data (calc, eval, exp)."""
    print("\n" + "=" * 80)
    print("Preparing  Singlet and Triplet Auger GNN data (calc, eval, exp)")
    print("=" * 80)

    # --- Calculated training data ---
    print("\n Calculated Auger singlet training data...")
    sing_calc_data = bmg.build_graphs('calc_auger', 
                                      auger_spin='singlet', 
                                      auger_max_ke=args.max_ke,
                                      auger_max_spec_len=args.max_spec_len,
                                      DEBUG=args.debug)

    _print_graph_stats(sing_calc_data, "Auger calc singlet")

    ce.analyze_carbon_environments(sing_calc_data, verbose=args.verbose)

    _save_collated(sing_calc_data, os.path.join(DATA_PROCESSED_DIR, 
                                                "gnn_calc_auger_sing_data.pt"))

    print("\n Calculated Auger triplet training data...")
    trip_calc_data = bmg.build_graphs('calc_auger', 
                                      auger_spin='triplet', 
                                      auger_max_ke=args.max_ke,
                                      auger_max_spec_len=args.max_spec_len,
                                      DEBUG=args.debug)

    _print_graph_stats(trip_calc_data, "Auger calc triplet")

    ce.analyze_carbon_environments(trip_calc_data, verbose=args.verbose)

    _save_collated(trip_calc_data, os.path.join(DATA_PROCESSED_DIR, 
                                                "gnn_calc_auger_trip_data.pt"))
    # --- Calc. evaluation data ---
    print("\n Calc evaluation Auger singlet training data...")
    sing_eval_data = bmg.build_graphs('eval_auger', 
                                      auger_spin='singlet', 
                                      auger_max_ke=args.max_ke,
                                      auger_max_spec_len=args.max_spec_len,
                                      DEBUG=args.debug)

    _print_graph_stats(sing_eval_data, "Auger eval singlet")

    ce.analyze_carbon_environments(sing_calc_data, verbose=args.verbose)

    _save_collated(sing_eval_data, os.path.join(DATA_PROCESSED_DIR, 
                                                "gnn_eval_auger_sing_data.pt"))

    print("\n Calc evaluation Auger triplet training data...")
    trip_eval_data = bmg.build_graphs('eval_auger', 
                                      auger_spin='triplet', 
                                      auger_max_ke=args.max_ke,
                                      auger_max_spec_len=args.max_spec_len,
                                      DEBUG=args.debug)

    _print_graph_stats(trip_eval_data, "Auger eval triplet")

    ce.analyze_carbon_environments(trip_eval_data, verbose=args.verbose)

    _save_collated(trip_eval_data, os.path.join(DATA_PROCESSED_DIR, 
                                                "gnn_eval_auger_trip_data.pt"))

# =============================================================================
# AUGER CNN
# =============================================================================

def auger_gnn_to_cnn_dataframe(data_type, sing_pt_path, trip_pt_path, 
                               auger_max_spec_len=300, auger_max_ke=273):
    """
    Extract a per-carbon DataFrame from a saved Auger GNN .pt file.

    Each row is one carbon atom with its stick spectrum, molecular CEBE,
    normalized binding energy shift, and carbon environment class.
    Gaussian broadening is deferred to training time.

    Parameters
    ----------
    pt_path : str
        Path to collated Auger GNN ``(data, slices)`` file.
    auger_max_spec_len : int
        Max number of spectral peaks used during graph building (for reshape).
    auger_max_ke : float
        Max KE used to normalise energies during graph building.

    Returns
    -------
    pd.DataFrame
        One row per carbon atom with columns:
        ``mol_name, smiles, atom_idx, carbon_env_label, true_cebe,
        atomic_be, delta_be, stick_energies, stick_intensities``
    """
    sing_collated, sing_slices = torch.load(sing_pt_path, weights_only=False)
    trip_collated, trip_slices = torch.load(trip_pt_path, weights_only=False)
    dataset = []
    n_mols = len(sing_slices['x']) - 1

    for i in range(n_mols):
        # Node-level slices (for node_mask, carbon_env, atomic_be, etc.)
        ns, ne = int(sing_slices['x'][i]), int(sing_slices['x'][i + 1])
        n_atoms = ne - ns

        mol_name = sing_collated.mol_name[i]
        smiles = sing_collated.smiles[i]
        node_mask = sing_collated.node_mask[ns:ne]
        carbon_env = sing_collated.carbon_env_labels[ns:ne]
        true_cebe = sing_collated.true_cebe[ns:ne]
        atomic_be = sing_collated.atomic_be[ns:ne]

        # y slices — y was stored as view(-1, 1), so each molecule has
        # n_atoms * max_spec_len * 2 rows in collated.y
        ys, ye = int(sing_slices['y'][i]), int(sing_slices['y'][i + 1])
        sing_y_flat = sing_collated.y[ys:ye].view(-1)
        sing_y_2d = sing_y_flat.view(n_atoms, auger_max_spec_len * 2)
        # Undo the transpose+reshape: (N, 2*L) → (N, L, 2)
        sing_y_spec = sing_y_2d.view(n_atoms, 2, auger_max_spec_len).transpose(1, 2)

        tys, tye = int(trip_slices['y'][i]), int(trip_slices['y'][i + 1])
        trip_y_flat = trip_collated.y[tys:tye].view(-1)
        trip_y_2d = trip_y_flat.view(n_atoms, auger_max_spec_len * 2)
        # Undo the transpose+reshape: (N, 2*L) → (N, L, 2)
        trip_y_spec = trip_y_2d.view(n_atoms, 2, auger_max_spec_len).transpose(1, 2)

        if data_type == 'eval':
            exp_spec_path = os.path.join(DATA_RAW_DIR, 'eval_auger', f"{mol_name}_exp.txt")
            exp_spec      = np.loadtxt(exp_spec_path)

        for j in range(n_atoms):
            if node_mask[j].item() < 0.5:
                continue  # skip non-carbons

            sing_spec = sing_y_spec[j]                      # (max_spec_len, 2)
            trip_spec = trip_y_spec[j]                      # (max_spec_len, 2)
            # Remove zero-padding rows
            sing_valid = sing_spec[:, 1].abs() > 0
            sing_spec_valid = sing_spec[sing_valid]
            trip_valid = trip_spec[:, 1].abs() > 0
            trip_spec_valid = trip_spec[trip_valid]

            # De-normalise energies (spectra were stored as E/max_ke)
            sing_energies = sing_spec_valid[:, 0].numpy() * auger_max_ke
            sing_intensities = sing_spec_valid[:, 1].numpy()
            trip_energies = trip_spec_valid[:, 0].numpy() * auger_max_ke
            trip_intensities = trip_spec_valid[:, 1].numpy()

            cebe_val = true_cebe[j].item()
            be_val = atomic_be[j].item()
            if data_type == 'calc': 
              dataset.append({
                  'mol_name': mol_name,
                  'smiles': smiles,
                  'atom_idx': j,
                  'carbon_env_label': carbon_env[j].item(),
                  'true_cebe': cebe_val,
                  'atomic_be': be_val,
                  'delta_be': be_val - cebe_val,
                  'sing_stick_energies': sing_energies,
                  'sing_stick_intensities': sing_intensities,
                  'trip_stick_energies': trip_energies,
                  'trip_stick_intensities': trip_intensities,
              })
            if data_type == 'eval': 
              dataset.append({
                  'mol_name': mol_name,
                  'smiles': smiles,
                  'atom_idx': j,
                  'carbon_env_label': carbon_env[j].item(),
                  'true_cebe': cebe_val,
                  'atomic_be': be_val,
                  'delta_be': be_val - cebe_val,
                  'sing_stick_energies': sing_energies,
                  'sing_stick_intensities': sing_intensities,
                  'trip_stick_energies': trip_energies,
                  'trip_stick_intensities': trip_intensities,
                  'exp_spec': exp_spec 
              })

    df = pd.DataFrame(dataset)
    print(f"  CNN DataFrame: {len(df)} carbons from {n_mols} molecules")
    return df


def prepare_auger_cnn(args):
    """
    Prepare Auger CNN data by extracting per-carbon DataFrames
    from the already-processed Auger GNN .pt files.

    Saves one pickle per split (calc singlet, calc triplet, eval singlet, eval triplet).
    """
    print("\n" + "=" * 80)
    print("Preparing Auger CNN DataFrames from Auger GNN graphs")
    print("=" * 80)

    gnn_files = {
        'calc_sing': 'gnn_calc_auger_sing_data.pt',
        'calc_trip': 'gnn_calc_auger_trip_data.pt',
        'eval_sing': 'gnn_eval_auger_sing_data.pt',
        'eval_trip': 'gnn_eval_auger_trip_data.pt',
    }

    sing_calc_pt_path = os.path.join(DATA_PROCESSED_DIR, gnn_files['calc_sing'])
    trip_calc_pt_path = os.path.join(DATA_PROCESSED_DIR, gnn_files['calc_trip'])
    sing_eval_pt_path = os.path.join(DATA_PROCESSED_DIR, gnn_files['eval_sing'])
    trip_eval_pt_path = os.path.join(DATA_PROCESSED_DIR, gnn_files['eval_trip'])

    calc_df = auger_gnn_to_cnn_dataframe('calc',
            sing_calc_pt_path, trip_calc_pt_path,
            auger_max_spec_len=args.max_spec_len,
            auger_max_ke=args.max_ke,
        )
    eval_df = auger_gnn_to_cnn_dataframe('eval',
            sing_eval_pt_path, trip_eval_pt_path, 
            auger_max_spec_len=args.max_spec_len,
            auger_max_ke=args.max_ke,
        )

    calc_out_path = os.path.join(DATA_PROCESSED_DIR, f"cnn_auger_calc.pkl")
    eval_out_path = os.path.join(DATA_PROCESSED_DIR, f"cnn_auger_eval.pkl")
    calc_df.to_pickle(calc_out_path)
    eval_df.to_pickle(eval_out_path)
    print(f"    Saved: {calc_out_path}")
    print(f"    Saved: {eval_out_path}")

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
    parser.add_argument('--max_ke', default=273, type=int,
                        help='Max KE to normalize auger spec energies by')
    parser.add_argument('--max_spec_len', default=300, type=int,
                        help='Max number of final states in auger spec')
    args = parser.parse_args()

    print("╔" + "═" * 78 + "╗")
    print("║" + "  DATA PREPARATION — AUGER-NET".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Raw data     : {DATA_RAW_DIR}")
    print(f"  Output       : {DATA_PROCESSED_DIR}")
    print(f"  Debug mode   : {args.debug}")
    print(f"  Verbose      : {args.verbose}")

    #make cebe gnn graphs
    prepare_cebe_gnn(args)

    #make auger gnn graphs
    prepare_auger_gnn(args)

    #make auger cnn dataframes (from gnn graphs)
    prepare_auger_cnn(args)

    # ---- Final summary ----
    print("\n" + "=" * 80)
    print("ALL DATA PREPARATION COMPLETE")
    print("=" * 80)

    print("\nReady for training!")

if __name__ == "__main__":
    main()
