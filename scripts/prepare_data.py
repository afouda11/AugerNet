"""
Data Preparation for All Model Pipelines
=================================================

Outputs are written to  data/processed/  under the project root.

Feature Store Approach
------------------------------------
All possible node features are computed and stored as separate attributes
during preparation.  ``data.x`` contains only the category_feature.
Feature selection is implemented at training time via
``feature_assembly.assemble_node_features()``.

File naming:
    Molecular graphs for GNN
        gnn_calc_cebe_data.pt           (CEBE calculated training)
        gnn_exp_cebe_data.pt            (CEBE experimental evaluation)
        gnn_calc_auger_data.pt          (Auger calculated)
        gnn_eval_auger_data.pt          (Auger evaluation)

    Carbon atom Pandas dataframe for CNN
        cnn_auger_calc.pkl              (Calculated training)
        cnn_auger_eval.pkl              (Calc. + Exp. evaluation)

CLI Reference
-------------
  python prepare_data.py [OPTIONS]

    --from-zenodo       Download pre-built processed data files from Zenodo (skips local graph building)
    --with-raw          Also download and unpack raw data archives from Zenodo 
                        (use with --from-zenodo to also regenerate graphs locally)
    --debug             Only generate first 5 in mol_list.txt for testing
    --verbose, -v       Print detailed per-molecule environment tables
    --max_spec_len       Max number of final states in auger spec, default 300

"""

import os
import warnings
import argparse
import tarfile
import urllib.request
import numpy as np
import pandas as pd
from collections import Counter

# =============================================================================
# ZENODO DOWNLOAD
# =============================================================================

ZENODO_PROCESSED = {
    'gnn_calc_cebe_data.pt': 'https://zenodo.org/records/19688196/files/gnn_calc_cebe_data.pt?download=1',
    'gnn_exp_cebe_data.pt':  'https://zenodo.org/records/19688196/files/gnn_exp_cebe_data.pt?download=1',
}

ZENODO_RAW = {
    'calc_cebe.tar.gz': 'https://zenodo.org/records/19688196/files/calc_cebe.tar.gz?download=1',
    'exp_cebe.tar.gz':  'https://zenodo.org/records/19688196/files/exp_cebe.tar.gz?download=1',
}


def _download(url, dest_path):
    """Download a file from url to dest_path with a progress message."""
    filename = os.path.basename(dest_path)
    if os.path.exists(dest_path):
        print(f"    Already exists, skipping: {filename}")
        return
    print(f"    Downloading {filename} ...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"    Saved: {dest_path}")


def download_processed(processed_dir):
    """Download pre-built processed .pt files from Zenodo."""
    print("\n" + "=" * 80)
    print("Downloading processed data from Zenodo")
    print("=" * 80)
    os.makedirs(processed_dir, exist_ok=True)
    for filename, url in ZENODO_PROCESSED.items():
        _download(url, os.path.join(processed_dir, filename))
    print("\nProcessed data ready in:", processed_dir)


def download_raw(raw_dir):
    """Download raw tar.gz archives from Zenodo and unpack them."""
    print("\n" + "=" * 80)
    print("Downloading raw data from Zenodo")
    print("=" * 80)
    os.makedirs(raw_dir, exist_ok=True)
    for filename, url in ZENODO_RAW.items():
        archive_path = os.path.join(raw_dir, filename)
        _download(url, archive_path)
        folder_name = filename.replace('.tar.gz', '')
        dest_dir = os.path.join(raw_dir, folder_name)
        if os.path.exists(dest_dir):
            print(f"    Already unpacked, skipping: {folder_name}/")
        else:
            print(f"    Unpacking {filename} ...")
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(raw_dir)
            print(f"    Unpacked to: {dest_dir}")
    print("\nRaw data ready in:", raw_dir)

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

def _debug_suffix(filename, debug):
    """Add _debug suffix before extension if debug mode is active."""
    if not debug:
        return filename
    base, ext = os.path.splitext(filename)
    return f"{base}_debug{ext}"


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
    if label in ['CEBE calc ', 'CEBE exp ']:
        y_dim = set(g.cebe_y.shape[1] for g in data_list)
        print(f"    {label}: {len(data_list)} mols, "
            f"node_dim={node_dims}, edge_dim={edge_dims}, "
            f"y_dim={y_dim}, "
            f"({min(n_nodes)}-{max(n_nodes)})")
    if label in ['Auger calc', 'Auger eval']:
        y_cebe_dim = set(g.cebe_y.shape[1] for g in data_list)
        y_sing_dim = set(g.sing_y.shape[1:] for g in data_list)
        y_trip_dim = set(g.trip_y.shape[1:] for g in data_list)
        sing_mask_dim = set(g.sing_mask_bin.shape[1:] for g in data_list)
        trip_mask_dim = set(g.trip_mask_bin.shape[1:] for g in data_list)
        print(f"    {label}: {len(data_list)} mols, "
            f"node_dim={node_dims}, edge_dim={edge_dims}, "
            f"cebe y_dim={y_cebe_dim}, "
            f"sing y_dim={y_sing_dim}, "
            f"trip y_dim={y_trip_dim}, "
            f"sing mask_dim={sing_mask_dim}, "
            f"trip mask_dim={trip_mask_dim}, "
            f"({min(n_nodes)}-{max(n_nodes)})")
    
    # Show stored feature attributes (feature-store)
    g0 = data_list[0]
    from augernet.feature_assembly import FEATURE_NAMES
    known_attrs = set(FEATURE_NAMES.values())
    feat_attrs = [a for a in sorted(dir(g0)) if a in known_attrs]
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

    print("\n Calculated CEBE training data ...")
    calc_data = bmg.build_graphs('calc_cebe', DEBUG=args.debug)

    _print_graph_stats(calc_data, "CEBE calc ")

    ce.analyze_carbon_environments(calc_data, verbose=args.verbose)

    calc_path = _debug_suffix("gnn_calc_cebe_data.pt", args.debug)
    _save_collated(calc_data, os.path.join(DATA_PROCESSED_DIR, calc_path))

    print("\n Experimental CEBE evaluation data ...")
    exp_data = bmg.build_graphs('exp_cebe', DEBUG=args.debug)

    _print_graph_stats(exp_data, "CEBE exp ")

    ce.analyze_carbon_environments(exp_data, verbose=args.verbose)

    exp_path = _debug_suffix("gnn_exp_cebe_data.pt", args.debug)
    _save_collated(exp_data, os.path.join(DATA_PROCESSED_DIR, exp_path))

# =============================================================================
# AUGER GNN
# =============================================================================

def prepare_auger_gnn(args):
    """Prepare Singlet and Triplet Auger GNN data (calc, eval, exp)."""
    print("\n" + "=" * 80)
    print("Preparing  Singlet and Triplet Auger GNN data (calc, eval, exp)")
    print("=" * 80)

    print("\n Calculated Auger training data...")
    calc_data = bmg.build_graphs('calc_auger', 
                                      auger_max_spec_len=args.max_spec_len,
                                      DEBUG=args.debug)

    _print_graph_stats(calc_data, "Auger calc")

    ce.analyze_carbon_environments(calc_data, verbose=args.verbose)

    calc_path = _debug_suffix("gnn_calc_auger_data.pt", args.debug)
    _save_collated(calc_data, os.path.join(DATA_PROCESSED_DIR, calc_path))


    print("\n Calc evaluation Auger training data...")
    eval_data = bmg.build_graphs('eval_auger', 
                                      auger_max_spec_len=args.max_spec_len,
                                      DEBUG=args.debug)

    _print_graph_stats(eval_data, "Auger eval")

    ce.analyze_carbon_environments(eval_data, verbose=args.verbose)

    eval_path = _debug_suffix("gnn_eval_auger_data.pt", args.debug)
    _save_collated(eval_data, os.path.join(DATA_PROCESSED_DIR, eval_path))

# =============================================================================
# AUGER CNN
# =============================================================================

def auger_gnn_to_cnn_dataframe(data_type, pt_path, auger_max_spec_len=300):
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

    Returns
    -------
    pd.DataFrame
        One row per carbon atom with columns:
        ``mol_name, smiles, atom_idx, carbon_env_label, true_cebe,
        atomic_be, delta_be, stick_energies, stick_intensities``
    """
    collated, slices = torch.load(pt_path, weights_only=False)
    dataset = []
    n_mols = len(slices['x']) - 1

    auger_norm_stats_path = os.path.join(DATA_PROCESSED_DIR, 'auger_norm_stats.pt')
    auger_norm_stats = torch.load(auger_norm_stats_path, weights_only=False)
    maxE = float(auger_norm_stats['maxE'])
    maxI = float(auger_norm_stats['maxI'])

    for i in range(n_mols):
        # Node-level slices (for node_mask, carbon_env, atomic_be, etc.)
        ns, ne = int(slices['x'][i]), int(slices['x'][i + 1])
        n_atoms = ne - ns

        mol_name = collated.mol_name[i]
        smiles = collated.smiles[i]
        node_mask = collated.node_mask[ns:ne]
        carbon_env_labels = collated.carbon_env_labels[i]   # list of strings
        carbon_env_indices = collated.carbon_env_indices[ns:ne]  # slice tensor
        true_cebe = collated.true_cebe[ns:ne]
        atomic_be = collated.atomic_be_eV[ns:ne]
        sing_y_spec = collated.sing_y[ns:ne]
        trip_y_spec = collated.trip_y[ns:ne]

        if data_type == 'eval':
            exp_spec_path = os.path.join(DATA_RAW_DIR, 'eval_auger', f"{mol_name}_exp.txt")
            exp_spec      = np.loadtxt(exp_spec_path)

        for j in range(n_atoms):
            if node_mask[j].item() == 0.:
                continue  # skip non-carbons

            sing_spec = sing_y_spec[j]                      # (max_spec_len, 2)
            trip_spec = trip_y_spec[j]                      # (max_spec_len, 2)

            #un-normalize
            sing_energies = sing_spec[:, 0].numpy() * maxE   
            trip_energies = trip_spec[:, 0].numpy() * maxE   
            sing_intensities = sing_spec[:, 1].numpy() * maxI
            trip_intensities = trip_spec[:, 1].numpy() * maxI

            cebe_val = true_cebe[j].item()
            be_val = atomic_be[j].item()
            if data_type == 'calc': 
              dataset.append({
                  'mol_name': mol_name,
                  'smiles': smiles,
                  'atom_idx': j,
                  'carbon_env_label': carbon_env_labels[j],
                  'carbon_env_index': carbon_env_indices[j].item(),
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
                  'carbon_env_label': carbon_env_labels[j],
                  'carbon_env_index': carbon_env_indices[j].item(),
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

    Saves one pickle for calc (singlet + triplet) and eval (singlet + triplet).
    """
    print("\n" + "=" * 80)
    print("Preparing Auger CNN DataFrames from Auger GNN graphs")
    print("=" * 80)

    gnn_files = {
        'calc': _debug_suffix("gnn_calc_auger_data.pt", args.debug),
        'eval': _debug_suffix("gnn_eval_auger_data.pt", args.debug),
    }

    calc_pt_path = os.path.join(DATA_PROCESSED_DIR, gnn_files['calc'])
    eval_pt_path = os.path.join(DATA_PROCESSED_DIR, gnn_files['eval'])

    calc_df = auger_gnn_to_cnn_dataframe('calc',
            calc_pt_path, auger_max_spec_len=args.max_spec_len
        )
    eval_df = auger_gnn_to_cnn_dataframe('eval',
            eval_pt_path, auger_max_spec_len=args.max_spec_len,
        )
    
    calc_out_path = _debug_suffix("cnn_auger_calc.pkl", args.debug)
    eval_out_path = _debug_suffix("cnn_auger_eval.pkl", args.debug)
    calc_df.to_pickle(os.path.join(DATA_PROCESSED_DIR, calc_out_path))
    eval_df.to_pickle(os.path.join(DATA_PROCESSED_DIR, eval_out_path))
    print(f"Saved: {os.path.join(DATA_PROCESSED_DIR, calc_out_path)}")
    print(f"Saved: {os.path.join(DATA_PROCESSED_DIR, eval_out_path)}")

# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Data preparation for CEBE GNN, Auger GNN, and Auger CNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--from-zenodo', action='store_true',
                        help='Download pre-built processed data files from Zenodo '
                             '(skips local graph building)')
    parser.add_argument('--with-raw', action='store_true',
                        help='Also download and unpack raw data archives from Zenodo '
                             '(use with --from-zenodo to also regenerate graphs locally)')
    parser.add_argument('--debug', action='store_true',
                        help='Only generate first 5 in mol_list.txt for testing')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed per-molecule environment tables')
    parser.add_argument('--max_spec_len', default=300, type=int,
                        help='Max number of final states in auger spec')
    args = parser.parse_args()

    print("=" * 80)
    print("  DATA PREPARATION -- AUGER-NET")
    print("=" * 80)
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Raw data     : {DATA_RAW_DIR}")
    print(f"  Output       : {DATA_PROCESSED_DIR}")

    # --from-zenodo: download processed files and optionally raw archives
    if args.from_zenodo:
        download_processed(DATA_PROCESSED_DIR)
        if args.with_raw:
            download_raw(DATA_RAW_DIR)
        print("\nDone. To regenerate graphs from raw data, run without --from-zenodo.")
        return

    # --with-raw alone: download raw archives then build graphs
    if args.with_raw:
        download_raw(DATA_RAW_DIR)

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
