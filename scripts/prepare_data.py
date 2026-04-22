"""
Data Preparation for All Model Pipelines
=================================================

All outputs are written to  data/processed/  under the project root.

Feature Store Approach
------------------------------------
ALL possible node features are computed and stored as separate attributes
during preparation.  ``data.x`` contains only the category_feature.
Feature selection is implemented at training time via
``feature_assembly.assemble_node_features()``.

File naming:
    Molecular graphs for GNN
        gnn_calc_cebe_data.pt           (CEBE calculated training)
        gnn_exp_cebe_data.pt            (CEBE experimental evaluation)
        gnn_sing_calc_auger_data.pt     (Auger singlet calculated)
        gnn_trip_calc_auger_data.pt     (Auger triplet calculated)
        gnn_sing_eval_auger_data.pt     (Auger singlet evaluation)
        gnn_trip_eval_auger_data.pt     (Auger triplet evaluation)

    Carbon atom Pandas dataframe for CNN
        cnn_auger_calc.pkl              (Calculated training)
        cnn_auger_eval.pkl              (Calc. + Exp. evaluation)

CLI Reference
-------------
  python prepare_data.py [OPTIONS]

    --debug              Only generate first 5 in mol_list.txt for testing
    --verbose, -v        Print detailed per-molecule environment tables
    --max_ke             Max KE to normalize auger spec energies by, default 273
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

# SkipAtom pre-trained embeddings are bundled in data/raw/skipatom.tar.gz
# (MIT License, lantunes et al., arXiv:2107.14664)
_SCRIPTS_DIR    = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT      = os.path.dirname(_SCRIPTS_DIR)
SKIPATOM_ARCHIVE = os.path.join(_REPO_ROOT, 'data', 'raw', 'skipatom.tar.gz')
SKIPATOM_DIR     = os.path.join(_REPO_ROOT, 'data', 'raw', 'skipatom')


def _unpack_skipatom():
    """Unpack skipatom.tar.gz into data/raw/ if not already unpacked."""
    if os.path.isdir(SKIPATOM_DIR):
        return
    if not os.path.exists(SKIPATOM_ARCHIVE):
        raise FileNotFoundError(
            f"SkipAtom archive not found: {SKIPATOM_ARCHIVE}\n"
            f"Make sure data/raw/skipatom.tar.gz is present in the repository."
        )
    print("  Unpacking skipatom.tar.gz ...")
    with tarfile.open(SKIPATOM_ARCHIVE, 'r:gz') as tar:
        tar.extractall(os.path.join(_REPO_ROOT, 'data', 'raw'))
    print(f"  Unpacked to: {SKIPATOM_DIR}")


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
    print(f"    {label}: {len(data_list)} mols, "
          f"node_dim={node_dims}, edge_dim={edge_dims}, "
          f"nodes/mol={np.mean(n_nodes):.1f}+/-{np.std(n_nodes):.1f} "
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
    parser.add_argument('--max_ke', default=273, type=int,
                        help='Max KE to normalize auger spec energies by')
    parser.add_argument('--max_spec_len', default=300, type=int,
                        help='Max number of final states in auger spec')
    args = parser.parse_args()

    print("=" * 80)
    print("  DATA PREPARATION -- AUGER-NET")
    print("=" * 80)
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Raw data     : {DATA_RAW_DIR}")
    print(f"  Output       : {DATA_PROCESSED_DIR}")

    # Always unpack skipatom if not already done
    _unpack_skipatom()

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

    # ---- Final summary ----
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)

    print("\nReady for training!")

if __name__ == "__main__":
    main()
