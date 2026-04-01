"""
Evaluate CEBE GNN Model — Standalone Evaluation Script

Can be used in two ways:

1. **Standalone CLI** — evaluate any saved model:

       python evaluate_cebe_model.py /path/to/model.pth \\
           --feature-keys 035 \\
           --layer-type IN --hidden-channels 64 --n-layers 10 \\
           --output-dir eval_outputs --png-dir eval_pngs

2. **Imported by train_driver.py** — called automatically after training
   when ``run_evaluation: true``:

       from evaluate_cebe_model import run_evaluation
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torch_geometric.loader import DataLoader
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Allow imports from project root
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from augernet import gnn_train_utils as gtu
from augernet.feature_assembly import (
    assemble_dataset, compute_feature_tag, describe_features,
    parse_feature_keys,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
PRED_TYPE = 'CEBE'
ATOM_SYMBOLS = ['H', 'C', 'N', 'O', 'F']


# =============================================================================
#  LOAD MODEL
# =============================================================================

def load_model(
    model_path: str,
    in_channels: int,
    edge_dim: int,
    *,
    layer_type: str = 'EQ',
    hidden_channels: int = 64,
    n_layers: int = 3,
    dropout: float = 0.0,
) -> tuple:
    """
    Load a saved CEBE model from a ``.pth`` file.

    Parameters
    ----------
    model_path : str
        Absolute path to the saved ``.pth`` state dict.
    in_channels : int
        Number of input node features (must match training config).
    edge_dim : int
        Number of edge features (must match training config).
    layer_type : str
        ``'EQ'`` (equivariant) or ``'IN'`` (invariant).
    hidden_channels : int
        Hidden channel width.
    n_layers : int
        Number of GNN layers.
    dropout : float
        Dropout probability between message passing layers (must match training config).

    Returns
    -------
    model : torch.nn.Module
    device : torch.device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = gtu.MPNN(
        num_layers=n_layers, emb_dim=hidden_channels,
        in_dim=in_channels, edge_dim=edge_dim,
        out_dim=1, layer_type=layer_type, pred_type=PRED_TYPE,
        dropout=dropout,
    ).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model from: {model_path}  ({n_params:,} params)")
    return model, device


# =============================================================================
#  EVALUATION
# =============================================================================

def run_evaluation(
    model: torch.nn.Module,
    device: torch.device,
    exp_data: list,
    output_dir: str,
    *,
    fold: Optional[int] = None,
    norm_stats_file: str = None,
    png_dir: str,
    train_results: list = None,
    model_id: str = 'cebe',
    config_id: str = None,
    param_file_prefix: str = None,
):
    """
    Evaluate a CEBE model on experimental data.

    Molecule names and sizes are read directly from the processed graph
    Data objects (``data.mol_name`` and ``data.pos``), so no raw XYZ
    files or external molecule lists are required.

    Produces:
      1. Training / validation loss curves (if ``train_results`` provided)
      2. Per-atom predicted vs true CEBE (label_results file)
      3. Per-molecule MAE summary table
      4. Scatter plot with R², MAE, STD

    Parameters
    ----------
    model : torch.nn.Module
        Trained CEBE model (already in eval mode).
    device : torch.device
        Device the model lives on.
    exp_data : list
        List of PyG Data objects for experimental molecules.
    output_dir : str
        Directory for text output files.
    fold : int, optional
        Fold number (used in filenames). ``None`` for standalone evaluation.
    norm_stats_file : str, optional
        Path to ``cebe_normalization_stats.pt``.
    png_dir : str
        Directory for PNG plots.
    train_results : list, optional
        List of ``[epoch, train_loss, val_loss]`` for loss-curve plotting.
    model_id : str
        Unified filename stem (e.g. ``'cebe_gnn_035_random_EQ3_h64'``).
        All output files are named ``{model_id}_fold{fold}_<type>.<ext>``.
    config_id : str, optional
        Param-search config identifier (e.g. ``'cfg003'``).
        Appended to filenames to prevent overwrites during param search.
    param_file_prefix : str, optional
        Prefix prepended to all output filenames (e.g. the ``search_id``
        from a param search). Produces filenames like
        ``{param_file_prefix}_{model_id}_fold{fold}_{config_id}_<type>.<ext>``.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    # Load normalization stats
    norm_stats = torch.load(norm_stats_file, weights_only=False)
    mean = norm_stats['mean']
    std = norm_stats['std']

    # Unified file-stem: [{param_file_prefix}_]{model_id}_fold{fold}[_{config_id}]
    file_stem = f"{model_id}_fold{fold}" if fold is not None else model_id
    if config_id is not None:
        file_stem = f"{file_stem}_{config_id}"
    if param_file_prefix is not None:
        file_stem = f"{param_file_prefix}_{file_stem}"

    print("\n" + "=" * 80)
    print(f"EVALUATION: Testing on experimental data{f'  (fold {fold})' if fold else ''}")
    print("=" * 80)

    # ------------------------------------------------------------------
    #  1) Training / validation loss curves  (publication style)
    # ------------------------------------------------------------------
    if train_results is not None and len(train_results) > 0:
        epochs     = np.array([r[0] for r in train_results])
        train_loss = np.array([r[1] for r in train_results])
        val_loss   = np.array([r[2] for r in train_results])

        best_epoch    = int(np.argmin(val_loss))
        best_val_loss = val_loss[best_epoch]

        fig, ax = plt.subplots(figsize=(5, 4))

        ax.semilogy(epochs, train_loss, color='#0072B2', lw=1.6,
                    label='Train', zorder=3)
        ax.semilogy(epochs, val_loss,   color='#E69F00', lw=1.6,
                    label='Validation', alpha=0.92, zorder=3)

        # Vertical dashed line at best val epoch
        ax.axvline(best_epoch, color='#d62728', ls='--', lw=1.3, alpha=0.8)

        ax.set_xlabel('Epoch', fontsize=10, fontweight='bold')
        ax.set_ylabel('Loss (MSE)', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, framealpha=0.85, loc='lower left')
        ax.tick_params(axis='both', labelsize=9)
        ax.set_xlim(0, epochs[-1] + 2)
        ax.grid(True, alpha=0.3, linewidth=1.0, axis='both', zorder=0)

        loss_plot_path = os.path.join(png_dir, f"{file_stem}_loss.png")
        loss_pdf_path  = os.path.join(png_dir, f"{file_stem}_loss.pdf")
        fig.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        fig.savefig(loss_pdf_path, bbox_inches='tight')
        print(f"Loss curves saved to: {loss_plot_path}")
        plt.close(fig)

        # Write raw loss data to a text file
        loss_txt_path = os.path.join(output_dir, f"{file_stem}_loss.txt")
        with open(loss_txt_path, 'w') as f:
            f.write(f"{'epoch':>8s}  {'train_loss':>14s}  {'val_loss':>14s}\n")
            for ep, tl, vl in zip(epochs, train_loss, val_loss):
                f.write(f"{int(ep):>8d}  {tl:>14.8f}  {vl:>14.8f}\n")
        print(f"Loss data saved to:   {loss_txt_path}")

    # ------------------------------------------------------------------
    #  2) Inference on experimental data
    # ------------------------------------------------------------------
    test_loader = DataLoader(exp_data, batch_size=1, shuffle=False)

    all_pred_out = []
    all_true_out = []
    molecule_results = {}

    for i, data in enumerate(test_loader):
        data = data.to(device)
        with torch.no_grad():
            out = model(data)

        pred_out = out.cpu().detach().numpy()
        true_out = data.y.cpu().detach().numpy()

        atomic_be = data.atomic_be.cpu().numpy() if isinstance(data.atomic_be, torch.Tensor) else np.array(data.atomic_be)

        # Original CEBE values (eV) stored at prep time
        has_true_cebe = hasattr(data, 'true_cebe') and data.true_cebe is not None
        if has_true_cebe:
            true_cebe = data.true_cebe.cpu().numpy()

        # Atom symbols from data
        atom_syms = data.atom_symbols
        # If DataLoader wrapped it in another list, unwrap it
        if isinstance(atom_syms, list):
            atom_syms = atom_syms[0]  # Unwrap the outer list
        # Ensure all elements are strings
        atom_syms = [str(s).strip() for s in atom_syms]

        # Get molecule name from graph Data object
        mol_name_raw = data.mol_name
        if isinstance(mol_name_raw, list):
            mol_name_raw = mol_name_raw[0]  # DataLoader wraps strings in list
        mol_name = mol_name_raw.split('.')[0] if '.' in mol_name_raw else mol_name_raw

        # --- Per-atom results for this molecule ---
        mol_atom_rows = []
        for j, val in enumerate(true_out):
            sym = atom_syms[j] if j < len(atom_syms) else '?'
            if val != -1:
                pred_be = atomic_be[j] - ((pred_out[j] * std) + mean)

                if has_true_cebe:
                    true_be_f = float(true_cebe[j])
                else:
                    true_be = atomic_be[j] - ((true_out[j] * std) + mean)
                    true_be_f = float(np.squeeze(true_be))

                pred_be_f = float(np.squeeze(pred_be))
                error = pred_be_f - true_be_f
                mol_atom_rows.append((sym, true_be_f, pred_be_f, error))

                all_pred_out.append(pred_be_f)
                all_true_out.append(true_be_f)
            else:
                mol_atom_rows.append((sym, -1.0, -1.0, -1.0))

        if mol_name not in molecule_results:
            molecule_results[mol_name] = {
                'true': [], 'pred': [],
                'atom_rows': mol_atom_rows,
                'n_atoms': data.pos.size(0),   # total atoms from graph
            }
        molecule_results[mol_name]['true'] = [r[1] for r in mol_atom_rows if r[1] != -1.0]
        molecule_results[mol_name]['pred'] = [r[2] for r in mol_atom_rows if r[1] != -1.0]

    # ------------------------------------------------------------------
    #  3) Save label_results file
    # ------------------------------------------------------------------
    label_path = os.path.join(output_dir, f"{file_stem}_labels.txt")
    with open(label_path, 'w') as out_file:
        out_file.write(f"# CEBE Evaluation Results\n")
        out_file.write(f"# Columns: atom_symbol  true_BE(eV)  pred_BE(eV)  error(eV)\n")
        out_file.write(f"# Non-carbon atoms marked with -1 sentinels\n")
        out_file.write(f"#\n")
        for mol_name, res in molecule_results.items():
            out_file.write(f"# --- {mol_name} ---\n")
            for sym, true_be, pred_be, error in res['atom_rows']:
                if true_be == -1.0:
                    out_file.write(f"{sym:>3s}    {'—':>10s}    {'—':>10s}    {'—':>10s}\n")
                else:
                    out_file.write(f"{sym:>3s}    {true_be:10.4f}    {pred_be:10.4f}    {error:10.4f}\n")
            out_file.write(f"\n")
    print(f"Label results saved to {label_path}")

    # Compact numeric results (carbon atoms only)
    np.savetxt(
        os.path.join(output_dir, f"{file_stem}_results.txt"),
        np.column_stack((all_true_out, all_pred_out)),
    )

    # ------------------------------------------------------------------
    #  4) Per-molecule summary (MAE + STD)
    # ------------------------------------------------------------------
    print(f"\n{'Molecule':<22s} {'MAE (eV)':>10s} {'STD (eV)':>10s} {'N_C':>5s} {'N_atoms':>8s}")
    print("-" * 60)
    for mol_name, res in molecule_results.items():
        true_arr = np.array(res['true'])
        pred_arr = np.array(res['pred'])
        if len(true_arr) > 0:
            errors = np.abs(true_arr - pred_arr)
            mol_mae = np.mean(errors)
            mol_std = np.std(errors) if len(errors) > 1 else 0.0
        else:
            mol_mae = float('nan')
            mol_std = float('nan')

        n_carbon = len(true_arr)

        # Get total molecule size from the processed graph data
        mol_size = res.get('n_atoms', 'N/A')

        print(f"{mol_name:<22s} {mol_mae:10.4f} {mol_std:10.4f} {n_carbon:>5d} {str(mol_size):>8s}")

    # ------------------------------------------------------------------
    #  5) Scatter plot (predicted vs experimental)
    # ------------------------------------------------------------------
    print("\nGenerating evaluation plots...")

    all_true_arr = np.array(all_true_out)
    all_pred_arr = np.array(all_pred_out)

    mae = np.mean(np.abs(all_true_arr - all_pred_arr))
    residuals = all_true_arr - all_pred_arr
    std_res = np.std(residuals)
    r2 = r2_score(all_true_arr, all_pred_arr)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(all_true_arr, all_pred_arr, alpha=0.6, s=80, edgecolors='k', linewidth=1)

    ax.text(
        0.05, 0.95,
        f'R$^{{2}}$ = {r2:.2f}\nMAE = {mae:.2f} eV\nSTD = {std_res:.2f} eV',
        ha='left', va='top', transform=ax.transAxes,
        fontsize=22, verticalalignment='top', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.8),
    )

    max_val = max(np.max(all_true_arr), np.max(all_pred_arr))
    min_val = min(np.min(all_true_arr), np.min(all_pred_arr))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3)

    ax.set_xlabel('Experimental CEBE (eV)', fontsize=20, fontweight='bold')
    ax.set_ylabel('GNN Predicted CEBE (eV)', fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=1.2)
    ax.tick_params(axis='both', which='major', labelsize=24, width=2, length=6)

    plt.tight_layout()

    plot_path = os.path.join(png_dir, f"{file_stem}_scatter.png")
    plt.savefig(plot_path, dpi=600, bbox_inches='tight')
    print(f"Scatter plot saved to: {plot_path}")
    plt.close()

    # ------------------------------------------------------------------
    #  6) Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"  R2 Score:  {r2:.4f}")
    print(f"  MAE:       {mae:.4f} eV")
    print(f"  STD:       {std_res:.4f} eV")
    print("=" * 80)

    return {'r2': r2, 'mae': mae, 'std': std_res}


# =============================================================================
#  PES EVALUATION
# =============================================================================

# Molecule metadata for PES scans
PES_MOLECULES = {
    'ethene':        {'n_structures': 20, 'coord_label': 'Dihedral angle (°)',
                      'n_atoms': 6},
    'fluoromethane': {'n_structures': 20, 'coord_label': 'Bond length (Å)',
                      'n_atoms': 5},
    'methane':       {'n_structures': 20, 'coord_label': 'Bond length (Å)',
                      'n_atoms': 5},
    'methanol':      {'n_structures': 20, 'coord_label': 'Bond length (Å)',
                      'n_atoms': 6},
}


def run_pes_evaluation(
    model: torch.nn.Module,
    device: torch.device,
    pes_data: list,
    *,
    norm_stats_file: str = None,
    output_dir: str = None,
    png_dir: str,
    pes_raw_dir: str,
    model_id: str = 'cebe',
):
    """
    Evaluate a CEBE model on PES (Potential Energy Surface) scan data.

    For each molecule, plots the calculated CEBE and GNN-predicted CEBE
    as a function of the geometric coordinate (dihedral angle or bond
    length).

    Parameters
    ----------
    model : torch.nn.Module
        Trained CEBE model (already in eval mode).
    device : torch.device
        Device the model lives on.
    pes_data : list
        List of 80 PyG Data objects (4 molecules × 20 structures).
    norm_stats_file : str, optional
        Path to ``cebe_normalization_stats.pt``.
    output_dir : str, optional
        Directory for CSV output files.  One CSV per molecule is saved with
        columns ``coordinate, calc_cebe, gnn_cebe``.
    png_dir : str
        Directory for PNG plots.
    pes_raw_dir : str
        Path to ``data/raw/cebe_pes_eval/`` containing ``*_pes.txt`` files.
    model_id : str
        Unified filename stem (e.g. ``'cebe_gnn_035_random_EQ3_h64'``).
    """
    os.makedirs(png_dir, exist_ok=True)

    # Load normalization stats
    norm_stats = torch.load(norm_stats_file, weights_only=False)
    mean = norm_stats['mean']
    std = norm_stats['std']

    print("\n" + "=" * 80)
    print("PES EVALUATION: CEBE along potential energy surfaces")
    print("=" * 80)

    # ── Build a name→graph lookup ────────────────────────────────────────
    graph_by_name = {}
    for g in pes_data:
        graph_by_name[g.name] = g

    # ── Run inference on all molecules ───────────────────────────────────
    #    Skip graphs with isolated nodes (bond dissociation beyond RDKit
    #    cutoff).  These can be included later by increasing the distance
    #    cutoff in build_molecular_graphs.py → DetermineBonds.
    test_loader = DataLoader(pes_data, batch_size=1, shuffle=False)

    predictions = {}   # name → pred_cebe per atom (only carbon)
    n_skipped = 0
    for data in test_loader:
        # Check for isolated nodes (nodes with no edges)
        n_nodes = data.x.size(0)
        nodes_in_edges = set(data.edge_index[0].tolist() + data.edge_index[1].tolist())
        if len(nodes_in_edges) < n_nodes:
            name = data.name[0] if isinstance(data.name, list) else data.name
            true_cebe = data.true_cebe.cpu().numpy().flatten()
            predictions[name] = {
                'pred_cebe': np.full(n_nodes, np.nan),
                'true_cebe': true_cebe,
                'skipped': True,
            }
            n_skipped += 1
            continue

        data = data.to(device)
        with torch.no_grad():
            out = model(data)

        pred_out = out.cpu().numpy().flatten()
        atomic_be = data.atomic_be.cpu().numpy().flatten()
        true_cebe = data.true_cebe.cpu().numpy().flatten()

        # Convert network output (normalised delta) → absolute CEBE
        n_atoms = len(pred_out)
        pred_cebe = np.full(n_atoms, -1.0)
        for j in range(n_atoms):
            if true_cebe[j] != -1.0:
                pred_cebe[j] = atomic_be[j] - ((pred_out[j] * std) + mean)

        name = data.name[0] if isinstance(data.name, list) else data.name
        predictions[name] = {
            'pred_cebe': pred_cebe,
            'true_cebe': true_cebe,
            'skipped': False,
        }

    if n_skipped > 0:
        print(f"  :wSkipped {n_skipped} structures with isolated nodes "
              f"(bond dissociation beyond RDKit cutoff)")

    # ── Plot each molecule ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax_idx, (mol_name, mol_info) in enumerate(PES_MOLECULES.items()):
        ax = axes[ax_idx]
        n_struct = mol_info['n_structures']
        coord_label = mol_info['coord_label']

        # Load geometric coordinate from *_pes.txt
        pes_file = os.path.join(pes_raw_dir, f'{mol_name}_pes.txt')
        if not os.path.exists(pes_file):
            print(f"  PES file not found: {pes_file}")
            ax.set_title(f'{mol_name} (no PES file)')
            continue

        pes_data_arr = np.loadtxt(pes_file)
        coords = pes_data_arr[:, 0]   # geometric coordinate
        calc_cebe_from_file = pes_data_arr[:, 1]  # calculated CEBE

        # Collect predictions for each structure
        pred_cebe_list = []
        true_cebe_list = []
        n_skip_mol = 0
        for i in range(n_struct):
            struct_name = f'{mol_name}_{i:02d}'
            if struct_name not in predictions:
                print(f"  Missing graph: {struct_name}")
                pred_cebe_list.append(np.nan)
                true_cebe_list.append(np.nan)
                continue

            res = predictions[struct_name]

            # Skipped due to isolated nodes → NaN prediction, keep true CEBE
            if res.get('skipped', False):
                carbon_mask = res['true_cebe'] != -1.0
                carbon_true = res['true_cebe'][carbon_mask]
                true_cebe_list.append(np.mean(carbon_true))
                pred_cebe_list.append(np.nan)
                n_skip_mol += 1
                continue

            # Get CEBE for carbon atoms only (exclude −1 sentinels)
            carbon_mask = res['true_cebe'] != -1.0
            carbon_true = res['true_cebe'][carbon_mask]
            carbon_pred = res['pred_cebe'][carbon_mask]

            # For ethene: 2 equivalent carbons → average (should be identical)
            # For others: 1 carbon
            true_cebe_list.append(np.mean(carbon_true))
            pred_cebe_list.append(np.mean(carbon_pred))

        true_cebe_arr = np.array(true_cebe_list)
        pred_cebe_arr = np.array(pred_cebe_list)

        # Compute per-molecule error stats
        valid = ~np.isnan(pred_cebe_arr) & ~np.isnan(true_cebe_arr)
        if valid.sum() > 0:
            errors = np.abs(pred_cebe_arr[valid] - true_cebe_arr[valid])
            mae = np.mean(errors)
            max_err = np.max(errors)
        else:
            mae, max_err = float('nan'), float('nan')

        # Plot
        ax.plot(coords, true_cebe_arr, 'o-', color='tab:blue', linewidth=2,
                markersize=5, label='Calculated')
        ax.plot(coords, pred_cebe_arr, 's--', color='tab:red', linewidth=2,
                markersize=5, label='GNN Predicted')

        ax.set_xlabel(coord_label, fontsize=12)
        ax.set_ylabel('CEBE (eV)', fontsize=12)
        ax.set_title(mol_name.capitalize(), fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        stat_text = f'MAE = {mae:.3f} eV\nMax = {max_err:.3f} eV'
        if n_skip_mol > 0:
            stat_text += f'\n({n_skip_mol} skipped: dissociated)'
        ax.text(
            0.97, 0.05, stat_text,
            ha='right', va='bottom', transform=ax.transAxes,
            fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

        skip_str = f"  ({n_skip_mol} skipped)" if n_skip_mol > 0 else ""
        print(f"  {mol_name:<16s}  MAE = {mae:.4f} eV,  Max error = {max_err:.4f} eV  "
              f"({valid.sum()}/{n_struct} structures){skip_str}")

        # ── Save CSV for this molecule ──────────────────────────────────
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(
                output_dir,
                f"{model_id}_pes_{mol_name}.csv",
            )
            with open(csv_path, 'w') as f:
                f.write("coordinate,calc_cebe,gnn_cebe\n")
                for c, t, p in zip(coords, true_cebe_arr, pred_cebe_arr):
                    f.write(f"{c},{t},{p}\n")
            print(f"    CSV saved: {csv_path}")

    plt.suptitle('CEBE along Potential Energy Surface Scans', fontsize=16, fontweight='bold')
    plt.tight_layout()

    plot_path = os.path.join(png_dir, f"{model_id}_pes.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n  PES plot saved to: {plot_path}")
    plt.close()

    # ── Overall summary ─────────────────────────────────────────────────
    all_true = []
    all_pred = []
    for mol_name, mol_info in PES_MOLECULES.items():
        for i in range(mol_info['n_structures']):
            struct_name = f'{mol_name}_{i:02d}'
            if struct_name in predictions:
                res = predictions[struct_name]
                if res.get('skipped', False):
                    continue
                carbon_mask = res['true_cebe'] != -1.0
                all_true.extend(res['true_cebe'][carbon_mask].tolist())
                all_pred.extend(res['pred_cebe'][carbon_mask].tolist())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    overall_mae = np.mean(np.abs(all_true - all_pred))
    overall_r2 = r2_score(all_true, all_pred)

    n_total = sum(m['n_structures'] for m in PES_MOLECULES.values())
    print(f"\n  Overall PES:  R$^{{2}}$ = {overall_r2:.4f},  MAE = {overall_mae:.4f} eV  "
          f"({n_total - n_skipped}/{n_total} structures evaluated)")
    print("=" * 80)

    return {'r2': overall_r2, 'mae': overall_mae}


# =============================================================================
#  STANDALONE CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved CEBE GNN model on experimental data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with default settings (feature keys '035')
  python evaluate_cebe_model.py train_results/models/cebe_model_fold3_035.pth

  # Evaluate with custom feature keys and architecture
  python evaluate_cebe_model.py /path/to/model.pth \\
      --feature-keys 0356 \\
      --layer-type IN --hidden-channels 64 --n-layers 10

  # Evaluate with custom output directory
  python evaluate_cebe_model.py /path/to/model.pth \\
      --output-dir my_eval/outputs --png-dir my_eval/pngs
        """,
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default="/Users/foudaae/AUGER-NET/cebe_pred/train_results/models/cebe_model_fold3_035_EQ_3.pth",
        help="Path to the saved model .pth file.",
    )

    # ---- Feature configuration ----
    parser.add_argument(
        '--feature-keys', '-fk',
        type=str, default='035',
        help="Feature keys as a compact string (e.g. '035' = skipatom_200 + atomic_be + e_score).",
    )

    # ---- Model architecture ----
    parser.add_argument(
        '--layer-type', '-lt',
        type=str, default='EQ',
        choices=['IN', 'EQ'],
        help="GNN layer type (default: IN).",
    )
    parser.add_argument(
        '--hidden-channels', '-hc',
        type=int, default=64,
        help="Hidden channels (default: 64).",
    )
    parser.add_argument(
        '--n-layers', '-nl',
        type=int, default=3,
        help="Number of GNN layers (default: 10).",
    )

    parser.add_argument(
        '--norm-stats-file',
        type=str, default=None,
        help="Path to cebe_normalization_stats.pt (default: auto-detect in cebe_pred/).",
    )

    # ---- Output directories ----
    parser.add_argument(
        '--output-dir',
        type=str, default=None,
        help="Directory for text outputs (default: eval_results/outputs).",
    )
    parser.add_argument(
        '--png-dir',
        type=str, default=None,
        help="Directory for PNG plots (default: eval_results/pngs).",
    )

    # ---- Optional fold label ----
    parser.add_argument(
        '--fold',
        type=int, default=None,
        help="Fold number to include in output filenames (optional).",
    )

    # ---- Data paths ----
    parser.add_argument(
        '--data-path',
        type=str, default=None,
        help="Path to data/ directory (default: auto-detect from project root).",
    )

    # ---- PES evaluation mode ----
    parser.add_argument(
        '--pes', action='store_true',
        help="Run PES (Potential Energy Surface) evaluation instead of standard evaluation.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Resolve paths -------------------------------------------------------
    data_path = args.data_path or os.path.join(PROJECT_ROOT, 'data')
    pes_raw_dir = os.path.join(data_path, 'raw', 'cebe_pes_eval')
    norm_stats_file = args.norm_stats_file or os.path.join(SCRIPT_DIR, 'cebe_normalization_stats.pt')

    output_dir = args.output_dir or os.path.join(SCRIPT_DIR, 'eval_results', 'outputs')
    png_dir = args.png_dir or os.path.join(SCRIPT_DIR, 'eval_results', 'pngs')

    feature_keys = parse_feature_keys(args.feature_keys)
    feature_tag = compute_feature_tag(feature_keys)

    # Build model_id for consistent filenames (same format as config.py)
    model_id = (
        f"cebe_gnn_{feature_tag}_standalone"
        f"_{args.layer_type}{args.n_layers}_h{args.hidden_channels}"
    )

    mode_label = "PES" if args.pes else "Experimental"

    print("=" * 80)
    print(f"  CEBE MODEL EVALUATION — {mode_label} (standalone)")
    print("=" * 80)
    print(f"  Model:          {args.model_path}")
    print(f"  Model ID:       {model_id}")
    print(f"  Feature keys:   {feature_keys}  ({describe_features(feature_keys)})")
    print(f"  Architecture:   {args.layer_type}, {args.hidden_channels}h, {args.n_layers}L")
    print(f"  Data path:      {data_path}")
    print(f"  Output dir:     {output_dir}")
    print(f"  PNG dir:        {png_dir}")
    if args.pes:
        print(f"  PES raw dir:    {pes_raw_dir}")
    print("=" * 80)

    # ---- Choose dataset to load ----------------------------------------------
    if args.pes:
        data_file = 'gnn_pes_cebe_data.pt'
    else:
        data_file = 'gnn_exp_cebe_data.pt'

    print(f"\nLoading {mode_label.lower()} data from: {data_path}/{data_file}")
    ds = gtu.LoadDataset(data_path, file_name=data_file)
    eval_data = [ds[i] for i in range(len(ds))]
    print(f"Loaded {len(eval_data)} molecules/structures")

    # Assemble features
    print(f"Assembling features {feature_keys}")
    assemble_dataset(eval_data, feature_keys)

    in_channels = eval_data[0].x.size(1)
    edge_dim = eval_data[0].edge_attr.size(1)
    print(f"  x.shape[1] = {in_channels},  edge_attr.shape[1] = {edge_dim}")

    # ---- Load model ----------------------------------------------------------
    model, device = load_model(
        args.model_path,
        in_channels=in_channels,
        edge_dim=edge_dim,
        layer_type=args.layer_type,
        hidden_channels=args.hidden_channels,
        n_layers=args.n_layers,
    )

    # ---- Run evaluation ------------------------------------------------------
    if args.pes:
        run_pes_evaluation(
            model, device, eval_data,
            norm_stats_file=norm_stats_file,
            output_dir=output_dir,
            png_dir=png_dir,
            pes_raw_dir=pes_raw_dir,
            model_id=model_id,
        )
    else:
        run_evaluation(
            model, device, eval_data,
            output_dir=output_dir,
            fold=args.fold,
            norm_stats_file=norm_stats_file,
            png_dir=png_dir,
            train_results=None,
            model_id=model_id,
        )

    print("\n Evaluation Complete")


if __name__ == "__main__":
    main()
