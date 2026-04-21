"""
Evaluate CEBE GNN Model
    
Imported by train_driver.py called automatically after training
   when ``run_evaluation: true``:

       from evaluate_cebe_model import run_evaluation
"""

import os
import sys
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

    # Build loss_stem: same as file_stem but with expval/expeval scrubbed from the
    # prefix so the loss plot has a consistent name regardless of which exp split
    # is being evaluated. Any legitimate param-search prefix is preserved.
    if param_file_prefix is not None:
        loss_prefix = param_file_prefix
        for token in ('_expval', '_expeval', 'expval', 'expeval'):
            loss_prefix = loss_prefix.replace(token, '')
        loss_prefix = loss_prefix.strip('_') or None
    else:
        loss_prefix = None

    loss_stem = f"{model_id}_fold{fold}" if fold is not None else model_id
    if config_id is not None:
        loss_stem = f"{loss_stem}_{config_id}"
    if loss_prefix:
        loss_stem = f"{loss_prefix}_{loss_stem}"

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

        loss_plot_path = os.path.join(png_dir, f"{loss_stem}_loss.png")
        loss_pdf_path  = os.path.join(png_dir, f"{loss_stem}_loss.pdf")
        fig.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        fig.savefig(loss_pdf_path, bbox_inches='tight')
        print(f"Loss curves saved to: {loss_plot_path}")
        plt.close(fig)

        # Write raw loss data to a text file
        loss_txt_path = os.path.join(output_dir, f"{loss_stem}_loss.txt")
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

        atomic_be = data.atomic_be_eV.cpu().numpy() if isinstance(data.atomic_be_eV, torch.Tensor) else np.array(data.atomic_be_eV)

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
