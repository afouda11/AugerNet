"""
Publication Plots — Auger GNN Stick vs Fitted comparison
=========================================================

Generates publication-quality comparison plots (2×2 or 3×2 grids)
of GNN-predicted Auger spectra vs calculated and experimental
reference spectra.

Supports **multiple feature sets** (e.g. ``035`` and ``045``) so that
stick and fitted predictions from different node-feature models can
be compared side-by-side on the same plot.

Reuses the prediction / broadening / PCC helpers from
``augernet.evaluation_scripts.evaluate_auger_model`` so the
comparison methodology is identical to the main evaluation.

Usage
-----
    cd AugerNet/auger_pred
    python publication_plots.py                       # 035 only, 2×2
    python publication_plots.py --layout 3x2          # 035 only, 3×2
    python publication_plots.py --feature-sets 035 045 # both sets, 2×2
    python publication_plots.py --feature-sets 035 045 --layout 3x2
    python publication_plots.py --molecules acetylene benzene butane formamide

Edit the MOLECULES_* lists or pass ``--molecules`` to choose panels.
"""

from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate

# ── project path ----------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from augernet import DATA_DIR, DATA_RAW_DIR
from augernet import gnn_train_utils as gtu
from augernet.feature_assembly import (
    assemble_dataset, parse_feature_keys, describe_features,
)
from augernet.backend import _load_model_from_path

# Import the exact helpers used in the evaluation script
from augernet.evaluation_scripts.evaluate_auger_model import (
    _broaden_sticks,
    _predict_stick,
    _predict_fitted,
    _load_eval_metadata,
    _load_experimental_spectrum,
    _load_all_calc_sticks,
    _compute_pcc,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Default molecule selections (override via --molecules CLI arg)
MOLECULES_2x2 = ["acetylene", "butane", "formamide", "tetrafluoromethane"]

MOLECULES_3x2 = [
    "acetaldehyde", "acetone",   "butane",
    "ethane",      "formamide",    "tetrafluoromethane",
]

# Feature sets to compare (override via --feature-sets CLI arg)
# Each entry produces stick + fitted curves on the same plot.
DEFAULT_FEATURE_SETS = ["035"]          # e.g. ["035", "045"] for both

# ── Model / spectrum parameters (must match training config) ──────────────
MODEL_DIR    = os.path.join(SCRIPT_DIR, "train_results", "models")
MODEL_STEM   = "auger_{fk}_butina_EQ3_h64"   # {fk} is replaced by feature key
FOLD         = 1

LAYER_TYPE   = "EQ"
HIDDEN       = 64
N_LAYERS     = 3
DROPOUT      = 0.1

MAX_SPEC_LEN = 300          # stick output dim (per half)
MAX_KE       = 273
MIN_KE       = 200.0
N_POINTS     = 731          # fitted grid dim
FWHM         = 3.768        # Gaussian broadening width
KE_SHIFT     = -2.0         # theory → experiment offset

# ── Colours per feature set ───────────────────────────────────────────────
#    Each feature set gets a (stick_colour, fitted_colour) pair.
C_EXP   = "k"               # experimental      (black, solid)
C_CALC  = "#2ca02c"         # ab initio calc     (green, solid)

FEATURE_SET_STYLES = {
    # feature_keys: (stick_colour, fitted_colour, stick_ls, fitted_ls)
    "035": ("#ff7f0e", "#1f77b4", ":",  "--"),   # orange dotted / blue dashed
    "045": ("#d62728", "#9467bd", ":",  "--"),   # red dotted / purple dashed
}
# Fallback for any key not in the dict above
_DEFAULT_STYLE = ("#ff7f0e", "#1f77b4", ":", "--")

# ── Output ────────────────────────────────────────────────────────────────
PNG_DIR = os.path.join(SCRIPT_DIR, "pngs")


# ═══════════════════════════════════════════════════════════════════════════════
#  Data & model loading  (parameterised by feature_keys)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_eval_data(feature_keys: str):
    """Load and prepare eval datasets assembled with *feature_keys*.

    Returns fresh copies each time (assemble_dataset is in-place),
    so different feature sets don't clobber each other.
    """
    feature_keys_parsed = parse_feature_keys(feature_keys)

    sing_ds = gtu.LoadDataset(DATA_DIR, file_name="gnn_eval_auger_sing_data.pt")
    trip_ds = gtu.LoadDataset(DATA_DIR, file_name="gnn_eval_auger_trip_data.pt")
    eval_sing = [sing_ds[i] for i in range(len(sing_ds))]
    eval_trip = [trip_ds[i] for i in range(len(trip_ds))]

    # reshape y from (n_atoms*600, 1) → (n_atoms, 600)
    for dlist in (eval_sing, eval_trip):
        for d in dlist:
            n_atoms = d.x.size(0)
            d.y = d.y.view(n_atoms, 600)

    assemble_dataset(eval_sing, feature_keys_parsed)
    assemble_dataset(eval_trip, feature_keys_parsed)

    return eval_sing, eval_trip


def _model_id_for(feature_keys: str) -> str:
    """Build the model-ID stem for a given feature-key string."""
    return MODEL_STEM.format(fk=feature_keys)


def _load_stick_models(sample_data, feature_keys: str):
    """Load singlet + triplet stick models for *feature_keys*."""
    model_id = _model_id_for(feature_keys)
    kw = dict(
        layer_type=LAYER_TYPE, hidden_channels=HIDDEN,
        n_layers=N_LAYERS, dropout=DROPOUT,
        pred_type="AUGER", spectrum_type="stick",
        spectrum_dim=MAX_SPEC_LEN,
    )
    sing_path = os.path.join(MODEL_DIR, f"singlet_{model_id}_fold{FOLD}.pth")
    trip_path = os.path.join(MODEL_DIR, f"triplet_{model_id}_fold{FOLD}.pth")

    if not os.path.exists(sing_path) or not os.path.exists(trip_path):
        print(f"  ⚠ Stick models not found for feature set '{feature_keys}'")
        return None, None, None

    sing_model, device = _load_model_from_path(sing_path, sample_data, **kw)
    trip_model, _      = _load_model_from_path(trip_path, sample_data, **kw)
    return sing_model, trip_model, device


def _load_fitted_model(sample_data, feature_keys: str):
    """Load fitted model for *feature_keys* (returns None tuple if missing)."""
    model_id = _model_id_for(feature_keys)
    fitted_path = os.path.join(MODEL_DIR, f"fitted_{model_id}_fold{FOLD}.pth")
    if not os.path.exists(fitted_path):
        print(f"  ⚠ Fitted model not found for feature set '{feature_keys}'")
        return None, None

    kw = dict(
        layer_type=LAYER_TYPE, hidden_channels=HIDDEN,
        n_layers=N_LAYERS, dropout=DROPOUT,
        pred_type="AUGER", spectrum_type="fitted",
        spectrum_dim=N_POINTS,
    )
    return _load_model_from_path(fitted_path, sample_data, **kw)


def _combine_stick_preds(sing_preds, trip_preds, n_molecules):
    """Merge singlet + triplet stick predictions per molecule.

    Returns ``{mol_idx: np.ndarray(N, 2)}`` with columns [energy, intensity].
    """
    combined = {}
    for mol_idx in range(n_molecules):
        specs = []
        for preds in (sing_preds, trip_preds):
            if mol_idx in preds:
                for a_idx in sorted(preds[mol_idx]):
                    specs.append(preds[mol_idx][a_idx])
        if specs:
            arr = np.row_stack(specs)
            if arr.shape[0] > 0:
                combined[mol_idx] = arr
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def _get_style(fk: str):
    """Return ``(c_stick, c_fit, ls_stick, ls_fit)`` for a feature set."""
    return FEATURE_SET_STYLES.get(fk, _DEFAULT_STYLE)


def _plot_grid(
    mol_names: list[str],
    nrows: int,
    ncols: int,
    *,
    all_stick: dict[str, dict],
    all_fitted: dict[str, dict | None],
    feature_sets: list[str],
    mol_list: list[str],
    c_num: np.ndarray,
    eval_dir: str,
    filename: str = "comparison.png",
    show_vlines: bool = True,
):
    """
    Plot a nrows×ncols grid comparing Exp / Calc / GNN predictions
    for one or more feature sets.

    Parameters
    ----------
    all_stick : dict[str, dict]
        ``{feature_keys: {mol_idx: np.ndarray(N,2)}}``
    all_fitted : dict[str, dict | None]
        ``{feature_keys: {mol_idx: {atom_idx: np.ndarray(n_pts,)}} | None}``
    feature_sets : list[str]
        Ordered list of feature-key strings to plot.
    show_vlines : bool
        If False, suppress vertical stick lines (calc and GNN) for a
        cleaner look.  Default True.
    """
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(9 * ncols, 3.5 * nrows),
        squeeze=False,
    )
    ax = axes.ravel()

    name_to_idx = {n: i for i, n in enumerate(mol_list)}
    energy_grid = np.linspace(MIN_KE, MAX_KE, N_POINTS)
    multi = len(feature_sets) > 1          # label suffix needed?

    pcc_data = []

    for plot_idx, mol_name in enumerate(mol_names):
        if mol_name not in name_to_idx:
            print(f"  ⚠ '{mol_name}' not in eval set — skipping")
            ax[plot_idx].set_visible(False)
            continue

        mol_idx = name_to_idx[mol_name]
        n_c = int(c_num[mol_idx])

        # ── Experimental ──────────────────────────────────────────────
        exp_spec = _load_experimental_spectrum(eval_dir, mol_name)
        exp_base = fit_exp_norm = None

        if exp_spec is not None:
            exp_min, exp_max = exp_spec[:, 0].min(), exp_spec[:, 0].max()
            exp_base = np.linspace(exp_min, exp_max, N_POINTS)
            fit_exp = interpolate.interp1d(
                exp_spec[:, 0], exp_spec[:, 1],
            )(exp_base)
            fit_exp_norm = fit_exp / fit_exp.max()

            ax[plot_idx].plot(
                exp_spec[:, 0],
                exp_spec[:, 1] / exp_spec[:, 1].max(),
                lw=2.5, color=C_EXP, ls="-", label="Exp.", alpha=0.8,
            )

        # ── Calculated (ab initio) ────────────────────────────────────
        calc_sticks = _load_all_calc_sticks(eval_dir, mol_name, n_c)
        fit_calc_norm = calc_grid = None

        if calc_sticks and exp_spec is not None:
            calc_all = np.row_stack(calc_sticks)
            calc_all[:, 0] += KE_SHIFT
            fit_calc = _broaden_sticks(
                calc_all[:, 0], calc_all[:, 1],
                FWHM, exp_min - 1, exp_max + 1, N_POINTS,
            )
            fit_calc_norm = fit_calc[:, 1] / fit_calc[:, 1].max()
            calc_grid = fit_calc[:, 0]
            ax[plot_idx].plot(
                calc_grid, fit_calc_norm,
                lw=2.5, color=C_CALC, ls="-", label="Calc.", alpha=0.8,
            )
            if show_vlines:
                ax[plot_idx].vlines(
                    calc_all[:, 0], 0,
                    calc_all[:, 1] / fit_calc[:, 1].max(),
                    lw=1.5, color=C_CALC, ls="--", alpha=0.5,
                )

        # ── PCC accumulators ──────────────────────────────────────────
        pcc_entry = dict(molecule=mol_name, pcc_calc=None)

        if exp_base is not None and fit_exp_norm is not None:
            if fit_calc_norm is not None and calc_grid is not None:
                calc_interp = np.interp(exp_base, calc_grid, fit_calc_norm)
                pcc_entry["pcc_calc"] = _compute_pcc(calc_interp, fit_exp_norm)

        # ── GNN curves — loop over feature sets ──────────────────────
        for fk in feature_sets:
            c_st, c_ft, ls_st, ls_ft = _get_style(fk)
            tag = f" ({fk})" if multi else ""

            stick_combined = all_stick.get(fk, {})
            fitted_preds   = all_fitted.get(fk)

            # ---- Stick ----
            gnn_stick_norm = gnn_stick_grid = None
            if mol_idx in stick_combined and exp_spec is not None:
                gnn_stick_spec = stick_combined[mol_idx].copy()
                gnn_stick_spec[:, 0] += KE_SHIFT
                gnn_fit = _broaden_sticks(
                    gnn_stick_spec[:, 0], gnn_stick_spec[:, 1],
                    FWHM, exp_min - 1, exp_max + 1, N_POINTS,
                )
                gnn_max = gnn_fit[:, 1].max()
                if gnn_max > 0:
                    gnn_stick_norm = gnn_fit[:, 1] / gnn_max
                    gnn_stick_grid = gnn_fit[:, 0]
                    ax[plot_idx].plot(
                        gnn_stick_grid, gnn_stick_norm,
                        lw=3.5, color=c_st, ls=ls_st,
                        label=f"GNN Stick{tag}", alpha=0.8,
                    )
                    if show_vlines:
                        ax[plot_idx].vlines(
                            gnn_stick_spec[:, 0], 0,
                            gnn_stick_spec[:, 1] / gnn_max,
                            lw=2.0, color=c_st, alpha=0.4,
                        )

            # ---- Fitted ----
            gnn_fitted_norm = None
            if fitted_preds is not None and mol_idx in fitted_preds:
                gnn_total = np.zeros(N_POINTS)
                for a_idx in fitted_preds[mol_idx]:
                    gnn_total += fitted_preds[mol_idx][a_idx]
                if gnn_total.max() > 0:
                    gnn_fitted_norm = gnn_total / gnn_total.max()
                    ax[plot_idx].plot(
                        energy_grid, gnn_fitted_norm,
                        lw=3.5, color=c_ft, ls=ls_ft,
                        label=f"GNN Fit{tag}", alpha=0.8,
                    )

            # ---- PCC for this feature set ----
            if exp_base is not None and fit_exp_norm is not None:
                try:
                    pcc_st = pcc_ft = None
                    if gnn_stick_norm is not None and gnn_stick_grid is not None:
                        interp = np.interp(exp_base, gnn_stick_grid, gnn_stick_norm)
                        pcc_st = _compute_pcc(interp, fit_exp_norm)
                    if gnn_fitted_norm is not None:
                        interp = np.interp(exp_base, energy_grid, gnn_fitted_norm)
                        pcc_ft = _compute_pcc(interp, fit_exp_norm)
                    pcc_entry[f"stick_{fk}"] = pcc_st
                    pcc_entry[f"fitted_{fk}"] = pcc_ft
                except Exception as e:
                    print(f"  ⚠ PCC error for {mol_name}/{fk}: {e}")

        pcc_data.append(pcc_entry)

        # ── Annotation ────────────────────────────────────────────────
        ax[plot_idx].text(
            0.05, 0.95, mol_name,
            transform=ax[plot_idx].transAxes,
            fontsize=20, va="top", fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        ax[plot_idx].set_xlim(220, 275)
        ax[plot_idx].set_ylim(0, 1.15)
        ax[plot_idx].tick_params(axis="y", labelleft=False, labelsize=14)

        # Row index for this panel (0-based)
        row_idx = plot_idx // ncols
        if row_idx < nrows - 1:
            # First / middle rows: hide x-axis tick labels
            ax[plot_idx].tick_params(axis="x", labelbottom=False, labelsize=24)
        else:
            # Last row: show x-axis tick labels, larger font
            ax[plot_idx].tick_params(axis="x", labelbottom=True, labelsize=24)

    # Hide unused panels
    for j in range(len(mol_names), nrows * ncols):
        ax[j].set_visible(False)

    # Legend on first panel
    ax[0].legend(loc="upper right", fontsize=19, framealpha=0.85)

    # Shared labels
    fig.text(0.5, 0.005, "Kinetic Energy (eV)", ha="center", fontsize=24)
    fig.text(0.005, 0.5, "Normalized Intensity (arb. units)",
             va="center", rotation="vertical", fontsize=24)

    plt.tight_layout(rect=[0.02, 0.02, 1, 1])

    os.makedirs(PNG_DIR, exist_ok=True)
    out_path = os.path.join(PNG_DIR, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved {out_path}")
    plt.close(fig)

    # ── Print PCC table ───────────────────────────────────────────────
    _f = lambda v: f"{v:.4f}" if v is not None else "  —   "

    # Build dynamic header
    hdr = f"  {'Molecule':<22} {'Calc':>8}"
    for fk in feature_sets:
        tag = f"_{fk}" if multi else ""
        hdr += f" {'St' + tag:>10} {'Ft' + tag:>10}"
    print(f"\n{hdr}")
    print(f"  {'-' * (22 + 8 + 20 * len(feature_sets))}")

    for e in pcc_data:
        row = f"  {e['molecule']:<22} {_f(e.get('pcc_calc')):>8}"
        for fk in feature_sets:
            row += f" {_f(e.get(f'stick_{fk}')):>10} {_f(e.get(f'fitted_{fk}')):>10}"
        print(row)
    print()

# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Publication comparison plots for Auger GNN models.",
    )
    parser.add_argument(
        "--layout", default="2x2", choices=["2x2", "3x2"],
        help="Panel grid layout (default: 2x2).",
    )
    parser.add_argument(
        "--molecules", nargs="+", default=None,
        help="Molecule names to plot (overrides default lists).",
    )
    parser.add_argument(
        "--feature-sets", nargs="+", default=None, dest="feature_sets",
        help="Feature-key strings to compare, e.g. 035 045 (default: DEFAULT_FEATURE_SETS).",
    )
    parser.add_argument(
        "--filename", default=None,
        help="Output PNG filename (default: comparison_<layout>.png).",
    )
    args = parser.parse_args()

    feature_sets: list[str] = args.feature_sets or list(DEFAULT_FEATURE_SETS)

    # Determine grid + molecule list + per-layout defaults
    if args.layout == "2x2":
        nrows, ncols = 2, --layout2
        mol_names = args.molecules or MOLECULES_2x2
    else:
        nrows, ncols = 3, 2
        mol_names = args.molecules or MOLECULES_3x2
        # 3×2 layout defaults to 035 only (override with --feature-sets)
        if args.feature_sets is None:
            feature_sets = ["035"]

    n_panels = nrows * ncols
    if len(mol_names) > n_panels:
        print(f"  ⚠ {len(mol_names)} molecules but only {n_panels} panels "
              f"— truncating to first {n_panels}")
        mol_names = mol_names[:n_panels]

    filename = args.filename or f"comparison_{args.layout}.png"

    print("=" * 70)
    print("  Publication Plots — Multi-Feature-Set Auger GNN Comparison")
    print("=" * 70)
    print(f"  Layout:       {nrows}×{ncols}")
    print(f"  Feature sets: {feature_sets}")
    for fk in feature_sets:
        print(f"    {fk} → {describe_features(fk)}")
    print(f"  Molecules:    {mol_names}")
    print(f"  Output:       {os.path.join(PNG_DIR, filename)}")
    print("=" * 70)

    # ── Shared metadata ───────────────────────────────────────────────
    eval_dir = os.path.join(DATA_RAW_DIR, "eval_auger")
    mol_list, c_num = _load_eval_metadata(eval_dir)

    # ── Loop over feature sets ────────────────────────────────────────
    all_stick: dict[str, dict] = {}
    all_fitted: dict[str, dict | None] = {}

    for fk in feature_sets:
        mid = _model_id_for(fk)
        print(f"\n─── Feature set {fk}  (model stem: {mid}) ───")

        # Load fresh copies of eval data with this feature assembly
        eval_sing, eval_trip = _load_eval_data(fk)
        print(f"  {len(eval_sing)} singlet + {len(eval_trip)} triplet eval molecules")

        # Stick models + predict
        print(f"  Loading stick models ({fk}) ...")
        sing_model, trip_model, device = _load_stick_models(eval_sing, fk)

        print(f"  Generating stick predictions ({fk}) ...")
        sing_preds = _predict_stick(sing_model, eval_sing, device, MAX_KE, MAX_SPEC_LEN)
        trip_preds = _predict_stick(trip_model, eval_trip, device, MAX_KE, MAX_SPEC_LEN)
        stick_combined = _combine_stick_preds(sing_preds, trip_preds, len(eval_sing))
        all_stick[fk] = stick_combined
        print(f"  {len(stick_combined)} molecules with stick predictions")

        # Fitted model + predict
        print(f"  Loading fitted model ({fk}) ...")
        fitted_model, fitted_device = _load_fitted_model(eval_sing, fk)
        if fitted_model is not None:
            print(f"  Generating fitted predictions ({fk}) ...")
            fitted_preds = _predict_fitted(fitted_model, eval_sing, fitted_device)
            all_fitted[fk] = fitted_preds
            print(f"  {len(fitted_preds)} molecules with fitted predictions")
        else:
            all_fitted[fk] = None
            print(f"  Fitted model not available for {fk} — stick-only")

    # ── Plot ──────────────────────────────────────────────────────────
    print(f"\nCreating {nrows}×{ncols} plot ...")
    _plot_grid(
        mol_names, nrows, ncols,
        all_stick=all_stick,
        all_fitted=all_fitted,
        feature_sets=feature_sets,
        mol_list=mol_list,
        c_num=c_num,
        eval_dir=eval_dir,
        filename=filename,
    )

    print("=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()