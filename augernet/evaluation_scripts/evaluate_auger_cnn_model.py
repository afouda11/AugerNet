"""
Evaluate Auger CNN Model — Carbon Environment Classification
=============================================================

Two evaluation modes:

1. **Calculated spectra** — broadened from singlet/triplet sticks
   (same pipeline as training).
2. **Experimental spectra** — pre-broadened digitised measurements stored
   in the ``exp_spec`` column of the eval pickle.  Only molecules with a
   single unique carbon environment are used (unambiguous ground truth).

Usage:

1. **Imported by backend_cnn.py** — called automatically after training::

       from augernet.evaluation_scripts.evaluate_auger_cnn_model import (
           run_evaluation, load_model, get_input_length,
       )

2. **Standalone CLI** — evaluate any saved CNN model::

       python -m augernet.evaluation_scripts.evaluate_auger_cnn_model \\
           --model /path/to/cnn_model.pth \\
           --output-dir eval_results/outputs
"""

from __future__ import annotations

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
from typing import Optional, Tuple

from augernet import DATA_PROCESSED_DIR
from augernet import cnn_train_utils as ctu
from augernet.cnn_train_utils import AugerCNN1D
from augernet import carbon_dataframe as cdf
from augernet.class_merging import (
    apply_label_merging,
    get_num_classes,
    get_merged_class_names,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Load model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    model_path: str,
    *,
    architecture: dict = None,
    device_str: str = 'auto',
    input_length: int = 732,
    merge_scheme: str = 'none',
) -> Tuple[torch.nn.Module, torch.device]:
    """Load a saved CNN model from a ``.pth`` state-dict file.

    Parameters
    ----------
    model_path : str
        Path to the saved ``.pth`` file.
    architecture : dict, optional
        Architecture dict for ``AugerCNN1D``.  Defaults to the recommended
        preset defined in ``cnn_train_utils``.
    device_str : str
        ``'auto'``, ``'cuda'``, ``'mps'``, or ``'cpu'``.
    input_length : int
        CNN input length (spectrum points + optional augmentation channel).
    merge_scheme : str
        Class merging scheme (``'none'`` | ``'chemical'`` | ``'conservative'``
        | ``'practical'`` | ``'aggressive'``).

    Returns
    -------
    model : torch.nn.Module
    device : torch.device
    """
    device = ctu.get_device(device_str, verbose=True)

    num_classes = ctu.NUM_CARBON_CLASSES
    if merge_scheme != 'none':
        num_classes = get_num_classes(merge_scheme)

    if architecture is None:
        architecture = ctu.ARCHITECTURE_PRESETS['recommended']

    model = AugerCNN1D(input_length, num_classes, architecture)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Loaded model  ← {model_path}  ({n_params:,} params)")
    return model, device


# ─────────────────────────────────────────────────────────────────────────────
#  Input length helper
# ─────────────────────────────────────────────────────────────────────────────

def get_input_length(
    df: pd.DataFrame,
    use_augmented: bool,
    augmented_scaled: bool,
    n_spectrum_points: int = 731,
) -> int:
    """Determine CNN input_length (spectra are always broadened to a grid)."""
    n_spec = n_spectrum_points
    return n_spec + (1 if (use_augmented or augmented_scaled) else 0)


# ─────────────────────────────────────────────────────────────────────────────
#  Experimental spectra helpers
# ─────────────────────────────────────────────────────────────────────────────

def _filter_single_env_molecules(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only molecules that have a single unique carbon environment.

    Experimental spectra are measured per-molecule, so we can only use them
    for unambiguous classification when every carbon atom in the molecule
    belongs to the same environment class.
    """
    mol_n_envs = df.groupby('mol_name')['carbon_env_label'].nunique()
    single_env_mols = mol_n_envs[mol_n_envs == 1].index
    filtered = df[df['mol_name'].isin(single_env_mols)].copy()
    n_mols = filtered['mol_name'].nunique()
    print(f"  Filtered to {n_mols} single-env molecules "
          f"({len(filtered)} atoms) from {df['mol_name'].nunique()} total")
    return filtered


class ExpSpectraDataset(torch.utils.data.Dataset):
    """Dataset for pre-broadened experimental spectra.

    Each molecule has one experimental spectrum (Nx2 array of
    ``[energy, intensity]`` pairs on a non-uniform grid).  This dataset
    interpolates each spectrum onto the standard uniform energy grid and
    normalises to [0, 1].

    For single-env molecules every carbon atom gets the same spectrum
    (the molecule's experimental spectrum).

    Parameters
    ----------
    df : pd.DataFrame
        Carbon DataFrame filtered to single-env molecules.
        Must contain ``exp_spec`` column (Nx2 arrays).
    energy_min, energy_max : float
        Uniform energy grid limits (eV).
    n_points : int
        Number of uniform grid points.
    include_augmentation : bool
        Whether to prepend delta_be (matches calc dataset format).
    augmentation_type : str
        ``'normalized'`` or ``'scaled'``.
    delta_be_scale : float
        Divisor for scaled augmentation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        energy_min: float = 200.0,
        energy_max: float = 273.0,
        n_points: int = 731,
        include_augmentation: bool = False,
        augmentation_type: str = 'normalized',
        delta_be_scale: float = 100.0,
    ):
        self.df = df.reset_index(drop=True)
        self.include_augmentation = include_augmentation
        self.augmentation_type = augmentation_type
        self.delta_be_scale = delta_be_scale

        n_atoms = len(self.df)
        target_grid = np.linspace(energy_min, energy_max, n_points)
        self._spectra = np.zeros((n_atoms, n_points), dtype=np.float32)

        # Build one interpolated spectrum per molecule, share across atoms
        mol_spectra = {}
        for mol_name in self.df['mol_name'].unique():
            row = self.df[self.df['mol_name'] == mol_name].iloc[0]
            raw = np.asarray(row['exp_spec'], dtype=np.float64)
            if raw.ndim == 2 and raw.shape[1] == 2:
                energies = raw[:, 0]
                intensities = raw[:, 1]
            else:
                warnings.warn(
                    f"Unexpected exp_spec shape for {mol_name}: {raw.shape}. "
                    "Skipping."
                )
                mol_spectra[mol_name] = np.zeros(n_points, dtype=np.float32)
                continue

            # Sort by energy (digitised data may not be monotonic)
            order = np.argsort(energies)
            energies = energies[order]
            intensities = intensities[order]

            # Interpolate onto the uniform grid (zero outside measured range)
            interp = np.interp(target_grid, energies, intensities,
                               left=0.0, right=0.0)
            mol_spectra[mol_name] = interp.astype(np.float32)

        for i in range(n_atoms):
            mol_name = self.df.iloc[i]['mol_name']
            self._spectra[i] = mol_spectra[mol_name]

        # delta_be normalisation (same as CarbonDataset)
        if 'delta_be' in df.columns:
            mu = df['delta_be'].mean()
            std = df['delta_be'].std() + 1e-8
            self._delta_be_norm = ((df['delta_be'] - mu) / std).values
        else:
            self._delta_be_norm = np.zeros(n_atoms, dtype=np.float32)

        n_mols = self.df['mol_name'].nunique()
        print(f"✓ ExpSpectraDataset: {n_atoms} atoms ({n_mols} molecules), "
              f"grid {energy_min}–{energy_max} eV ({n_points} pts)")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        spectrum = torch.from_numpy(self._spectra[idx].copy())

        # Per-atom [0, 1] normalisation
        smax = spectrum.max()
        smin = spectrum.min()
        if smax - smin > 1e-8:
            spectrum = (spectrum - smin) / (smax - smin)
        else:
            spectrum = torch.zeros_like(spectrum)

        # Optional delta_be augmentation
        if self.include_augmentation:
            if self.augmentation_type == 'normalized':
                dbe = float(self._delta_be_norm[idx])
            else:
                dbe = float(row['delta_be'] / self.delta_be_scale)
            spectrum = torch.cat([torch.tensor([dbe]), spectrum])

        label = torch.tensor(row['carbon_env_label'], dtype=torch.long)
        return spectrum, label


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    model: torch.nn.Module,
    device: torch.device,
    eval_data_path: str,
    output_dir: str,
    *,
    fold: Optional[int] = None,
    use_augmented: bool = False,
    augmented_scaled: bool = False,
    delta_be_scale: float = 100.0,
    broadening_fwhm: Optional[float] = None,
    energy_min: float = 200.0,
    energy_max: float = 273.0,
    n_spectrum_points: int = 731,
    merge_scheme: str = 'none',
    cv_suffix: str = '',
    # Legacy kwarg — accepted but ignored (exp data now comes from eval pkl)
    exp_data_path: str = None,
):
    """Evaluate on calc spectra + experimental spectra (from eval pkl).

    Parameters
    ----------
    model : torch.nn.Module
        Trained CNN model (already in eval mode).
    device : torch.device
        Device the model lives on.
    eval_data_path : str
        Path to the evaluation pickle.  Must contain stick-spectrum columns
        for calc evaluation and an ``exp_spec`` column for experimental
        evaluation.
    output_dir : str
        Directory for output files.
    fold : int, optional
        Fold number for filenames.
    use_augmented : bool
        Whether normalised delta_be augmentation was used.
    augmented_scaled : bool
        Whether scaled delta_be augmentation was used.
    delta_be_scale : float
        Scale factor for delta_be.
    broadening_fwhm : float, optional
        FWHM for on-the-fly Gaussian broadening (calc spectra only).
    energy_min, energy_max : float
        Energy grid limits.
    n_spectrum_points : int
        Number of spectrum grid points.
    merge_scheme : str
        Class merging scheme.
    cv_suffix : str
        Suffix for filenames.
    exp_data_path : str
        **Deprecated / ignored.**  Experimental spectra are now read from
        the ``exp_spec`` column of *eval_data_path*.
    """
    os.makedirs(output_dir, exist_ok=True)

    augmentation_type = ('normalized' if use_augmented
                         else ('scaled' if augmented_scaled else 'normalized'))

    # Merged class names for display
    merged_names = (get_merged_class_names(merge_scheme)
                    if merge_scheme != 'none' else None)
    merged_num = (get_num_classes(merge_scheme)
                  if merge_scheme != 'none' else None)

    # Suffix for CSV filenames
    csv_suffix = cv_suffix
    if use_augmented:
        csv_suffix += '_be_norm'
    elif augmented_scaled:
        csv_suffix += '_be_scale'
    if fold is not None:
        csv_suffix += f'_fold{fold}'

    # ── Load eval DataFrame once (used for both calc and exp) ─────────────
    if not eval_data_path or not os.path.exists(eval_data_path):
        print(f"⚠ Eval data not found: {eval_data_path}")
        return

    eval_df = cdf.load_carbon_dataframe(eval_data_path)
    if merge_scheme != 'none':
        eval_df = apply_label_merging(eval_df, merge_scheme, verbose=False)

    # ── 1. Evaluation on calculated hold-out (broadened sticks) ───────────
    print("\n" + "=" * 70)
    print(f"EVALUATION ON CALCULATED SPECTRA{f'  (fold {fold})' if fold else ''}")
    print("=" * 70)

    eval_base = cdf.CarbonDataset(
        eval_df,
        include_augmentation=use_augmented or augmented_scaled,
        augmentation_type=augmentation_type,
        delta_be_scale=delta_be_scale, normalize_delta_be=True,
        broadening_fwhm=broadening_fwhm,
        energy_min=energy_min, energy_max=energy_max,
        n_points=n_spectrum_points,
    )
    eval_ds = ctu.CarbonLabelDataset(eval_base, eval_df)

    ctu.evaluate_with_molecule_details(
        eval_df, model, device, eval_ds,
        output_dir=output_dir,
        eval_type='calc', csv_suffix=csv_suffix,
        class_names_override=merged_names,
        num_classes_override=merged_num,
    )

    # ── 2. Evaluation on experimental spectra ─────────────────────────────
    has_exp = 'exp_spec' in eval_df.columns
    if not has_exp:
        print("\n⚠ No 'exp_spec' column in eval data — skipping exp eval.")
        return

    # Check if any exp_spec values are non-null / non-empty
    non_null = eval_df['exp_spec'].apply(
        lambda x: x is not None and (
            (hasattr(x, '__len__') and len(x) > 0) or
            (isinstance(x, np.ndarray) and x.size > 0)
        )
    )
    if not non_null.any():
        print("\n⚠ All exp_spec values are empty — skipping exp eval.")
        return

    # Keep only rows with valid exp_spec
    exp_df = eval_df[non_null].copy()

    # Filter to single-carbon-environment molecules
    exp_df = _filter_single_env_molecules(exp_df)
    if len(exp_df) == 0:
        print("⚠ No single-env molecules with exp spectra — skipping.")
        return

    print("\n" + "=" * 70)
    print(f"EVALUATION ON EXPERIMENTAL DATA{f'  (fold {fold})' if fold else ''}")
    print("  (single-carbon-environment molecules only)")
    print("=" * 70)

    exp_base = ExpSpectraDataset(
        exp_df,
        energy_min=energy_min,
        energy_max=energy_max,
        n_points=n_spectrum_points,
        include_augmentation=use_augmented or augmented_scaled,
        augmentation_type=augmentation_type,
        delta_be_scale=delta_be_scale,
    )
    # ExpSpectraDataset already returns (spectrum, label) tuples —
    # no need for CarbonLabelDataset wrapper.

    ctu.evaluate_with_molecule_details(
        exp_df, model, device, exp_base,
        output_dir=output_dir,
        eval_type='exp', csv_suffix=csv_suffix,
        class_names_override=merged_names,
        num_classes_override=merged_num,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Auger CNN model on calculated/experimental data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m augernet.evaluation_scripts.evaluate_auger_cnn_model \\
      --model train_results/models/cnn_fold1_size_fwhm1pt6.pth

  python -m augernet.evaluation_scripts.evaluate_auger_cnn_model \\
      --model /path/to/model.pth --merge-scheme practical
        """,
    )

    parser.add_argument('--model', type=str, required=True,
                        help="Path to the saved model .pth file.")
    parser.add_argument('--merge-scheme', type=str, default='none',
                        choices=['none', 'chemical', 'conservative',
                                 'practical', 'aggressive'],
                        help="Class merging scheme (default: none).")
    parser.add_argument('--use-augmented', action='store_true', default=True,
                        help="Include normalised delta_be augmentation.")
    parser.add_argument('--no-augmented', action='store_true', default=False,
                        help="Disable delta_be augmentation.")
    parser.add_argument('--delta-be-scale', type=float, default=100.0,
                        help="Scale factor for delta_be (default: 100.0).")
    parser.add_argument('--broadening-fwhm', type=float, default=1.6,
                        help="FWHM for Gaussian broadening (eV). Default: 1.6.")
    parser.add_argument('--energy-min', type=float, default=200.0,
                        help="Energy grid minimum (eV). Default: 200.0.")
    parser.add_argument('--energy-max', type=float, default=273.0,
                        help="Energy grid maximum (eV). Default: 273.0.")
    parser.add_argument('--n-spectrum-points', type=int, default=731,
                        help="Number of spectrum grid points. Default: 731.")
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Directory for output files.")
    parser.add_argument('--fold', type=int, default=None,
                        help="Fold number for output filenames (optional).")
    parser.add_argument('--eval-data', type=str, default=None,
                        help="Path to evaluation pickle (calc + exp_spec).")
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help="Device to use (default: auto).")
    return parser.parse_args()


def main():
    args = _parse_args()

    use_augmented = args.use_augmented and not args.no_augmented
    augmented_scaled = False

    eval_data_path = args.eval_data or os.path.join(
        DATA_PROCESSED_DIR, 'cnn_auger_eval.pkl')
    train_data_path = os.path.join(
        DATA_PROCESSED_DIR, 'cnn_auger_calc.pkl')
    output_dir = args.output_dir or os.path.join(
        os.getcwd(), 'eval_results', 'outputs')

    print("=" * 80)
    print("  AUGER CNN MODEL EVALUATION (standalone)")
    print("=" * 80)
    print(f"  Model:           {args.model}")
    print(f"  Merge scheme:    {args.merge_scheme}")
    print(f"  Augmented:       {use_augmented}")
    print(f"  Broadening FWHM: {args.broadening_fwhm} eV")
    print(f"  Energy range:    {args.energy_min}–{args.energy_max} eV "
          f"({args.n_spectrum_points} pts)")
    print(f"  Eval data:       {eval_data_path}")
    print(f"  Output dir:      {output_dir}")
    print(f"  Device:          {args.device}")
    print("=" * 80)

    # ── Determine input length from training data ─────────────────────────
    print(f"\nLoading data to determine input length: {train_data_path}")
    tmp_df = cdf.load_carbon_dataframe(train_data_path)
    input_length = get_input_length(
        tmp_df, use_augmented, augmented_scaled,
        n_spectrum_points=args.n_spectrum_points,
    )
    print(f"  Input length: {input_length}")

    # ── Load model ────────────────────────────────────────────────────────
    model, device = load_model(
        args.model,
        device_str=args.device,
        input_length=input_length,
        merge_scheme=args.merge_scheme,
    )

    # ── Build cv_suffix for filenames ─────────────────────────────────────
    fwhm_str = f"fwhm{str(args.broadening_fwhm).replace('.', 'pt')}"
    cv_suffix = f"_{fwhm_str}"
    if args.merge_scheme != 'none':
        cv_suffix = f"_{args.merge_scheme}{cv_suffix}"

    # ── Run evaluation (calc + exp) ───────────────────────────────────────
    run_evaluation(
        model, device,
        eval_data_path=eval_data_path,
        output_dir=output_dir,
        fold=args.fold,
        use_augmented=use_augmented,
        augmented_scaled=augmented_scaled,
        delta_be_scale=args.delta_be_scale,
        broadening_fwhm=args.broadening_fwhm,
        energy_min=args.energy_min,
        energy_max=args.energy_max,
        n_spectrum_points=args.n_spectrum_points,
        merge_scheme=args.merge_scheme,
        cv_suffix=cv_suffix,
    )

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
