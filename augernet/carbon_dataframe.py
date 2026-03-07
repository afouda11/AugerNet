"""
Auger CNN Data Processing - Pandas-based Carbon-centric Format

This module provides utilities to store and load Auger spectra data as pandas DataFrames,
with one row per carbon atom (not per molecule or per all atoms).

Structure:
- Each row represents ONE CARBON ATOM from ONE MOLECULE
- Columns include: molecular info, carbon info, spectra, features
- Filtering to carbon atoms only happens automatically
- Data is stored in efficient Parquet format

Usage:
    # Generate data
    df = generate_carbon_dataframe(data_list)
    df.to_parquet("auger_calc.parquet")
    
    # Load data
    df = pd.read_parquet("auger_calc.parquet")
    dataset = CarbonDataset(df)
    loader = DataLoader(dataset, batch_size=32)
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from augernet.spec_utils import fit_spectrum_to_grid


def generate_carbon_dataframe(data_list: List[Any], molecule_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convert PyTorch Geometric Data objects into a pandas DataFrame with one row per carbon atom.
    
    Supports both fitted (broadened) and stick (raw peaks) spectrum formats.
    
    Parameters
    ----------
    data_list : List[Data]
        List of PyTorch Geometric Data objects from build_auger_graphs_cnn_mod.process_auger_data()
    molecule_names : List[str], optional
        List of molecule names corresponding to data_list
        If None, will use data.name field
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - mol_name: Molecule identifier
        - smiles: SMILES string for the molecule (empty string if not available)
        - mol_idx: Index in molecule list
        - atom_idx: Atom index in XYZ file (0-indexed)
        - carbon_env_label: Carbon environment class index
        - carbon_env_onehot: One-hot encoded carbon environment (44 dims)
        - e_neg_score: Electronegativity score for this atom
        - atomic_be: Binding energy for this atom
        - delta_be: Raw delta BE (atomic_be - mol_cebe)
        - delta_be_norm: Z-score normalized delta BE
        - morgan_fp: Morgan fingerprint (1024 dims)
        - spectrum_type: 'fitted' or 'stick'
        
        When spectrum_type == 'fitted':
        - spectrum: Fitted Auger spectrum (n_points, 2) with [energy, intensity]
        - spectrum_intensity_only: Just the intensity values (n_points,)
        
        When spectrum_type == 'stick':
        - stick_energies: Raw stick peak energies (variable length)
        - stick_intensities: Raw stick peak intensities (variable length)
        
        - node_mask: Whether this is a carbon atom (1.0 for carbons, 0.0 for non-carbons)
    """
    records = []
    
    for mol_idx, data in enumerate(data_list):
        mol_name = data.name if hasattr(data, 'name') else (molecule_names[mol_idx] if molecule_names else f"mol_{mol_idx}")
        mol_smiles = str(data.smiles) if hasattr(data, 'smiles') and data.smiles is not None else ''
        
        # Detect spectrum type
        spec_type = getattr(data, 'spectrum_type', 'fitted')
        
        if spec_type == 'stick':
            n_atoms = len(data.stick_spectra)
        else:
            n_atoms = data.y.shape[0]
        
        # Extract per-atom features
        for atom_idx in range(n_atoms):
            # Only include carbons (node_mask == 1.0 or carbon_env_label >= 0)
            if hasattr(data, 'node_mask'):
                is_carbon = data.node_mask[atom_idx].item() > 0.5
            else:
                is_carbon = data.carbon_env_labels[atom_idx].item() >= 0
            
            if not is_carbon:
                continue  # Skip non-carbon atoms
            
            # Build record
            record = {
                'mol_name': mol_name,
                'smiles': mol_smiles,
                'mol_idx': mol_idx,
                'atom_idx': atom_idx,
                'carbon_env_label': data.carbon_env_labels[atom_idx].item(),
                'e_neg_score': data.e_neg_scores[atom_idx].item() if hasattr(data, 'e_neg_scores') else np.nan,
                'atomic_be': data.atomic_be[atom_idx].item() if hasattr(data, 'atomic_be') else np.nan,
                'delta_be': data.delta_be[atom_idx].item() if hasattr(data, 'delta_be') else np.nan,
                'spectrum_type': spec_type,
            }
            
            if spec_type == 'stick':
                # Store raw stick peaks (variable length)
                stick_arr = data.stick_spectra[atom_idx]
                if stick_arr.shape[0] > 0:
                    record['stick_energies'] = stick_arr[:, 0].copy()
                    record['stick_intensities'] = stick_arr[:, 1].copy()
                else:
                    record['stick_energies'] = np.array([], dtype=np.float32)
                    record['stick_intensities'] = np.array([], dtype=np.float32)
                # No pre-broadened spectrum
                record['spectrum'] = None
                record['spectrum_intensity_only'] = None
            else:
                # Legacy fitted path
                spectrum_flat = data.y[atom_idx].numpy()
                n_points = len(spectrum_flat) // 2
                spectrum_2d = spectrum_flat.reshape(n_points, 2)
                energy_grid = spectrum_2d[:, 0]
                intensity_grid = spectrum_2d[:, 1]
                record['spectrum'] = spectrum_2d
                record['spectrum_intensity_only'] = intensity_grid
                record['stick_energies'] = None
                record['stick_intensities'] = None
            
            # Add optional fields
            if hasattr(data, 'delta_be_norm'):
                record['delta_be_norm'] = data.delta_be_norm[atom_idx].item()
            else:
                record['delta_be_norm'] = np.nan
            
            if hasattr(data, 'carbon_env_onehot'):
                record['carbon_env_onehot'] = data.carbon_env_onehot[atom_idx].numpy()
            else:
                record['carbon_env_onehot'] = np.zeros(44)
            
            if hasattr(data, 'morgan_fp'):
                record['morgan_fp'] = data.morgan_fp[atom_idx].numpy()
            else:
                record['morgan_fp'] = np.zeros(1024)
            
            if hasattr(data, 'node_mask'):
                record['node_mask'] = data.node_mask[atom_idx].item()
            else:
                record['node_mask'] = 1.0 if is_carbon else 0.0
            
            records.append(record)
    
    df = pd.DataFrame(records)
    
    spec_type_str = df['spectrum_type'].iloc[0] if len(df) > 0 else 'unknown'
    print(f"\n{'='*80}")
    print(f"CARBON DATAFRAME GENERATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total molecules: {len(data_list)}")
    print(f"Total carbon atoms extracted: {len(df)}")
    print(f"Spectrum type: {spec_type_str}")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nCarbon environment distribution:")
    print(df['carbon_env_label'].value_counts().sort_index().head(10))
    print(f"{'='*80}\n")
    
    return df


def save_carbon_dataframe(df: pd.DataFrame, filepath: str, format: str = 'auto'):
    """
    Save carbon DataFrame to efficient format.
    
    Parameters
    ----------
    df : pd.DataFrame
        Carbon-centric DataFrame from generate_carbon_dataframe()
    filepath : str
        Path to save file (.parquet or .pkl)
    format : str, default='auto'
        Format to use: 'parquet', 'pickle', or 'auto' (tries parquet, falls back to pickle)
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'auto':
        # Determine format from filepath
        if filepath.endswith('.pkl') or filepath.endswith('.pickle'):
            format = 'pickle'
        elif filepath.endswith('.parquet'):
            format = 'parquet'
        else:
            format = 'parquet'  # Default to parquet
    
    try:
        if format == 'parquet':
            try:
                df.to_parquet(filepath, compression='snappy')
                print(f"✓ Saved {len(df)} carbon atoms to: {filepath} (Parquet)")
            except ImportError:
                print(f"⚠ Parquet not available, falling back to pickle...")
                # Replace extension and use pickle
                pkl_filepath = filepath.replace('.parquet', '.pkl')
                df.to_pickle(pkl_filepath)
                print(f"✓ Saved {len(df)} carbon atoms to: {pkl_filepath} (Pickle)")
        else:  # pickle
            df.to_pickle(filepath)
            print(f"✓ Saved {len(df)} carbon atoms to: {filepath} (Pickle)")
    except Exception as e:
        print(f"❌ Failed to save DataFrame: {e}")
        raise


def load_carbon_dataframe(filepath: str, format: str = 'auto') -> pd.DataFrame:
    """
    Load carbon DataFrame from file.
    
    Parameters
    ----------
    filepath : str
        Path to .parquet or .pkl file
    format : str, default='auto'
        Format: 'parquet', 'pickle', or 'auto' (auto-detect from extension)
    
    Returns
    -------
    pd.DataFrame
        Carbon-centric DataFrame
    """
    if format == 'auto':
        # Determine format from filepath
        if filepath.endswith('.pkl') or filepath.endswith('.pickle'):
            format = 'pickle'
        elif filepath.endswith('.parquet'):
            format = 'parquet'
        else:
            format = 'parquet'  # Default to parquet
    
    try:
        if format == 'parquet':
            try:
                df = pd.read_parquet(filepath)
                print(f"✓ Loaded {len(df)} carbon atoms from: {filepath} (Parquet)")
            except ImportError:
                print(f"⚠ Parquet not available, trying pickle...")
                pkl_filepath = filepath.replace('.parquet', '.pkl')
                if Path(pkl_filepath).exists():
                    df = pd.read_pickle(pkl_filepath)
                    print(f"✓ Loaded {len(df)} carbon atoms from: {pkl_filepath} (Pickle)")
                else:
                    raise FileNotFoundError(f"Neither {filepath} nor {pkl_filepath} found")
        else:  # pickle
            df = pd.read_pickle(filepath)
            print(f"✓ Loaded {len(df)} carbon atoms from: {filepath} (Pickle)")
        
        return df
    except Exception as e:
        print(f"❌ Failed to load DataFrame: {e}")
        raise


class CarbonDataset(Dataset):
    """
    PyTorch Dataset for carbon-centric Auger spectra data.
    
    One sample = one carbon atom from one molecule.
    
    Supports two spectrum storage modes:
    - **fitted** (legacy): Pre-broadened spectra stored in ``spectrum_intensity_only``.
    - **stick**: Raw stick peaks stored in ``stick_energies`` / ``stick_intensities``.
      Gaussian broadening is applied on-the-fly during ``__getitem__()`` using
      ``fit_spectrum_to_grid()``.  This allows changing FWHM without
      re-running data preparation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Carbon-centric DataFrame with columns: spectrum, carbon_env_label, etc.
    include_augmentation : bool, default=False
        If True, prepend delta_be feature to spectrum
    augmentation_type : str, default='normalized'
        Type of delta_be augmentation: 'normalized' (z-score) or 'scaled' (raw/scale_factor)
    delta_be_scale : float, default=100.0
        Scale factor for 'scaled' augmentation type
    normalize_delta_be : bool, default=True
        If True, normalize delta_be values (z-score) across dataset
    normalize_spectrum : bool, default=True
        If True, normalize broadened spectrum to [0, 1] per atom
    broadening_fwhm : float or None, default=None
        FWHM for Gaussian broadening (eV). Required when spectrum_type=='stick'.
        Ignored when spectrum_type=='fitted'.
    energy_min : float, default=200.0
        Minimum energy for fitted grid (eV)
    energy_max : float, default=273.0
        Maximum energy for fitted grid (eV)
    n_points : int, default=731
        Number of points in the output grid
    """
    
    def __init__(self, df: pd.DataFrame, 
                 include_augmentation: bool = False,
                 augmentation_type: str = 'normalized',
                 delta_be_scale: float = 100.0,
                 normalize_delta_be: bool = True,
                 normalize_spectrum: bool = True,
                 broadening_fwhm: float = None,
                 energy_min: float = 200.0,
                 energy_max: float = 273.0,
                 n_points: int = 731):
        self.df = df.reset_index(drop=True)
        self.include_augmentation = include_augmentation
        self.augmentation_type = augmentation_type
        self.delta_be_scale = delta_be_scale
        self.normalize_delta_be = normalize_delta_be
        self.normalize_spectrum = normalize_spectrum
        
        # Stick-spectra broadening parameters
        self.broadening_fwhm = broadening_fwhm
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.n_points = n_points
        
        # Detect spectrum type from data
        self.is_stick = (
            'spectrum_type' in df.columns
            and df['spectrum_type'].iloc[0] == 'stick'
        )
        
        if self.is_stick and broadening_fwhm is None:
            raise ValueError(
                "broadening_fwhm must be specified when the DataFrame "
                "contains stick spectra (spectrum_type='stick')"
            )
        
        # Pre-broaden ALL stick spectra once (instead of per-epoch in __getitem__)
        self._broadened_cache = None
        if self.is_stick:
            import time as _time
            _t0 = _time.time()
            energy_grid = np.linspace(energy_min, energy_max, n_points)
            n_atoms = len(df)
            cache = np.zeros((n_atoms, n_points), dtype=np.float32)
            full_cache = np.zeros((n_atoms, n_points, 2), dtype=np.float32)
            for i in range(n_atoms):
                row = df.iloc[i]
                energies = row['stick_energies']
                intensities = row['stick_intensities']
                if energies is not None and len(energies) > 0:
                    _, intensity_grid = fit_spectrum_to_grid(
                        energies, intensities,
                        fwhm=broadening_fwhm,
                        energy_min=energy_min,
                        energy_max=energy_max,
                        n_points=n_points,
                    )
                    cache[i] = intensity_grid
                full_cache[i, :, 0] = energy_grid
                full_cache[i, :, 1] = cache[i]
            self._broadened_cache = cache
            self._full_cache = full_cache
            _elapsed = _time.time() - _t0
            print(f"  Pre-broadened {n_atoms} stick spectra (FWHM={broadening_fwhm} eV) "
                  f"in {_elapsed:.1f}s")
        
        # Normalize delta_be if requested
        if normalize_delta_be and 'delta_be_norm' not in df.columns:
            self.df['delta_be_norm'] = (
                (df['delta_be'] - df['delta_be'].mean()) / 
                (df['delta_be'].std() + 1e-8)
            )
        
        # Compute scaled delta_be if needed
        if augmentation_type == 'scaled' and 'delta_be_scaled' not in df.columns:
            self.df['delta_be_scaled'] = df['delta_be'] / delta_be_scale
        
        spec_info = f"stick→FWHM={broadening_fwhm} (pre-broadened)" if self.is_stick else "pre-broadened"
        print(f"✓ Created CarbonDataset with {len(self.df)} carbon atoms")
        print(f"  Spectrum: {spec_info}")
        print(f"  Augmentation: {include_augmentation} ({augmentation_type})")
        if include_augmentation and augmentation_type == 'scaled':
            print(f"  Delta BE scale factor: {delta_be_scale}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _broaden_stick(self, row) -> np.ndarray:
        """Apply Gaussian broadening to stick peaks and return intensity grid."""
        energies = row['stick_energies']
        intensities = row['stick_intensities']
        
        if len(energies) == 0:
            return np.zeros(self.n_points, dtype=np.float32)
        
        _, intensity_grid = fit_spectrum_to_grid(
            energies, intensities,
            fwhm=self.broadening_fwhm,
            energy_min=self.energy_min,
            energy_max=self.energy_max,
            n_points=self.n_points,
        )
        return intensity_grid
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get one carbon atom's data.
        
        Returns
        -------
        Dict with keys:
            - spectrum: intensity values (n_points,) — broadened on-the-fly if stick
            - spectrum_full: [energy, intensity] pairs (n_points, 2)
            - label: carbon environment label (scalar)
            - onehot: one-hot encoded label (44,)
            - e_neg: electronegativity score (scalar)
            - atomic_be: binding energy (scalar)
            - delta_be: raw delta BE (scalar)
            - delta_be_norm: normalized delta BE (scalar)
            - morgan_fp: Morgan fingerprint (1024,)
            - mol_name: molecule identifier (str)
            - atom_idx: atom index in molecule (int)
        """
        row = self.df.iloc[idx]
        
        # ----- Spectrum intensity -----
        if self.is_stick:
            # Use pre-broadened cache (computed once in __init__)
            spectrum_input = torch.FloatTensor(self._broadened_cache[idx])
            spectrum_full = torch.FloatTensor(self._full_cache[idx])
        else:
            # Pre-broadened spectrum
            spectrum_input = torch.FloatTensor(row['spectrum_intensity_only'])
            spectrum_full = torch.FloatTensor(row['spectrum'])
        
        # Normalize spectrum intensity to [0, 1] per-atom
        if self.normalize_spectrum:
            spec_min = spectrum_input.min()
            spec_max = spectrum_input.max()
            if spec_max - spec_min > 1e-8:
                spectrum_input = (spectrum_input - spec_min) / (spec_max - spec_min)
            else:
                spectrum_input = torch.zeros_like(spectrum_input)
        
        # Optionally prepend delta_be to spectrum for augmentation
        if self.include_augmentation:
            if self.augmentation_type == 'normalized':
                delta_be_value = float(row['delta_be_norm'])
            elif self.augmentation_type == 'scaled':
                delta_be_value = float(row['delta_be'] / self.delta_be_scale)
            else:
                raise ValueError(f"Unknown augmentation_type: {self.augmentation_type}")
            
            # Prepend delta_be as a single channel
            spectrum_input = torch.cat([
                torch.tensor([delta_be_value]),
                spectrum_input
            ])
        
        return {
            'spectrum': spectrum_input,
            'spectrum_full': spectrum_full,
            'label': torch.LongTensor([row['carbon_env_label']]).squeeze(),
            'onehot': torch.FloatTensor(row['carbon_env_onehot']),
            'e_neg': torch.FloatTensor([row['e_neg_score']]).squeeze(),
            'atomic_be': torch.FloatTensor([row['atomic_be']]).squeeze(),
            'delta_be': torch.FloatTensor([row['delta_be']]).squeeze(),
            'delta_be_norm': torch.FloatTensor([row['delta_be_norm']]).squeeze(),
            'morgan_fp': torch.FloatTensor(row['morgan_fp']),
            'mol_name': row['mol_name'],
            'atom_idx': int(row['atom_idx']),
        }


def create_dataloader(df: pd.DataFrame,
                      batch_size: int = 32,
                      shuffle: bool = True,
                      num_workers: int = 0,
                      include_augmentation: bool = False) -> DataLoader:
    """
    Create a PyTorch DataLoader from carbon DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Carbon-centric DataFrame
    batch_size : int, default=32
        Batch size
    shuffle : bool, default=True
        Whether to shuffle data
    num_workers : int, default=0
        Number of data loading workers
    include_augmentation : bool, default=False
        Whether to include delta_be in spectrum
    
    Returns
    -------
    DataLoader
        PyTorch DataLoader yielding batches of carbon atom data
    """
    dataset = CarbonDataset(df, include_augmentation=include_augmentation)
    
    def collate_fn(batch):
        """Custom collate to handle variable-length and nested data."""
        result = {}
        for key in batch[0].keys():
            if key in ['mol_name']:
                result[key] = [item[key] for item in batch]
            elif key in ['atom_idx']:
                result[key] = torch.LongTensor([item[key] for item in batch])
            else:
                result[key] = torch.stack([item[key] for item in batch])
        return result
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


# ============================================================================
# ANALYSIS AND VISUALIZATION
# ============================================================================

def analyze_carbon_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze carbon DataFrame and return summary statistics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Carbon-centric DataFrame
    
    Returns
    -------
    Dict
        Summary statistics
    """
    stats = {
        'total_carbons': len(df),
        'unique_molecules': df['mol_name'].nunique(),
        'unique_environments': df['carbon_env_label'].nunique(),
        'environment_distribution': df['carbon_env_label'].value_counts().to_dict(),
        'delta_be_mean': df['delta_be'].mean(),
        'delta_be_std': df['delta_be'].std(),
        'delta_be_min': df['delta_be'].min(),
        'delta_be_max': df['delta_be'].max(),
        'e_neg_mean': df['e_neg_score'].mean(),
        'e_neg_std': df['e_neg_score'].std(),
    }
    
    print(f"\n{'='*80}")
    print(f"CARBON DATAFRAME ANALYSIS")
    print(f"{'='*80}")
    print(f"Total carbons: {stats['total_carbons']}")
    print(f"Unique molecules: {stats['unique_molecules']}")
    print(f"Unique environments: {stats['unique_environments']}")
    print(f"\nEnvironment distribution:")
    for env_label, count in sorted(stats['environment_distribution'].items())[:10]:
        print(f"  Environment {env_label}: {count} atoms")
    print(f"\nDelta BE statistics:")
    print(f"  Mean: {stats['delta_be_mean']:.4f}")
    print(f"  Std: {stats['delta_be_std']:.4f}")
    print(f"  Range: [{stats['delta_be_min']:.4f}, {stats['delta_be_max']:.4f}]")
    print(f"{'='*80}\n")
    
    return stats


def get_broadened_spectrum(row, fwhm: float = 1.5,
                           energy_min: float = 200.0,
                           energy_max: float = 273.0,
                           n_points: int = 731) -> np.ndarray:
    """
    Get a broadened intensity array from a DataFrame row.

    Works transparently for both fitted and stick DataFrames:
    - fitted: returns the stored ``spectrum_intensity_only`` array.
    - stick:  applies Gaussian broadening on-the-fly.

    Parameters
    ----------
    row : pd.Series
        One row of a carbon DataFrame.
    fwhm : float
        FWHM for Gaussian broadening (only used for stick data).
    energy_min, energy_max, n_points : float / int
        Grid parameters (only used for stick data).

    Returns
    -------
    np.ndarray, shape (n_points,)
        Broadened intensity values normalised to [0, 1].
    """
    spec_type = row.get('spectrum_type', 'fitted')
    if spec_type == 'stick':
        energies = row['stick_energies']
        intensities = row['stick_intensities']
        if len(energies) == 0:
            return np.zeros(n_points, dtype=np.float64)
        _, intensity = fit_spectrum_to_grid(
            energies, intensities,
            fwhm=fwhm,
            energy_min=energy_min,
            energy_max=energy_max,
            n_points=n_points,
        )
        return intensity.astype(np.float64)
    else:
        return np.asarray(row['spectrum_intensity_only'], dtype=np.float64)
