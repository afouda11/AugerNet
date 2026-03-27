"""
Unified Molecule-Level K-Fold Splitting (for CNN carbon DataFrames)
====================================================================

Provides the same molecule-level train/val split used by the GNN backend
(``backend._get_fold_split``) but operating on a pandas DataFrame with
one row per carbon atom.

Splitting methods:
  ``random`` — sklearn KFold (shuffle=True).
  ``butina`` — Butina (scaffold) clustering via ``build_molecular_graphs``,
               then GroupKFold so similar molecules stay in the same fold.
  ``size``   — StratifiedKFold by molecular size (# carbons), binned.

Public API (consumed by ``backend_cnn.py``):
  get_stratified_kfold_split(carbon_df, ...)   → mol-level (train, val) indices
  mol_indices_to_carbon_indices(carbon_df, ...) → row-level (train, val) indices
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple

from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unique_mol_names(carbon_df) -> List[str]:
    """Ordered unique molecule names from a carbon DataFrame."""
    seen: dict = {}
    for name in carbon_df['mol_name']:
        if name not in seen:
            seen[name] = len(seen)
    return list(seen.keys())


def _mol_smiles(carbon_df, mol_names: List[str]) -> List[str]:
    """Extract one SMILES per molecule (first occurrence)."""
    mol_smi: dict = {}
    for name, smi in zip(carbon_df['mol_name'], carbon_df['smiles']):
        if name not in mol_smi and smi:
            mol_smi[name] = str(smi)
    return [mol_smi.get(n, '') for n in mol_names]


# ─────────────────────────────────────────────────────────────────────────────
#  Splitting back-ends
# ─────────────────────────────────────────────────────────────────────────────

def _random_folds(n_molecules, n_folds, random_state):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return list(kf.split(np.arange(n_molecules)))


def _butina_folds(smiles_list, n_folds):
    from augernet.build_molecular_graphs import get_butina_clusters
    cluster_ids = get_butina_clusters(smiles_list, cutoff=0.65)
    n = len(smiles_list)
    gkf = GroupKFold(n_splits=n_folds)
    return list(gkf.split(np.arange(n), groups=cluster_ids))


def _size_folds(carbon_df, mol_names, n_folds, random_state, n_bins=5):
    counts = carbon_df.groupby('mol_name', sort=False).size()
    sizes = np.array([counts.get(n, 1) for n in mol_names])
    try:
        bins = pd.qcut(sizes, q=n_bins, labels=False, duplicates='drop')
    except ValueError:
        bins = pd.qcut(sizes, q=min(3, len(np.unique(sizes))),
                        labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=random_state)
    return list(skf.split(np.arange(len(mol_names)), bins))


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_stratified_kfold_split(
    carbon_df,
    *,
    n_folds: int = 5,
    fold: int = 1,
    random_state: int = 42,
    split_method: str = 'size',
    split_file=None,          # accepted for compat, not used
    force_recompute=False,    # accepted for compat
    verbose: bool = True,
    **_extra,                 # absorb any unknown kwargs
) -> Tuple[List[int], List[int]]:
    """Molecule-level k-fold split for a carbon DataFrame.

    Parameters
    ----------
    carbon_df : pd.DataFrame
        One row per carbon atom; must have ``mol_name`` and ``smiles`` columns.
    n_folds, fold, random_state : int
        Standard CV parameters.  ``fold`` is **1-indexed**.
    split_method : str
        ``'random'`` | ``'butina'`` | ``'size'``
    verbose : bool
        Print a one-line summary.

    Returns
    -------
    (train_mol_idx, val_mol_idx) : Tuple[List[int], List[int]]
        Molecule-level indices (into the ordered unique mol_name list).
    """
    mol_names = _unique_mol_names(carbon_df)
    n_mol = len(mol_names)

    if split_method == 'random':
        folds = _random_folds(n_mol, n_folds, random_state)
    elif split_method == 'butina':
        smiles = _mol_smiles(carbon_df, mol_names)
        folds = _butina_folds(smiles, n_folds)
    elif split_method == 'size':
        folds = _size_folds(carbon_df, mol_names, n_folds, random_state)
    else:
        raise ValueError(
            f"split_method must be 'random', 'butina', or 'size'; "
            f"got '{split_method}'"
        )

    train_idx, val_idx = folds[fold - 1]
    train_idx = train_idx.tolist() if hasattr(train_idx, 'tolist') else list(train_idx)
    val_idx = val_idx.tolist() if hasattr(val_idx, 'tolist') else list(val_idx)

    if verbose:
        print(f"  {split_method.upper()} molecule split  "
              f"(fold {fold}/{n_folds}): "
              f"{len(train_idx)} train / {len(val_idx)} val molecules")

    return train_idx, val_idx


def mol_indices_to_carbon_indices(
    carbon_df,
    mol_train_idx: List[int],
    mol_val_idx: List[int],
) -> Tuple[List[int], List[int]]:
    """Convert molecule-level indices → carbon-row indices in the DataFrame."""
    mol_names = _unique_mol_names(carbon_df)
    train_mols = {mol_names[i] for i in mol_train_idx}
    val_mols = {mol_names[i] for i in mol_val_idx}

    carbon_train, carbon_val = [], []
    for row_idx, name in enumerate(carbon_df['mol_name']):
        if name in train_mols:
            carbon_train.append(row_idx)
        elif name in val_mols:
            carbon_val.append(row_idx)
    return carbon_train, carbon_val
