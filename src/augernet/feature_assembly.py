"""
Feature Assembly Module
=======================

Provides runtime feature selection and assembly for GNN training.

During data preparation, ALL possible node features are computed and stored
as separate attributes on each PyG Data object.  At training time, the user
selects which features to include via a list of integer keys, and this module
concatenates + optionally scales them into ``data.x``.

Feature Catalog
---------------
Key  Name            Dim   Description
───  ──────────────  ────  ─────────────────────────────────────────
 0   skipatom_200    200   SkipAtom atom-type embedding (200-dim)
 1   skipatom_30      30   SkipAtom atom-type embedding (30-dim)
 2   onehot            5   Element one-hot encoding (H, C, N, O, F)
 3   atomic_be         1   Isolated-atom 1s BE (Hartree, raw)
 4   mol_be            1   Molecular CEBE for C, atomic for others (Hartree, raw)
 5   e_score           1   Electronegativity-difference score (raw)
 6   env_onehot       ~8   Carbon environment one-hot (NUM_CARBON_CATEGORIES)
 7   morgan_fp       256   Per-atom Morgan fingerprint (ECFP2, radius=1)

Only the ``category_feature`` ([1,0,0], [0,1,0], [0,0,1]) is placed in
``data.x`` at preparation time.  Everything else lives in ``data.feat_*``
attributes and is assembled here at training time.

Usage
-----
>>> from augernet.feature_assembly import assemble_node_features, parse_feature_keys
>>> feature_keys_parsed = parse_feature_keys('035')  # [0, 3, 5]
>>>
>>> # Before creating DataLoader — modifies data.x in-place
>>> for data in data_list:
...     assemble_node_features(data, feature_keys_parsed)
"""

import torch
from copy import copy
from typing import List, Sequence

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE CATALOG
# ─────────────────────────────────────────────────────────────────────────────
# Maps integer key → attribute name on the Data object (set during preparation)
FEATURE_CATALOG = {
    0: 'feat_skipatom_200',   # (N, 200)
    1: 'feat_skipatom_30',    # (N, 30)
    2: 'feat_onehot',         # (N, 5) — H, C, N, O, F
    3: 'feat_atomic_be',      # (N, 1)
    4: 'feat_mol_be',         # (N, 1)
    5: 'feat_e_score',        # (N, 1)
    6: 'feat_env_onehot',     # (N, NUM_CARBON_CATEGORIES)
    7: 'feat_morgan_fp',      # (N, MORGAN_N_BITS) — per-atom Morgan FP (ECFP2)
}

FEATURE_NAMES = {
    0: 'skipatom_200',
    1: 'skipatom_30',
    2: 'onehot',
    3: 'atomic_be',
    4: 'mol_be',
    5: 'e_score',
    6: 'env_onehot',
    7: 'morgan_fp',
}


def compute_feature_tag(feature_keys: Sequence[int]) -> str:
    """
    Compute a compact filename-safe tag from sorted feature keys.

    >>> compute_feature_tag([3, 0, 5])
    '035'
    """
    return ''.join(str(k) for k in sorted(feature_keys))


def parse_feature_keys(tag: str) -> List[int]:
    """
    Parse a compact feature-key string into a sorted list of ints.

    Each character in the string is one feature key digit.

    >>> parse_feature_keys('035')
    [0, 3, 5]
    >>> parse_feature_keys('7')
    [7]
    """
    tag = str(tag).strip()
    if not tag:
        return []
    keys = sorted(int(ch) for ch in tag)
    unknown = [k for k in keys if k not in FEATURE_CATALOG]
    if unknown:
        raise ValueError(
            f"Unknown feature key(s) {unknown} in '{tag}'. "
            f"Valid keys: {sorted(FEATURE_CATALOG.keys())}"
        )
    return keys


def get_feature_dim(data, feature_keys: Sequence[int]) -> int:
    """
    Compute the total node-feature dimension that ``assemble_node_features``
    will produce (category_feature columns + selected feature columns).

    Parameters
    ----------
    data : torch_geometric.data.Data
        A single graph from the dataset (used to read tensor shapes).
    feature_keys : sequence of int
        Feature keys to include.

    Returns
    -------
    int
        Total ``data.x`` width after assembly.
    """
    # Use stashed category_feature if available (after first assembly),
    # otherwise fall back to current data.x (before first assembly).
    base = getattr(data, '_category_feature', data.x)
    cat_dim = base.size(1) if base is not None else 0
    feat_dim = 0
    for key in feature_keys:
        attr_name = FEATURE_CATALOG[key]
        tensor = getattr(data, attr_name, None)
        if tensor is None:
            raise ValueError(
                f"Feature key {key} ({FEATURE_NAMES[key]}) not found on Data object. "
                f"Available attributes: {[a for a in dir(data) if a.startswith('feat_')]}"
            )
        if tensor.dim() == 1:
            feat_dim += 1
        else:
            feat_dim += tensor.size(1)
    return cat_dim + feat_dim


def _scale_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    Apply per-molecule (per-graph) scaling to a feature tensor.

    Parameters
    ----------
    t : torch.Tensor
        Shape (N, D) or (N,)

    Returns
    -------
    torch.Tensor  — same shape, scaled.
    """

    if t.dim() == 1:
        t = t.unsqueeze(1)
    
    mu = t.mean(dim=0, keepdim=True)
    sigma = t.std(dim=0, keepdim=True)
    sigma = sigma.clamp(min=1e-8)  # avoid division by zero for single-atom molecules
    return (t - mu) / sigma



def assemble_node_features(
    data,
    feature_keys: Sequence[int],
    inplace: bool = True,
):
    """
    Concatenate selected node features into ``data.x``.

    The existing ``data.x`` (category_feature, shape [N, 3]) is kept as the
    **first** columns.  Selected features are scaled and appended.

    Parameters
    ----------
    data : torch_geometric.data.Data
        A single graph.  Must have ``feat_*`` attributes set during preparation.
    feature_keys : sequence of int
        Which features to include (see FEATURE_CATALOG).
    inplace : bool
        If True, modifies ``data.x`` directly.  If False, returns a copy.

    Returns
    -------
    data : the (possibly modified) Data object.
    """
    if not inplace:
        data = copy(data)

    # On first call, stash the original category_feature so that
    # subsequent calls (e.g. param search with different feature_keys)
    # always start from the base columns, not previously assembled ones.
    if not hasattr(data, '_category_feature'):
        data._category_feature = data.x.clone()

    parts = [data._category_feature]

    # Features that should NOT be scaled (categorical / pre-normalized)
    no_scale_keys = {0, 1, 2, 6, 7}

    for key in sorted(feature_keys):
        attr_name = FEATURE_CATALOG[key]
        tensor = getattr(data, attr_name, None)
        if tensor is None:
            raise ValueError(
                f"Feature key {key} ({FEATURE_NAMES[key]}) not found on Data object."
            )
        
        # Ensure 2D
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(1)

        # Scale scalar features only
        if key in no_scale_keys:
            parts.append(tensor.float())
        else:
            parts.append(_scale_tensor(tensor.float()))

    data.x = torch.cat(parts, dim=1)
    return data


def assemble_dataset(
    data_list: list,
    feature_keys: Sequence[int],
) -> list:
    """
    Apply ``assemble_node_features`` to every graph in a list (in-place).

    Returns the same list for convenience.
    """
    for data in data_list:
        assemble_node_features(data, feature_keys, inplace=True)
    return data_list


def describe_features(feature_keys: Sequence[int]) -> str:
    """
    Return a human-readable description of the selected feature set.

    >>> describe_features([0, 3, 5])
    'skipatom_200 (200) + atomic_be (1) + e_score (1)'
    """
    parts = []
    for key in sorted(feature_keys):
        name = FEATURE_NAMES.get(key, f'unknown_{key}')
        parts.append(name)
    return ' + '.join(parts)
