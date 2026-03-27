"""
CEBE-GNN Backend — model-specific hooks for train_driver.py
============================================================

Provides the five hooks that train_driver needs:
  load_data, train_single_run, load_saved_model,
  load_param_model, run_evaluation, run_unit_tests
"""

from __future__ import annotations

import os
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple
from sklearn.model_selection import KFold, GroupKFold

from augernet import gnn_train_utils as gtu
from augernet.feature_assembly import (
    assemble_dataset, compute_feature_tag, describe_features,
    parse_feature_keys,
)

from augernet import PROJECT_ROOT, DATA_DIR, DATA_PROCESSED_DIR


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_exp_split_names(split):
    """Load mol-name sets for the experimental val/eval split.

    Reads ``mol_list_val.txt`` and ``mol_list_eval.txt`` from the raw
    exp_cebe directory.  Returns ``(val_names, eval_names)`` as sets.
    """
    from augernet import DATA_RAW_DIR
    exp_dir = os.path.join(DATA_RAW_DIR, 'exp_cebe')

    def _read(fname):
        path = os.path.join(exp_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Exp split file not found: {path}\n"
                f"Run  generate_exp_split.py  to create it."
            )
        with open(path) as f:
            return {line.strip() for line in f if line.strip()}

    return _read('mol_list_val.txt'), _read('mol_list_eval.txt')


def _split_exp_data(exp_data_all, cfg):
    """Partition experimental data according to ``cfg.exp_split``.

    Returns ``(exp_val_data, exp_eval_data)``.  Depending on the mode:

    - ``'all'``  — both lists contain all 113 molecules (legacy behaviour)
    - ``'val'``  — val = 63 validation, eval = empty
    - ``'eval'`` — val = empty, eval = 50 evaluation
    - ``'both'`` — val = 63, eval = 50  (run evaluation on each separately)
    """
    split = cfg.exp_split

    if split == 'all':
        return exp_data_all, exp_data_all

    val_names, eval_names = _load_exp_split_names(split)

    exp_val  = [d for d in exp_data_all if d.mol_name in val_names]
    exp_eval = [d for d in exp_data_all if d.mol_name in eval_names]

    # Sanity check
    found = {d.mol_name for d in exp_val} | {d.mol_name for d in exp_eval}
    missing_val  = val_names  - {d.mol_name for d in exp_val}
    missing_eval = eval_names - {d.mol_name for d in exp_eval}
    if missing_val:
        print(f"  ⚠ Val split: {len(missing_val)} names not found in data: "
              f"{sorted(missing_val)[:5]}")
    if missing_eval:
        print(f"  ⚠ Eval split: {len(missing_eval)} names not found in data: "
              f"{sorted(missing_eval)[:5]}")

    if split == 'val':
        return exp_val, []
    elif split == 'eval':
        return [], exp_eval
    elif split == 'both':
        return exp_val, exp_eval
    else:
        raise ValueError(
            f"Unknown exp_split '{split}'. "
            f"Supported: 'all', 'val', 'eval', 'both'."
        )


def load_data(cfg) -> Dict[str, Any]:
    """Load CEBE calculated + experimental data (raw, without feature assembly).

    Feature assembly is deferred to ``train_single_run`` / ``run_evaluation``
    so that param-search can override ``feature_keys`` per configuration.
    """
    print(f"\nLoading training data from: {DATA_PROCESSED_DIR}")
    print(f"Feature keys: {cfg.feature_keys}  ({describe_features(cfg.feature_keys_parsed)})")
    print(f"Feature tag:  {cfg.feature_tag}")
    print(f"Model ID:     {cfg.model_id}")

    ds = gtu.LoadDataset(DATA_DIR, file_name='gnn_calc_cebe_data.pt')
    calc_data = [ds[i] for i in range(len(ds))]
    print(f"Loaded feature-store data: {len(calc_data)} molecules")

    # Experimental data — load all, then split
    exp_ds = gtu.LoadDataset(DATA_DIR, file_name='gnn_exp_cebe_data.pt')
    exp_data_all = [exp_ds[i] for i in range(len(exp_ds))]

    exp_val, exp_eval = _split_exp_data(exp_data_all, cfg)

    exp_split = cfg.exp_split
    if exp_split == 'all':
        print(f"Exp split: all ({len(exp_data_all)} molecules)")
    else:
        print(f"Exp split: {exp_split}  "
              f"(val={len(exp_val)}, eval={len(exp_eval)})")

    # Assemble features using the default config keys.
    # train_single_run will re-assemble if overrides change feature_keys.
    feature_keys = cfg.feature_keys_parsed
    print(f"Assembling features {cfg.feature_keys}")
    assemble_dataset(calc_data, feature_keys)
    # Assemble on all exp molecules (val + eval may overlap for 'all')
    assemble_dataset(exp_data_all, feature_keys)
    print(f"Calculated data: {len(calc_data)} molecules, "
          f"x.shape[1]={calc_data[0].x.size(1)}")

    return {
        'calc_data': calc_data,
        'exp_data': exp_data_all,      # full set (backward compat)
        'exp_val_data': exp_val,       # validation subset
        'exp_eval_data': exp_eval,     # evaluation subset
        'assembled_feature_keys': cfg.feature_keys,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Single training run
# ─────────────────────────────────────────────────────────────────────────────

def train_single_run(
    data: Dict[str, Any],
    fold: int,
    n_folds: int,
    *,
    save_dir: str,
    output_dir: str,
    cfg,
    verbose: bool = True,
    **overrides,
) -> Dict[str, Any]:
    """
    Train a single CEBE GNN on one fold.

    Returns dict with keys: model, device, fold, best_val_loss,
    best_train_loss, best_val_epoch, n_epochs, model_path, train_results,
    model_id, final_train_loss, final_val_loss
    """
    # Merge cfg with any per-config overrides (from param search)
    layer_type      = overrides.get('layer_type',      cfg.layer_type)
    hidden_channels = overrides.get('hidden_channels', cfg.hidden_channels)
    n_layers        = overrides.get('n_layers',        cfg.n_layers)
    num_epochs      = overrides.get('num_epochs',      cfg.num_epochs)
    patience        = overrides.get('patience',        cfg.patience)
    batch_size      = overrides.get('batch_size',      cfg.batch_size)
    learning_rate   = overrides.get('learning_rate',   cfg.learning_rate)
    random_seed     = overrides.get('random_seed',     cfg.random_seed)
    split_method    = overrides.get('split_method',    cfg.split_method)
    optimizer_type  = overrides.get('optimizer_type',  cfg.optimizer_type)
    weight_decay    = overrides.get('weight_decay',    cfg.weight_decay)
    gradient_clip_norm = overrides.get('gradient_clip_norm', cfg.gradient_clip_norm)
    warmup_epochs   = overrides.get('warmup_epochs',   cfg.warmup_epochs)
    min_lr          = overrides.get('min_lr',          cfg.min_lr)
    dropout         = overrides.get('dropout',         cfg.dropout)
    param_file_prefix = overrides.pop('param_file_prefix', None)

    # ── Handle feature_keys override ─────────────────────────────────────
    # If param search overrides feature_keys, re-assemble features on data
    override_fk = overrides.get('feature_keys')
    if override_fk is not None:
        fk_parsed = parse_feature_keys(override_fk)
        fk_tag = compute_feature_tag(fk_parsed)
        # Only re-assemble if features actually changed
        if fk_tag != data.get('assembled_feature_keys', cfg.feature_keys):
            print(f"  Re-assembling features for key override: {fk_tag}")
            assemble_dataset(data['calc_data'], fk_parsed)
            assemble_dataset(data['exp_data'], fk_parsed)
            data['assembled_feature_keys'] = fk_tag
    else:
        fk_tag = cfg.feature_tag

    # Compute a per-config model_id that reflects actual hyperparams
    model_id = (
        f"cebe_{fk_tag}_{split_method}"
        f"_{layer_type}{n_layers}_h{hidden_channels}"
    )

    calc_data = data['calc_data']

    gtu.seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(save_dir,   exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    in_channels = calc_data[0].x.size(1)
    edge_dim = calc_data[0].edge_attr.size(1)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"CEBE MODEL TRAINING — Fold {fold}/{n_folds}")
        print(f"{'=' * 70}")
        print(f"  Layer type:  {layer_type}")
        print(f"  Hidden:      {hidden_channels}")
        print(f"  Layers:      {n_layers}")
        print(f"  LR:          {learning_rate}")
        print(f"  Dropout:     {dropout}")
        print(f"  Batch:       {batch_size}")
        print(f"  Split:       {split_method}")
        print(f"{'=' * 70}")

    n_molecules = len(calc_data)
    if split_method == 'random':
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        folds = list(kf.split(np.arange(n_molecules)))
    elif split_method == 'butina':
        from augernet.build_molecular_graphs import get_butina_clusters
        smiles_list = [d.smiles for d in calc_data]
        cluster_ids = get_butina_clusters(smiles_list, cutoff=0.65)
        n_clusters = len(set(cluster_ids))
        if verbose:
            print(f"  Butina clustering → {n_clusters} clusters "
                  f"(cutoff=0.65)")
        gkf = GroupKFold(n_splits=n_folds)
        folds = list(gkf.split(np.arange(n_molecules),
                                groups=cluster_ids))
    else:
        raise ValueError(
            f"Unknown split_method '{split_method}'. "
            f"Supported: 'random', 'butina'."
        )
    train_idx, val_idx = folds[fold - 1]  # fold is 1-indexed
    train_idx = train_idx.tolist() if hasattr(train_idx, 'tolist') else list(train_idx)
    val_idx = val_idx.tolist() if hasattr(val_idx, 'tolist') else list(val_idx)

    train_data = [calc_data[i] for i in train_idx]
    val_data   = [calc_data[i] for i in val_idx]

    if verbose:
        print(f"  Split → {len(train_data)} train / {len(val_data)} val molecules")

    # ── build model ──────────────────────────────────────────────────────
    model = gtu.MPNN(
        num_layers=n_layers, emb_dim=hidden_channels,
        in_dim=in_channels, edge_dim=edge_dim,
        out_dim=1, layer_type=layer_type, pred_type='CEBE',
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  Parameters:  {n_params:,}")

    # ── train ────────────────────────────────────────────────────────────
    train_results = gtu.train_loop(
        train_data, model, device,
        num_epochs=num_epochs, batch_size=batch_size,
        max_lr=learning_rate,
        verbose=verbose, layer_type=layer_type, pred_type='CEBE',
        val_data_list=val_data, patience=patience,
        optimizer_type=optimizer_type, weight_decay=weight_decay,
        gradient_clip_norm=gradient_clip_norm,
        warmup_epochs=warmup_epochs, min_lr=min_lr,
    )
    model.eval()

    # ── save ─────────────────────────────────────────────────────────────
    model_filename = f"{model_id}_fold{fold}.pth"
    if param_file_prefix:
        model_filename = f"{param_file_prefix}_{model_filename}"
    model_path = os.path.join(save_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    if verbose:
        print(f"\n✓ Saved model  → {model_path}")

    # ── results ──────────────────────────────────────────────────────────
    train_losses = [r[1] for r in train_results]
    val_losses   = [r[2] for r in train_results]
    best_val_idx   = int(np.argmin(val_losses))
    best_val_loss  = val_losses[best_val_idx]
    best_train_loss = train_losses[best_val_idx]

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"FOLD {fold} COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Epochs run:    {len(train_results)}")
        print(f"  Best Val Loss: {best_val_loss:.6f}  "
              f"(train loss: {best_train_loss:.6f}, epoch {best_val_idx + 1})")

    return {
        'model': model,
        'device': device,
        'fold': fold,
        'best_val_loss': best_val_loss,
        'best_train_loss': best_train_loss,
        'best_val_epoch': best_val_idx + 1,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'n_epochs': len(train_results),
        'model_path': model_path,
        'train_results': train_results,
        'model_id': model_id,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_model_from_path(
    model_path: str,
    calc_data: list,
    *,
    layer_type: str,
    hidden_channels: int,
    n_layers: int,
    dropout: float = 0.0,
) -> Tuple[torch.nn.Module, torch.device]:
    """Load a CEBE GNN from a .pth file."""
    from .evaluation_scripts.evaluate_cebe_model import load_model as load_eval_model

    in_channels = calc_data[0].x.size(1)
    edge_dim = calc_data[0].edge_attr.size(1)
    return load_eval_model(
        model_path, in_channels=in_channels, edge_dim=edge_dim,
        layer_type=layer_type, hidden_channels=hidden_channels,
        n_layers=n_layers, dropout=dropout,
    )


def load_saved_model(save_dir, fold, data, cfg):
    """Load a saved CEBE model by fold number."""
    calc_data = data['calc_data']
    model_filename = f"{cfg.model_id}_fold{fold}.pth"
    model_path = os.path.join(save_dir, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved model '{model_filename}' found in:\n  {save_dir}"
        )
    return _load_model_from_path(
        model_path, calc_data,
        layer_type=cfg.layer_type,
        hidden_channels=cfg.hidden_channels,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    )


def load_param_model(model_path, data, cfg, best_params):
    """Load a CEBE model from a param-search result path."""
    calc_data = data['calc_data']

    # If best_params overrode feature_keys, ensure data is assembled correctly
    fk_override = best_params.get('feature_keys')
    if fk_override is not None:
        fk_parsed = parse_feature_keys(fk_override)
        fk_tag = compute_feature_tag(fk_parsed)
        if fk_tag != data.get('assembled_feature_keys', cfg.feature_keys):
            assemble_dataset(calc_data, fk_parsed)
            assemble_dataset(data['exp_data'], fk_parsed)
            data['assembled_feature_keys'] = fk_tag

    return _load_model_from_path(
        model_path, calc_data,
        layer_type=best_params.get('layer_type', cfg.layer_type),
        hidden_channels=best_params.get('hidden_channels', cfg.hidden_channels),
        n_layers=best_params.get('n_layers', cfg.n_layers),
        dropout=best_params.get('dropout', cfg.dropout),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(model_result, data, fold, output_dir, png_dir, cfg,
                   train_results=None, config_id=None, param_file_prefix=None,
                   exp_split=None):
    """Run CEBE evaluation (calls evaluate_cebe_model.run_evaluation).

    Parameters
    ----------
    exp_split : str or None
        Which experimental subset to evaluate on.
        'val'  : validation molecules only (for HP tuning / CV).
        'eval' : held-out evaluation molecules only (blind test).
        'both' : run evaluation separately on val and eval subsets.
        'all'  : use the full experimental set (legacy behaviour).
        If *None*, falls back to ``cfg.exp_split``.
    """

    from .evaluation_scripts.evaluate_cebe_model import run_evaluation as _run_eval

    if isinstance(model_result, dict):
        model = model_result['model']
        device = model_result['device']
        train_results = model_result.get('train_results', train_results)
        model_id = model_result.get('model_id', cfg.model_id)
    elif isinstance(model_result, tuple):
        model, device = model_result
        model_id = cfg.model_id
    else:
        model = model_result
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_id = cfg.model_id

    split = exp_split if exp_split is not None else cfg.exp_split

    def _call(exp_data, suffix=''):
        pfx = param_file_prefix
        if suffix and pfx:
            pfx = f"{pfx}_{suffix}"
        elif suffix:
            pfx = suffix
        return _run_eval(
            model, device, exp_data,
            output_dir=output_dir, fold=fold,
            png_dir=png_dir,
            train_results=train_results,
            norm_stats_file=cfg.norm_stats_file,
            model_id=model_id,
            config_id=config_id,
            param_file_prefix=pfx or None,
        )

    if split == 'val' and data.get('exp_val_data'):
        return _call(data['exp_val_data'])
    elif split == 'eval' and data.get('exp_eval_data'):
        return _call(data['exp_eval_data'])
    elif split == 'both' and data.get('exp_val_data') and data.get('exp_eval_data'):
        val_metrics = _call(data['exp_val_data'], suffix='expval')
        _call(data['exp_eval_data'], suffix='expeval')
        # Return val metrics so upstream selectors (param search, CV) use
        # validation performance for ranking.
        return val_metrics
    else:
        # 'all' or lists not available → full experimental set
        return _call(data['exp_data'])


# ─────────────────────────────────────────────────────────────────────────────
#  Unit tests
# ─────────────────────────────────────────────────────────────────────────────

def run_unit_tests(model, data, cfg):
    """Run GNN symmetry tests on CEBE model."""
    if isinstance(model, dict):
        model = model['model']
    model.eval()
    gtu.run_unit_tests(model, data['calc_data'], layer_type=cfg.layer_type)


# ─────────────────────────────────────────────────────────────────────────────
#  Predict (inference on arbitrary .xyz files)
# ─────────────────────────────────────────────────────────────────────────────

def run_predict(*, model_path: str, predict_dir: str, fold, cfg):
    """Build graphs from .xyz files, run inference, and write output files.

    The model is trained on carbon 1s CEBEs only.  Predictions for
    non-carbon atoms are not meaningful and are marked with ``*`` in
    the labels file.  The numeric ``_results.txt`` contains carbon
    predictions only.

    Produces ``{model_id}_fold{fold}_labels.txt`` and
    ``{model_id}_fold{fold}_results.txt`` in ``predict_results/outputs/``.
    """
    from augernet.build_molecular_graphs import (
        _mol_from_xyz_order,
        _build_node_and_edge_features,
        _initialize_all_atom_encoders,
    )
    from augernet import DATA_RAW_DIR, DATA_PROCESSED_DIR
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    # ── Discover .xyz files ──────────────────────────────────────────────
    xyz_files = sorted(
        f for f in os.listdir(predict_dir) if f.endswith('.xyz')
    )
    if not xyz_files:
        raise FileNotFoundError(
            f"No .xyz files found in: {predict_dir}"
        )

    mol_names = [os.path.splitext(f)[0] for f in xyz_files]

    print(f"\n  Predict directory: {predict_dir}")
    print(f"  Found {len(xyz_files)} .xyz files")

    # ── Build graphs ─────────────────────────────────────────────────────
    skipatom_dir = os.path.join(DATA_RAW_DIR, 'skipatom')
    all_encoders = _initialize_all_atom_encoders(skipatom_dir)

    # Load normalization stats (needed for output denormalization)
    norm_stats_file = cfg.norm_stats_file
    norm_stats = torch.load(norm_stats_file, weights_only=False)
    mean = norm_stats['mean']
    std  = norm_stats['std']

    category_feature = np.array([1, 0, 0])   # CEBE category
    feature_keys = cfg.feature_keys_parsed
    data_list = []

    print(f"  Building molecular graphs...")
    for xyz_file, mol_name in zip(xyz_files, mol_names):
        xyz_path = os.path.join(predict_dir, xyz_file)
        mol, xyz_symbols, pos, smiles = _mol_from_xyz_order(xyz_path, labeled_atoms=False)

        # For prediction we have no CEBE values — pass all -1 sentinels
        n_atoms = mol.GetNumAtoms()
        dummy_cebe = np.full(n_atoms, -1.0)

        node_features, x, edge_index, edge_attr, atomic_be, carbon_env_indices = \
            _build_node_and_edge_features(mol, all_encoders, category_feature, dummy_cebe)

        data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            pos=torch.tensor(pos, dtype=torch.float),
            atomic_be=atomic_be,
            atom_symbols=xyz_symbols,
            smiles=smiles,
            mol_name=mol_name,
        )

        # Attach all feat_* attributes
        for attr_name, tensor in node_features.items():
            setattr(data, attr_name, tensor)

        data_list.append(data)

    print(f"  Assembled {len(data_list)} graphs")

    # ── Assemble features ────────────────────────────────────────────────
    print(f"  Assembling features {cfg.feature_keys}")
    assemble_dataset(data_list, feature_keys)

    # ── Load model ───────────────────────────────────────────────────────
    in_channels = data_list[0].x.size(1)
    edge_dim = data_list[0].edge_attr.size(1)
    model, device = _load_model_from_path(
        model_path, data_list,
        layer_type=cfg.layer_type,
        hidden_channels=cfg.hidden_channels,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    )

    # ── Inference ────────────────────────────────────────────────────────
    output_dir = os.path.join(cfg.predict_output_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    file_stem = f"{cfg.model_id}_fold{fold}" if fold is not None else cfg.model_id

    print(f"\n{'=' * 80}")
    print(f"  PREDICT: Running inference on {len(data_list)} molecules")
    print(f"  NOTE: Model is trained on carbon 1s CEBEs only.")
    print(f"        Predictions for non-carbon atoms are not meaningful.")
    print(f"{'=' * 80}")

    loader = DataLoader(data_list, batch_size=1, shuffle=False)

    all_pred = []
    all_atoms = []
    molecule_results = {}

    for i, data in enumerate(loader):
        # Check for isolated nodes (bond dissociation beyond RDKit cutoff)
        n_nodes = data.x.size(0)
        nodes_in_edges = set(
            data.edge_index[0].tolist() + data.edge_index[1].tolist()
        )
        if len(nodes_in_edges) < n_nodes:
            # Skip — graph is disconnected; inference would fail
            mol_name_raw = data.mol_name
            if isinstance(mol_name_raw, list):
                mol_name_raw = mol_name_raw[0]
            atom_syms = data.atom_symbols
            if isinstance(atom_syms, list):
                atom_syms = atom_syms[0]
            atom_syms = [str(s).strip() for s in atom_syms]
            mol_rows = [(sym, float('nan')) for sym in atom_syms]
            molecule_results[mol_name_raw] = mol_rows
            all_pred.extend([float('nan')] * n_nodes)
            all_atoms.extend(atom_syms)
            print(f"    ⚠  Skipping {mol_name_raw}: disconnected graph "
                  f"({len(nodes_in_edges)}/{n_nodes} nodes in edges)")
            continue

        data = data.to(device)
        with torch.no_grad():
            out = model(data)

        pred_out = out.cpu().numpy()
        atomic_be_vals = data.atomic_be.cpu().numpy()

        # atom_symbols and mol_name are already on the graph from build step
        atom_syms = data.atom_symbols
        # If DataLoader wrapped it in another list, unwrap it
        if isinstance(atom_syms, list):
            atom_syms = atom_syms[0]  # Unwrap the outer list
         # Ensure all elements are strings
        atom_syms = [str(s).strip() for s in atom_syms]

        mol_name_raw = data.mol_name
        if isinstance(mol_name_raw, list):
            mol_name_raw = mol_name_raw[0]  # DataLoader wraps strings in list
        mol_name = mol_name_raw
        mol_rows = []

        for j in range(len(pred_out)):
            sym = atom_syms[j] if j < len(atom_syms) else '?'
            ref = atomic_be_vals[j]

            # Denormalize: predicted CEBE = atomic_BE - (pred * std + mean)
            pred_be = float(ref - (pred_out[j] * std + mean))

            mol_rows.append((sym, pred_be))
            all_pred.append(pred_be)
            all_atoms.append(sym)

        molecule_results[mol_name] = mol_rows

    # ── Write _labels.txt ────────────────────────────────────────────────
    label_path = os.path.join(output_dir, f"{file_stem}_labels.txt")
    with open(label_path, 'w') as f:
        f.write(f"# CEBE Predictions\n")
        f.write(f"# Model: {cfg.model_id}\n")
        f.write(f"# NOTE: Only carbon (C) predictions are meaningful.\n")
        f.write(f"#       Non-carbon rows are marked with * and should be ignored.\n")
        f.write(f"# Columns: atom_symbol  pred_BE(eV)\n")
        f.write(f"#\n")
        for mol_name, rows in molecule_results.items():
            f.write(f"# --- {mol_name} ---\n")
            for sym, pred_be in rows:
                marker = ' ' if sym == 'C' else '*'
                f.write(f"{sym:>3s}{marker}   {pred_be:10.4f}\n")
            f.write(f"\n")
    print(f"  Label results saved to {label_path}")

    # ── Write _results.txt (numeric, carbon atoms only) ──────────────────
    carbon_preds = [p for s, p in zip(all_atoms, all_pred) if s == 'C']
    results_path = os.path.join(output_dir, f"{file_stem}_results.txt")
    np.savetxt(results_path, np.array(carbon_preds).reshape(-1, 1))
    print(f"  Numeric results saved to {results_path}")

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'Molecule':<22s} {'N_C':>5s} {'N_atoms':>8s} {'Mean C_1s (eV)':>15s}")
    print("-" * 55)
    for mol_name, rows in molecule_results.items():
        c_preds = [p for s, p in rows if s == 'C']
        n_c = len(c_preds)
        n_tot = len(rows)
        mean_c = np.mean(c_preds) if c_preds else float('nan')
        print(f"{mol_name:<22s} {n_c:>5d} {n_tot:>8d} {mean_c:>15.4f}")
