"""
CNN Training Utilities for Auger Spectra Classification

Contains:
- CNN model architecture (AugerCNN1D — flexible, dict-configured)
- Cross entropy loss function with inverse-frequency class weights
- Dataset wrapper
- Trainer class with early stopping
- Evaluation with molecule-by-molecule details
- Diagnostics

"""

import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score

from augernet import carbon_dataframe as cdf
from augernet.carbon_environment import (
    CARBON_ENVIRONMENT_PATTERNS,
    CARBON_ENV_TO_IDX,
    IDX_TO_CARBON_ENV,
    NUM_CARBON_CATEGORIES,
)

# =============================================================================
# CONSTANTS
# =============================================================================

CARBON_ENVIRONMENT_NAMES = list(CARBON_ENVIRONMENT_PATTERNS.keys())
NUM_CARBON_CLASSES = NUM_CARBON_CATEGORIES


# =============================================================================
# SEED
# =============================================================================

def seed(seed_val=0):
    """Set all random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed_val)
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



# =============================================================================
# DEVICE UTILITIES
# =============================================================================

def get_device(device_str: str = 'auto', verbose: bool = True) -> torch.device:
    """Get PyTorch device with fallback support (CUDA > MPS > CPU)."""
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            if verbose:
                print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            if verbose:
                print("Using device: mps (Apple Silicon)")
        else:
            device = torch.device('cpu')
            if verbose:
                print("Using device: cpu")
    else:
        device = torch.device(device_str)
        if verbose:
            print(f"Using device: {device_str}")
    return device


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class AugerCNN1D(nn.Module):
    """
    Flexible 1D CNN for Auger spectra — architecture fully defined by a dict.

    Parameters
    ----------
    input_length int  : Length of the input spectrum (+ optional extra features).
    num_classes  int  :  Number of output classes.
    architecture dict :
        Keys:
            conv_filters    list[int] : filters per conv block, e.g. [16, 32, 64]
            conv_kernels    list[int] : kernel size per block,  e.g. [7, 5, 3]
            pool_size       int       : max-pool kernel (default 4)
            fc_hidden       list[int] : hidden FC layer sizes,  e.g. [128]
            use_batch_norm  bool      : BatchNorm after each conv (default False)
            dropout         float     : Dropout rate for fully-connected layers.
            dropout_conv    float     : dropout after each conv block (default 0.0)
    """

    def __init__(self, input_length: int, num_classes: int,
                 architecture: dict):
        super().__init__()

        filters   = architecture['conv_filters']
        kernels   = architecture['conv_kernels']
        pool_size = architecture.get('pool_size', 4)
        fc_hidden = architecture.get('fc_hidden', [64])
        use_bn    = architecture.get('use_batch_norm', False)
        dropout   = architecture.get('dropout', 0.2)
        drop_conv = architecture.get('dropout_conv', 0.0)

        n_blocks = len(filters)
        if len(kernels) != n_blocks:
            raise ValueError(
                f"conv_filters has {n_blocks} entries but "
                f"conv_kernels has {len(kernels)}"
            )

        self.architecture = architecture

        # ---- build conv blocks ------------------------------------------------
        self.conv_blocks = nn.ModuleList()
        in_ch = 1
        for out_ch, ks in zip(filters, kernels):
            block = nn.Sequential()
            block.append(nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=ks // 2))
            if use_bn:
                block.append(nn.BatchNorm1d(out_ch))
            block.append(nn.ReLU())
            if drop_conv > 0:
                block.append(nn.Dropout(drop_conv))
            self.conv_blocks.append(block)
            in_ch = out_ch

        self.pool = nn.MaxPool1d(pool_size)

        # ---- compute flattened size -------------------------------------------
        conv_out_len = input_length
        for _ in range(n_blocks):
            conv_out_len = conv_out_len // pool_size
        self.flat_size = filters[-1] * conv_out_len

        # ---- build FC layers --------------------------------------------------
        self.fc_layers = nn.ModuleList()
        prev = self.flat_size
        for h in fc_hidden:
            self.fc_layers.append(nn.Linear(prev, h))
            prev = h
        self.fc_out = nn.Linear(prev, num_classes)
        self.dropout_fc = nn.Dropout(dropout)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x):
        # x shape: (batch, 1, input_length)
        for block in self.conv_blocks:
            x = self.pool(block(x))
        x = x.flatten(1)

        for fc in self.fc_layers:
            x = self.dropout_fc(F.relu(fc(x)))
        x = self.fc_out(x)
        return x

# =============================================================================
# DATASET WRAPPER
# =============================================================================

class CarbonLabelDataset(Dataset):
    """Wrapper dataset that provides carbon environment labels."""

    def __init__(self, base_dataset: cdf.CarbonDataset, df: pd.DataFrame):
        self.base_dataset = base_dataset
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        if isinstance(item, tuple):
            spectrum, _ = item
        else:
            spectrum = item['spectrum']
        label = self.df.iloc[idx]['carbon_env_label']
        return spectrum, torch.LongTensor([label]).squeeze()


# =============================================================================
# TRAINER CLASS
# =============================================================================

class CNNTrainer:
    """Training and evaluation handler for Auger CNN models."""

    def __init__(self, model: nn.Module, device: torch.device,
                 learning_rate: float = 5e-4, weight_decay: float = 1e-4,
                 patience: int = 20, use_cosine_schedule: bool = False,
                 cosine_T_max: int = None,
                 class_weights: torch.Tensor = None):
        self.model = model.to(device)
        self.device = device
        self.patience = patience
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # ---- optional cosine-annealing LR schedule ---------------------------
        self.use_cosine_schedule = use_cosine_schedule
        self.scheduler = None
        if use_cosine_schedule:
            T_max = cosine_T_max if cosine_T_max is not None else 500
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=1e-6
            )
            print(f"  Using CosineAnnealingLR schedule (T_max={T_max})")

        # ---- loss function ---------------------------------------------------
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(device),
            )
            print(f"  Using CrossEntropyLoss with inverse-frequency class weights")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print(f"  Using CrossEntropyLoss with no class weights")

        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        self.best_model_state = None

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            if isinstance(batch, dict):
                spectra = batch['spectrum'].to(self.device, dtype=torch.float32)
                labels = batch['label'].to(self.device, dtype=torch.long)
            else:
                spectra, labels = batch
                spectra = spectra.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)
            if spectra.dim() == 2:
                spectra = spectra.unsqueeze(1)

            self.optimizer.zero_grad()

            logits = self.model(spectra)
            loss = self.criterion(logits, labels)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            total += labels.size(0)
        return total_loss / len(train_loader), 100.0 * correct / total

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    spectra = batch['spectrum'].to(self.device, dtype=torch.float32)
                    labels = batch['label'].to(self.device, dtype=torch.long)
                else:
                    spectra, labels = batch
                    spectra = spectra.to(self.device, dtype=torch.float32)
                    labels = labels.to(self.device, dtype=torch.long)
                if spectra.dim() == 2:
                    spectra = spectra.unsqueeze(1)
                logits = self.model(spectra)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        return total_loss / len(val_loader), 100.0 * correct / total

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            num_epochs: int = 100, verbose: bool = True) -> Dict[str, List[float]]:
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            # Step the LR scheduler (if active)
            if self.scheduler is not None:
                self.scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            if verbose:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return self.history


# =============================================================================
# CLASS WEIGHTS
# =============================================================================

def get_class_weights_and_counts(
    df: pd.DataFrame,
    num_classes: int = None,
) -> Tuple[torch.Tensor, Dict[int, int]]:
    """Compute inverse-frequency class weights and per-class counts.

    Returns
    -------
    weights : torch.Tensor
        Shape ``(n_classes,)``.  Inverse-frequency weights normalised so that
        the active (non-zero) entries have unit mean.
    counts : Dict[int, int]
        ``{class_idx: sample_count}`` for every class index in
        ``range(n_classes)``, with ``0`` for classes absent from *df*.
    """
    label_col = 'carbon_env_label'
    n_classes = NUM_CARBON_CLASSES if num_classes is None else num_classes

    raw_counts = df[label_col].value_counts().to_dict()
    counts = {i: raw_counts.get(i, 0) for i in range(n_classes)}

    weights = torch.zeros(n_classes, dtype=torch.float32)
    total_samples = len(df)
    for class_idx, count in raw_counts.items():
        if 0 <= class_idx < n_classes:
            weights[class_idx] = total_samples / (n_classes * count)
    active_mask = weights > 0
    if active_mask.sum() > 0:
        weights[active_mask] = weights[active_mask] / weights[active_mask].mean()

    return weights, counts


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def diagnose_dataframe(df: pd.DataFrame) -> None:
    """Run diagnostics on the carbon DataFrame."""
    print("\n" + "=" * 70)
    print("CARBON DATAFRAME DIAGNOSTICS")
    print("=" * 70)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Total carbon atoms: {len(df)}")
    print(f"Total molecules: {df['mol_name'].nunique()}")

    class_dist = df['carbon_env_label'].value_counts().sort_index()
    print(f"\nClass distribution ({len(class_dist)} active classes):")

    print(f"\n  Unique classes: {len(class_dist)}")
    print(f"  Min class count: {class_dist.min()}")
    print(f"  Max class count: {class_dist.max()}")
    print(f"  Imbalance ratio: {class_dist.max() / class_dist.min():.1f}x")

    # Detect stick columns (new format: sing/trip separate; old: combined)
    has_sing = 'sing_stick_energies' in df.columns
    has_combined = 'stick_energies' in df.columns

    if has_sing:
        se = df['sing_stick_energies']
        te = df['trip_stick_energies']
        n_peaks = [len(s) + len(t) for s, t in zip(se, te)]
        print(f"\nSpectrum type: STICK (sing+trip, broadened on-the-fly)")
        print(f"  Peaks per atom: min={min(n_peaks)}, max={max(n_peaks)}, "
              f"mean={np.mean(n_peaks):.1f}")
        all_e = np.concatenate([e for e in se if e is not None and len(e) > 0]
                               + [e for e in te if e is not None and len(e) > 0])
        print(f"  Energy range: [{all_e.min():.1f}, {all_e.max():.1f}] eV")
    elif has_combined:
        n_peaks = [len(e) for e in df['stick_energies'].values if e is not None]
        print(f"\nSpectrum type: STICK (combined, broadened on-the-fly)")
        print(f"  Peaks per atom: min={min(n_peaks)}, max={max(n_peaks)}, "
              f"mean={np.mean(n_peaks):.1f}")
        all_e = np.concatenate([e for e in df['stick_energies'] if e is not None and len(e) > 0])
        print(f"  Energy range: [{all_e.min():.1f}, {all_e.max():.1f}] eV")
    elif 'spectrum_intensity_only' in df.columns:
        spectra = np.stack(df['spectrum_intensity_only'].values)
        print(f"\nSpectrum type: PRE-BROADENED")
        print(f"  Shape: {spectra.shape}")
        print(f"  Min: {spectra.min():.4f}, Max: {spectra.max():.4f}")

    print("=" * 70)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_with_molecule_details(
    df: pd.DataFrame, model: nn.Module, device: torch.device,
    dataset: Dataset, output_dir: str = None,
    eval_type: str = 'test', csv_suffix: str = '',
    class_names_override: list = None,
    num_classes_override: int = None,
) -> Dict[str, Any]:
    """Evaluate model with detailed molecule-by-molecule results and CSV output."""
    model.eval()

    label_col = 'carbon_env_label'
    class_names = class_names_override if class_names_override else CARBON_ENVIRONMENT_NAMES
    num_classes = num_classes_override if num_classes_override else NUM_CARBON_CLASSES

    all_preds, all_probs, all_labels = [], [], []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                spectra, labels = batch[0], batch[1]
            elif isinstance(batch, dict):
                spectra, labels = batch['spectrum'], batch['label']
            else:
                raise ValueError(f"Unknown batch type: {type(batch)}")
            spectra = spectra.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            if spectra.dim() == 2:
                spectra = spectra.unsqueeze(1)
            logits = model(spectra)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            all_preds.append(pred.cpu().item())
            all_probs.append(probs.cpu().numpy()[0])
            all_labels.append(labels.cpu().item())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print("\n" + "=" * 130)
    print(f"EVALUATION ({eval_type.upper()}) - {num_classes} CLASSES")
    print("=" * 130)

    df_reset = df.reset_index(drop=True)
    molecules = df_reset.groupby('mol_name')
    csv_records = []

    for mol_name, mol_group_indices in molecules.groups.items():
        mol_group = df_reset.loc[mol_group_indices].sort_values('atom_idx')
        print("\n" + "-" * 160)
        print(f"Molecule: {mol_name}  |  Carbons: {len(mol_group)}")
        print("-" * 160)

        print(f"{'XYZ Idx':<10} {'True Category':<50} {'Pred Category':<50} "
              f"{'Conf':<10} {'Status':<10}")
        print("-" * 160)
        for idx in mol_group_indices:
            row = df_reset.iloc[idx]
            atom_idx = int(row['atom_idx'])
            true_label = int(row[label_col])
            pred_label = int(all_preds[idx])
            confidence = all_probs[idx][pred_label] * 100
            # Use class_names list (works for both original and merged labels)
            true_name = class_names[true_label] if true_label < len(class_names) else f"class_{true_label}"
            pred_name = class_names[pred_label] if pred_label < len(class_names) else f"class_{pred_label}"
            status = "CORRECT" if pred_label == true_label else "WRONG"
            mark = "✓" if status == "CORRECT" else "✗"
            print(f"{atom_idx:<10} {true_name:<50} {pred_name:<50} "
                  f"{confidence:>6.1f}%   {mark} {status}")
            # Strip 'C_' prefix for cleaner CSV output
            true_display = true_name.removeprefix('C_')
            pred_display = pred_name.removeprefix('C_')
            csv_records.append({
                'Molecule': mol_name,
                'True Environment': true_display,
                'CNN Prediction (Confidence %)': f"{pred_display} ({confidence:.1f}%)",
            })

    if output_dir:
        csv_df = pd.DataFrame(csv_records)
        # Drop duplicate rows (same molecule, same environment, same prediction+confidence)
        #csv_df = csv_df.drop_duplicates().reset_index(drop=True)
        csv_path = Path(output_dir) / f"eval_{eval_type}{csv_suffix}.csv"
        csv_df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved evaluation CSV: {csv_path}")
        print(f"  Rows: {len(csv_df)} (deduplicated from {len(csv_records)} atoms)")

    # ---- Deduplicated metrics -------------------------------------------------
    # Collapse atoms that are truly identical: same molecule, same true label,
    # same predicted label.  This avoids double-counting symmetry-
    # equivalent carbons (e.g. benzene 6× aromatic = 1 entry) while still
    # keeping atoms that share a label but have distinguishable spectra and
    # different predictions (e.g. pyrimidine C2 vs C4/C6, all C_arom_N but
    # C2 sits between two nitrogens and may be predicted differently).
    dedup_set = set()  # (mol_name, true_label, pred_label)
    dedup_labels = []
    dedup_preds = []
    for mol_name, mol_group_indices in molecules.groups.items():
        for idx in mol_group_indices:
            true_label = int(df_reset.iloc[idx][label_col])
            pred_label = int(all_preds[idx])
            key = (mol_name, true_label, pred_label)
            if key not in dedup_set:
                dedup_set.add(key)
                dedup_labels.append(true_label)
                dedup_preds.append(pred_label)

    dedup_labels = np.array(dedup_labels)
    dedup_preds = np.array(dedup_preds)
    # Suppress sklearn warning when #classes > 50% of #samples (small eval sets)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dedup_accuracy = accuracy_score(dedup_labels, dedup_preds)
        dedup_f1_macro = f1_score(dedup_labels, dedup_preds, average='macro', zero_division=0)
        dedup_f1_weighted = f1_score(dedup_labels, dedup_preds, average='weighted', zero_division=0)

    print("\n" + "=" * 160)
    print("SUMMARY STATISTICS")
    print("=" * 160)

    print(f"\n  Per-atom metrics (all {len(all_labels)} atoms):")
    print(f"    Accuracy:    {accuracy*100:.1f}%  ({(all_preds == all_labels).sum()}/{len(all_labels)})")
    print(f"    F1-Macro:    {f1_macro:.4f}")
    print(f"    F1-Weighted: {f1_weighted:.4f}")

    print(f"\n  Deduplicated metrics ({len(dedup_labels)} unique environment–molecule pairs):")
    print(f"    Accuracy:    {dedup_accuracy*100:.1f}%  ({(dedup_preds == dedup_labels).sum()}/{len(dedup_labels)})")
    print(f"    F1-Macro:    {dedup_f1_macro:.4f}")
    print(f"    F1-Weighted: {dedup_f1_weighted:.4f}")

    print("=" * 160)

    print("\nPer-Class Performance (deduplicated):")
    print("-" * 100)
    for label in sorted(np.unique(dedup_labels)):
        mask = dedup_labels == label
        support = mask.sum()
        correct = ((dedup_preds == dedup_labels) & mask).sum()
        acc = correct / support if support > 0 else 0
        name = class_names[label] if label < len(class_names) else f"class_{label}"
        print(f"  {name:40s}: {correct:3d}/{support:3d} ({acc*100:5.1f}%)")

    return {
        'accuracy': accuracy, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted,
        'dedup_accuracy': dedup_accuracy, 'dedup_f1_macro': dedup_f1_macro,
        'dedup_f1_weighted': dedup_f1_weighted,
        'predictions': all_preds, 'labels': all_labels, 'probabilities': all_probs,
        'dedup_predictions': dedup_preds, 'dedup_labels': dedup_labels,
    }


# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

def plot_training_history(history: Dict[str, List[float]], output_dir: str) -> None:
    """Save training loss/accuracy plots."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(history['train_acc'], label='Train', linewidth=2)
        axes[0, 1].plot(history['val_acc'], label='Val', linewidth=2)
        axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(history['train_loss'], label='Train', linewidth=2)
        axes[1, 0].plot(history['val_loss'], label='Val', linewidth=2)
        axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Loss (Last 50 Epochs)')
        axes[1, 0].set_xlim(max(0, len(history['val_loss']) - 50), len(history['val_loss']))
        axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

        best_val_loss_idx = np.argmin(history['val_loss'])
        best_val_acc_idx = np.argmax(history['val_acc'])
        axes[1, 1].axis('off')
        metrics_text = (
            f"  Best Val Loss: {min(history['val_loss']):.4f} (Epoch {best_val_loss_idx + 1})\n"
            f"  Best Val Acc:  {max(history['val_acc']):.2f}% (Epoch {best_val_acc_idx + 1})\n\n"
            f"  Final Train Acc: {history['train_acc'][-1]:.2f}%\n"
            f"  Final Val Acc:   {history['val_acc'][-1]:.2f}%\n\n"
            f"  Total Epochs: {len(history['train_loss'])}"
        )
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                        family='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plot_path = Path(output_dir) / 'training_plots.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved training plots: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"⚠ Failed to save plots: {e}")
