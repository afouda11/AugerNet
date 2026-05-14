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

class FiLMLayer(nn.Module):
    """Feature-wise affine transform: y = gamma * x + beta"""
    def forward(self, x, gamma, beta):
        return gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)

class FiLMGenerator(nn.Module):
    """ Maps z-score normalised conditioning inputs to gamma and beta parameters
        for every FiLM layer in the FiLM'd network.
        film_dim: 1 (be only or mol_size only) or 2 (both)
    """
    def __init__(self, channels_per_layer: list, hidden_dim: int = 64, film_dim: int = 2):
        super().__init__()
        #one gamma and one beta per channel, per FiLM layer
        total = 2 * sum(channels_per_layer)
        self.channels_per_layer = channels_per_layer
        self.mlp = nn.Sequential(
            nn.Linear(film_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, total),
        )
        # zero last layer so gamma starts at 1 and beta 0, so CNN starts as un-FiLM'd
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, film_cond: torch.Tensor):
        # film_cond: (B, 2) = [delta_be_norm, mol_size_norm]
        out = self.mlp(film_cond)
        # split into (gamma_i, beta_i) pairs per layer
        sizes = []
        for c in self.channels_per_layer:
            sizes.extend([c, c])
        parts = torch.split(out, sizes, dim=1)
        # gamma = 1 + delta (so identity init means gamma == 1)
        return [(1.0 + parts[2*i], parts[2*i + 1])
                for i in range(len(self.channels_per_layer))]

class AugerCNN1D_FiLMd(nn.Module):
    def __init__(self,
        input_length: int,
        num_classes: int,
        parallel_kernel_sizes=(5,10,15),
        parallel_filters=(12,12,12),
        sequential_kernel_size=(15,15),
        sequential_filters=(12,12),
        conv_dropout=0.2,
        pool_kernel=32,
        pool_stride=2,
        film_hidden=64,
        film_inputs='both',   # 'none' | 'be' | 'mol_size' | 'both'
    ):
        super().__init__()

        # FiLM conditioning mode
        _valid = ('none', 'be', 'mol_size', 'both')
        if film_inputs not in _valid:
            raise ValueError(f"film_inputs must be one of {_valid}, got '{film_inputs}'")
        self.film_inputs = film_inputs
        film_dim = {'none': 0, 'be': 1, 'mol_size': 1, 'both': 2}[film_inputs]

        self.parallel_convs = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=f, kernel_size=k, 
                      stride=1, padding='same') for f, k in zip(parallel_filters, parallel_kernel_sizes)
        ])
        concat_channels = sum(parallel_filters)

        self.seqconv1 = nn.Conv1d(in_channels=concat_channels, out_channels=sequential_filters[0], 
                             kernel_size=sequential_kernel_size[0], stride=1, padding='same')
        self.seqconv2 = nn.Conv1d(in_channels=sequential_filters[0], out_channels=sequential_filters[1], 
                             kernel_size=sequential_kernel_size[1], stride=1, padding='same')
        
        # FiLM bits
        # first try modulating after parallel convs and after each seq conv
        film_channel_counts = [concat_channels, sequential_filters[0], sequential_filters[1]]

        if film_dim > 0:
            self.film_generator = FiLMGenerator(film_channel_counts, hidden_dim=film_hidden, film_dim=film_dim)
            self.film_layers = nn.ModuleList([FiLMLayer() for _ in film_channel_counts])
        else:
            self.film_generator = None
            self.film_layers = None

        self.conv_dropout = nn.Dropout(conv_dropout)
        self.pool = nn.AdaptiveAvgPool1d(pool_kernel)
        # AdaptiveAvgPool1d always outputs exactly pool_kernel time-steps
        flat_size = sequential_filters[1] * pool_kernel
        print(f"AugerCNN1D_FiLMd: film_inputs='{film_inputs}', film_dim={film_dim}, flat_size={flat_size}")
        self.fc = nn.Linear(flat_size, num_classes)

    def forward(self, x: torch.Tensor, film_cond: torch.Tensor) -> torch.Tensor:

        # x: (B, 1, spec)
        # film_cond: (B, 2) = [delta_be_norm, mol_size_norm]  (always passed full)
        # Select the conditioning columns based on film_inputs config:
        if self.film_inputs == 'none':
            cond = None
        elif self.film_inputs == 'be':
            cond = film_cond[:, 0:1]   # (B, 1)
        elif self.film_inputs == 'mol_size':
            cond = film_cond[:, 1:2]   # (B, 1)
        else:  # 'both'
            cond = film_cond           # (B, 2)

        film = self.film_generator(cond) if cond is not None else None

        # parallel branches -> ReLU -> concat -> FiLM -> dropout
        parallel_outs = [F.relu(conv(x)) for conv in self.parallel_convs]
        x = torch.cat(parallel_outs, dim=1)
        if film is not None:
            g, b = film[0]
            x = self.film_layers[0](x, g, b)
        x = self.conv_dropout(x)

        # seqconv1 -> FiLM -> RelU -> dropout
        x = self.seqconv1(x)
        if film is not None:
            g, b = film[1]
            x = self.film_layers[1](x, g, b)
        x = F.relu(x)
        x = self.conv_dropout(x)

        # seqconv2 -> FiLM -> RelU -> dropout
        x = self.seqconv2(x)
        if film is not None:
            g, b = film[2]
            x = self.film_layers[2](x, g, b)
        x = F.relu(x)
        x = self.conv_dropout(x)

        # pool, flatten, classify
        # AdaptiveAvgPool1d is not supported on MPS when input is not
        # evenly divisible by output size, so fall back to CPU for this op.
        if x.device.type == 'mps':
            x = self.pool(x.cpu()).to(x.device)
        else:
            x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x


class AugerCNN1D(nn.Module):
    def __init__(self,
                 input_length: int, 
                 num_classes: int,
                 parralel_kernal_size: tuple = (5, 10, 15),
                 parralel_filters: tuple = (12, 12, 12),
                 sequential_kernel_size: tuple = (15, 15),
                 sequential_filters: tuple = (12, 12),
                 conv_dropout: float = 0.2,
                 pool_kernel:         int = 2,
                 pool_stride:         int = 2,
                 ):
        super().__init__()
        self.parallel_convs = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=f, kernel_size=k, 
                      stride=1, padding='same') for f, k in zip(parralel_filters, parralel_kernal_size)
        ])

        concat_channels = sum(parralel_filters)

        self.seqconv1 = nn.Conv1d(in_channels=concat_channels, out_channels=sequential_filters[0], 
                             kernel_size=sequential_kernel_size[0], stride=1, padding='same')
        self.seqconv2 = nn.Conv1d(in_channels=sequential_filters[0], out_channels=sequential_filters[1], 
                             kernel_size=sequential_kernel_size[1], stride=1, padding='same')

        self.conv_dropout = nn.Dropout(conv_dropout)
        self.pool = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride)
        pooled_length = (input_length - pool_kernel) // pool_stride + 1
        flat_size = sequential_filters[1] * pooled_length
        self.fc = nn.Linear(flat_size, num_classes)

        self.seq_block = nn.Sequential(
            self.seqconv1,
            nn.ReLU(),
            self.conv_dropout,
            self.seqconv2,
            nn.ReLU(),
            self.conv_dropout,
            self.pool,
            nn.Flatten(),
            self.fc,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        parralel_outs = [F.relu(conv(x)) for conv in self.parallel_convs]
        x = torch.cat(parralel_outs, dim=1)
        x = self.conv_dropout(x)
        x = self.seq_block(x)
        return x



# =============================================================================
# TRAINER CLASS
# =============================================================================

class CNNTrainer:
    """Training and evaluation handler for Auger CNN models."""

    def __init__(self, model: nn.Module, device: torch.device,
                 learning_rate: float = 5e-4, weight_decay: float = 1e-4,
                 patience: int = 20, scheduler_type: str = 'cosine',
                 cosine_T_max: int = None,
                 class_weights: torch.Tensor = None,
                 label_smoothing: float = 0.0,
                 noise_std: float = 0.0):

        self.model = model.to(device)
        self.device = device
        self.patience = patience
        self.noise_std = noise_std
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # ---- LR schedule -----------------------------------------------------
        self.scheduler = None
        self.scheduler_per_batch = False
        if scheduler_type == 'onecycle':
            # OneCycleLR requires train_loader length — deferred to fit()
            self._onecycle_max_lr = learning_rate
            self._onecycle_cosine_T_max = cosine_T_max  # used as num_epochs
            self.scheduler_per_batch = True
            print(f"  Scheduler: OneCycleLR  (will be initialised at fit())")
        else:
            # CosineAnnealingLR — per-epoch
            T_max = cosine_T_max if cosine_T_max is not None else 500
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=1e-6
            )
            print(f"  Using CosineAnnealingLR schedule (T_max={T_max})")

        # ---- loss function ---------------------------------------------------
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(device),
                label_smoothing=label_smoothing,
            )
            print(f"  Using CrossEntropyLoss with inverse-frequency class weights"
                  f"{f' + label_smoothing={label_smoothing}' if label_smoothing else ''}")
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            print(f"  Using CrossEntropyLoss with no class weights"
                  f"{f' + label_smoothing={label_smoothing}' if label_smoothing else ''}")

        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
        self.best_model_state = None

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
#             if isinstance(batch, dict):
#                 spectra = batch['spectrum'].to(self.device, dtype=torch.float32)
#                 labels = batch['label'].to(self.device, dtype=torch.long)
#             else:
            spectra, delta_be, mol_size, labels = batch
            spectra = spectra.to(self.device, dtype=torch.float32)
            delta_be = delta_be.to(self.device, dtype=torch.float32)
            mol_size = mol_size.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)
            if spectra.dim() == 2:
                spectra = spectra.unsqueeze(1)

            # Online augmentation: Gaussian noise (train only)
            if self.noise_std > 0.0:
                spectra = spectra + torch.randn_like(spectra) * self.noise_std
                spectra = spectra.clamp(min=0.0)

            self.optimizer.zero_grad()

            film_cond = torch.stack([delta_be, mol_size], dim=1)  # (B, 2)
            logits = self.model(spectra, film_cond)
            loss = self.criterion(logits, labels)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Per-batch LR step (OneCycleLR)
            if self.scheduler_per_batch and self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            total += labels.size(0)
        return total_loss / len(train_loader), 100.0 * correct / total

    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                spectra, delta_be, mol_size, labels = batch
                spectra = spectra.to(self.device, dtype=torch.float32)
                delta_be = delta_be.to(self.device, dtype=torch.float32)
                mol_size = mol_size.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)
                if spectra.dim() == 2:
                    spectra = spectra.unsqueeze(1)
                film_cond = torch.stack([delta_be, mol_size], dim=1)  # (B, 2)
                logits = self.model(spectra, film_cond)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                all_preds.append(predicted.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        val_f1 = f1_score(
            np.concatenate(all_labels), np.concatenate(all_preds),
            average='macro', zero_division=0,
        )
        return total_loss / len(val_loader), 100.0 * correct / total, val_f1

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            num_epochs: int = 100, verbose: bool = True) -> Dict[str, List[float]]:
        # Deferred OneCycleLR init (needs train_loader length)
        if self.scheduler_per_batch and self.scheduler is None:
            from torch.optim.lr_scheduler import OneCycleLR
            epochs = getattr(self, '_onecycle_cosine_T_max', None) or num_epochs
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self._onecycle_max_lr,
                steps_per_epoch=len(train_loader),
                epochs=epochs,
            )
            total_steps = len(train_loader) * epochs
            if verbose:
                print(f"  Scheduler: OneCycleLR  (per-batch, {total_steps} total steps)")

        best_val_f1 = 0.0
        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_f1 = self.validate(val_loader)

            # Per-epoch LR step (CosineAnnealingLR)
            if not self.scheduler_per_batch and self.scheduler is not None:
                self.scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)

            if verbose:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                      f"Val F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
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

    label_col = 'carbon_env_index'
    class_names = class_names_override if class_names_override else CARBON_ENVIRONMENT_NAMES
    num_classes = num_classes_override if num_classes_override else NUM_CARBON_CLASSES

    all_preds, all_probs, all_labels = [], [], []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch in loader:
            spectra, delta_be, mol_size, labels = batch
            spectra = spectra.to(device, dtype=torch.float32)
            delta_be = delta_be.to(device, dtype=torch.float32)
            mol_size = mol_size.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            if spectra.dim() == 2:
                spectra = spectra.unsqueeze(1)
            film_cond = torch.stack([delta_be, mol_size], dim=1)  # (B, 2)
            logits = model(spectra, film_cond)
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

    print(f"\n== EVALUATION ({eval_type.upper()}) - {num_classes} CLASSES ==")

    df_reset = df.reset_index(drop=True)
    molecules = df_reset.groupby('mol_name')
    csv_records = []

    # Compact per-molecule table: one header, one row per molecule
    col_w = 38
    print(f"\n  {'Molecule':<30} {'C':>4} {'ok':>4} {'wrong':>6}   Misclassified (true -> pred)")
    print("  " + "-" * 110)
    for mol_name, mol_group_indices in molecules.groups.items():
        mol_group = df_reset.loc[mol_group_indices].sort_values('atom_idx')
        n_carbon = len(mol_group)
        wrong_items = []
        for idx in mol_group_indices:
            row = df_reset.iloc[idx]
            atom_idx  = int(row['atom_idx'])
            true_label = int(row[label_col])
            pred_label = int(all_preds[idx])
            confidence = all_probs[idx][pred_label] * 100
            true_name = class_names[true_label] if true_label < len(class_names) else f"class_{true_label}"
            pred_name = class_names[pred_label] if pred_label < len(class_names) else f"class_{pred_label}"
            # Strip 'C_' prefix for CSV
            true_display = true_name.removeprefix('C_')
            pred_display = pred_name.removeprefix('C_')
            csv_records.append({
                'Molecule': mol_name,
                'True Environment': true_display,
                'CNN Prediction (Confidence %)': f"{pred_display} ({confidence:.1f}%)",
            })
            if pred_label != true_label:
                wrong_items.append(
                    f"{true_name.removeprefix('C_')}->{pred_name.removeprefix('C_')}"
                )
        n_correct = n_carbon - len(wrong_items)
        wrong_str = ', '.join(wrong_items) if wrong_items else 'all correct'
        print(f"  {mol_name:<30} {n_carbon:>4} {n_correct:>4} {len(wrong_items):>6}   {wrong_str}")

    if output_dir:
        csv_df = pd.DataFrame(csv_records)
        # Drop duplicate rows (same molecule, same environment, same prediction+confidence)
        #csv_df = csv_df.drop_duplicates().reset_index(drop=True)
        csv_path = Path(output_dir) / f"eval_{eval_type}{csv_suffix}.csv"
        csv_df.to_csv(csv_path, index=False)
        print(f"\nSaved evaluation CSV: {csv_path}")
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

    print("\n== SUMMARY STATISTICS ==")

    print(f"  Per-atom  ({len(all_labels)} atoms):  "
          f"Acc={accuracy*100:.1f}%  F1-mac={f1_macro:.4f}  F1-wt={f1_weighted:.4f}")
    print(f"  Deduped   ({len(dedup_labels)} pairs):  "
          f"Acc={dedup_accuracy*100:.1f}%  F1-mac={dedup_f1_macro:.4f}  F1-wt={dedup_f1_weighted:.4f}")

    print("\n  Per-Class (deduplicated):")
    print("  " + "-" * 70)
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
        print(f"Saved training plots: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"Failed to save plots: {e}")
