import os
import random
import numpy as np
from pathlib import Path
import torch
from torch_geometric.data import InMemoryDataset
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import train_test_split
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from torch.nn import Linear, ReLU, Tanh, Sequential as Seq
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from scipy.stats import ortho_group


def seed(seed=0):
    os.environ["PYTHONHASHSEED"]      = str(seed)  # enforce hash-based ops order
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # deterministic GEMMs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)

def get_next_model_filename(directory):
    files = [f for f in os.listdir(directory) if f.startswith("model_") and f.endswith(".pth")]
    numbers = [int(f.split("_")[1].split(".")[0]) for f in files if f.split("_")[1].split(".")[0].isdigit()]
    next_number = max(numbers) + 1 if numbers else 1
    return f"model_{next_number}.pth"

def get_latest_model_filename(directory):
    files = [f for f in os.listdir(directory) if f.startswith("model_") and f.endswith(".pth")]
    if not files:
        raise FileNotFoundError("No saved model found in the directory.")
    files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]), reverse=True)
    return os.path.join(directory, files[0])

class LoadDataset(InMemoryDataset):
    """
    Generic wrapper around a pre-collated (data, slices) file.

    Parameters
    ----------
    root : str | Path
        Directory that contains the processed file.
    file_name : str, default "data.pt"
        Name of the processed file to load.
    **kwargs
        Forwarded to `InMemoryDataset`.
    """
    def __init__(self, root: str | Path, *, file_name: str = "data.pt", **kwargs):
        self._processed_name = file_name        # store before super().__init__
        super().__init__(root, **kwargs)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    # ── hooks that InMemoryDataset expects ──────────────────────────────────
    @property
    def raw_file_names(self):        # nothing raw to manage
        return []

    @property
    def processed_file_names(self):
        # use whatever name the user passed
        return [self._processed_name]

    def download(self):              # no download step
        pass

    def process(self):               # already processed
        pass

class InvariantMPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=4, aggr='add'):
        """
        Message Passing Neural Network Layer
        This layer is equivariant to 3D rotations and translations.

        Args:
            emb_dim: (int) - hidden dimension `d`
            edge_dim: (int) - edge feature dimension `d_e`
            aggr: (str) - aggregation function `⊕` (sum/mean/max)
        """
        # Set the aggregation function
        super().__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        # --- Define the MLPs for the layer ---
        # MLP for the message function (ψ)
        # Input: concatenation of [h_i, h_j, edge_attr, d_ij^2]
        # where d_ij^2 = ||pos_i - pos_j||^2 (an invariant)
        self.mlp_msg = Seq(
            Linear(2 * emb_dim + edge_dim + 1, emb_dim),
            ReLU(),
            Linear(emb_dim, emb_dim)
        )

        # MLP for updating node features (φ)
        # Input: concatenation of [old h, aggregated feature message]
        self.mlp_upd = Seq(
            Linear(2 * emb_dim, emb_dim),
            ReLU(),
            Linear(emb_dim, emb_dim)
        )

    def forward(self, h, pos, edge_index, edge_attr):
        """
        Forward pass: one round of message passing.

        Args:
            h: (n, d) - initial node features
            pos: (n, 3) - initial node coordinates
            edge_index: (2, e) - edge index tensor with shape [2, num_edges]
            edge_attr: (e, d_e) - edge features

        Returns:
            out: tuple of [(n, d), (n, 3)] - updated node features and coordinates
        """
        # The propagate function will call message(), aggregate(), and update() for us.
        out = self.propagate(edge_index, h=h, pos=pos, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, pos_i, pos_j, edge_attr):
        """
        Message function.

        For each edge (i, j):
          - Compute the invariant squared distance: d2 = ||pos_i - pos_j||^2.
          - Compute a feature message based on h_i, h_j, edge_attr, and d2.
          - Compute a scalar weight (via mlp_coord) and form the coordinate message as:
              weight * (pos_i - pos_j)

        Returns a tuple of (feature_message, coordinate_message).
        """
        # Invariant: squared Euclidean distance (remains the same under rotations and translations)
        #d2 = torch.sum((pos_i - pos_j)**2, dim=-1, keepdim=True)  # shape: (E, 1)
        d = torch.norm(pos_i - pos_j, p=2, dim=-1, keepdim=True)  # shape: (E, 1)
        d2 = d**2

        # Concatenate inputs for the message MLP
        msg = torch.cat([h_i, h_j, edge_attr, d2], dim=-1)

        return self.mlp_msg(msg)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        """
        Aggregates messages from neighboring nodes.

        Since message() returns a tuple (feature_message, coordinate_message),
        we aggregate each component separately using the chosen aggregator.
        """
#         return (agg_feat, agg_coord)
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)

    def update(self, aggr_out, h, pos):
        """
        Updates the node features and coordinates.

        - The new node features are computed as φ(concat(old features, aggregated feature messages)).
          This update is invariant.
        - The new coordinates are given by pos + (aggregated coordinate messages).
          Because the coordinate messages are equivariant, this update is equivariant.
        """

        h_updated = self.mlp_upd(torch.cat([h, aggr_out], dim=-1))
#         return (h_updated, pos_updated)
        return h_updated

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})'

class EquivariantMPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=4, aggr='add'):
        """
        Message Passing Neural Network Layer
        This layer is equivariant to 3D rotations and translations.

        Args:
            emb_dim: (int) - hidden dimension `d`
            edge_dim: (int) - edge feature dimension `d_e`
            aggr: (str) - aggregation function `⊕` (sum/mean/max)
        """
        # Set the aggregation function
        super().__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        # --- Define the MLPs for the layer ---
        # MLP for the message function (ψ)
        # Input: concatenation of [h_i, h_j, edge_attr, d_ij^2]
        # where d_ij^2 = ||pos_i - pos_j||^2 (an invariant)
        self.mlp_msg = Seq(
            Linear(2 * emb_dim + edge_dim + 1, emb_dim),
            ReLU(),
            Linear(emb_dim, emb_dim)
        )

        # MLP for coordinate update weight
        # Input: message from mlp_msg, output: a scalar weight
        self.mlp_coord = Seq(
            Linear(emb_dim, emb_dim),
            ReLU(),
            Linear(emb_dim, 1),
            Tanh()
        )

        # MLP for updating node features (φ)
        # Input: concatenation of [old h, aggregated feature message]
        self.mlp_upd = Seq(
            Linear(2 * emb_dim, emb_dim),
            ReLU(),
            Linear(emb_dim, emb_dim)
        )

    def forward(self, h, pos, edge_index, edge_attr):
        """
        Forward pass: one round of message passing.

        Args:
            h: (n, d) - initial node features
            pos: (n, 3) - initial node coordinates
            edge_index: (2, e) - edge index tensor with shape [2, num_edges]
            edge_attr: (e, d_e) - edge features

        Returns:
            out: tuple of [(n, d), (n, 3)] - updated node features and coordinates
        """
        # The propagate function will call message(), aggregate(), and update() for us.
        out = self.propagate(edge_index, h=h, pos=pos, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, pos_i, pos_j, edge_attr):
        """
        Message function.

        For each edge (i, j):
          - Compute the invariant squared distance: d2 = ||pos_i - pos_j||^2.
          - Compute a feature message based on h_i, h_j, edge_attr, and d2.
          - Compute a scalar weight (via mlp_coord) and form the coordinate message as:
              weight * (pos_i - pos_j)

        Returns a tuple of (feature_message, coordinate_message).
        """
        # Invariant: squared Euclidean distance (remains the same under rotations and translations)
        #d2 = torch.sum((pos_i - pos_j)**2, dim=-1, keepdim=True)  # shape: (E, 1)
        d = torch.norm(pos_i - pos_j, p=2, dim=-1, keepdim=True)  # shape: (E, 1)
        d2 = d**2

        # Concatenate inputs for the message MLP
        msg_input = torch.cat([h_i, h_j, edge_attr, d2], dim=-1)
        msg = self.mlp_msg(msg_input)  # shape: (E, emb_dim)

        # Compute a scalar weight from the message for coordinate update
        w = self.mlp_coord(msg)  # shape: (E, 1)

        # Equivariant coordinate message: scales the relative position
        msg_coord = w * (pos_i - pos_j)  # shape: (E, 3)

        # Return both messages
        return (msg, msg_coord)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        """
        Aggregates messages from neighboring nodes.

        Since message() returns a tuple (feature_message, coordinate_message),
        we aggregate each component separately using the chosen aggregator.
        """
        msg_feat, msg_coord = inputs
        agg_feat = scatter(msg_feat, index, dim=0, reduce=self.aggr)
        agg_coord = scatter(msg_coord, index, dim=0, reduce=self.aggr)

        counts = scatter(torch.ones(msg_coord.size(0), device=msg_coord.device), index, dim=0, reduce="sum")

        #scale = 1.0 / (counts - 1).clamp(min=1)
        scale = 1.0 / counts.clamp(min=1)
        scale = scale.unsqueeze(-1)
        agg_coord = agg_coord * scale

        return (agg_feat, agg_coord)

    def update(self, aggr_out, h, pos):
        """
        Updates the node features and coordinates.

        - The new node features are computed as φ(concat(old features, aggregated feature messages)).
          This update is invariant.
        - The new coordinates are given by pos + (aggregated coordinate messages).
          Because the coordinate messages are equivariant, this update is equivariant.
        """
        agg_feat, agg_coord = aggr_out
        # Feature update: combine old features with aggregated messages
        h_updated = self.mlp_upd(torch.cat([h, agg_feat], dim=-1))
        # Coordinate update: add aggregated coordinate messages to the original coordinates
        pos_updated = pos + agg_coord

        return (h_updated, pos_updated)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})'

class MPNN(nn.Module):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1,
                layer_type="IN", pred_type="AUGER", spectrum_type='stick', spectrum_dim=300,
                dropout=0.0):

        """Message Passing Neural Network model for graph property prediction

        This model uses both node features and coordinates as inputs, and
        is invariant to 3D rotations and translations (the constituent MPNN layers
        are equivariant to 3D rotations and translations).

        Args:
            num_layers: (int) - number of message passing layers `L`
            emb_dim: (int) - hidden dimension `d`
            in_dim: (int) - initial node feature dimension `d_n`
            edge_dim: (int) - edge feature dimension `d_e`
            out_dim: (int) - output dimension (CEBE only, fixed to 1)
            spectrum_type: (str) - 'stick' or 'fitted'
                stick:  two heads (energy + intensity), each → spectrum_dim (default 300)
                        total output = 2 * spectrum_dim = 600
                fitted: single intensity head to spectrum_dim (default 731)
            spectrum_dim: (int) - per-head output dimension
                stick mode:  300  (energy 300 + intensity 300 = 600)
                fitted mode: 731  (intensity only on common energy grid)
            dropout: (float) - dropout probability between message passing layers (0 = off)
        """
        super().__init__()

        # Linear projection for initial node features
        # dim: d_n -> d
        self.lin_in = Linear(in_dim, emb_dim)

        # Stack of MPNN layers with LayerNorm after each
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer_type == "EQ":
                self.convs.append(EquivariantMPNNLayer(emb_dim, edge_dim, aggr='add'))
            elif layer_type == "IN":
                self.convs.append(InvariantMPNNLayer(emb_dim, edge_dim, aggr='add'))
            self.norms.append(nn.LayerNorm(emb_dim))

        if pred_type == "CEBE":
            # Linear prediction head
            # dim: d -> out_dim
            self.lin_pred = Linear(emb_dim, out_dim)
        elif pred_type == "AUGER":
            # --- node-level decoder (deeper, wider) ---
            # Intermediate dim: 2x embedding for more capacity
            dec_mid = emb_dim * 2
            # Intensity head: 4-layer decoder with Softplus output
            self.dec_int = nn.Sequential(
                nn.Linear(emb_dim, dec_mid),
                nn.LayerNorm(dec_mid),
                nn.Softplus(beta=2.0),
                nn.Dropout(p=0.10),
                nn.Linear(dec_mid, dec_mid),
                nn.Softplus(beta=2.0),
                nn.Dropout(p=0.05),
                nn.Linear(dec_mid, spectrum_dim),
                nn.Softplus(beta=1.0),
            )
            if spectrum_type == 'stick':
                # Energy head: only used for stick spectra
                self.dec_eng = nn.Sequential(
                    nn.Linear(emb_dim, dec_mid),
                    nn.LayerNorm(dec_mid),
                    nn.Softplus(beta=2.0),
                    nn.Dropout(p=0.10),
                    nn.Linear(dec_mid, dec_mid),
                    nn.Softplus(beta=2.0),
                    nn.Dropout(p=0.05),
                    nn.Linear(dec_mid, spectrum_dim),
                    nn.Softplus(beta=1.0),
                )

        self.layer_type = layer_type
        self.pred_type  = pred_type
        self.spectrum_type = spectrum_type
        self.spectrum_dim = spectrum_dim
        self.dropout = dropout

    def forward(self, data):
        """
        Args:
            data: (PyG.Data) - batch of PyG graphs

        Returns:
            out: (batch_size, out_dim) - prediction for each graph
        """
        h = self.lin_in(data.x) # (n, d_n) -> (n, d)

        if self.layer_type == "EQ":

            pos = data.pos

            for conv, norm in zip(self.convs, self.norms):
                # Message passing layer
                h_update, pos_update = conv(h, pos, data.edge_index, data.edge_attr)

                # Residual connection + LayerNorm
                h = norm(h + h_update) # (n, d) -> (n, d)

                # Dropout (only active during training)
                h = F.dropout(h, p=self.dropout, training=self.training)

                # Update node coordinates
                pos = pos_update # (n, 3) -> (n, 3)
        elif self.layer_type == "IN":

            pos = data.pos

            for conv, norm in zip(self.convs, self.norms):
                # Message passing layer
                h_update = conv(h, pos, data.edge_index, data.edge_attr)

                # Residual connection + LayerNorm
                h = norm(h + h_update) # (n, d) -> (n, d)

                # Dropout (only active during training)
                h = F.dropout(h, p=self.dropout, training=self.training)
        elif self.layer_type == "PE":
            for conv, norm in zip(self.convs, self.norms):
                h_update = conv(h, data.edge_index, data.edge_attr)
                h = norm(h + h_update)

                # Dropout (only active during training)
                h = F.dropout(h, p=self.dropout, training=self.training)

        if self.pred_type == "CEBE":
            out = self.lin_pred(h)
        elif self.pred_type == "AUGER":
            if self.spectrum_type == 'fitted':
                # Fitted: intensity-only output on common energy grid
                out = self.dec_int(h)
            else:
                # Stick: concatenate energy and intensity heads
                out_int = self.dec_int(h)
                out_eng = self.dec_eng(h)
                out = torch.cat([out_eng, out_int], dim=-1)

        return out

def train_mpnn(data_loader: DataLoader, model: nn.Module, optimizer: torch.optim.Optimizer, layer_type, pred_type, schedular):
    total_loss = 0
    total_samples = 0

    model.train()

    for data in data_loader:
        optimizer.zero_grad()
        #out = model(data, layer_type, pred_type)
        out = model(data)
        #print("Batch y:", data.y)
        #print("Output:", out)
        if pred_type == "CEBE":
            loss = F.mse_loss(out, data.y)
        elif pred_type == "AUGER":
            idx   = data.node_mask.nonzero(as_tuple=True)[0]
            # Handle both 300-dim (split) and 600-dim (original) mask_bin for backward compatibility
            mask = data.mask_bin[idx]
            out_sel = out[idx]
            y_sel = data.y[idx]

            # Ensure mask and y/out have same dimensions
            if mask.shape != y_sel.shape:
                if y_sel.shape[1] == 300 and mask.shape[1] == 600:
                    # Mask is 600-dim, y is 300-dim: take first 300 of mask
                    mask = mask[:, :300]
                elif y_sel.shape[1] == 600 and mask.shape[1] == 600:
                    # Both 600-dim: keep as is
                    pass

            loss = ((out_sel - y_sel)**2 * mask).sum() / mask.sum()
        loss.backward()
        optimizer.step()
        if pred_type == "AUGER":
            schedular.step()

        #print(out.detach().abs().mean())

        total_loss += loss.item()
        total_samples += 1

    return total_loss / total_samples

def eval_mpnn(data_loader, model, device, layer_type, pred_type, spectrum_type='stick'):
    """One pass over data_loader without gradient to compute mean loss.

    Args:
        spectrum_type: 'stick' (600-dim energy+intensity) or 'fitted' (n_points intensity)
    """
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out = model(data)
            if pred_type == "CEBE":
                idx  = data.node_mask.nonzero(as_tuple=True)[0]
                loss = F.mse_loss(out[idx], data.y[idx])
#                 loss = F.mse_loss(out, data.y)
            elif pred_type == "AUGER":
                idx  = data.node_mask.nonzero(as_tuple=True)[0]
                out_sel = out[idx]

                if spectrum_type == 'fitted':
                    # Fitted: target is data.y_fitted (N, n_points), no mask needed
                    y_sel = data.y_fitted[idx]
                    loss = F.mse_loss(out_sel, y_sel)
                else:
                    # Stick: target is data.y (N, 600) with mask
                    y_sel = data.y[idx]
                    mask = data.mask_bin[idx]
                    if mask.shape != y_sel.shape:
                        if y_sel.shape[1] == 300 and mask.shape[1] == 600:
                            mask = mask[:, :300]
                    loss = ((out_sel - y_sel)**2 * mask).sum() / mask.sum()
            total_loss += loss.item()
            n_batches  += 1
    return total_loss / n_batches


class CosineAnnealingWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine Annealing with Linear Warmup scheduler.

    During warmup phase: linearly increases LR from 0 to max_lr
    During cosine phase: decreases LR using cosine annealing to min_lr

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs for linear warmup
        max_epochs: Total number of epochs
        min_lr: Minimum learning rate (default: 1e-7)
        last_epoch: The index of last epoch (default: -1)
    """
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int,
                 min_lr: float = 1e-7, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate for current epoch."""
        current_epoch = self.last_epoch

        if current_epoch < self.warmup_epochs:
            # Linear warmup phase
            lr_range = self.base_lrs[0] - self.min_lr
            return [self.min_lr + lr_range * current_epoch / self.warmup_epochs
                    for _ in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            lr_range = self.base_lrs[0] - self.min_lr
            return [self.min_lr + lr_range * cosine_decay for _ in self.base_lrs]


def train_loop(data_list: list, model: nn.Module, device, num_epochs: int = 100, batch_size=64, max_lr=1e-2,
                pct_start=0.6, verbose = True, layer_type="IN", pred_type="AUGER", plot_results=False, val_data_list=None,
                patience=50, optimizer_type='adamw', weight_decay=1e-4, gradient_clip_norm=0.5, warmup_epochs=10, min_lr=1e-7,
                spectrum_type='stick', scheduler_type='cosine'):
    """
    Advanced training loop with gradient clipping, configurable optimizer and LR scheduler.

    Args:
        data_list: Training data
        model: Neural network model
        device: Device to train on
        num_epochs: Number of training epochs
        batch_size: Batch size
        max_lr: Maximum learning rate
        pct_start: For OneCycleLR, percentage of training steps allocated to warmup
        verbose: Whether to print training progress
        layer_type: Layer type (IN/EQ/PE)
        pred_type: Prediction type (CEBE/AUGER)
        plot_results: Whether to plot training results
        val_data_list: Validation data (if None, will split from training data)
        optimizer_type: 'adam', 'adamw' (default: 'adamw')
        weight_decay: L2 regularization weight
        gradient_clip_norm: Max gradient norm for clipping (default: 1.0)
        warmup_epochs: Number of epochs for warmup in cosine scheduler (default: 10)
        min_lr: Minimum learning rate for cosine scheduler (default: 1e-7)
        spectrum_type: 'stick' (600-dim energy+intensity with mask) or
                       'fitted' (n_points intensity on common grid, no mask)
        scheduler_type: 'cosine' (CosineAnnealingWarmup, per-epoch) or
                        'onecycle' (OneCycleLR, per-batch — original AUGER schedule)
    """

    split_seed = 42

    seed(0)
    gen = torch.Generator().manual_seed(0)

    # If val_data_list is provided, use it directly; otherwise perform internal split
    if val_data_list is not None:
        # External split already performed (e.g., for combined singlet+triplet training)
        train_set = data_list
        val_set = val_data_list
    else:
        # Internal split for backward compatibility (e.g., for CEBE training)
        train_set, val_set = train_test_split(data_list, test_size=0.10, random_state=split_seed)

    print(f"Training samples: {len(train_set)}, carbons: {sum(s == 'C' for d in train_set for s in d.atom_symbols)}")
    print(f"Validation samples: {len(val_set)}, carbons: {sum(s == 'C' for d in val_set for s in d.atom_symbols)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, generator=gen,
                                pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, generator=gen,
                                pin_memory=(device.type == "cuda"))

    # ============================================================================
    # OPTIMIZER SETUP - Use AdamW for better generalization
    # ============================================================================
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, weight_decay=weight_decay)

    # ============================================================================
    # SCHEDULER SETUP
    # ============================================================================
    # Determine whether scheduler steps per-batch or per-epoch
    scheduler_per_batch = False

    if scheduler_type == 'onecycle':
        # OneCycleLR: steps per BATCH — aggressive peak then smooth decay.
        # Well-suited for masked regression (stick spectra).
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=len(train_loader),
            epochs=num_epochs,
            pct_start=pct_start,
        )
        scheduler_per_batch = True
        if verbose:
            total_steps = len(train_loader) * num_epochs
            print(f"  Scheduler: OneCycleLR  (per-batch, {total_steps} total steps, "
                  f"pct_start={pct_start})")
    else:
        # CosineAnnealingWarmup: steps per EPOCH — smoother schedule.
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=num_epochs,
            min_lr=min_lr,
        )
        scheduler_per_batch = False
        if verbose:
            print(f"  Scheduler: CosineAnnealingWarmup  (per-epoch, "
                  f"warmup={warmup_epochs} epochs)")

    train_results = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = patience  # Early stopping patience
    best_model_state = None  # Track best model weights

    for epoch in range(num_epochs):

        model.train()
        running_loss, n_batches = 0.0, 0

        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            out = model(data)
            if pred_type == "CEBE":
                idx  = data.node_mask.nonzero(as_tuple=True)[0]
                loss = F.mse_loss(out[idx], data.y[idx])
                #loss = F.mse_loss(out, data.y)
            elif pred_type == "AUGER":
                idx  = data.node_mask.nonzero(as_tuple=True)[0]
                out_sel = out[idx]

                # DEBUG: Print shapes on first batch of first epoch
                if epoch == 0 and n_batches == 0:
                    print(f"DEBUG AUGER ({spectrum_type}): idx.shape={idx.shape}")
                    print(f"DEBUG AUGER: out_sel.shape={out_sel.shape}")

                if spectrum_type == 'fitted':
                    # Fitted: target is data.y_fitted (N, n_points), no mask needed
                    y_sel = data.y_fitted[idx]
                    if epoch == 0 and n_batches == 0:
                        print(f"DEBUG AUGER: y_fitted_sel.shape={y_sel.shape}")
                    loss = F.mse_loss(out_sel, y_sel)
                else:
                    # Stick: target is data.y (N, 600) with binary mask
                    y_sel = data.y[idx]
                    mask = data.mask_bin[idx]
                    if epoch == 0 and n_batches == 0:
                        print(f"DEBUG AUGER: data.y.shape={data.y.shape}, data.mask_bin.shape={data.mask_bin.shape}")
                        print(f"DEBUG AUGER: y_sel.shape={y_sel.shape}, mask.shape={mask.shape}")
                    # Ensure mask and y/out have same dimensions
                    if mask.shape != y_sel.shape:
                        if y_sel.shape[1] == 300 and mask.shape[1] == 600:
                            mask = mask[:, :300]
                    loss = ((out_sel - y_sel)**2 * mask).sum() / mask.sum()

            loss.backward()

            # ============================================================================
            # GRADIENT CLIPPING - Prevent gradient explosion
            # ============================================================================
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)

            optimizer.step()

            # OneCycleLR steps per batch
            if scheduler_per_batch:
                scheduler.step()

            running_loss += loss.item()
            n_batches    += 1

        train_loss = running_loss / n_batches
        val_loss   = eval_mpnn(val_loader, model, device, layer_type, pred_type, spectrum_type=spectrum_type)
        train_results.append([epoch, train_loss, val_loss])

        # CosineAnnealingWarmup steps per epoch
        if not scheduler_per_batch:
            scheduler.step()

        # Early stopping with model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model weights
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} with val_loss={val_loss:.5f}, best_val_loss={best_val_loss:.5f}")
                # Restore best model weights before returning
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    if verbose:
                        print(f"Restored model weights from epoch with best validation loss")
                break

        if verbose:
            print(f"Epoch {epoch:03d} │ train {train_loss:.5f} │ val {val_loss:.5f}")

    return train_results

def permute_graph(data, perm):
    """Helper function for permuting PyG Data object attributes consistently.
    """
    # Permute the node attribute ordering
    data.x = data.x[perm]
    data.pos = data.pos[perm]

    # Permute optional per-node attributes if they exist
    if hasattr(data, 'z') and data.z is not None:
        data.z = data.z[perm]
    if hasattr(data, 'batch') and data.batch is not None:
        data.batch = data.batch[perm]
    if hasattr(data, 'y') and data.y is not None and data.y.size(0) == perm.size(0):
        data.y = data.y[perm]
    if hasattr(data, 'node_mask') and data.node_mask is not None:
        data.node_mask = data.node_mask[perm]

    # Permute the edge index
    adj = to_dense_adj(data.edge_index)
    adj = adj[:, perm, :]
    adj = adj[:, :, perm]
    data.edge_index = dense_to_sparse(adj)[0]

    # Note:
    # (1) While we originally defined the permutation matrix P as only having
    #     entries 0 and 1, its implementation via `perm` uses indexing into
    #     torch tensors, instead.
    # (2) It is cumbersome to permute the edge_attr, so we set it to constant
    #     dummy values. For any experiments beyond unit testing, all GNN models
    #     use the original edge_attr.

    return data

def permutation_equivariance_unit_test_model(module, dataloader):
    """Unit test for checking whether a **node-level** GNN model is
    permutation equivariant.

    For a node-level model (no global pooling), permuting the input nodes
    should permute the output rows in the same way:
        out(π(G))[i]  ==  out(G)[π⁻¹(i)]   ⟺   out_2 == out_1[perm]

    Note: The old test checked ``out_1 == out_2`` which is *invariance* —
    correct only for graph-level (pooled) models, not node-level ones.
    """
    it = iter(dataloader)
    data = next(it)

    # Set edge_attr to dummy values (for simplicity)
    data.edge_attr = torch.zeros(data.edge_attr.shape)

    # Forward pass on original example
    out_1 = module(data)

    # Create random permutation
    perm = torch.randperm(data.x.shape[0])
    data = permute_graph(data, perm)

    # Forward pass on permuted example
    out_2 = module(data)

    # Node-level equivariance: output rows should follow the permutation
    return torch.allclose(out_1[perm], out_2, atol=1e-04)


def permutation_equivariance_unit_test_layer(module, dataloader, lin_in=None):
    """Unit test for checking whether a single MPNN layer is
    permutation equivariant.

    Parameters
    ----------
    module : MessagePassing layer
    dataloader : DataLoader
    lin_in : nn.Module, optional
        The model's input projection (``model.lin_in``).  If provided the
        raw node features ``data.x`` are projected to ``emb_dim`` before
        being fed to the layer, which avoids a dimension mismatch.
    """
    it = iter(dataloader)
    data = next(it)

    # Set edge_attr to dummy values (for simplicity)
    data.edge_attr = torch.zeros(data.edge_attr.shape)

    # Project raw features to embedding dim if lin_in is provided
    h = lin_in(data.x) if lin_in is not None else data.x

    # Forward pass on original example
    if isinstance(module, EquivariantMPNNLayer):
        out_1, _ = module(h, data.pos, data.edge_index, data.edge_attr)
    elif isinstance(module, InvariantMPNNLayer):
        out_1 = module(h, data.pos, data.edge_index, data.edge_attr)
    else:
        out_1 = module(h, data.edge_index, data.edge_attr)

    # Create random permutation
    perm = torch.randperm(data.x.shape[0])
    data = permute_graph(data, perm)
    h = h[perm]  # permute the projected features consistently

    # Forward pass on permuted example
    if isinstance(module, EquivariantMPNNLayer):
        out_2, _ = module(h, data.pos, data.edge_index, data.edge_attr)
    elif isinstance(module, InvariantMPNNLayer):
        out_2 = module(h, data.pos, data.edge_index, data.edge_attr)
    else:
        out_2 = module(h, data.edge_index, data.edge_attr)

    # Check whether output varies after applying transformations
    return torch.allclose(out_1[perm], out_2, atol=1e-04)


def random_orthogonal_matrix(dim=3):
  """Helper function to build a random orthogonal matrix of shape (dim, dim)
  """
  Q = torch.tensor(ortho_group.rvs(dim=dim)).float()
  return Q


def rot_trans_invariance_unit_test(module, dataloader, lin_in=None):
    """Unit test for checking whether a module (GNN model/layer) is
    rotation and translation invariant.

    Parameters
    ----------
    lin_in : nn.Module, optional
        The model's input projection (``model.lin_in``).  When testing a
        bare layer, this projects raw ``data.x`` to ``emb_dim`` first.
    """
    it = iter(dataloader)
    data = next(it)

    # Forward pass on original example
    if isinstance(module, MPNN):
        out_1 = module(data)
    else:
        h = lin_in(data.x) if lin_in is not None else data.x
        if isinstance(module, EquivariantMPNNLayer):
            out_1, _ = module(h, data.pos, data.edge_index, data.edge_attr)
        elif isinstance(module, InvariantMPNNLayer):
            out_1 = module(h, data.pos, data.edge_index, data.edge_attr)
        else:
            out_1 = module(h, data.edge_index, data.edge_attr)

    Q = random_orthogonal_matrix(dim=3)
    t = torch.rand(3)

    # Perform random rotation + translation on data.
    data.pos = data.pos @ Q.T + t

    # Forward pass on rotated + translated example
    if isinstance(module, MPNN):
        out_2 = module(data)
    else:
        # h is unchanged (features are not rotated, only positions)
        if isinstance(module, EquivariantMPNNLayer):
            out_2, _ = module(h, data.pos, data.edge_index, data.edge_attr)
        elif isinstance(module, InvariantMPNNLayer):
            out_2 = module(h, data.pos, data.edge_index, data.edge_attr)
        else:
            out_2 = module(h, data.edge_index, data.edge_attr)

    # Check whether output varies after applying transformations.
    return torch.allclose(out_1, out_2, atol=1e-04)

def rot_trans_equivariance_unit_test(module, dataloader, lin_in=None):
    """Unit test for checking whether a module (GNN layer) is
    rotation and translation equivariant.

    Parameters
    ----------
    lin_in : nn.Module, optional
        The model's input projection (``model.lin_in``).
    """
    it = iter(dataloader)
    data = next(it)

    h = lin_in(data.x) if lin_in is not None else data.x

    out_1, pos_1 = module(h, data.pos, data.edge_index, data.edge_attr)

    Q = random_orthogonal_matrix(dim=3)
    t = torch.rand(3)

    # Perform random rotation + translation on data.
    data.pos = data.pos @ Q.T + t

    # Forward pass on rotated + translated example
    out_2, pos_2 = module(h, data.pos, data.edge_index, data.edge_attr)

    # Check whether output varies after applying transformations.
    # Node features should be invariant (same regardless of rotation/translation).
    features_invariant = torch.allclose(out_1, out_2, atol=1e-04)
    # Coordinates should be equivariant: pos_2 ≈ pos_1 @ Q.T + t
    coords_equivariant = torch.allclose(pos_1 @ Q.T + t, pos_2, atol=1e-04)
    return features_invariant and coords_equivariant


# =====================================================================
#  run_unit_tests — convenience wrapper for all symmetry unit tests
# =====================================================================

def run_unit_tests(model, data_list, layer_type='IN', batch_size=1):
    """Run permutation and rotation/translation symmetry unit tests on a
    trained GNN model and its first message-passing layer.

    For a **node-level** model (no global pooling), the correct symmetry
    property is **permutation equivariance** — permuting the input nodes
    should permute the output rows in the same way.

    For the layer-level tests, ``model.lin_in`` is used to project the raw
    node features down to ``emb_dim`` before feeding them into the bare
    layer, avoiding a dimension mismatch.

    Args:
        model:      (MPNN) — the trained model (in eval mode).
        data_list:  list[Data] — dataset (at least 1 graph).
        layer_type: (str) — 'EQ' or 'IN'.
        batch_size: (int) — batch size for the test dataloader (default 1).

    Returns:
        results: dict mapping test name → bool (pass/fail).
    """
    import copy

    model.eval()
    results = {}

    # The input-projection layer is needed for layer-level tests so that
    # data.x (in_dim) is mapped to the layer's expected emb_dim.
    lin_in = model.lin_in

    print(f"\n{'=' * 60}")
    print("  SYMMETRY UNIT TESTS")
    print(f"{'=' * 60}")

    # ── 1. Model-level permutation equivariance ─────────────────────────
    #       (node-level model: permuting inputs permutes outputs)
    try:
        loader_copy = DataLoader(copy.deepcopy(data_list[:1]), batch_size=batch_size, shuffle=False)
        passed = permutation_equivariance_unit_test_model(model, loader_copy)
        results['permutation_equivariance_model'] = passed
        status = 'PASS' if passed else 'FAIL'
        print(f"  {status}  Permutation equivariance  (model)")
    except Exception as e:
        results['permutation_equivariance_model'] = False
        print(f"  ERROR  Permutation equivariance  (model): {e}")

    # ── 2. Layer-level permutation equivariance ─────────────────────────
    first_layer = model.convs[0]
    try:
        loader_copy = DataLoader(copy.deepcopy(data_list[:1]), batch_size=batch_size, shuffle=False)
        passed = permutation_equivariance_unit_test_layer(first_layer, loader_copy, lin_in=lin_in)
        results['permutation_equivariance_layer'] = passed
        status = 'PASS' if passed else 'FAIL'
        print(f"  {status}  Permutation equivariance (layer)")
    except Exception as e:
        results['permutation_equivariance_layer'] = False
        print(f"  ERROR  Permutation equivariance (layer): {e}")

    # ── 3. Rotation+translation invariance (model) ──────────────────────
    try:
        loader_copy = DataLoader(copy.deepcopy(data_list[:1]), batch_size=batch_size, shuffle=False)
        passed = rot_trans_invariance_unit_test(model, loader_copy)
        results['rot_trans_invariance_model'] = passed
        status = 'PASS' if passed else 'FAIL'
        print(f"  {status}  Rotation+translation invariance  (model)")
    except Exception as e:
        results['rot_trans_invariance_model'] = False
        print(f"  ERROR  Rotation+translation invariance  (model): {e}")

    # ── 4. Rotation+translation invariance (IN layer) or equivariance (EQ layer)
    if layer_type == 'IN':
        try:
            loader_copy = DataLoader(copy.deepcopy(data_list[:1]), batch_size=batch_size, shuffle=False)
            passed = rot_trans_invariance_unit_test(first_layer, loader_copy, lin_in=lin_in)
            results['rot_trans_invariance_layer'] = passed
            status = 'PASS' if passed else 'FAIL'
            print(f"  {status}  Rotation+translation invariance  (IN layer)")
        except Exception as e:
            results['rot_trans_invariance_layer'] = False
            print(f"  ERROR  Rotation+translation invariance  (IN layer): {e}")
    elif layer_type == 'EQ':
        try:
            loader_copy = DataLoader(copy.deepcopy(data_list[:1]), batch_size=batch_size, shuffle=False)
            passed = rot_trans_equivariance_unit_test(first_layer, loader_copy, lin_in=lin_in)
            results['rot_trans_equivariance_layer'] = passed
            status = 'PASS' if passed else 'FAIL'
            print(f"  {status}  Rotation+translation equivariance (EQ layer)")
        except Exception as e:
            results['rot_trans_equivariance_layer'] = False
            print(f"  ERROR  Rotation+translation equivariance (EQ layer): {e}")

    # ── Summary ─────────────────────────────────────────────────────────
    n_pass = sum(v for v in results.values())
    n_total = len(results)
    print(f"\n  Summary: {n_pass}/{n_total} tests passed")
    print(f"{'=' * 60}\n")

    return results
