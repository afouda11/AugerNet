import os
import csv
import random
import numpy as np
import re
from pathlib import Path
import torch
from torch_geometric.data import InMemoryDataset
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
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

############################################################################ 
# GNN Architecture: 
# Invariant or equivariant layer definition and message passing layer stacking
############################################################################


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
                layer_type="IN", pred_type="AUGER", spectrum_dim=300, dropout=0.0, 
                task_type='single', n_var=2):

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
            spectrum_dim: (int) - per-head output dimension
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
        self.layer_type = layer_type
        self.pred_type  = pred_type
        self.spectrum_dim = spectrum_dim
        self.dropout = dropout
        self.task_type = task_type
        # Learnable log-variance for uncertainty weighting (Kendall et al. 2018)
        # log_var[0] for CEBE, log_var[1] for Auger, log_var[2] for alpha (optional) 
        self.log_var = nn.Parameter(torch.zeros(n_var))
        #different weighting approach, from https://github.com/Mikoto10032/AutomaticWeightedLoss
        #self.awl = AutomaticWeightedLoss(num=n_var)   # CEBE, Auger, alpha

        # Multi-task adapters (only for pred_type == 'AUGER')
        if task_type == 'multi' and pred_type == 'AUGER':
            self.adapter_cebe  = nn.Linear(emb_dim, emb_dim)
            self.adapter_auger = nn.Linear(emb_dim, emb_dim)
            # CEBE scalar prediction head (shared encoder -> adapter -> scalar)
            self.lin_pred = nn.Linear(emb_dim, 1)

    def forward(self, data, return_embedding=False):
        """
        Args:
            data: (PyG.Data) - batch of PyG graphs
            return_embedding: if True, return node embeddings h before the
                              prediction head instead of predictions.

        Returns:
            out: (n_nodes, emb_dim) if return_embedding else (n_nodes, out_dim)
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

        if return_embedding:
            return h

        if self.pred_type == "CEBE":
            out = self.lin_pred(h)
        elif self.pred_type == "AUGER":
            if self.task_type == 'multi':
                # Multi-task: route through task-specific adapters
                h_cebe  = F.silu(self.adapter_cebe(h))
                h_auger = F.silu(self.adapter_auger(h))
                cebe_out = self.lin_pred(h_cebe)
                auger_out = self.dec_int(h_auger)
                return cebe_out, auger_out
            else:
                out = self.dec_int(h)
        return out

class AutomaticWeightedLoss(nn.Module):
    """Liebel & Körner (2018) automatically weighted multi-task loss.
        Prevents negative loss, from https://github.com/Mikoto10032/AutomaticWeightedLoss."""
    def __init__(self, num=2):
        super().__init__()
        self.params = nn.Parameter(torch.ones(num))

    def forward(self, *losses):
        total = 0.0
        for i, L in enumerate(losses):
            total = total + 0.5 / (self.params[i] ** 2) * L + torch.log(1 + self.params[i] ** 2)
        return total

############################################################################ 
# Utility function: 
# LR warmup, training history write and paramter splitting for weight decay
############################################################################

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

############################################################################
# Uncertainty weighting (UW) helpers
############################################################################
# Kendall et al. (2018) derive the uncertainty-weighted multi-task loss from a
# Gaussian likelihood, which pairs with a *squared-error* task loss:
#
#     L_uw = 1/(2*sigma^2) * L_mse + log sigma
#          = 0.5 * ( exp(-s) * L_mse + s ),        s := log sigma^2
#
# The Laplace likelihood for L1 loss gives
#
#     L_uw = 1/b * L_mae + log b
#          = 1.0 * ( exp(-s) * L_mae + s ),        s := log b
#
# Learnt parameter ``model.log_var`` therefore means log sigma^2 for an MSE head but
# log b for an MAE head — it is a log scale parameter either way.
#
# Weight that actually multiplies a task loss is
#     w = scale * exp(-log_var)

_UW_SCALE      = {'mse': 0.5, 'mae': 1.0}                 # keyed by loss name
_UW_SCALE_FN   = {F.mse_loss: 0.5, F.l1_loss: 1.0}        # keyed by loss fn
_UW_LIKELIHOOD = {'mse': 'Gaussian', 'mae': 'Laplace'}


def _uw_scale(loss_spec, task_name=""):
    """Per-task prefactor of the UW loss term: 0.5 for MSE, 1.0 for MAE.

    ``loss_spec`` may be a loss name ('mse'/'mae') or the corresponding
    ``torch.nn.functional`` callable (``F.mse_loss``/``F.l1_loss``).
    """
    if isinstance(loss_spec, str):
        if loss_spec in _UW_SCALE:
            return _UW_SCALE[loss_spec]
    elif loss_spec in _UW_SCALE_FN:
        return _UW_SCALE_FN[loss_spec]
    raise ValueError(
        f"Unsupported UW loss{' for ' + task_name if task_name else ''}: "
        f"{loss_spec!r} — must be 'mae'/'mse' or F.l1_loss/F.mse_loss."
    )


def _uw_loss(loss_cebe, loss_auger, log_var, scale_cebe, scale_auger):
    """Uncertainty-weighted sum of the two task losses.

    Single expression covering all MAE/MSE combinations (including mixed
    ones):  ``scale * ( exp(-s) * L + s )`` per task.
    """
    return (scale_cebe  * (torch.exp(-log_var[0]) * loss_cebe  + log_var[0]) +
            scale_auger * (torch.exp(-log_var[1]) * loss_auger + log_var[1]))


def _uw_weights(log_var, scale_cebe, scale_auger):
    """Effective task weights actually applied to each loss: scale*exp(-s).

    Returns
    -------
    (float, float)
        ``(w_cebe, w_auger)`` — plain Python floats, detached.
    """
    lv = log_var.detach().cpu()
    return (scale_cebe  * float(torch.exp(-lv[0])),
            scale_auger * float(torch.exp(-lv[1])))


# Loss-history CSV
# Column order for the per-run loss-history CSV.
# ``w_cebe`` / ``w_auger`` are the EFFECTIVE weights (scale * exp(-log_var)),
# consistent with the UW formulation selected by cebe_loss / auger_loss.
_HISTORY_FIELDS = [
    "epoch", "stage", "lr",
    "train_loss", "val_loss",
    "train_cebe", "val_cebe",
    "train_auger", "val_auger",
    "w_cebe", "w_auger",
    "log_var_cebe", "log_var_auger",
]

def _write_loss_history(out_dir, run_tag, history):
    """Write the full per-epoch loss / uncertainty-weight history to one CSV.

    Parameters
    ----------
    out_dir : str``cfg.outputs_dir``
    run_tag : str Run identifier, derived from model_tag
    history : list[dict] One dict per epoch see ``_HISTORY_FIELDS`` above

    Returns
    -------
    str
        Path of history csv file.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{run_tag}_loss_history.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_HISTORY_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k, "") for k in _HISTORY_FIELDS})
    return path

# Recursive function for getting all model parameter names in PyTorch tree structure
# Taken from Hugging Face github transformers/src/transformers/trainer_pt_utils.py
def get_parameter_names(model, forbidden_layer_types, forbidden_layer_names=None):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    forbidden_layer_patterns = (
        [re.compile(pattern) for pattern in forbidden_layer_names] if forbidden_layer_names is not None else []
    )
    result = []
    for name, child in model.named_children():
        child_params = get_parameter_names(child, forbidden_layer_types, forbidden_layer_names)
        result += [
            f"{name}.{n}"
            for n in child_params
            if not isinstance(child, tuple(forbidden_layer_types))
            and not any(pattern.search(f"{name}.{n}".lower()) for pattern in forbidden_layer_patterns)
        ]
    # Add model specific parameters that are not in any child
    result += [
        k for k in model._parameters if not any(pattern.search(k.lower()) for pattern in forbidden_layer_patterns)
    ]

    return result


############################################################################ 
# validation run
############################################################################

def validate_mpnn(data_loader, model, device, pred_type, cebe_loss_fn, auger_loss_fn,
                  task_type='single'):
    """
        One pass over data_loader without gradient to compute mean loss.
    """

    model.eval()
    total_loss, n_batches = 0.0, 0
    run_cebe, run_auger, n_joint = 0.0, 0.0, 0
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out = model(data)

            if task_type == 'multi':
                # Multi-task: out is (cebe_out, auger_out) tuple
                cebe_out, auger_out = out
                idx = data.node_mask.nonzero(as_tuple=True)[0]
                # CEBE loss
                loss_cebe = cebe_loss_fn(cebe_out[idx], data.cebe_y[idx])
                # Auger loss
                out_sel = auger_out[idx]
                y_sel = data.y_fitted[idx]
                loss_auger = auger_loss_fn(out_sel, y_sel)

                run_cebe  += loss_cebe.item()
                run_auger += loss_auger.item()
                n_joint   += 1

                # Uncertainty-weighted combined loss.  Same formulation as the
                # training loop: scale * (exp(-log_var)*L + log_var) per task,
                # with scale = 0.5 for MSE (Gaussian) / 1.0 for MAE (Laplace).
                # Mixed MSE/MAE combinations are handled automatically.
                loss = _uw_loss(loss_cebe, loss_auger, model.log_var,
                                _uw_scale(cebe_loss_fn,  "CEBE"),
                                _uw_scale(auger_loss_fn, "Auger"))

            elif task_type == 'single':
                if isinstance(out, tuple):
                    out = out[1]
                if pred_type == "CEBE":
                    idx  = data.node_mask.nonzero(as_tuple=True)[0]
                    loss = cebe_loss_fn(out[idx], data.cebe_y[idx])
                elif pred_type == "AUGER":
                    idx  = data.node_mask.nonzero(as_tuple=True)[0]
                    out_sel = out[idx]
                    y_sel = data.y_fitted[idx]
                    loss = auger_loss_fn(out_sel, y_sel)

            total_loss += loss.item()
            n_batches  += 1
    val_loss = total_loss / n_batches
    comp = ({'cebe':  run_cebe  / n_joint,
             'auger': run_auger / n_joint} if n_joint > 0 else None)
    return val_loss, comp

    #return total_loss / n_batches


############################################################################
# Training run
############################################################################

def train_loop(train_list: list, val_list: list, model: nn.Module, device,
                num_epochs: int = 100, batch_size=64, max_lr=1e-2, pct_start=0.6, 
                verbose = True, pred_type="AUGER", cebe_loss='mse', auger_loss='mse', 
                patience=50, random_seed=0, optimizer_type='adamw', weight_decay=1e-4, gradient_clip_norm=0.5, 
                warmup_epochs=10, min_lr=1e-7, scheduler_type='cosine', task_type='single', 
                mt_warmup_epochs=10, mt_finetune_auger=False, mt_finetune_epochs=50,
                out_dir=None, run_tag=None
                ):
    """
    Training loop with gradient clipping, configurable optimizer and LR scheduler.

    Also multi-task training for:
    Step 1: CEBE warmup
    Step 2: joint CEBE + Auger training with uncertainty weighting
    Step 3: fine-tuning Auger model

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
        val_data_list: Validation data (if None, will split from training data)
        optimizer_type: 'adam', 'adamw' (default: 'adamw')
        weight_decay: L2 regularization weight
        gradient_clip_norm: Max gradient norm for clipping (default: 1.0)
        warmup_epochs: Number of epochs for warmup in cosine scheduler (default: 10)
        min_lr: Minimum learning rate for cosine scheduler (default: 1e-7)
        scheduler_type: 'cosine' (CosineAnnealingWarmup, per-epoch) or
                        'onecycle' (OneCycleLR, per-batch — original AUGER schedule)
        out_dir: directory for the per-epoch loss-history CSV.  If None
                 (or run_tag is None) no history file is written.
        run_tag: run identifier used as the CSV stem, i.e.
                 ``{out_dir}/{run_tag}_loss_history.csv``.  Should match the
                 model ``.pth`` stem — see ``_write_loss_history``.
    """

    seed(random_seed)
    gen = torch.Generator().manual_seed(0)

    # dict to toggle loss function options
    _loss_fn = {'mse': F.mse_loss, 'mae': F.l1_loss}
    cebe_loss_fn  = _loss_fn[cebe_loss]
    auger_loss_fn = _loss_fn[auger_loss]

    # UW prefactors implied by the chosen task losses (0.5 MSE / 1.0 MAE).
    # Resolved once here so the objective, the history CSV and the epoch
    # printout all use the same formulation.  Raises for anything unsupported.
    uw_scale_cebe  = _uw_scale(cebe_loss,  "CEBE")
    uw_scale_auger = _uw_scale(auger_loss, "Auger")

    train_set = train_list
    val_set = val_list
    print(f"Training samples: {len(train_set)}, carbons: {sum(s == 'C' for d in train_set for s in d.atom_symbols)}")
    print(f"Validation samples: {len(val_set)}, carbons: {sum(s == 'C' for d in val_set for s in d.atom_symbols)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, generator=gen,
                                pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, generator=gen,
                                pin_memory=(device.type == "cuda"))

    #Split model paramters into:
    #   Those which should under weight decay in AdamW Optimizer: model weights
    #   Those which should not: LayerNrom, bias, log_var (for multitask)
    #   Uses recursive get_parametr_names I copied in from hugging face transformers
    decay_names  = get_parameter_names(model, [nn.LayerNorm])
    decay_names  = [n for n in decay_names if "bias" not in n and n != "log_var"]

    param_groups = [
        {"params": [p for n, p in model.named_parameters() if n in decay_names],
            "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if n not in decay_names],
            "weight_decay": 0.0},
    ]

    # Optimizer 
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=max_lr, betas=(0.9, 0.999))
    else:
        optimizer = torch.optim.Adam(param_groups,  lr=max_lr)

    # Scheduler
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
        # CosineAnnealingWarmup: steps per epoch — smoother schedule.
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
    history = []          # per-epoch records for the loss-history CSV
    write_history = bool(out_dir) and bool(run_tag)
    best_val_loss = float('inf')
    patience_counter = 0
    patience = patience  # Early stopping patience
    best_model_state = None  # Track best model weights

    for epoch in range(num_epochs):

        model.train()
        # LR the epoch starts on.  With OneCycleLR (per-batch) this drifts
        # within the epoch; recorded here as the representative value.
        epoch_lr = optimizer.param_groups[0]['lr']
        running_loss, n_batches = 0.0, 0
        run_cebe, run_auger, n_joint = 0.0, 0.0, 0

        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            out = model(data)
            idx = data.node_mask.nonzero(as_tuple=True)[0]
            if task_type == 'multi':
                # Multi-task: out is (cebe_out, auger_out)
                cebe_out, auger_out = out
                # CEBE loss
                loss_cebe = cebe_loss_fn(cebe_out[idx], data.cebe_y[idx])
                run_cebe += loss_cebe.item()
                if epoch < mt_warmup_epochs:
                    # Stage 1: CEBE-only warmup — stabilise encoder first
                    loss = loss_cebe
                else:
                    # Stage 2: joint training with uncertainty weighting
                    out_sel = auger_out[idx]
                    y_sel = data.y_fitted[idx]
                    loss_auger = auger_loss_fn(out_sel, y_sel)

                    # NOTE: loss_cebe is accumulated into run_cebe once, above,
                    # for every batch in both stages.  Do not accumulate it
                    # again here — doing so doubled the reported train CEBE
                    # loss in the joint stage.
                    run_auger += loss_auger.item()
                    n_joint   += 1

                    # Uncertainty-weighted combined loss.
                    #   scale * ( exp(-log_var) * L + log_var )   per task
                    # with scale = 0.5 for an MSE head (Gaussian likelihood,
                    # log_var = log sigma^2) and 1.0 for an MAE head (Laplace
                    # likelihood, log_var = log b).  Mixed MSE/MAE combinations
                    # fall out of the same expression.
                    # See Kendall et al. for the UW derivation.
                    loss = _uw_loss(loss_cebe, loss_auger, model.log_var,
                                    uw_scale_cebe, uw_scale_auger)

                    #loss = model.awl(loss_cebe, loss_auger, loss_alpha)
                    if epoch == mt_warmup_epochs and n_batches == 0 and verbose:
                        print(f"  [multi] Switching to joint UW training at epoch {epoch}"
                              f" │ UW form: cebe={cebe_loss}/{_UW_LIKELIHOOD[cebe_loss]}"
                              f" (scale {uw_scale_cebe}),"
                              f" auger={auger_loss}/{_UW_LIKELIHOOD[auger_loss]}"
                              f" (scale {uw_scale_auger})")
                        print("           reported w(c/a) = scale * exp(-log_var)"
                              " (effective weight on each task loss)")

            elif pred_type == "CEBE": 
                loss = cebe_loss_fn(out[idx], data.cebe_y[idx])    
            elif pred_type == "AUGER":

                out_sel = out[idx]
                # DEBUG: Print shapes on first batch of first epoch
                if epoch == 0 and n_batches == 0:
                    print(f"DEBUG AUGER: idx.shape={idx.shape}")
                    print(f"DEBUG AUGER: out_sel.shape={out_sel.shape}")
                # Fitted: target is data.y_fitted (N, n_points), no mask needed
                y_sel = data.y_fitted[idx]
                if epoch == 0 and n_batches == 0:
                    print(f"DEBUG AUGER: y_fitted_sel.shape={y_sel.shape}")
                loss = auger_loss_fn(out_sel, y_sel)

            loss.backward()

            # Gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)

            optimizer.step()

            # OneCycleLR steps per batch
            if scheduler_per_batch:
                scheduler.step()

            running_loss += loss.item()
            n_batches    += 1

        train_loss = running_loss / n_batches

        val_loss, val_comp  = validate_mpnn(val_loader, model, device, pred_type,
                                   cebe_loss_fn, auger_loss_fn,
                                   task_type=task_type)
        
        train_results.append([epoch, train_loss, val_loss])

        # ── Loss-history record ──────────────────────────────────────────
        # Built here (rather than next to the verbose print) so that the
        # epoch that trips early stopping is still recorded, matching
        # train_results exactly.
        if write_history:
            row = {"epoch": epoch, "lr": epoch_lr,
                   "train_loss": train_loss, "val_loss": val_loss}
            if task_type == 'multi':
                lv_rec = model.log_var.detach().cpu()
                # Effective weights: scale * exp(-log_var), i.e. what actually
                # multiplies each task loss under the selected UW formulation
                # (0.5 for an MSE head, 1.0 for an MAE head).
                w_cebe_eff, w_auger_eff = _uw_weights(model.log_var,
                                                      uw_scale_cebe,
                                                      uw_scale_auger)
                row["stage"] = ("warmup" if epoch < mt_warmup_epochs
                                else "joint")
                row["train_cebe"]    = run_cebe / max(n_batches, 1)
                row["w_cebe"]        = w_cebe_eff
                row["w_auger"]       = w_auger_eff
                row["log_var_cebe"]  = lv_rec[0].item()
                row["log_var_auger"] = lv_rec[1].item()
                if n_joint > 0:
                    row["train_auger"] = run_auger / n_joint
                if val_comp is not None:
                    row["val_cebe"]  = val_comp["cebe"]
                    row["val_auger"] = val_comp["auger"]
            else:
                # Single-task: mirror the loss into its component column so
                # the CSV is parsed uniformly across run types.
                row["stage"] = pred_type.lower()
                if pred_type == "CEBE":
                    row["train_cebe"],  row["val_cebe"]  = train_loss, val_loss
                else:
                    row["train_auger"], row["val_auger"] = train_loss, val_loss
            history.append(row)
            _write_loss_history(out_dir, run_tag, history)

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
            #print(f"Epoch {epoch:03d} │ train {train_loss:.5f} │ val {val_loss:.5f}")
            msg = f"Epoch {epoch:03d} │ train {train_loss:.5f} │ val {val_loss:.5f}"
            #if task_type == 'multi' and (epoch % 10 == 0 or epoch == num_epochs - 1):
            if task_type == 'multi':
                # Effective UW weights (scale * exp(-log_var)) so the printed
                # w(c/a) matches the objective for MAE, MSE and mixed runs.
                w_c, w_a = _uw_weights(model.log_var,
                                       uw_scale_cebe, uw_scale_auger)
                c  = run_cebe / max(n_batches, 1)
                if epoch < mt_warmup_epochs:
                    msg += f" │ trnL(c)={c:.4f} (warmup)"
                elif n_joint > 0:
                    a = run_auger / n_joint
                    msg += (f" │ trnL(c/a)={c:.4f}/{a:.4f}"
                            f" │ w(c/a)={w_c:.3f}/{w_a:.3f}")
                    if val_comp is not None:
                        msg += (f" │ valL(c/a)="
                                f"{val_comp['cebe']:.4f}/{val_comp['auger']:.4f}")
            print(msg)

    # If training finished all epochs without early stopping, the model still
    # holds last-epoch weights. Restore the best-val checkpoint so the returned
    # model is always the best-val one, regardless of whether patience fired.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ── Optional multi-task Stage 3: Auger-only fine-tune ────────────────────
    if task_type == 'multi' and mt_finetune_auger and mt_finetune_epochs > 0:
        if verbose:
            print(f"\n[multi] Stage 3: Auger-only fine-tune for {mt_finetune_epochs} epochs")
        # Freeze CEBE adapter and lin_pred; unfreeze Auger decoder + adapter
        for name, p in model.named_parameters():
            if 'adapter_cebe' in name or 'lin_pred' in name:
                p.requires_grad_(False)
        ft_decay, ft_no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:          # adapter_cebe / lin_pred are frozen here
                continue
            (ft_decay if n in decay_names else ft_no_decay).append(p)

#         ft_optimizer = torch.optim.AdamW(
#             filter(lambda p: p.requires_grad, model.parameters()),
#             lr=max_lr * 0.1, weight_decay=weight_decay
#         )
        ft_optimizer = torch.optim.AdamW(
            [{"params": ft_decay,    "weight_decay": weight_decay},
            {"params": ft_no_decay, "weight_decay": 0.0}],
            lr=max_lr * 0.1, betas=(0.9, 0.999),
        )

        for ft_epoch in range(mt_finetune_epochs):
            model.train()
            ft_loss, ft_n = 0.0, 0
            for data in train_loader:
                ft_optimizer.zero_grad()
                data = data.to(device)
                _, auger_out = model(data)
                idx = data.node_mask.nonzero(as_tuple=True)[0]
                out_sel = auger_out[idx]
                y_sel = data.y_fitted[idx]
                loss = auger_loss_fn(out_sel, y_sel)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                ft_optimizer.step()
                ft_loss += loss.item(); ft_n += 1

            ft_train = ft_loss / ft_n

            ft_val, _  = validate_mpnn(val_loader, model, device, pred_type,
                                     cebe_loss_fn, auger_loss_fn,
                                     task_type='single')
            
            train_results.append([num_epochs + ft_epoch, ft_train, ft_val])

            if write_history:
                lv_rec = model.log_var.detach().cpu()
                w_cebe_eff, w_auger_eff = _uw_weights(model.log_var,
                                                      uw_scale_cebe,
                                                      uw_scale_auger)
                # Stage 3 optimises the plain Auger MAE, so train_loss /
                # val_loss are Auger-only here and the CEBE columns are left
                # empty.  log_var is frozen but recorded for continuity.
                history.append({
                    "epoch": num_epochs + ft_epoch,
                    "stage": "finetune",
                    "lr": ft_optimizer.param_groups[0]['lr'],
                    "train_loss": ft_train, "val_loss": ft_val,
                    "train_auger": ft_train, "val_auger": ft_val,
                    "w_cebe": w_cebe_eff, "w_auger": w_auger_eff,
                    "log_var_cebe": lv_rec[0].item(),
                    "log_var_auger": lv_rec[1].item(),
                })
                _write_loss_history(out_dir, run_tag, history)

            if verbose:
                print(f"  FT Epoch {ft_epoch:03d} │ train {ft_train:.5f} │ val {ft_val:.5f}")
        # Re-enable all parameters
        for p in model.parameters():
            p.requires_grad_(True)

    if write_history and verbose:
        print(f"\n  Loss history ({len(history)} epochs) written to "
              f"{os.path.join(out_dir, f'{run_tag}_loss_history.csv')}")

    return train_results

############################################################################
# Permuation, translation and rotation tests
############################################################################

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
    if hasattr(data, 'cebe_y') and data.cebe_y is not None and data.cebe_y.size(0) == perm.size(0):
        data.cebe_y = data.cebe_y[perm]
    if hasattr(data, 'node_mask') and data.node_mask is not None:
        data.node_mask = data.node_mask[perm]

    # Permute the edge index
    adj = to_dense_adj(data.edge_index)
    adj = adj[:, perm, :]
    adj = adj[:, :, perm]
    data.edge_index = dense_to_sparse(adj)[0]

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
    # Multi-task models return (cebe_out, auger_out) — test the Auger head
    if isinstance(out_1, tuple):
        out_1 = out_1[1]

    # Create random permutation
    perm = torch.randperm(data.x.shape[0])
    data = permute_graph(data, perm)

    # Forward pass on permuted example
    out_2 = module(data)
    if isinstance(out_2, tuple):
        out_2 = out_2[1]

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
        # Multi-task models return (cebe_out, auger_out) — test the Auger head
        if isinstance(out_1, tuple):
            out_1 = out_1[1]
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
        if isinstance(out_2, tuple):
            out_2 = out_2[1]
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