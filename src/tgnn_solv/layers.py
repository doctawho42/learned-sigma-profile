"""
Neural network layers for TGNN-Solv.

Contains:
  - MPNNLayer        : Message-passing with edge features and attention
  - GNNEncoder       : Multi-layer MPNN with solute/solvent role adapters
  - SoluteSolventCrossAttention : Cross-attention between solute and solvent
  - BipartiteMessagePassing : Bipartite message passing alternative
  - AttentionPooling : Learnable attention-based graph readout
  - PhysicsAwareReadout : Multi-strategy pooling (attention + Set2Set)
  - IdealSolubilityLayer : Hardcoded Φ(T) from SLE theory (0 params)
  - NRTLLayer        : Hardcoded NRTL activity coefficient model (0 params)
  - HansenDistanceLayer : Hardcoded Hansen Ra distance (0 params)

Physics layers have ZERO learnable parameters — all thermodynamic
equations are implemented as fixed differentiable functions.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import (
    MessagePassing,
    Set2Set,
)
from torch_geometric.utils import to_dense_batch

from .positional_encoding import PositionalEncoding
from .features import graph_feature_spec_from_config

if TYPE_CHECKING:
    from .config import TGNNSolvConfig

TensorList: TypeAlias = list[Tensor]
NRTLState: TypeAlias = dict[str, Tensor]


def make_temperature_features(T: Tensor) -> Tensor:
    """
    Build normalized temperature features.

    Returns (B, 3): [ (T-300)/50, (1/T - 1/300)*300, log(T/300) ].
    """
    t = torch.clamp(T.float(), min=1e-6)
    t_norm = (t - 300.0) / 50.0
    inv_norm = (1.0 / t - 1.0 / 300.0) * 300.0
    log_norm = torch.log(t / 300.0)
    return torch.stack([t_norm, inv_norm, log_norm], dim=-1)


# ================================================================== #
#  Scatter utilities (no torch_scatter dependency)                    #
# ================================================================== #

def scatter_add(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    dim_size: int | None = None,
) -> Tensor:
    """Scatter-add using native PyTorch."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    if src.dim() > 1 and index.dim() == 1:
        # .contiguous() materializes the broadcast: expand_as() yields a
        # zero-stride view that CPU scatter tolerates but the CUDA kernel
        # reads as illegal memory (zero-stride expanded idx).
        idx = index.unsqueeze(-1).expand_as(src).contiguous()
    else:
        idx = index
    out = torch.zeros(dim_size, *src.shape[1:],
                      dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, idx, src)


def scatter_mean(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    dim_size: int | None = None,
) -> Tensor:
    """Scatter-mean using native PyTorch."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    if src.dim() > 1 and index.dim() == 1:
        # See scatter_add: expand_as() is a zero-stride view; .contiguous()
        # is required for the CUDA scatter kernel (no-op cost on CPU).
        idx = index.unsqueeze(-1).expand_as(src).contiguous()
    else:
        idx = index
    out_sum = torch.zeros(dim_size, *src.shape[1:],
                          dtype=src.dtype, device=src.device)
    out_sum.scatter_add_(dim, idx, src)
    count = torch.zeros(dim_size, *src.shape[1:],
                        dtype=src.dtype, device=src.device)
    count.scatter_add_(dim, idx, torch.ones_like(src))
    return out_sum / count.clamp(min=1)


# ================================================================== #
#  Message-Passing GNN                                                #
# ================================================================== #

class MPNNLayer(MessagePassing):
    """
    Single message-passing layer with edge features and attention gating.

    message:  m_ij = MLP([h_i || h_j || e_ij]) * sigmoid(attn(m_ij))
    update:   h_i' = LayerNorm(h_i + MLP([h_i || agg_j m_ij]))
    """

    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        super().__init__(aggr="add")

        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        x_new = self.upd_mlp(torch.cat([x, out], dim=-1))
        return self.norm(x + x_new)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        msg = self.msg_mlp(msg_input)
        attn = torch.sigmoid(self.attn_mlp(msg))
        return msg * attn


class GNNEncoder(nn.Module):
    """
    Multi-layer GNN encoder with role-specific adapters.

    Two modes are supported:
      - ``shared_residual``: the full stack is shared, then lightweight role
        adapters provide a small late specialization.
      - ``split_late``: early layers are shared, while the last
        ``role_specific_layers`` are separate for solute and solvent.
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 6,
        role_mode: str = "shared_residual",
        role_specific_layers: int = 2,
    ) -> None:
        super().__init__()
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if role_mode not in {"shared_residual", "split_late"}:
            raise ValueError(f"Unsupported role_mode: {role_mode}")

        self.role_mode = role_mode
        if role_mode == "split_late":
            self.role_specific_layers = min(
                max(int(role_specific_layers), 0),
                max(n_layers - 1, 0),
            )
        else:
            self.role_specific_layers = 0
        self.shared_layer_count = n_layers - self.role_specific_layers

        self.node_embed = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_feat_dim, hidden_dim)

        self.shared_layers = nn.ModuleList(
            [
                MPNNLayer(hidden_dim, hidden_dim)
                for _ in range(self.shared_layer_count)
            ]
        )
        self.solute_layers = nn.ModuleList(
            [
                MPNNLayer(hidden_dim, hidden_dim)
                for _ in range(self.role_specific_layers)
            ]
        )
        self.solvent_layers = nn.ModuleList(
            [
                MPNNLayer(hidden_dim, hidden_dim)
                for _ in range(self.role_specific_layers)
            ]
        )

        self.solute_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        self.solvent_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )

        self.temp_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

    def _run_layer_stack(
        self,
        h: Tensor,
        e: Tensor,
        edge_index: Tensor,
        layers: nn.ModuleList,
        batch: Optional[Tensor],
        temp_feat: Optional[Tensor],
    ) -> Tensor:
        for layer in layers:
            h = layer(h, edge_index, e)
            if temp_feat is not None and batch is not None:
                h = self._apply_temp_film(h, batch, temp_feat)
        return h

    def _apply_temp_film(
        self,
        h: Tensor,
        batch: Tensor,
        temp_feat: Tensor,
    ) -> Tensor:
        gamma_beta = self.temp_mlp(temp_feat)  # (B, 2D)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma[batch]
        beta = beta[batch]
        return h * (1.0 + gamma) + beta

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        role: str = "solute",
        batch: Optional[Tensor] = None,
        temp_feat: Optional[Tensor] = None,
    ) -> Tensor:
        h = self.node_embed(x)
        e = self.edge_embed(edge_attr)

        h = self._run_layer_stack(
            h,
            e,
            edge_index,
            self.shared_layers,
            batch,
            temp_feat,
        )

        if self.role_mode == "split_late":
            role_layers = (
                self.solute_layers if role == "solute" else self.solvent_layers
            )
            h = self._run_layer_stack(
                h,
                e,
                edge_index,
                role_layers,
                batch,
                temp_feat,
            )
        elif role == "solute":
            h = h + self.solute_adapter(h)
        else:
            h = h + self.solvent_adapter(h)

        if temp_feat is not None and batch is not None:
            h = self._apply_temp_film(h, batch, temp_feat)

        return h


class GPSLayer(nn.Module):
    """GPS block: local message passing plus per-graph global attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        use_edge_attr: bool,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.use_edge_attr = use_edge_attr
        self.local_mpnn = MPNNLayer(hidden_dim, hidden_dim)
        self.global_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.local_dropout = nn.Dropout(dropout)
        self.global_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def _batched_attention(
        self,
        h: Tensor,
        batch: Tensor,
    ) -> Tensor:
        """Apply dense self-attention independently to each graph in the batch."""
        h_dense, mask = to_dense_batch(h, batch)
        h_attn, _ = self.global_attn(
            h_dense,
            h_dense,
            h_dense,
            key_padding_mask=~mask,
            need_weights=False,
        )
        return h_attn[mask]

    def forward(
        self,
        h: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        base_h = h
        if batch is None:
            batch = torch.zeros(
                h.size(0),
                dtype=torch.long,
                device=h.device,
            )

        local_edge_attr = edge_attr
        if not self.use_edge_attr:
            local_edge_attr = torch.zeros_like(edge_attr)

        local_out = self.local_mpnn(base_h, edge_index, local_edge_attr)
        local_delta = local_out - base_h
        global_out = self._batched_attention(base_h, batch)

        h = self.norm1(base_h + self.local_dropout(local_delta))
        h = self.norm2(h + self.global_dropout(global_out))
        h = self.norm3(h + self.ffn(h))
        return h


class GPSEncoder(nn.Module):
    """Role-aware encoder that augments local MPNN updates with global attention."""

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 6,
        role_mode: str = "shared_residual",
        role_specific_layers: int = 2,
        *,
        gps_num_heads: int = 8,
        gps_use_edge_attr: bool = True,
        gps_positional_encoding: str = "laplacian",
        gps_pe_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if role_mode not in {"shared_residual", "split_late"}:
            raise ValueError(f"Unsupported role_mode: {role_mode}")

        self.role_mode = role_mode
        if role_mode == "split_late":
            self.role_specific_layers = min(
                max(int(role_specific_layers), 0),
                max(n_layers - 1, 0),
            )
        else:
            self.role_specific_layers = 0
        self.shared_layer_count = n_layers - self.role_specific_layers

        self.positional_encoder = PositionalEncoding(
            gps_pe_dim,
            gps_positional_encoding,
        )
        self.node_embed = nn.Linear(node_feat_dim + gps_pe_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_feat_dim, hidden_dim)

        self.shared_layers = nn.ModuleList(
            [
                GPSLayer(
                    hidden_dim,
                    gps_num_heads,
                    dropout,
                    gps_use_edge_attr,
                )
                for _ in range(self.shared_layer_count)
            ]
        )
        self.solute_layers = nn.ModuleList(
            [
                GPSLayer(
                    hidden_dim,
                    gps_num_heads,
                    dropout,
                    gps_use_edge_attr,
                )
                for _ in range(self.role_specific_layers)
            ]
        )
        self.solvent_layers = nn.ModuleList(
            [
                GPSLayer(
                    hidden_dim,
                    gps_num_heads,
                    dropout,
                    gps_use_edge_attr,
                )
                for _ in range(self.role_specific_layers)
            ]
        )

        self.solute_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        self.solvent_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )

        self.temp_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

    def _run_layer_stack(
        self,
        h: Tensor,
        e: Tensor,
        edge_index: Tensor,
        layers: nn.ModuleList,
        batch: Optional[Tensor],
        temp_feat: Optional[Tensor],
    ) -> Tensor:
        for layer in layers:
            h = layer(h, edge_index, e, batch=batch)
            if temp_feat is not None and batch is not None:
                h = self._apply_temp_film(h, batch, temp_feat)
        return h

    def _apply_temp_film(
        self,
        h: Tensor,
        batch: Tensor,
        temp_feat: Tensor,
    ) -> Tensor:
        gamma_beta = self.temp_mlp(temp_feat)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma[batch]
        beta = beta[batch]
        return h * (1.0 + gamma) + beta

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        role: str = "solute",
        batch: Optional[Tensor] = None,
        temp_feat: Optional[Tensor] = None,
    ) -> Tensor:
        if batch is None:
            batch = torch.zeros(
                x.size(0),
                dtype=torch.long,
                device=x.device,
            )

        pe = self.positional_encoder(edge_index, x.size(0), batch=batch).to(x.dtype)
        h = self.node_embed(torch.cat([x, pe], dim=-1))
        e = self.edge_embed(edge_attr)

        h = self._run_layer_stack(
            h,
            e,
            edge_index,
            self.shared_layers,
            batch,
            temp_feat,
        )

        if self.role_mode == "split_late":
            role_layers = (
                self.solute_layers if role == "solute" else self.solvent_layers
            )
            h = self._run_layer_stack(
                h,
                e,
                edge_index,
                role_layers,
                batch,
                temp_feat,
            )
        elif role == "solute":
            h = h + self.solute_adapter(h)
        else:
            h = h + self.solvent_adapter(h)

        if temp_feat is not None and batch is not None:
            h = self._apply_temp_film(h, batch, temp_feat)

        return h


class TIMPLayer(nn.Module):
    """Thermodynamics-informed message passing with dispersive and polar channels."""

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError("TIMP requires an even hidden_dim.")
        half_dim = hidden_dim // 2
        self.hidden_dim = hidden_dim
        self.half_dim = half_dim
        self.phi_disp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, half_dim),
        )
        self.w_disp = nn.Parameter(torch.ones(1))
        self.b_disp = nn.Parameter(torch.zeros(1))
        self.phi_polar = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, half_dim),
        )
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        alpha: Tensor,
        charges: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        src, dst = edge_index
        h_dst = h[dst]
        h_src = h[src]
        inp = torch.cat([h_dst, h_src, edge_attr], dim=-1)

        m_disp = self.phi_disp(inp)
        disp_scale = (
            self.w_disp
            * torch.sqrt((alpha[src] * alpha[dst]).clamp(min=0.0) + 1.0e-8)
            + self.b_disp
        )
        m_disp = m_disp * disp_scale.unsqueeze(-1)

        m_polar = self.phi_polar(inp)
        polar_scale = F.softplus(charges[src] * charges[dst])
        m_polar = m_polar * polar_scale.unsqueeze(-1)

        a_disp = scatter_add(m_disp, dst, dim=0, dim_size=h.size(0))
        a_polar = scatter_add(m_polar, dst, dim=0, dim_size=h.size(0))
        a = self.combine(torch.cat([a_disp, a_polar], dim=-1))
        h_new = h + self.dropout(self.update(torch.cat([h, a], dim=-1)))
        h_new = self.norm(h_new)
        return h_new, a_disp, a_polar


class TIMPEncoder(nn.Module):
    """Role-aware TIMP stack with the same external API as the standard encoders."""

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 6,
        role_mode: str = "shared_residual",
        role_specific_layers: int = 2,
        *,
        alpha_idx: int,
        charge_idx: int | None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if role_mode not in {"shared_residual", "split_late"}:
            raise ValueError(f"Unsupported role_mode: {role_mode}")
        if hidden_dim % 2 != 0:
            raise ValueError("TIMP requires an even hidden_dim.")

        self.role_mode = role_mode
        if role_mode == "split_late":
            self.role_specific_layers = min(
                max(int(role_specific_layers), 0),
                max(n_layers - 1, 0),
            )
        else:
            self.role_specific_layers = 0
        self.shared_layer_count = n_layers - self.role_specific_layers

        self.alpha_idx = int(alpha_idx)
        self.charge_idx = charge_idx if charge_idx is None else int(charge_idx)
        self.channel_dim = hidden_dim // 2

        self.node_embed = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_feat_dim, hidden_dim)
        self.shared_layers = nn.ModuleList(
            [
                TIMPLayer(hidden_dim, hidden_dim, dropout)
                for _ in range(self.shared_layer_count)
            ]
        )
        self.solute_layers = nn.ModuleList(
            [
                TIMPLayer(hidden_dim, hidden_dim, dropout)
                for _ in range(self.role_specific_layers)
            ]
        )
        self.solvent_layers = nn.ModuleList(
            [
                TIMPLayer(hidden_dim, hidden_dim, dropout)
                for _ in range(self.role_specific_layers)
            ]
        )
        self.solute_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        self.solvent_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        self.temp_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

    def _apply_temp_film(
        self,
        h: Tensor,
        batch: Tensor,
        temp_feat: Tensor,
    ) -> Tensor:
        gamma_beta = self.temp_mlp(temp_feat)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return h * (1.0 + gamma[batch]) + beta[batch]

    def _run_layer_stack(
        self,
        h: Tensor,
        e: Tensor,
        edge_index: Tensor,
        layers: nn.ModuleList,
        alpha: Tensor,
        charges: Tensor,
        batch: Optional[Tensor],
        temp_feat: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        a_disp = torch.zeros(
            h.size(0),
            self.channel_dim,
            device=h.device,
            dtype=h.dtype,
        )
        a_polar = torch.zeros_like(a_disp)
        for layer in layers:
            h, a_disp, a_polar = layer(h, edge_index, e, alpha, charges)
            if temp_feat is not None and batch is not None:
                h = self._apply_temp_film(h, batch, temp_feat)
        return h, a_disp, a_polar

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        role: str = "solute",
        batch: Optional[Tensor] = None,
        temp_feat: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if batch is None:
            batch = torch.zeros(
                x.size(0),
                dtype=torch.long,
                device=x.device,
            )
        h = self.node_embed(x)
        e = self.edge_embed(edge_attr)
        alpha = x[:, self.alpha_idx].to(h)
        if self.charge_idx is None:
            charges = torch.zeros_like(alpha)
        else:
            charges = x[:, self.charge_idx].to(h)

        h, a_disp, a_polar = self._run_layer_stack(
            h,
            e,
            edge_index,
            self.shared_layers,
            alpha,
            charges,
            batch,
            temp_feat,
        )

        if self.role_mode == "split_late":
            role_layers = (
                self.solute_layers if role == "solute" else self.solvent_layers
            )
            h, a_disp, a_polar = self._run_layer_stack(
                h,
                e,
                edge_index,
                role_layers,
                alpha,
                charges,
                batch,
                temp_feat,
            )
        elif role == "solute":
            h = h + self.solute_adapter(h)
        else:
            h = h + self.solvent_adapter(h)

        if temp_feat is not None and batch is not None:
            h = self._apply_temp_film(h, batch, temp_feat)
        return h, a_disp, a_polar


def build_graph_encoder(
    node_feat_dim: int,
    edge_feat_dim: int,
    cfg: "TGNNSolvConfig",
) -> nn.Module:
    """Construct the configured graph encoder while keeping the downstream API stable."""
    encoder_type = getattr(cfg, "encoder_type", "mpnn")
    if encoder_type == "gps":
        return GPSEncoder(
            node_feat_dim,
            edge_feat_dim,
            hidden_dim=cfg.hidden_dim,
            n_layers=cfg.n_gnn_layers,
            role_mode=cfg.encoder_role_mode,
            role_specific_layers=cfg.encoder_role_specific_layers,
            gps_num_heads=cfg.gps_num_heads,
            gps_use_edge_attr=cfg.gps_use_edge_attr,
            gps_positional_encoding=cfg.gps_positional_encoding,
            gps_pe_dim=cfg.gps_pe_dim,
            dropout=cfg.dropout,
        )
    if encoder_type == "timp":
        spec = graph_feature_spec_from_config(cfg)
        return TIMPEncoder(
            node_feat_dim,
            edge_feat_dim,
            hidden_dim=cfg.hidden_dim,
            n_layers=cfg.n_gnn_layers,
            role_mode=cfg.encoder_role_mode,
            role_specific_layers=cfg.encoder_role_specific_layers,
            alpha_idx=spec.polarizability_idx,
            charge_idx=spec.gasteiger_charge_idx,
            dropout=cfg.dropout,
        )
    return GNNEncoder(
        node_feat_dim,
        edge_feat_dim,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_gnn_layers,
        role_mode=cfg.encoder_role_mode,
        role_specific_layers=cfg.encoder_role_specific_layers,
    )


# ================================================================== #
#  Cross-Attention                                                    #
# ================================================================== #

class SoluteSolventCrossAttention(nn.Module):
    """
    Transformer-style cross-attention: solute atoms attend to solvent atoms.

    One block = MultiheadAttention + LayerNorm + FFN + LayerNorm.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 8,
        *,
        use_thermo_cross_attention: bool = False,
        thermo_cross_attention_beta_init: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
            )
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.use_thermo_cross_attention = use_thermo_cross_attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(0.1)
        if use_thermo_cross_attention:
            self.energy_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.acc_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(hidden_dim // 4, 1),
            )
            self.beta = nn.Parameter(
                torch.tensor(float(thermo_cross_attention_beta_init))
            )
        else:
            self.energy_mlp = None
            self.acc_mlp = None
            self.register_parameter("beta", None)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.temp_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

    def _shape_heads(self, x: Tensor) -> Tensor:
        """Reshape `(B, N, D)` tensors into `(B, H, N, Dh)`."""
        B, N, _ = x.shape
        return x.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        h_solute: Tensor,
        h_solvent: Tensor,
        solute_mask: Optional[Tensor] = None,
        solvent_mask: Optional[Tensor] = None,
        temp_feat: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        h_solute : (B, N_sol, D)
        h_solvent : (B, N_slv, D)
        solute_mask : (B, N_sol) bool — True for real atoms
        solvent_mask : (B, N_slv) bool — True for real atoms

        Returns
        -------
        h_out : (B, N_sol, D)
        attn_weights : (B, N_sol, N_slv)
        """
        q = self._shape_heads(self.q_proj(h_solute))
        k = self._shape_heads(self.k_proj(h_solvent))
        v = self._shape_heads(self.v_proj(h_solvent))
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if self.use_thermo_cross_attention:
            energy_input = torch.cat(
                [
                    h_solute[:, :, None, :].expand(-1, -1, h_solvent.size(1), -1),
                    h_solvent[:, None, :, :].expand(-1, h_solute.size(1), -1, -1),
                ],
                dim=-1,
            )
            energy_ij = self.energy_mlp(energy_input).squeeze(-1)
            acc_i = torch.sigmoid(self.acc_mlp(h_solute))
            acc_j = torch.sigmoid(self.acc_mlp(h_solvent))
            e_contact = energy_ij * acc_i * acc_j.transpose(1, 2)
            attn_logits = attn_logits + self.beta * e_contact.unsqueeze(1)

        if solvent_mask is not None:
            attn_logits = attn_logits.masked_fill(
                ~solvent_mask[:, None, None, :],
                float("-inf"),
            )

        attn_weights_heads = torch.softmax(attn_logits, dim=-1)
        if solvent_mask is not None and not solvent_mask.all():
            attn_weights_heads = torch.nan_to_num(attn_weights_heads, nan=0.0)
        attn_weights_heads = self.attn_dropout(attn_weights_heads)
        h_cross = torch.matmul(attn_weights_heads, v)
        h_cross = h_cross.transpose(1, 2).contiguous().view(
            h_solute.size(0), h_solute.size(1), self.hidden_dim
        )
        h_cross = self.out_proj(h_cross)
        attn_weights = attn_weights_heads.mean(dim=1)

        h_out = self.norm1(h_solute + h_cross)
        if temp_feat is not None:
            gamma_beta = self.temp_mlp(temp_feat)  # (B, 2D)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            h_out = h_out * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        h_out = self.norm2(h_out + self.ffn(h_out))
        return h_out, attn_weights


class BipartiteMessagePassing(nn.Module):
    """
    Bipartite message passing between solute and solvent atoms.

    Each block aggregates messages across the complete bipartite
    graph (solute ↔ solvent) using MLP-based pairwise transforms.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.msg_sol = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.msg_slv = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm_sol = nn.LayerNorm(hidden_dim)
        self.norm_slv = nn.LayerNorm(hidden_dim)
        self.temp_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

    def forward(
        self,
        h_sol: Tensor,
        h_slv: Tensor,
        sol_mask: Optional[Tensor] = None,
        slv_mask: Optional[Tensor] = None,
        temp_feat: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        B, Ns, D = h_sol.shape
        Nv = h_slv.shape[1]

        sol_exp = h_sol[:, :, None, :].expand(-1, -1, Nv, -1)
        slv_exp = h_slv[:, None, :, :].expand(-1, Ns, -1, -1)

        pair_sol = torch.cat([sol_exp, slv_exp], dim=-1)
        msg_sol = self.msg_sol(pair_sol)  # (B, Ns, Nv, D)
        if slv_mask is not None:
            msg_sol = msg_sol * slv_mask[:, None, :, None]
            denom = slv_mask.sum(dim=1).clamp(min=1).view(B, 1, 1)
        else:
            denom = float(Nv)
        agg_sol = msg_sol.sum(dim=2) / denom

        pair_slv = torch.cat([slv_exp, sol_exp], dim=-1)
        msg_slv = self.msg_slv(pair_slv)  # (B, Ns, Nv, D)
        if sol_mask is not None:
            msg_slv = msg_slv * sol_mask[:, :, None, None]
            denom = sol_mask.sum(dim=1).clamp(min=1).view(B, 1, 1)
        else:
            denom = float(Ns)
        agg_slv = msg_slv.sum(dim=1) / denom

        h_sol_out = self.norm_sol(h_sol + agg_sol)
        h_slv_out = self.norm_slv(h_slv + agg_slv)

        if temp_feat is not None:
            gamma_beta = self.temp_mlp(temp_feat)  # (B, 2D)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            h_sol_out = h_sol_out * (1.0 + gamma[:, None, :]) + beta[:, None, :]
            h_slv_out = h_slv_out * (1.0 + gamma[:, None, :]) + beta[:, None, :]

        return h_sol_out, h_slv_out


def pad_atom_features(
    h_atoms_list: TensorList,
) -> tuple[Tensor, Tensor]:
    """
    Pad variable-length atom tensors into a single (B, N_max, D) batch.

    Returns
    -------
    padded : (B, N_max, D)
    mask   : (B, N_max) bool — True for real atoms
    """
    B = len(h_atoms_list)
    D = h_atoms_list[0].shape[-1]
    N_max = max(h.shape[0] for h in h_atoms_list)

    device = h_atoms_list[0].device
    padded = torch.zeros(B, N_max, D, device=device)
    mask = torch.zeros(B, N_max, dtype=torch.bool, device=device)

    for i, h in enumerate(h_atoms_list):
        n = h.shape[0]
        padded[i, :n, :] = h
        mask[i, :n] = True

    return padded, mask


def build_batch_from_lists(
    h_atoms_list: TensorList,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Build a batch vector for concatenated per-graph tensors."""
    device = h_atoms_list[0].device
    if dtype is None:
        dtype = torch.long
    chunks = []
    for i, h in enumerate(h_atoms_list):
        chunks.append(
            torch.full((h.shape[0],), i, device=device, dtype=dtype)
        )
    return torch.cat(chunks, dim=0)


# ================================================================== #
#  Graph-level Readout                                                #
# ================================================================== #

class AttentionPooling(nn.Module):
    """Learnable attention-weighted sum pooling over graph nodes."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, h: Tensor, batch: Tensor) -> Tensor:
        """
        Parameters
        ----------
        h : (N_total, D)  — all nodes in the batch
        batch : (N_total,) — graph membership indices

        Returns
        -------
        (B, D) graph-level vectors
        """
        attn_logits = self.gate(h)  # (N_total, 1)

        # Subtract per-graph mean for numerical stability
        attn_mean = scatter_mean(attn_logits, batch, dim=0)  # (B, 1)
        attn_logits = attn_logits - attn_mean[batch]

        attn_weights = torch.exp(attn_logits)
        attn_sum = scatter_add(attn_weights, batch, dim=0)  # (B, 1)
        attn_weights = attn_weights / (attn_sum[batch] + 1e-8)

        return scatter_add(h * attn_weights, batch, dim=0)


class PhysicsAwareReadout(nn.Module):
    """
    Multi-strategy readout: attention + Set2Set pooling → concat.

    Output dimension = hidden_dim * 3.
    """

    def __init__(
        self,
        hidden_dim: int,
        set2set_steps: int = 3,
        *,
        channel_aware: bool = False,
        channel_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.attn_pool = AttentionPooling(hidden_dim)
        self.set2set = Set2Set(hidden_dim, processing_steps=set2set_steps)
        self.channel_aware = channel_aware
        self.channel_dim = (
            (hidden_dim // 2) if channel_dim is None else int(channel_dim)
        )
        if channel_aware:
            self.disp_pool = AttentionPooling(self.channel_dim)
            self.polar_pool = AttentionPooling(self.channel_dim)
        else:
            self.disp_pool = None
            self.polar_pool = None
        self.output_dim = hidden_dim * 3 + (
            2 * self.channel_dim if channel_aware else 0
        )

    def forward(
        self,
        h: Tensor,
        batch: Tensor,
        *,
        a_disp: Tensor | None = None,
        a_polar: Tensor | None = None,
        return_parts: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        h_attn = self.attn_pool(h, batch)       # (B, D)
        h_s2s = self.set2set(h, batch)          # (B, 2D)
        combined = torch.cat([h_attn, h_s2s], dim=-1)  # (B, 3D)

        g_disp: Tensor | None = None
        g_polar: Tensor | None = None
        if (
            self.channel_aware
            and a_disp is not None
            and a_polar is not None
            and self.disp_pool is not None
            and self.polar_pool is not None
        ):
            g_disp = self.disp_pool(a_disp, batch)
            g_polar = self.polar_pool(a_polar, batch)
            value = torch.cat([combined, g_disp, g_polar], dim=-1)
        else:
            value = combined
            if self.channel_aware:
                zeros = combined.new_zeros(combined.size(0), self.channel_dim)
                g_disp = zeros
                g_polar = zeros

        if return_parts:
            payload = {
                "value": value,
                "combined": combined,
                "disp": g_disp,
                "polar": g_polar,
            }
            return payload
        return value


# ================================================================== #
#  Physics Layers (0 learnable parameters)                            #
# ================================================================== #

class IdealSolubilityLayer(nn.Module):
    """
    Ideal solubility contribution Φ(T).

    Φ = (ΔH_fus / R) · (1/T - 1/T_m)
        - (ΔCp_fus / R) · [(T_m/T - 1) - ln(T_m/T)]

    x_ideal = exp(-Φ)
    """

    def __init__(self, R: float = 8.314) -> None:
        super().__init__()
        self.R = R

    def forward(
        self,
        T: Tensor,
        T_m: Tensor,
        dH_fus: Tensor,
        dCp_fus: Tensor,
    ) -> Tensor:
        """Return Φ (always ≥ 0 when T ≤ T_m)."""
        term1 = (dH_fus / self.R) * (1.0 / T - 1.0 / T_m)
        ratio = T_m / T
        psi = (ratio - 1.0) - torch.log(ratio + 1e-10)
        term2 = -(dCp_fus / self.R) * psi
        return term1 + term2


class NRTLLayer(nn.Module):
    """
    Non-Random Two-Liquid model for activity coefficients.

    Supported parameterizations:
      ref_invT : tau(T) = tau_ref + tau_inv · (T_ref / T - 1)
      abc      : tau(T) = a_ij + b_ij / T + c_ij · ln(T/T_ref)
      legacy   : converted internally to the ABC form

    G_ij = exp(-α_ij · τ_ij)
    ln γ_2, ln γ_1, ln γ_∞  — standard NRTL expressions.

    Zero learnable parameters.
    """

    def __init__(
        self,
        R: float = 8.314,
        T_ref: float = 298.15,
        tau_clamp: float = 30.0,
        eps: float = 1e-10,
    ) -> None:
        super().__init__()
        self.R = R
        self.T_ref = T_ref
        self.tau_clamp = tau_clamp
        self.eps = eps

    def compute_tau_G(
        self,
        a_12: Tensor,
        b_12: Tensor,
        c_12: Tensor,
        a_21: Tensor,
        b_21: Tensor,
        c_21: Tensor,
        alpha_12: Tensor,
        T: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute τ and G parameters from temperature coefficients."""
        log_term = torch.log(T / self.T_ref + self.eps)
        tau_12 = a_12 + b_12 / T + c_12 * log_term
        tau_21 = a_21 + b_21 / T + c_21 * log_term
        tau_12 = torch.clamp(tau_12, -self.tau_clamp, self.tau_clamp)
        tau_21 = torch.clamp(tau_21, -self.tau_clamp, self.tau_clamp)
        G_12 = torch.exp(-alpha_12 * tau_12)
        G_21 = torch.exp(-alpha_12 * tau_21)
        return tau_12, tau_21, G_12, G_21

    def compute_tau_G_ref_invT(
        self,
        tau_ref_12: Tensor,
        tau_inv_12: Tensor,
        tau_ref_21: Tensor,
        tau_inv_21: Tensor,
        alpha_12: Tensor,
        T: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute tau(T) from a compact reference-temperature parameterization."""
        inv_ratio = (self.T_ref / T) - 1.0
        tau_12 = tau_ref_12 + tau_inv_12 * inv_ratio
        tau_21 = tau_ref_21 + tau_inv_21 * inv_ratio
        tau_12 = torch.clamp(tau_12, -self.tau_clamp, self.tau_clamp)
        tau_21 = torch.clamp(tau_21, -self.tau_clamp, self.tau_clamp)
        G_12 = torch.exp(-alpha_12 * tau_12)
        G_21 = torch.exp(-alpha_12 * tau_21)
        return tau_12, tau_21, G_12, G_21

    def params_to_abc(
        self,
        nrtl_params: NRTLState,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Convert supported NRTL parameter layouts into the ABC solver form."""
        alpha_12 = nrtl_params["alpha_12"]

        if "tau_a12" in nrtl_params:
            return (
                nrtl_params["tau_a12"],
                nrtl_params["tau_b12"],
                nrtl_params["tau_c12"],
                nrtl_params["tau_a21"],
                nrtl_params["tau_b21"],
                nrtl_params["tau_c21"],
                alpha_12,
            )

        if "tau_ref_12" in nrtl_params:
            tau_ref_12 = nrtl_params["tau_ref_12"]
            tau_ref_21 = nrtl_params["tau_ref_21"]
            tau_inv_12 = nrtl_params["tau_inv_12"]
            tau_inv_21 = nrtl_params["tau_inv_21"]
            zeros = torch.zeros_like(tau_ref_12)
            return (
                tau_ref_12 - tau_inv_12,
                tau_inv_12 * self.T_ref,
                zeros,
                tau_ref_21 - tau_inv_21,
                tau_inv_21 * self.T_ref,
                zeros,
                alpha_12,
            )

        if "dg_12" in nrtl_params:
            a_T12 = nrtl_params["a_T12"]
            a_T21 = nrtl_params["a_T21"]
            zeros = torch.zeros_like(a_T12)
            return (
                -a_T12,
                nrtl_params["dg_12"] / self.R + a_T12 * self.T_ref,
                zeros,
                -a_T21,
                nrtl_params["dg_21"] / self.R + a_T21 * self.T_ref,
                zeros,
                alpha_12,
            )

        raise KeyError("Unsupported NRTL parameter layout")

    def compute_tau_G_from_params(
        self,
        nrtl_params: NRTLState,
        T: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute tau(T) and G(T) from any supported NRTL parameter layout."""
        if "ln_gamma_inf" in nrtl_params:
            tau_12 = torch.clamp(
                nrtl_params["ln_gamma_inf"], -self.tau_clamp, self.tau_clamp
            )
            tau_21 = torch.zeros_like(tau_12)
            G_12 = torch.ones_like(tau_12)
            G_21 = torch.ones_like(tau_12)
            return tau_12, tau_21, G_12, G_21

        if "ln_gamma_inf_ref" in nrtl_params:
            inv_ratio = (self.T_ref / T) - 1.0
            tau_12 = torch.clamp(
                nrtl_params["ln_gamma_inf_ref"]
                + nrtl_params["ln_gamma_inf_inv"] * inv_ratio,
                -self.tau_clamp,
                self.tau_clamp,
            )
            tau_21 = torch.zeros_like(tau_12)
            G_12 = torch.ones_like(tau_12)
            G_21 = torch.ones_like(tau_12)
            return tau_12, tau_21, G_12, G_21

        if "tau_12" in nrtl_params and "tau_21" in nrtl_params:
            tau_12 = torch.clamp(
                nrtl_params["tau_12"], -self.tau_clamp, self.tau_clamp
            )
            tau_21 = torch.clamp(
                nrtl_params["tau_21"], -self.tau_clamp, self.tau_clamp
            )
            alpha_12 = nrtl_params["alpha_12"]
            G_12 = torch.exp(-alpha_12 * tau_12)
            G_21 = torch.exp(-alpha_12 * tau_21)
            return tau_12, tau_21, G_12, G_21

        if "tau_ref_12" in nrtl_params:
            return self.compute_tau_G_ref_invT(
                nrtl_params["tau_ref_12"],
                nrtl_params["tau_inv_12"],
                nrtl_params["tau_ref_21"],
                nrtl_params["tau_inv_21"],
                nrtl_params["alpha_12"],
                T,
            )

        a_12, b_12, c_12, a_21, b_21, c_21, alpha_12 = self.params_to_abc(
            nrtl_params
        )
        return self.compute_tau_G(
            a_12, b_12, c_12, a_21, b_21, c_21, alpha_12, T
        )

    def ln_gamma_1(
        self,
        x1: Tensor,
        x2: Tensor,
        tau_12: Tensor,
        tau_21: Tensor,
        G_12: Tensor,
        G_21: Tensor,
    ) -> Tensor:
        """ln γ_1 (solvent activity coefficient)."""
        A = x2 + x1 * G_12 + self.eps
        B = x1 + x2 * G_21 + self.eps
        term1 = tau_21 * (G_21 / B) ** 2
        term2 = tau_12 * G_12 / A ** 2
        return x2 ** 2 * (term1 + term2)

    def ln_gamma_2(
        self,
        x1: Tensor,
        x2: Tensor,
        tau_12: Tensor,
        tau_21: Tensor,
        G_12: Tensor,
        G_21: Tensor,
    ) -> Tensor:
        """ln γ_2 (solute activity coefficient)."""
        A = x2 + x1 * G_12 + self.eps
        B = x1 + x2 * G_21 + self.eps
        term1 = tau_12 * (G_12 / A) ** 2
        term2 = tau_21 * G_21 / B ** 2
        return x1 ** 2 * (term1 + term2)

    def ln_gamma_inf(self, tau_12: Tensor, tau_21: Tensor, G_21: Tensor) -> Tensor:
        """ln γ_2^∞ (infinite dilution: x_2 → 0)."""
        return tau_12 + tau_21 * G_21

    def forward(
        self,
        x2: Tensor,
        a_12: Tensor,
        b_12: Tensor,
        c_12: Tensor,
        a_21: Tensor,
        b_21: Tensor,
        c_21: Tensor,
        alpha_12: Tensor,
        T: Tensor,
    ) -> NRTLState:
        """Full NRTL forward: returns ln γ_2 and intermediate quantities."""
        x1 = 1.0 - x2
        tau_12, tau_21, G_12, G_21 = self.compute_tau_G(
            a_12, b_12, c_12, a_21, b_21, c_21, alpha_12, T
        )
        lng2 = self.ln_gamma_2(x1, x2, tau_12, tau_21, G_12, G_21)
        return {
            "ln_gamma_2": lng2,
            "tau_12": tau_12,
            "tau_21": tau_21,
            "G_12": G_12,
            "G_21": G_21,
        }


def _segment_matvec(p: Tensor, E: Tensor) -> Tensor:
    """``sum_n E[m,n] * p[...,n]`` for ``p`` of shape (B,G) or (B,K,G), ``E`` of (B,G,G).

    The (B,K,G) form solves K segment profiles that SHARE one exchange matrix per row in a
    single kernel. That is what makes the mixture and pure-solute solves of
    ``_residual_ln_gamma2`` one launch instead of two: the segment fixed point is independent
    per row, so stacking is exact rather than an approximation. ``E`` is symmetric by
    construction here, but the transpose is written out instead of relied upon.

    Why it matters: a COSMO-SAC forward is launch-bound, not flop-bound. Measured on a Kaggle
    T4 (``scripts/kaggle/make_bottleneck_notebook.py``, 2026-10-02) the loader delivers
    648-714 rows/s while the compute ceiling is 193 rows/s at 331 ms/step for batch 64 -- with
    the GPU only 25% busy (p90 34%). The work per kernel is tiny (64x51x51); the cost is
    issuing ~1.8k of them per step from Python.
    """
    if p.dim() == 3:
        return torch.matmul(p, E.transpose(-1, -2))
    return torch.bmm(E, p.unsqueeze(-1)).squeeze(-1)


class CosmoSacLayer(nn.Module):
    """
    Differentiable COSMO-SAC activity-coefficient model (Lin & Sandler 2002 /
    VT-2005 conventions).

    The molecule is represented by a sigma-profile ``p(sigma)`` — area (Å²) per
    screening-charge-density bin, summing to the cavity surface area ``A``. Given
    solute & solvent profiles the residual (restoring) activity coefficient comes
    from a segment-activity-coefficient fixed point Gamma(sigma) — the same kind
    of damped fixed point as the SLE solver, fully differentiable. An optional
    Staverman–Guggenheim combinatorial term uses molecular volume/area.

    Zero learnable parameters. All math runs in float32.
    """

    def __init__(self, cfg=None) -> None:
        super().__init__()
        g = (lambda name, default: float(getattr(cfg, name, default)) if cfg is not None else default)
        gi = (lambda name, default: int(getattr(cfg, name, default)) if cfg is not None else default)
        gb = (lambda name, default: bool(getattr(cfg, name, default)) if cfg is not None else default)
        n_bins = gi("cosmo_sac_n_bins", 51)
        sigma_min = g("cosmo_sac_sigma_min", -0.025)
        sigma_max = g("cosmo_sac_sigma_max", 0.025)
        self.a_eff = g("cosmo_sac_a_eff", 7.5)            # Å² effective segment area
        self.alpha_prime = g("cosmo_sac_alpha_prime", 16466.72)  # kcal·Å⁴/mol/e²
        self.c_hb = g("cosmo_sac_c_hb", 85580.0)
        self.sigma_hb = g("cosmo_sac_sigma_hb", 0.0084)
        self.R_kcal = g("cosmo_sac_R_kcal", 1.987204e-3)  # kcal/mol/K
        self.coord_z = g("cosmo_sac_coord_z", 10.0)
        self.q0 = g("cosmo_sac_q0", 79.53)                # Å² SG area normalizer
        self.r0 = g("cosmo_sac_r0", 66.69)                # Å³ SG volume normalizer
        self.use_combinatorial = gb("cosmo_sac_use_combinatorial", True)
        self.n_iter_train = gi("cosmo_sac_gamma_iter_train", 8)
        self.n_iter_eval = gi("cosmo_sac_gamma_iter_eval", 30)
        self.damping = g("damping", 0.7)
        self.exp_clamp = g("tau_clamp", 30.0)
        self.eps = g("eps", 1e-10)
        self.track_convergence = gb("cosmo_sac_track_convergence", False)
        #: Заполняется только при track_convergence; см. _segment_ln_gamma.
        self.last_convergence: dict[str, float] | None = None

        sigma_grid = torch.linspace(sigma_min, sigma_max, n_bins)
        self.register_buffer("sigma_grid", sigma_grid, persistent=False)
        # Exchange-energy matrix delta_w(sigma_m, sigma_n) depends only on the
        # fixed grid -> precompute once (kcal/mol).
        sm = sigma_grid.view(-1, 1)
        sn = sigma_grid.view(1, -1)
        misfit = 0.5 * self.alpha_prime * (sm + sn) ** 2
        sigma_acc = torch.maximum(sm, sn)
        sigma_don = torch.minimum(sm, sn)
        hb = self.c_hb * torch.clamp(sigma_acc - self.sigma_hb, min=0.0) * torch.clamp(
            sigma_don + self.sigma_hb, max=0.0
        )
        self.register_buffer("delta_w_base", misfit + hb, persistent=False)
        self.n_bins = n_bins
        # --- Arm C (Tier-3 closure-fix): learnable low-rank symmetric residual on the
        # exchange-energy kernel, Δ = B·diag(a)·Bᵀ over the fixed σ-grid. Zero-init ⇒ Δ=0
        # ⇒ bit-identical to exact COSMO-SAC at R=0 (nested). Sees only the σ-grid (never x),
        # so it corrects the MAP (B_clos), never injects intermediate information (B_insuff).
        R = gi("cosmo_sac_kernel_residual_rank", 0)
        self.kernel_rank = R
        self.kernel_smooth_penalty = g("cosmo_sac_kernel_residual_penalty", 0.0)
        if R > 0:
            # kernel_B needs a small non-zero init: the residual (B*a)@B^T is quadratic in B, so at
            # B=0 the gradient w.r.t. B vanishes and the residual would never co-train (dead init).
            # kernel_a stays 0 so the residual is exactly 0 at init (nested with exact COSMO-SAC),
            # yet d(residual)/d(a) = B_r B_r^T != 0 bootstraps training. (run_local_closure_fix.py
            # re-inits kernel_B the same way before its post-hoc fit.)
            self.kernel_B = nn.Parameter(0.01 * torch.randn(n_bins, R))  # 51·R params (persistent)
            self.kernel_a = nn.Parameter(torch.zeros(R))                 #    R params
            D2 = torch.zeros(n_bins - 2, n_bins)                   # 2nd-difference smoothness op
            idx = torch.arange(n_bins - 2)
            D2[idx, idx], D2[idx, idx + 1], D2[idx, idx + 2] = 1.0, -2.0, 1.0
            self.register_buffer("_kernel_D2", D2, persistent=False)
        else:
            self.kernel_B = None
            self.kernel_a = None

    def _segment_ln_gamma(self, p_norm: Tensor, E: Tensor, n_iter: int) -> Tensor:
        """Solve the segment activity-coefficient fixed point; return ln Gamma (B,51).

        Gamma(sigma_m) = 1 / sum_n p_norm(sigma_n) Gamma(sigma_n) exp(-dW/RT).
        ``p_norm`` (B,51) -- or (B,K,51) for K profiles sharing one E -- sums to 1 over the
        last axis; ``E`` (B,51,51) = exp(-delta_w/RT). Shape in, same shape out.
        """
        gamma = torch.ones_like(p_norm)
        for _ in range(n_iter):
            # sum_n E[m,n] * p_norm[n] * gamma[n]
            denom = _segment_matvec(p_norm * gamma, E)
            gamma_new = 1.0 / (denom + self.eps)
            gamma = self.damping * gamma_new + (1.0 - self.damping) * gamma
            gamma = gamma.clamp(1e-8, 1e8)
        if self.track_convergence:
            self.last_convergence = self._fixed_point_residual(p_norm, E, gamma, n_iter)
        return torch.log(gamma + self.eps)

    @staticmethod
    def _weighted(x: Tensor, w: Tensor) -> Tensor:
        return (w * x).sum(-1) / w.sum(-1).clamp_min(1e-30)

    def _fixed_point_residual(self, p_norm: Tensor, E: Tensor, gamma: Tensor,
                              n_iter: int) -> dict[str, float]:
        """Невязка неподвижной точки, ВЗВЕШЕННАЯ ПО МАССЕ профиля.

        В точке решения Gamma_m * sum_n p_n Gamma_n E_mn = 1 для каждого бина. Брать максимум
        этого выражения ПО ВСЕМ бинам бессмысленно и даёт ~1.0 даже после 20000 итераций: в
        пустых бинах Gamma вырождена (делится на ~eps, упирается в clamp 1e8), а в итоговую
        свёртку она входит с весом p и потому на результат не влияет. Диагностика 2026-09-30
        сначала мерила именно так и читалась как «не сходится» -- ошибка была в метрике, а не в
        решателе. Поэтому невязка взвешивается тем же весом, с каким бин входит в ответ.
        """
        with torch.no_grad():
            den = _segment_matvec(p_norm * gamma, E)
            r = (gamma * den - 1.0).abs()
            per_row = self._weighted(r, p_norm)
        return {"n_iter": int(n_iter), "residual_mean": float(per_row.mean()),
                "residual_max": float(per_row.max())}

    def _effective_delta_w(self) -> Tensor:
        """Exchange-energy kernel with the optional Arm-C low-rank residual folded in."""
        if self.kernel_rank and self.kernel_rank > 0:
            Ba = self.kernel_B * self.kernel_a.unsqueeze(0)      # (n_bins, R)
            return self.delta_w_base + Ba @ self.kernel_B.t()    # symmetric by construction
        return self.delta_w_base

    def kernel_smoothness_loss(self) -> Tensor:
        """Optional 2nd-difference smoothness penalty on the Arm-C residual columns."""
        if self.kernel_rank and self.kernel_rank > 0 and self.kernel_smooth_penalty > 0.0:
            return self.kernel_smooth_penalty * (self._kernel_D2 @ self.kernel_B).pow(2).mean()
        return self.delta_w_base.new_zeros(())

    def _E_matrix(self, T: Tensor) -> Tensor:
        """Boltzmann matrix exp(-delta_w_eff/(R T)) per sample. Returns (B,51,51)."""
        rt = (self.R_kcal * T).clamp_min(self.eps).view(-1, 1, 1)
        dw = self._effective_delta_w()
        arg = (-dw.unsqueeze(0) / rt).clamp(-self.exp_clamp, self.exp_clamp)
        return torch.exp(arg)

    def _residual_ln_gamma2(
        self,
        p2: Tensor,
        p1: Tensor,
        A2: Tensor,
        A1: Tensor,
        x2: Tensor,
        T: Tensor,
        *,
        n_iter: int,
    ) -> Tensor:
        """Restoring ln gamma2 of solute (2) in solvent (1) at mole fraction x2."""
        x1 = 1.0 - x2
        E = self._E_matrix(T)
        # Mixture area-fraction profile (sums to 1 over bins).
        A_mix = (x2 * A2 + x1 * A1).clamp_min(self.eps)
        p_mix = (x2.unsqueeze(-1) * p2 + x1.unsqueeze(-1) * p1) / A_mix.unsqueeze(-1)
        p2_pure = p2 / A2.clamp_min(self.eps).unsqueeze(-1)
        # One stacked solve, not two sequential ones: both profiles share E and the
        # fixed point is row-independent, so this is exact and halves the inner loop's
        # kernel launches -- see _segment_matvec for why launches are the binding cost.
        ln_gamma_both = self._segment_ln_gamma(
            torch.stack((p_mix, p2_pure), dim=1), E, n_iter
        )
        ln_gamma_mix, ln_gamma_2pure = ln_gamma_both[:, 0], ln_gamma_both[:, 1]
        # ln gamma2_res = (A2/a_eff) * sum_m p2_pure(m) (ln Gamma_mix - ln Gamma_2pure)
        contrib = p2_pure * (ln_gamma_mix - ln_gamma_2pure)
        return (A2 / self.a_eff) * contrib.sum(dim=-1)

    def _combinatorial_ln_gamma2(
        self, A2: Tensor, A1: Tensor, V2: Tensor, V1: Tensor, x2: Tensor
    ) -> Tensor:
        """Staverman–Guggenheim combinatorial ln gamma2 (component 2)."""
        x1 = 1.0 - x2
        r2 = V2 / self.r0
        r1 = V1 / self.r0
        q2 = A2 / self.q0
        q1 = A1 / self.q0
        r_bar = (x1 * r1 + x2 * r2).clamp_min(self.eps)
        q_bar = (x1 * q1 + x2 * q2).clamp_min(self.eps)
        z = self.coord_z
        l1 = (z / 2.0) * (r1 - q1) - (r1 - 1.0)
        l2 = (z / 2.0) * (r2 - q2) - (r2 - 1.0)
        # x2-cancelled ratios so the term stays finite at infinite dilution (x2->0):
        #   phi2/x2 = r2/r_bar ;  theta2/phi2 = (q2 r_bar)/(r2 q_bar)
        phi2_over_x2 = r2 / r_bar
        theta2_over_phi2 = (q2 * r_bar) / (r2 * q_bar).clamp_min(self.eps)
        ln_gamma_c = (
            torch.log(phi2_over_x2.clamp_min(self.eps))
            + (z / 2.0) * q2 * torch.log(theta2_over_phi2.clamp_min(self.eps))
            + l2
            - phi2_over_x2 * (x1 * l1 + x2 * l2)
        )
        return ln_gamma_c

    def ln_gamma_2(
        self,
        x1: Tensor,
        x2: Tensor,
        p2: Tensor,
        p1: Tensor,
        A2: Tensor,
        A1: Tensor,
        V2: Tensor | None,
        V1: Tensor | None,
        T: Tensor,
    ) -> Tensor:
        """Total ln gamma2 (residual + optional combinatorial) at composition x2."""
        n_iter = self.n_iter_train if self.training else self.n_iter_eval
        lng_res = self._residual_ln_gamma2(p2, p1, A2, A1, x2, T, n_iter=n_iter)
        if self.use_combinatorial and V2 is not None and V1 is not None:
            lng_res = lng_res + self._combinatorial_ln_gamma2(A2, A1, V2, V1, x2)
        # Safety rail: physical ln γ is O(±10); the residual prefactor A2/a_eff can
        # be ~30 and each segment ln Γ is bounded only to ±18, so a strongly
        # mismatched profile can produce ±hundreds. Bound to ±50 so it cannot
        # inject pathological magnitudes into the SLE exponent or the activity loss
        # (defense-in-depth behind the solver's exp-argument clamp).
        return lng_res.clamp(-50.0, 50.0)

    def ln_gamma_inf(
        self,
        p2: Tensor,
        p1: Tensor,
        A2: Tensor,
        A1: Tensor,
        V2: Tensor | None,
        V1: Tensor | None,
        T: Tensor,
    ) -> Tensor:
        """Infinite-dilution ln gamma2 (x2 -> 0): mixture profile == pure solvent."""
        x2 = torch.zeros_like(A2)
        return self.ln_gamma_2(1.0 - x2, x2, p2, p1, A2, A1, V2, V1, T)


class CosmoSac2010Layer(nn.Module):
    """Differentiable COSMO-SAC-2010/dsp (Hsieh, Lin & Sandler, *Ind. Eng. Chem.
    Res.* 2010; the NIST ``COSMO3`` model).

    Extends the 2002 (VT-2005) layer with the three mechanisms that define the
    modern closure, matched bit-for-bit to the NIST reference C++ (usnistgov/COSMOSAC,
    ``get_DELTAW``/``get_lngamma_*``); validated by ``validate_cosmo2010_layer.py``:

    1. **Three typed sigma-profiles** (NHB / OH / OT), concatenated to a 153-grid;
       the segment-activity fixed point runs on the full 153-grid.
    2. **Temperature-dependent electrostatic misfit** ``c_ES(T)=A_ES+B_ES/T^2`` and a
       **donor/acceptor-typed hydrogen-bond kernel** ``c_hb(type_m,type_n)`` (the
       single-parameter 2002 kernel split into OH-OH / OH-OT / OT-OT), applied only
       where ``sigma_m*sigma_n<0``:
       ``DELTAW = c_ES(sigma_m+sigma_n)^2 - c_hb*(sigma_m-sigma_n)^2``.
    3. **Dispersive free-energy term** ``A = w*(0.5(e0+e1)-sqrt(e0*e1))`` from
       per-molecule ``eps/kB`` (the ``dsp`` in ``2010/dsp``); at infinite dilution the
       solute contribution is ``A``.

    Combinatorial term is the same Staverman-Guggenheim as the 2002 layer. Zero
    learnable parameters (float32). Constants are the exact NIST COSMO3 defaults.
    """

    N_TYPE = 3   # NHB, OH, OT

    def __init__(self, cfg=None) -> None:
        super().__init__()
        g = (lambda name, default: float(getattr(cfg, name, default)) if cfg is not None else default)
        gi = (lambda name, default: int(getattr(cfg, name, default)) if cfg is not None else default)
        gb = (lambda name, default: bool(getattr(cfg, name, default)) if cfg is not None else default)
        n_bins = gi("cosmo_sac_n_bins", 51)
        sigma_min = g("cosmo_sac_sigma_min", -0.025)
        sigma_max = g("cosmo_sac_sigma_max", 0.025)
        # --- exact NIST COSMO3 constants ---
        self.A_ES = g("cosmo3_A_ES", 6525.69)
        self.B_ES = g("cosmo3_B_ES", 1.4859e8)
        self.c_OH_OH = g("cosmo3_c_OH_OH", 4013.78)
        self.c_OT_OT = g("cosmo3_c_OT_OT", 932.31)
        self.c_OH_OT = g("cosmo3_c_OH_OT", 3016.43)
        self.AEFFPRIME = g("cosmo3_AEFFPRIME", 7.25)
        self.R_kcal = g("cosmo_sac_R_kcal", 1.987204e-3)
        self.disp_w = g("cosmo3_disp_w", 0.27027)   # magnitude; sign per class-pair (passed in)
        # SG combinatorial (same as 2002)
        self.coord_z = g("cosmo_sac_coord_z", 10.0)
        self.q0 = g("cosmo_sac_q0", 79.53)
        self.r0 = g("cosmo_sac_r0", 66.69)
        self.use_combinatorial = gb("cosmo_sac_use_combinatorial", True)
        self.use_dispersion = gb("cosmo3_use_dispersion", True)
        self.n_iter_train = gi("cosmo_sac_gamma_iter_train", 8)
        self.n_iter_eval = gi("cosmo_sac_gamma_iter_eval", 200)
        self.damping = g("damping", 0.5)
        self.exp_clamp = g("tau_clamp", 30.0)
        self.eps = g("eps", 1e-10)

        # The grid is built the way the NIST reference builds it -- sigma_min + delta*m in
        # float64 -- and NOT with torch.linspace, because the hydrogen-bond mask below tests
        # sigma_m*sigma_n against zero and linspace does not put the centre bin ON zero.
        # linspace gives sigma_25 = -8.149e-10 (float32) / +8.674e-19 (float64); NIST's
        # `-0.025 + (2*0.025/50)*m` gives exactly 0.0.  With a non-zero centre bin, sigma_25*sigma_n
        # and sigma_25*sigma_(50-n) carry OPPOSITE signs, so exactly one of every mirrored pair
        # passes the strict test: 25 of 51 entries per block-pair, 75 of 153 on the centre row,
        # each awarding a full c_hb*(sigma_m-sigma_n)^2 to a segment that carries no screening
        # charge and cannot hydrogen-bond.  NIST has the same strict comparison
        # (`c_hb = (sigma_m*sigma_n >= 0) ? 0 : cc`, COSMO.hpp:341) and is spared only by its
        # exact zero, so this was a DIVERGENCE from the reference, not a shared convention.
        # The float64 form below is bit-identical to NIST's on the default grid.
        # WHAT IT DOES NOT DO, measured rather than assumed: it does not move the NIST comparison
        # at all.  validate_cosmo2010_layer.py over 300 UD pairs returns RMSE 1.9e-4 / max 1.8e-3
        # both before and after, to every printed digit and the same worst pairs, so the residual
        # discrepancy against cCOSMO has some other cause.  The reason the fix is invisible there
        # is that on all 80 UD compounds the OH and OT blocks carry EXACTLY zero area in the centre
        # bin (NHB carries a median 7.7%, and c_hb(NHB, anything) = 0) -- a tabulated profile puts
        # a zero-charge segment in the non-hydrogen-bonding type by construction.
        # NOTHING IN THE TREE MOVES, checked and not assumed: the four scripts that use this layer
        # (run_b_insuff_convention_audit, run_b_insuff_representative_audit,
        # run_fidelity_lever_fair2002, run_fidelity_lever_inhouse) all feed TABULATED typed
        # profiles from UD sigma3/, and no script anywhere pairs this layer with a checkpoint, so
        # the learned-sigma -> 2010 path this file's validator docstring mentions does not exist.
        # WHY IT IS KEPT ANYWAY: the guarantee that spares the reference is a property of the
        # TABLE, not of the closure -- a predicted typed profile with any OH/OT mass at sigma = 0
        # would fire the artefact, silently, and this repository predicts sigma-profiles for a
        # living.  The fix also makes this kernel's charge-conjugation invariance exact
        # (max|dw - J dw J| goes 2.5086 -> 0.0000e+00 in both precisions), which the gauge audit
        # in run_charge_conjugation_gauge.py reads.
        # NOT applied to CosmoSacLayer (2002): its kernel thresholds at +-sigma_hb = 0.0084, so a
        # centre bin off by 1e-10 crosses nothing there, and changing it would perturb every
        # deposited 2002 number for no gain.
        _delta = (sigma_max - sigma_min) / (n_bins - 1)
        sigma_grid = (sigma_min + _delta * torch.arange(n_bins, dtype=torch.float64)).to(
            torch.get_default_dtype())                                  # (51,)
        sigma_cat = sigma_grid.repeat(self.N_TYPE)                      # (153,) NHB|OH|OT
        self.n_bins = n_bins
        self.n_grid = self.N_TYPE * n_bins
        sm = sigma_cat.view(-1, 1)
        sn = sigma_cat.view(1, -1)
        # T-independent pieces of DELTAW (assembled per-T in _delta_w)
        self.register_buffer("sumsq", (sm + sn) ** 2, persistent=False)          # electrostatic
        self.register_buffer("diffsq", (sm - sn) ** 2, persistent=False)         # HB shape
        self.register_buffer("sign_neg", (sm * sn < 0).to(sigma_cat.dtype), persistent=False)
        # typed HB coefficient per 51x51 block: c_hb(type_m, type_n)
        c_hb = {(1, 1): self.c_OH_OH, (2, 2): self.c_OT_OT, (1, 2): self.c_OH_OT, (2, 1): self.c_OH_OT}
        chb = torch.zeros(self.n_grid, self.n_grid)
        for i in range(self.N_TYPE):        # 0=NHB, 1=OH, 2=OT
            for j in range(self.N_TYPE):
                val = c_hb.get((i, j), 0.0)  # NHB with anything -> 0
                chb[i * n_bins:(i + 1) * n_bins, j * n_bins:(j + 1) * n_bins] = val
        self.register_buffer("chb", chb, persistent=False)

    def _delta_w(self, T: Tensor) -> Tensor:
        """Exchange-energy kernel (B, 153, 153) with T-dependent electrostatic misfit."""
        c_ES = (self.A_ES + self.B_ES / (T * T)).view(-1, 1, 1)            # (B,1,1)
        hb = (self.chb * self.diffsq * self.sign_neg).unsqueeze(0)          # (1,153,153)
        return c_ES * self.sumsq.unsqueeze(0) - hb

    def _segment_ln_gamma(self, p_norm: Tensor, E: Tensor, n_iter: int) -> Tensor:
        """Segment fixed point on the 153-grid; return ln Gamma.

        ``p_norm`` is (B,153), or (B,K,153) for K profiles sharing one E; same shape out.
        """
        gamma = torch.ones_like(p_norm)
        for _ in range(n_iter):
            denom = _segment_matvec(p_norm * gamma, E)
            gamma_new = 1.0 / (denom + self.eps)
            gamma = self.damping * gamma_new + (1.0 - self.damping) * gamma
            gamma = gamma.clamp(1e-8, 1e8)
        return torch.log(gamma + self.eps)

    def _residual_ln_gamma2(self, p2, p1, A2, A1, x2, T, *, n_iter):
        """Restoring ln gamma2 of solute (2) in solvent (1); p2,p1 are (B,153) area/bin."""
        x1 = 1.0 - x2
        rt = (self.R_kcal * T).clamp_min(self.eps).view(-1, 1, 1)
        E = torch.exp((-self._delta_w(T) / rt).clamp(-self.exp_clamp, self.exp_clamp))
        A_mix = (x2 * A2 + x1 * A1).clamp_min(self.eps)
        p_mix = (x2.unsqueeze(-1) * p2 + x1.unsqueeze(-1) * p1) / A_mix.unsqueeze(-1)
        p2_pure = p2 / A2.clamp_min(self.eps).unsqueeze(-1)
        # One stacked solve, not two sequential ones: both profiles share E and the
        # fixed point is row-independent, so this is exact and halves the inner loop's
        # kernel launches -- see _segment_matvec for why launches are the binding cost.
        ln_gamma_both = self._segment_ln_gamma(
            torch.stack((p_mix, p2_pure), dim=1), E, n_iter
        )
        ln_gamma_mix, ln_gamma_2pure = ln_gamma_both[:, 0], ln_gamma_both[:, 1]
        contrib = p2_pure * (ln_gamma_mix - ln_gamma_2pure)
        return (A2 / self.AEFFPRIME) * contrib.sum(dim=-1)

    def _combinatorial_ln_gamma2(self, A2, A1, V2, V1, x2):
        """Staverman-Guggenheim combinatorial ln gamma2 (identical to the 2002 layer)."""
        x1 = 1.0 - x2
        r2, r1 = V2 / self.r0, V1 / self.r0
        q2, q1 = A2 / self.q0, A1 / self.q0
        r_bar = (x1 * r1 + x2 * r2).clamp_min(self.eps)
        q_bar = (x1 * q1 + x2 * q2).clamp_min(self.eps)
        z = self.coord_z
        l1 = (z / 2.0) * (r1 - q1) - (r1 - 1.0)
        l2 = (z / 2.0) * (r2 - q2) - (r2 - 1.0)
        phi2_over_x2 = r2 / r_bar
        theta2_over_phi2 = (q2 * r_bar) / (r2 * q_bar).clamp_min(self.eps)
        return (
            torch.log(phi2_over_x2.clamp_min(self.eps))
            + (z / 2.0) * q2 * torch.log(theta2_over_phi2.clamp_min(self.eps))
            + l2
            - phi2_over_x2 * (x1 * l1 + x2 * l2)
        )

    def _dispersion_inf(self, eps2, eps1, w):
        """Dispersive ln gamma2 at infinite dilution: A = w*(0.5(e0+e1)-sqrt(e0 e1))."""
        return w * (0.5 * (eps2 + eps1) - torch.sqrt((eps2 * eps1).clamp_min(0.0)))

    def ln_gamma_inf(
        self, p2, p1, A2, A1, V2, V1, T,
        eps2: Tensor | None = None, eps1: Tensor | None = None, disp_w: Tensor | None = None,
    ) -> Tensor:
        """Infinite-dilution ln gamma2 (residual + optional combinatorial + dispersion).

        ``p2``/``p1`` are (B, 153) area-per-bin profiles (NHB|OH|OT). ``eps2``/``eps1``
        are per-molecule dispersion eps/kB; ``disp_w`` is the (signed) class-pair
        coefficient (default +0.27027)."""
        x2 = torch.zeros_like(A2)
        n_iter = self.n_iter_train if self.training else self.n_iter_eval
        out = self._residual_ln_gamma2(p2, p1, A2, A1, x2, T, n_iter=n_iter)
        if self.use_combinatorial and V2 is not None and V1 is not None:
            out = out + self._combinatorial_ln_gamma2(A2, A1, V2, V1, x2)
        if self.use_dispersion and eps2 is not None and eps1 is not None:
            w = disp_w if disp_w is not None else out.new_full(out.shape, self.disp_w)
            out = out + self._dispersion_inf(eps2, eps1, w)
        return out.clamp(-50.0, 50.0)


class HansenDistanceLayer(nn.Module):
    """
    Hansen distance Ra between two molecules.

    Ra² = 4·(δd1 - δd2)² + (δp1 - δp2)² + (δh1 - δh2)²

    Zero learnable parameters.
    """

    def forward(
        self, hansen_1: Tensor, hansen_2: Tensor
    ) -> Tensor:
        dd = hansen_1[:, 0] - hansen_2[:, 0]
        dp = hansen_1[:, 1] - hansen_2[:, 1]
        dh = hansen_1[:, 2] - hansen_2[:, 2]
        Ra_sq = 4.0 * dd ** 2 + dp ** 2 + dh ** 2
        return torch.sqrt(Ra_sq + 1e-10)
