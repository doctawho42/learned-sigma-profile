"""On-the-fly structural positional encodings for molecular graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import Tensor


PositionalEncodingKind = Literal["laplacian", "rwse"]


def _ensure_batch(batch: Optional[Tensor], num_nodes: int, device: torch.device) -> Tensor:
    """Return a valid per-node graph assignment vector."""
    if batch is None:
        return torch.zeros(num_nodes, dtype=torch.long, device=device)
    return batch


def _dense_adjacency_from_local_edges(
    local_edge_index: Tensor,
    num_nodes: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Build a dense unweighted adjacency matrix for one graph."""
    adj = torch.zeros((num_nodes, num_nodes), device=device, dtype=dtype)
    if local_edge_index.numel() == 0 or num_nodes == 0:
        return adj
    row, col = local_edge_index
    adj[row, col] = 1.0
    adj[col, row] = 1.0
    adj.fill_diagonal_(0.0)
    return adj


def _laplacian_positional_encoding(
    adj: Tensor,
    pe_dim: int,
) -> Tensor:
    """Return absolute non-trivial Laplacian eigenvectors padded to `pe_dim`."""
    num_nodes = adj.size(0)
    if num_nodes == 0 or pe_dim <= 0:
        return adj.new_zeros((num_nodes, pe_dim))
    if num_nodes == 1:
        return adj.new_zeros((1, pe_dim))

    degree = adj.sum(dim=-1)
    laplacian = torch.diag(degree) - adj

    eig_device = laplacian.device
    if eig_device.type == "mps":
        eigvals, eigvecs = torch.linalg.eigh(laplacian.cpu())
        eigvecs = eigvecs.to(device=eig_device, dtype=adj.dtype)
    else:
        _, eigvecs = torch.linalg.eigh(laplacian)

    start = 1
    stop = min(num_nodes, pe_dim + 1)
    selected = eigvecs[:, start:stop].abs()
    if selected.size(1) < pe_dim:
        selected = torch.nn.functional.pad(selected, (0, pe_dim - selected.size(1)))
    return selected


def _rwse_positional_encoding(
    adj: Tensor,
    pe_dim: int,
) -> Tensor:
    """Return diagonal random-walk return probabilities for steps `1..pe_dim`."""
    num_nodes = adj.size(0)
    if num_nodes == 0 or pe_dim <= 0:
        return adj.new_zeros((num_nodes, pe_dim))

    degree = adj.sum(dim=-1)
    inv_degree = torch.where(
        degree > 0,
        degree.reciprocal(),
        torch.zeros_like(degree),
    )
    transition = inv_degree[:, None] * adj

    powers = []
    current = transition
    eye = torch.eye(num_nodes, device=adj.device, dtype=adj.dtype)
    for _ in range(pe_dim):
        powers.append(torch.diagonal(current, dim1=-2, dim2=-1))
        current = current @ transition if num_nodes > 1 else eye

    return torch.stack(powers, dim=-1)


@dataclass(frozen=True)
class _GraphSlice:
    node_indices: Tensor
    edge_index: Tensor


def _iter_graph_slices(
    edge_index: Tensor,
    batch: Tensor,
    num_nodes: int,
) -> list[_GraphSlice]:
    """Collect per-graph local node ids and local edge indices."""
    slices: list[_GraphSlice] = []
    num_graphs = int(batch.max().item()) + 1 if num_nodes > 0 else 0
    for graph_id in range(num_graphs):
        node_mask = batch == graph_id
        node_indices = node_mask.nonzero(as_tuple=False).view(-1)
        node_count = int(node_indices.numel())
        if node_count == 0:
            continue
        local_index = torch.full(
            (num_nodes,),
            -1,
            dtype=torch.long,
            device=batch.device,
        )
        local_index[node_indices] = torch.arange(
            node_count,
            dtype=torch.long,
            device=batch.device,
        )
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        graph_edges = edge_index[:, edge_mask]
        if graph_edges.numel() > 0:
            graph_edges = local_index[graph_edges]
        slices.append(_GraphSlice(node_indices=node_indices, edge_index=graph_edges))
    return slices


class PositionalEncoding(torch.nn.Module):
    """Batch-aware graph positional encodings computed from the current graph."""

    def __init__(
        self,
        pe_dim: int,
        kind: PositionalEncodingKind = "laplacian",
    ) -> None:
        super().__init__()
        if pe_dim <= 0:
            raise ValueError("pe_dim must be positive")
        if kind not in {"laplacian", "rwse"}:
            raise ValueError(f"Unsupported positional encoding kind: {kind}")
        self.pe_dim = int(pe_dim)
        self.kind = kind

    def forward(
        self,
        edge_index: Tensor,
        num_nodes: int,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """Return per-node positional features with shape `(N, pe_dim)`."""
        if num_nodes == 0:
            device = edge_index.device
            return torch.zeros((0, self.pe_dim), dtype=torch.float32, device=device)

        batch = _ensure_batch(batch, num_nodes, edge_index.device)
        pe = torch.zeros(
            (num_nodes, self.pe_dim),
            dtype=torch.float32,
            device=edge_index.device,
        )
        graph_slices = _iter_graph_slices(edge_index, batch, num_nodes)
        for graph_slice in graph_slices:
            node_count = int(graph_slice.node_indices.numel())
            adj = _dense_adjacency_from_local_edges(
                graph_slice.edge_index,
                node_count,
                device=edge_index.device,
                dtype=pe.dtype,
            )
            if self.kind == "laplacian":
                graph_pe = _laplacian_positional_encoding(adj, self.pe_dim)
            else:
                graph_pe = _rwse_positional_encoding(adj, self.pe_dim)
            pe[graph_slice.node_indices] = graph_pe
        return pe
