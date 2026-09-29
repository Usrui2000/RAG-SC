"""Sparse refinement-token transport with explicit position side information."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class SparseTokenPacket:
    """Sparse values (B,K,D), selected indices (B,K), and dense token count."""

    values: torch.Tensor
    indices: torch.Tensor
    token_count: int

    @property
    def index_bits_per_packet(self) -> int:
        return self.indices.shape[1] * math.ceil(math.log2(max(self.token_count, 2)))


def pack_sparse_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> SparseTokenPacket:
    """Extract equal-cardinality selected tokens and their source positions."""

    if tokens.ndim != 3:
        raise ValueError("tokens must have shape (B,N,D).")
    if mask.shape not in {
        (tokens.shape[0], tokens.shape[1]),
        (tokens.shape[0], tokens.shape[1], 1),
    }:
        raise ValueError("mask must have shape (B,N) or (B,N,1).")
    selected = mask.squeeze(-1).to(dtype=torch.bool)
    counts = selected.sum(dim=1)
    if counts.numel() == 0 or not torch.equal(counts, counts[:1].expand_as(counts)):
        raise ValueError("the selected count must be equal for every batch item.")
    count = int(counts[0].item())
    if count:
        indices = torch.stack(
            [selected[i].nonzero(as_tuple=False).squeeze(1) for i in range(tokens.shape[0])]
        )
    else:
        indices = torch.empty(tokens.shape[0], 0, dtype=torch.long, device=tokens.device)
    values = torch.gather(tokens, 1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
    return SparseTokenPacket(values=values, indices=indices, token_count=tokens.shape[1])


def unpack_sparse_tokens(packet: SparseTokenPacket, *, fill_value: float = 0.0) -> torch.Tensor:
    """Restore a dense (B,N,D) tensor for an unchanged token decoder."""

    batch, _, dim = packet.values.shape
    dense = packet.values.new_full((batch, packet.token_count, dim), fill_value)
    return dense.scatter_(1, packet.indices.unsqueeze(-1).expand(-1, -1, dim), packet.values)
