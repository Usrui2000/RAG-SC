"""Sensitivity-Aware Refinement Token Selection (SART)."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_spatial_map(value: torch.Tensor) -> torch.Tensor:
    """Normalize each view by its spatial mean, as in the paper."""

    return value / value.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)


def image_gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    """Return an RGB-averaged first-order edge magnitude map."""

    dx = (image[..., :, :, 1:] - image[..., :, :, :-1]).abs().mean(
        dim=2,
        keepdim=True,
    )
    dy = (image[..., :, 1:, :] - image[..., :, :-1, :]).abs().mean(
        dim=2,
        keepdim=True,
    )
    return F.pad(dx, (0, 1, 0, 0)) + F.pad(dy, (0, 0, 0, 1))


@dataclass
class SARTSelection:
    tokens: torch.Tensor
    mask: torch.Tensor
    scores: torch.Tensor
    keep_ratio: torch.Tensor


class SensitivityAwareRefinementSelector(nn.Module):
    """Rank refinement tokens using geometry, edges, and token magnitude."""

    def __init__(
        self,
        num_tokens_per_view: int,
        sensitivity_weight: float = 1.0,
        geometry_weight: float = 1.0,
        edge_weight: float = 0.25,
    ) -> None:
        super().__init__()
        grid_size = math.isqrt(num_tokens_per_view)
        if grid_size * grid_size != num_tokens_per_view:
            raise ValueError("num_tokens_per_view must be a square number.")
        self.num_tokens_per_view = num_tokens_per_view
        self.grid_size = grid_size
        self.sensitivity_weight = sensitivity_weight
        self.geometry_weight = geometry_weight
        self.edge_weight = edge_weight

    def importance_map(
        self,
        appearance: torch.Tensor,
        disparity: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        """Build the spatial rendering-sensitivity proxy Q."""

        geometry = confidence.detach().clamp(0.0, 1.0) * normalize_spatial_map(
            disparity.detach().clamp_min(0.0)
        )
        edges = normalize_spatial_map(
            image_gradient_magnitude(appearance.detach()).clamp_min(0.0)
        )
        return normalize_spatial_map(
            self.geometry_weight * geometry + self.edge_weight * edges
        )

    def token_importance(self, importance_map: torch.Tensor) -> torch.Tensor:
        """Pool Q to the token grid and flatten views into one packet."""

        batch, views, _, height, width = importance_map.shape
        pooled = F.adaptive_avg_pool2d(
            importance_map.reshape(batch * views, 1, height, width),
            (self.grid_size, self.grid_size),
        )
        return pooled.flatten(2).reshape(
            batch,
            views * self.num_tokens_per_view,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        importance_map: torch.Tensor,
        keep_ratio: float | torch.Tensor,
    ) -> SARTSelection:
        """Keep a fixed-cardinality top-K subset of refinement tokens."""

        if tokens.ndim != 3:
            raise ValueError("tokens must have shape (B, V*N, C).")
        batch, count, _ = tokens.shape
        importance = self.token_importance(importance_map)
        if importance.shape != tokens.shape[:2]:
            raise ValueError(
                f"Importance shape {tuple(importance.shape)} does not match "
                f"token shape {tuple(tokens.shape[:2])}."
            )

        if isinstance(keep_ratio, torch.Tensor):
            ratio = keep_ratio.to(device=tokens.device, dtype=tokens.dtype)
            ratio = ratio.reshape(-1)
            if ratio.numel() == 1:
                ratio = ratio.expand(batch)
        else:
            ratio = tokens.new_full((batch,), float(keep_ratio))
        if ratio.numel() != batch:
            raise ValueError("keep_ratio must be scalar or contain one value per batch item.")
        ratio = ratio.clamp(0.0, 1.0)

        magnitude = tokens.detach().square().mean(dim=-1)
        normalized = importance / importance.mean(dim=1, keepdim=True).clamp_min(1e-6)
        scores = magnitude * (1.0 + self.sensitivity_weight * normalized)
        keep_counts = (ratio * count).round().long().clamp(0, count)
        mask = torch.zeros(batch, count, 1, device=tokens.device, dtype=tokens.dtype)
        for batch_index, keep_count in enumerate(keep_counts.tolist()):
            if keep_count:
                selected = scores[batch_index].topk(keep_count).indices
                mask[batch_index, selected, 0] = 1.0
        return SARTSelection(tokens * mask, mask, scores, ratio)


__all__ = [
    "SARTSelection",
    "SensitivityAwareRefinementSelector",
    "image_gradient_magnitude",
    "normalize_spatial_map",
]
