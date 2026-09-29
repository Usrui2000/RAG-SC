"""Minimal host-independent Gaussian representation used by TGTR."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GaussianPrimitives:
    """Batched primitives with means (B,G,3), covariances (B,G,3,3),
    harmonics (B,G,3,D), and opacities (B,G)."""

    means: torch.Tensor
    covariances: torch.Tensor
    harmonics: torch.Tensor
    opacities: torch.Tensor



__all__ = ["GaussianPrimitives"]
