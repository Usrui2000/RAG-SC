"""Framework-level composition of the three RAG-SC modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .prtb import ProgressiveRenderTokenBottleneck
from .sart import SensitivityAwareRefinementSelector
from .tgtr import TokenGuidedTextureRestorer, decode_joint_context


@dataclass
class RAGSCOutput:
    rendered: torch.Tensor
    bootstrap_render: torch.Tensor
    bootstrap_depth: torch.Tensor
    decoded_features: torch.Tensor
    decoded_context: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    gaussians: Any
    bootstrap_gaussians: Any
    gaussian_update_gate: torch.Tensor | None
    rate_proxy: torch.Tensor
    channel_stats: dict[str, float | torch.Tensor]


class RAGSC(nn.Module):
    """Compose P-RTB, SART, Gaussian prediction, rendering, and TGTR.

    ``gaussian_predictor`` and ``depth_renderer`` are adapters supplied by the
    host 3DGS model. Their expected call signatures are documented in
    ``forward`` so this core package remains independent of a particular
    Gaussian parameterization or rasterizer.
    """

    def __init__(
        self,
        prtb: ProgressiveRenderTokenBottleneck,
        gaussian_predictor: nn.Module,
        depth_renderer: nn.Module,
        sart: SensitivityAwareRefinementSelector | None = None,
        tgtr: TokenGuidedTextureRestorer | None = None,
    ) -> None:
        super().__init__()
        self.prtb = prtb
        self.gaussian_predictor = gaussian_predictor
        self.depth_renderer = depth_renderer
        self.sart = sart
        self.tgtr = tgtr

        if sart is not None:
            if (
                not prtb.sensitivity_token_selection
                or prtb.token_selection_mode != "sart"
            ):
                raise ValueError(
                    "SART requires P-RTB with sensitivity_token_selection=True "
                    "and token_selection_mode=\"sart\"."
                )

    def forward(
        self,
        features: torch.Tensor,
        context_appearance: torch.Tensor,
        context_disparity: torch.Tensor,
        context_confidence: torch.Tensor,
        point_cloud: torch.Tensor,
        context_extrinsics: torch.Tensor,
        context_intrinsics: torch.Tensor,
        target_extrinsics: torch.Tensor,
        target_intrinsics: torch.Tensor,
    ) -> RAGSCOutput:
        """Run the transmit-and-render path.

        The Gaussian adapter is called as
        ``predictor(features, appearance, disparity, confidence)``. The depth
        renderer is called as ``renderer(gaussians, extrinsics, intrinsics)``
        and must return target-view RGB and depth tensors.
        """

        importance = None
        if self.sart is not None:
            importance = self.sart.importance_map(
                context_appearance,
                context_disparity,
                context_confidence,
            )
        auxiliary = torch.cat([context_disparity, context_confidence], dim=2)
        compressed = self.prtb(
            features=features,
            point_cloud=point_cloud,
            extrinsics=context_extrinsics,
            intrinsics=context_intrinsics,
            aux_maps=auxiliary,
            image_maps=context_appearance,
            token_importance_maps=importance,
        )
        decoded_features, rate_proxy, _ = compressed
        decoded_appearance, decoded_depth, decoded_confidence = decode_joint_context(
            compressed
        )
        decoded_disparity = decoded_depth.reciprocal()
        gaussians = self.gaussian_predictor(
            decoded_features,
            decoded_appearance,
            decoded_disparity,
            decoded_confidence,
        )
        bootstrap, bootstrap_depth = self.depth_renderer(
            gaussians,
            target_extrinsics,
            target_intrinsics,
        )
        rendered = bootstrap
        refined_gaussians = gaussians
        gaussian_update_gate = None
        if self.tgtr is not None:
            gaussian_dc_update, _, _, _, gaussian_update_gate = self.tgtr(
                gaussians,
                bootstrap,
                bootstrap_depth,
                (decoded_appearance, decoded_depth, decoded_confidence),
                target_extrinsics,
                target_intrinsics,
                context_extrinsics,
                context_intrinsics,
            )
            views = target_extrinsics.shape[1]
            refined_views = []
            view_gaussians = []
            for view_index in range(views):
                refined_view_gaussians = self.tgtr.apply_gaussian_appearance_update(
                    gaussians,
                    gaussian_dc_update[:, view_index],
                )
                refined_view, _ = self.depth_renderer(
                    refined_view_gaussians,
                    target_extrinsics[:, view_index : view_index + 1],
                    target_intrinsics[:, view_index : view_index + 1],
                )
                refined_views.append(refined_view)
                view_gaussians.append(refined_view_gaussians)
            rendered = torch.cat(refined_views, dim=1)
            refined_gaussians = tuple(view_gaussians)
        return RAGSCOutput(
            rendered=rendered,
            bootstrap_render=bootstrap,
            bootstrap_depth=bootstrap_depth,
            decoded_features=decoded_features,
            decoded_context=(
                decoded_appearance,
                decoded_depth,
                decoded_confidence,
            ),
            gaussians=refined_gaussians,
            bootstrap_gaussians=gaussians,
            gaussian_update_gate=gaussian_update_gate,
            rate_proxy=rate_proxy,
            channel_stats=dict(self.prtb.last_channel_stats),
        )


__all__ = ["RAGSC", "RAGSCOutput"]
