from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .types import GaussianPrimitives


SH_C0 = 0.28209479177387814


@dataclass
class TextureRestoreCfg:
    hidden_dim: int = 32
    num_blocks: int = 4
    residual_scale: float = 0.2
    occlusion_tau: float = 0.12
    reliability_gating: bool = False
    guidance_blur_kernel: int = 5
    consistency_tau: float = 0.08
    coverage_gamma: float = 1.0
    detail_gate_floor: float = 0.05
    flat_denoise_strength: float = 0.12
    flat_edge_tau: float = 0.04
    gradient_weight: float = 0.15
    laplacian_weight: float = 0.1
    residual_weight: float = 0.005
    gaussian_update_scale: float = 1.0
    gaussian_depth_tau: float = 0.12


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.net(features)


class TextureResidualNet(nn.Module):
    """Predict a bounded target-view field for Gaussian appearance refinement."""

    def __init__(
        self,
        hidden_dim: int = 32,
        num_blocks: int = 4,
        residual_scale: float = 0.2,
        reliability_gating: bool = False,
        guidance_blur_kernel: int = 5,
        consistency_tau: float = 0.08,
        coverage_gamma: float = 1.0,
        detail_gate_floor: float = 0.05,
        flat_denoise_strength: float = 0.12,
        flat_edge_tau: float = 0.04,
    ) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.reliability_gating = reliability_gating
        if guidance_blur_kernel < 3 or guidance_blur_kernel % 2 == 0:
            raise ValueError("guidance_blur_kernel must be an odd integer >= 3.")
        self.guidance_blur_kernel = guidance_blur_kernel
        self.consistency_tau = max(consistency_tau, 1e-4)
        self.coverage_gamma = max(coverage_gamma, 0.0)
        self.detail_gate_floor = min(max(detail_gate_floor, 0.0), 1.0)
        self.flat_denoise_strength = min(max(flat_denoise_strength, 0.0), 1.0)
        self.flat_edge_tau = max(flat_edge_tau, 1e-4)
        input_channels = 11
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, 1, 1),
            nn.GELU(),
        )
        self.local_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.coarse_blocks = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, 1, 1),
            nn.GELU(),
            ResidualBlock(hidden_dim),
        )
        self.residual_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 3, 3, 1, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    @staticmethod
    def _normalized_log_depth(depth: torch.Tensor) -> torch.Tensor:
        log_depth = depth.clamp_min(1e-4).log()
        mean = log_depth.mean(dim=(-2, -1), keepdim=True)
        scale = log_depth.std(dim=(-2, -1), keepdim=True).clamp_min(1e-3)
        return ((log_depth - mean) / scale).clamp(-3.0, 3.0) / 3.0

    def _blur(self, image: torch.Tensor) -> torch.Tensor:
        padding = self.guidance_blur_kernel // 2
        padded = F.pad(
            image,
            (padding, padding, padding, padding),
            mode="reflect",
        )
        return F.avg_pool2d(
            padded,
            kernel_size=self.guidance_blur_kernel,
            stride=1,
        )

    def _flat_region_gate(self, image: torch.Tensor) -> torch.Tensor:
        dx = F.pad(
            (image[..., :, 1:] - image[..., :, :-1]).abs().mean(1, keepdim=True),
            (0, 1, 0, 0),
        )
        dy = F.pad(
            (image[..., 1:, :] - image[..., :-1, :]).abs().mean(1, keepdim=True),
            (0, 0, 0, 1),
        )
        return torch.exp(-(dx + dy) / self.flat_edge_tau)

    def _predict_residual(self, inputs: torch.Tensor) -> torch.Tensor:
        local = self.local_blocks(self.stem(inputs))
        coarse = F.avg_pool2d(local, kernel_size=2, stride=2)
        coarse = self.coarse_blocks(coarse)
        coarse = F.interpolate(
            coarse,
            size=local.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused = self.fusion(torch.cat([local, coarse], dim=1))
        return self.residual_scale * torch.tanh(self.residual_head(fused))

    def forward(
        self,
        rendered: torch.Tensor,
        warped_texture: torch.Tensor,
        coverage: torch.Tensor,
        depth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.reliability_gating:
            warped_high = warped_texture - F.avg_pool2d(
                warped_texture,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            inputs = torch.cat(
                [
                    rendered,
                    warped_texture,
                    warped_high,
                    coverage,
                    self._normalized_log_depth(depth),
                ],
                dim=1,
            )
            residual = self._predict_residual(inputs)
            restored = (rendered + residual).clamp(0.0, 1.0)
            return restored, residual

        rendered_smooth = self._blur(rendered)
        warped_smooth = self._blur(warped_texture)
        warped_high = warped_texture - warped_smooth

        # Reject reprojected low-frequency color drift. Only geometrically
        # covered, photometrically consistent details are used as guidance.
        color_error = (warped_smooth - rendered_smooth).abs().mean(
            dim=1,
            keepdim=True,
        )
        reliability = coverage.clamp(0.0, 1.0).pow(self.coverage_gamma)
        reliability = reliability * torch.exp(-color_error / self.consistency_tau)
        detail_gate = self.detail_gate_floor + (
            1.0 - self.detail_gate_floor
        ) * reliability
        guided_high = warped_high * reliability

        # Apply a small edge-aware smoothing only in flat regions. The learned
        # residual then restores reliable high frequencies without sharpening
        # channel artifacts.
        denoised = rendered + (
            self.flat_denoise_strength
            * self._flat_region_gate(rendered)
            * (rendered_smooth - rendered)
        )
        guided_texture = denoised + guided_high
        inputs = torch.cat(
            [
                denoised,
                guided_texture,
                guided_high,
                reliability,
                self._normalized_log_depth(depth),
            ],
            dim=1,
        )
        predicted_residual = self._predict_residual(inputs)
        residual_low = self._blur(predicted_residual)
        residual_high = predicted_residual - residual_low
        learned_residual = residual_low + detail_gate * residual_high
        residual = learned_residual + (denoised - rendered)
        restored = (rendered + residual).clamp(0.0, 1.0)
        return restored, residual


def decode_joint_context(
    compressor_output: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode the joint compressor output without accessing raw context images."""

    if not isinstance(compressor_output, tuple) or len(compressor_output) < 3:
        raise RuntimeError("Joint compressor did not return token-decoded auxiliary maps.")
    raw_aux = compressor_output[2]
    if not isinstance(raw_aux, torch.Tensor) or raw_aux.shape[2] < 5:
        raise RuntimeError(
            "Expected token-decoded RGB, disparity, and confidence channels."
        )
    image = torch.sigmoid(raw_aux[:, :, :3])
    disparity = F.softplus(raw_aux[:, :, 3:4]) + 1e-4
    depth = disparity.reciprocal()
    confidence = torch.sigmoid(raw_aux[:, :, 4:5])
    return image, depth, confidence


def _target_to_context_grid(
    target_depth: torch.Tensor,
    target_extrinsic: torch.Tensor,
    target_intrinsic: torch.Tensor,
    context_extrinsic: torch.Tensor,
    context_intrinsic: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project target pixels into one context view.

    Dataset extrinsics are camera-to-world transforms and intrinsics operate in
    normalized image coordinates.
    """

    height, width = target_depth.shape
    device = target_depth.device
    dtype = target_depth.dtype
    y, x = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    pixels = torch.stack([x, y, torch.ones_like(x)], dim=-1).reshape(-1, 3)
    rays = pixels @ torch.linalg.inv(target_intrinsic).transpose(0, 1)
    points_target = rays * target_depth.reshape(-1, 1)
    points_target_h = torch.cat(
        [points_target, torch.ones_like(points_target[:, :1])],
        dim=-1,
    )
    points_world = points_target_h @ target_extrinsic.transpose(0, 1)
    points_context = points_world @ torch.linalg.inv(
        context_extrinsic
    ).transpose(0, 1)
    xyz_context = points_context[:, :3]
    projected = xyz_context @ context_intrinsic.transpose(0, 1)
    projected_depth = xyz_context[:, 2]
    uv = projected[:, :2] / projected[:, 2:3].clamp_min(1e-6)
    grid = uv.reshape(height, width, 2) * 2.0 - 1.0
    valid = (
        (projected_depth > 1e-4)
        & (grid[..., 0].reshape(-1) >= -1.0)
        & (grid[..., 0].reshape(-1) <= 1.0)
        & (grid[..., 1].reshape(-1) >= -1.0)
        & (grid[..., 1].reshape(-1) <= 1.0)
    )
    return grid, projected_depth.reshape(height, width) * valid.reshape(height, width)


def fuse_reprojected_context(
    rendered: torch.Tensor,
    target_depth: torch.Tensor,
    context_image: torch.Tensor,
    context_depth: torch.Tensor,
    context_confidence: torch.Tensor,
    target_extrinsics: torch.Tensor,
    target_intrinsics: torch.Tensor,
    context_extrinsics: torch.Tensor,
    context_intrinsics: torch.Tensor,
    occlusion_tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse token-decoded context textures in each target camera."""

    batch, target_views, _, height, width = rendered.shape
    context_views = context_image.shape[1]
    fused_views = []
    coverage_views = []
    for batch_index in range(batch):
        for target_index in range(target_views):
            numerator = torch.zeros(
                3,
                height,
                width,
                dtype=rendered.dtype,
                device=rendered.device,
            )
            denominator = torch.zeros(
                1,
                height,
                width,
                dtype=rendered.dtype,
                device=rendered.device,
            )
            for context_index in range(context_views):
                grid, projected_depth = _target_to_context_grid(
                    target_depth[batch_index, target_index],
                    target_extrinsics[batch_index, target_index],
                    target_intrinsics[batch_index, target_index],
                    context_extrinsics[batch_index, context_index],
                    context_intrinsics[batch_index, context_index],
                )
                sample_grid = grid.unsqueeze(0)
                warped_rgb = F.grid_sample(
                    context_image[batch_index, context_index].unsqueeze(0),
                    sample_grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )[0]
                warped_depth = F.grid_sample(
                    context_depth[batch_index, context_index].unsqueeze(0),
                    sample_grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )[0]
                warped_confidence = F.grid_sample(
                    context_confidence[batch_index, context_index].unsqueeze(0),
                    sample_grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )[0]
                valid = (projected_depth > 0).to(rendered.dtype).unsqueeze(0)
                depth_error = (
                    projected_depth.clamp_min(1e-4).log().unsqueeze(0)
                    - warped_depth.clamp_min(1e-4).log()
                ).abs()
                visibility = (
                    valid
                    * warped_confidence.clamp(0.0, 1.0)
                    * torch.exp(-depth_error / max(occlusion_tau, 1e-4))
                )
                numerator = numerator + warped_rgb * visibility
                denominator = denominator + visibility
            coverage = denominator.clamp(0.0, 1.0)
            fused = numerator / denominator.clamp_min(1e-4)
            fallback = rendered[batch_index, target_index]
            fused = torch.where(denominator > 1e-4, fused, fallback)
            fused_views.append(fused)
            coverage_views.append(coverage)
    return (
        torch.stack(fused_views).reshape(batch, target_views, 3, height, width),
        torch.stack(coverage_views).reshape(batch, target_views, 1, height, width),
    )


class TokenGuidedTextureRestorer(nn.Module):
    """Feed token-guided target-view texture back into Gaussian appearance."""

    def __init__(self, cfg: TextureRestoreCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.network = TextureResidualNet(
            hidden_dim=cfg.hidden_dim,
            num_blocks=cfg.num_blocks,
            residual_scale=cfg.residual_scale,
            reliability_gating=cfg.reliability_gating,
            guidance_blur_kernel=cfg.guidance_blur_kernel,
            consistency_tau=cfg.consistency_tau,
            coverage_gamma=cfg.coverage_gamma,
            detail_gate_floor=cfg.detail_gate_floor,
            flat_denoise_strength=cfg.flat_denoise_strength,
            flat_edge_tau=cfg.flat_edge_tau,
        )

    def forward(
        self,
        gaussians: GaussianPrimitives,
        rendered: torch.Tensor,
        target_depth: torch.Tensor,
        decoded_context: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target_extrinsics: torch.Tensor,
        target_intrinsics: torch.Tensor,
        context_extrinsics: torch.Tensor,
        context_intrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context_image, context_depth, context_confidence = decoded_context
        warped, coverage = fuse_reprojected_context(
            rendered,
            target_depth,
            context_image,
            context_depth,
            context_confidence,
            target_extrinsics,
            target_intrinsics,
            context_extrinsics,
            context_intrinsics,
            self.cfg.occlusion_tau,
        )
        flat_rendered = rearrange(rendered, "b v c h w -> (b v) c h w")
        flat_warped = rearrange(warped, "b v c h w -> (b v) c h w")
        flat_coverage = rearrange(coverage, "b v c h w -> (b v) c h w")
        flat_depth = rearrange(target_depth, "b v h w -> (b v) () h w")
        _, residual = self.network(
            flat_rendered,
            flat_warped,
            flat_coverage,
            flat_depth,
        )
        batch, views = rendered.shape[:2]
        residual = rearrange(
            residual,
            "(b v) c h w -> b v c h w",
            b=batch,
            v=views,
        )
        gaussian_dc_update, gaussian_gate = self._project_gaussian_appearance_update(
            gaussians,
            residual,
            coverage,
            target_depth,
            target_extrinsics,
            target_intrinsics,
        )
        return (
            gaussian_dc_update,
            residual,
            warped,
            coverage,
            gaussian_gate,
        )

    def _project_gaussian_appearance_update(
        self,
        gaussians: GaussianPrimitives,
        appearance_update: torch.Tensor,
        coverage: torch.Tensor,
        target_depth: torch.Tensor,
        target_extrinsics: torch.Tensor,
        target_intrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lift a target-view RGB update to visible Gaussian SH-DC updates.

        Geometry and opacity remain shared and unchanged. Returning only the
        per-view DC updates avoids replicating the full spherical-harmonic
        tensor for every target camera; callers apply and render one target
        view at a time.
        """

        batch, views, _, height, width = appearance_update.shape
        if gaussians.means.shape[0] != batch:
            raise ValueError(
                "Gaussian batch does not match the target-view update batch."
            )

        means_h = torch.cat(
            [gaussians.means, torch.ones_like(gaussians.means[..., :1])],
            dim=-1,
        )
        world_to_camera = torch.linalg.inv(target_extrinsics)
        camera_h = torch.einsum("bvij,bgj->bvgi", world_to_camera, means_h)
        camera_xyz = camera_h[..., :3]
        projected = torch.einsum(
            "bvij,bvgj->bvgi",
            target_intrinsics,
            camera_xyz,
        )
        camera_depth = camera_xyz[..., 2]
        uv = projected[..., :2] / projected[..., 2:3].clamp_min(1e-6)
        sample_grid = (uv * 2.0 - 1.0).reshape(batch * views, -1, 1, 2)

        flat_update = appearance_update.reshape(batch * views, 3, height, width)
        sampled_update = F.grid_sample(
            flat_update,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled_update = sampled_update.squeeze(-1).transpose(1, 2)
        sampled_update = sampled_update.reshape(batch, views, -1, 3)

        flat_coverage = coverage.reshape(batch * views, 1, height, width)
        sampled_coverage = F.grid_sample(
            flat_coverage,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled_coverage = sampled_coverage.squeeze(1).squeeze(-1)
        sampled_coverage = sampled_coverage.reshape(batch, views, -1)

        flat_depth = target_depth.reshape(batch * views, 1, height, width)
        sampled_depth = F.grid_sample(
            flat_depth,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled_depth = sampled_depth.squeeze(1).squeeze(-1)
        sampled_depth = sampled_depth.reshape(batch, views, -1)

        valid = (
            (camera_depth > 1e-4)
            & (uv[..., 0] >= 0.0)
            & (uv[..., 0] <= 1.0)
            & (uv[..., 1] >= 0.0)
            & (uv[..., 1] <= 1.0)
            & (sampled_depth > 1e-4)
        )
        depth_error = (
            camera_depth.clamp_min(1e-4).log()
            - sampled_depth.clamp_min(1e-4).log()
        ).abs()
        depth_gate = torch.exp(
            -depth_error / max(self.cfg.gaussian_depth_tau, 1e-4)
        )
        gaussian_gate = (
            valid.to(appearance_update.dtype)
            * sampled_coverage.clamp(0.0, 1.0)
            * depth_gate
        )

        dc_update = (
            self.cfg.gaussian_update_scale
            * gaussian_gate[..., None]
            * sampled_update
            / SH_C0
        )
        return dc_update, gaussian_gate

    @staticmethod
    def apply_gaussian_appearance_update(
        gaussians: GaussianPrimitives,
        dc_update: torch.Tensor,
    ) -> GaussianPrimitives:
        """Apply one target view's RGB update to Gaussian SH-DC coefficients."""

        if dc_update.shape != gaussians.harmonics[..., 0].shape:
            raise ValueError(
                "Expected a Gaussian DC update with shape "
                f"{tuple(gaussians.harmonics[..., 0].shape)}, got "
                f"{tuple(dc_update.shape)}."
            )
        harmonics = gaussians.harmonics.clone()
        harmonics[..., 0] = harmonics[..., 0] + dc_update
        return GaussianPrimitives(
            means=gaussians.means,
            covariances=gaussians.covariances,
            harmonics=harmonics,
            opacities=gaussians.opacities,
        )


def gradient_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def laplacian_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    channels = prediction.shape[-3]
    kernel = prediction.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).reshape(1, 1, 3, 3)
    kernel = kernel.expand(channels, 1, 3, 3)
    flat_prediction = prediction.reshape(-1, channels, *prediction.shape[-2:])
    flat_target = target.reshape(-1, channels, *target.shape[-2:])
    pred_laplacian = F.conv2d(flat_prediction, kernel, padding=1, groups=channels)
    target_laplacian = F.conv2d(flat_target, kernel, padding=1, groups=channels)
    return F.l1_loss(pred_laplacian, target_laplacian)
