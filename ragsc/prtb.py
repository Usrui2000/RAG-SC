from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _square_grid_size(num_tokens: int) -> int:
    grid_size = int(math.isqrt(num_tokens))
    if grid_size * grid_size != num_tokens:
        raise ValueError(f"num_tokens must be a square number, got {num_tokens}.")
    return grid_size


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(min=1e-4, max=30.0)
    return value + torch.log(-torch.expm1(-value))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.gelu(self.norm1(x))
        residual = self.conv1(residual)
        residual = F.gelu(self.norm2(residual))
        return x + self.conv2(residual)


class JointAnalysisTransform(nn.Module):
    """Encode aligned feature, appearance, and geometry maps into one stream."""

    def __init__(
        self,
        feature_channels: int,
        aux_channels: int,
        num_tokens: int,
        token_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.grid_size = _square_grid_size(num_tokens)

        feature_hidden = hidden_dim // 2
        image_hidden = hidden_dim // 4
        aux_hidden = hidden_dim - feature_hidden - image_hidden
        self.feature_stem = nn.Sequential(
            nn.Conv2d(feature_channels, feature_hidden, 3, 1, 1),
            nn.GroupNorm(_group_count(feature_hidden), feature_hidden),
            nn.GELU(),
            ResidualBlock(feature_hidden),
        )
        self.image_stem = nn.Sequential(
            nn.Conv2d(3, image_hidden, 3, 1, 1),
            nn.GroupNorm(_group_count(image_hidden), image_hidden),
            nn.GELU(),
            ResidualBlock(image_hidden),
        )
        self.aux_stem = nn.Sequential(
            nn.Conv2d(aux_channels, aux_hidden, 3, 1, 1),
            nn.GroupNorm(_group_count(aux_hidden), aux_hidden),
            nn.GELU(),
            ResidualBlock(aux_hidden),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 5, 2, 2),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim),
        )
        self.to_tokens = nn.Conv2d(hidden_dim, token_dim, 1)
        self.token_norm = nn.LayerNorm(token_dim)
        self.channel_projection = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        image_signal: torch.Tensor,
        aux_signal: torch.Tensor,
    ) -> torch.Tensor:
        encoded = torch.cat(
            [
                self.feature_stem(features),
                self.image_stem(image_signal),
                self.aux_stem(aux_signal),
            ],
            dim=1,
        )
        encoded = self.downsample(encoded)
        encoded = F.adaptive_avg_pool2d(encoded, (self.grid_size, self.grid_size))
        tokens = self.to_tokens(encoded).flatten(2).transpose(1, 2)
        return self.channel_projection(self.token_norm(tokens))


class JointSynthesisTransform(nn.Module):
    """Decode one token stream into task-aligned receiver conditioning maps."""

    def __init__(
        self,
        feature_channels: int,
        aux_channels: int,
        num_tokens: int,
        token_dim: int,
        hidden_dim: int = 128,
        zero_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.grid_size = _square_grid_size(num_tokens)
        self.from_channel = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, hidden_dim),
        )
        self.grid_blocks = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )
        self.up_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
                    nn.GELU(),
                    ResidualBlock(hidden_dim),
                )
                for _ in range(3)
            ]
        )
        self.feature_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, feature_channels, 3, 1, 1),
        )
        self.image_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 3, 3, 1, 1),
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, aux_channels, 3, 1, 1),
        )
        if zero_output:
            for head in (self.feature_head, self.image_head, self.aux_head):
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    def forward(
        self,
        tokens: torch.Tensor,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, count, _ = tokens.shape
        if count != self.num_tokens:
            raise ValueError(f"Expected {self.num_tokens} tokens, got {count}.")
        decoded = self.from_channel(tokens)
        decoded = decoded.transpose(1, 2).reshape(
            batch,
            -1,
            self.grid_size,
            self.grid_size,
        )
        decoded = self.grid_blocks(decoded)

        target_sizes = [
            (min(height, self.grid_size * 2), min(width, self.grid_size * 2)),
            (min(height, self.grid_size * 4), min(width, self.grid_size * 4)),
            (height, width),
        ]
        for target_size, block in zip(target_sizes, self.up_blocks):
            if decoded.shape[-2:] != target_size:
                decoded = F.interpolate(
                    decoded,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            decoded = block(decoded)
        return (
            self.feature_head(decoded),
            self.image_head(decoded),
            self.aux_head(decoded),
        )


class JointTokenSemanticCompressor(nn.Module):
    """Pure-token joint bottleneck with base and progressive refinement streams.

    This class matches the existing ``SemanticCompressor`` constructor and
    forward contract, but it does not instantiate independent image or aux
    codecs. ``bottleneck_dim`` is the base-token dimension. The existing
    ``image_token_dim`` setting is intentionally reused as the refinement-token
    dimension for this isolated experiment path.
    """

    def __init__(
        self,
        in_features_dim: int,
        aux_channels: int = 0,
        use_aux_tokens: bool = False,
        aux_num_tokens: int = 1024,
        aux_token_dim: int = 64,
        image_channels: int = 3,
        use_image_tokens: bool = True,
        image_num_tokens: int = 1024,
        image_token_dim: int = 32,
        feature_codec_mode: str = "grid",
        token_residual_enabled: bool = True,
        token_residual_num_tokens: int | None = None,
        token_residual_keep_ratio: float = 1.0,
        token_residual_adaptive: bool = False,
        token_residual_min_keep: float = 0.25,
        token_residual_clean_keep: float = 1.0,
        token_residual_snr_mid: float = 10.0,
        token_residual_snr_scale: float = 4.0,
        token_residual_scale: float = 1.0,
        sensitivity_token_selection: bool = False,
        token_selection_mode: str = "sart",
        sensitivity_token_weight: float = 1.0,
        num_tokens: int = 1024,
        bottleneck_dim: int = 96,
        strict_bottleneck: bool = True,
        transmit_geometry: bool = True,
        channel_model: str = "snr",
        eval_channel: bool = False,
        snr_db: float = 20.0,
        train_channel_prob: float = 1.0,
        train_snr_min: float = 0.0,
        train_snr_max: float = 20.0,
        snr_noise_scale: float = 1.0,
        snr_power_normalize: bool = True,
        snr_channel_type: str = "awgn",
        adaptive_snr: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        if feature_codec_mode != "grid":
            raise ValueError("JointTokenSemanticCompressor only supports grid mode.")
        if aux_channels <= 0:
            raise ValueError("Joint token mode requires image/geometry auxiliary inputs.")
        if image_channels != 3:
            raise ValueError(f"Expected RGB image inputs, got {image_channels} channels.")

        self.num_tokens = num_tokens
        self.bottleneck_dim = bottleneck_dim
        self.refinement_num_tokens = token_residual_num_tokens or num_tokens
        self.refinement_dim = image_token_dim
        self.aux_channels = aux_channels
        self.image_channels = image_channels
        self.use_image_tokens = use_image_tokens
        self.use_aux_tokens = use_aux_tokens
        self.feature_codec_mode = feature_codec_mode
        self.strict_bottleneck = strict_bottleneck
        self.transmit_geometry = transmit_geometry
        self.channel_model = channel_model
        self.eval_channel = eval_channel
        self.snr_db = snr_db
        self.train_channel_prob = train_channel_prob
        self.train_snr_min = train_snr_min
        self.train_snr_max = train_snr_max
        self.snr_noise_scale = snr_noise_scale
        self.snr_power_normalize = snr_power_normalize
        if snr_channel_type not in {"awgn", "rayleigh"}:
            raise ValueError(
                "snr_channel_type must be awgn or rayleigh, got "
                f"{snr_channel_type}."
            )
        self.snr_channel_type = snr_channel_type
        self.adaptive_snr = adaptive_snr
        self.token_residual_enabled = token_residual_enabled
        self.token_residual_keep_ratio = token_residual_keep_ratio
        self.token_residual_adaptive = token_residual_adaptive
        self.token_residual_min_keep = token_residual_min_keep
        self.token_residual_clean_keep = token_residual_clean_keep
        self.token_residual_snr_mid = token_residual_snr_mid
        self.token_residual_snr_scale = max(token_residual_snr_scale, 1e-6)
        self.token_residual_scale = token_residual_scale
        self.sensitivity_token_selection = sensitivity_token_selection
        if token_selection_mode not in {"energy", "random", "sart"}:
            raise ValueError(
                "token_selection_mode must be energy, random, or sart, got "
                f"{token_selection_mode}."
            )
        self.token_selection_mode = token_selection_mode
        self.sensitivity_token_weight = sensitivity_token_weight

        self.joint_base_encoder = JointAnalysisTransform(
            in_features_dim,
            aux_channels,
            num_tokens,
            bottleneck_dim,
        )
        self.joint_base_decoder = JointSynthesisTransform(
            in_features_dim,
            aux_channels,
            num_tokens,
            bottleneck_dim,
        )
        if token_residual_enabled:
            self.joint_refinement_encoder = JointAnalysisTransform(
                in_features_dim,
                aux_channels,
                self.refinement_num_tokens,
                self.refinement_dim,
            )
            self.joint_refinement_decoder = JointSynthesisTransform(
                in_features_dim,
                aux_channels,
                self.refinement_num_tokens,
                self.refinement_dim,
                zero_output=True,
            )
        else:
            self.joint_refinement_encoder = None
            self.joint_refinement_decoder = None

        self.last_channel_stats: dict[str, float | torch.Tensor] = {}
        self.last_reconstructed_features: torch.Tensor | None = None
        self.last_base_features: torch.Tensor | None = None
        self.last_token_residual_features: torch.Tensor | None = None

    def _canonical_inputs(
        self,
        features: torch.Tensor,
        image_maps: torch.Tensor,
        aux_maps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_logits = torch.logit(image_maps.clamp(1e-3, 1.0 - 1e-3))
        disp_logits = _inverse_softplus(aux_maps[:, :, :1])
        if self.aux_channels > 1:
            pdf_logits = torch.logit(aux_maps[:, :, 1:].clamp(1e-4, 1.0 - 1e-4))
            aux_logits = torch.cat([disp_logits, pdf_logits], dim=2)
        else:
            aux_logits = disp_logits
        return features, image_logits, aux_logits

    def _sample_channel_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, bool]:
        if self.training:
            active = torch.rand((), device=device).item() < self.train_channel_prob
            low = min(self.train_snr_min, self.train_snr_max)
            high = max(self.train_snr_min, self.train_snr_max)
            snr = low + torch.rand(batch_size, 1, device=device, dtype=dtype) * (high - low)
        else:
            active = self.eval_channel
            snr = torch.full((batch_size, 1), self.snr_db, device=device, dtype=dtype)
        return snr, active

    def _apply_channel(
        self,
        tokens: torch.Tensor,
        snr_db: torch.Tensor,
        active: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        centered = tokens - tokens.mean(dim=(1, 2), keepdim=True)
        signal_rms = centered.detach().square().mean(dim=(1, 2), keepdim=True).clamp_min(1e-8).sqrt()
        if not active:
            noise_std = torch.zeros_like(signal_rms)
            channel_gain = torch.ones_like(signal_rms)
            return tokens, {
                "signal_rms": signal_rms.mean(),
                "noise_std": noise_std.mean(),
                "effective_noise_std": noise_std.mean(),
                "fading_gain": channel_gain.mean(),
            }

        snr_linear = (10.0 ** (snr_db / 10.0)).view(-1, 1, 1).clamp_min(1e-8)
        channel_gain = torch.ones_like(signal_rms)
        if self.snr_channel_type == "rayleigh":
            # One unit-power flat-fading coefficient per sample/token packet.
            # Perfect receiver CSI equalizes before the joint token decoder.
            channel_gain = torch.sqrt(
                torch.randn_like(signal_rms).square()
                + torch.randn_like(signal_rms).square()
            ).clamp_min(1e-3) / (2.0**0.5)
        if self.snr_power_normalize:
            transmitted = centered / signal_rms
            normalized_noise = snr_linear.rsqrt() * self.snr_noise_scale
            received = (
                channel_gain * transmitted
                + torch.randn_like(transmitted) * normalized_noise
            )
            if self.snr_channel_type == "rayleigh":
                received = received / channel_gain
            received = received * signal_rms + tokens.mean(dim=(1, 2), keepdim=True)
            effective_noise = normalized_noise * signal_rms / channel_gain
            noise_std = normalized_noise
        else:
            effective_noise = signal_rms * snr_linear.rsqrt() * self.snr_noise_scale
            received = (
                channel_gain * tokens
                + torch.randn_like(tokens) * effective_noise
            )
            if self.snr_channel_type == "rayleigh":
                received = received / channel_gain
                effective_noise = effective_noise / channel_gain
            noise_std = effective_noise
        return received, {
            "signal_rms": signal_rms.mean(),
            "noise_std": noise_std.mean(),
            "effective_noise_std": effective_noise.mean(),
            "fading_gain": channel_gain.mean(),
        }

    def add_noise(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.training:
            return tokens + torch.rand_like(tokens) - 0.5
        return torch.round(tokens)

    def get_rate(self, tokens: torch.Tensor) -> torch.Tensor:
        variance = torch.var(tokens, dim=(0, 1), keepdim=True) + 1e-6
        return 0.5 * torch.log2(2 * torch.pi * torch.e * variance).sum()

    def _refinement_keep_ratio(
        self,
        snr_db: torch.Tensor,
        active: bool,
    ) -> torch.Tensor:
        batch_size = snr_db.shape[0]
        if not self.token_residual_adaptive or not active:
            value = self.token_residual_clean_keep if not active else self.token_residual_keep_ratio
            return torch.full_like(snr_db, value).clamp(0.0, 1.0)
        progress = torch.sigmoid(
            (snr_db - self.token_residual_snr_mid) / self.token_residual_snr_scale
        )
        return self.token_residual_min_keep + (
            self.token_residual_clean_keep - self.token_residual_min_keep
        ) * progress

    def _importance_from_maps(
        self,
        importance_maps: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if importance_maps is None or not self.sensitivity_token_selection:
            return None
        batch, views, _, height, width = importance_maps.shape
        grid_size = _square_grid_size(self.refinement_num_tokens)
        pooled = F.adaptive_avg_pool2d(
            importance_maps.reshape(batch * views, 1, height, width).clamp_min(0.0),
            (grid_size, grid_size),
        )
        return pooled.flatten(2).reshape(batch, views * self.refinement_num_tokens)

    def _apply_keep_mask(
        self,
        tokens: torch.Tensor,
        keep_ratio: torch.Tensor,
        importance: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, count, _ = tokens.shape
        keep_counts = (keep_ratio.view(batch) * count).round().long().clamp(0, count)
        energy = tokens.detach().square().mean(dim=-1)
        score = (
            torch.rand_like(energy)
            if self.sensitivity_token_selection
            and self.token_selection_mode == "random"
            else energy
        )
        if importance is not None and self.token_selection_mode == "sart":
            if importance.shape != score.shape:
                raise ValueError(
                    f"Importance shape {tuple(importance.shape)} does not match {tuple(score.shape)}."
                )
            normalized = importance / importance.mean(dim=1, keepdim=True).clamp_min(1e-6)
            score = score * (1.0 + self.sensitivity_token_weight * normalized)
        mask = torch.zeros(batch, count, 1, dtype=tokens.dtype, device=tokens.device)
        for batch_idx in range(batch):
            amount = int(keep_counts[batch_idx].detach().cpu())
            if amount > 0:
                indices = score[batch_idx].topk(amount).indices
                mask[batch_idx, indices, 0] = 1.0
        return tokens * mask, mask.mean(), mask

    def _encode(
        self,
        encoder: JointAnalysisTransform,
        feature_signal: torch.Tensor,
        image_signal: torch.Tensor,
        aux_signal: torch.Tensor,
    ) -> torch.Tensor:
        batch, views, channels, height, width = feature_signal.shape
        return encoder(
            feature_signal.reshape(batch * views, channels, height, width),
            image_signal.reshape(batch * views, 3, height, width),
            aux_signal.reshape(batch * views, self.aux_channels, height, width),
        ).reshape(batch, views * encoder.num_tokens, encoder.token_dim)

    def _decode(
        self,
        decoder: JointSynthesisTransform,
        tokens: torch.Tensor,
        batch: int,
        views: int,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_map, image_logits, aux_logits = decoder(
            tokens.reshape(batch * views, decoder.num_tokens, decoder.token_dim),
            height,
            width,
        )
        return (
            feature_map.reshape(batch, views, feature_map.shape[1], height, width),
            image_logits.reshape(batch, views, 3, height, width),
            aux_logits.reshape(batch, views, self.aux_channels, height, width),
        )

    def forward(
        self,
        features: torch.Tensor,
        point_cloud: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        aux_maps: torch.Tensor | None = None,
        image_maps: torch.Tensor | None = None,
        token_importance_maps: torch.Tensor | None = None,
        residual_weight: float | torch.Tensor | None = None,
    ):
        del point_cloud, extrinsics, intrinsics
        if image_maps is None or aux_maps is None:
            raise ValueError("Joint token mode requires both image_maps and aux_maps.")
        batch, views, channels, height, width = features.shape
        feature_target, image_target, aux_target = self._canonical_inputs(
            features,
            image_maps,
            aux_maps,
        )
        snr_db, active = self._sample_channel_state(batch, features.device, features.dtype)

        base_tokens = self._encode(
            self.joint_base_encoder,
            feature_target,
            image_target,
            aux_target,
        )
        if self.channel_model == "snr":
            base_received, base_stats = self._apply_channel(base_tokens, snr_db, active)
        else:
            base_received = self.add_noise(base_tokens)
            zero = torch.zeros((), dtype=features.dtype, device=features.device)
            base_stats = {
                "signal_rms": base_tokens.detach().square().mean().sqrt(),
                "noise_std": zero,
                "effective_noise_std": zero,
                "fading_gain": torch.ones_like(zero),
            }
        rate = self.get_rate(base_received)
        base_feature, base_image, base_aux = self._decode(
            self.joint_base_decoder,
            base_received,
            batch,
            views,
            height,
            width,
        )
        self.last_base_features = base_feature

        reconstructed_feature = base_feature
        reconstructed_image = base_image
        reconstructed_aux = base_aux
        keep_ratio = torch.zeros_like(snr_db)
        actual_keep = torch.zeros((), dtype=features.dtype, device=features.device)
        refinement_stats: dict[str, torch.Tensor] = {}
        self.last_token_residual_features = None

        if self.joint_refinement_encoder is not None:
            residual_tokens = self._encode(
                self.joint_refinement_encoder,
                feature_target - base_feature.detach(),
                image_target - base_image.detach(),
                aux_target - base_aux.detach(),
            )
            keep_ratio = self._refinement_keep_ratio(snr_db, active)
            importance = self._importance_from_maps(token_importance_maps)
            residual_tokens, actual_keep, keep_mask = self._apply_keep_mask(
                residual_tokens,
                keep_ratio,
                importance,
            )
            if self.channel_model == "snr":
                residual_received, refinement_stats = self._apply_channel(
                    residual_tokens,
                    snr_db,
                    active,
                )
            else:
                residual_received = self.add_noise(residual_tokens)
            # Missing refinement packets must stay absent after channel noise.
            residual_received = residual_received * keep_mask
            rate = rate + self.get_rate(residual_received)
            residual_feature, residual_image, residual_aux = self._decode(
                self.joint_refinement_decoder,
                residual_received,
                batch,
                views,
                height,
                width,
            )
            residual_feature = residual_feature * self.token_residual_scale
            self.last_token_residual_features = residual_feature
            reconstructed_feature = reconstructed_feature + residual_feature
            reconstructed_image = reconstructed_image + residual_image * self.token_residual_scale
            reconstructed_aux = reconstructed_aux + residual_aux * self.token_residual_scale

        self.last_reconstructed_features = reconstructed_feature
        self.last_channel_stats = {
            "active": float(active),
            "snr_db": snr_db.mean().detach(),
            "signal_rms": base_stats["signal_rms"].detach(),
            "noise_std": base_stats["noise_std"].detach(),
            "effective_noise_std": base_stats.get("effective_noise_std", base_stats["noise_std"]).detach(),
            "noise_scale": torch.as_tensor(self.snr_noise_scale, dtype=features.dtype, device=features.device),
            "power_normalize": float(self.snr_power_normalize),
            "adaptive_snr": float(self.adaptive_snr),
            "channel_rayleigh": float(self.snr_channel_type == "rayleigh"),
            "fading_gain": base_stats["fading_gain"].detach(),
            "token_channel_snr": float(self.channel_model == "snr"),
            "joint_token_stream": 1.0,
            "independent_image_tokens": 0.0,
            "independent_aux_tokens": 0.0,
            "base_num_tokens": float(self.num_tokens),
            "base_token_dim": float(self.bottleneck_dim),
            "refinement_num_tokens": float(
                self.refinement_num_tokens if self.joint_refinement_encoder is not None else 0
            ),
            "refinement_token_dim": float(
                self.refinement_dim if self.joint_refinement_encoder is not None else 0
            ),
            "base_payload_symbols": float(views * self.num_tokens * self.bottleneck_dim),
            "refinement_payload_symbols": (
                actual_keep.detach()
                * float(views * self.refinement_num_tokens * self.refinement_dim)
            ),
            "total_payload_symbols": (
                actual_keep.detach()
                * float(views * self.refinement_num_tokens * self.refinement_dim)
                + float(views * self.num_tokens * self.bottleneck_dim)
            ),
            "token_residual_enabled": float(self.joint_refinement_encoder is not None),
            "token_residual_keep_ratio": keep_ratio.mean().detach(),
            "token_residual_actual_keep": actual_keep.detach(),
            "token_residual_scale": torch.as_tensor(
                self.token_residual_scale,
                dtype=features.dtype,
                device=features.device,
            ),
            "sensitivity_token_selection": float(
                self.sensitivity_token_selection and token_importance_maps is not None
            ),
            "selection_mode_random": float(
                self.sensitivity_token_selection
                and self.token_selection_mode == "random"
            ),
            "selection_mode_sart": float(
                self.sensitivity_token_selection
                and self.token_selection_mode == "sart"
            ),
        }
        if refinement_stats:
            self.last_channel_stats["refinement_signal_rms"] = refinement_stats[
                "signal_rms"
            ].detach()
            self.last_channel_stats["refinement_noise_std"] = refinement_stats[
                "noise_std"
            ].detach()
            self.last_channel_stats["refinement_fading_gain"] = refinement_stats[
                "fading_gain"
            ].detach()

        if residual_weight is None:
            residual_weight = 0.0 if self.strict_bottleneck else 1.0
        if isinstance(residual_weight, torch.Tensor):
            residual_weight = residual_weight.to(features.device, features.dtype)
        output_feature = (
            reconstructed_feature
            if self.strict_bottleneck
            else (1.0 - residual_weight) * reconstructed_feature + residual_weight * features
        )
        output_aux = torch.cat([reconstructed_image, reconstructed_aux], dim=2)
        return output_feature, rate, output_aux


# Public paper-facing name. The module never patches a host renderer at import.
ProgressiveRenderTokenBottleneck = JointTokenSemanticCompressor


__all__ = [
    "JointAnalysisTransform",
    "JointSynthesisTransform",
    "JointTokenSemanticCompressor",
    "ProgressiveRenderTokenBottleneck",
]
