# RAG-SC Core Modules

This directory contains only the new, framework-specific components of **Rendering-Aware Gaussian Semantic Communication (RAG-SC)**. It intentionally excludes the original MVSplat backbone, Gaussian rasterizer, datasets, checkpoints, training/evaluation scripts, and third-party communication baselines.

## Included components

- `ragsc/prtb.py`: Progressive Render-Token Bottleneck (P-RTB). It jointly encodes feature, RGB appearance, disparity, and confidence maps into base and refinement token streams, and supports AWGN or flat Rayleigh token-channel simulation.
- `ragsc/sart.py`: Sensitivity-Aware Refinement Token selection (SART). It combines token energy with a geometry and image-edge sensitivity proxy to rank refinement tokens.
- `ragsc/tgtr.py`: Token-Guided Texture Restoration (TGTR). It reprojects token-decoded context texture into a requested target view, estimates a bounded residual, and projects it to visible Gaussian SH-DC appearance coefficients before the final rasterization.
- `ragsc/framework.py`: A host-agnostic composition layer. The host provides a Gaussian predictor and a depth-aware renderer.
- `ragsc/types.py`: Minimal Gaussian primitive container used at the RAG-SC boundary.
- `ragsc/packet.py`: Sparse refinement-token packet utilities. They make selected token positions explicit for a deployable implementation.

## Installation

```bash
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
```

## Host integration contract

RAG-SC does not include a 3D Gaussian renderer. The surrounding host model must provide:

1. `gaussian_predictor(features, appearance, disparity, confidence) -> GaussianPrimitives`.
2. `depth_renderer(gaussians, extrinsics, intrinsics) -> (rgb, depth)`, where `rgb` is `(B,V,3,H,W)` and `depth` is `(B,V,H,W)`.
3. A converter between the host Gaussian type and `GaussianPrimitives`, whose fields are `means`, `covariances`, `harmonics`, and `opacities`.

P-RTB receives feature maps of shape `(B,V,C,H,W)`, RGB context maps `(B,V,3,H,W)`, and auxiliary maps `(B,V,A,H,W)`. The legacy `point_cloud`, `extrinsics`, and `intrinsics` arguments remain in its call signature for drop-in integration, but are not consumed by the token codec itself.

## Sparse-token transport

The in-model masking path is useful for differentiable training and channel simulation, but an arbitrary SART subset cannot be decoded from token values alone. A practical link must transmit token indices, transmit a mask, or use a deterministic selection rule that both ends can reproduce. Use `pack_sparse_tokens` and `unpack_sparse_tokens` when sparse positions are sent explicitly:

```python
from ragsc import pack_sparse_tokens, unpack_sparse_tokens

packet = pack_sparse_tokens(refinement_tokens, selection.mask)
# Send packet.values and packet.indices through the chosen digital/analog link.
refinement_tokens = unpack_sparse_tokens(packet)
index_bits = packet.index_bits_per_packet
```

Accordingly, the scalar-token payload used by an analog-token experiment is a symbol-count proxy, not a complete digital bitrate. Any comparison using coded bits must include quantization, modulation/channel coding, and this index side information.

## Minimal composition

```python
from ragsc import (
    ProgressiveRenderTokenBottleneck, SensitivityAwareRefinementSelector,
    TextureRestoreCfg, TokenGuidedTextureRestorer, RAGSC,
)

prtb = ProgressiveRenderTokenBottleneck(
    in_features_dim=128, aux_channels=2, num_tokens=1024,
    bottleneck_dim=96, image_token_dim=32, strict_bottleneck=True,
    token_residual_enabled=True, token_residual_adaptive=True,
    sensitivity_token_selection=True, token_selection_mode="sart",
)
sart = SensitivityAwareRefinementSelector(num_tokens_per_view=1024)
tgtr = TokenGuidedTextureRestorer(TextureRestoreCfg(hidden_dim=32, num_blocks=4))
model = RAGSC(prtb, host_gaussian_predictor, host_depth_renderer, sart=sart, tgtr=tgtr)
```

## Verification

```bash
python -m py_compile ragsc/*.py
```

Runtime verification additionally requires a Python environment with PyTorch and einops installed. The package was kept separate from MVSplat so that its three proposed modules can be reviewed or integrated with another Gaussian-splatting host without copying the host codebase.
