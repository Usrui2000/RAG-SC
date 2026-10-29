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
