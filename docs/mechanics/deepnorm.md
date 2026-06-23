# deepnorm — Summary

**Category:** normalization  
**DSL spec:** [`mechanics/deepnorm.neuro`](../../mechanics/deepnorm.neuro)

## Overview

Up-scales the residual branch by α and down-scales sublayer init by β so post-norm transformers train at extreme depth. Enables stable training up to ~1000 layers without the instability that normally afflicts post-LayerNorm architectures.

## Equation

`x_{l+1} = LayerNorm(α · x_l + Sublayer(x_l))`  
Sublayer projection weights are down-scaled at init: `W ← β · W_init`.  
Decoder-only formulas: `α = (2M)^(1/4)`, `β = (8M)^(−1/4)` where M is the layer count.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 1.0 | Residual up-scale α (DEPTH-DERIVED). Default 1.0 = no scaling (placeholder) |
| `beta` | 1.0 | Init down-scale β (DEPTH-DERIVED). Default 1.0 = no scaling (placeholder) |

## When to Use / When NOT to Use

**Use when:** building very deep transformers (dozens to ~1000 layers); you want post-norm quality without training instability; scaling depth rather than width.

**Avoid when:** shallow/moderate-depth models (≤~24 layers); pre-norm architectures; leaving α/β at placeholder defaults.

## References

- Wang, Ma, Dong, Huang, Zhang, Wei (2022) DeepNet: Scaling Transformers to 1,000 Layers. arXiv 2203.00555
- Liu et al. (2020) Understanding the Difficulty of Training Transformers (Admin init). arXiv 2004.08249
