# rmsnorm — Summary

**Category:** normalization  
**DSL spec:** [`mechanics/rmsnorm.neuro`](../../mechanics/rmsnorm.neuro)

## Overview

Re-scales activations by their root-mean-square only — LayerNorm without mean subtraction or bias. The de-facto default normalizer in LLaMA, Mistral, Gemma, and most modern decoder stacks; cheaper than LayerNorm with no measured quality loss.

## Equation

`RMS(x) = √((1/d) · Σᵢ xᵢ² + eps)`  
`ŷᵢ = (xᵢ / RMS(x)) · gᵢ`  

Skips mean subtraction (μ) and the bias term (b) relative to LayerNorm; re-scaling invariance provides the stabilising effect.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eps` | 1e-6 | Numerical floor added inside the RMS square root |
| `elementwise_affine` | true | Learn a per-feature gain g. When false, RMSNorm is parameter-free |

## When to Use / When NOT to Use

**Use when:** building a modern decoder-only transformer; LayerNorm-level stability at lower compute and parameter cost; half-precision training; matching LLaMA/Mistral/Gemma-style architectures.

**Avoid when:** architectures relying specifically on mean-centering; exact reproduction of original Transformer / BERT recipes with LayerNorm + bias; learned bias is known to be load-bearing.

## References

- Zhang, Sennrich (2019) Root Mean Square Layer Normalization. arXiv 1910.07467 (NeurIPS 2019)
- Touvron et al. (2023) LLaMA: Open and Efficient Foundation Language Models. arXiv 2302.13971
