# differential_attention — Noise-Cancelling Attention via Differential Softmax

**Category:** attention  
**Implementation:** `neuroslm.modules.differential_attention.DifferentialAttention`  
**DSL spec:** [`mechanics/differential_attention.neuro`](../../mechanics/differential_attention.neuro)

## Overview

Differential Attention (Ye et al. 2024) replaces the single softmax attention map with the difference of two softmax maps, A₁ − λA₂. The two maps attend to the same content using separate Q/K projections; the subtraction cancels common-mode noise (uniform background attention) while amplifying signal (structured attention peaks). The mechanism inherits causal masking and is GQA-compatible.

## Equation

```
[Q₁, Q₂] = split(W_Q · x)    [K₁, K₂] = split(W_K · x)
A₁ = softmax(Q₁Kᵀ₁/√(d/2))    A₂ = softmax(Q₂Kᵀ₂/√(d/2))
Output = (A₁ − λ·A₂) @ V     per-head RMSNorm applied before projection
λ = softplus(λ_init)  (learnable, lower-bounded away from 0)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_init` | `0.8` | Initial λ in softplus units; controls differential cancellation strength |

## When to Use

When attention maps contain spurious uniform-background tokens (common with long contexts or noisy corpora). The differential cancellation removes content-independent attention mass, sharpening the effective attention pattern. Particularly effective when attention entropy is high.

## When NOT to Use

- Very short contexts (T < 16): insufficient tokens for the two maps to diverge
- λ ≫ 1: over-cancellation — A₁ and A₂ become nearly identical, output ≈ 0
- Architectures that already use sparse/local attention (redundant noise cancellation)

## Properties

- **Noise cancellation**: subtracts common-mode uniform attention
- **Bounded λ**: softplus ensures λ > 0 always; prevents sign flip
- **GQA compatible**: can share K/V across heads
- **Causal preserved**: causal mask applied before both softmaxes

## Empirical Evidence

Ye, Dongrui, Chao Luo, et al. (2024) "Differential Attention." arXiv:2410.05258. Consistent improvements on long-context tasks (≥2k tokens) across multiple model scales.

## References

- Ye et al. (2024) Differential Attention. arXiv:2410.05258
- Internal: `neuroslm/modules/differential_attention.py`
