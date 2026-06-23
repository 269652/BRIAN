# cosine_lm_head — Norm-Invariant Output Projection

**Category:** training_dynamics  
**Implementation:** `neuroslm.dsl.nn_ops.cosine_lm_head`  
**DSL spec:** [`mechanics/cosine_lm_head.neuro`](../../mechanics/cosine_lm_head.neuro)

## Overview

Replaces the standard linear LM head (`z_i = h·W_i`) with a cosine-similarity projection (`z_i = τ·cos(h, W_i)`). Eliminates representation magnitude as a degree of freedom: confidence is determined purely by angular proximity. Logits are bounded in [−τ, +τ]; τ is a learnable scalar initialised to √d_model. Part of GIF-6.

**Root cause addressed**: standard heads can encode domain-specific confidence in ‖h‖ and ‖W_i‖, making OOD tokens structurally under-confident regardless of angular proximity.

## Equation

```
ĥ   = h / ‖h‖
Ŵ_i = W_i / ‖W_i‖
z_i = τ · (ĥ · Ŵ_i)     τ_init = √d_model
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_init` | `-1.0` | Initial temperature; -1 = √d_model |
| `learnable_tau` | `true` | Whether τ adapts during training |

## When to Use

When you observe norm-mediated confidence asymmetry: in-distribution tokens grow increasingly confident while OOD tokens stay under-confident even as they have high angular similarity. Most powerful when h is not already fixed-norm.

## When NOT to Use

- h is already LayerNorm'd before the head (cosine head adds no invariance)
- learnable_tau=false with small τ: model cannot be confident on rare tokens

## Properties

- **Norm-invariant**: logits depend only on angle, not magnitude
- **Bounded**: z_i ∈ [−τ, +τ] always
- **Single new parameter**: only τ added vs linear head
- **Drop-in**: replaces nn.Linear without architecture change

## Empirical Evidence

Liu et al. (2022): cosine head consistently matches or outperforms linear head on downstream generalisation. Internal: wired as GIF-6; isolated ablation pending.

## References

- Liu, Jia, Bhatt (2022) Same Pre-training Loss, Better Downstream. arXiv:2205.00214
- Internal: `lib/gif.neuro` (gif_cosine_lm_head equation)
