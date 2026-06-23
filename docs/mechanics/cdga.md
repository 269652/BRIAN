# cdga — Cross-Distribution Gradient Alignment

**Category:** regularizer  
**Implementation:** `neuroslm.regularizers.CDGARegularizer`  
**DSL spec:** [`mechanics/cdga.neuro`](../../mechanics/cdga.neuro)

## Overview

CDGA (gradient surgery) projects out the component of the training gradient that conflicts with an OOD anchor gradient. The conflict coefficient c measures how much the training gradient would hurt the anchor; the aligned gradient removes exactly that projection. Applied as a gradient modifier rather than an auxiliary loss.

## Equation

```
g_train  = ∇_θ L(x_train; θ)
g_anchor = ∇_θ L(x_anchor; θ)   # OOD anchor batch
c = max(0, −⟨g_train, g_anchor⟩ / ‖g_anchor‖²)  # conflict only when negative
g_aligned = g_train − α·c·g_anchor
θ ← θ − lr·g_aligned
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha_max` | `1.0` | Maximum projection strength (1.0 = full surgery) |
| `warmup_steps` | `2000` | Steps to ramp α from 0 to alpha_max |
| `refresh_every` | `4` | Re-sample anchor batch every N steps |

## When to Use

When you can hold an OOD anchor batch in memory and observe that the training gradient conflicts with it (c > 0 frequently). Warmup of 2000 steps is important — before warmup, both gradients are noisy and surgery misfires.

## When NOT to Use

- No anchor loader (CDGARegularizer.enabled automatically = false if anchor batch absent)
- refresh_every=1: doubles step time; use 4 or higher
- Anchor domain radically different from training: over-constrains gradient

## Properties

- **No-harm floor**: surgery only fires when c > 0 (gradient actively conflicts)
- **Gradient surgery**: no auxiliary loss; direct gradient modification
- **PCGrad analogue**: single-task version where "task 2" is the OOD distribution

## Empirical Evidence

Yu et al. (2020) PCGrad: 15% task improvement on multi-task benchmarks. CDGA is a single-task special case. Internal: not yet ablated.

## References

- Yu, T. et al. (2020) PCGrad: Gradient Surgery for Multi-Task Learning. NeurIPS
- Internal: `neuroslm/regularizers.py::CDGARegularizer`
