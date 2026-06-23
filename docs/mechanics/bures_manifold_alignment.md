# bures_manifold_alignment — Sliced W₂ Erank Guard

**Category:** training_dynamics  
**Implementation:** `neuroslm.harness.BrainHarness._compute_bma_loss`  
**DSL spec:** [`mechanics/bures_manifold_alignment.neuro`](../../mechanics/bures_manifold_alignment.neuro)

## Overview

Computes sliced Wasserstein₂ between the trunk's hidden-state distribution and the frozen expert ensemble's hidden-state distribution. Gradient flows into the trunk only (expert is detached). The trunk is pushed to match the expert's variance profile — preventing erank collapse while the expert provides the reference geometry.

**Critical**: ramp_end=1000 (not 3000) — erank collapse begins at step ~300; ramp must reach full weight before that.

## Equation

```
W₂²(H_trunk, H_expert) ≈ (1/J)·Σⱼ W₁²(H_trunk·vⱼ, H_expert.detach()·vⱼ)
vⱼ ~ Uniform(S^{d-1})   (random projections)
L_BMA = w_bma(t) · W₂²(H_trunk, H_expert.detach())
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bma_weight` | `0.05` | Peak W₂ loss weight |
| `bma_n_projections` | `64` | J = random projections for sliced W₂ |
| `bma_ramp_start` | `0` | Ramp start (0 = immediate) |
| `bma_ramp_end` | `1000` | Ramp end (must be < step 300 for erank protection) |

## When to Use

When a multi-cortex ensemble is present and erank collapses early (observed: 53→7 by step 300). BMA fires immediately and reaches full weight before collapse.

## When NOT to Use

- No frozen expert ensemble (no reference distribution)
- bma_weight > 0.1: W₂ loss dominates CE

## Properties

- **Expert detached**: only trunk gets gradient
- **Sliced W₂**: O(N·J) cost; J=64 gives accurate approximation
- **Erank guard**: prevents trunk representation from collapsing

## Empirical Evidence

Internal: erank 53→7 by step 300 without BMA. With ramp_end=1000: collapse slowed. Rabin et al. (2011) sliced W₂.

## References

- Kolouri et al. (2016) Sliced and Radon Wasserstein Barycenters. CVPR
- Internal: `neuroslm/harness.py::_compute_bma_loss`
