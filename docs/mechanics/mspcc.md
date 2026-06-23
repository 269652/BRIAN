# mspcc — Multi-Scale Predictive Coding Cascade

**Category:** training_dynamics  
**Implementation:** `neuroslm.emergent.mspcc.mspcc_loss`  
**DSL spec:** [`mechanics/mspcc.neuro`](../../mechanics/mspcc.neuro)

## Overview

Generalises single-waist VBB into a per-layer cascade. For each adjacent layer pair (ℓ, ℓ+1) in the trunk, one MDRV-VBB term is contributed: prediction residual r_ℓ, KL divergence KL_ℓ, and Posterior Entropy Commitment PEC_ℓ (anti-collapse). Layer weights λ_ℓ decay geometrically from the deepest pair (bowtie waist dominates) to the shallowest. No new hyperparameters beyond base_weight and layer_weight_decay.

## Equation

```
r_ℓ    = ‖h_{ℓ+1} − W_ℓ·h_ℓ‖²
KL_ℓ   = ½·Σ_d (σ_d² + μ_d² − 1 − log σ_d²)
PEC_ℓ  = −η·½·E[log σ_ℓ²]            # → +∞ as σ→0: collapse Lyapunov-unstable
L_ℓ    = β_ℓ·r_ℓ − log β_ℓ + α·KL_ℓ + PEC_ℓ
λ_ℓ    = base_weight · decay^{(L-2)-ℓ}
L_MSPCC = Σ_{ℓ=0}^{L-2} λ_ℓ·L_ℓ
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_weight` | `0.02` | λ₀ for the deepest layer pair |
| `layer_weight_decay` | `0.5` | Geometric decay per layer away from deepest |

## When to Use

As a generalisation of VBB when you want IB pressure at every layer, not just the bottleneck. Enable alongside or as a replacement for VBB.

## When NOT to Use

- Shallow models (L < 4): too few terms
- base_weight > 0.1: shallow terms dominate CE

## Properties

- **Geometric cascade**: waist pair dominates
- **MDRV stabilisers**: free-bits, β-ceiling, PEC applied per pair
- **Additive with VBB**: composes cleanly
- **Hierarchical IB**: each layer pair captures different abstraction level

## References

- Sønderby et al. (2016) Ladder VAE. NeurIPS
- Felleman & Van Essen (1991) Distributed hierarchical processing. Cereb Cortex
- Internal: `neuroslm/emergent/mspcc.py`
