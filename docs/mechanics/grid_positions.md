# grid_positions — Multi-Scale Grid-Cell Positions

**Category:** attention  
**Implementation:** `neuroslm.dsl.nn_ops.grid_cell_positions`  
**DSL spec:** [`mechanics/grid_positions.neuro`](../../mechanics/grid_positions.neuro)

## Overview

Multi-scale sinusoidal positional encoding at K scales spaced by the golden ratio φ≈1.618. Each scale k encodes position t as (sin(2πt/P_k), cos(2πt/P_k)) with P_k = base_period·φᵏ. A zero-init projection maps 2K features to d_model. The φ-ratio ensures aperiodic tiling (algebraically irrational periods) which gives provable out-of-context-length extrapolation (H16 hypothesis). Biologically motivated by entorhinal grid cells (Sargolini 2006).

## Equation

```
P_k = base_period · φᵏ            k = 0..K-1
f_t = concat([sin(2πt/Pk), cos(2πt/Pk)] for k in range(K))   ∈ ℝ^{2K}
pos_emb_t = W_grid · f_t           W_grid: zero-init → step-0 pos_emb=0
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_scales` | `4` | K scales; output = 2K features |
| `scale_ratio` | `1.6180339887` | φ (golden ratio); other irrationals also valid |
| `base_period` | `16.0` | Period of finest scale |

## When to Use

When you want positional encoding that generalises beyond training context length, or when you want a biologically-motivated position signal that composes with RoPE.

## When NOT to Use

- Position-agnostic tasks (sets, bags)
- base_period >> max_context_length (finest scale is DC)

## Properties

- **ReZero contract**: W_grid zero-init → pos_emb=0 at step 0
- **φ-ratio**: algebraically irrational → aperiodic → OOD length extrapolation
- **Biologically grounded**: entorhinal grid cells use multi-scale tiling
- **2K compact**: K=4 → 8 features → linear to d_model

## Empirical Evidence

Sargolini et al. (2006): EC grid cells tile space at multiple φ-ratio scales. H16 hypothesis: enabled in master arch (n_scales=4, scale_ratio=φ); extrapolation ablation pending.

## References

- Sargolini, F. et al. (2006) Conjunctive Representation of Position, Direction, Speed. Science 312
- Stensola, H. et al. (2012) The Entorhinal Grid Map is Discretized. Nature 492
