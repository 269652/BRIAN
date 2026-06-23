# geglu — Summary

**Category:** feedforward  
**DSL spec:** [`mechanics/geglu.neuro`](../../mechanics/geglu.neuro)

## Overview

GELU-gated FFN: one projection is GELU-activated and gates another before down-projection. The default FFN in the Gemma model family; near-equivalent in quality to SwiGLU.

## Equation

`g = x W_gate`, `u = x W_up`  
`GeGLU(x) = (GELU(g) ⊙ u) W_down`  
where `GELU(z) = z · Φ(z)` (standard-normal CDF). Hidden dim `h = (2/3)·4d ≈ 2.6667·d` for parameter matching.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ff_mult` | 2.6667 | Hidden width as a multiple of d_model; 2/3·4 keeps params ≈ a dense 4d FFN |
| `bias` | false | Add bias terms to the three linear projections |

## When to Use / When NOT to Use

**Use when:** a gated FFN with GELU lineage is preferred (Gemma, T5 ablations); upgrading a dense GELU FFN at matched parameters; the rest of the stack already uses GELU.

**Avoid when:** extremely parameter-constrained models where the third matrix is costly; quantisation schemes requiring piecewise-linear activation; drop-in replacement of a pre-trained 2-matrix FFN.

## References

- Shazeer (2020) GLU Variants Improve Transformer. arXiv 2002.05202
- Gemma Team (2024) Gemma: Open Models Based on Gemini Research and Technology. arXiv 2403.08295
- Hendrycks & Gimpel (2016) Gaussian Error Linear Units (GELUs). arXiv 1606.08415
