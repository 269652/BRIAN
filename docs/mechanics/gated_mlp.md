# gated_mlp — Summary

**Category:** feedforward  
**DSL spec:** [`mechanics/gated_mlp.neuro`](../../mechanics/gated_mlp.neuro)

## Overview

Generic gated-linear-unit FFN with a configurable gate activation. SwiGLU, GeGLU, ReGLU, and bilinear variants are all special cases controlled by the `activation` parameter.

## Equation

`g = x W_gate`, `u = x W_up`  
`GatedMLP(x) = (σ(g) ⊙ u) W_down`  
where σ is `silu` (SwiGLU), `gelu` (GeGLU), `relu` (ReGLU), etc.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ff_mult` | 4.0 | Hidden width as a multiple of d_model; 4.0 is NOT param-matched for 3 matrices (use 2.6667 for parity) |
| `activation` | `"silu"` | Gate activation: silu\|gelu\|relu\|sigmoid\|tanh\|identity |
| `bias` | false | Add bias terms to the three linear projections |

## When to Use / When NOT to Use

**Use when:** A/B testing different gate activations from one code path; running GLU-variant ablations; the activation is an experimental knob set at compile time.

**Avoid when:** the activation is already fixed — use `swiglu` or `geglu` directly; parameter-tight budgets at ff_mult=4.0; a non-gated dense FFN is needed.

## References

- Shazeer (2020) GLU Variants Improve Transformer. arXiv 2002.05202
- Dauphin et al. (2017) Language Modeling with Gated Convolutional Networks. arXiv 1612.08083
- Touvron et al. (2023) LLaMA: Open and Efficient Foundation Language Models. arXiv 2302.13971
