# swiglu — Summary

**Category:** feedforward  
**DSL spec:** [`mechanics/swiglu.neuro`](../../mechanics/swiglu.neuro)

## Overview

Swish-gated FFN: one projection gates another via SiLU before the down-projection. The de-facto default FFN in LLaMA, PaLM, Mistral and most 2023+ LMs; consistently lowers perplexity vs ReLU/GELU dense FFNs at matched parameter count.

## Equation

`g = x W_gate`, `u = x W_up`  
`SwiGLU(x) = (Swish(g) ⊙ u) W_down`  
where `Swish(z) = z · sigmoid(z)`.  
Hidden dim convention: `h = (2/3) · 4d ≈ 2.6667 · d` → `ff_mult = 2.6667` for parameter parity with a 4d dense FFN.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ff_mult` | 2.6667 | Hidden width as a multiple of d_model; 2/3·4 keeps params ≈ a dense 4d FFN |
| `bias` | false | Add bias terms to the three linear projections (LLaMA/PaLM omit them) |

## When to Use / When NOT to Use

**Use when:** building a modern decoder-only transformer trunk; strict quality upgrade over ReLU/GELU dense FFN at matched parameters; the third projection matrix cost is acceptable.

**Avoid when:** extremely parameter-constrained models; architectures requiring a strictly monotone activation; drop-in replacement of a pre-trained 2-matrix FFN without re-training.

## References

- Shazeer (2020) GLU Variants Improve Transformer. arXiv 2002.05202
- Touvron et al. (2023) LLaMA: Open and Efficient Foundation Language Models. arXiv 2302.13971
- Chowdhery et al. (2022) PaLM: Scaling Language Modeling with Pathways. arXiv 2204.02311
