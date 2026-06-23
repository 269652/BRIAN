# mixture_of_depths — Sparse Token Routing

**Category:** routing  
**Implementation:** `neuroslm.modules.mixture_of_depths.MoDBlock`  
**DSL spec:** [`mechanics/mixture_of_depths.neuro`](../../mechanics/mixture_of_depths.neuro)

## Overview

Mixture of Depths (Raposo et al. 2024) routes a fraction C of tokens through a full expensive block (attention + MLP) while the remaining tokens skip it via a residual passthrough. A lightweight MLP router scores each token; top-C tokens are processed; all tokens are scatter-merged. Reduces FLOP count by (1−C) on the expensive block while maintaining sequence length.

## Equation

```
score_t = MLP_router(h_t)             # scalar logit per token
C       = floor(capacity_ratio · T)   # how many tokens to process
top_C   = argtopk(score, C)           # token indices
h_top   = expensive_block(h[top_C])   # full computation on C tokens
h_out   = h.scatter(top_C, h_top)     # merge: processed + passthrough
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `capacity_ratio` | `0.5` | Fraction of tokens processed per block (0.5 = 50%) |

## When to Use

When the model is compute-bottlenecked by deep attention blocks and you can tolerate some tokens not being updated at each layer. Works best when tokens are heterogeneous in importance (e.g. content vs filler tokens).

## When NOT to Use

- capacity_ratio = 1.0: all tokens processed — MoD overhead with no savings
- Very short sequences: routing overhead dominates
- Tasks requiring every token to be processed at every layer

## Properties

- **FLOP reduction**: (1−capacity_ratio) savings on the expensive block
- **Differentiable**: routing is continuous; top-k is approximated at training
- **Causal preserved**: scatter restores original sequence order

## Empirical Evidence

Raposo et al. (2024) demonstrate isoFLOP curves: MoD matches dense transformer quality at fewer FLOPs by spending compute only on salient tokens.

## References

- Raposo, D. et al. (2024) Mixture of Depths: Dynamically allocating compute in transformers. arXiv:2404.02258
