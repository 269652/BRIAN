# expert_choice_routing — Summary

**Category:** routing  
**DSL spec:** [`mechanics/expert_choice_routing.neuro`](../../mechanics/expert_choice_routing.neuro)

## Overview

Inverted MoE routing: each expert picks its top-capacity tokens rather than tokens choosing experts. This gives exact load balance by construction with no auxiliary loss needed.

## Equation

`S = softmax(x W_r)ᵀ` (n_experts × B·T). Each expert selects its top-C tokens:  
`C = capacity_factor · (B·T / n_experts)`.  
Output: `y_t = Σ_{i : t ∈ I_i} G_{i,t} · E_i(x_t)`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_experts` | 8 | Number of parallel FFN experts |
| `capacity_factor` | 2.0 | Tokens kept per expert as C = capacity_factor · (B·T / n_experts) |

## When to Use / When NOT to Use

**Use when:** load imbalance and auxiliary balancing loss are the main pain points of token-choice MoE; every expert should be fully and equally utilised; variable per-token compute is acceptable.

**Avoid when:** strictly causal autoregressive decoding without chunking; every token must receive exactly the same compute; some tokens receiving zero experts is unacceptable without a shared fallback.

## References

- Zhou et al. (2022) Mixture-of-Experts with Expert Choice Routing. arXiv 2202.09368
- Fedus, Zoph, Shazeer (2021) Switch Transformers. arXiv 2101.03961
