# shared_expert — Summary

**Category:** routing  
**DSL spec:** [`mechanics/shared_expert.neuro`](../../mechanics/shared_expert.neuro)

## Overview

DeepSeekMoE shared-expert isolation: a few always-on experts capture common knowledge so routed experts can specialise more sharply. Every token passes through the shared path unconditionally before adding the sparse routed contribution.

## Equation

Shared path (dense, no routing): `y_shared = Σ_{m=1..n_shared} S_m(x)`  
Routed path (top-k sparse): `p = softmax(x W_r)`, `y_routed = Σ_{j∈top_k} g_j · E_j(x)`  
Combined: `y = y_shared + y_routed`

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_shared` | 1 | Number of always-on shared experts every token passes through |
| `n_routed` | 64 | Number of fine-grained routed experts the router dispatches among |
| `top_k` | 6 | Routed experts activated per token |

## When to Use / When NOT to Use

**Use when:** fine-grained MoE with routed experts redundantly relearning common patterns; guaranteed dense fallback path needed so no token is left without compute; following the DeepSeekMoE recipe (large n_routed + shared isolation + fine-grained top_k).

**Avoid when:** coarse MoE with few large experts; ultra-tight FLOP budgets where an always-on dense expert is too costly; fully conditional compute is required with no dense path.

## References

- Dai et al. (2024) DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models. arXiv 2401.06066
- DeepSeek-AI (2024) DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv 2405.04434
- Fedus, Zoph, Shazeer (2021) Switch Transformers. arXiv 2101.03961
