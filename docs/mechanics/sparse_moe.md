# sparse_moe — Summary

**Category:** routing  
**DSL spec:** [`mechanics/sparse_moe.neuro`](../../mechanics/sparse_moe.neuro)

## Overview

Sparsely-gated MoE: a router sends each token to its top-k experts, with a load-balancing auxiliary loss. Decouples model capacity (parameters) from per-token compute — experts scale, FLOPs do not.

## Equation

`p = softmax(x W_r)`, `idx = top_k_indices(p, k = top_k)`, `g = p[idx] / Σ p[idx]`.  
`y = Σ_{j∈idx} g_j · E_j(x)`.  
Load-balance loss: `L_aux = aux_loss_weight · n_experts · Σ_i f_i · P_i` added to the task loss.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_experts` | 8 | Number of parallel FFN experts the router can dispatch to |
| `top_k` | 2 | Experts each token is routed to (k=1 is Switch; k=2 is GShard/Mixtral) |
| `capacity_factor` | 1.25 | Slack multiplier on per-expert token capacity; >1 reduces token dropping |
| `aux_loss_weight` | 0.01 | Weight α on the load-balancing auxiliary loss |

## When to Use / When NOT to Use

**Use when:** growing model capacity without proportionally growing per-token compute; training at scale where a dense model of equivalent quality would be too expensive; systems complexity of routing and all-to-all dispatch is acceptable.

**Avoid when:** small models / single-GPU setups where routing overhead exceeds benefit; latency-critical inference where token dropping is unacceptable; deterministic balance-free forward pass is needed.

## References

- Shazeer et al. (2017) Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. arXiv 1701.06538
- Fedus, Zoph, Shazeer (2021) Switch Transformers: Scaling to Trillion Parameter Models. arXiv 2101.03961
- Jiang et al. (2024) Mixtral of Experts. arXiv 2401.04088
