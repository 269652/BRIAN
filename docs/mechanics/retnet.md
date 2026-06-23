# retnet — Summary

**Category:** sequence_mixer  
**DSL spec:** [`mechanics/retnet.neuro`](../../mechanics/retnet.neuro)

## Overview

Retention with parallel / recurrent / chunkwise duality: because there is no softmax, the three forms compute identical outputs from the same weights. Fixed exponential decay γ per head gives multi-scale memory without a learned gate.

## Equation

Parallel: `Retention(X) = (Q Kᵀ ⊙ D) V` where `D_{n,m} = γ^{n-m}` (causal decay mask).  
Recurrent: `S_n = γ · S_{n-1} + k_nᵀ v_n`, `o_n = q_n S_n`.  
Chunkwise: intra-chunk parallel + inter-chunk recurrence carry → O(N) overall.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_heads` | 8 | Number of retention heads; each gets a distinct decay γ_h |
| `decay_min` | 0.96875 | Smallest per-head decay (fastest forgetting); 1 − 2^−5 |
| `decay_max` | 0.998 | Largest per-head decay (longest memory) |

## When to Use / When NOT to Use

**Use when:** attention-quality parallel training AND O(1) recurrent decoding from the same weights with no approximation gap; long-context training throughput matters; simple fixed-decay memory is preferable to a learned gate.

**Avoid when:** content-dependent forgetting is needed (γ is a fixed geometric schedule); short sequences with no chunkwise payoff; precise arbitrary-token copy beyond fixed-state capacity.

## References

- Sun, Dong, Huang et al. (2023) Retentive Network: A Successor to Transformer for Large Language Models. arXiv 2307.08621
- Sun, Dong, Patra et al. (2022) A Length-Extrapolatable Transformer (xPos). arXiv 2212.10554
