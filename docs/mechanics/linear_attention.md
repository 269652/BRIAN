# linear_attention — Summary

**Category:** sequence_mixer  
**DSL spec:** [`mechanics/linear_attention.neuro`](../../mechanics/linear_attention.neuro)

## Overview

Attention without softmax: a kernel feature map φ replaces the exponential, enabling the KV sum to be hoisted out of the per-query loop. This gives O(N) total complexity and a fixed-size recurrent state for constant-memory decoding.

## Equation

Replace `exp(q·k)` with `φ(q)·φ(k)`. Define running state `S_t = S_{t-1} + φ(k_t) v_tᵀ` and normalizer `z_t = z_{t-1} + φ(k_t)`.  
`o_t = (φ(q_t)ᵀ S_t) / (φ(q_t)ᵀ z_t + eps)`

Default feature map: `φ(x) = elu(x) + 1 ≥ 0`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `feature_map` | `"elu"` | Positive feature map φ; `"elu"` uses φ(x)=elu(x)+1. Also `"relu"`, `"softmax"`, `"identity"` |
| `eps` | 1e-6 | Stabilizer added to the denominator to avoid divide-by-zero |

## When to Use / When NOT to Use

**Use when:** sequence length is large and the O(T²) matrix dominates; constant-memory autoregressive generation is needed; a simple attention-shaped drop-in is preferred over a full SSM.

**Avoid when:** tasks dominated by precise long-range copy / associative recall; short sequences where O(T²) is already cheap; exact softmax distribution is required.

## References

- Katharopoulos, Vyas, Pappas, Fleuret (2020) Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. arXiv 2006.16236
- Choromanski et al. (2021) Rethinking Attention with Performers. arXiv 2009.14794
