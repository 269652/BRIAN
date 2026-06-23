# gated_linear_attention — Summary

**Category:** sequence_mixer  
**DSL spec:** [`mechanics/gated_linear_attention.neuro`](../../mechanics/gated_linear_attention.neuro)

## Overview

Linear attention plus a learned per-channel forget gate, trained in a chunked parallel form. The data-dependent gate lets the model selectively forget or retain state per channel, addressing the state-saturation issue of plain linear attention.

## Equation

`α_t = σ(Linear_α(x_t)) ∈ (0,1)^{d_k}` (per-channel forget gate).  
`S_t = (α_t 𝟙ᵀ) ⊙ S_{t-1} + k_t v_tᵀ`  
`o_t = q_tᵀ S_t`

Training uses a chunkwise parallel form (intra-chunk quadratic + inter-chunk recurrence) for O(N) total work on tensor cores.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `expand_k` | 0.5 | Key projection expansion ratio; d_k = expand_k · d_model |
| `expand_v` | 1.0 | Value projection expansion ratio; d_v = expand_v · d_model |
| `chunk_size` | 64 | Sequence chunk length C for the chunkwise parallel scan |

## When to Use / When NOT to Use

**Use when:** O(N) cost is needed but plain linear attention's state saturates; per-channel content-dependent forgetting is desirable; training throughput matters.

**Avoid when:** short sequences where O(T²) is cheap; tasks needing exact arbitrary-token copy; hardware lacking the chunkwise kernel.

## References

- Yang, Wang, Shen, Panda, Kim (2023) Gated Linear Attention Transformers with Hardware-Efficient Training. arXiv 2312.06635
- Katharopoulos, Vyas, Pappas, Fleuret (2020) Transformers are RNNs. arXiv 2006.16236
