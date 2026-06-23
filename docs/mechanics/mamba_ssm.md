# mamba_ssm — Summary

**Category:** sequence_mixer  
**DSL spec:** [`mechanics/mamba_ssm.neuro`](../../mechanics/mamba_ssm.neuro)

## Overview

Selective SSM: input-dependent A, B, C parameters run as an O(N) hardware-aware parallel scan. Unlike LTI SSMs, the selection mechanism lets the model context-dependently keep or forget state based on content.

## Equation

Discretized recurrence: `h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_t`, `y_t = Σ_n C_t,n · h_t,n + D ⊙ x_t`.  
`Δ`, `B`, `C` are functions of the input. `Ā = exp(Δ ⊙ A)`, `B̄ = Δ ⊙ B` (ZOH discretization).  
Training uses an associative parallel scan kept in SRAM (fused kernel).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_state` | 16 | SSM hidden state size N per channel |
| `d_conv` | 4 | Kernel width of the causal depthwise conv applied before the selective scan |
| `expand` | 2 | Inner expansion factor; d_inner = expand · d_model |
| `dt_rank` | `"auto"` | Low-rank dim of the Δ projection; `"auto"` = ceil(d_model / 16) |

## When to Use / When NOT to Use

**Use when:** long sequences (≥ 2k–8k) where quadratic attention is the bottleneck; constant-memory autoregressive inference; strong recurrent/streaming structure (audio, DNA, long-document LM).

**Avoid when:** very short sequences where scan overhead is not amortized; tasks needing exact long-range associative recall; hardware without a fused selective-scan kernel.

## References

- Gu, Dao (2023) Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv 2312.00752
- Gu, Goel, Ré (2021) Efficiently Modeling Long Sequences with Structured State Spaces (S4). arXiv 2111.00396
- Dao, Gu (2024) Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2). arXiv 2405.21060
