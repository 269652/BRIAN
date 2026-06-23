# hyena — Summary

**Category:** sequence_mixer  
**DSL spec:** [`mechanics/hyena.neuro`](../../mechanics/hyena.neuro)

## Overview

Subquadratic attention substitute: interleaved implicit long convolutions and elementwise gating. Implicit filters parameterized by a small MLP give global receptive field at cost independent of sequence length.

## Equation

Recurrence of N (gate, long-conv) stages:  
`z⁰ = v`; `zⁿ = x^n ⊙ (h^n ∗ z^{n-1})` for n = 1..N; `y = Linear(z^N)`.  
Each long causal conv is evaluated via FFT: `h ∗ z = IFFT(FFT(pad(h)) ⊙ FFT(pad(z)))`, giving O(N·T log T) total cost.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `order` | 2 | Number N of (gate, long-conv) stages in the Hyena recurrence |
| `filter_order` | 64 | Width of the implicit-filter MLP parameterizing the long convolution kernels |
| `short_filter_len` | 3 | Kernel width of the short causal depthwise conv applied to each projected branch |

## When to Use / When NOT to Use

**Use when:** long context with a global-receptive subquadratic token mixer; implicit parameterization decouples filter length from parameter count; convolutional inductive bias plus content gating is useful.

**Avoid when:** short sequences where FFT overhead is not amortized; strict O(1)-per-step streaming decode; tasks dominated by exact arbitrary-token associative recall.

## References

- Poli, Massaroli, Nguyen et al. (2023) Hyena Hierarchy: Towards Larger Convolutional Language Models. arXiv 2302.10866
- Romero, Kuzina, Bekkers, Tomczak, Hoogendoorn (2022) CKConv: Continuous Kernel Convolution. arXiv 2102.02611
