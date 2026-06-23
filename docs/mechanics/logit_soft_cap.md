# logit_soft_cap — Summary

**Category:** attention  
**DSL spec:** [`mechanics/logit_soft_cap.neuro`](../../mechanics/logit_soft_cap.neuro)

## Overview

Tanh soft-cap that smoothly bounds attention scores and final logits to ±cap — taming spikes without hard clipping. Unlike hard clipping, the tanh map has a positive gradient everywhere so saturated entries can still recover during training.

## Equation

`soft_cap(x, c) = c · tanh(x / c)`  
Applied to pre-softmax attention scores: `S = soft_cap(Q Kᵀ / √d, attn_cap)`, then `A = softmax(S)`.  
Applied to final vocabulary logits: `z = soft_cap(W_O h, final_logit_cap)`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `attn_cap` | 50.0 | Soft-cap applied to pre-softmax attention scores (Gemma 2 default) |
| `final_logit_cap` | 30.0 | Soft-cap applied to final vocabulary logits before the loss (Gemma 2 default) |

## When to Use / When NOT to Use

**Use when:** training exhibits loss spikes from runaway logits; final-layer logits risk softmax/cross-entropy overflow; reproducing the Gemma 2 recipe.

**Avoid when:** the model already trains stably with no logit blow-up; fused attention kernels cannot apply a pre-softmax score cap; exactly-calibrated raw logits are needed.

## References

- Team Gemma (2024) Gemma 2: Improving Open Language Models at a Practical Size. arXiv 2408.00118
- Bello, Zoph, Vaswani, Shlens, Le (2017) Neural Combinatorial Optimization with Reinforcement Learning. arXiv 1611.09940
