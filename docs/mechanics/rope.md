# rope — Summary

**Category:** position  
**DSL spec:** [`mechanics/rope.neuro`](../../mechanics/rope.neuro)

## Overview

Encodes absolute position by rotating Q/K pairs; dot products become relative-position dependent. Parameter-free and KV-cache friendly — the de-facto default for decoder-only LMs (LLaMA, Mistral, Qwen, GPT-NeoX).

## Equation

`θ_i = rope_base^(−2i / d)` (inverse frequencies). Rotation at position m: `R(m·θ_i)` applied per 2D pair.  
Key identity: `⟨q'_m, k'_n⟩ = f(q_m, k_n, m − n)` — result depends only on the relative offset `m − n`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rope_base` | 10000 | Base θ for inverse-frequency geometric series; larger = longer effective context |
| `partial_rotary_factor` | 1.0 | Fraction of head dim to rotate; <1.0 leaves tail dims position-free (GPT-NeoX/Phi style) |

## When to Use / When NOT to Use

**Use when:** training a decoder-only LM from scratch; relative-position behaviour without an explicit bias table; parameter-free position scheme that plays well with KV-caching.

**Avoid when:** encoder/bidirectional models where learned absolute embeddings suffice; tasks needing exact absolute index recovery; abruptly changing rope_base on a model trained with a different base.

## References

- Su, Lu, Pan, Wen, Liu (2021) RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv 2104.09864
- Peng, Quesnelle, Fan, Shippole (2023) YaRN: Efficient Context Window Extension of Large Language Models. arXiv 2309.00071
