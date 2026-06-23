# nope — Summary

**Category:** position  
**DSL spec:** [`mechanics/nope.neuro`](../../mechanics/nope.neuro)

## Overview

No explicit position signal; the causal mask alone lets the network infer token order. The count of visible keys grows monotonically with position, providing an implicit clock even without a parametric position encoding.

## Equation

`q'_m = q_m`, `k'_n = k_n` (no modification).  
`scores_{i,j} = (q_i · k_j) / √d_k`, `A = softmax(mask_causal(scores))`.  
Order recovery relies entirely on the asymmetric causal mask.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | true | When true, no position signal is injected. When false, acts as a no-op pass-through |

## When to Use / When NOT to Use

**Use when:** decoder-only (causal) model where the mask already encodes order; length generalization on algorithmic/synthetic tasks where explicit position schemes overfit; cleanest baseline before adding RoPE/ALiBi.

**Avoid when:** bidirectional / encoder models (no causal mask = no order signal); very large-context decoders where RoPE measurably outperforms implicit order recovery; tasks requiring exact absolute index arithmetic.

## References

- Kazemnejad, Padhi, Natesan Ramamurthy, Das, Reddy (2023) The Impact of Positional Encoding on Length Generalization in Transformers. arXiv 2305.19466
- Haviv, Ram, Press, Izsak, Levy (2022) Transformer Language Models without Positional Encodings Still Learn Positional Information. arXiv 2203.16634
