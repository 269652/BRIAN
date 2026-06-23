# qk_norm — Summary

**Category:** attention  
**DSL spec:** [`mechanics/qk_norm.neuro`](../../mechanics/qk_norm.neuro)

## Overview

Normalize Q and K (per-head) before the dot product so attention logits cannot blow up during training. A learned per-head gain restores controllable temperature while bounding magnitude growth. Near-default in modern large-scale recipes (Gemma 2, Chameleon, ViT-22B).

## Equation

With `norm_type = "rmsnorm"` (default):  
`Q̂ = RMSNorm(Q) · g_q`, `K̂ = RMSNorm(K) · g_k`  
With `norm_type = "l2"`:  
`Q̂ = Q / (‖Q‖₂ + eps)`, `K̂ = K / (‖K‖₂ + eps)`  
Then `A = softmax(Q̂ K̂ᵀ / √d)`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eps` | 1e-6 | Numerical floor in the normalization denominator |
| `norm_type` | `"rmsnorm"` | Normalizer applied to Q and K: `"rmsnorm"` or `"l2"` |

## When to Use / When NOT to Use

**Use when:** training large or deep transformers where attention logits grow unbounded; loss spikes traced to exploding max-attention-logits; half-precision training; scaling to ViT-22B-like regimes.

**Avoid when:** tiny models that already train stably; fine-tuning a pre-trained MHA checkpoint where changed logit temperature matters for ablation parity.

## References

- Henry, Dachapally, Pawar, Chen (2020) Query-Key Normalization for Transformers. Findings of EMNLP
- Dehghani, Djolonga, Mustafa et al. (2023) Scaling Vision Transformers to 22 Billion Parameters. arXiv 2302.05442
- Team Gemma (2024) Gemma 2: Improving Open Language Models at a Practical Size. arXiv 2408.00118
- Chameleon Team (2024) Chameleon: Mixed-Modal Early-Fusion Foundation Models. arXiv 2405.09818
