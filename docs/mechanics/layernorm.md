# layernorm — Summary

**Category:** normalization  
**DSL spec:** [`mechanics/layernorm.neuro`](../../mechanics/layernorm.neuro)

## Overview

Normalizes each token's features to zero mean and unit variance, then applies a learned gain and optional bias. The original Transformer / BERT / GPT-2 normalizer; batch-independent and identical at train and inference time.

## Equation

`μ = (1/d) · Σᵢ xᵢ`, `σ² = (1/d) · Σᵢ (xᵢ − μ)²`  
`x̂ᵢ = (xᵢ − μ) / √(σ² + eps)`, `yᵢ = gᵢ · x̂ᵢ + bᵢ`

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eps` | 1e-5 | Numerical floor added to the variance before the square root |
| `bias` | true | Learn a per-feature additive bias b. When false, only the gain g is learned |

## When to Use / When NOT to Use

**Use when:** reproducing the original Transformer, BERT, or GPT-2 exactly; mean-centering is known to matter; most widely-validated conservative choice is needed.

**Avoid when:** cost-sensitive large decoder stacks (RMSNorm is cheaper with no measured quality loss); convolutional vision backbones; mean-subtraction and bias are pure overhead.

## References

- Ba, Kiros, Hinton (2016) Layer Normalization. arXiv 1607.06450
- Xiong et al. (2020) On Layer Normalization in the Transformer Architecture. arXiv 2002.04745
- Vaswani et al. (2017) Attention Is All You Need. arXiv 1706.03762
