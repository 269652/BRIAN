# dar — Distributional Adversarial Reweighting

**Category:** regularizer  
**Implementation:** `neuroslm.regularizers.DARReweighter`  
**DSL spec:** [`mechanics/dar.neuro`](../../mechanics/dar.neuro)

## Overview

DAR trains a domain discriminator on top of the encoder via a Gradient Reversal Layer (GRL). The encoder is pushed to produce domain-invariant representations (it gets a reversed gradient from the discriminator loss). Simultaneously, per-sample weights are recomputed to up-weight minority-domain samples. The result: representations that generalise across domains, with minority samples receiving stronger gradient signal.

## Equation

```
GRL_α(x) = x forward; −α·∇ backward
z   = encoder(x)
L_d = BCE(discriminator(GRL_α(z)), domain_label)   # adversarial
w_i = exp(λ·L_ce_i · 1[minority_i])                 # reweighting
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lam` | `1.0` | Reweighting exponent |
| `hidden` | `64` | Discriminator hidden size |
| `grl_alpha` | `0.1` | GRL gradient reversal strength |
| `weight` | `0.1` | Scale on discriminator loss |

## When to Use

When training on mixed-domain data (chat + prose + code) and representations are domain-separable (the discriminator achieves high accuracy). The adversarial pressure forces the encoder to become domain-invariant.

## When NOT to Use

- Single-domain training (discriminator collapses trivially)
- grl_alpha > 1.0: reversed gradient overwhelms LM gradient
- Domains that are fundamentally different in vocabulary (code vs prose): domain-invariance may harm specialisation

## Properties

- **Adversarial training**: encoder and discriminator compete
- **GRL**: exact reversal of gradient — no separate optimiser
- **Minority reweighting**: orthogonal to the adversarial component

## Empirical Evidence

Ganin et al. (2016) DANN: domain-invariant features via GRL improve transfer accuracy by 3-15% on standard domain adaptation benchmarks.

## References

- Ganin, Y. et al. (2016) Domain-Adversarial Training of Neural Networks. JMLR
- Saito, K. et al. (2018) Maximum Classifier Discrepancy for Unsupervised Domain Adaptation. CVPR
