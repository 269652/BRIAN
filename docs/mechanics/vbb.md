# vbb — Variational Bowtie Bottleneck

**Category:** training_dynamics  
**Implementation:** `neuroslm.regularizers.VBBLoss`  
**DSL spec:** [`mechanics/vbb.neuro`](../../mechanics/vbb.neuro)

## Overview

A Friston-style variational free energy at the bowtie bottleneck. The narrowest hidden state is mapped to a reparameterised Gaussian posterior q(z|h) = N(μ,σ²I); a KL penalty pushes q toward the standard Gaussian prior p(z)=N(0,I). The information bottleneck effect prevents memorisation of training-distribution attractors. Scheduled via GIF-1 (loose during infancy, tight at maturity).

## Equation

```
μ, log_σ = split(Linear(h_bot))
z = μ + σ·ε,  ε ~ N(0,I)
KL = ½Σ_j (σ_j² + μ_j² − 1 − log σ_j²)
L_VBB = α_vbb(t) · KL     # α schedule from gif_vbb_alpha_schedule
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha_max` | `0.1` | Peak KL weight (reached at warmup end) |
| `warmup_steps` | `2000` | Steps to ramp α from 0 to alpha_max |
| `d_z` | `-1` | Bottleneck latent dim; -1 = same as d_bot |

## When to Use

At the bowtie waist of a bowtie-topology model, to add an information bottleneck. Start with alpha_max=0.1. Observe KL over training; healthy KL grows from 0 to ~10-50 nats/dim. Enable GIF-7C KL floor to prevent collapse.

## When NOT to Use

- Non-bowtie models (no clear bottleneck)
- alpha_max > 0.5: KL dominates CE before bottleneck stabilises
- VBB collapse pathology: KL can collapse from 38000 to ~100 at step ~2040 (use vbb_kl_floor to guard)

## Properties

- **Friston free energy**: F = KL(q‖p) − E[log p(x|z)]
- **IB implicit**: KL = upper bound on I(z;h)
- **GIF-1 scheduled**: loose during infancy, tight at maturity
- **Zero-init**: α_vbb(0) = 0 → clean baseline at step 0

## Empirical Evidence

Kingma & Welling (2014) VAE: competitive reconstructions with compact latent. Internal: VBB collapse at step ~2040 motivated MDRV stabilisers (free-bits, β-ceiling, PEC).

## References

- Kingma, D. & Welling, M. (2014) Auto-Encoding Variational Bayes. ICLR
- Friston, K. (2010) The free-energy principle. Nat Rev Neurosci
- Internal: `lib/gif.neuro` (gif_vbb_alpha_schedule)
