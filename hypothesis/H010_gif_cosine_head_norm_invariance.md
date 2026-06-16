---
code_refs:
  - neuroslm/dsl/nn_ops.py (cosine_lm_head)
  - neuroslm/dsl/nn_lang.py (DSLLanguageModel, DSLLanguageCortex — cosine_head branches)
  - neuroslm/dsl/training_config.py (cosine_head: bool)
  - neuroslm/train_dsl.py (build_dsl_lm_harness cosine_head passthrough)
  - lib/gif.neuro (gif_cosine_lm_head)
created_at: "2026-06-16T23:00:00Z"
id: H010
proof_path: null
proof_status: missing
references:
  - docs/OOD_MECHANISMS.md
  - Gidaris & Komodakis (2018) — Dynamic Few-Shot Visual Learning without Forgetting
  - Chen et al. (2020) — A Simple Framework for Contrastive Learning (SimCLR)
  - Wang et al. (2017) — NormFace: L₂ Hypersphere Embedding for Face Verification
status: stated
tags: [gif, ood, cosine-head, norm-invariance, gap-ratio, structural]
test_refs:
  - tests/dsl/test_cosine_head.py
theorem_name: Brian.CosineHeadEliminatesMagnitudeBias
title: "GIF-6: Cosine LM head eliminates norm-mediated confidence asymmetry"
updated_at: "2026-06-17T00:00:00Z"
---

## H010 — GIF-6: Cosine LM head eliminates norm-mediated confidence asymmetry

### Statement

Replace the standard linear LM head $z_i = h^\top w_i$ with a cosine head:

$$z_i = \tau \cdot \frac{h}{\lVert h \rVert} \cdot \frac{w_i}{\lVert w_i \rVert}$$

where $\tau$ is a learnable temperature initialised at $\sqrt{d_\text{model}}$.

**Claim:** The cosine head bounds logits to $[-\tau, +\tau]$, forcing the
model to map from $\mathbb{S}^{d-1} \to \Delta^{V-1}$ and eliminating
all magnitude degrees of freedom. This:

1. Reduces gap ratio $G(t) = \text{PPL}\_\text{ood} / \text{PPL}\_\text{train}$
   to $G \le 2.0$ by step 10 000 (vs $G \approx 2.8$ with linear head).
2. Achieves OOD PPL ≤ 200 by step 8 000.
3. Does not degrade train PPL by more than 5% at matched step count.

### Root cause analysis

With RMSNorm + linear head, the model has two magnitude degrees of freedom:

- $\lVert h \rVert$: hidden-state norm (proportional to input familiarity)
- $\lVert w_i \rVert$: token embedding norm (proportional to token frequency)

These allow logits $\lVert h \rVert \cdot \lVert w_i \rVert \cdot \cos\theta$
to be artificially inflated for in-distribution patterns. The model encodes
**domain-specific confidence** in these norms — FineWeb-Edu patterns get
high-norm representations, out-of-distribution contexts get low-norm.

The cosine head projects both $h$ and $w_i$ to the unit hypersphere,
so the only remaining degree of freedom is the angle $\theta$ between
hidden state and token embedding. Temperature $\tau$ controls sharpness
globally, not per-domain.

### Mechanism details

```
Logit computation (linear):    z = h @ W.T          → unbounded
Logit computation (cosine):    z = τ * F.normalize(h) @ F.normalize(W).T → [-τ, τ]
```

**Gradient flow**: The temperature $\tau$ is a scalar `nn.Parameter` with
`requires_grad=True`. It learns the entropy of the output distribution
without the ability to modulate it per-input.

**DSL surface**: Exposed as `cosine_head: true` in arch.neuro. The DSL
equation `gif_cosine_lm_head(h, W, τ)` is reusable across architectures.

### Ablation protocol

| Variant | Head | τ init | Expected |
|---------|------|--------|----------|
| Baseline | linear | — | Gap ~2.8, OOD PPL ~180 at 10k |
| Cosine (default) | cosine | √d | Gap ≤2.0, OOD PPL ≤200 at 8k |
| Cosine (low τ) | cosine | √(d/4) | Softer logits, possibly under-fit |
| Cosine (high τ) | cosine | √(4d) | Sharper logits, may re-create gap |
| Cosine (fixed τ) | cosine | √d (frozen) | No temperature adaptation |
| Cosine + no GIF-4 | cosine | √d | Structural fix alone, no label smoothing |
| Cosine + no GIF-5 | cosine | √d | Structural fix alone, no diversity |

### Key commits

- `d3cc00c` — feat(gif-6): cosine LM head — norm-invariant logits
- `7523d59` — dsl: add GIF-6 equation to lib/gif.neuro

### Config

```neuro
training: {
  cosine_head: true
}
```

### Empirical evidence

- Instance 41248189 (commit `e689d61`): Deployed with full GIF stack
  (GIF-1 through GIF-6). Still booting at time of hypothesis creation.
  This is the first instance with cosine head active.

- **Pre-GIF-6 trajectory** (instance 41238415, gap driven by linear head):
  - Step 3000: train PPL 97.5, OOD 215, gap=2.20
  - Step 5000: train PPL 77.9, OOD 189, gap=2.43
  - Step 6500: train PPL 66.0, OOD 177, gap=2.68
  - Trend: gap monotonically increasing despite GIF-1 through GIF-5.

- **Expected GIF-6 trajectory**: Gap should plateau or decrease as
  the model cannot encode domain confidence in norms. Temperature τ
  will converge to a value that balances sharpness across domains.

### Theoretical justification

The linear head's output space is $\mathbb{R}^V$ — unconstrained.
The cosine head's output space is $\tau \cdot \mathbb{S}^{d-1} \to [-\tau,\tau]^V$
— bounded. Since cross-entropy only uses log-softmax, what matters is
**relative** logit ordering, not absolute magnitude. The cosine head
preserves ordering while eliminating the magnitude channel through
which domain-specific confidence leaks.

This is analogous to L₂-normalised embeddings in metric learning
(FaceNet, ArcFace), where normalisation forces the model to learn
angular features rather than magnitude-dependent ones.
