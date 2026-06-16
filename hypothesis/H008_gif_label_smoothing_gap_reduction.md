---
code_refs:
  - neuroslm/emergent/gif.py (GIFController.label_smoothing)
  - neuroslm/harness.py (_compute_loss_from_logits, GIF-4 override)
  - lib/gif.neuro (gif_label_smoothing)
created_at: "2026-06-16T20:00:00Z"
id: H008
proof_path: null
proof_status: missing
references:
  - docs/OOD_MECHANISMS.md
  - Szegedy et al. (2016) — label smoothing
  - Müller et al. (2019) — When Does Label Smoothing Help?
status: stated
tags: [gif, ood, label-smoothing, confidence, simplex]
test_refs:
  - tests/dsl/test_gif_config.py::TestGapDrivenLabelSmoothing
theorem_name: Brian.GIFLabelSmoothingReducesGap
title: "GIF-4: Gap-driven label smoothing reduces OOD PPL"
updated_at: "2026-06-17T00:00:00Z"
---

## H008 — GIF-4: Gap-driven label smoothing reduces OOD PPL

### Statement

Let $\varepsilon(t)$ be the label smoothing coefficient at step $t$,
driven by the gap ratio:

$$\varepsilon(t) = \varepsilon_0 \cdot \text{clamp}\!\left(\frac{G(t)}{G^*} - 1,\; 0,\; 1\right)$$

**Claim:** Adding GIF-4 label smoothing to GIF-1/2/3 reduces the OOD PPL
by ≥15% at step 5000 compared to GIF-1/2/3 alone, and the gap ratio
$G(t)$ remains below $2\,G^*$ throughout training.

### Mechanism

The one-hot CE target lives at a vertex of the probability simplex.
Label smoothing moves the target toward the barycentre (uniform) by $\varepsilon$:

$$q'_i = (1 - \varepsilon)\,q_i + \varepsilon / V$$

This penalises over-confident predictions (output distribution concentrated
on a single token), which is the proximate cause of the train-PPL / OOD-PPL
divergence: the model sharpens on FineWeb-Edu co-occurrences, producing
wrong-but-confident OOD predictions.

**Self-correcting:** as the gap falls back to target, $\varepsilon \to 0$.
**Saturates at** $\varepsilon_0$ when $G(t) \ge 2\,G^*$.

### Ablation protocol

| Variant | GIF-1/2/3 | GIF-4 | ε₀ | Expected |
|---------|-----------|-------|----|----------|
| Baseline | ✓ | ✗ | 0 | Gap ~3.0 by step 6k |
| GIF-4 low | ✓ | ✓ | 0.02 | Gap ~2.5 |
| GIF-4 mid | ✓ | ✓ | 0.05 | Gap ~2.0 (deployed) |
| GIF-4 high | ✓ | ✓ | 0.10 | Gap ~1.8 but train PPL hurt |
| GIF-4 static | ✓ | static=0.05 | 0.05 | Smoothing active even when gap OK → train PPL penalty |

### Limitation discovered

GIF-4 saturates at $\varepsilon_0 = 0.05$. On instance 41228110:
- Step 4500: gap = 2.20 (GIF-4 active, ε at ceiling)
- Step 6000: gap = 2.81 (rebound — train PPL kept falling, OOD flat)

The ceiling exists because label smoothing attacks **output-side** confidence
but leaves the **encoder-side** redundancy untouched. The model compensates
by encoding domain-specific features in attention head similarity
and hidden state norms.

### Key commits

- `12644c9` — feat(gif): GIF-4 gap-driven label smoothing
- `7523d59` — dsl: add GIF-4 equation to lib/gif.neuro

### Config

```neuro
gif: {
  label_smooth_max: 0.05
}
```

### Empirical evidence

- Instance 41228110: gap 3.08→2.20 (step 4500), then rebound to 2.81
  (step 6000). OOD PPL 419→190 (train PPL 420→66).
