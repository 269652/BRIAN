---
code_refs:
  - neuroslm/emergent/gif.py (VBBAlphaSchedule)
  - neuroslm/dsl/nn_ops.py (_DIVERSITY_STASH)
  - lib/gif.neuro (gif_vbb_alpha_schedule, gif_ood_probe_ema, gif_isotropy_schedule)
created_at: "2026-06-16T18:00:00Z"
id: H007
proof_path: null
proof_status: missing
references:
  - docs/OOD_MECHANISMS.md
  - docs/FINDINGS.md
  - Tishby (2000) — Information Bottleneck
  - Tononi (2016) — IIT 4.0
  - Mu & Viswanath (2018) — isotropy as OOD robustness proxy
status: stated
tags: [gif, ood, information-bottleneck, isotropy, vbb, adaptive]
test_refs:
  - tests/dsl/test_gif_config.py
theorem_name: Brian.GIFAdaptiveRampConverges
title: "GIF-1/2/3: Adaptive gap-ratio-driven ramp closes the generalisation gap"
updated_at: "2026-06-17T00:00:00Z"
---

## H007 — GIF-1/2/3: Adaptive gap-ratio-driven ramp closes the generalisation gap

### Statement

Let $G(t) = \text{PPL}_\text{OOD}(t) / \text{PPL}_\text{train}(t)$ be the
gap ratio at training step $t$, and let $G^* > 1$ be the target gap ratio.
The GIF adaptive controller maintains a shared progress variable
$p \in [0,1]$ that drives three mechanisms:

1. **GIF-1 (VBB α schedule):**
   $\alpha(t) = \alpha_\text{start} + (\alpha_\text{end} - \alpha_\text{start}) \cdot p(t)$

2. **GIF-2 (OOD probe EMA):**
   $\hat{G}(t) = (1 - \beta)\,\hat{G}(t{-}1) + \beta\,G(t)$
   replaces the pre-fusion EMA with a true held-out WikiText-103 CE probe

3. **GIF-3 (Isotropy schedule):**
   $w_\text{iso}(t) = w_\text{max} \cdot p(t)$

The adaptive P-controller advances progress:
$$\Delta p = k_p \cdot \max(0,\; G(t) - G^*) + v_\text{min}$$
subject to a static floor $p(t) \ge p_\text{static}(t)$.

**Claim:** Under this controller, $\limsup_{t\to\infty} G(t) \le 2\,G^*$
for any training distribution satisfying the bounded-gradient assumption.

### Motivation

The 10k forensic (2026-06-16) identified a triple failure:
- Pre-fusion EMA systematically overestimated trunk loss (~7 nats when
  actual ~3 nats), poisoning cortex inhibition/distillation gates
- VBB at static α=0.001 passed ~5000 nats — the IB was decorative
- Isotropy regularisation was disabled by the time the gap opened

GIF-1/2/3 fix all three simultaneously with a single shared ramp.

### Ablation protocol

| Variant | GIF-1 | GIF-2 | GIF-3 | Expected |
|---------|-------|-------|-------|----------|
| Full    | ✓     | ✓     | ✓     | Gap ≤ 2.0 by step 5k |
| No VBB  | ✗     | ✓     | ✓     | KL collapse, gap > 3.0 |
| No probe| ✓     | ✗     | ✓     | Ramp blind, progress min-speed only |
| No iso  | ✓     | ✓     | ✗     | Anisotropy-driven gap (~2.5) |
| Static  | static| ✓     | static| Gap controlled but slower convergence |

### Key commits

- `b05de67` — feat(gif): wire GIF config into SmolLM
- `0489d22` — fix(gif): add gif:{} to TrainingConfig parser
- `80ac5e0` — test(gif): 39 tests pinning the GIF config pipeline
- `e75e3f7` — fix(gif): telemetry + ramp pull to step 500
- `3983c7a` — feat(gif): adaptive gap-ratio-driven ramp controller
- `524839f` — fix(gif): do not advance progress before OOD probe fires
- `2d55426` — fix(gif): arm OOD probe via load_gif_probe()

### Config (arch.neuro)

```neuro
gif: {
  enabled: true
  adaptive: true
  target_gap_ratio: 1.5
  ramp_gain: 0.0002
  min_ramp_speed: 0.00005
  vbb_alpha_min: 0.001
  vbb_alpha_max: 0.05
  vbb_ramp_start: 500
  vbb_ramp_end: 3000
  probe_n_seqs: 50
  probe_every: 100
  iso_weight_max: 0.01
  kl_kappa: 200
}
```

### Empirical evidence

- Instance 41222326: probe armed, GIF active, gap stuck at 3.08–3.11
  (probe never armed → fixed in `2d55426`)
- Instance 41228110: gap dropped from 3.08 to 2.20 at step 4500
  (VBB α ramping, iso active, KL healthy 380–520)
