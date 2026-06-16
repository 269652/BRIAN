---
code_refs:
  - neuroslm/harness.py (load_gif_probe — arm_probe=True)
  - neuroslm/train_dsl.py (load_gif_probe call)
  - neuroslm/emergent/gif.py (ood_ema updates)
created_at: "2026-06-16T23:30:00Z"
id: H011
proof_path: null
proof_status: missing
references:
  - docs/OOD_MECHANISMS.md
  - docs/FINDINGS.md
status: stated
tags: [gif, ood, probe, ema, bugfix, correctness]
test_refs: []
theorem_name: Brian.OODProbeArmingIsPrerequisite
title: "OOD probe arming is a prerequisite for all gap-reactive GIF mechanisms"
updated_at: "2026-06-17T00:00:00Z"
---

## H011 — OOD probe arming is a prerequisite for all gap-reactive GIF mechanisms

### Statement

The GIF controller's OOD EMA field (`ood_ema`) must be armed (set to a
non-zero initial value from the first probe evaluation) before any
gap-reactive mechanism can activate.

**Claim:** Without probe arming, `ood_ema = 0.0` persists indefinitely,
causing:
- Gap ratio $G(t) = 0/\text{PPL}\_\text{train} = 0$ — all gap-reactive
  mechanisms see zero gap and produce zero weight
- GIF-4 label smoothing: $\varepsilon = 0$ (no smoothing)
- GIF-5 head diversity: $w_\text{div} = w_0 \cdot \ln(1 + 0) = 0$
- GIF-1 VBB: α schedule proceeds but P-controller has no error signal

### Bug description

Prior to the fix, the probe arming sequence had a race condition:
1. `load_gif_probe()` was called before training loop
2. It loaded WikiText-103 sequences and ran initial evaluation
3. But the resulting PPL was **not written to `gif.ood_ema`**
4. `ood_ema` remained at default `0.0`
5. First non-zero `ood_ema` only appeared when the first in-loop
   probe evaluation fired (every 100 steps)
6. For the first 100 steps, all gap-reactive mechanisms were dead

The fix ensured `load_gif_probe()` explicitly arms the probe by
writing the initial evaluation result to `gif.ood_ema`, so gap-reactive
mechanisms are active from step 0.

### Impact

Without arming, the first 100 training steps run with:
- Zero label smoothing → confidence overshoot on early batches
- Zero diversity weight → heads converge to identical patterns early
- Zero gap signal → VBB P-controller has no error to correct

These early steps matter disproportionately because:
1. Gradient norms are largest at initialisation
2. Attention pattern formation happens in the first ~500 steps
3. Once heads converge, diversity loss has to fight against established patterns

### Ablation protocol

| Variant | Probe armed | Expected |
|---------|-------------|----------|
| Broken (pre-fix) | ✗ | ood_ema=0 for first 100 steps, gap-reactive dead |
| Fixed (post-fix) | ✓ | ood_ema=initial_ppl from step 0, all mechanisms active |
| Delayed arming | Armed at step 50 | Partial early convergence of heads |

### Key commits

- `2d55426` — fix: arm OOD probe so ood_ema is non-zero from step 0
- `524839f` — fix: don't advance OOD iterator before probe fires

### Empirical evidence

- Instance 41222326 (pre-fix): `ood=0.00` in early gif[] telemetry lines,
  gap-reactive mechanisms inactive. OOD PPL trajectory showed delayed
  improvement until probe fired at step 100.
- All post-fix instances: `ood=<actual_ppl>` from step 0.
