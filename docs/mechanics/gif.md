# gif — Geometric Information Funnel

**Category:** training_dynamics  
**Implementation:** `neuroslm.emergent.gif.GIFController`  
**DSL spec:** [`mechanics/gif.neuro`](../../mechanics/gif.neuro)

## Overview

GIF is a composite of 7 interlocking sub-mechanisms designed to close the train-PPL / OOD-PPL gap. It addresses a triple failure mode observed in every 10k run: (1) pre-fusion EMA gives cortex gates a wrong signal, (2) VBB KL weight is too small to constrain, (3) zero OOD regularisation after aux losses collapse. GIF-1 schedules the VBB α; GIF-2 replaces the EMA with an OOD probe; GIF-3 schedules isotropy; GIF-4 adds gap-reactive label smoothing; GIF-5 adds attention head diversity; GIF-6 replaces the LM head with a cosine head; GIF-7A/B add gradient and LR stability.

## Sub-mechanisms

| Sub-mechanism | Summary |
|---------------|---------|
| GIF-1 | VBB α schedule: α_start → α_end linearly |
| GIF-2 | OOD probe EMA: 50 WikiText-103 seqs, eval every 100 steps |
| GIF-3 | Isotropy schedule: ramp w_iso in lockstep with GIF-1 |
| GIF-4 | Gap-reactive label smoothing (ε proportional to gap/target − 1) |
| GIF-5 | Attention head diversity loss (pairwise cosine penalty) |
| GIF-6 | Cosine LM head (norm-invariant output) |
| GIF-7A | Divisive gradient normalisation (smooth gain control) |
| GIF-7B | Loss-variance metaplastic LR damping (BCM rule) |
| GIF-7C | VBB KL floor (anti-collapse guard) |

## Key Parameters

`vbb_alpha_start=0.001`, `vbb_alpha_end=0.05`, `vbb_ramp_start=1000`, `vbb_ramp_end=5000`, `iso_weight_max=0.05`, `label_smooth_max=0.1`, `target_gap=2.0`, `head_div_w0=0.01`, `cosine_head=true`, `divisive_grad_c=1.0`, `loss_var_window=50`, `kl_floor=10.0`.

## When to Use

Drop in as OOD-gap closing package when gap_ratio grows after step ~1000 in a multi-domain training run. Start with defaults; check per-component loss logs to see which sub-mechanism dominates.

## When NOT to Use

- Single-domain (GIF-2/4/5 cannot function)
- Very short runs (< 2k steps): VBB ramp idles
- No bottleneck: VBB encoder has nowhere to operate

## Empirical Evidence

Internal: root-cause analysis of triple failure post-H22 forensic. Isolated sub-mechanism ablations ongoing.

## References

- Tishby & Schwartz-Ziv (2017) Opening the Black Box of Deep Neural Networks. ITW
- Bienenstock, Cooper, Munro (1982) BCM rule. J Neurosci
- Internal: `lib/gif.neuro`, `neuroslm/emergent/gif.py`, `neuroslm/emergent/gif7.py`
