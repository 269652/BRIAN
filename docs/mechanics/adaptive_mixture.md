# adaptive_mixture — Adaptive Mixture Controller

**Category:** regularizer  
**Implementation:** `neuroslm.regularizers.AdaptiveMixtureController`  
**DSL spec:** [`mechanics/adaptive_mixture.neuro`](../../mechanics/adaptive_mixture.neuro)

## Overview

A closed-loop PI controller that adjusts the chat/prose training-data ratio to keep OOD entropy at a target level. It measures token entropy on a held-out prose probe every N steps, applies an EMA, and adjusts the chat fraction proportionally. Three damping mechanisms prevent oscillation: warmup, slew-rate limit, and EMA smoothing.

**Critical bug history**: `direction='amplify'` (legacy) caused `chat_ratio` to run from 0.60→0.80 in <100 steps on 2026-06-03, collapsing prose PPL. Always use `direction='balance'`.

## Equation

```
H_t = entropy(model on prose_probe)
H_ema = (1−α_ema)·H_ema + α_ema·H_t
chat_ratio += Δ = clip((H_target/H_ema)^γ − 1, −max_delta, +max_delta)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_entropy` | `4.5` | Target prose entropy in nats |
| `probe_interval` | `100` | Measure every N steps |
| `gamma` | `2.0` | Control gain exponent |
| `direction` | `"balance"` | **MUST be "balance"** (not "amplify") |
| `min_ratio` / `max_ratio` | `0.10` / `0.50` | Chat fraction bounds |
| `controller_warmup_steps` | `2000` | No updates before this step |
| `max_step_delta` | `0.03` | Slew-rate limit per update |
| `entropy_ema_alpha` | `0.1` | EMA smoothing coefficient |

## When to Use

Multi-domain training where model capability on prose must remain stable. Requires a held-out prose probe (≥50 sequences).

## When NOT to Use

- Single-domain training
- Probe set < 20 sequences (entropy estimate too noisy)
- `direction='amplify'`: this is the 2026-06-03 bug — do not use

## Properties

- **Closed-loop**: measures and adjusts in real time
- **Three damping mechanisms**: warmup + slew-rate + EMA
- **No parameter change**: pure data curriculum controller

## Empirical Evidence

Internal: `direction='amplify'` caused chat_ratio 0.60→0.80 in <100 steps and prose PPL collapsed. `direction='balance'`: stable ratio control confirmed.

## References

- Internal: `neuroslm/regularizers.py::AdaptiveMixtureController`
- Internal: `docs/findings.md` (2026-06-03 PR-B analysis)
