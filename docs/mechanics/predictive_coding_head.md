# predictive_coding_head — PC Reentry Between Layer States

**Category:** training_dynamics  
**Implementation:** `neuroslm.modules.predictive_coding_residual.PredictiveCodingResidual`  
**DSL spec:** [`mechanics/predictive_coding_head.neuro`](../../mechanics/predictive_coding_head.neuro)

## Overview

Implements Rao & Ballard (1999) predictive coding at the bowtie waist: the motor (top-down) population predicts the sensory (bottom-up) population via a learned projection W_pc. The prediction residual adds to the LM loss; gradient flows through BOTH populations, enforcing mutual predictability. An NT gate (DA/GABA) modulates the constraint strength. Can be upgraded to Friston variational free energy via vbb_alpha > 0.

## Equation

```
ĥ_sensory = W_pc · h_motor
r_pc = ‖h_sensory − ĥ_sensory‖²
gate = clip(1 + 0.5·DA − 0.7·GABA, 0, ∞)
L_pc = pc_reentry_weight · gate · r_pc
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pc_reentry_weight` | `0.1` | Weight on prediction residual |
| `pc_reentry_nt_gate` | `true` | Gate by DA/GABA |
| `vbb_alpha` | `0.001` | KL weight for VBB upgrade |
| `vbb_beta_init` | `1.0` | Initial precision β |

## When to Use

When the model has distinct sensory/motor populations (bowtie topology) and you want to enforce a world-model constraint: motor must predict sensory.

**Warning**: at pc_reentry_weight > 0.5 without VBB, the self-distillation amplifier effect kicks in (train-PPL halved, OOD-PPL barely moved, gap ratio 4.3→6.8 by step 4000).

## When NOT to Use

- No sensory/motor population split
- pc_reentry_weight > 0.5 without VBB upgrade

## Properties

- **Bidirectional gradient**: both sensory and motor populations updated
- **NT-gated**: DA (curiosity) strengthens; GABA (inhibition) weakens
- **Active inference**: implements Friston & Stephan (2007) when VBB is enabled

## References

- Rao, R. & Ballard, D. (1999) Predictive coding in the visual cortex. Nat Neurosci
- Friston, K. (2010) Free-energy principle. Nat Rev Neurosci
