# multi_cortex — Multi-Expert LM Ensemble with KL Distillation

**Category:** routing  
**Implementation:** `neuroslm.experts.LMExpertEnsemble` + `neuroslm.harness.BrainHarness._cortex_fusion_aux_step`  
**DSL spec:** [`mechanics/multi_cortex.neuro`](../../mechanics/multi_cortex.neuro)

## Overview

Frozen pretrained LM experts (GPT-2, CodeGPT, Qwen2.5) are routed per-token via a ThalamicRouter (lexical bias + learnable head + NE temperature gain). Expert logits are combined and the trunk learns an additive correction. Capacity-Funneled Distillation (CFD) transfers expertise via KL with top-K sparsification, entropy-matched temperature, and a gradient-alignment gate. A cortex inhibition gate auto-retires an expert when the trunk surpasses it.

## Equation

```
logits_expert = Σᵢ gate_i · VocabBridge(logits_e_i)
logits_fused  = logits_expert.detach() + α·logits_trunk   # additive correction
# CFD distillation:
L_KL = λ_eff · KL(student/T_eff ‖ teacher_topK/T_eff) · T_eff²
# Inhibition:
inhibit_t = (1−α_inh)·inhibit_{t-1} + α_inh·σ(lm_loss − cx_loss)
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fusion_mode` | `"additive_correction"` | trunk learns delta over expert ensemble |
| `fusion_init` | `0.1` | Initial α for trunk delta |
| `cfd_enabled` | `true` | Capacity-Funneled Distillation |
| `cfd_topk_start/end` | `4/32` | K annealed from 4→32 over 1500 steps |
| `distillation_lambda_max` | `1.0` | Peak KL weight |
| `inhibition_enabled` | `true` | Auto-retire expert when trunk surpasses it |

## When to Use

Bootstrap a small trunk with domain expertise from larger pretrained LMs. The additive correction means the trunk starts predicting corrections to the expert ensemble — not competing with it.

## When NOT to Use

- No pretrained experts
- Cross-family experts with different tokenizers (byte-fallback bridge needed)
- distillation_lambda_max > 2.0 without CFD (teacher-too-strong pathology from H22)

## Properties

- **VocabBridge**: eliminates ln(V) CE floor from random projection
- **CFD no-harm floor**: gradient-alignment gate zeroes KL when it conflicts with LM grad
- **Expert retirement**: inhibition gate → ~20× FLOP savings when trunk surpasses expert
- **NT-modulated**: NE→router temp, 5HT/DA→distill λ, GABA→lateral inhibition

## Empirical Evidence

H22-H24: VocabBridge lowers initial CE from ~10.85 to ~3-5 nats. CFD prevents teacher-too-strong pathology. Additive correction fusion is current best config.

## References

- Hinton, Vinyals, Dean (2015) Distilling the Knowledge. NeurIPS Workshop
- Internal: `neuroslm/experts.py`, `tests/training/test_lm_expert_harness_integration.py`
