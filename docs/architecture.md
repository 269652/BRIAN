# NeuroSLM architecture notes

## Module map

| Module | Brain analog | File |
|---|---|---|
| LanguageCortex | Wernicke + Broca | `modules/language.py` |
| TextSensoryCortex | primary sensory cortex (+ superior colliculus) | `modules/sensory.py` |
| AssociationCortex | multimodal association cortex | `modules/association.py` |
| WorldModel | parietal / posterior cortex | `modules/world_model.py` |
| SelfModel | insula / TPJ | `modules/self_model.py` |
| GlobalWorkspace | thalamic relay + frontoparietal network | `modules/workspace.py` |
| DefaultModeNetwork | DMN | `modules/dmn.py` |
| Hippocampus | DG / CA3 / CA1 | `modules/hippocampus.py` |
| PrefrontalCortex | dlPFC | `modules/pfc.py` |
| BasalGanglia | striatum (Go/NoGo) | `modules/basal_ganglia.py` |
| ForwardModel | cerebellum | `modules/forward_model.py` |
| Evaluator | ACC / OFC | `modules/evaluator.py` |
| MotorCortex | M1 | `modules/motor.py` |
| Neuromodulators | VTA/LC/raphe/basal forebrain | `modules/neuromods.py` |

## Forward pass during pretraining (`Brain.forward_lm`)

1. LanguageCortex: tokens → logits + comprehension embedding `sem`.
2. SensoryCortex: salience-gated `sem` → sensory token.
3. AssociationCortex: fuses modality streams → unified percept.
4. WorldModel + SelfModel: update recurrent state.
5. GlobalWorkspace: competitive attention over candidate embeddings.
6. DMN: produces hippocampal query + stop signal.
7. Hippocampus: recall + novelty.
8. PFC: select thought + replace-gate decision.
9. BasalGanglia: action proposal (DA-modulated).
10. ForwardModel: predicted next world/self.
11. Evaluator: scalar value of predicted state.

Loss = w_lm · CE(next-token) + w_world · MSE(world prediction) + small reg.

## Inference (`Brain.cognitive_step`)

Same path, but with floating-thought updating, action conditioning of the
language head via `MotorCortex → LanguageCortex.from_sem`, and hippocampal
writes. `Brain.generate` loops this for `max_new` tokens.

## Limitations of this prototype

- World/self/forward losses are weak proxies (no real environment / RL signal).
- Neuromodulator scalars are computed from zero baselines during pretraining.
- Hippocampus is reset per-process; persistent storage is not yet implemented.
- Backprop through everything; no local learning rules yet (planned phase 2).
