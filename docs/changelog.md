# Changelog

Reverse-chronological summary of every commit on the branch. Maintained
by `brian ai document` from `git log`.

<!-- Entries grouped by month, shape:

## 2026-06

- `0ba70e2` · 2026-05-31 · arch · genetics + multi-scale + NFG-v3 + DSL extensions
- `35d946f` · 2026-05-31 · fix · chunked CE + expandable-segments allocator
- ...
-->

## 2026-06

- `(staged)` · 2026-06-14 · feat(MoE+NT) · Items 2/3/4/5/6 — neuro-modulated
  mixture-of-experts surgery (B5 prep).
  Five disciplined-TDD additions, all GREEN end-to-end:
    * **Item 5** — Expert alias registry (`_EXPERT_ALIAS_REGISTRY` +
      `resolve_expert_alias()` + `register_expert_alias()` in
      `neuroslm/experts.py`; 14 tests). Any HF model id is now a valid
      `multi_cortex.experts[i].id` — `gpt2`, `Qwen/Qwen2.5-0.5B`,
      `smollm2_360m`, `codegpt_py`, `hf://owner/repo`, etc. all resolve
      uniformly via `LMExpert.__init__`.
    * **Item 2** — NE-driven router temperature (new
      `router_temp_nt_gain` knob on `MultiCortexConfig`;
      `ThalamicRouter.set_nt_levels({"NE": ...})` + clamped
      `mult = clamp(1+k·z_NE, 0.1, 10.0)` applied before softmax;
      12 tests). Default 0.0 = back-compat identity.
    * **Item 3** — 5HT/DA-driven distillation λ multiplier
      (`distillation_5ht_gain`, `distillation_da_gain`;
      `BRIANHarness._distillation_lambda` multiplies the gap-ramp
      value by `clamp(1+k_5HT·z_5HT − k_DA·z_DA, 0, 2)`; 14 tests).
    * **Item 4** — GABA-gated lateral expert inhibition (new
      `LateralInhibition` module in `neuroslm/cortex.py` with divisive
      normalisation `w_i / (1 + κ·rival_mass_i)` and renormalisation;
      `lateral_inhibition_kappa` DSL knob; `LMExpertEnsemble` accepts a
      `lateral_inhibition=...` kwarg; 16 tests). Gini-sharpness
      contract pins WTA semantics.
    * **Item 6** — Trainable W coupling matrix in `DrivenNTSystem`
      (now `nn.Module`; optional `W_param: nn.Parameter(7, 5)` exposed
      under `trainable_W=True`; new differentiable readout
      `predict_nt_tensor(drivers)`; `_RunningStats.peek()` for
      idempotent z-scores; 17 tests). Float OU dynamics unchanged
      (`step_full` uses `W.detach()`), so `levels()` is bit-identical
      regardless of the flag. Shipped DISABLED in arch.neuro until a
      future surgery plumbs `predict_nt_tensor` into the consumers.
    * **Harness wiring** — Single `BRIANHarness.distribute_nt_levels()`
      seam called auto-magically by `compute_loss(nt_levels=...)`; the
      training loop in `train_dsl.py` sources the live NT dict from
      the observer's `DrivenNTSystem` each step. 6 tests pin the
      seam, polymorphism vs legacy ensemble, None-is-no-op, and the
      auto-distribute call site.
  arch.neuro activates Items 2/3/4 with conservative gains
  (`router_temp_nt_gain: 0.5`, `distillation_5ht_gain: 0.4`,
  `distillation_da_gain: 0.4`, `lateral_inhibition_kappa: 0.5`).
  79 new tests added across 5 files; all green.


  (`neuroslm/fitness.py` — JSON-serialisable `FitnessConfig` / `FitnessObjective`
  / `FitnessAdaptation` with `compute_loss(metrics)`, save/load helpers and a
  `create_fitness_mutation_vesicle()` builder for mid-training evolutionary
  mutations.  Sits alongside the runtime `FitnessComposer` from `5344ca0`.)
- `95d9638` · 2026-06-07 · feat(metabolic) · Phase B/F3 — NRCSTKController for
  metabolic-market neuron pruning
  (`neuroslm/modules/nrcstk.py` + `tests/test_nrcstk_metabolic.py` — EMA-driven
  per-neuron demand statistic, hinge-squared budget loss, hard-zero pruning
  mask that starts all-ones and tightens after first `observe()`; 24 tests
  across 5 classes.  Completes the C → A → B Multi-Objective-Fitness order.)
- `5344ca0` · 2026-06-07 · feat(fitness) · Phase A/F1+F2 — `fitness { ... }`
  DSL block + `FitnessComposer` runtime (absorbed alongside a UTF-8 colab fix).
  `neuroslm/dsl/training_config.py` adds `FitnessConfig` / `FitnessObjective`
  with the `_VALID_FITNESS_OBJECTIVES = {lm, phi, nis_plus, symbolic, piso,
  metabolic}` / `_VALID_FITNESS_SCHEDULES = {constant, gated, linear}`
  whitelist; `neuroslm/fitness.py` adds `LossBundle` + `FitnessComposer(nn.Module)`
  with per-objective phase-gate table.  39 tests in `tests/test_fitness_parser.py`
  + `tests/test_fitness_composer.py`.
- `ad07f0f` · 2026-06-07 · feat(symbolic) · Phase C/F5 — SymbolicHyperNeuron
  for mathematical invention
  (`neuroslm/modules/symbolic_unit.py` + `tests/test_symbolic_unit.py` —
  `OperatorBank` over `{identity, add, sub, mul, exp, sin, tanh}` (with
  `_safe_exp` clamped to ±20) plus Gumbel-softmax selection over two inputs
  and one operator per unit; produces a sparsity-loss term and human-readable
  `expression_strings()`.  36 tests across 6 classes.)
