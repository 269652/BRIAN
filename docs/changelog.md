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

- `700664b` · 2026-06-07 · feat · Fitness configuration system with self-improving objectives
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
