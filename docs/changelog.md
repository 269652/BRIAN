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

- `XXXXXXX` · 2026-06-17 · feat(cli) · **HF Hub checkpoint listing/download,
  always-on chat daemon, deploy --resume / --latest**.
  Four user-facing capabilities wired end-to-end across CLI / deploy /
  trainer / vast.ai bash loop:
    * **`brian hf list / pull / latest`** (`neuroslm/hf_checkpoints.py` NEW) —
      Read-side companion to `neuroslm.checkpoint_push`. Lists every
      `.pt` on the HF Hub repo (newest-step first), pulls a specific
      checkpoint by `path_in_repo` or `--latest` into
      `lfs_checkpoints/<RUN_DIR>/step<N>.pt` (layout-preserving so the
      existing resume globber finds it), prints the newest URI.
      Accepts `hf://owner/repo/path` shorthand. Auth chain mirrors
      `checkpoint_push`: explicit token → `HF_TOKEN` env → cached
      `~/.huggingface/token`. Never raises — failures print + return
      empty/None. **27 new TDD tests.**
    * **`brian chat`** (`neuroslm/chat_daemon.py` NEW) — Boots a
      checkpoint into an always-on inference daemon with three
      concurrent surfaces: conversation (USER ↔ BRIAN), idle thoughts
      (model self-prompts during user inactivity), CLI dashboard (one
      ANSI screen with `memory` / `thoughts` / `chat` panes). Single
      `threading.Lock()` serialises user-turn and thought-tick
      generates (transformer KV-cache not thread-safe). `_stop.wait()`
      interrupt instead of `time.sleep` so process shutdown is
      instant. REPL slash commands: `/quit`, `/clear`, `/think`,
      `/render`. Resolution chain for the checkpoint: positional arg
      → `--latest` (HF pull) → local highest-step in
      `lfs_checkpoints/`. **34 new TDD tests with deterministic stub
      `GenerateFn` (zero torch dependency in suite).**
    * **`brian deploy --resume PATH_OR_URI`** + **`--latest`**
      (`neuroslm/cli.py`, `neuroslm/train_dsl.py`, `_deploy_train.py`,
      `scripts/vast_train_{dsl,dna}_loop.sh`) — Resume training from
      a specific checkpoint (local path or `hf://` URI), or auto-pick
      the highest-step from HF Hub with `--latest`. Five-hop env-var
      propagation: CLI → `extra['RESUME_FROM']` → `_deploy_train.py`
      ONSTART export → bash `RESUME_ARGS` array → trainer's new
      `--resume_from` flag. The trainer downloads `hf://` URIs into
      `--ckpt_dir` and resumes at the saved step. Backwards-compat:
      legacy `--resume` globber still works when `RESUME_FROM` is
      empty. **23 new TDD tests covering parser flags, handler
      dispatch, env propagation chain, bash array expansion.**
  **Net: 5 new test files, 120 new tests, all green in 2.4s.**
  Zero new top-level dependencies (lazy `huggingface_hub` import,
  no `rich`, no `curses`). Daemon works on Windows + git-bash + Linux.

- `5cec369` · 2026-06-17 · feat(trunk-opt) · **SpectralPowerLawProbe**
  (NOVEL intrinsic geometric invariant) + loss-space budget proxy +
  `isotropy_activation_step` (early erank guard).
  Three coupled changes shipped together as a closed measurement →
  telemetry → intervention loop:
    * **SpectralPowerLawProbe** (`neuroslm/emergent/trunk_opt.py`) —
      One SVD/step yields three intrinsic invariants of the
      representation manifold: power-law exponent `α`, goodness-of-fit
      `R²`, and Wegner participation ratio `D_PR`. Provably scale-
      invariant ($H \to cH$) and orthogonally-invariant
      ($H \to HQ$ for $Q \in O(d)$). Tracks the biological 1/f cortical
      signature (He 2014, Voytek 2015). Log format extended to
      `trunk[... α=1.34 R²=0.97 PR=12.5]`. Live numerical verification
      recovers $\alpha \in \{1, 1.5, 2, 3\}$ to $10^{-3}$ precision.
      10 new TDD tests in `TestSpectralPowerLawProbe`. **Formal hypothesis: H012**.
    * **Budget loss-space proxy** (`neuroslm/emergent/trunk_opt.py`,
      `neuroslm/harness.py`) — Fixes `trunk[budget=0.00]` blackout
      observed in Colab telemetry for all 1400 steps. When no
      LM-only backward is available, `B_L = L_LM / L_tot` is used
      instead of `B_∇ = ||∇L_LM|| / ||∇L_tot||`. Exact when
      `L_aux = 0`, directionally consistent otherwise. **Formal hypothesis: H013**.
    * **`isotropy_activation_step`** (`neuroslm/regularizers.py`,
      `neuroslm/dsl/regularization.py`, `architectures/SmolLM/arch.neuro`) —
      Per-intervention override (default `-1` = use global gate).
      Lets whitening loss fire *before* the global activation step,
      preventing the erank collapse 40→2.3 observed during the
      LM-only warmup. SmolLM recipe activates at step 1000 (3000
      steps before DAR/PCC). **Formal hypothesis: H014**.
  13 new TDD tests across `TestSpectralPowerLawProbe` (10) +
  `TestActivationStep` (3). 1004 broader regression tests pass.
  Pre-commit hook ran 911 tests green and regenerated README.
  Three hypothesis files added (H012/H013/H014) + `docs/formal_framework.md`
  §15 TRUNK-OPT — Measurement & Provability Layer.


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
