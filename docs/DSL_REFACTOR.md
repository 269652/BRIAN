# DSL → Brain Compilation Refactor

Branch: `arch/dsl-compiled-brain`
Goal: make `neuroslm/dsl/rcc_bowtie.neuro` (or a similar `.neuro` file) the
**single source of truth** for the model architecture. All future changes to
the architecture happen in the DSL file; `neuroslm.brain.Brain` is regenerated
from it.

## Why

The current workflow has architectural state spread across:

- `neuroslm/config.py` (270-line BrainConfig with ~50 feature flags + 9 presets)
- `neuroslm/brain.py` (3000 lines of `Brain.__init__` + `forward_lm`)
- `neuroslm/modules/{language,smooth_gated_bus,predictive_coding_trunk,...}.py`
- `neuroslm/intelligence/orchestrator.py` (11-stage pipeline)
- `neuroslm/neurochem/{projections,growth}.py`

A single architectural change (e.g. closing the NT-modulation leak in P2)
required touching `config.py`, `brain.py`, and adding a new preset. That is
the shape of a system whose architecture is implicit. Making the DSL explicit
turns architecture changes into single-file edits.

## What exists already

- **DSL syntax**: `neurotransmitter`, `population`, `synapse`, `modulation`,
  `sheaf`, `formal_spec` blocks (~7 top-level keywords).
- **Parser**: `neuroslm/dsl/compiler.py` parses to an IR (PopulationIR,
  SynapseIR, NeurotransmitterSystemIR, ModulationIR, SheetIR, FormalSpecIR).
- **Architecture file**: `neuroslm/dsl/rcc_bowtie.neuro` (406 lines) — the
  current architecture's structure is already partially captured.
- **Evolutionary loop**: `evolutionary.py`, `fitness.py`, `mutations.py` use
  DSL IR for variant generation + selection.

## What's missing for the full vision

The current DSL describes the **structural topology** but not:

1. Training hyperparameters (lr, wd, warmup, dropout, label_smoothing, …)
2. Architectural feature flags (`use_rcc_bowtie`, `rcc_freeze_nt_modulation`,
   `use_predictive_coding_trunk`, `use_smooth_gated_bus`, `fe_gate_enable`,
   `use_branching_ema`, `detach_trunk_from_aux`, `maturity_ratchet`,
   `freeze_pruning_after_maturation`, …)
3. Layer dimensions (`d_hidden`, `d_sem`, `lang_layers`, `lang_heads`, ...)
4. Sub-module-specific config (PCT predictor hidden mult, SGB gate
   centers/widths, BEMA gamma/alpha_cap, ...)
5. The forward path: which operations run, in what order, gated by what
6. The training loop: warmup ramps, awakening transitions, trophic
   plasticity schedule
7. The OOD eval pipeline configuration

## Phases

### Phase 1 — Config from DSL  ← **this session**

Add a `config { ... }` block to the DSL that captures everything in
`BrainConfig`. Compile produces a `BrainConfig` instance.

**Deliverables**:
- New `config { ... }` syntax in DSL
- `ConfigIR` node in compiler IR
- `compile_to_brain_config(filepath) -> BrainConfig` function
- Update `rcc_bowtie.neuro` to include the full `rcc_bowtie_30m_p2` config
- Equivalence test: `compile_to_brain_config('rcc_bowtie.neuro') == PRESETS['rcc_bowtie_30m_p2']()`

**Out of scope**: changing how `Brain` is built. After Phase 1 the user can
edit `rcc_bowtie.neuro`'s `config` block instead of editing `config.py`.

### Phase 2 — Brain skeleton from DSL  *(next session)*

The DSL needs to express which submodules a Brain instance contains
(LanguageCortex, SmoothGatedBus, PredictiveCodingTrunk, ...) and their
construction args.

**Deliverables**:
- New `module <name> ( <args> )` syntax — e.g. `module language_cortex(d_hidden=384, lang_layers=4, ...)`
- `ModuleIR` node
- Codegen that emits a Python file `neuroslm/_generated_brain.py` containing
  a `BrainFromDSL` class with all `__init__` registrations
- Equivalence test: same param count, same state_dict keys as direct
  `Brain(cfg)` construction

### Phase 3 — Forward path from DSL  *(later)*

The hardest piece. The DSL needs to express the dataflow graph: which tensor
flows where, what gates apply, when each module fires.

**Approach options**:
- (a) Express as a sequence of named operations with input/output bindings
  — closest to a computation graph in DSL form
- (b) Keep the forward path in Python, but make its structure introspectable
  from the DSL (gates, branches all named, DSL captures the graph topology)

Likely (b) is the pragmatic answer: the DSL captures the *topology + gates*,
the Python code is a thin executor that interprets the graph.

### Phase 4 — Switch consumers  *(later)*

- `train.py` imports `from neuroslm._generated_brain import Brain`
- `brian_ood_test.py` same
- Old `neuroslm/brain.py` deleted (or kept as historical reference)
- All architectural changes happen in `rcc_bowtie.neuro` (or a new `.neuro`
  file copied + modified)

### Phase 5 — DSL → multi-architecture system  *(later)*

Once the loop closes, the evolutionary discovery infrastructure
(`evolutionary.py`, `mutations.py`, `fitness.py`) can mutate the DSL,
codegen the mutated Brain, train, and select — fully automated.

## Acceptance criteria for "complete"

1. `arch/rcc-bowtie-p2`'s working architecture is fully expressible in
   `rcc_bowtie.neuro`.
2. `python -m neuroslm.dsl.compile rcc_bowtie.neuro` emits a working
   Python file.
3. `python -m neuroslm.train --preset rcc_bowtie_30m_p2 ...` is replaced by
   `python -m neuroslm.train --neuro neuroslm/dsl/rcc_bowtie.neuro ...`.
4. Same trained model state-dict (modulo random-seed-dependent paths).
5. Editing a flag in `rcc_bowtie.neuro` is a one-file change with no
   Python edits required.

## Honest expected duration

- Phase 1: 2-3h (this session)
- Phase 2: 1-2 days
- Phase 3: 3-5 days  *(the hard one)*
- Phase 4: 1-2 days
- Phase 5: 1 day
- **Total: 8-13 days** of focused work.

This document tracks progress across sessions.
