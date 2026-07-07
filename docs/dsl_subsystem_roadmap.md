# DSL Full-Coverage Roadmap — porting every BRIAN subsystem

Goal: express the **entire** `Brain` architecture — every subsystem, plus
its gradient flow — formally in the `.neuro` DSL, with the compiler and
BRIAN harness supporting whatever language features that requires. This
is the path to `train_dsl.py` producing a model semantically equal to the
hand-written `Brain`.

This is a large, multi-week effort. The doc breaks it into the **new
language constructs** required and the **subsystems** each unblocks, so
work can proceed in dependency order with strict TDD (every ported
subsystem gets a `torch.allclose` equivalence test against the Brain
reference).

---

## Status: what already exists

| Capability | Status |
|---|---|
| Algebraic population equations (`y = f(x)`) | ✅ Phase 7 S1 |
| ODE population dynamics (`dV/dt = …`) + Euler + stability | ✅ Phase 7 S2 |
| Synapse + modulation equations | ✅ multifile S5 |
| Multi-file architectures, imports/exports, lib | ✅ multifile S1–7 |
| `training { … }` config (loss clipping, optimizer, etc.) | ✅ Phase A |
| BRIAN harness: embed + LM head + loss + clip + sched + AMP + resume | ✅ Phase A / A.2 |

## The missing language constructs (in dependency order)

### L1 — `state` blocks with explicit update rules  ⟵ foundational
Most subsystems are *stateful*: a per-population (or per-synapse) value
that persists across steps and updates by a declared rule.

```neuro
population hippo {
    state mem { shape: "(M, d_sem)", init: "zeros" }
    equation: "y = read(mem, x)"
    update mem: "mem + write_gate * outer(x, key)"
}
```

Unblocks: vesicle pools, trophic state, Hebbian fast weights, hippocampus
memory bank, maturity scalar, NT concentration kinetics, DNC temporal-link
matrix.

### L2 — `auxiliary_loss` declarations
Subsystems contribute extra loss terms beyond the LM cross-entropy.

```neuro
auxiliary_loss phi {
    equation: "-mutual_information(modules)",
    weight: 0.1,
    gated_by: "maturity > 0.3"
}
```

Unblocks: Φ-integration objective, actual-causation head, world-model
prediction loss, VQ commitment loss, NEMORI predictive-forgetting,
ReasoningCortex causality loss, survival-causal-head ΔS loss.

### L3 — `param_scope` + gradient routing  ⟵ **Phase B, next**
Declarative gradient isolation (the p3 fix) and separate optimizer groups.

```neuro
param_scope trunk { populations: [sensory, thalamus, gws, pfc, motor, ...] }
param_scope bio  { populations: [amygdala, hippo, vta, ...],
                   gradient: "detached_from_main_loss" }
```

Unblocks: p3 parameter-closure isolation, trunk/bio optimizer
partitioning, frozen-during-infancy params.

### L4 — event triggers (`on …`)
Discrete events fire actions: spikes, ignition, sleep onset.

```neuro
population amygdala {
    ode: "tau * dV/dt = -(V - V_rest) + R * x"
    on V > V_thresh { emit: spike, reset: "V = V_rest" }
}
on schedule(every: 2000) { run: sleep_cycle }
```

Unblocks: integrate-and-fire spiking (full LIF), GWS ignition gate,
sleep-cycle CLS scheduling, NEMORI gating events.

### L5 — conditional execution (`when …`)
Maturation-gated paths and adaptive compute.

```neuro
when maturity > 0.3 { enable: [auxiliary_loss.phi, auxiliary_loss.causal] }
when calm_halt(token) { early_exit }
```

Unblocks: topological maturation (infancy → awakening), adaptive compute
(MoD + CALM early exit).

### L6 — structured ops (attention, codebook, gather/scatter)
First-class ML primitives beyond elementwise + matmul.

```neuro
population thought_transformer {
    equation: "y = attention(q: Wq @ x, k: Wk @ x, v: Wv @ x)"
}
population bg {
    equation: "y = vq_lookup(codebook, x)",
    state codebook { shape: "(K, d_sem)", init: "kaiming" }
}
```

Unblocks: ThoughtTransformer (real self-attention), BG VQH (vector-
quantized codebook), DNC content-addressable read/write, geometry adapter.

### L7 — sheaf / cohomology operators
Domain-specific operators for the formal-spec subsystems.

```neuro
sheaf narrative { sections: episodes, gluing: overlap_consistency }
formal_spec phi { operator: "integrated_information", over: modules }
```

Unblocks: H¹ contradiction detection + SUPERSEDES, IIT 4.0 Φ measurement,
actual-causation (κ_cause).

---

## Subsystem coverage matrix

| Brain subsystem (arch.md ref) | Needs | Phase |
|---|---|---|
| Parameter-closure isolation (p3) | L3 | **B** |
| NT-modulation leak cut (p2) | (done — wiring) | B-audit |
| Trunk transformer body | L6 (attention) | C |
| Topological maturation | L1 + L5 | C |
| Adaptive compute (MoD/CALM) | L5 | C |
| Vesicle pools (κ_cause, κ_neg) | L1 + L2 | D.1 |
| Trophic system (BDNF) | L1 + L2 | D.2 |
| Hebbian fast weights | L1 | D.3 |
| Sleep-cycle CLS | L1 + L4 | D.4 |
| WorldModel + SelfModel | L2 + L6 | D.5 |
| Hippocampus DNC + sheaf | L1 + L6 + L7 | D.6 |
| GWS ignition gate | L4 | D.7 |
| BG VQH + NAcc RPE | L1 + L2 + L6 | D.8 |
| DMN + ThoughtTransformer + Claustrum | L6 | D.9 |
| Amygdala (full LIF) + LHb + Insula | L4 | D.10 |
| Qualia + homeostatic warp | L1 + L2 | D.11 |
| BRIAN narrative + sheaf stack | L1 + L7 | D.12 |
| NEMORI predictive forgetting | L2 + L4 | D.13 |
| Personality vector + trust | L1 | D.14 |
| Cognitive closure (gridworld loop) | L1 + L2 + L4 | D.15 |
| ActualCausationHead (IIT 4.0) | L2 + L7 | D.16 |

## Execution order (strict-sequential, TDD-gated)

1. **Phase B** — L3 (`param_scope`) → p3 fix declarative. *(starting now)*
2. **Phase C** — L1 + L5 + L6-attention → trunk parity + maturation.
3. **Phase D.1–D.16** — one subsystem per stage, each with an equivalence
   test vs the Brain reference impl, in the dependency order above.
4. **Phase E** — real data loader, BEMA, multi-stream checkpoints,
   episode recording.
5. **Phase F** — bit-identical forward + benchmark parity (HellaSwag/
   ARC/MMLU) between `train.py` (Brain) and `train_dsl.py` (DSL).

Realistic calendar: **6–12 weeks** of focused work for full coverage.
Each phase ships independently; the vast deploy can re-launch on the DSL
path at any phase boundary with strictly more Brain-equivalent behavior.

## TDD discipline

Every subsystem port lands a test of the form:

```python
def test_<subsystem>_matches_brain_reference():
    brain_out = reference_brain_subsystem(x, seed=0)
    dsl_out   = compiled_dsl_subsystem(x, seed=0)
    assert torch.allclose(brain_out, dsl_out, atol=1e-5)
```

No phase merges to master unless its equivalence test is green.

---

# NGL — the Neuro-Genetic Language (algorithm-discovery substrate)

## Why a fourth language

Layers A–D describe *architectures* (what tensors flow where). None of
them can describe an **ML algorithm** — an optimizer update rule, a
gradient/flow-modulation policy, a learning rule — because those need
**persistent state** and **control**, which the straight-line SSA DAG of
`nn_lang.py` deliberately forbids. NGL fills that gap. It is the
substrate on which "search the language space to discover a novel ML
mechanism" is actually tractable, and it is the proven design: AutoML-Zero
(Real et al., 2020) and the symbolic discovery of **Lion** (Chen et al.,
2023) both search exactly this kind of linear register-machine program
space.

## Core model — a typed register machine

An NGL `Program` is an ordered list of `Instruction`s over a typed
`Memory` of registers:

- **scalar bank** `s0..sK` — Python/0-d float registers (lr, betas, EI gates)
- **tensor bank** `t0..tK` — shape-polymorphic torch tensors (grad, param,
  momentum, velocity, activations)

An `Instruction` is `(op, out_reg, *in_regs, const?)`. Each `op` is drawn
from `OpRegistry`, and **every op has a closed-form semantics** (the
lowering table *is* the spec), so a program is a composition of known
differentiable functions — the same formal-semantics guarantee the rest
of the DSL family upholds.

Calling convention for an **update rule** (optimizer):

```
setup(shape)                 -> initialises state registers (m, v, step)
step(g, p) reads {g,p,state} -> writes delta into a designated out reg
p <- p - delta               (the harness applies the write)
```

Because state registers persist across `step` calls, momentum/Adam/Lion
are expressible; because the op set includes `select` (conditional) and a
bounded `repeat`, the language is **Turing-complete in the linear
register-machine sense** (equivalent to a bounded-tape register machine;
resource-bounded in practice for tractable search — we do not claim
unbounded-tape completeness, and that honesty is the point).

## Grammar space (op registry)

| Family | Ops |
|---|---|
| arithmetic | `add sub mul div neg abs sign square sqrt exp log clip` |
| reduction / norm | `mean sum norm rms max_r min_r` |
| control / compare | `gt select min max` |
| nonlinear | `tanh sigmoid relu silu softmax` |
| linear algebra | `matmul transpose outer` |
| constants / state | `const scale read write` |

This set spans the update-rule and flow-modulation grammar (it contains
SGD, Momentum, RMSProp, Adam, AdamW, Lion, Lookahead, sign-SGD, and
divisive/multiplicative gradient modulation as sub-programs). New ops are
added to `OpRegistry` with a semantics function + an exact-match test.

## Intrinsic semantic space

Every program embeds to a fixed vector (`Program.semantic_vector()`):
op-family histogram + structural features (length, state size, register
reuse, control depth). This is the metric space novelty search and
diversity preservation operate in — the "inherent ability to form and
encode novel algorithms by searching the language space."

## Genetic operators

`mutate` (point-change op / operand / const, insert, delete),
`crossover` (instruction-list splice), each re-validated by execution on
a probe. `auto_evolve` runs a Pareto GA (reusing `dsl/fitness.pareto_*`)
over a population of programs.

## CPU discovery harness

`neuroslm/genetic/discovery.py` + `brian discover`:

1. **Optimizer search** — evolve `step` programs; fitness trains a tiny
   CPU model (`synthetic_tasks.parity/modular_addition`, or a small MLP)
   for K steps and scores **(final loss ↓, steps-to-threshold ↓,
   throughput ↑)** as a multi-objective Pareto vector. Reproduces the
   AutoML-Zero / Lion result category: rediscover-or-beat SGD on CPU.
2. **Flow-modulation search** — evolve programs that gate activations /
   gradients; fitness adds an **effective-information / synergy** proxy
   from `neuroslm.information` (`net_synergy`, `pid_synergy`) so the
   search rewards topology/modulation that raises integration, not just
   lowers loss.

Discovered programs are recorded through the existing `DiscoveryStore`
ledger (`D###` records) so the audit trail matches the rest of the repo.

## What this session ships vs. what it enables

Shipped (tested, on CPU): the NGL core + op registry, exact-match proofs
that SGD/Momentum/RMSProp/Adam/Lion are NGL programs, the genetic
operators, and a runnable optimizer/flow-modulation discovery that beats
the SGD baseline on a tiny CPU benchmark. This is the *machinery* the
"outperform GPT-2 param-matched" goal requires; the large-scale training
that would cash that claim runs through `brian deploy`, not on this CPU.

## NGL, part 2 — arch bridge, algebraic simplifier, neuroanatomic auto-evolve

Three capabilities turn NGL from an optimizer-search substrate into a full
architecture-discovery loop (findings H32):

- **`compile_arch.py` — arch → NGL.** Lower an `nn_lang` forward graph into an
  NGL `Program`: SSA value → register, op/binop → instruction, params → pre-bound
  tensor registers. The composite NN ops (`linear`, `rmsnorm`, `layernorm`,
  `swiglu`, `gelu`, `embedding`) were added to `REGISTRY`, delegating to `nn_ops`,
  so the lowering is near 1:1 and byte-equivalent (verified on an FFN block).
  Scalar-config ops (attention `n_heads`) raise `UnsupportedLowering`. This is how
  discovery/simplification run on the real architecture rather than toy programs.

- **`rewrite.py` — verified algebraic simplifier.** Program → expression DAG
  (forward symbolic eval) → value-preserving rewrite rules applied to a fixpoint
  (`add-0`, `sub-0`, `mul-1`, `neg-neg`, `transpose²`, `(a+b)-b → a`, `cscale`
  constant folding, like-term combination `a·x + b·x → (a+b)·x`) → lower back with
  CSE. **Every accepted rewrite is globally probe-verified**, so unsound
  shape-dependent rewrites are rejected. Wired into `simplify()`. A bloated
  compiled FFN simplifies 6 → 3 instructions (dead code + `(h+h)-h → h`).

- **`neuro_evolve.py` — neuroanatomically-constrained trunk auto-evolve.** Evolve
  an NGL neuromodulation on a tiny CPU LM's residual stream (`h ← h · g(h)`),
  fitness = Pareto `(−val_PPL, +neuroanatomic_plausibility)`. The realism prior
  rewards divisive normalization, multiplicative gain, and homeostatic saturation,
  penalizes runaway amplification, and scores the dead-code-eliminated program.
  A bounded-`tanh` gain cut tiny-LM val PPL −9.4%. `brian discover trunk`.

The honest boundary: competitive **SmolLM** PPL is a GPU claim (`brian deploy`);
what runs on CPU here is the search engine + a tiny-LM demonstration + the bridge
that a deploy would use to wire a discovered gain law into the real trunk.

## NGL, part 3 — full-mechanic lowering, modulation store, GPU guardrail

Findings H33. The substrate now covers every model mechanic and treats discovered
modulations as managed artifacts:

- **Config-carrying instructions.** `Instruction` gained a `config` tuple and
  `OpSpec.uses_config`, so scalar-config ops (attention's `n_heads`, …) lower as
  opaque nodes. `causal_self_attention` is a registered NGL op;
  `compile_layer_to_ngl(source, bindings=…)` splits tensor vs config args. The
  **entire TransformerBlock** compiles to NGL byte-equivalent and is
  simplifiable/evolvable — not just the FFN. `config` is preserved through
  `to_source`, the peephole pass, and the algebraic rewriter.

- **Shape-correct verification.** `compile_arch.make_probes` builds real
  param+input tensors from the reference layer, and `simplify(..., probes=…)`
  verifies rewrites against non-degenerate values (generic all-zero probes made
  any rewrite look equivalent). Attention survives simplification (opaque);
  residual simplification around it is genuinely verified.

- **`modulation_store.py` — `modulations/*.neuro`.** An NGL program serializes to
  a `.neuro` `modulation { program { … } }` block (round-trip exact, incl. config
  ops). `ModulationStore` provides save / list / show / drop / **merge** (compose
  gains `g₂(g₁(h))`). `brian discover trunk --save NAME` persists the discovered
  gain law; `brian modulation {list,show,drop,merge}` manages the store — a
  discovered modulation is versionable, mergeable, throwaway-able.

- **Novelty + efficiency.** `discover optimizer --novelty W` adds semantic-space
  distance to the objective (hunt novel rules); cost objectives + the trunk
  prior's metabolic-economy term + the simplifier cover "more efficient".

- **GPU guardrail (enforced by design).** CPU discovery yields *candidates*
  proven on tiny models. A param-matched GPT-2 competitor comes **only** from GPU
  exploration + extensive GPU training (`brian deploy`). The CLI prints this;
  `--save` + a future `brian deploy --modulation NAME` are the CPU→GPU bridge.

### NGL device support + Colab exploration

NGL execution is device-aware: `Memory` tracks the live device from written
tensors and `execute` aligns every op's operands (and constants/eps guards) to
it, so a program runs correctly on cuda — a `cuda_tensor + cpu_scalar` mismatch
(which would silently fall back and corrupt the math) can't happen. The discovery
harness threads `device=` through `benchmark_optimizer` / `run_optimizer_discovery`
/ `run_trunk_evolution` / `run_flow_modulation_discovery`, with `_resolve_device`
degrading `cuda`→`cpu` when no GPU is present (so the same cell runs on a free CPU
runtime). CLI: `brian discover {optimizer,trunk,flow} --device auto`.

`colab_run.ipynb` ships two phone-friendly cells: a **T4 exploration cell**
(`brian discover … --device auto`, free — runs on the Colab GPU) and a **vast.ai
deploy cell** (rents a GPU for extensive training). Exploration finds candidate
mechanics/modulations on tiny models; the param-matched competitor still comes
from the deploy path.

## NGL, part 4 — abstraction, mechanism-search, compiler passes, prior-art gate

Four extensions toward evolving *complex, novel* algorithms (findings H34):

- **Macros / ADFs** (`macros.py`): named reusable sub-programs invoked via a
  `call` instruction; `expand_macros` inlines with fresh temps + copy-in input
  isolation + cycle guard. `Program.library` auto-flattens on execute. The GA can
  graft a whole macro as one gene (`mutate(library=…)` → `insert_call`), and
  `auto_evolve(macro_library=…)` threads it through selection — so the search
  composes higher-order structure instead of re-deriving primitives. Built-ins:
  divisive_norm, bounded_gain, sign_interp, rms_scale.
- **Attention as primitives** (`attention_primitives.py` + axis-aware ops
  `softmax_last`, `l2norm_last`, `causal_mask`): single-head causal attention
  expressed as an NGL program, byte-exact vs a torch reference — so the attention
  *mechanism* is mutable/searchable, not an opaque composite op.
- **Standard compiler passes** (`rewrite.py`): `cse` (common-subexpression
  elimination via hash-consed DAG), `constant_fold`, and `optimize` (DCE → CSE →
  const-fold → algebraic, to a fixpoint).
- **Prior-art gate** (`known.py`): a registry of known algorithms (SGD/Momentum/
  RMSProp/Adam/Lion + the trivial gradient rule) compared in hyperparameter-
  invariant semantic space; `discover optimizer --avoid-known` penalizes
  rediscovering them so the budget goes to novelty. `--macros` enables ADF
  grafting.

## NGL, part 5 — flow/compute heat, geometric topology, modulation auto-push

Records where information flows and where computation is heavy, analyses the
topology geometrically, and streams discoveries back during runs (findings H35).
The existing `neuroslm/evolution/` loop (heatmap→propose→gate keep-if-better,
already wired into training) supplies the online-mutation machinery; these add the
signal + geometry it lacked:

- `profile.py` — per-op information flow (output norm) + compute (est FLOPs) via a
  `recorder` hook on execute; `ExecutionProfile.{heavy_compute,hot_flow,low_hanging}`
  ranks bottlenecks and cheap-high-effect edges. `low_hanging` is the "search this
  first" signal.
- `topology.py` — weighted DiGraph + betweenness / articulation points / max-flow-
  min-cut / algebraic connectivity; `propose_edits` → bypass/parallelise/prune from
  geometry. (Graph theory, not a literal fluid-flow sim — same signal, far cheaper.)
- `modulation_pusher.py` — scoped git commit+push of `modulations/*.neuro` during a
  run. `discover trunk --save --push`; Colab explore cell `PUSH`.
- CLI: `brian discover profile --layer-file X --binding D=16 …`.

## NGL, part 6 — exploration wired into training + persistent search ledger

Findings H36. An online modulation explorer with a keep-if-better gate and a
persistent, cross-run dedup ledger:

- `ledger.py` — `SearchLedger`: semantic-signature-keyed, JSON-persistent record of
  searched patterns; `is_dud` skips patterns prior runs found unhelpful so a new
  run never re-searches the same space.
- `training_explorer.py` — `TrainingExplorer.maybe_explore(step, score_fn)` fires
  every `explore_every` steps, searches (skipping ledger duds), A/B keeps-if-better,
  records to the ledger. `run_training_with_exploration` wires it to a tiny CPU LM;
  the same explorer attaches to the real trunk via a `score_fn`.
- CLI: `brian discover explore --explore-every 500 --ledger PATH`,
  `brian discover ledger [--clear]`.

Real-trunk wiring is one `explorer.maybe_explore(...)` call in the training loop +
a trunk `score_fn`; validated on a `brian deploy` run.

## NGL, part 7 — baselines/seed-from, artifact push, quality-diversity manifold search

Findings H37:
- `baselines.py` — standard optimizers with tradeoffs (cost/memory/stability);
  `run_optimizer_discovery(seed_from=["adam"])` starts from the arch's current
  algorithm. CLI `discover optimizer --seed-from adam,lion`, `discover baselines`.
- `modulation_pusher.push_artifacts(paths)` — push the modulation store + search
  ledger + run JSONs during a run. `discover explore --push`.
- `qd_search.py` — MAP-Elites over a structural descriptor (length × op-family
  diversity = manifold coordinates); illuminates the space into a diverse zoo of
  high-performers across shapes. `discover qd --iters N`. This is the computable
  version of "novel algorithms emerging from the geometry of the semantic space";
  paired with H35's flow/topology it covers the flow-perturbation idea without a
  literal fluid-flow sim.

## NGL, part 8 — prior-art ledger seeding + mechanic optimizer

Findings H39:
- `known.seed_ledger_with_known(ledger)` — record every known algorithm/mechanic as
  NGL programs with outcome="known" so the explorer's is_dud gate skips them; only
  novel mechanics get searched. `discover ledger --seed-known`; `discover explore`
  auto-seeds (default on).
- `mechanic_optimizer.py` — optimize_mechanic (CSE + algebraic + superopt report),
  shared_subexpressions (compute-once targets across mechanics), analyze_common_
  mechanics. `discover optimize-mechanics`. Found adam 27→22, lion 8→4 reducible.

## NGL, part 9 — semantic description language, full mechanic catalog, shared-macro lift

Findings H40:
- `semantics.py` — abstract interpretation of an NGL program over a boolean value
  lattice (bounded / nonneg / normalized / sign_only / mixes). `analyze()` runs
  per-op transfer functions to a `SemanticSummary` (role, bounded, normalizing,
  elementwise, sign_based, stateful, inputs/state); `describe()` is its
  human-readable projection ("what it does / when to use"); `interchangeable(a,b)`
  is the substitution gate the CSE / mechanic-reuse search consults. Role and
  state fall out of the analysis (read-modify-write buffer ⇒ stateful ⇒ update
  rule; softmax_last+matmul ⇒ attention). `discover semantics --known NAME`.
- `catalog.py` — `MechanicCatalog` loads **all 74** `mechanics/`+`dynamics/`+
  `structures/` `*.neuro` specs through the existing `mechanic_parser` (the rich
  `summary`/`when_to_use`/`not_for`/`properties` blocks *are* the human-facing
  semantic-description language). `catalog_names()` is prior art for the novelty
  gate; `discover mechanics [--category C] [--describe NAME]`.
- `shared_macros.py` — `extract_shared_as_macros(mechanics)` lifts multi-op
  subexpressions shared by ≥2 mechanics into `Macro`s and rewrites every mechanic
  to `call` them (probe-verified; stateful/reused-internal cases left untouched) —
  CSE across the whole mechanic set, so an improvement is reused everywhere, not
  just where found. `promote_modulation(store, name)` stamps a validated
  modulation as the reference implementation. `discover extract-shared`.
- `normalize.py` — **semantic normalization** as a compiler step. `canonical_form`
  reduces a program to its normal form under the convergent rewrite system
  (`optimize`+`simplify` to a fixpoint); `semantic_signature` is that form's
  structural key (equal ⇒ rewrite-provably equal). `normalize_semantics(programs,
  counts, prefer)` clusters equivalent programs (rewrite-equal signatures, then
  probe-equal merges gated by semantic role **and** statefulness) and substitutes
  one canonical representative per class — most-used (`prefer="frequency"`) or
  lowest-complexity (`prefer="simplest"`). Stateful programs are never probe-merged
  (single-shot probes zero the state, so sgd and momentum would falsely unify).
  Wired into `TrainingExplorer` (`ExploreConfig.normalize=True`): candidates are
  canonicalized *before* the ledger's dud-skip/dedup, so exploration never
  re-searches a syntactic variant. `discover normalize [--prefer simplest]`.
  **Boundary (honest):** general equivalence is undecidable (Rice) and the minimal
  program is uncomputable (Kolmogorov), so this is complete only *relative to its
  rewrite theory + probe budget* — sound, practically strong on NGL's bounded
  fragment, not a universal intent-minimizer.
