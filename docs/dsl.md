# The `.neuro` Architecture DSL — Reference

BRIAN's brain architectures are not Python classes. They are declarative
`.neuro` files compiled to PyTorch `nn.Module`s at runtime. Every
population, synapse, and modulation carries an *explicit mathematical
equation* — algebraic, ODE, or a reference to a reusable macro — that
lowers to torch ops via a SymPy-backed equation IR.

This document is the full reference. For the high-level story of why
this exists, see [`architecture.md` § The `.neuro` DSL](architecture.md).
For commit-by-commit history of how it got built, see the
`arch(dsl-*)` commits on master.

---

## 1. Why math-first?

Before the DSL, an architecture lived in `neuroslm.brain.Brain` as
hand-written PyTorch — 28 sub-modules, hard-coded wiring, hard-coded
neurotransmitter modulation. Two problems:

1. **Opaque semantics.** "Thalamus" was a class name; the actual gating
   maths were buried in `forward()`. You couldn't ask the system "what
   is the fixed point of this circuit under steady input?" without
   re-deriving the equations from the code.
2. **Hard to evolve.** Mutating an architecture meant editing Python —
   no clean perturbation model for an evolutionary search loop, no easy
   ablation of just the gating mechanism.

The math-first DSL turns the architecture into a *symbolic, analyzable
object*:

* Every dynamic is an equation in scope of named variables.
* Fixed points are computed by `sympy.nsolve(rhs == 0)`.
* Linearised stability falls out of the Jacobian.
* The evolutionary engine mutates equation strings, not Python code.
* Codegen lowers the same equation IR to the same torch ops the legacy
  hand-written modules produced (verified `torch.allclose` byte-equal).

---

## 2. Folder layout

A BRIAN architecture lives in its own folder under `architectures/`:

```
architectures/
  rcc_bowtie/
    arch.neuro                      ← package config + global wiring (entry point)
    modules/
      sensory.neuro                 ← one file per brain region
      thalamus.neuro
      world.neuro                   ← (world + self_m grouped as state models)
      amygdala.neuro                ← (amygdala + insula as affect)
      qualia.neuro
      gws.neuro                     ← (gws + neural_geometry as workspace)
      hippocampus.neuro             ← (hippo + entorhinal + cerebellum as memory)
      pfc.neuro                     ← (pfc + acc as cognitive control)
      bg.neuro                      ← (bg + forward_m + evaluator as executive)
      dmn.neuro                     ← (dmn + thought_transformer + claustrum)
      motor.neuro
      nuclei.neuro                  ← 6 neuromodulatory nuclei
    lib/
      dynamics.neuro                ← reusable named dynamics + helper functions
```

* **`arch.neuro`** is the *package config*: declares architecture
  metadata, the 7 global neurotransmitter systems, every module
  `import`, and all cross-module wiring (synapses, modulations,
  formal specs). It does *not* define populations itself.
* **`modules/*.neuro`** each export the populations of one brain region.
  A folder containing `index.neuro` is also a module (mjs-style); the
  `import` resolves to the index file.
* **`lib/*.neuro`** carry shared mechanics (`dynamics`, `function`) that
  any module may import.

---

## 3. Top-level constructs

### `architecture`

```neuro
architecture rcc_bowtie {
    d_sem: 256,         ← default semantic dimension for populations
    dt: 0.01            ← default ODE integration timestep
}
```

Lives only at the root of `arch.neuro`. Sets defaults the rest of the
file inherits.

### `neurotransmitter`

```neuro
neurotransmitter dopamine {
    base_concentration: 0.10,
    release_rate: 0.20,
    reuptake_rate: 0.80,
    diffusion_rate: 0.02
}
```

A globally-visible NT pool with first-order kinetics. Concentrations
are supplied at forward-pass time via `nt_levels={"dopamine": 0.7,
...}`. Future stages will integrate the kinetics ODE in-line.

### `population`

```neuro
export population pfc {
    count: 256,                            ← neuron count (informational; codegen uses d_sem)
    dynamics: "rate_code",                 ← enum macro reference
    equation: "y = ReLU(x)",               ← OR explicit algebraic equation
    timescale: 0.02,                       ← time constant (informational for now)
    capacity: 1.0                          ← capacity (informational)
}

export population amygdala {
    count: 32,
    dynamics: "integrate_and_fire",
    ode: "dV/dt = (-V + x) / tau",          ← OR explicit ODE
    timescale: 0.005
}
```

Precedence: `ode:` > `equation:` > `dynamics:` enum macro > passthrough
fallback. The `export` keyword makes a population visible to imports
from other files; without it the population is private to its module.

Variable conventions inside a population's equation/ode:
* `x` — input (sum of all incoming synapse contributions, or the
  top-level `sensory_input` for input-pop)
* `y` — output
* `s` — persistent state (algebraic; auto-registered as a buffer)
* `V`, or any other identifier on the LHS of `dV/dt` — ODE state
* `d_sem` — bound to `self.d_sem` (the population's dimension)
* Any other free symbol — auto-promoted to a learnable
  `nn.Parameter(torch.zeros(1))`. Override by importing a `dynamics`
  decl from a lib file that declares `params:`, `state:`, `constants:`.

### `synapse`

```neuro
synapse pfc -> bg {
    weight: 0.6,
    neurotransmitter: "glutamate",
    equation: "y = weight * (x_pre @ W)"
}
```

A linear (by default) projection between two populations. Variables in
the equation:
* `x_pre` — source population's current-step output (or last-step, for
  back-edges in re-entry loops)
* `W` — the synapse weight matrix (random-init buffer of shape
  `(d_sem, d_sem)`)
* `weight` — scalar from the `weight:` field (default 1.0)
* `y` — contribution added to the target's input

If `equation:` is omitted, the canonical legacy form
`y = weight * (x_pre @ W)` is used.

Note the PyTorch batched-tensor convention: `x_pre @ W` (not the
mathematician's `W @ x_pre`) because `x_pre` carries a batch dim.

### `modulation`

```neuro
modulation dopamine -> pfc {
    effect: "multiplicative",
    gain: 0.6,
    equation: "y = output * (c * gain)"
}
```

A neurotransmitter's effect on a target population. Variables:
* `output` — target's current output tensor
* `c` — NT concentration from `nt_levels` at forward time
* `gain` — scalar from the `gain:` field
* `y` — new output value

Canonical legacy forms:
* `multiplicative`: `y = output * (c * gain)`
* `additive`:        `y = output + (c * gain)`

### `dynamics` (lib only)

```neuro
export dynamics lif_neuron {
    ode: "dV/dt = (-V + x) / tau",
    state: { V: "torch.zeros(1, d_sem)" },
    constants: { tau: 0.05, dt: 0.01 }
}
```

A reusable dynamics definition. After
`import { lif_neuron } from "@/lib/dynamics"`, any population in the
importing file can use it via `dynamics: "lif_neuron"`. The `params:`,
`state:`, and `constants:` fields tell codegen how to register
learnable parameters, persistent state buffers, and scalar locals.

### `function` (lib only)

```neuro
export function decay(x, alpha) {
    equation: "(1 - alpha) * x"
}
```

A reusable equation fragment. Function calls in an equation string get
*inlined* at compile time. (Inlining lands in a follow-up stage; for
now `function` declarations parse and store correctly but aren't
yet invoked from population equations.)

### `formal_spec` / `sheaf`

```neuro
formal_spec phi_integration {
    rule: "integrated_information",
    metric: "phi"
}

sheaf narrative_consistency {
    contradiction_threshold: 0.7,
    mechanism: "h1_cohomology_proxy"
}
```

Constraint declarations recognised by the parser and stored in the IR;
the larger BRIAN runtime consumes them for Φ scoring and contradiction
detection. They don't participate in codegen.

### `fitness` (training sub-block)

```neuro
training {
    fitness: {
        enabled: true,
        objectives: {
            lm:        { weight: 1.0,   enabled: true },
            phi:       { weight: 0.02,  enabled: true, schedule: "gated" },
            symbolic:  { weight: 0.05,  enabled: true, schedule: "gated" },
            metabolic: { weight: 0.001, enabled: true, schedule: "gated" }
        },

        # SymbolicHyperNeuron (only read when `symbolic.enabled`):
        symbolic_n_units:         8,
        symbolic_n_features:      64,
        symbolic_tau_init:        1.0,
        symbolic_tau_final:       0.1,
        symbolic_sparsity_weight: 0.01,

        # NRCSTKController (only read when `metabolic.enabled`):
        metabolic_budget:          0.7,
        metabolic_prune_threshold: 0.05
    }
}
```

Lives **inside** `training { ... }`, not at top level. Parsed by
`neuroslm.dsl.training_config.parse_training_config` into
`TrainingConfig.fitness : FitnessConfig`; consumed at runtime by
`neuroslm.fitness.FitnessComposer`.

| Field                         | Type                 | Notes                                                                                    |
|-------------------------------|----------------------|------------------------------------------------------------------------------------------|
| `enabled`                     | bool                 | Master switch.  False ⇒ composer passes `bundle.lm` through unmodified.                  |
| `objectives.<name>`           | `FitnessObjective`   | One entry per active objective.                                                          |
| `objectives.<n>.enabled`      | bool, default `false`| Per-objective on/off; disabled objectives contribute zero and are omitted from telemetry.|
| `objectives.<n>.weight`       | float, default `0.0` | Multiplier applied before the schedule factor.                                           |
| `objectives.<n>.schedule`     | enum                 | `"constant"` (default), `"gated"` (phase-gate from `dsl.maturity`), or `"linear"`.       |
| `symbolic_n_units` …          | int / float          | Construction args for `SymbolicHyperNeuron`; ignored when the `symbolic` objective is off.|
| `metabolic_budget` …          | float                | Construction args for `NRCSTKController`; ignored when the `metabolic` objective is off. |

**Valid objective names** (validated at parse time, source of truth in
`_VALID_FITNESS_OBJECTIVES`):

```python
{"lm", "phi", "nis_plus", "symbolic", "piso", "metabolic"}
```

**Valid schedules** (`_VALID_FITNESS_SCHEDULES`):

```python
{"constant", "gated", "linear"}
```

The `"gated"` schedule multiplies each objective by a maturity-driven
phase gate centred per-objective in `neuroslm.fitness._GATE_TABLE`
(e.g., `metabolic` opens at maturity 0.65 to avoid pruning topology
before it stabilises).  This matches the legacy `AuxWeights` curve so
migrating an objective from the hard-coded `total_loss_config` into
this block preserves its training-time activation bit-for-bit.

See `architecture.md` §7.5 for the runtime composition pipeline and
`tests/test_fitness_parser.py` for every supported field shape.

---

## 4. Imports and exports

### Path specifiers

```neuro
import { lif_neuron }         from "@/lib/dynamics"       ← absolute (from arch root)
import { core }               from "./layers"             ← relative (same dir)
import { thalamus }           from "../thalamus"          ← relative (parent dir)
```

* `@/foo` is anchored at the architecture root (the folder containing
  `arch.neuro`). The `@` is a literal marker, not a variable.
* `./foo` and `../foo` are relative to the current file.
* Paths that escape the architecture root are rejected.
* Trailing `.neuro` is optional; `import "@/lib/dynamics"` and
  `import "@/lib/dynamics.neuro"` are equivalent.
* If `foo` is a folder containing `index.neuro`, the specifier
  resolves there (mjs-style folder-as-module).

### Named, aliased, and side-effect imports

```neuro
import { foo, bar }                from "@/lib/x"        ← named
import { foo as bar, baz as qux }  from "@/lib/y"        ← aliased
import "@/lib/setup"                                      ← side-effect (no binding)
```

### Exports

Mark any declaration with `export`:

```neuro
export population output { ... }       ← visible to importers
population helper { ... }              ← private (file-local)
export dynamics custom_dyn { ... }
export synapse a -> b { ... }
```

Importing a non-exported name from another file is a `ResolverError`
at compile time.

---

## 5. The macro library

Built into `neuroslm.dsl.equations.DYNAMICS_DECLS`. These are the
seven canonical neural dynamics; their equations are what `dynamics:
"foo"` expands to:

| Macro                 | Form        | Canonical equation/ODE                              |
|-----------------------|-------------|-----------------------------------------------------|
| `rate_code`           | algebraic   | `y = ReLU(x)`                                       |
| `winner_take_all`     | algebraic   | `y = softmax(x / 0.1) * d_sem`                      |
| `gated`               | algebraic   | `y = ReLU(x) * sigmoid(gate)`   *(gate ∈ Param)*    |
| `attractor_network`   | algebraic   | `y = (1 - alpha) * s + alpha * ReLU(x)`             |
| `attention_pool`      | algebraic   | `y = softmax(x) * ReLU(x)`                          |
| `static`              | algebraic   | `y = x`                                             |
| `integrate_and_fire`  | ODE         | `dV/dt = (-V + x) / tau`         *(tau = 0.05)*     |

Writing the canonical equation explicitly produces a byte-equivalent
`nn.Module` to using the enum form. This is the central guarantee — see
[`tests/dsl/test_codegen.py`](../tests/dsl/test_codegen.py)
`TestEquationVsMacroParity`.

---

## 6. Compilation pipeline

```
architectures/rcc_bowtie/
    │
    ▼
multifile.Resolver
    │  walks the folder, parses every .neuro file into a ModuleAST,
    │  resolves every `import` to a target file + validates exports,
    │  builds user_dynamics / user_functions tables
    ▼
multifile.compile_folder
    │  emits declarations in canonical order:
    │    NT systems → populations (per arch.neuro import order)
    │    → synapses → modulations → formal_specs/sheaves
    │  pipes the synthetic single-file source through
    ▼
compiler.NeuroMLCompiler
    │  regex-extracts populations/synapses/modulations/etc.
    │  into a ProgramIR
    ▼
codegen.CodeGenerator
    │  for each population: resolve DynamicsDecl → lower equation to
    │  torch ops → emit a nested nn.Module class
    │  for the circuit: instantiate populations, wire synapses (with
    │  forward/back-edge classification), apply modulations
    │  validates generated source with ast.parse, exec()s into a
    │  namespace, returns the class
    ▼
Compiled nn.Module class
    │
    ▼
circuit = Cls(d_sem=128)
output_dict = circuit(sensory_input, nt_levels={...})
```

Cycle handling: populations are evaluated in declaration order. A
synapse whose source is *later* in declaration order than its target
reads `self.last_{source}`, a `(1, d_sem)` buffer updated to the
mean of the source's output at the end of each forward pass.
Forward-edges read the current step's `outputs[source]` directly.

---

## 7. Symbolic analysis

The same equation IR that drives codegen supports static analysis.

### Fixed points

```python
from neuroslm.dsl.equations import parse_equation, find_fixed_point

eq = parse_equation("y = sigmoid(x)")
fp = find_fixed_point(eq, input_symbol="x", guess=0.5)
# fp ≈ 0.6590    (the unique fixed point of x = sigmoid(x))
```

For ODEs, the fixed point is where `dV/dt = 0`:

```python
from neuroslm.dsl.equations import parse_ode, ode_fixed_point

ode = parse_ode("dV/dt = -V + I")
fp = ode_fixed_point(ode, param_bindings={"I": 1.5})
# fp == 1.5    (V* = I for a leaky integrator)
```

### Stability

Linearized stability at a fixed point — Jacobian sign:

```python
from neuroslm.dsl.equations import ode_stable_at

ode = parse_ode("dV/dt = -V")
ode_stable_at(ode, point=0.0)   # True    (slope -1, contracting)

ode = parse_ode("dV/dt = 2 * V")
ode_stable_at(ode, point=0.0)   # False   (slope +2, expanding)
```

For algebraic recurrences `x ← f(x)`, use `jacobian_at(eq, "x",
point=...)`: `|J| < 1` is locally stable.

---

## 8. Tests

The DSL has 215 tests across 8 files. The semantic-equivalence
guarantee — equation form, enum form, and reference torch
implementation all produce `torch.allclose` byte-equal output — is
pinned in three places:

* [`tests/dsl/test_codegen.py`](../tests/dsl/test_codegen.py) —
  parameterised per-dynamics reference parity, macro-vs-explicit-
  equation parity, synapse routing, NT modulation.
* [`tests/dsl/test_codegen_syn_mod.py`](../tests/dsl/test_codegen_syn_mod.py)
  — synapse/modulation legacy-vs-equation parity.
* [`tests/dsl/test_codegen_rcc.py`](../tests/dsl/test_codegen_rcc.py)
  — end-to-end: rcc_bowtie folder compiles, every population has an
  explicit equation/ode, forward pass produces well-formed output for
  all 28 regions, NT modulation responds.

The multi-file infrastructure is covered by:
* `test_multifile_paths.py` — path resolution + folder discovery
* `test_multifile_parser.py` — `module`/`import`/`export` parsing
* `test_multifile_resolver.py` — cross-file reference linking
* `test_multifile_lib.py` — `dynamics`/`function` defs and lookup

---

## 9. Worked example — adding a custom population

Say you want to add an *insula* sub-region with a non-standard
dynamic: leaky integration over the sensory bottleneck, scaled by a
learnable gain.

1. **Define the dynamic** in a lib file (or inline in the module).
   In `architectures/rcc_bowtie/lib/dynamics.neuro`:

   ```neuro
   export dynamics scaled_leak {
       equation: "y = scale * ((1 - alpha) * s + alpha * ReLU(x))",
       state: { s: "torch.zeros(1, d_sem)" },
       constants: { alpha: 0.1 },
       params: { scale: "torch.ones(1)" }
   }
   ```

2. **Use it** in `architectures/rcc_bowtie/modules/amygdala.neuro`:

   ```neuro
   import { scaled_leak } from "@/lib/dynamics"

   export population insula_v2 {
       count: 64,
       dynamics: "scaled_leak"
   }
   ```

3. **Wire it** in `architectures/rcc_bowtie/arch.neuro`:

   ```neuro
   import { insula_v2 } from "@/modules/amygdala"
   synapse sensory -> insula_v2 { weight: 0.4, equation: "y = weight * (x_pre @ W)" }
   ```

4. **Run a test** that compiles the folder and checks the new
   population is present:

   ```python
   from neuroslm.dsl.multifile import compile_folder
   ir = compile_folder("architectures/rcc_bowtie")
   assert any(p.name == "insula_v2" for p in ir.populations)
   ```

No Python edits required. The codegen picks up the new dynamic from the
lib import, registers `s` as a buffer and `scale` as a learnable param,
and emits the forward pass that computes
`scale * ((1-0.1)*s + 0.1*ReLU(x))`.

---

## 10. Roadmap

Landed (`arch(dsl-*)` commits on master):
* Phase 7 Stage 1 — algebraic equations + macro library
* Phase 7 Stage 2 — ODE dynamics + Euler integration + stability
* Multi-file Stage 1 — path resolver + folder loader
* Multi-file Stage 2 — module / import / export parser
* Multi-file Stage 3 — cross-file reference resolver
* Multi-file Stage 4 — `dynamics` / `function` defs in lib
* Multi-file Stage 5 — synapse / modulation equation codegen
* Multi-file Stage 6 — rcc_bowtie migrated to folder layout
* Multi-file Stage 7 — legacy `.neuro` file removed

Planned:
* **Function inlining** — `equation: "y = decay(x, 0.1)"` resolves the
  call to a body from a `function` decl and substitutes parameters.
  Parsing is in place; the codegen-side rewrite is next.
* **Brian2-compatible grammar** — adopt threshold/reset events for
  spiking neurons, unit annotations (`: volt`), and the broader Brian2
  equation block. Compatibility target: parse selected Brian2 tutorial
  examples unmodified.
* **Multi-line import statements** — currently single-line only;
  adding parser support is a small ergonomic fix.
* **Auto-prefixing of module-local names** — for architectures with
  name collisions across modules. rcc_bowtie has none so this hasn't
  been needed yet.
* **Plasticity declarations** — `synapse pfc -> bg { plasticity:
  hebbian }` where `hebbian` is a reusable rule from `lib/plasticity`.
