# Static Analysis Layer for the NeuroSLM DSL — Detailed Plan

**Status**: planning. Implementation does not begin until this document has
been validated externally (NotebookLM + Perplexity) and Phase 2–5 of the
DSL refactor (`docs/DSL_REFACTOR.md`) is complete.

**Goal**: turn the `.neuro` DSL from a passive description language into an
*analyzable program*. Once an architecture is expressed in the DSL, we can
verify properties, optimize structure, and search the design space using
static-analysis techniques that classical software compilers use — applied
to neural-flow graphs.

This document is structured for handoff to NotebookLM / Perplexity:
each section names a discipline (type theory, abstract interpretation,
geometric algorithms, …), cites the published work that grounds it,
states what we'd build, and gives an honest difficulty estimate.

---

## Why this matters for this project specifically

We've spent four iterations (RCC P1 → P2 → P3, plus synth-v1/v2/BEMA)
chasing leaks in the bowtie. Each leak was the *same class of bug*:
a closed-loop write path between cognitive sidecar and language trunk.
Each one was discovered empirically by running training, observing a
PPL spike, tracing the cause in code.

A static-analysis pass on the DSL — given the right abstractions —
**would have flagged every one of those leaks before training started**.
The forward-write-back, NT-modulation, and parameter-mutation leaks
are graph properties of the architecture, detectable from the topology
alone if the topology is in machine-readable form.

So this isn't research-for-research's-sake. The static-analysis layer is
the engineering tool that prevents the next four iterations of trial and
error.

---

# Part A — Tractable Engineering (Foundations)

These layers are well-grounded, have working implementations in
adjacent fields, and can be shipped incrementally. Estimated total
duration **3–5 weeks** for someone with PL-background experience.

## A1. Type System for Tensor Shapes

**Discipline**: dependent type theory / refinement types applied to tensors.

**Published grounding**:
- Hasktorch, Dex (Google Research; Maclaurin et al. 2019, *Dex: array
  programming with typed indices*) — dependent types for tensor shapes
  in a functional language.
- Hindley–Milner-style inference over shape variables: PyTorch's
  `torch.fx`, TVM's Relay IR.
- `tsanley` (Singh et al. 2021) — runtime-checked tensor shapes, but
  the same ideas apply at compile time.

**What we'd build**:
- Extend `module` DSL block with `inputs:` and `outputs:` shape signatures:
  ```
  module language_cortex {
      class: "LanguageCortex"
      inputs:  { ids: tensor[B, T, dtype=long] }
      outputs: { logits: tensor[B, T, vocab_size], sem: tensor[B, d_sem], h: tensor[B, T, d_hidden] }
      ...
  }
  ```
- Type checker that walks the DSL graph, unifies shape variables across
  edges (the output shape of one module must match the input shape of
  the next), and rejects ill-typed compositions at DSL compile time.
- Specifically catches: `Linear(d_hidden, d_sem) ∘ Linear(d_sem, d_hidden)`
  composes correctly (good); but `Linear(d_hidden, d_sem) ∘ Linear(d_hidden, d_sem)`
  fails because the second's input expects `d_hidden`, sees `d_sem`.

**Difficulty**: **MEDIUM**. ~1 week. Existing libraries (`tsalib`, type
inference algorithms) provide patterns.

**Output**: any `.neuro` file with shape mismatches fails to compile
with a precise error before training is launched.

## A2. Control-Flow Graph (CFG) of Neural Data Flow

**Discipline**: classical compiler dataflow analysis.

**Published grounding**:
- Aho/Sethi/Ullman *Compilers: Principles, Techniques, & Tools* —
  standard CFG construction + reaching definitions + live variables.
- LLVM IR's CFG and MLIR's region-based control flow.
- For neural networks: ONNX, TVM Relay IR, JAX's jaxpr.

**What we'd build**:
- From the DSL's `module` declarations + `connect` declarations
  (added in Phase 3 of DSL refactor), construct a directed graph
  where nodes are modules and edges are tensor flows.
- Standard graph algorithms applied:
  - **Reachability**: which modules reach the LM head? Modules that
    don't are dead code — flag them.
  - **Dominator tree**: which module's output ALL downstream modules
    depend on? These are bottleneck modules.
  - **Strongly Connected Components (SCC)**: detects feedback loops.
    Each SCC > size 1 is a closed loop in the architecture.
  - **Forward/backward dataflow**: track which modules read/write
    which tensors; flag conflicting writes.

**What this catches concretely**:
- RCC P1's forward write-back (motor → h_lang) is an edge in the CFG
  that creates an SCC including the language trunk. The analyzer
  would flag it: "module `motor` writes to `h_lang` which is read by
  `language_cortex` which writes to `motor`'s input feed — cycle
  detected."
- RCC P2's NT modulation: same pattern, different edge — the trophic
  system's NT field is an SCC participant with the trunk.
- RCC P3's parameter mutation: requires extending the CFG with
  parameter-mutation edges (in addition to data-flow edges) — same
  algorithm, richer graph.

**Difficulty**: **MEDIUM**. ~1 week. Standard textbook algorithms;
implementation is straightforward graph manipulation in Python.

**Output**: any `.neuro` file with closed-loop architecture, dead modules,
or shared-write conflicts fails compilation with a precise error.

## A3. Abstract Interpretation for Architectural Invariants

**Discipline**: abstract interpretation (Cousot & Cousot 1977).

**Published grounding**:
- Cousot & Cousot 1977 — original abstract interpretation paper.
- α-CROWN, CertNN (NeurIPS) — abstract interpretation applied to
  neural network *robustness verification* (different goal, same
  technique).
- Polyhedral abstract domains (Bertrand-Goeman 2010) — encode tensor
  shape constraints, value-range constraints.

**What we'd build**:
- Lattice-valued abstract domains over the DSL graph:
  - **Sign domain**: gate values are in {≥0, ≤0, sign(unknown)}.
  - **Interval domain**: scalar values are in `[lo, hi]`.
  - **Bool domain**: feature flags True/False/⊤.
- A fixed-point analyzer that walks the graph, propagates abstract
  values forward, and *proves* architectural invariants:
  - "All temporal-gate outputs are bounded in [0, 1]" — required for
    SGB / FE-ramp / BEMA correctness.
  - "Free-energy loss is non-negative" — required for it to act as
    a regularizer not a destabilizer.
  - "No closed-loop write path from cognitive sidecar to language
    trunk parameters" — the RCC bowtie invariant, machine-checked.

**What this catches concretely**:
- A new mechanism's gate that *might* go negative in some code path
  → invariant violated → DSL compile error.
- A mutation that re-introduces a leak we previously fixed → caught
  before training.
- Properties that the codebase informally assumes ("the trunk's
  attention temperature is bounded") become explicitly asserted.

**Difficulty**: **HARD**. ~2 weeks. Requires defining the abstract
domains, choosing a sound widening operator, and validating each
invariant against the existing architecture.

**Output**: each `.neuro` file ships with a list of *proven* invariants
about its behavior. Any future modification that violates an invariant
fails compilation.

## A4. Shape-Constrained Neural Architecture Search

**Discipline**: NAS (Neural Architecture Search) + constraint
satisfaction.

**Published grounding**:
- Zoph & Le 2017 — original RL-based NAS.
- DARTS (Liu et al. 2019) — differentiable NAS.
- Constrained NAS via type systems: TabNAS, NAS-Bench-301.

**What we'd build**:
- Extend the existing `neuroslm/dsl/evolutionary.py` mutation set to
  ONLY produce candidates that pass A1+A2+A3. Today the evolutionary
  loop generates many invalid architectures and discards them at runtime
  (expensive). Static-analysis-constrained NAS rejects them at the
  symbolic level (cheap).
- Bias mutations toward dimensions of high-uncertainty fitness — if
  param counts are over- or under-represented in the candidate space,
  push mutations toward the unexplored regime.

**Difficulty**: **MEDIUM**. ~1 week. The existing evolutionary loop is
the framework; we just gate its proposals through the static analyzers.

**Output**: search efficiency improves by an estimated 10–100× (the
mostly-invalid 99% of the candidate space gets pruned upfront).

---

# Part B — Geometric / Manifold Analysis (Research Territory)

These layers connect the DSL to *measured* training dynamics. They
require partial training runs and have less-mature published
foundations. Estimated total duration **6–10 weeks**.

## B1. Loss-Landscape Geometry as Fitness Signal

**Discipline**: PAC-Bayes + flatness/sharpness analysis.

**Published grounding**:
- Foret et al. 2020 — *Sharpness-Aware Minimization* (SAM).
- Dziugaite & Roy 2017 — *Computing nonvacuous generalization bounds*
  via PAC-Bayes.
- Salvatori et al. 2023 (NeurIPS) — predictive-coding networks
  empirically reach flatter minima.

**What we'd build**:
- After a brief training run (say, 500–1000 steps) on a candidate
  architecture, compute:
  - Hessian's top-k eigenvalues (via stochastic Lanczos).
  - PAC-Bayes flatness bound.
  - Effective rank of the trunk's representations.
- Use these as fitness signals in NAS (alongside training loss).

**Difficulty**: **MEDIUM-HARD**. ~2 weeks for the basic Hessian-spectrum
computation; another week to validate that the signal correlates with
final OOD performance.

**Risk**: the bound is loose enough to be noise relative to actual
OOD performance, providing no signal. Mitigation: use it as a *gate*
("reject candidates with eigenvalue > X"), not a *ranking* signal.

## B2. Latent Manifold Geometry of Representations

**Discipline**: manifold learning + intrinsic dimension estimation.

**Published grounding**:
- Recanatesi et al. 2019, *Predictive learning extracts latent space
  representations* (Nature Comm. Neurosci.) — intrinsic dimension of
  neural-trained representations correlates with generalization.
- Hénaff et al. 2019 (NeurIPS) — *Temporal straightening* — the
  trunk's per-step trajectory geometry predicts next-token entropy.
- Stringer et al. 2019 (Nature) — empirical manifold geometry in
  cortical representations.

**What we'd build**:
- Per-layer, compute the intrinsic dimension of the latent
  representation (via Two-NN estimator).
- Per-step, compute the trajectory curvature.
- Track these throughout training; fit a regression of "early-training
  manifold geometry" → "final OOD gap_ratio".
- Use the regression as a fitness signal.

**Difficulty**: **HARD**. ~3 weeks. Intrinsic-dimension estimation is
noisy; the correlation we hope for may not be strong.

**Risk**: published manifold-generalization results are on toy models;
no guarantee they transfer to 100M-scale LMs.

## B3. Effective Information (EI) Bounds at Compile Time

**Discipline**: integrated information theory + causal abstraction.

**Published grounding**:
- Hoel et al. 2013 (PNAS) — *Quantifying causal emergence shows that
  macro can beat micro*. Effective Information.
- Klein et al. 2024 — Effective Information in neural networks.
- Albantakis & Tononi 2015 (IIT 3.0) — Φ_MIP upper bound via
  min-partition.

**What we'd build**:
- For each module in the DSL, derive its state-space size from the
  declared dimensions.
- Compute an upper bound on the module's contribution to EI as
  `log|States(M)|`.
- Compute the bowtie min-cut Φ upper bound: `Φ ≤ I(narrowing; widening)
  ≤ min(H(bottleneck))`.
- Report at DSL compile time.

**Difficulty**: **MEDIUM**. ~1 week for the upper bounds (easy compute
from dimensions); tight bounds are research-tier.

**Risk**: the bounds will be loose. A 256-dim bottleneck has EI ≤ 256
bits which is huge — the bound is technically true but not useful for
discriminating architectures.

---

# Part C — Geometric Algorithm Discovery (Speculative)

The layer the user described as "geometrical algorithms to discover
pathways that boost performance based on geometrical topology phenomena
… latent manifold shape-based evolutionary improvements which
autodiscovers better algorithmic neural flow modulation and processing."

This is genuinely research-thesis tier. Estimated **3–6 months** for
a publishable proof-of-concept.

## C1. Differential-Geometric NAS

**Discipline**: Riemannian geometry + neural tangent kernel.

**Published grounding**:
- Jacot et al. 2018 — *NTK: Convergence and generalization in NN*.
- Belghazi et al. 2018 — *Mutual Information Neural Estimation*.
- Brehmer & Mishra 2019 — *Manifold-aware NAS*. (Less-known.)

**What we'd build**:
- Compute the NTK at init for each candidate architecture.
- Use NTK spectral properties (top-k eigenvalues, condition number)
  as a *static* fitness signal.
- Combine with the dynamic signals from B1/B2.

**Difficulty**: **VERY HARD**. NTK computation scales poorly; the
correlation with final performance is empirical and noisy.

## C2. Topology-Aware Search via Persistent Homology

**Discipline**: topological data analysis.

**Published grounding**:
- Carlsson 2009 — *Topology and data* (Bull. AMS).
- Naitzat et al. 2020 — *Topology of deep neural networks* (JMLR) —
  layer-by-layer persistent-homology evolution.

**What we'd build**:
- For each module in a DSL architecture, measure persistent-homology
  signatures of its output activations on a probe set.
- Mutate architecture in directions that simplify the persistent
  homology (Naitzat shows simpler topology correlates with better
  generalization on small models).

**Difficulty**: **VERY HARD**. ~6 months including paper. Tools exist
(ripser, gudhi) but applying them to 1024-dim activations is
computationally heavy.

## C3. Self-Improving Architecture (the "autodiscover" layer)

**Discipline**: meta-learning + program synthesis.

**Published grounding**:
- Real et al. 2020 — *AutoML-Zero* (ICML) — evolution discovers
  ML algorithms from scratch.
- Hospedales et al. 2021 — *Meta-learning in neural networks: a survey*.

**What we'd build**:
- The DSL has mutation rules (already in `mutations.py`). Add
  meta-mutations: mutations on the *mutation rules themselves*.
- The fitness signal is the rate at which child architectures discover
  good new mutations.

**Difficulty**: **THESIS-TIER**. 6–12 months realistic.

---

# Phasing and Dependencies

```
Foundations (Part A) — 3–5 weeks, deterministic engineering
   A1: type system           ────┐
   A2: CFG dataflow          ────┼─→  Part-A complete: every .neuro file
   A3: abstract interp       ────┘    statically verified before training
   A4: constrained NAS

   (depends on DSL Phase 2–5 being complete — modules + forward in DSL)

Measurement (Part B) — 6–10 weeks, real research
   B1: loss-landscape fitness
   B2: manifold geometry
   B3: EI / Φ bounds
       — each independent; can parallelize across people

Discovery (Part C) — 3–6 months, thesis-tier
   C1: differential-geometric NAS
   C2: persistent-homology
   C3: self-improving meta-arch

   (depends on A + B yielding usable fitness signals)
```

# Honest cost / value summary

| Layer | Engineering cost | Likelihood of payoff | Value if it works |
|---|---|---|---|
| **A1** type system | 1 wk | 100% (well-trodden path) | catches shape bugs at compile time |
| **A2** CFG dataflow | 1 wk | 100% | catches every RCC-style closed-loop bug |
| **A3** abstract interp | 2 wk | 80% (HARD but established) | machine-checked invariants |
| **A4** constrained NAS | 1 wk | 95% | 10–100× more efficient search |
| **B1** loss-landscape | 2 wk | 50% | useful fitness signal — or noise |
| **B2** manifold geometry | 3 wk | 30% | research-quality signal if it works |
| **B3** EI/Φ bounds | 1 wk | 40% (tight bounds) / 100% (loose) | reportable architectural claims |
| **C1** NTK NAS | 4–6 wk | 20% | novel direction; uncertain payoff |
| **C2** persistent homology | 6 wk | 20% | gorgeous math; thin ML track record |
| **C3** meta-arch discovery | 12+ wk | 10% | publishable paper if anything works |

# Recommendation for sequencing

1. **First, complete DSL Phase 2–5** (`docs/DSL_REFACTOR.md`) so the
   architecture is fully expressible in `.neuro` files. **Without that,
   none of the static analysis has an input to analyze.**
2. **Then ship A1–A2** (type system + CFG). 2 weeks. Catches every
   RCC-style bug at DSL compile time.
3. **A3 + A4** next. Another 3 weeks. Now NAS is efficient.
4. **B1** (loss-landscape) as the first real-data fitness signal.
5. **B2 and B3** in parallel if multiple people available.
6. **Part C** only after A + B have produced reproducible wins.

# What's deliberately not in this plan

- A custom IR like MLIR or Relay. Existing tools (jaxpr, torch.fx,
  Relay) are more mature; we'd just be reinventing.
- A new tensor compiler. Torch + TVM already do this layer.
- Replacing PyTorch as the runtime. The DSL is a *meta-layer*; the
  models still run on PyTorch.
- Formal verification in Coq/Lean. Mentioned earlier ("formally
  validated sheaf cohomology"). Real formal verification of an ML
  system is a multi-year project; not in scope.

---

**External review questions for NotebookLM / Perplexity**:

1. Does Hasktorch's shape-type system handle the operator-overloading
   we'd need for cross-cohort architectures (PCT predictors + standard
   transformer blocks)?
2. Has anyone applied Cousot-style abstract interpretation to NN
   architectural invariants (not robustness — invariants like "no
   closed-loop write from cognitive to trunk")?
3. Have Recanatesi-style intrinsic-dimension fitness signals been
   validated as predictive of *OOD* generalization specifically (vs.
   training-set generalization)?
4. Is there a working open-source implementation of persistent-homology
   feature extraction for high-dim NN activations that we could reuse,
   or is this a from-scratch project?
5. Does AutoML-Zero's program-synthesis loop scale to architectural-level
   mutations (not just per-layer ops)?
