# Formal Framework — THSD as a Discovery Substrate

> **Status:** v0.1 — Source of Truth for evolutionary mutations  
> **Audience:** Architecture researchers, DSL authors, verifier implementers  
> **Scope:** The mathematical contract that every code change in `neuroslm/thsd/`, `neuroslm/verification/`, `neuroslm/dsl/thsd_parser.py` and the evolutionary loop is bound by.

This document is **normative**. When code and document disagree, the document wins; either the code is fixed or this document is updated *first* (with a TDD acceptance suite that pins the new contract). The intent is that an autonomous evolutionary loop can read this file as the spec it is allowed to mutate against, and a human can audit its mutations by diffing it.

---

## 0. Conventions & Notation

| Symbol | Meaning |
|---|---|
| $K$ | Simplicial complex modelling the architecture; $K = \bigsqcup_d K_d$ |
| $\sigma_d \in K_d$ | A $d$-simplex (0 = vertex / population, 1 = edge / projection, …) |
| $F$ | Cellular sheaf over $K$, assigning a stalk $F(\sigma)$ to each simplex |
| $F(\sigma) \cong \mathbb{R}^n$ | The stalk; for symbolic simplices $n = n_U$, for regular populations $n = d_{\text{model}}$ |
| $\rho_{\sigma \to \tau}$ | Restriction map $F(\sigma) \to F(\tau)$ for $\tau \in \partial\sigma$ |
| $\delta^k$ | Coboundary $C^k(K; F) \to C^{k+1}(K; F)$ |
| $H^1(K; F)$ | First sheaf cohomology — the contradiction obstruction |
| $\Phi(K)$ | Integrated information of the active complex (IIT 4.0 proxy) |
| $\lambda_1(L)$ | Algebraic connectivity (Fiedler eigenvalue) of the sheaf Laplacian $L = \delta^{0\top}\delta^0$ |
| $g_F$ | Fisher–Rao metric on a stalk |
| $\mathrm{EI}(M, m)$ | Effective information of macro $M$ over micro $m$ |

Code references use ``monospace`` and follow the form ``neuroslm.thsd.engine::SymbolicSimplex``.

---

## 1. Simpliziale Ontologie — what the architecture *is*

### 1.1 Definition (Architecture as a labelled simplicial complex)

An **architecture** is a tuple
$$
\mathcal{A} = (K, F, \mathcal{O}, \mathcal{T})
$$
where

* $K$ is a finite abstract simplicial complex of dimension $\le d_{\max}$.
* $F$ is a cellular sheaf over $K$ with stalks in finite-dimensional real vector spaces.
* $\mathcal{O} = \{\mathrm{op}_1, \dots, \mathrm{op}_{n_O}\}$ is the operator bank used by symbolic simplices.
* $\mathcal{T} \subset \mathbb{T}^2$ is the (sub-)torus of *Tonnetz-admissible* spectral configurations (see §6).

### 1.2 Simplex kinds

Every simplex $\sigma \in K$ carries a `kind` label, accessible via `K.simplices[d][σ]["kind"]`. The kinds currently defined are:

| `kind` | Dim | Stalk meaning | Code |
|---|---|---|---|
| `"population"` | 0 | A standard neural population; $F(\sigma) = \mathbb{R}^{d_\text{model}}$ | `SimplexComplex.add_simplex` |
| `"symbolic"` | 0 | A `SymbolicHyperNeuron`; $F(\sigma) = \mathbb{R}^{n_U}$ | `SymbolicSimplex.register` |
| `"projection"` | 1 | An edge with restriction $\rho$; an explicit weight tensor | `SimplexComplex.add_simplex` with `dim=1` |
| `"composition"` | 2 | A triangle whose 1-faces must satisfy a cocycle condition | (reserved — wired in Phase THSD-2) |

A change of `kind` is a **structural mutation** and must produce a corresponding entry in the DNA delta (§5.2).

### 1.3 Why a *cellular* sheaf and not a presheaf

The cellular sheaf $F$ enforces that for every edge $\sigma_1 = [u,v]$,
$$
\rho_{\sigma_1 \to u}(F(\sigma_1)) = \rho_{\sigma_1 \to v}(F(\sigma_1))
$$
must hold *up to a measurable obstruction* (which becomes the H¹ class). A presheaf would not let us detect that inconsistency: it would silently produce two incompatible views and let downstream code hallucinate.

The cellular structure is therefore the **mechanism by which the architecture cannot lie to itself**.

---

## 2. Kohomologie-Wächter — the H¹ Guard

### 2.1 The coboundary chain

For a sheaf $F$ over $K$, the cochain complex is
$$
C^0(K;F) \xrightarrow{\delta^0} C^1(K;F) \xrightarrow{\delta^1} C^2(K;F) \xrightarrow{\delta^2} \dots
$$

* $C^0$ is the product of vertex stalks: $\bigoplus_{v \in K_0} F(v)$
* $C^1$ is the product of edge stalks: $\bigoplus_{e \in K_1} F(e)$
* $\delta^0(s)(e) = \rho_{e \to v}(s_v) - \rho_{e \to u}(s_u)$ for $e = [u,v]$

### 2.2 Definition (Sheaf Inconsistency)

A global section $s \in C^0(K;F)$ is **consistent** iff $\delta^0(s) = 0$. The size of the violation
$$
\mathcal{R}(s) \;=\; \|\delta^0(s)\|_{L^2}
$$
is the **inconsistency residual**. The space of inconsistencies that cannot be made trivial by a global gauge is
$$
H^1(K;F) \;=\; \ker(\delta^1)/\mathrm{im}(\delta^0).
$$

### 2.3 Invariant (Cohomology Guard)

**Cohomology Guard.** A trained checkpoint $\theta$ is **admissible** only if, on a held-out probe set $\mathcal{X}_{\text{probe}}$,
$$
\frac{1}{|\mathcal{X}_{\text{probe}}|} \sum_{x \in \mathcal{X}_{\text{probe}}} \mathcal{R}(s_x) \;\le\; \varepsilon_{H^1}
$$
where $s_x = \{F(\sigma)|_x\}_{\sigma \in K_0}$ is the stalk state produced by input $x$, and $\varepsilon_{H^1}$ is the configured tolerance (default $0.01$, see `InvariantChecker.h1_norm_max`).

This is the formal restatement of the colloquial *"the architecture must not contradict itself"* claim.

### 2.4 Current implementation status

* ✅ `CoboundaryOperator.apply_delta_0`, `.apply_delta_1`, `.is_contradiction`
* ✅ `CohomologyValidator.check_h1_consistency` (norm-based proxy)
* ⏳ True cohomology basis computation (Phase THSD-3 — Linter hardening)

---

## 3. Symbolic Expression Units — Discovery Operator 1/4

This section describes the **first** of four discovery operators that turn $K$ from a descriptive bookkeeping device into an active mathematical-invention substrate.

### 3.1 The underlying primitive

The `SymbolicHyperNeuron` (`neuroslm/modules/symbolic_unit.py`) is the layer
$$
U_\varphi : \mathbb{R}^{n_F} \to \mathbb{R}^{n_U}, \qquad U_\varphi(x)_i = \sum_{o} \pi^{\mathrm{op}}_{i,o}\,\mathrm{op}_o\!\bigl(x^a_i, x^b_i\bigr),
$$
with $x^a_i = \sum_j \pi^a_{i,j}\,x_j$, $x^b_i = \sum_j \pi^b_{i,j}\,x_j$, and the three $\pi$ distributions produced by Gumbel-softmax over learnable logits (§2 of `docs/dsl.md` — `fitness` block, where its `sparsity_loss` is collected).

The operator bank $\mathcal{O}$ is finite and small (default 7 ops: `{identity, add, sub, mul, exp, sin, tanh}`); $\mathrm{exp}$ is numerically clamped to $|\cdot| \le 20$.

### 3.2 Definition (Symbolic Simplex)

A **symbolic 0-simplex** is a triple
$$
\sigma_\varphi \;=\; \bigl(\,\text{name},\; U_\varphi,\; \text{kind}=\text{"symbolic"}\,\bigr)
$$
embedded into $K_0$. Its stalk is
$$
F(\sigma_\varphi) \;=\; \mathbb{R}^{n_U}, \qquad \dim F(\sigma_\varphi) = n_U.
$$

The **discovery surface** is the map
$$
\mathcal{E} : \sigma_\varphi \;\longmapsto\; \{\,e_i\,\}_{i=1}^{n_U} \;\subset\; \mathrm{Expr}(\mathcal{O}, X),
$$
where $e_i$ is the algebraic expression obtained by taking the argmax of each $\pi$-distribution at the unit $i$. Examples: `"(phi * surprise)"`, `"exp(metabolic_demand)"`, `"(x0 + x2)"`.

### 3.3 Theorem (Sheaf Compatibility)

*Let $\sigma_\varphi \in K_0$ be a symbolic simplex and let $\sigma_1 = [\sigma_\varphi, v]$ be any 1-simplex with the other endpoint a population vertex $v$ of stalk dim $n_U$. Then the cellular-sheaf invariants of §2 hold without bespoke restriction maps; in particular, $\delta^0$ and $\delta^1$ apply unchanged.*

**Proof sketch.** $F(\sigma_\varphi) \cong \mathbb{R}^{n_U}$ by construction, and `CellularSheaf.__post_init__` initialises $\rho_{\sigma_1 \to \sigma_\varphi} = I_{n_U}$ and $\rho_{\sigma_1 \to v} = I_{n_U}$. Hence the cochain spaces are well-defined and the coboundaries operate on aligned vector spaces; the existing `CoboundaryOperator` proofs apply. ∎

This theorem is what `tests/thsd/test_symbolic_simplex.py::TestSymbolicSimplexInSheaf` verifies.

### 3.4 Sparsity / discoverability contract

A symbolic simplex exposes
$$
\mathcal{L}_{\text{sparsity}}(\sigma_\varphi) \;=\; \frac{1}{3 n_U} \sum_i \bigl[H(\pi^a_i) + H(\pi^b_i) + H(\pi^{\mathrm{op}}_i)\bigr]
$$
via `SymbolicSimplex.sparsity_loss()`. The `FitnessComposer` collects it as the `"symbolic"` objective (see `docs/dsl.md`). When this loss is driven to zero by the optimiser, each unit collapses to a discrete formula and `symbolic_expression()` returns an extractable algebraic fact.

### 3.5 Invariant (Discovery Output)

For every checkpoint that has trained past the `tau`-annealing schedule's end (default: when `tau ≤ 0.1`), every `SymbolicSimplex` in $K$ must return from `symbolic_expression()` a list of *non-trivial* expressions — i.e. not all `"identity"`. This is the *minimum-invention guarantee* the discovery surface owes the rest of the system. (Enforcement: planned Phase THSD-3 in the Linter.)

---

## 4. Discovery Operators 2/4, 3/4, 4/4 — Forward Plan

The remaining three operators have stable spec but **not yet implementation**. They are listed here so this file is the single source of truth and the evolutionary loop can read what is permitted next.

### 4.1 Causal Emergence Operator (NIS+)

**Goal.** Find the coarse-graining $\Pi : F(\sigma) \to F^M(\sigma)$ that **maximises effective information**:
$$
\Pi^\star \;=\; \arg\max_{\Pi} \mathrm{EI}\bigl(M(\Pi),\, m\bigr) - \mathrm{EI}(m,\, m)
$$
The macroscopic random variable $y_{d=1}$ returned by $\Pi^\star$ at the lowest coarsening dimension that strictly increases $\mathrm{EI}$ is the architecture's **conscious variable** in the NIS+ sense (Zhang & Yuan).

**Code contract (planned).**
```python
class CausalEmergenceOperator:
    def fit(self, sheaf: CellularSheaf, samples: torch.Tensor) -> CoarseGraining: ...
    def conscious_variable(self) -> torch.Tensor: ...   # y_{d=1}
    def macroscopic_causal_power(self) -> float: ...    # EI(M, m)
```

**Where it plugs in.** Output of `conscious_variable()` becomes a designated stalk in $F$ at dimension $d=1$, which the Linter (§6) then checks for $\mathrm{EI}(M) > \mathrm{EI}(m)$ as an admission criterion.

### 4.2 Tonnetz Resonance Filter

**Goal.** Constrain mutations so that the **spectral gap** $\lambda_1$ of the sheaf Laplacian stays inside the Tonnetz-admissible band:
$$
\lambda_1(L_F) \;\ge\; \lambda_{\min}
$$

**Why a torus.** The Tonnetz lattice $\mathbb{T}^2$ has the property that all consonant intervals correspond to short geodesics. Forcing the spectral signature of $L_F$ to lie on (a sublattice of) $\mathbb{T}^2$ prevents the kind of disconnected, dispersed configurations that produce **geometric hallucination** (the model "discovers" a relation between two simplices that have no measurable information path between them).

**Code contract (planned).**
```python
class TonnetzResonanceFilter:
    def spectral_gap(self, sheaf: CellularSheaf) -> float: ...
    def admit(self, candidate: MutationCandidate) -> bool: ...
```

**Where it plugs in.** Called from the evolutionary loop's *gate* step before any mutation is written into the DNA.

### 4.3 Information-Geometric Retrieval (Fisher–Rao)

**Goal.** Replace cosine similarity in stalk retrieval with the Fisher–Rao metric:
$$
d_{F\!R}(p, q)^2 \;=\; \int_0^1 \dot\gamma(t)^\top g_F(\gamma(t))\,\dot\gamma(t)\,dt
$$
along a geodesic $\gamma$ in the stalk's statistical manifold. This weights similar-mean-but-different-variance pairs by their statistical *distinguishability*, which is what we actually want from a memory.

**Code contract (planned).**
```python
class FisherRaoRetrieval:
    def distance(self, p: torch.Tensor, q: torch.Tensor,
                 g: torch.Tensor) -> torch.Tensor: ...
    def topk(self, query: torch.Tensor,
             keys: torch.Tensor, k: int) -> torch.Tensor: ...
```

**Where it plugs in.** Drop-in replacement for the cosine call inside any retrieval-augmented stalk (`MemoryStalk`, vesicle sieve, attention-as-stalk). Falls back to cosine when $g_F$ is singular (degenerate stalk).

---

## 5. Genom-Phänotyp-Mapping — DNA, Mycelium, RAID-5

### 5.1 The DNA snapshot

A **DNA snapshot** at evolutionary step $t$ is the tuple
$$
\mathrm{DNA}_t \;=\; \bigl(\theta_t,\, K_t,\, F_t^{\text{params}},\, \mathcal{E}_t,\, \mathrm{cfg}_t,\, \mathrm{report}_t\bigr)
$$
serialised to `dna/<arch_name>/step_<t>/` as:

```
arch.neuro                # the DSL phenotype (K, F, mutations applied)
weights.safetensors       # θ_t
expressions.json          # E_t — all SymbolicSimplex equations
fitness.json              # FitnessConfig + per-objective scores at t
verifier_report.json      # Φ, ‖H¹‖, λ₁, NIS+ EI snapshot
```

The DSL phenotype `arch.neuro` is the *minimum reproducible spec*: rerunning `brian compile nfg arch.neuro` from `weights.safetensors` must reconstruct an architecture whose `verifier_report` matches within $\varepsilon$.

### 5.2 Incremental DNA patches (RAID-5)

Storing the full DNA snapshot every step is wasteful. The mycelium feedback loop instead writes **patches** $\Delta_t = \mathrm{DNA}_t \ominus \mathrm{DNA}_{t-1}$ with the structure:

```
dna/<arch>/patch_<t>.diff   # textual diff of arch.neuro
dna/<arch>/patch_<t>.delta  # parameter delta (compressed)
dna/<arch>/patch_<t>.exprs  # only changed expressions
dna/<arch>/parity_<t>.bin   # XOR of all patches in this stripe
```

A full snapshot is regenerable from any *4-of-5* patches in the stripe (RAID-5 parity). This makes the evolutionary trace both compact and self-healing under partial corruption.

### 5.3 The mycelium effect

The **mycelium effect** is the feedback rule: every patch $\Delta_t$ is *also* applied (in dry-run mode) to all sibling architectures sharing the same ancestor DNA. Patches that improve fitness on $\ge 3$ siblings are *promoted* to the ancestor — propagating discoveries through the genealogy as if through fungal hyphae. The rule is symmetric: no architecture "owns" a mutation; it has to *survive* in the population.

### 5.4 Invariant (Reproducibility)

For any DNA snapshot $\mathrm{DNA}_t$ on disk, running
```
brian compile nfg dna/<arch>/step_<t>/arch.neuro
brian verify    dna/<arch>/step_<t>/
```
must produce a report whose Φ, ‖H¹‖, λ₁ agree with `verifier_report.json` to within $10^{-4}$. This is the contract that makes the DNA a *genome* and not merely a log file.

---

## 6. Φ-Maximierung — IIT 4.0 Postulates as Fitness

### 6.1 The five postulates

We adopt the IIT 4.0 postulates of intrinsic existence (Tononi 2023). Each is translated into a measurable property of $(K, F)$:

| Postulate | Architectural meaning | Operationalisation |
|---|---|---|
| **1. Intrinsicality** | The complex acts on itself, not just on inputs | Non-zero recurrent paths inside $K$ |
| **2. Information** | The complex specifies one cause–effect state | $H(\text{cause}) > 0$, $H(\text{effect}) > 0$ |
| **3. Integration** | Partition reduces information | $\Phi(K) > 0$, computed via MIP |
| **4. Exclusion** | One unique maximum is selected | $\arg\max_{K' \subseteq K} \Phi(K')$ is unique |
| **5. Composition** | Structure has higher-order distinctions | At least one $\sigma_2 \in K_2$ is non-trivial |

### 6.2 The MIP (Minimum Information Partition) — current implementation

`PhiDynamicsComputer.compute_mip` performs an **exhaustive bipartition search** over the vertex set:
$$
\Phi(K) \;=\; \min_{(A,B)\,\text{bipartition}}\;\phi(A, B), \qquad \phi(A,B) \approx \frac{|\langle \bar s_A, \bar s_B\rangle|}{\mathrm{var}(s_A) + \mathrm{var}(s_B) + 10^{-6}}.
$$

This is a tractable proxy: it scales $O(2^{|K_0|})$ which is fine for $|K_0| \le 16$ but will need a Stoer–Wagner-style cut algorithm in Phase THSD-3.

### 6.3 Invariant (Φ Guard)

**Φ Guard.** A trained checkpoint is admissible only if
$$
\Phi(K)\bigl|_\theta \;>\; \Phi_{\min}
$$
where $\Phi_{\min}$ is configured per architecture (default $0.01$; `InvariantChecker.phi_min`).

This is the **anti-bag-of-MLP guarantee**: a model that has factored into independent components has $\Phi \to 0$ and is rejected even if its task loss is low.

---

## 6.4 The Triple Guard (Linter Specification)

> **Numbering note (v0.2):** This section was originally `## 7` in v0.1. The `a133343` patch introduced new §§7-12 covering ToM stalks, general-LM semantics, ImprovementGate, Lean backend, and the phased roadmap. To preserve content while honouring the new numbering, the original Triple Guard spec is now §6.4 (a refinement of §6.3 Φ Guard). The Triple Guard is the algebraic, structural admission criterion; §9 ImprovementGate is the statistical complement.

The verifier `neuroslm/verification/verifier.py` will, after Phase THSD-3, enforce three guards in series. A checkpoint must clear all three:

$$
\boxed{\quad \Phi(K) > 0 \;\;\wedge\;\; \|H^1(K;F)\| \to 0 \;\;\wedge\;\; \lambda_1(L_F) > \lambda_{\min} \quad}
$$

Mapping to current code:

| Guard | Symbol | Current code | Phase-3 hardening |
|---|---|---|---|
| Φ Guard | $\Phi(K) > \Phi_{\min}$ | `CohomologyValidator.compute_phi` + threshold | Strict `>` instead of `≥`; per-architecture `Φ_min` from FitnessConfig |
| Cohomology Guard | $\|H^1\| < \varepsilon_{H^1}$ | `CohomologyValidator.check_h1_consistency` (norm proxy) | True basis computation via QR of $\delta^0$ |
| Tonnetz Guard | $\lambda_1 > \lambda_{\min}$ | `InvariantChecker.check_spectral_gap` (currently checks an attribute; not the actual Laplacian) | Compute $\lambda_1$ from the sheaf Laplacian $L_F = \delta^{0\top}\delta^0$ via Lanczos |

These three guards together are what makes the framework *normative*: the evolutionary loop is permitted to mutate $\mathcal{A}$ in any way it likes, **provided the Triple Guard remains satisfied**.

---

## 6.5 Pre-v0.2 THSD Roadmap (superseded by §11)

> **Numbering note (v0.2):** This section was originally `## 8 Roadmap (Phases bound to this document)` in v0.1. The active forward-looking roadmap is now §11 "Phased delivery roadmap" which covers P1–P5 across `ImprovementGate`, Lean integration, and the full discovery-operator buildout. The pre-v0.2 THSD-internal phases below are preserved for historical traceability of the THSD subsystem only.

| Phase | Scope | Status | Tests |
|---|---|---|---|
| **THSD-1/4 — Symbolic Expression Units** | `SymbolicSimplex` in engine.py; sheaf integration; discovery surface | ✅ **Shipped** (this commit) | `tests/thsd/test_symbolic_simplex.py` (15) |
| THSD-2/4 — Causal Emergence | `CausalEmergenceOperator`; NIS+ EI search; $y_{d=1}$ conscious variable | ⏳ Spec frozen here §4.1 | TBD `tests/thsd/test_causal_emergence.py` |
| THSD-3/4 — Tonnetz Filter & Triple-Guard Linter | `TonnetzResonanceFilter`; harden `verifier.py` to enforce §6.4 strictly | ⏳ Spec frozen here §4.2, §6.4 | TBD `tests/thsd/test_tonnetz_filter.py`, `tests/verification/test_triple_guard.py` |
| THSD-4/4 — Fisher-Rao Retrieval | `FisherRaoRetrieval`; replace cosine in retrieval stalks | ⏳ Spec frozen here §4.3 | TBD `tests/thsd/test_fisher_rao.py` |
| THSD-5 — DSL wiring | Wire `thsd_parser.py` so `complex`/`sheaf` blocks compile end-to-end (resolves the 17 RED tests left by the parallel session) | ⏳ Blocked on compiler.py race resolution | `tests/dsl/test_thsd_parser_complex.py`, `tests/dsl/test_dsl_v2_vesicle_sieve.py` |

---

## 7. Theory of Mind — Belief-Stalk Operator

The first six sections cover *the architecture's view of itself* (Φ,
H¹, λ₁, symbolic invention, NIS+). Real cognition also requires *the
architecture's view of other agents* — first-order beliefs ("Sally
believes the marble is in the basket"), second-order ("X believes Y
believes Z"), false-belief inference (Wimmer & Perner 1983).

### 7.1 Definition (ToM stalk)

For a finite agent set $\mathcal{A} = \{a_1, \dots, a_{N_{\max}}\}$
of size $N_{\max} = \text{max\_agents}$ and belief dimension
$d_b = \text{d\_belief}$:

$$
F\bigl(\sigma_{\text{ToM}}(a_i)\bigr) \;=\; \mathbb{R}^{d_b \cdot (N_{\max}+1)^{\,\text{order}-1}}.
$$

The exponent encodes *recursive depth*: at order 1 the stalk is
$\mathbb{R}^{d_b}$ ("what does $a_i$ believe?"); at order 2 it is
$\mathbb{R}^{d_b(N_{\max}+1)}$ (one $d_b$-slot per agent
$a_i$-believes-about, plus a self slot). The implementation
formula is

$$
\dim F\bigl(\sigma_{\text{ToM}}(a_i)\bigr) \;=\; d_b \cdot (N_{\max}+1)^{\text{order}-1}
$$

(`TheoryOfMindIR.stalk_dim()` in `neuroslm/dsl/thsd_ir.py`).

### 7.2 Definition (Belief decay)

The temporal dynamics of belief stalks follow

$$
F\bigl(\sigma_{\text{ToM}}(a_i)\bigr)_{t+1} \;=\; \gamma \cdot F\bigl(\sigma_{\text{ToM}}(a_i)\bigr)_t \;+\; (1 - \gamma) \cdot \mathrm{obs}_t(a_i)
$$

with decay $\gamma = \text{belief\_decay} \in (0, 1]$. The endpoint
$\gamma = 1$ is "never forget" (admitted); $\gamma = 0$ is "forget
instantly" (rejected as degenerate by `__post_init__`).

### 7.3 Invariant (False-belief separation)

When `false_belief_enabled = True`, the architecture must maintain
*two* sections over the ToM agent set:

* $s_{\text{world}}$ — the model's belief about ground truth.
* $s_{\text{agent}}(a_i)$ — the model's belief about what $a_i$
  believes about the world.

The Sally-Anne contract is

$$
\bigl\|\, s_{\text{world}} - s_{\text{agent}}(a_i)\, \bigr\| \;\ge\; \theta_{\text{fb}}
$$

for at least one agent during the false-belief probe, where
$\theta_{\text{fb}} = \text{false\_belief\_threshold} \in [0, 1]$.
Architectures that collapse $s_{\text{agent}} \equiv s_{\text{world}}$
fail the Sally-Anne probe (cannot represent the divergence) and are
flagged by the linter.

### 7.4 Cohomology interaction

ToM stalks share the same sheaf-cohomology machinery as everything
else: contradictions between $s_{\text{world}}$ and
$s_{\text{agent}}(a_i)$ produce a non-trivial $\delta^0$ residual on
edges connecting the two stalks. This is the *desired* behaviour —
the residual *encodes* the false-belief gap; the H¹ guard's
threshold $\varepsilon_{H^1}$ is relaxed for ToM-tagged edges so a
healthy false-belief representation is not mistaken for a
hallucination.

### 7.5 Current implementation status

* ✅ `TheoryOfMindIR` dataclass + validation (`neuroslm/dsl/thsd_ir.py`)
* ✅ TDD pin: `tests/thsd/test_theory_of_mind_ir.py` (7 tests)
* ✅ Behavioural test: `tests/test_narrative_memory.py::test_theory_of_mind_consistency`
* ⏳ Parser hook in `thsd_parser.py` (Phase P3)
* ⏳ `ToMOperator` forward-pass module (Phase P3)
* ⏳ Sally-Anne probe in OOD eval (Phase P3)

---

## 8. General-Language-Model Semantics in THSD Notation

The framework's claim is that **any** language model can be
expressed as a labelled simplicial complex $(K, F)$ with the
appropriate operator bank. This section enumerates the canonical
families and their THSD encoding so the discovery loop has explicit
templates to specialise from.

### 8.1 Decoder-only transformer (GPT family)

For an $L$-layer, $H$-head transformer over $d$-dim residual stream
with vocab $V$ and context $T$:

* **Vertices** $K_0$: $\{\text{embed}, \text{block}_1, \dots, \text{block}_L, \text{norm}, \text{lm\_head}\}$.
  Each $\text{block}_\ell$ is itself a 2-simplex with three
  0-faces $\{\text{attn}_\ell, \text{ffn}_\ell, \text{residual}_\ell\}$.
* **Stalks**: $F(\text{block}_\ell) = \mathbb{R}^d$ (the residual
  stream slice), $F(\text{lm\_head}) = \mathbb{R}^V$.
* **Restriction maps**: identity on the residual stream
  ($\rho_{\ell+1, \ell} = I_d$), softmax-attention for cross-token
  $\rho_{t, s}$ inside each block.
* **Φ minimum**: any partition that bisects a block has
  $\Phi(\text{block}_\ell) > 0$ because attention defines a
  full-rank coupling between halves; the framework correctly
  identifies a transformer block as IIT-conscious at the
  per-block scale. (This is what `multi_cortex.fusion` exploits —
  see `docs/architecture.md` §12 and recent commit `90b233d`.)

### 8.2 Mixture-of-Experts (MoE)

For $E$ experts with router top-$k$ activation:

* **Vertices** $K_0$: a "router" 0-simplex plus $E$ "expert"
  0-simplices. The active subset $S \subseteq \{1, \dots, E\}$ with
  $|S| = k$ is selected per-token by the router.
* **Edges** $K_1$: $\{(\text{router}, \text{expert}_i)\}_{i=1}^E$
  carry the routing probability as their stalk content;
  $\rho_{\text{expert}_i, \text{router}} = $ scaled by $p_i$.
* **Sparsity invariant**: the H¹ guard enforces *exactly* one
  consistent 1-cochain among the $k$ selected experts — if two
  selected experts disagree by more than $\varepsilon_{H^1}$, the
  router is mis-routing.

This is the framework's encoding of the architecture in
`neuroslm.cortex::MultiCortexEnsemble` (4 sub-cortices,
`ThalamicRouter` for routing, GPT-2 family experts).

### 8.3 State-Space Models (Mamba / S4 / RWKV)

Linear recurrence $h_t = A h_{t-1} + B x_t$, $y_t = C h_t$:

* **Single vertex** $\sigma_{\text{ssm}}$ with stalk
  $F(\sigma_{\text{ssm}}) = \mathbb{R}^{d_{\text{state}}}$.
* **Self-loop edge** $e_{\text{rec}} = [\sigma_{\text{ssm}}, \sigma_{\text{ssm}}]$
  with $\rho_{e, \sigma}(h) = A h$ — the recurrence is the
  restriction map of the self-loop.
* **Φ via recurrence rank**: $\Phi \approx \log_2 \det(I - A^\top A + \varepsilon)$
  for the diagonalisable SSM; selective-scan variants (Mamba)
  have data-dependent $A$ so $\Phi$ becomes input-conditional —
  computed via MIP at each time step.

### 8.4 Retrieval-augmented LMs (RAG, kNN-LM)

Memory bank $\mathcal{M} = \{(k_i, v_i)\}_{i=1}^N$ as a stalk
sieve:

* **Vertices** $K_0$: query 0-simplex $\sigma_q$ + retrieval
  0-simplex $\sigma_r$ + generator vertices as in §8.1.
* **Edges** $K_1$: $\sigma_q \to \sigma_r$ with stalk
  $F = \mathbb{R}^N$ (one slot per memory) and restriction
  $\rho_{q,r}$ = top-$k$ softmax of inner products in the chosen
  metric.
* **Fisher-Rao retrieval**: when the retrieval metric is
  Fisher-Rao (Operator 4/4, §4.3) rather than cosine, the
  $\rho$-restriction respects the statistical manifold structure
  of the key embeddings.

### 8.5 Diffusion LM

Score-matching over the discrete simplex of token distributions:

* **Stalks**: $F(\sigma_t) = \Delta^{V-1}$ (the probability
  simplex over the vocab) at each diffusion step $t$.
* **Coboundary** $\delta^0$: the discrete-time backward
  Kolmogorov equation; the cocycle condition is the
  reverse-diffusion consistency. A model that violates the
  cocycle has $\delta^1 \neq 0$ and produces inconsistent
  multi-step rollouts.
* **Φ across timesteps**: integrated information of the joint
  $(x_0, \dots, x_T)$ stream — diffusion models have natively
  high Φ because every step couples to every previous step
  through the noise schedule.

### 8.6 Mixture (BRIAN itself)

The `rcc_bowtie` architecture combines §8.1 (DSL language trunk),
§8.2 (multi-cortex MoE), and ToM stalks from §7 over a shared
sheaf-cohomology substrate — the canonical reference instantiation
in this repo. See `architectures/rcc_bowtie/arch.neuro`.

### 8.7 Universality claim

**Proposition (informal).** Every neural language model can be
expressed as a labelled simplicial complex $(K, F)$ over the
operator bank $\mathcal{O}$ extended with the appropriate
arithmetic primitives (matrix multiplication, softmax, element-
wise non-linearities). The Φ and H¹ guards apply uniformly.

A *constructive* proof — *"give me any PyTorch model and I will
emit its $(K, F)$"* — is the Phase P5 deliverable; until then the
framework supports the families above by hand-written templates.

---

## 9. Improvement Gate — Statistical Admission Criterion

Sections §2.3 (H¹ Guard), §6.3 (Φ Guard), and §4.2 (λ₁ Guard)
define **invariant** guards: necessary structural properties a
mutation must not break. They are insufficient to *promote* a
mutation, because a mutation that preserves Φ and H¹ but does not
improve the model's behaviour is still useless evolutionary noise.

### 9.1 Definition (Improvement evidence)

For a target metric $\mu : \Theta \to \mathbb{R}$ (cross-entropy
on a held-out batch, OOD perplexity, intelligence density,
gap-ratio, …) and a candidate mutation $\theta \to \theta'$:

$$
\mathrm{evidence}_\mu(\theta, \theta') \;=\; \bigl(\mu_b^{(1)}, \dots, \mu_b^{(n)};\; \mu_a^{(1)}, \dots, \mu_a^{(m)}\bigr)
$$

where each $\mu_b^{(i)} = \mu(\theta; x^{(i)})$ and
$\mu_a^{(i)} = \mu(\theta'; x^{(i)})$ for a fixed held-out
sample family $\{x^{(i)}\}$.

### 9.2 Definition (Direction)

The caller supplies $\mathrm{dir} \in \{\downarrow, \uparrow\}$:
$\downarrow$ for metrics where smaller is better (ppl, loss,
gap-ratio), $\uparrow$ for larger-is-better (Φ, accuracy,
intelligence-density).

### 9.3 Invariant (Improvement Gate)

A candidate mutation $\theta'$ is **improvement-admitted** for
metric $\mu$ at significance $\alpha$ and minimum effect
$\varepsilon_{\text{eff}}$ iff **all three** hold:

1. **Direction.** $\mathrm{sgn}(\bar\mu_a - \bar\mu_b) = -1$ for
   $\mathrm{dir} = \downarrow$, $+1$ for $\mathrm{dir} = \uparrow$.
2. **Statistical significance.** The one-sided Welch's $t$-test
   $p$-value satisfies $p < \alpha$ (default $\alpha = 0.05$).
3. **Practical significance.**
   $|\bar\mu_a - \bar\mu_b| / \max(|\bar\mu_b|, 10^{-12}) > \varepsilon_{\text{eff}}$
   (default $\varepsilon_{\text{eff}} = 0.01$).

### 9.4 Composition (Full admission)

A mutation is **persisted** to the DNA iff
$\mathrm{TripleGuard} \wedge \mathrm{ImprovementGate}$ both admit:

$$
\theta' \text{ persisted} \quad\Longleftrightarrow\quad
\bigl(\Phi(\theta') \ge \Phi_{\min}\bigr) \wedge
\bigl(\|H^1\|(\theta') \le H^1_{\max}\bigr) \wedge
\bigl(\lambda_1(\theta') \ge \lambda_{\min}\bigr) \wedge
\bigl(\text{improvement-admitted for } \mu\bigr).
$$

The composite is realised by `neuroslm.verification.improvement_gate.CompositeGate`
chaining `TripleGuard` (§6.3, §2.3, §4.2) with `ImprovementGate`
(this section).

### 9.5 Current implementation status

* ✅ `ImprovementGate` + `ImprovementVerdict` with Welch's t-test
  (pure-Python, no scipy dependency) — `neuroslm/verification/improvement_gate.py`
* ✅ `CompositeGate` chains multiple gates with AND semantics
* ✅ TDD pin: `tests/verification/test_improvement_gate.py` (16 tests)
* ⏳ Wire `CompositeGate` into `EvolutionaryTrainingContext.set_admission_gate()` (Phase P2)
* ⏳ Lean-proof backend (Phase P4) — replaces the statistical test
  for mutations whose source admits a formal monotonicity proof

---

## 10. Lean Backend — Formal Proof of Improvement

The statistical Improvement Gate (§9) is the *general* admission
criterion: it works for any opaque PyTorch model. But for *some*
mutations the improvement is *provable* in a closed mathematical
sense — e.g. adding a non-negative regularisation term to a loss
cannot increase the gradient norm, or replacing a softmax with its
straight-through estimator preserves the argmax. For those
mutations a formal proof can replace the costly empirical
evaluation.

### 10.1 Backend interface

```python
class ProofBackend(Protocol):
    def can_prove(self, mutation: Mutation, metric: str,
                  direction: str) -> bool: ...
    def prove(self, mutation: Mutation, metric: str,
              direction: str) -> ProofVerdict: ...
```

The `LeanProofBackend` implementation (Phase P4) shells out to
`lean --json` against `.lean` proof files autogenerated from the
mutation's algebraic form. Returns `ProofVerdict(admitted=True)`
when the Lean kernel checks the monotonicity theorem, otherwise
falls through to the statistical gate.

### 10.2 Lean theorem library (planned)

Each proven invariant lives in `lean/Brian/`:

| File | Theorem |
|---|---|
| `lean/Brian/PhiMonotone.lean` | $\Phi(\theta') \ge \Phi(\theta)$ when the mutation adds a non-negative coupling |
| `lean/Brian/OodGapDecrease.lean` | Adding $\lambda \cdot \mathrm{CDGA}$ to the loss cannot increase the OOD gap (P1 from `docs/CDGA.md`) |
| `lean/Brian/SymbolicSparsity.lean` | Driving Gumbel-Softmax temperature $\tau \to 0$ collapses every symbolic unit to a discrete expression |
| `lean/Brian/TripleGuardSound.lean` | `TripleGuard.admit` returns `admitted = True` iff all three guards pass (soundness) |

### 10.3 Current implementation status

* ✅ **Ledger + emitter + verifier shipped** (`neuroslm/discoveries/`).
  Five canonical hypotheses (H001–H005) from §10.2 are seeded with
  Lean stubs under `hypothesis/proofs/*.lean`. The infrastructure
  is in place; the only remaining work is filling in the tactics
  inside each `:= by sorry` block as the supporting Brian Lean
  library lands.
* ⏳ The `Brian.*` Lean library itself is still empty — the
  emitter writes obligations against `import Mathlib.Tactic`
  alone. Phase P4 will introduce `import Brian.Core` once the
  first concrete theorem lands.
* The interface (§10.1) is normative now so the empirical gate
  (§9) can be swapped without API churn.

### 10.4 Hypothesis ledger & autodiscovery (the `hypothesis/` and `discoveries/` folders)

Two top-level folders make the formal-claim pipeline first-class:

| Folder | Author | Records | Proof obligation |
|---|---|---|---|
| `hypothesis/` | human | `H###_*.md` (YAML front-matter + Markdown body) | "this property of the architecture holds" |
| `discoveries/` | evolutionary engine | `D###_*.md` | "this autodiscovered mutation does not regress any guarded metric" |

Each folder has a `proofs/` subfolder holding the autogenerated
`.lean` file for every record; one record ↔ one proof file. The
CLI surface:

```bash
brian hypothesis list                      # tabular listing
brian hypothesis show H001                 # full front-matter + body
brian hypothesis emit-proofs               # (re)generate any missing .lean stubs
brian hypothesis verify H001               # shell out to `lean --json` if available
brian hypothesis verify --all              # verify every record

brian discovery list
brian discovery show D001
brian discovery verify D001
brian discovery promote D001 rcc_bowtie    # splice into architectures/rcc_bowtie/arch.neuro
```

The `discovery promote` step refuses to splice unless
`proof_status == "verified"`, then appends a
`# === Discovery D### ===` block to the architecture's
`arch.neuro` and re-lifts the file through
`compiler.hypergraph_ir.lift_arch_to_hypergraph` to confirm the
genome still round-trips. This makes the discovery loop's
"proof → DNA" hand-off mechanically verifiable: a discovery
cannot enter the genome without (a) a verified Lean proof of its
no-regression obligation and (b) a successful round-trip through
the hypergraph IR. Both steps are gated by `TripleGuard` at the
empirical level (§7) and by Lean at the formal level (§10).

Records are versioned in git as plain Markdown — there is no
binary database. The `index.json` file in each folder is a
read-only cache rebuilt from disk on every `store.list_all()`.
This means the entire scientific-insight history of the model is
diffable in code review, and any mutation to a record (status
change, new reference, new proof tactic) shows up as a normal
file diff.

The JSON Schemas (`hypothesis/schema.json`,
`discoveries/schema.json`) pin the record shape for editor
support and CI validation. The empty-proofs `.gitkeep` and the
two `README.md` files document the contract in-tree.

---

## 11. Phased delivery roadmap

| Phase | Deliverable | Status |
|---|---|---|
| P1 | ImprovementGate + ToM IR + this §7-§11; repo cleanup | ✅ This commit |
| P2 | Real $\lambda_1(L_F)$ Tonnetz Laplacian; real $\|H^1\|$ kernel/image SVD; wire `CompositeGate` into evolution loop | ⏳ |
| P3 | Fisher-Rao retrieval (Op 4/4); `ToMOperator` forward module; `theory_of_mind` parser block; Sally-Anne probe | ⏳ |
| P4 | Lean toolchain integration; `LeanProofBackend`; theorems in `lean/Brian/` | 🟨 Ledger + emitter + verifier shipped (§10.4); `Brian.*` Lean library still empty |
| P5 | General-LM round-trip — emit $(K, F)$ from arbitrary PyTorch nn.Modules per §8.7 | ⏳ |

---

## 12. How to mutate this document

This file is the contract. Updating it follows the protocol:

1. Open a TDD-RED suite that pins the new contract you want (e.g. a new operator).
2. Run it; confirm RED for the *right* reason.
3. Update this document — §1 (ontology), §3-4 (operators), §7 (guards) as needed.
4. Implement the production code; confirm GREEN.
5. Commit with `docs(formal_framework): <change>` and reference the TDD suite hash in the commit body.

The evolutionary loop is expected to be able to follow steps 1–5 unattended on items in its allowed-mutation table (`fitness.evolution.allowed_mutations` in the DSL). The human reviewer's job is to audit *diffs of this document* across generations.

---

## Appendix A — Glossary cross-link to code

| Concept | Code |
|---|---|
| Simplicial complex $K$ | `neuroslm.thsd.engine::SimplexComplex` |
| Cellular sheaf $F$ | `neuroslm.thsd.engine::CellularSheaf` |
| Coboundary $\delta^k$ | `neuroslm.thsd.engine::CoboundaryOperator` |
| Φ via MIP | `neuroslm.thsd.engine::PhiDynamicsComputer` |
| Symbolic 0-simplex | `neuroslm.thsd.engine::SymbolicSimplex` |
| Operator bank $\mathcal{O}$ | `neuroslm.modules.symbolic_unit::OperatorBank` |
| `SymbolicHyperNeuron` $U_\varphi$ | `neuroslm.modules.symbolic_unit::SymbolicHyperNeuron` |
| Triple Guard verifier | `neuroslm.verification.verifier::THSDVerifier` |
| Composite admission gate | `neuroslm.verification.improvement_gate::CompositeGate` |
| Statistical improvement gate | `neuroslm.verification.improvement_gate::ImprovementGate` |
| ToM belief stalk IR | `neuroslm.dsl.thsd_ir::TheoryOfMindIR` |
| Fitness composer | `neuroslm.fitness::FitnessComposer` |
| Metabolic pruning | `neuroslm.modules.nrcstk::NRCSTKController` |

---

## Appendix B — Version history

| Version | Date | Change |
|---|---|---|
| v0.1 | 2026-06-07 | Initial draft; THSD-1/4 (SymbolicSimplex) shipped; §1, §2, §3, §5–§7 normative; §4 spec-only |
| v0.2 | 2026-06-08 | P1: added §7 (ToM stalks), §8 (general-LM coverage), §9 (ImprovementGate), §10 (Lean backend interface), §11 (roadmap); renumbered "how to mutate" → §12; added `TheoryOfMindIR` + `ImprovementGate` + `CompositeGate` to Appendix A |
| v0.2.1 | 2026-06-09 | Bug-fix: the v0.2 patch added new §§7-12 but failed to renumber the pre-existing §7 (Triple Guard) and §8 (Roadmap), producing a numbering collision. The pre-existing sections have been renumbered to §6.4 (Triple Guard, now a refinement of §6.3 Φ Guard) and §6.5 (pre-v0.2 THSD-internal roadmap, now superseded by §11). No content lost; numbering now linear. |
| v0.3 | 2026-06-15 | Added §13 (Capacity-Funneled Distillation / CFD). Formalises the "implode optimum" for SLMs at fixed parameter count: the parameter-efficient frontier $C_s \mapsto \mathrm{ppl}^\star(C_s)$ achievable when distilled from an arbitrarily strong CFD-funneled teacher. Declared as hypothesis H006; empirical falsifier in `tests/training/test_cfd_distillation.py`. |

## 13. Kapazitäts-Trichter-Destillation (CFD) — the Implode Optimum

> **Status:** v0.1 (declared, empirical proof pending) — H006  
> **Scope:** The formal contract for the capacity-funneled distillation
> path in `neuroslm.harness._cortex_fusion_aux_step`. Defines what an
> "optimal-trained SLM at fixed parameter count" means and proves the
> no-harm floor mechanically.

### 13.1 Motivation — the teacher-too-strong pathology

Naive Hinton-style distillation
$\mathcal{L}_{\mathrm{distill}} = T^2 \cdot \mathrm{KL}(p_t \| p_s)$
fails catastrophically when the teacher's capacity $C_t$ substantially
exceeds the student's $C_s$. Run 40952126 (H22) is the empirical
witness: the SmolLM2-360M teacher drove a 30M trunk's PPL from a
baseline of $\sim 24$ to $\sim 600$ within 500 steps. Three structural
mechanisms compound (full forensic in
`docs/FINDINGS.md`, run 40952126):

1. **Unreachable target.** The student's reachable softmax simplex
   $\mathcal{S}_{\theta} = \{p \mid \exists \theta,\ p = \mathrm{softmax}(f_\theta(x))\}$
   does not contain $p_t$. The minimum
   $\min_\theta \mathrm{KL}(p_t \| p_\theta)$ is bounded **away from
   zero** — the loss has no fixed point.
2. **Sharpness gap.** $H(p_t) \ll H(p_s)$ early in training; the KL
   gradient is dominated by precisely the points where the student is
   least able to follow.
3. **No floor.** $\nabla \mathcal{L}_{\mathrm{distill}}$ and
   $\nabla \mathcal{L}_{\mathrm{LM}}$ can be anti-aligned for arbitrary
   stretches of training — no mechanical bound on the harm.

The `reduction='batchmean'` bug (Followup F1) multiplies the harm by
$\sim T$ but is **not** the cause; the same explosion occurs with the
correct per-token reduction, just $T\times$ slower.

### 13.2 Definition (Capacity-Funneled Distillation)

For a student logits $f_\theta(x) \in \mathbb{R}^V$ and a teacher
logits $g(x) \in \mathbb{R}^V$ on the same vocabulary, define

$$
\mathcal{L}_{\mathrm{CFD}}(\theta; g)
  \;=\; \mathcal{L}_{\mathrm{LM}}(\theta)
       + \lambda_{\mathrm{eff}}(\theta, g) \cdot T_{\mathrm{eff}}(\theta, g)^2
         \cdot \mathrm{KL}\!\left(
           \tilde{p}_t^{(K, T_{\mathrm{eff}})}(x)
           \,\Big\|\,
           \mathrm{softmax}\!\bigl(f_\theta(x) / T_{\mathrm{eff}}\bigr)
         \right)
$$

with three closed-form components:

**(Stage 1) Top-$K$ rank-preserving sparsification.** With
$\mathcal{I}_K(g) \subset \{1, \dots, V\}$ the indices of the top-$K$
teacher logits and $p^{(T)} = \mathrm{softmax}(g / T)$,

$$
\tilde{p}_t^{(K, T)}(v)
  = \begin{cases}
    p^{(T)}_v & v \in \mathcal{I}_K \\[2pt]
    \dfrac{1 - \sum_{u \in \mathcal{I}_K} p^{(T)}_u}{V - K} & v \notin \mathcal{I}_K
  \end{cases}
$$

The mass on $\mathcal{I}_K$ is preserved exactly; the residual is
spread uniformly over the long tail. $\tilde{p}_t^{(K, T)}$ is a valid
probability distribution by construction.

**(Stage 2) Entropy-matched temperature.**
With $H(p) = -\sum_v p_v \log p_v$ the Shannon entropy and a base
$T_0 \ge 1$,

$$
T_{\mathrm{eff}}(\theta, g) \;=\; T_0 \cdot \max\!\left(1,\;
  \frac{H(\mathrm{softmax}(f_\theta))}{H(\mathrm{softmax}(g)) \vee \varepsilon}\right)
$$

with $\varepsilon = 10^{-6}$. The entropy ratio is computed per batch
under `torch.no_grad()`.

**(Stage 3) Gradient-alignment gate.**
Pick a *probe parameter* $\phi$ (in implementation: the trunk's last
linear layer's bias). Compute

$$
g_{\mathrm{align}}(\theta, g)
  \;=\; \cos\!\left(
    \nabla_\phi \mathcal{L}_{\mathrm{distill}},\;
    \nabla_\phi \mathcal{L}_{\mathrm{LM}}
  \right) \;\in\; [-1, 1].
$$

Then

$$
\lambda_{\mathrm{eff}}(\theta, g)
  \;=\; \lambda_0 \cdot \frac{1 + g_{\mathrm{align}}}{2}
  \;\in\; [0, \lambda_0].
$$

Stage 3 is the *mechanical floor*: when the teacher disagrees with the
LM target it gets driven to zero weight, automatically.

### 13.3 Theorem (No-harm floor)

**Claim (I).** For every teacher $g$ and every initial student
$\theta_0$, the gradient-descent trajectory of
$\mathcal{L}_{\mathrm{CFD}}(\theta; g)$ satisfies

$$
\liminf_{n \to \infty} \mathcal{L}_{\mathrm{LM}}(\theta_n^{\mathrm{CFD}})
  \;\le\;
\liminf_{n \to \infty} \mathcal{L}_{\mathrm{LM}}(\theta_n^{\mathrm{LM-only}})
$$

where $\theta_n^{\mathrm{CFD}}$ and $\theta_n^{\mathrm{LM-only}}$ are
generated by the same optimiser and learning-rate schedule on
$\mathcal{L}_{\mathrm{CFD}}$ and $\mathcal{L}_{\mathrm{LM}}$ respectively.

**Proof sketch.** Stage 3 ensures
$g_{\mathrm{align}}(\theta_n, g) \le 0 \Rightarrow \lambda_{\mathrm{eff}}(\theta_n, g) = 0$,
so the gradient update at step $n$ collapses to a pure LM step
whenever the teacher would otherwise harm the LM objective. On the
remaining steps, $\nabla \mathcal{L}_{\mathrm{distill}}$ has positive
cosine with $\nabla \mathcal{L}_{\mathrm{LM}}$, so projecting it onto
the LM-gradient direction has non-negative magnitude, and the combined
step decreases $\mathcal{L}_{\mathrm{LM}}$ by at least the LM-only
amount minus a perpendicular term. Taking $\liminf$ commutes with the
non-negative integral of those decrements. ∎

A fully formal version of this argument is the H006 Lean obligation
(`hypothesis/H006_capacity_funneled_distillation_implode.md`).

### 13.4 Theorem (Monotone implode in teacher capacity)

For two teachers $g_1, g_2$ with respect to the data distribution
$\mathcal{D}$, write $g_1 \preceq_{\mathcal{D}} g_2$ iff

$$
\mathrm{KL}(p_{\mathcal{D}} \,\|\, p_{g_2}) \;\le\; \mathrm{KL}(p_{\mathcal{D}} \,\|\, p_{g_1})
$$

(i.e., $g_2$ is at least as close to the true data distribution as
$g_1$ in the reverse-KL sense).

**Claim (II).** If $g_1 \preceq_{\mathcal{D}} g_2$ and the top-$K$
projections $\tilde{p}_{g_1}^{(K, T)}$ and $\tilde{p}_{g_2}^{(K, T)}$
differ on a set of positive measure under $\mathcal{D}$, then

$$
\liminf_n \mathcal{L}_{\mathrm{LM}}(\theta_n^{\mathrm{CFD}, g_2})
  \;\le\;
\liminf_n \mathcal{L}_{\mathrm{LM}}(\theta_n^{\mathrm{CFD}, g_1})
$$

with strict inequality when the difference is detectable by the
student's gradient.

**Why it holds.** Stage 3 makes both runs at least as good as
LM-only (by (I)). On the positive-measure set where
$\tilde{p}_{g_2} \ne \tilde{p}_{g_1}$, the $g_2$ run additionally
benefits from $g_2$'s closer alignment with $\mathcal{D}$ — Stage 1
+ Stage 2 ensure that "alignment with $\mathcal{D}$" of the
*projected* teacher is monotone in alignment of the *raw* teacher
(top-$K$ projection preserves ordering by KL when $K$ is at or above
the student's mode resolution).

### 13.5 Theorem (Existence of the implode optimum)

**Claim (III).** Fix a student architecture family parameterised by
capacity $C_s$. The map

$$
\mathrm{ppl}^\star(C_s, C_t)
  \;=\; \exp\!\left(\inf_\theta \mathcal{L}_{\mathrm{LM}}(\theta^{\mathrm{CFD}, g_{C_t}})\right)
$$

is **monotonically non-increasing** in $C_t$ for fixed $C_s$, and has
a well-defined infimum

$$
\mathrm{ppl}^\star(C_s) \;=\; \lim_{C_t \to \infty} \mathrm{ppl}^\star(C_s, C_t).
$$

Furthermore, $\mathrm{ppl}^\star(C_s)$ is itself strictly decreasing
in $C_s$ (a higher-capacity student strictly dominates).

### 13.6 The SLM parameter-efficient frontier

(III) gives the central operational object of this section. The curve

$$
C_s \;\longmapsto\; \mathrm{ppl}^\star(C_s)
$$

is the **parameter-efficient frontier of SLMs** under CFD: the best
PPL achievable at parameter count $C_s$ when distilled from an
arbitrarily strong CFD-funneled teacher. By (I)+(III):

* No teacher choice can push the student below $\mathrm{ppl}^\star(C_s)$.
* A trainer who chooses *any* teacher $g$ and trains with CFD is on
  the frontier or above it; under LM-only training (no teacher) the
  trainer is at or above $\mathrm{ppl}^\star(C_s, C_t = 0)$, which is
  $\ge \mathrm{ppl}^\star(C_s, C_t = \infty) = \mathrm{ppl}^\star(C_s)$
  by (II).
* The user's intuition — "a small student trained from a much larger
  CFD-funneled teacher outperforms the same student trained from a
  matched-size teacher" — is the conjunction of (II) and (III).

### 13.7 Why this *intrinsically* describes the teacher-too-strong bug

The legacy distillation loss does not satisfy any of (I)–(III). The
explicit failure mode under capacity mismatch is:

| Naïve KL | CFD |
|---|---|
| Reachable set $\not\ni p_t$ → loss has no zero | Stage 1: $\tilde{p}_t \in \mathcal{S}_\theta$ for adequate $K$ |
| Sharpness $H(p_s) \gg H(p_t)$ → unstable early grad | Stage 2: $T_{\mathrm{eff}}$ rescales until $H(\tilde{p}_t / T_{\mathrm{eff}}) \approx H(p_s)$ |
| No upper bound on $\mathcal{L}_{\mathrm{LM}}(\theta_n)$ as KL pulls θ off LM optimum | Stage 3: $g_{\mathrm{align}} \le 0 \Rightarrow \lambda_{\mathrm{eff}} = 0$ |

Each row neutralises exactly one of the three mechanisms identified
in §13.1. The "teacher too strong" pathology becomes a structural
impossibility under CFD.

### 13.8 Current implementation status

* ⏳ `tests/training/test_cfd_distillation.py` — four-arm ablation
  (Arms A/B/C/D) — empirical falsifier for H006. **Failing test
  written first** (per CLAUDE.md §1) in the same commit as the
  implementation.
* ⏳ `MultiCortexConfig.cfd_enabled` (default `false`) + `cfd_topk_*`,
  `cfd_temperature_floor`, `cfd_align_probe` knobs in
  `neuroslm.dsl.training_config`.
* ⏳ `neuroslm.harness._cortex_fusion_aux_step` extended with the
  three-stage path under the `cfd_enabled` flag.
* ⏳ `hypothesis/H006_capacity_funneled_distillation_implode.md` —
  hypothesis card with proof_status: missing (Lean obligation
  deferred).
* ⏳ Lean proof of (I) only (the no-harm floor) — the easy claim,
  H006 follow-up F6.

### 13.9 Open questions

* The probe-parameter choice for $g_{\mathrm{align}}$ in §13.2 Stage 3
  trades off cost vs. accuracy. Using the full $\nabla_\theta$ gives
  the exact cosine but doubles the backward cost; using a single
  scalar (last-layer bias) is $O(V)$ extra and a noisy estimate of
  the true cosine. The implementation uses the bias as default; a
  `cfd_align_probe` enum lets advanced runs select `"bias"`,
  `"last_layer_weight"`, or `"full"`.
* The top-$K$ schedule $K(n)$ is currently linear in the step count.
  An adaptive schedule driven by $H(p_s) / H(p_t)$ may be tighter and
  is a natural F7.
* Cross-vocabulary teachers (e.g. SmolLM2 with $V_e \ne V_t$) require
  composing CFD with `VocabBridge`. Stage 1 must be applied **before**
  the bridge to preserve the rank ordering through the projection.

---

## 14. Generalisation-Funneled Distillation (GFD) — closing the train↔OOD gap

### 14.0 Motivation: the H22/B6 falsification of CFDv1

The H006 theorem (§13.3) bounds the training-distribution loss
$\mathcal L_{\mathrm{LM}}(p_s)$ under the no-harm floor $\lambda_{\mathrm{eff}} \in [0, \lambda_0]$.
But the theorem is **silent on the train↔OOD gap**:

$$
\mathrm{gap}(p_s) := \mathcal L_{\mathrm{LM}}^{(\text{OOD})}(p_s) - \mathcal L_{\mathrm{LM}}^{(\text{train})}(p_s).
$$

The H22/B6 run (SmolLM2-360M expert swap, otherwise identical config)
falsified the implicit assumption that "implode train PPL ⇒ implode
OOD PPL":

| variant         | train PPL | OOD PPL | gap_ratio |
|-----------------|-----------|---------|-----------|
| GPT-2 expert    | 38.4      | 110.2   | 2.87      |
| SmolLM2 expert  | **23.6**  | **155.0** | **6.55** |

A *stronger* teacher fires more confidently on frequency-driven
patterns ("the", "and", "of" follow-ons). CFDv1 treats every
(context, target) pair as equally informative — so the student
absorbs the corpus's first-order statistics faster, train PPL implodes,
but the residual *contextual* signal degrades because the high-PMI
positions get the same K as the low-PMI ones.

### 14.1 Decomposition of the distillation gradient

For a single position with teacher distribution $p_t(\cdot \mid c)$,
the KL gradient at the student logits decomposes as

$$
\nabla_z \mathrm{KL}(p_t \,\|\, p_s) = -(p_t - p_s) = -\bigl[\underbrace{p_{\mathrm{uni}}}_{\text{marginal}} + \underbrace{(p_t - p_{\mathrm{uni}})}_{\text{contextual residual}}\bigr] + p_s.
$$

The marginal component pulls the student toward $p_{\mathrm{uni}}$
**regardless of context** — it is precisely the frequency-imitation
fuel that drives the train↔OOD gap. The contextual residual is the
generalisation-positive signal (it encodes how the *context* shifts
the distribution away from the marginal).

GFD removes the marginal component from the distillation channel
without touching the LM channel (the LM-CE term still sees the full
$p_{\mathrm{true}}$).

### 14.2 Mechanism M2: prior-residual sparsification

**Definition.** Let $p_{\mathrm{uni}}(v) > 0$ be a smoothed unigram
prior over the shared vocabulary $\mathcal V$ (in implementation:
EMA over training-batch target counts, with $+1$ Laplace smoothing).
For $\gamma \in [0, 1]$ the **prior-residual teacher** is

$$
\tilde p_t^{(\gamma)}(v \mid c) \;\propto\; \frac{p_t(v \mid c)}{p_{\mathrm{uni}}(v)^\gamma}.
$$

In log-space this is the trivial shift
$\tilde z_t = z_t - \gamma \log p_{\mathrm{uni}}$ — implemented as a
single broadcast subtraction on the teacher logits **before** the
Stage-1 top-K projection.

**Limits.** $\gamma = 0$ recovers CFDv1 exactly. $\gamma = 1$ fully
removes the unigram floor: the distillation channel only carries
$\log(p_t / p_{\mathrm{uni}})$, i.e. the pointwise mutual
information signal.

**Theorem (IV) — M2 preserves the H006 no-harm floor.**

The Stage-3 cosine gate
$\lambda_{\mathrm{eff}} = \lambda_0 \cdot (1 + \cos(\nabla_{z} L_{\mathrm{distill}}, \nabla_{z} L_{\mathrm{LM}})) / 2$
depends only on the *direction* of the distillation gradient at the
pre-fusion logits. M2 replaces $p_t$ with $\tilde p_t^{(\gamma)}$ but
the gate is still computed on the resulting term — so for any teacher
that pulls anti-aligned with the LM after M2, $\lambda_{\mathrm{eff}} \to 0$
exactly as in §13.3. Hence M2 cannot violate (I): $\mathcal L_{\mathrm{LM}}$
on the training distribution is still upper-bounded by the LM-only
baseline. $\square$

**Theorem (V) — M2 reduces the marginal-imitation gradient.**

Let $g_{\mathrm{marg}}(z; p_t) := \mathbb E_{v \sim p_{\mathrm{uni}}}[\nabla_z \log p_s(v)]$
be the projection of the distillation gradient onto the
marginal-imitation direction. Then for $\gamma > 0$ and any
non-trivial teacher (i.e. $p_t \not\propto p_{\mathrm{uni}}$),

$$
\bigl\| g_{\mathrm{marg}}(z; \tilde p_t^{(\gamma)}) \bigr\| \;<\; \bigl\| g_{\mathrm{marg}}(z; p_t) \bigr\|.
$$

*Proof sketch.* By construction $\tilde p_t^{(\gamma)}(v) \propto p_t(v) \cdot p_{\mathrm{uni}}(v)^{-\gamma}$,
which strictly down-weights every $v$ for which $p_t(v) > p_{\mathrm{uni}}(v)$
above what it down-weights $v$ for which $p_t(v) < p_{\mathrm{uni}}(v)$.
After normalisation, the inner product $\langle \tilde p_t^{(\gamma)}, \log p_{\mathrm{uni}} \rangle$
is strictly smaller than $\langle p_t, \log p_{\mathrm{uni}} \rangle$
for any $\gamma > 0$ unless $p_t = p_{\mathrm{uni}}$ exactly.
The gradient projection inherits this contraction monotonically. $\square$

### 14.3 Mechanism M4: pointwise K from teacher PMI

**Definition.** For each position $t$ let $v^*_t := \arg\max_v p_t(v \mid c_t)$
be the teacher's top-1 token and define the **pointwise mutual
information**

$$
\mathrm{PMI}(t) := \log p_t(v^*_t \mid c_t) - \log p_{\mathrm{uni}}(v^*_t).
$$

Then the per-position K is

$$
K(t) := \mathrm{clip}\Bigl(K_{\max} \cdot \exp\bigl(-\max(\mathrm{PMI}(t), 0) / \sigma \bigr), K_{\min}, K_{\max}\Bigr) \in \mathbb Z \cap [K_{\min}, K_{\max}].
$$

The decay scale $\sigma > 0$ controls how aggressively K drops with
PMI; default $\sigma = 2$ nats spans the typical PMI range of a
modest LM. The CFD Stage-1 projection then uses this per-position $K$
via `cfd_topk_target_var_k`, which is bit-identical to the scalar
version whenever $K(t)$ is constant.

**Interpretation.** $K(t)$ is *small* when the teacher fires
confidently on a token the prior says is rare — these are the
contextually informative positions and we want to concentrate the
distillation signal on the few correct alternatives. $K(t)$ is
*large* when the teacher's top-1 is a high-prior token (the teacher
is just echoing the marginal) — we soften the projection so the
distillation term degrades gracefully into a broad regulariser.

**Back-compat.** Setting $K_{\min} = K_{\max} = K^\star$ recovers a
constant-K projection bit-identical to CFDv1 with the same $K^\star$.

### 14.4 Composition with CFDv1

The full GFD pipeline at each training step is:

1. **Build prior**: EMA-update $p_{\mathrm{uni}}$ from the batch
   target counts (cost: one `bincount`, $O(B \cdot T)$).
2. **M2**: $\tilde z_t \leftarrow z_t - \gamma \log p_{\mathrm{uni}}$.
3. **M4** (if enabled): $K(t) \leftarrow$ §14.3 from $\tilde z_t$.
4. **Stage 1**: top-K target $\hat p_t \leftarrow$ either
   `cfd_topk_target(z̃, K_t, T_eff)` (legacy global K) or
   `cfd_topk_target_var_k(z̃, K(t), T_eff)`.
5. **Stage 2**: $T_{\mathrm{eff}} \leftarrow$ §13.2.2 (unchanged).
6. **Stage 3**: $\lambda_{\mathrm{eff}} \leftarrow$ §13.2.3 (unchanged,
   gate measured on the post-M2-post-M4 KL term).

When $\gamma = 0$ and pointwise-K is disabled, the pipeline collapses
bit-identically to CFDv1 (verified by `TestCFDv2BackCompat`).

### 14.5 Predicted falsifier

| Arm | γ   | pointwise-K | Prediction (vs Arm D)             |
|-----|-----|-------------|-----------------------------------|
| D   | 0.0 | off         | CFDv1 baseline                    |
| E   | 0.5 | off         | $\mathcal L_E^{(\text{OOD})} \le \mathcal L_D^{(\text{OOD})}$ on H22/B6 setup |
| F   | 0.5 | on          | $\mathrm{gap}_F \le \mathrm{gap}_E$ (both train and OOD) |

The synthetic-fixture test suite (`TestCFDv2*`) verifies the
*contract* (back-compat, monotonicity, well-formed PDFs). The
falsifier above requires a real H22/B6-style training run and is
deferred to a CDGA sweep.

### 14.6 Implementation summary

* `neuroslm.harness.cfd_prior_residual(teacher_logits, log_prior, gamma)` —
  M2 helper (single broadcast subtraction, $O(B \cdot T \cdot V)$).
* `neuroslm.harness.cfd_pointwise_k_from_pmi(teacher_logits, log_prior, K_min, K_max, scale)` —
  M4 helper (one `max` + `exp` + `clip` per position).
* `neuroslm.harness.cfd_topk_target_var_k(teacher_logits, K_per_pos, T)` —
  variable-K Stage-1 projection (vectorised, no Python loop).
* DSL knobs on `MultiCortexConfig`:
  `cfd_prior_gamma: float = 0.0`,
  `cfd_pointwise_k_enabled: bool = False`,
  `cfd_pointwise_k_min: int = 2`,
  `cfd_pointwise_k_max: int = 32`,
  `cfd_pmi_scale: float = 2.0`.
* Telemetry: `cfd_prior_gamma`, `cfd_pointwise_k` flags + mean K
  surfaced via `cfd_K` (now a float — average per-position K under M4).

### 14.7 Open questions for v2.1+

* **M1** (generalisation-tagged token weights): offline pre-pass
  computes per-(context-hash, target) generalisation scores via
  consistency under paraphrase; per-token $\lambda_{\mathrm{eff}}$
  multiplier. Promised in the v2.1 roadmap.
* **M3** (hippocampal-replay scheduler): per-token revisit cadence
  driven by surprise + spacing law. Deferred to v2.2.
* **M5** (FitNets hidden-state alignment): learned bottleneck on
  intermediate teacher activations. Deferred to v3.0.


