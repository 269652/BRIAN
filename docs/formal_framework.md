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

## 7. The Triple Guard (Linter Specification)

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

## 8. Roadmap (Phases bound to this document)

| Phase | Scope | Status | Tests |
|---|---|---|---|
| **THSD-1/4 — Symbolic Expression Units** | `SymbolicSimplex` in engine.py; sheaf integration; discovery surface | ✅ **Shipped** (this commit) | `tests/thsd/test_symbolic_simplex.py` (15) |
| THSD-2/4 — Causal Emergence | `CausalEmergenceOperator`; NIS+ EI search; $y_{d=1}$ conscious variable | ⏳ Spec frozen here §4.1 | TBD `tests/thsd/test_causal_emergence.py` |
| THSD-3/4 — Tonnetz Filter & Triple-Guard Linter | `TonnetzResonanceFilter`; harden `verifier.py` to enforce §7 strictly | ⏳ Spec frozen here §4.2, §7 | TBD `tests/thsd/test_tonnetz_filter.py`, `tests/verification/test_triple_guard.py` |
| THSD-4/4 — Fisher-Rao Retrieval | `FisherRaoRetrieval`; replace cosine in retrieval stalks | ⏳ Spec frozen here §4.3 | TBD `tests/thsd/test_fisher_rao.py` |
| THSD-5 — DSL wiring | Wire `thsd_parser.py` so `complex`/`sheaf` blocks compile end-to-end (resolves the 17 RED tests left by the parallel session) | ⏳ Blocked on compiler.py race resolution | `tests/dsl/test_thsd_parser_complex.py`, `tests/dsl/test_dsl_v2_vesicle_sieve.py` |

---

## 9. How to mutate this document

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
| Fitness composer | `neuroslm.fitness::FitnessComposer` |
| Metabolic pruning | `neuroslm.modules.nrcstk::NRCSTKController` |

---

## Appendix B — Version history

| Version | Date | Change |
|---|---|---|
| v0.1 | _this commit_ | Initial draft; THSD-1/4 (SymbolicSimplex) shipped; §1, §2, §3, §5–§7 normative; §4 spec-only |
