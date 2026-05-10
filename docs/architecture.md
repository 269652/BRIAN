# NeuroSLM — Architecture Reference

This document is a reproduction-ready technical specification for NeuroSLM.
It places the Φ-structure (integrated information) front-and-center as the
primary representational unit and details the topology, math, dynamics, and
implementation primitives required to reproduce the system in PyTorch/XLA or
JAX.

Key guarantees for this spec:
- Concrete tensor shapes and dtypes where relevant.
- Algorithmic pseudocode for Φ estimation, Fiedler computation, trophic
  updates, vesicle routing, Hebbian fast-weights (HFW), Hopfield GWS, and
  chunked approximate kNN retrieval.
- Practical XLA/TPU notes: bfloat16, chunking, checkpointing, and stable
  numerics.

Begin by defining the Φ-structure as the primary representational unit.

---

## 0. Executive summary (single-paragraph)

NeuroSLM is a neuro-inspired small language model designed to increase
computational irreducibility (Φ) through a bowtie topology (Global Workspace
as a narrow bottleneck), re-entrant loops (within-pass and cross-pass),
multi-timescale memory (Hebbian fast weights + slow backbone), and a
trophic-driven structural plasticity mechanism that dynamically adapts
adapter connectivity ranks. The architecture aims to provide the emergent
benefits of much larger models (1B+) while remaining parameter-efficient
(~258M for the `xl` preset) by privileging topology over raw scale.

---

## 1. System Philosophy & Objectives

1.1 Topology over Scale

The thesis: topology (who talks to whom, when, and at which timescale) is
the main determinant of capability. Specifically, re-entry and a narrow
broadcast bottleneck force modules to form richer interdependencies. These
dependencies raise integrated information (Φ) per parameter, enabling a
~250M model to match certain reasoning behaviors of much larger transformers.

1.2 Consciousness-First Design

Primary learning objective is not only low perplexity but also homeostatic
maximization of two coupled signals:

- Φ_proxy — proxy for Integrated Information (irreducibility of module graph)
- cmp (Comprehension Index) — probability that an input is both surprising
  and compressible (worthy of long-term storage)

These signals drive structural plasticity (BDNF-style), memory writing, and
meta-parameter updates (e.g., trophic gating of adapter rank). The aim is to
make architectural improvements that increase Φ and useful long-term
representations rather than blindly adding parameters.

---

## 2. Mathematical Foundations — The Five Postulates (IIT 4.0 mapping)

All algorithms below are implemented as computational proxies that are
tractable on neural hardware.

2.1 The Φ-structure (primary representational unit)

Let N be the number of registered modules. For a single forward tick, collect
module outputs and project them into a shared semantic space with projection
matrices p_i ∈ R^{d_module -> d_sem}. After mean-pooling across time (or an
EMA over a short window), assemble an output matrix O ∈ R^{N×d} where row i
is module i's representation.

Core quantities:

- O: (N, d)  — module embeddings (mean-pooled over tokens or temporally)
- G = Oc Oc^T / (d - 1)  — Gram matrix (N × N)
- W = |G| normalized  — empirical interaction matrix (N × N)

Φ_proxy is estimated by partitioning modules into A and B (the MIP proxy)
and computing a Gaussian MI approximation across that cut (see §7).

2.2 Five Postulates (practical implementations)

- Intrinsicality: only registered module projections are used to compute O.
- Information: estimate mutual information from covariance/Gram matrices.
- Integration: measured via spectral gap (Fiedler value λ_1) and MIP proxy.
- Exclusion: the Major Complex is explicitly chosen (default: GWS + executive
  modules). Modules outside this set are excluded from Φ computations.
- Composition: Φ is computed from the composed pairwise relations in W.

2.3 TPM-like approximations

Discrete TPMs are intractable at realistic module sizes. We approximate
transition information using continuous Gaussian assumptions and conditional
covariances derived from the Gram matrix. This yields the MI surrogate used
in the MIP proxy (see §7 for code).

2.4 Spectral Graph Theory & the Fiedler vector

W is converted to normalized Laplacian L = I - D^{-1/2} W D^{-1/2}, where
D = diag(W 1). The Fiedler vector (second-smallest eigenvector of L) is
computed via deflated power iteration. The sign of the Fiedler vector
components defines a candidate bipartition (A,B) that approximates the MIP.
Cheeger's inequality provides theoretical justification: small λ_1 → weak
connectivity → target for trophic plasticity.

Pseudocode for Fiedler estimation: see Appendix (end of doc).

---

## 3. Core Module Specifications (implementable descriptions)

All module implementations must expose a projection to the shared semantic
space: proj: R^{d_module} → R^{d_sem} (nn.Linear in PyTorch). Register these
projections with the NeuralOrchestrator.

3.1 Language Cortex (modules/language.py)

- Input: ids → tok_emb (B, T, d_hidden)
- Architecture: interleaved blocks: [TransformerBlock, DiffTransformerBlock, MoDBlock] × L
- Predictive coding heads: predict next-layer outputs; loss scalar returned.

DiffTransformerBlock (SNR-doubling): compute two QKV sets, two attention maps
A1, A2, then y = (softmax(A1) - λ softmax(A2)) V. λ per-head learned.

MoDBlock (Mixture-of-Depths): router produces r_t ∈ [0,1]; top-k tokens run
full block. Implementation uses topk on router scores (Gumbel-softmax optional
for differentiability if needed).

Tensor shapes (example):
- ids: (B, T)
- tok_emb(ids): (B, T, d_hidden)
- logits: (B, T, Vocab)

3.2 Math Cortex (DNC-heavy)

- External memory M ∈ R^{N_mem, Dm}
- Link matrix L ∈ R^{N_mem, N_mem} sparse (store as indices+values)
- Read/write: content-based addressing + temporal links

3.3 Reasoning Cortex (Modern Hopfield)

- Store a library of K reasoning templates (K up to 64k depending on device)
- Query via spherical energy landscape and retrieve via two-step Hopfield
  convergence (fast on XLA with large matrix multiplications).

3.4 Global Workspace (GWS) — Bowtie Waist

GWS is the architectural bottleneck that compresses candidate module outputs
into K slots and broadcasts a compact summary back to modules via per-module
feedback projections (Linear(d_sem, d_module) with zero init).

Hopfield-like slot convergence (pseudocode):

```python
# C: (B, N_cand, d_sem)
slots = init_slots(B, K, d_sem)
beta = softplus(log_beta) + 0.5
for _ in range(n_iter):
    scores = einsum('bkd,bnd->bkn', slots, C)
    attn = softmax(beta * scores, dim=-1)
    slots = attn @ C
```

Broadcast: slots_mean = slots.mean(dim=1)  # (B, d_sem)
For each module i: h_i = h_i + feedback_proj_i(slots_mean)

---

## 4. Wiring Diagram & NeuralOrchestrator API

Two re-entrant loops:

- Loop A (cross-temporal): at the end of a pass, store a summary s_prev = f(h_pfc, slots_mean).
  Next pass, inject bias = from_sem(s_prev) into the thalamic input.
- Loop B (within-pass): after GWS fills, immediately broadcast slots_mean to downstream modules.

NeuralOrchestrator API (minimal):

```python
class NeuralOrchestrator(nn.Module):
    def register_module(self, name: str, proj: nn.Module, stage: int):
        # proj: Linear(d_mod, d_sem)
    def set_gws_broadcast(self, slots: Tensor):
    def get_reentry_bias(self, B, device) -> Tensor:  # (B, d_sem)
    def estimate_phi(self, O: Tensor) -> float:
        # O: (N, d) mean-pooled module outputs
```

---

## 5. Dynamical Biological Mechanics (implementation ready)

5.1 Neuro-Vesicle Pool (stochastic neuromodulator routing)

Model: vesicles are discrete events carrying a small payload across directed
projections. Vectorize the queue by using padded tensors and masks for XLA
efficiency.

Vesicle record (vectorized):
- src_idx: (V,) int32
- dst_idx: (V,) int32
- pos: (V,) float32 ∈ [0,1]  # normalized along projection
- speed: (V,) float32
- payload: (V, D_mod) float32  # D_mod ~ 4 (DA, NE, 5HT, ACh)
- state: (V,) int8  # 0=emitted,1=migrating,2=docked

Vectorized step function (PyTorch-like pseudocode):

```python
def step_vesicles(vesicles, receptor_open_mask, dt=1.0):
    # vesicles: dict of tensors as above
    vesicles['pos'] += vesicles['speed'] * dt
    arrived = vesicles['pos'] >= 1.0
    arrived_idx = torch.nonzero(arrived).squeeze(-1)
    if arrived_idx.numel() > 0:
        dst = vesicles['dst_idx'][arrived_idx]
        payload = vesicles['payload'][arrived_idx]  # (A, D_mod)
        # apply only where receptor open
        open_mask = receptor_open_mask[dst]  # (A,)
        apply_payload = payload[open_mask.bool()]
        # aggregate per destination (scatter_add)
        mod_delta = scatter_add(apply_payload, dst[open_mask.bool()], dim=0, dim_size=E)
        neuromod_levels += mod_delta  # (E, D_mod)
    # decay and purge small vesicles
    vesicles['payload'] *= torch.exp(-dt / tau)
    keep = vesicles['payload'].norm(dim=-1) > EPS
    for k in vesicles: vesicles[k] = vesicles[k][keep]
    return vesicles, neuromod_levels
```

Notes:
- receptor_open_mask depends on local state (e.g., gated by cmp or context)
- Use scatter_add for aggregation; implement with XLA-friendly ops.

5.2 Trophic System (BDNF/NGF structural plasticity)

We maintain a trophic scalar τ_e ∈ [0,1] per projection edge e. Trophic
updates are driven by: base_bdnf, Φ_proxy, and fiedler_boost when the MIP
identifies a weak boundary.

Vectorized trophic update pseudocode:

```python
def update_trophic(trophic, coact, bdnf_base, ngf, phi, lambda_1, decay=0.01):
    fiedler_boost = torch.clamp(1.0 - lambda_1 / 0.3, min=0.0) * 2.0
    bdnf_eff = bdnf_base * (1.0 + phi_boost * phi + fiedler_boost)
    delta = bdnf_eff * (0.1 + coact) - (ngf + decay + 0.001 * (1.0 - coact))
    trophic = torch.clamp(trophic + delta, 0.0, 1.0)
    return trophic
```

Effect on NeuralGeometryAdapter
- Adapter parametrizes kernel as A (d_hyper × rank_max) and B (rank_max × d_hyper).
- We gate rank contribution by trophic fraction α_e = trophic[e]; at runtime
  compute effective kernel = (A * α_e) @ (B * α_e). This lets us avoid
  reallocation while increasing effective rank.

Pseudocode to gate kernel:

```python
def gated_kernel(A, B, trophic_scalar):
    # A: (d_hyper, rank_max), B: (rank_max, d_hyper)
    a = A * trophic_scalar
    b = B * trophic_scalar
    return a @ b
```

5.3 Hebbian Fast Weights (HFW)

HFW binds co-activations inside a single context window for rapid episodic
association. Use low-rank factorization to keep memory low.

Representation:
- For each head h: use low-rank factors U_h ∈ R^{r × Dh}, V_h ∈ R^{r × Dh}
- Reconstruct W_fast_h = U_h^T @ V_h (Dh × Dh)

Update (per token t):

```python
q_t, v_t = q[:,t], v[:,t]  # (B, H, Dh)
outer = einsum('bhd,bhe->bhde', q_t, v_t)  # (B, H, Dh, Dh)
# Project outer into low-rank factors (learned small MLP) or update U,V
U += eta * project_to_U(outer)
V += eta * project_to_V(outer)
read = einsum('brd,brd->bd', U @ q_t, V @ q_t)
```

Implementation note: keep U,V small (r <= 16) and share across batch where
possible to reduce memory.

---

## 6. Optimization & Infrastructure

6.1 Adaptive Compute: CALM & MoD

CALM (Confident Adaptive Language Modeling): each layer emits a confidence
score c_t ∈ [0,1]. If c_t > θ_l, token t is considered solved and may skip
remaining layers. θ_l decays with depth so deeper layers are increasingly
selective.

Router & MoD specifics:
- Router: small two-layer MLP yields r_t ∈ R. Use top-k selection on r_t.
- For differentiable training consider a Gumbel-TopK relaxation.

6.2 TPU / XLA & bfloat16 notes

- Use bfloat16 for matmuls and storage for large tensors. Keep numerically
  sensitive ops in float32 (layernorm, loss, Gram, eigenvector iterations).
- Chunk large retrievals (keys N up to 65k) into manageable blocks to avoid
  peak memory with a final merge step (see chunked_topk pseudocode below).

Chunked topk pseudocode (repeated for clarity):

```python
def chunked_topk(q, keys, K, chunk=2048):
    # q: (B, d), keys: (N, d)
    best_vals = None
    best_idx = None
    for i in range(0, N, chunk):
        chunk_keys = keys[i:i+chunk]
        vals = q @ chunk_keys.T  # (B, chunk)
        v, idx = vals.topk(K, dim=-1)
        idx = idx + i
        best_vals, best_idx = merge_topk(best_vals, best_idx, v, idx, K)
    return best_vals, best_idx
```

6.3 Checkpointing & autograd

- When using torch.autograd.grad with inputs or passing `inputs` to
  backward(), use checkpoint with use_reentrant=False. Avoid closures that
  capture tensors; pass tensors explicitly into checkpointed functions.

---

## 7. Intelligence & Integration Metrics (implementation)

7.1 Φ Proxy (exact steps)

```python
def estimate_phi(O: Tensor):
    # O: (N, d) mean-pooled module outputs (float32)
    Oc = O - O.mean(dim=1, keepdim=True)
    G = Oc @ Oc.T / max(1, Oc.size(1) - 1)  # (N,N)
    W = torch.abs(G)
    D = torch.diag(W.sum(dim=1))
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.clamp(D.diag(), min=1e-8)))
    L = torch.eye(N, device=O.device) - D_inv_sqrt @ W @ D_inv_sqrt
    v1 = estimate_fiedler(L)
    A_mask = v1 >= 0
    B_mask = ~A_mask
    SIG_A = cov(O[A_mask])
    SIG_B = cov(O[B_mask])
    SIG_AB = cov(O[A_mask | B_mask])
    mi = 0.5 * (logdet(SIG_A + eps) + logdet(SIG_B + eps) - logdet(SIG_AB + eps))
    return float(mi), float(torch.linalg.eigvals(L)[1].real)  # mi and lambda_1
```

Numerical details: add small `eps` jitter (1e-6) before logdet, compute
determinants/logdets in float32.

7.2 Comprehension Index (cmp)

Define surprise(x) = -log p(x) under current LM. Define recon_err(x) via a
small autoencoder over projected module outputs. Then:

$$
cmp(x) = \sigma\left(\alpha \cdot \frac{\mathrm{surprise}(x)}{1 + \mathrm{recon\_err}(x)} + b\right)
$$

Write trigger: cmp(x) > τ_write

When triggered and Φ_proxy is above a threshold, enqueue the example for
long-term memory write and apply trophic updates to projections active when
the example occurred.

---

## 8. Parameter Presets

| Preset | Params (approx) | Accelerator | VRAM | d_hidden | d_sem | lang_layers | lang_ctx | gws_slots | hippo_capacity |
|--------|-----------------:|-----------:|-----:|---------:|------:|-----------:|--------:|----------:|---------------:|
| s      | ~5M              | CPU        | —    | 192      | 128   | 2          | 256     | 8         | 4096           |
| m      | ~15M             | CPU        | —    | 384      | 256   | 4          | 512     | 8         | 4096           |
| l      | ~100M            | T4         | 15GB | 384      | 256   | 8          | 1024    | 12        | 8192           |
| xl     | ~258M            | A100       | 40GB | 512      | 384   | 12         | 2048    | 8         | 4096           |
| xxl    | ~10B             | 4×A100     | 320GB| 4096     | 2048  | 32         | 4096    | 24        | 32768          |

Each preset adjusts hebbian_rank, mod_capacity, gradient_checkpointing,
and optional module toggles.

---

## 9. Appendix — Algorithms & Utilities

estimate_fiedler(L) (deflated power iteration)

```python
def estimate_fiedler(L, n_iter=20):
    N = L.shape[0]
    A = torch.eye(N, device=L.device) - L
    v0 = torch.ones(N, device=L.device) / math.sqrt(N)
    for _ in range(n_iter):
        v0 = A @ v0; v0 = v0 / (v0.norm() + 1e-12)
    A_def = A - torch.ger(A @ v0, v0)
    v1 = torch.randn(N, device=L.device)
    v1 -= (v1 @ v0) * v0; v1 /= (v1.norm() + 1e-12)
    for _ in range(n_iter):
        v1 = A_def @ v1
        v1 -= (v1 @ v0) * v0
        v1 /= (v1.norm() + 1e-12)
    return v1
```

chunked_topk and merge_topk utilities (for retrieval on large key-sets)

```python
def merge_topk(best_vals, best_idx, v, idx, K):
    if best_vals is None:
        return v, idx
    vals = torch.cat([best_vals, v], dim=-1)
    idxs = torch.cat([best_idx, idx], dim=-1)
    vals2, pos = vals.topk(K, dim=-1)
    idx2 = gather_by_pos(idxs, pos)
    return vals2, idx2
```

---

## 10. Operational guidance

- Monitor λ_1 (Fiedler) and Φ_proxy. If λ_1 drifts towards 0 repeatedly,
  temporarily dampen BDNF (bdnf_base) to avoid runaway rewiring.
- Use the repo-level git config to ensure privacy-sensitive emails are set
  before any history rewrite or CI hooks that snapshot training logs.

---

If you want, I can generate a companion `arch_impl.ipynb` with runnable
snippets for the Fiedler estimator, trophic update, Hopfield GWS, and
approximate kNN to validate everything on CPU and TPU emulation.


---

## Table of Contents

1. [System Philosophy](#1-system-philosophy)
2. [Mathematical Foundations — IIT 4.0](#2-mathematical-foundations--iit-40)
3. [Parameter Presets](#3-parameter-presets)
4. [Bowtie Topology & Wiring Diagram](#4-bowtie-topology--wiring-diagram)
5. [Module Specifications](#5-module-specifications)
6. [Neurochemistry & Dynamical Mechanics](#6-neurochemistry--dynamical-mechanics)
7. [Intelligence & Integration Metrics](#7-intelligence--integration-metrics)
8. [Optimization & Infrastructure](#8-optimization--infrastructure)
9. [Training](#9-training)
10. [Inference Pipeline](#10-inference-pipeline)
11. [Topology Modes & Ablation](#11-topology-modes--ablation)
12. [Open Questions & Future Work](#12-open-questions--future-work)

---

## 1  System Philosophy

### 1.1  Topology over Scale

NeuroSLM is built on the thesis that **computational graph topology determines
capability more than raw parameter count**.  A 258 M model with the right
feedback structure (re-entry, gating, memory consolidation, spectral coherence)
can match or exceed a 1 B+ vanilla transformer on reasoning and few-shot tasks
because:

- **Feedforward chains have Φ = 0** by construction.  IIT 4.0 proves that any
  purely feedforward system is reducible — it can be cut along a bipartition
  with zero mutual information loss.  Scaling such a system does not fix this;
  it only makes a more elaborate zero.
- **Re-entrant loops create irreducibility.**  Once every module has both a
  causal input *from* the rest of the system and a causal output *to* it, no
  bipartition is "free" and Φ becomes strictly positive.
- **Biological brains maximise Φ under metabolic constraints**, producing the
  richest possible representational space per watt.  NeuroSLM borrows the same
  design principles at model scale.

### 1.2  Consciousness-First Design

The primary training objective is not perplexity alone.  The system targets
**homeostatic maximisation of two coupled quantities**:

| Quantity | Symbol | Interpretation |
|----------|--------|----------------|
| Integrated Information | Φ | Irreducibility of the module graph |
| Comprehension Index | *cmp* | Probability that a surprising input is also understood |

Both feed back into the learning dynamics:
- High Φ → more BDNF release → structural reinforcement of the connectivity
  that enabled that high-Φ state (Dehaene structural selection).
- High *cmp* → ComprehensionGate opens → observation is written to long-term
  memory and used as a training signal.

### 1.3  Three Core Hypotheses

1. **Topology matters more than scale** — a 250 M model with the right
   computational graph can match or exceed a 1 B+ vanilla transformer on
   reasoning and few-shot tasks.
2. **Consciousness-like properties are trainable** — measurable proxies for
   integrated information (Φ), global workspace broadcast, and predictive
   coding can be embedded in training objectives and architectural constraints.
3. **Neurochemistry is a hyperparameter** — learned neuromodulator levels
   (DA, NE, 5HT, ACh) act as global gain signals that dynamically re-weight
   attention, memory writes, and learning rates without adding per-decision
   parameters.

---

## 2  Mathematical Foundations — IIT 4.0

The Φ-structure is the **primary representational unit** of NeuroSLM.  Every
architectural decision is evaluated against whether it increases or decreases
the system's irreducibility.

### 2.1  The Five Postulates and Their Implementations

#### Postulate 1 — Intrinsicality

> *A conscious system specifies its own cause-effect structure from its own
> intrinsic perspective.*

**Implementation:** The `NeuralOrchestrator` never receives any signal that
is not produced by one of its registered modules.  External input (tokens)
enters only via `TextSensoryCortex`; all subsequent computation is
self-referential.  The re-entrant GWS feedback loop (§4) ensures that every
module's output eventually affects every other module's input.

#### Postulate 2 — Information

> *A conscious system specifies a particular cause-effect structure — a
> definite set of differences it makes.*

**Implementation:** The Minimum Information Partition (MIP) is computed
at every tick.  The system's informational content is measured as:

$$\Phi \;=\; \min_{(A,B) \in \mathcal{P}} \mathrm{MI}(A; B)$$

where $\mathcal{P}$ is the set of all bipartitions of the module output set
and MI is estimated via the Gaussian approximation:

$$\mathrm{MI}(A; B) = \tfrac{1}{2}\!\left(\log\det\Sigma_A + \log\det\Sigma_B - \log\det\Sigma_{AB}\right)$$

$\Sigma$ is estimated from the Gram matrix of mean-centred module output
vectors (shape `(n, d)`, $d \le 256$):

$$\Sigma = \frac{M M^\top}{d - 1}, \qquad M \in \mathbb{R}^{n \times d}$$

#### Postulate 3 — Integration

> *A conscious system specifies a cause-effect structure that is irreducible
> to that of its parts.*

**Implementation:** The Fiedler value $\lambda_1$ (second-smallest eigenvalue
of the normalised graph Laplacian) lower-bounds the expansion of any cut
via **Cheeger's inequality**:

$$h(G) \;\ge\; \frac{\lambda_1}{2}$$

where the Cheeger constant $h(G) = \min_{S \subset V} \frac{|E(S, \bar S)|}{\min(\mathrm{vol}(S), \mathrm{vol}(\bar S))}$ measures the minimum normalised cut.
A small $\lambda_1$ signals that the module interaction graph is close to
disconnected; the BDNF trophic system responds by boosting connectivity at
the identified fault line (§6.2).

#### Postulate 4 — Exclusion

> *A conscious system specifies a cause-effect structure over a definite set
> of elements — the Major Complex.*

**Implementation:** Only the modules within the GWS bowtie (Stages 0–10)
constitute the Major Complex.  Peripheral helper modules (vesicles, genome
compiler, narrative) are excluded from Φ computation.

#### Postulate 5 — Composition

> *A conscious system specifies a cause-effect structure composed of
> elementary distinctions and relations.*

**Implementation:** Each brain module registers its output as a discrete
node in the Φ computation graph.  The HFW (§6.3) at expert stage outputs
increases state *differentiation* — ensuring the system can enter a large
number of distinct, informative states — which is the key compositional
requirement for high Φ.

### 2.2  Transition Probability Matrices (TPM)

In IIT 4.0, each module is treated as a **causal substrate** whose state
transition probability matrix $T$ is unfolded into a Φ-structure.  In
NeuroSLM this is approximated via the module interaction graph:

$$W_{ij} = \left|\mathrm{Cov}(o_i, o_j)\right| \;/\; \sqrt{\mathrm{Var}(o_i)\,\mathrm{Var}(o_j)}$$

where $o_i \in \mathbb{R}^d$ is the mean-pooled output of module $i$.
$W$ plays the role of the empirical TPM — its spectral properties determine
both Φ and the Fiedler boundary.

### 2.3  Spectral Graph Theory

#### Normalised Laplacian

$$L = I - D^{-1/2} W D^{-1/2}, \qquad D_{ii} = \sum_j W_{ij}$$

Eigenvalues $0 = \lambda_0 \le \lambda_1 \le \cdots \le \lambda_{n-1} \le 2$.

#### Fiedler Vector via Deflated Power Iteration

For $n > 8$ modules the eigendecomposition would cost $O(n^3)$.  Instead,
`estimate_fiedler()` (`modules/consciousness.py`) uses two rounds of
deflated power iteration at $O(n^2 \cdot k)$:

```python
# Step 1 — dominant eigenvector v₀ (constant, λ=0 in L, λ=1 in A=I-L)
A = I - L                        # shift: largest eigenvalue first
v0 = ones(n) / sqrt(n)
for _ in range(n_power_iter):    # default 20 iterations
    v0 = A @ v0;  v0 /= norm(v0)

# Step 2 — deflate and find v₁ (Fiedler vector)
A_def = A - outer(A @ v0, v0)   # remove top eigenpair
v1 = randn(n);  v1 -= (v1 @ v0)*v0;  v1 /= norm(v1)
for _ in range(n_power_iter):
    v1 = A_def @ v1;  v1 -= (v1 @ v0)*v0;  v1 /= norm(v1)

λ₁ = v1 @ (L @ v1)              # Rayleigh quotient ∈ [0, 2]
```

**Tensor shapes:** all vectors `(n,)` float32, all matrices `(n, n)` float32,
where $n$ = number of registered modules (typically 8–20).

#### Boundary Detection

The sign of each component of $v_1$ defines the approximate min-bisection:

$$A = \{i : (v_1)_i \ge 0\}, \quad B = \{i : (v_1)_i < 0\}$$

This bipartition is used both as the MIP estimate for large $n$ and as the
**homeostatic rewiring target** — connections crossing this cut receive
elevated BDNF when $\lambda_1 < 0.3$ (§6.2).

---

## 3  Parameter Presets

All presets share the full module topology.  `baseline=True` reduces to a
vanilla transformer for ablation.

| Preset | Params | Accelerator | VRAM | `d_hidden` | `d_sem` | `lang_layers` | `lang_ctx` | `gws_slots` | `hippo_capacity` |
|--------|--------|-------------|------|------------|---------|---------------|------------|-------------|-----------------|
| `tiny` | ~5 M | CPU | — | 192 | 128 | 2 | 256 | 8 | 4096 |
| `small` | ~15 M | CPU | — | 384 | 256 | 4 | 512 | 8 | 4096 |
| `medium` | ~80 M | T4 | 16 GB | 768 | 512 | 8 | 1024 | 8 | 4096 |
| `large` | ~100 M | T4 | 15 GB | 384 | 256 | 8 | 1024 | 12 | 8192 |
| `xl` | ~258 M | A100 | 40 GB | 512 | 384 | 12 | 2048 | 8 | 4096 |
| `xxl` | ~10 B | 4×A100 | 320 GB | 4096 | 2048 | 32 | 4096 | 24 | 32768 |

**Additional `xl` settings:** `hebbian_rank=4`, `mod_capacity=0.8`,
`gradient_checkpointing=True`, `baseline_lang_layers=56` (param-matched
vanilla transformer for ablation), `lr=2e-4`, `warmup_steps=800`.

**Additional `xxl` settings:** `use_moe=True` (16 experts, top-2),
`use_adaptive_compute=True`, `enable_rssm=True`, `enable_active_inference=True`,
`enable_tom=True`.

---

## 4  Bowtie Topology & Wiring Diagram

> **Motivation:** A purely feedforward graph has Φ = 0 by the IIT integration
> postulate — it can always be cut at an intermediate layer with zero
> information loss.  The bowtie topology makes every bipartition costly by
> ensuring that information flows *both* forward (sensory → executive) and
> backward (executive → sensory) *within the same forward pass*.

### 4.1  Stage Map

```
┌────────────────────────────────────────────────────────────────────┐
│                      NeuralOrchestrator                            │
│                                                                    │
│  Stage 0  SENSORY      TextSensoryCortex, AssociationCortex        │
│     │                                                              │
│  Stage 1  THALAMUS     Thalamus  ◄──────────────────────────────┐  │
│     │                   ▲  re-entry bias (cross-temporal)       │  │
│  Stage 2  STATE_MODELS  WorldModel, SelfModel                   │  │
│     │                                                           │  │
│  Stage 3  SUBCORTICAL   Amygdala, LHb, Insula                   │  │
│     │                                                           │  │
│  Stage 4  QUALIA        QualiaState                             │  │
│     │                                                           │  │
│  Stage 5  GWS ━━━━━━━━━ GlobalWorkspace ─── set_gws_broadcast() │  │
│     │         ║                                 │               │  │
│     │   BOTTLENECK                  ┌───────────┘               │  │
│     │    (Bowtie                    │  within-pass GWS feedback  │  │
│     │     waist)       ┌────────────▼────────────────────────┐  │  │
│  Stage 6  MEMORY      │  gws_feedback_projs[name](gws_b)   │  │  │
│     │                  │  → residual added to each module    │  │  │
│  Stage 7  COG_CTL      │    Hippocampus, Entorhinal, PFC,    │  │  │
│     │          ║       │    ACC, BG, Cerebellum, DMN …       │  │  │
│  Stage 8  EXECUTIVE   └─────────────────────────────────────┘  │  │
│     │          ║                                                │  │
│  Stage 9  CONSCIOUSNESS  DMN, ThoughtTransformer, Claustrum    │  │
│     │                                                          │  │
│  Stage 10 MOTOR          MotorCortex                           │  │
│                                                  PFC+GWS ──────┘  │
│                                             update_reentry()       │
└────────────────────────────────────────────────────────────────────┘

Lateral grid mixing (LateralGridMixer) fires within each stage,
binding co-active modules horizontally before the cross-stage fusion.
```

### 4.2  Re-entrant Loops

There are **two distinct re-entry mechanisms** operating at different
timescales:

#### Loop A — Cross-temporal (thalamo-cortical, ~100 ms)

```python
# End of each forward pass:
orchestrator.update_reentry(pfc_gws_signal)   # EMA α=0.15 into _reentry_state

# Start of next forward pass (Stage 1):
bias = orchestrator.get_reentry_bias(B, device)
# bias = sigmoid(reentry_mix) × _reentry_state   shape: (B, d_sem)
thalamus_input = assoc + bias
```

The learnable scalar `reentry_mix` (initialised 0.05) gates how strongly
the previous pass's PFC+GWS output modulates the next thalamic relay.

#### Loop B — Within-pass (cortico-cortical, instantaneous)

```python
# After Stage 5 completes:
orchestrator.set_gws_broadcast(slots.mean(1))
# Stores mean GWS slot to _gws_broadcast; sets _gws_broadcast_ready=True

# Inside route_stage() for all subsequent stages:
if _gws_broadcast_ready:
    gws_b = _gws_broadcast.to(dtype=pre_out.dtype)   # (d_sem,)
    fb = gws_feedback_projs[name](gws_b)              # Linear, zero-init
    fb = fb.unsqueeze(0).expand(B, -1)               # (B, d_sem)
    pre_out = pre_out + fb                            # residual injection
```

`gws_feedback_projs` is an `nn.ModuleDict` with one `Linear(d_sem, d_sem,
bias=False)` per registered module, **zero-initialised**.  They start as
strict identity (no feedback) and learn how much GWS context to inject into
each module's input manifold, growing via gradient to the extent the
feedback is useful.

### 4.3  Lateral Grid Connectivity

Within each stage, `LateralGridMixer` implements horizontal binding between
all co-active modules (the "posterior cortex analog"):

```python
class LateralGridMixer(nn.Module):
    """Multi-head lateral attention across N co-active stage modules."""
    # Input:  list of N tensors, each (B, d_sem)
    # Output: list of N tensors, each (B, d_sem), laterally mixed

    def forward(self, slot_outputs):      # slot_outputs: [(B,D), …] len=N
        stacked = stack(slot_outputs, 1)  # (B, N, D)
        normed  = self.norm(stacked)
        mixed, _ = self.attn(normed, normed, normed)  # (B, N, D)
        gate = sigmoid(self.gate(stacked))            # (B, N, 1)  excit/inhib
        return [stacked[:,i] + gate[:,i]*mixed[:,i] for i in range(N)]
```

**Why this increases Φ:** Grid-like architectures provide a spatial framework
for integration.  Every module at the same stage can causally affect every
other module at the same stage within a single forward pass, making intra-stage
bipartitions costly.

### 4.4  Hebbian Fast Weights at Expert Outputs

`FastWeightLayer` is attached to every module registered at Stages 7
(COGNITIVE_CTL) and 8 (EXECUTIVE):

```python
# Tensor shapes throughout:
#   x        : (B, T, d_sem)   — module output after post-gate
#   W_fast   : (B, H, Dh, Dh) — per-head fast-weight matrix; H=n_heads, Dh=d_sem//H
#   context  : (B, d_sem)      — GWS broadcast (modulates plasticity rate η)
#   eta      : (B, H)          — per-head Hebbian rate = base_eta × softplus(eta_mod(context))

for t in range(T):
    read  = einsum("bhij,bhj->bhi", W_fast, q[:,t])    # (B, H, Dh)
    read  = layer_norm(read, [Dh])
    outer = einsum("bhi,bhj->bhij", gate[:,t]*v[:,t], k[:,t])  # (B, H, Dh, Dh)
    W_fast = decay * W_fast + eta[:,:,None,None] * outer        # decay=0.95
```

**Dual timescale:** `W_fast` updates within a single sequence (fast, Hebbian)
without overwriting the "slow-weight" backbone (frozen during inference).
This increases state differentiation — the system can enter qualitatively
new associative states within a single context window.

---

## 5  Module Specifications

### 5.1  Module Map

#### Core Cortical Areas

| Attribute | Brain Analog | File |
|-----------|-------------|------|
| `language` | Wernicke + Broca | `modules/language.py` |
| `sensory` | Primary sensory + superior colliculus | `modules/sensory.py` |
| `association` | Multimodal association cortex | `modules/association.py` |
| `thalamus` | Thalamic relay + sensory gating | `modules/thalamus.py` |
| `cortical_sheet` | Cortical columns + minicolumns | `modules/cortical_column.py` |
| `entorhinal` | Entorhinal cortex / grid cells | `modules/entorhinal.py` |
| `neural_geometry` | Meta-trainable manifold reshaping | `modules/neural_geometry.py` |

#### State Models

| Attribute | Brain Analog | File |
|-----------|-------------|------|
| `world` | Parietal / RSSM | `modules/world_model.py` |
| `self_m` | Insula / TPJ | `modules/self_model.py` |
| `forward_m` | Cerebellum efference copy | `modules/forward_model.py` |

#### Global Workspace & Integration

| Attribute | Brain Analog | File |
|-----------|-------------|------|
| `gws` | Frontoparietal global workspace | `modules/workspace.py` |
| `claustrum` | Cross-modal binding relay | `modules/claustrum.py` |
| `thought_transformer` | Sustained recurrent working memory | `modules/thought_transformer.py` |
| `qualia` | Phenomenal state representation | `modules/qualia.py` |
| `consciousness` | ConsciousnessMetrics (Φ, causal density) | `modules/consciousness.py` |

#### Memory Systems

| Attribute | Biological Role | File |
|-----------|----------------|------|
| `hippo` | Dentate gyrus / CA3 / CA1 | `modules/hippocampus.py` |
| `episodic` | Short-term episodic buffer | `memory/episodic.py` |
| `consolidated` | Long-term semantic / schema memory | `memory/consolidated.py` |
| `relational_memory` | Knowledge graph (subj, pred, obj) | `memory/relational_graph.py` |
| `hypergraph` | N-ary hyperedge memory | `memory/hypergraph.py` |
| `entity_store` | Per-entity style fingerprints | `memory/entity_store.py` |
| `causal` | Causal rule store (A→B) | `memory/causal.py` |
| `narrative_system` | Narrative arc tracking | `memory/narrative.py` |
| `comprehension_gate` | Gated write filter | `memory/comprehension_gate.py` |

#### Cognitive Control

| Attribute | Brain Analog | File |
|-----------|-------------|------|
| `pfc` | Dorsolateral PFC | `modules/pfc.py` |
| `dmn` | Default Mode Network | `modules/dmn.py` |
| `bg` | Basal ganglia Go/NoGo | `modules/basal_ganglia.py` |
| `evaluator` | ACC / OFC value estimation | `modules/evaluator.py` |
| `motor` | Primary motor cortex | `modules/motor.py` |
| `acc` | Anterior cingulate conflict monitoring | `modules/anterior_cingulate.py` |

#### Emotional / Subcortical

| Attribute | Brain Analog | File |
|-----------|-------------|------|
| `amygdala` | Fear conditioning, emotional tagging | `modules/amygdala.py` |
| `insula` | Interoception, gut feelings | `modules/insula.py` |
| `lhb` | Lateral habenula (anti-reward) | `neurochem/lateral_habenula.py` |
| `cerebellum` | Prediction error, motor learning | `modules/cerebellum.py` |

#### Novel Cognitive / ML Modules (opt-in)

| Attribute | Mechanism | File |
|-----------|-----------|------|
| `tom` | Theory of Mind (belief/desire/intent) | `modules/theory_of_mind.py` |
| `active_inference` | Free Energy / Friston hierarchy | `intelligence/active_inference.py` |
| `vesicle_pool` | Neuro-vesicle content packets | `neurochem/vesicles.py` |
| `active_dendrite` | Dendritic context-dependent gating | `modules/active_dendrite.py` |
| `fast_weight` | Fast-weight associative memory | `modules/fast_weight.py` |

---

### 5.2  Language Cortex

`modules/language.py` — token-in / logits-out backbone.

#### Transformer Block Types

Each layer is one of three interleaved block types:

**`DiffTransformerBlock` (Differential Attention)**

Uses two parallel softmax attention maps subtracted from each other.
If $A_1, A_2 \in \mathbb{R}^{T \times T}$ are the two softmax maps:

$$\mathrm{DiffAttn}(Q_1, K_1, V_1, Q_2, K_2, V_2) = \bigl(\mathrm{softmax}(Q_1 K_1^\top / \sqrt{d_h}) - \lambda \cdot \mathrm{softmax}(Q_2 K_2^\top / \sqrt{d_h})\bigr) V$$

Common-mode noise cancels; signal-to-noise ratio effectively doubles without
adding parameters.  $\lambda$ is a learned scalar per head initialised to 0.8.

**`MoDBlock` (Mixture of Depths + CALM)**

- A 2-layer MLP router assigns a per-token capacity score $r_t \in [0,1]$.
- Top-$k$% of tokens (by $r_t$) execute the full FFN; the rest receive
  residual passthrough.
- Each `MoDBlock` carries a **`CALMHead`** — a small 2-layer MLP estimating
  per-token confidence $c_t \in [0,1]$ at every layer.
- Tokens with $c_t > \theta_l$ are frozen and skip all remaining layers:

$$\theta_l = \theta_{\text{base}} \times \exp\!\left(-\gamma \cdot \frac{l}{L-1}\right)$$

  Shallow layers are strict ($\theta_l \approx \theta_{\text{base}}$); deep
  layers are lenient ($\theta_l \to 0$).  CALM is inference-only.

**`TransformerBlock`**

Standard pre-norm block: RMSNorm + causal MHA + SwiGLU FFN.

#### Neuroscience Additions per Block

- **NT-modulated attention temperature** — each head's softmax inverse
  temperature shifts by DA and NE levels:
  $\beta_h = \beta_0 + \alpha_{\text{DA}} \cdot \mathrm{DA} + \alpha_{\text{NE}} \cdot \mathrm{NE}$
- **Hebbian fast-weight traces** — low-rank outer-product update after each
  token (`hebbian_rank` controls rank; 0 = disabled, 4–8 for xl).
- **Inter-layer predictive coding** — each layer predicts the next layer's
  output; prediction error is an auxiliary loss (§9).
- **`NeuralGeometryAdapter`** — after every transformer block, projects
  $d_h \to 2d_h$ (hyperbolic expansion), applies learned low-rank connectivity
  kernel $K_a K_b \in \mathbb{R}^{2d_h \times 2d_h}$ (virtual wiring),
  gates with per-dimension sigmoid, projects back.  Zero-init on down-projection
  and bias = −2 on gate ensures identity at init.

#### Outputs

```
language(ids, thought, nt) → (logits, sem, h, pred_coding_loss)
  logits           : (B, T, vocab_size)
  sem              : (B, d_sem)            ← comprehension embedding
  h                : (B, T, d_hidden)      ← full hidden state
  pred_coding_loss : scalar
```

---

### 5.3  Global Workspace (GWS) — Bowtie Bottleneck

`modules/workspace.py` — implements Baars / Dehaene Global Workspace Theory
via Modern Hopfield Networks with ignition dynamics.

#### Hopfield Convergence

$$\text{slot}^{(t+1)} = \mathrm{softmax}\!\left(\beta \cdot C \cdot {\text{slot}^{(t)}}^\top\right) C$$

where $C$ is the candidate set (all module outputs projected into slot space)
and $\beta = \mathrm{softplus}(\log\beta) + 0.5$ is a learned inverse
temperature.  Two Hopfield iterations converge to the nearest energy minimum —
pattern *completion*, not mere selection.

#### Ignition Phase Transition (Dehaene 2011)

1. **Lateral competition.** Off-diagonal cosine similarity between slots
   drives 15% inhibition of redundant patterns (winner-take-all in feature
   space — ensures distinct, informative slot states).

   $$\text{slots} \leftarrow \text{slots} \cdot \bigl(1 - 0.15 \cdot \overline{\cos\text{-sim}_{\text{off-diag}}}\bigr)$$

2. **Ignition gate** (per slot, learnable threshold $\theta_k$):

   $$p_{\text{ign},k} = \tfrac{1}{2} + \tfrac{1}{2}\tanh\!\bigl(6 \cdot (\lVert\text{slot}_k\rVert - \theta_k)\bigr)$$

   The slope-6 $\tanh$ delivers a steeper transition than a $\sigma$ at
   the same effective gain — closer to a true phase change.

3. **Broadcast scale** is a clamped linear lift from sub-conscious leak to
   full ignition:

   $$\text{scale}_k = 0.3 + 0.7 \cdot p_{\text{ign},k}$$

   Pre-ignition slots therefore contribute at $\sim 0.3$ (sub-conscious),
   ignited slots at $\sim 1.0$ (broadcast). The default initial threshold
   is $\theta_k = 0.5$.

After Stage 5, `orchestrator.set_gws_broadcast(slots.mean(1))` stores the
mean slot for all subsequent within-pass feedback loops (§4.2 Loop B), and
`workspace._last_ignition` is exposed as a $(B,)$ tensor used by the
trainer's per-step log line (`ign` field).

---

### 5.4  Expert Cortices

#### Math Cortex (`modules/math.py`)

- **DNC-heavy topology**: differentiable neural computer with sparse temporal
  link matrix for structured, multi-step reasoning.
- **Sparse link matrices** limit interference between independent reasoning
  chains (each math problem activates an isolated memory trace).
- Receives extra BDNF when the forward-model prediction error is low
  (correct procedure → reinforce the wiring that found it).

#### Language Cortex (`modules/language.py`)

- **DiffTransformer SNR-doubling** (§5.2): dual-softmax cancels attention
  noise, giving twice the effective resolution per parameter at no cost.
- **CALM early exit**: easy tokens (function words, punctuation) exit at
  shallow layers; hard tokens (novel concepts, long-range dependencies) use
  the full depth.

#### Reasoning Cortex (`modules/reasoning.py`)

- **Modern Hopfield pattern completion**: the Hopfield energy landscape stores
  a library of reasoning patterns (analogy frames, logical templates).
- On each new query the network converges to the nearest stored pattern,
  composing the relevant reasoning strategy.
- `gws_feedback_projs["reasoning"]` carries the GWS broadcast back into the
  pattern library query, allowing the global context to bias which reasoning
  strategy is retrieved.

---

### 5.5  Thalamic Hub

`modules/thalamus.py` — re-entrant gating and lateral binding across cortices.

- Receives `assoc + reentry_bias` (Loop A, §4.2) as input.
- NE-gated relay: high NE → full pass-through; low NE → attenuate
  (sleep-like suppression).
- Acts as the **routing nexus**: the thalamus decides which sensory signal
  reaches the GWS bottleneck.
- `HomeostaticGate` at thalamus entry and exit stabilises signal magnitude
  (models thalamo-cortical gain control).

---

### 5.6  Hippocampus

`modules/hippocampus.py` — sparse key-value episodic memory implementing
Complementary Learning Systems with five recall streams.

#### Sub-regions

| Sub-region | Role | Implementation |
|------------|------|----------------|
| DG (Dentate Gyrus) | Sparse pattern separation | Fixed random projection → top-k winners |
| CA3 | Auto-associative completion | Learned QKV attention over stored memories |
| CA1 | Mismatch / novelty detection | MLP(expected, actual) → novelty ∈ [0,1] |

#### Five Recall Streams (fused via cross-attention before GWS integration)

1. **Semantic** — cosine kNN via chunked approximate kNN (§5.6.1).
2. **Temporal** — recency-weighted; recent writes score +0.4.
3. **Mood/emotional** — 0.4 × semantic + 0.4 × NT-state cosine + 0.2 × valence.
4. **Associative chain** — 2-hop: best semantic recall → second retrieval.
5. **DNC temporal traversal** — forward/backward via the DNC link matrix.

#### DNC Temporal Link Matrix (Graves 2016)

$$L \leftarrow (1 - w_w^\top \mathbf{1} - \mathbf{1} w_w^\top)_+ \circ L + w_w p^\top$$
$$p \leftarrow (1 - \textstyle\sum w_w) p + w_w$$

Forward traversal: $w_{\text{blend}} = 0.3 w_r + 0.7 (w_r L^\top)$
— recalls what came *after* the match.

Backward traversal: $w_{\text{blend}} = 0.3 w_r + 0.7 (w_r L)$
— recalls causal antecedents.

#### Chunked Approximate kNN

```python
def _approx_knn(query, keys, topk, chunk=2048):
    # query: (B, d),  keys: (N, d),  N up to 65 536
    best_vals, best_idx = [], []
    for start in range(0, N, chunk):
        scores = query @ keys[start:start+chunk].T   # (B, chunk)
        vals, idx = scores.topk(topk, dim=-1)
        best_vals.append(vals);  best_idx.append(idx + start)
    # merge across chunks
    all_v = cat(best_vals, -1);  all_i = cat(best_idx, -1)
    _, top = all_v.topk(topk, dim=-1)
    return all_v.gather(-1, top), all_i.gather(-1, top)
```

Peak memory: $O(B \times 2048 \times d)$ instead of $O(B \times N \times d)$.

---

## 6  Neurochemistry & Dynamical Mechanics

### 6.1  Neuromodulators

| Symbol | Nucleus | Primary Effects |
|--------|---------|-----------------|
| DA | VTA + SNc | Reward prediction error, attention sharpness, BG gating |
| NE | Locus Coeruleus | Arousal, thalamic gain, attention focus |
| 5HT | Raphe nuclei | Mood baseline, DMN suppression, temporal discounting |
| ACh | Nucleus Basalis | Memory encoding gain, PFC working memory |
| Glu | Cortical projections | Excitatory drive |
| GABA | BG → Thalamus | Inhibitory gating |
| eCB | Endocannabinoid | Language cortex disinhibition, memory erasure |

#### Receptor Banks

Each brain region has a `ReceptorBank` with signed weights.  Modulation
applies to the embedding before entering each region:
$x' = x \times (1 + \textstyle\sum_k r_k \cdot \mathrm{NT}_k)$

```
rcpt_pfc:   DA(+0.6), 5HT(+0.3), ACh(+0.4), GABA(-0.4)
rcpt_hippo: ACh(+0.5), Glu(+0.4)
rcpt_bg:    DA(+0.7), GABA(-0.5)
rcpt_thal:  NE(+0.5), GABA(-0.3)
rcpt_lang:  ACh(+0.3), eCB(-0.3)
rcpt_dmn:   5HT(-0.4), ACh(-0.2)
```

#### NT Release Pipeline (per forward pass)

1. VTA, LC, Raphe, NBM compute NT demand from novelty / RPE / arousal / mood.
2. `ProjectionGraph` releases NT along anatomical pathways.
3. `ReuptakeSystem` decays NT each step.
4. `ReceptorAdaptation` up/down-regulates receptor sensitivity.
5. `PlasticityGate` modulates per-parameter learning rate via NT milieu.
6. `TrophicSystem` updates axon trophic levels (§6.2).
7. `LateralHabenula` drives anti-reward: spikes when expected reward is not
   delivered, suppressing DA (learned aversion).

---

### 6.2  Trophic System — Φ-Gated & Fiedler-Gated BDNF

`neurochem/growth.py` — structural plasticity driven by integrated information
and spectral coherence.

#### Trophic Update Rule (per projection, per tick)

Each projection $(i \to j)$ maintains a scalar $\tau_{ij} \in [0,1]$
(trophic level).  Updated as:

$$\tau_{ij} \leftarrow \tau_{ij} + \Delta\tau, \qquad \Delta\tau = \text{BDNF}_{\text{eff}} \cdot (0.1 + \bar c_{ij}) - (\text{NGF} + \delta_{\text{decay}} + 0.001(1 - \bar c_{ij}))$$

where $\bar c_{ij}$ is the EMA of co-activation (cosine similarity of recent
source/destination activity scalars, $\alpha=0.05$).

#### Φ-Gated BDNF

High integrated information states release more trophic factor, structurally
locking the connectivity that enabled those states:

$$\text{BDNF}_\Phi = \text{BDNF} \times (1 + \phi_{\text{boost}} \cdot \max(0, \Phi))$$

Default `phi_boost = 2.0`.

#### Fiedler-Gated Homeostasis

When the spectral gap $\lambda_1$ is small (module graph near disconnection),
a homeostatic BDNF surge rewires the fault line identified by the Fiedler
vector:

$$\text{fiedler\_boost} = \max\!\left(0,\; 1 - \frac{\lambda_1}{0.3}\right) \times 2.0$$

$$\text{BDNF}_{\text{eff}} = \text{BDNF} \times \left(1 + \phi_{\text{boost}} \cdot \Phi + \text{fiedler\_boost}\right)$$

This means: when $\lambda_1 < 0.3$, BDNF can be up to $3\times$ the baseline,
specifically strengthening the projections that cross the identified minimum
cut.

#### Pseudo-code (PyTorch-compatible)

```python
def update(self, activities, bdnf, ngf, phi=0.0, fiedler=1.0):
    # Φ-gated + Fiedler-gated BDNF
    fiedler_boost = max(0.0, 1.0 - fiedler / 0.3) * 2.0
    bdnf_eff = bdnf * (1.0 + self.phi_boost * max(0.0, phi) + fiedler_boost)
    bdnf = min(0.05, max(0.0, bdnf_eff * 0.05))    # scale to [0, 0.05]
    ngf  = min(0.01, max(0.0, ngf  * 0.01))

    for i, proj in enumerate(self.graph.projections):
        a, b = activities.get(proj.src), activities.get(proj.dst)
        co = float((a * b).mean().clamp(0, 1)) if (a and b) else 0.0
        self.ema_coact[i] = 0.95 * self.ema_coact[i] + 0.05 * co

        growth = (bdnf + self.bdnf_baseline) * (0.1 + self.ema_coact[i])
        decay  = ngf + self.ngf_decay + 0.001 * (1.0 - self.ema_coact[i])
        self.trophic[i] = (self.trophic[i] + growth - decay).clamp(0, 1)

        # Pruning / re-sprouting
        if self.trophic[i] < self.prune_threshold:   self.active[i] = 0.0
        elif self.trophic[i] > 2*self.prune_threshold: self.active[i] = 1.0

    # gain(idx) = active[idx] * (0.2 + 1.6 * trophic[idx])  ← signal scaling
```

**Tensor shapes:**
- `trophic`: `(n_projections,)` float32
- `active`:  `(n_projections,)` float32 ∈ {0, 1}
- `ema_coact`: `(n_projections,)` float32

---

### 6.3  Hebbian Fast Weights (HFW)

`modules/fast_weight.py` — dual-timescale learning at expert cortex outputs.

#### Mechanism

The fast-weight matrix $W_F \in \mathbb{R}^{B \times H \times D_h \times D_h}$
accumulates outer-product associations within a single sequence:

$$W_F^{(t)} = \lambda W_F^{(t-1)} + \eta_t \cdot (g_t \odot v_t) \otimes k_t$$

- $\lambda = 0.95$ — exponential decay (recent associations weighted most).
- $\eta_t = \eta_0 \cdot \mathrm{softplus}(\text{MLP}(\text{context}))$ —
  context-dependent (GWS-modulated) per-head learning rate.
- $g_t = \sigma(\text{Linear}(x_t))$ — write gate, prevents catastrophic
  overwrite of earlier associations.

Retrieval:

$$y_t = \mathrm{LN}(W_F^{(t-1)} q_t)$$

#### Full PyTorch Pseudo-code

```python
class FastWeightLayer(nn.Module):
    # Tensor shapes annotated inline
    def forward(self, x, context=None, W_fast=None):
        B, T, D = x.shape                   # x: (B, T, d_sem)
        Dh = D // self.n_heads              # per-head dim

        if W_fast is None:
            W_fast = zeros(B, self.n_heads, Dh, Dh, device=x.device)
        if context is None:
            context = x.mean(1)             # (B, D)

        eta = self.base_eta * (self.eta_mod(context) + 1e-6)  # (B, H)

        k = self.k_proj(x).view(B,T,H,Dh).permute(0,2,1,3)   # (B,H,T,Dh)
        v = self.v_proj(x).view(B,T,H,Dh).permute(0,2,1,3)
        q = self.q_proj(x).view(B,T,H,Dh).permute(0,2,1,3)
        g = sigmoid(self.g_proj(x).view(B,T,H,Dh).permute(0,2,1,3))

        outs = []
        for t in range(T):
            read  = einsum("bhij,bhj->bhi", W_fast, q[:,:,t])   # (B,H,Dh)
            outs.append(layer_norm(read, [Dh]))
            outer = einsum("bhi,bhj->bhij", g[:,:,t]*v[:,:,t], k[:,:,t])
            W_fast = self.decay * W_fast + eta[:,:,None,None] * outer

        out_seq = stack(outs, 2).permute(0,2,1,3).reshape(B,T,D)  # (B,T,D)
        return self.ln(x + self.out_proj(out_seq)), W_fast.detach()
```

`W_fast.detach()` prevents gradients from flowing through the fast-weight
path across timesteps — the slow-weight backbone is unaffected.  Carry-over
state `_hfw_states[name]` persists within a dialogue session; cleared by
`orchestrator.reset_fast_weights()` between sequences.

---

### 6.4  Neuro-Vesicle Pool

`neurochem/vesicles.py` — discrete synaptic-vesicle-like packets providing
slow, long-range neuromodulation.

#### Vesicle Data Structure

```python
# All fields stored as tensors of shape (n_vesicles,) or (n_vesicles, d_sem)
content   : (n_vesicles, d_sem)  # semantic payload
lifetime  : (n_vesicles,)        # int; ticks until degradation
position  : (n_vesicles,)        # int; current module index ∈ [0, n_modules)
active    : (n_vesicles,)        # bool; alive flag
```

#### Life Cycle (one tick per `cognitive_step`)

| Phase | Tensor Operations |
|-------|-------------------|
| **Synthesis** | `novelty = (novelty_scalar * floating_thought).norm(dim=-1)` → synthesis gate MLP → `content[new_slot] = projection(signal)` |
| **Migration** | `T = softmax(log_T, dim=-1)` shape `(n_modules, n_modules)`; position updated by `multinomial(T[position])` per active vesicle |
| **Docking** | `dock_score = cosine(content_key[v], module_query[pos[v]])` → `delta = dock_proj(content[v]) * sigmoid(dock_score)` shape `(d_sem,)` |
| **Degradation** | `lifetime -= 1`; `active[lifetime <= 0] = False` |

**Modulation output:**
```python
mod_out: (B, n_modules, d_sem)   # summed dock deltas per module
# Applied to GWS slots after hippocampal enrichment step
```

**GPCR-like stateful gating:** the migration matrix $T$ is a learned
row-stochastic operator trained end-to-end.  Vesicles that reach a module with
high cosine affinity dock and release their payload — analogous to
neurotransmitter diffusion and GPCR binding, with docking probability set by
receptor affinity rather than hard routing.

**Config flags:**
- `enable_vesicles: bool = False`
- `n_vesicles: int = 32`
- `vesicle_lifetime: int = 16`

---

## 7  Intelligence & Integration Metrics

### 7.1  Φ Proxy — Algorithmic Steps

`intelligence/orchestrator.py::compute_phi_proxy()` and
`modules/consciousness.py::ConsciousnessMetrics._compute_phi_mip()`
provide two complementary Φ estimates:

**Orchestrator proxy (fast, correlation-based):**

```
1. Collect _last_stage_outputs (up to 16 recent stage outputs)
2. For each output: batch-mean → L2 normalise → unit vector v_i ∈ R^d
3. Compute pairwise cosine similarities c_{ij} = v_i · v_j for i < j
4. mean_c = mean(c_{ij});  std_c = std(c_{ij})
5. Φ_proxy = mean_c × (1 + std_c)   ∈ [0, 1]
```

High Φ requires both integration (nonzero mean coupling) and differentiation
(high variance of coupling).  If all modules output identical signals,
std_c = 0 and Φ_proxy saturates correctly low.

**ConsciousnessMetrics MIP estimate (rigorous, tick-level):**

```
1. Collect n module output vectors, each mean-pooled to (d,); cap n=8.
2. Centre: M = M - mean(M, dim=0);  compute Gram Σ = M @ M.T / (d-1)
3. If n ≤ 8:  enumerate all 2^(n-1)-1 bipartitions; return min MI(A;B)
4. If n > 8:  compute Fiedler bisection (§2.3); return MI at that cut
5. MI(A;B) = 0.5 * (logdet(Σ_A) + logdet(Σ_B) - logdet(Σ_AB))
6. Clamp result to [0, 10]; store in _phi_history (64-tick ring buffer)
```

`estimate_fiedler()` (§2.3) is called externally and its $\lambda_1$ result
is passed to `TrophicSystem.update(fiedler=λ₁)` (§6.2).

### 7.2  Comprehension Index (*cmp*)

`memory/comprehension_gate.py` — the write-gating filter.

$$\text{cmp}_t = \text{LM\_loss}_t \times \text{comprehension\_score}_t$$

The comprehension score is a learned MLP that estimates how well the model
"understands" the current input given its current hidden state.  High *cmp*
means: this observation is **surprising** (high LM loss) *and* understood
(high comprehension score).

Only observations with $\text{cmp}_t > \tau_t$ are written to episodic
memory.  The threshold $\tau_t$ is self-calibrating:

$$\tau_{t+1} = \tau_t + \alpha \cdot (\bar w_t - w^*)$$

where $\bar w_t$ is the recent write rate and $w^* = 0.10$ is the target.

This implements a **curiosity-driven memory filter**: the model prioritises
storing what it found surprising and managed to understand — new knowledge
at the boundary of its competence.

### 7.3  Full Metrics Dashboard

| Metric | Formula | Module |
|--------|---------|--------|
| **γ (gamma)** | Mean off-diagonal cosine sim of GWS slots | `consciousness.py` |
| **θ (theta)** | Mean CA1 novelty signal | `consciousness.py` |
| **α (alpha)** | Normalised routing entropy | `consciousness.py` |
| **Φ (phi)** | MIP lower bound (§7.1) | `consciousness.py` |
| **Coherence** | Cosine alignment of module outputs with GWS mean | `consciousness.py` |
| **Ignition** | Fraction of modules with norm > 0.6 | `consciousness.py` |
| **Metacognition** | $\sigma(\lVert\text{thought}\rVert - 1)$ | `consciousness.py` |
| **Binding** | γ × Coherence | `consciousness.py` |
| **λ₁ (Fiedler)** | Spectral gap via power iteration | `consciousness.py` |
| **Causal density** | Causal rules / observations | `intelligence/metrics.py` |
| **Narrative coherence** | Cosine sim of narrative buffer | `intelligence/metrics.py` |
| **Identity drift** | MSE of self-model across ticks | `orchestrator.py` |
| **Comprehension (cmp)** | LM loss × comprehension score | `memory/comprehension_gate.py` |

---

## 8  Optimization & Infrastructure

### 8.1  Adaptive Compute

**CALM (Confident Adaptive Language Modeling)**

Per-token early exit at every `MoDBlock`.  Threshold schedule:

$$\theta_l = \theta_{\text{base}} \times e^{-\gamma l / (L-1)}, \qquad \theta_{\text{base}} = 0.25, \quad \gamma = 2.0$$

Inference-only; does not affect training gradients.

**MoD (Mixture of Depths)**

Per-layer token routing: a 2-layer MLP assigns capacity score $r_t$;
top-$c$ fraction executes full FFN (default $c = 0.8$, configurable via
`mod_capacity`).

### 8.2  TPU/XLA Backend

The model is designed to run efficiently on Cloud TPUs with the following
constraints:

| Constraint | Implementation |
|------------|----------------|
| **bfloat16 precision** | All tensors cast to bf16 at forward entry; buffers stored in model's `.dtype` |
| **Dtype safety** | Every buffer read is cast via `.to(dtype=weight.dtype)` before use — prevents the float32/bf16 mismatch error (fixed in `world_model.py`, `growth.py`, `orchestrator.py`) |
| **Query-Key Normalisation** | `F.normalize(q, dim=-1)` and `F.normalize(k, dim=-1)` before attention in GWS and Hippo to stabilise bf16 dot-products |
| **Approximate kNN** | Chunked kNN (§5.6.1) avoids materialising `(B, N, d)` tensors; scales to 65K+ stored memories |
| **No torch.compile dynamic shapes** | All dynamic sequences are padded to fixed length or chunked to static sizes |
| **XLA-safe gradient checkpointing** | `torch.utils.checkpoint` conditionally disabled on XLA devices where it is unsupported |
| **Adafactor optimiser** | Used on TPU (`lr=None` → Adafactor adaptive step; `--optimizer adafactor`) |

**Graph operations (Φ, Fiedler)** are wrapped in `@torch.no_grad()` and use
float32 regardless of model dtype — they are metrics, not part of the
backward graph.  On Apple Silicon the power-iteration inner loop is SIMD-
friendly (sequential `matmul` of small square matrices).

### 8.3  HomeostaticGate

Every signal crossing a module boundary passes through a `HomeostaticGate`
that stabilises signal magnitude and models thalamo-cortical gain control:

```python
class HomeostaticGate(nn.Module):
    # x: (B, T, D) or (B, D)
    def forward(self, x):
        h = x + self.attn(self.norm(x))           # pre-synaptic refinement
        h = h + self.ff(self.ff_norm(h))
        # Online mean/var tracking (EMA, α=0.01)
        self.running_mean.lerp_(h.mean((0,1)), α)
        self.running_var.lerp_(h.var((0,1)), α)
        rms = (self.running_var + 1e-8).sqrt()
        return (h - self.running_mean) / rms * target_mag * self.gain + self.bias
```

`gain` and `bias` are learned per-dimension parameters.  `running_mean`/
`running_var` are non-gradient buffers (saved in checkpoint).

---

## 9  Training

### 9.1  Data Pipeline

`neuroslm/data.py` — interleaved streaming:

- **Text:** FineWeb-Edu (10B+ tokens), Cosmopedia, TinyStories
- **Chat:** OpenHermes-2.5, UltraChat-200k, WildChat-1M, SlimOrca, hh-rlhf, Dolly-15k
- **Mode `mix`:** `chat_ratio` fraction chat, remainder text (default 60/40)

### 9.2  Loss Function

$$\mathcal{L} = w_{\text{lm}} \cdot \mathcal{L}_{\text{CE}} + w_{\text{world}} \cdot \mathcal{L}_{\text{world}} + w_{\text{self}} \cdot \mathcal{L}_{\text{self}} + w_{\text{fwd}} \cdot \mathcal{L}_{\text{fwd}} + w_{\text{val}} \cdot \mathcal{L}_{\text{val}} + w_{\text{motor}} \cdot \mathcal{L}_{\text{motor}} + w_{\text{pc}} \cdot \mathcal{L}_{\text{pred\_coding}} + \ldots$$

| Term | Weight | Description |
|------|--------|-------------|
| $\mathcal{L}_{\text{CE}}$ | 1.0 | Cross-entropy next-token (chunked, T=128 slices) |
| $\mathcal{L}_{\text{world}}$ | 0.3 | World state MSE |
| $\mathcal{L}_{\text{self}}$ | 0.1 | Self-model MSE |
| $\mathcal{L}_{\text{fwd}}$ | 0.2 | Forward model MSE |
| $\mathcal{L}_{\text{val}}$ | 0.1 | Evaluator value MSE |
| $\mathcal{L}_{\text{motor}}$ | 0.05 | Action selection CE |
| $\mathcal{L}_{\text{pc}}$ | 0.1 | Inter-layer predictive coding |
| $\mathcal{L}_{\text{cpc}}$ | 0.05 | Contrastive predictive coding (optional) |
| $\mathcal{L}_{\text{KL}}$ | 0.1 | RSSM KL (if `enable_rssm`) |
| $\mathcal{L}_{\text{FE}}$ | 0.05 | Active inference free energy (if `enable_active_inference`) |
| $\mathcal{L}_{\text{social}}$ | 0.1 | ToM social prediction error (if `enable_tom`) |

### 9.3  Memory-Efficient Training

- **Chunked cross-entropy:** CE computed in $T=128$ token slices; avoids
  materialising the $(B \times T \times V)$ logit tensor.
- **Gradient checkpointing:** recomputes activations during backward; saves
  ~50% activation memory.  Forced on when `device == "cuda"`.
- **Gradient accumulation:** `--grad_accum N` gives effective batch
  $= B \times N$ at $1/N$ peak activation memory.
- **`del logits`** after motor logits frees one $(B, T, V)$ tensor before CE.

### 9.4  Optimiser

```
AdamW:   lr=2e-4 (xl),  weight_decay=0.1,  grad_clip=1.0
Adafactor (TPU):  lr=None (adaptive),  grad_clip=1.0
Schedule: linear warmup (800 steps) → cosine decay
NT modulation: PlasticityGate applies soft per-step LR multiplier ∈ [0.1, 3.0]
```

---

## 10  Inference Pipeline

`Brain.cognitive_step(token)` — full forward pass order:

```
token → LanguageCortex (thought + NT modulation, CALM early exit at MoDBlocks)
      → TextSensoryCortex  (salience gating, novelty)
      → AssociationCortex  (multimodal fusion)
      → Thalamus           (NE-gated relay + Loop-A re-entry bias)
      → WorldModel         (RSSM or GRU; updates h_world)
      → SelfModel          (interoceptive self-state)
      → Amygdala, LHb, Insula  (emotional colouring, aversion)
      → QualiaState        (phenomenal state vector update)
      ↓
      GlobalWorkspace      (Hopfield convergence → lateral competition → ignition)
      → orchestrator.set_gws_broadcast(slots.mean(1))     ← Loop B enabled
      → VesiclePool.tick() (synthesize → migrate → dock → degrade)
      ↓                    [GWS feedback now active for all below]
      → Hippocampus        (5-stream recall; ACh-gated enrichment)
      → EntorhinalCortex   (grid cell conceptual navigation)
      → Cerebellum         (prediction error update)
      → PFC                (thought selection + replace-gate)
                           [HFW active: W_fast updated]
      → ACC                (conflict monitoring)
      → BasalGanglia       (DA-modulated action selection)
                           [HFW active: W_fast updated]
      → ForwardModel + Evaluator
      → DMN                (every dmn_period steps: spontaneous reflection)
      → ThoughtTransformer + Claustrum
      → ToM update         (if enable_tom)
      → ActiveInference    (2-pass predictive coding; if enable_active_inference)
      → ConsciousnessMetrics.update() [Φ-MIP, γ/θ/α, ignition, λ₁]
      → MotorCortex → action-conditioned token bias
      → NT nuclei release + ProjectionGraph + Reuptake
      → TrophicSystem.update(phi=Φ, fiedler=λ₁)
      → orchestrator.update_reentry(pfc_gws_signal)       ← Loop A stored
      → EpisodicMemory.store() (gated by ComprehensionGate)
      → NarrativeSystem.update()
      → EntityStore.update()
      → HyperGraph.update()
```

`floating_thought` — EMA ($\alpha=0.3$) of PFC output — persists across
tokens as a "held thought" biasing the next language cortex forward pass.

---

## 11  Topology Modes & Ablation

```python
cfg.neural_topology = 'full'      # all modules active (default)
cfg.neural_topology = 'baseline'  # vanilla transformer only

cfg.baseline = True               # builds only LanguageCortex; skips all modules

brain.disable_module('hippo')     # bypass at runtime (zero passthrough)
brain.enable_module('tom')        # re-enable
brain.module_status()             # dict of enabled/disabled per module
```

Ablation cell (Colab, cell `8049a7fd`) trains **full model first** (results
are primary), then baseline (param-matched vanilla transformer with
`baseline_lang_layers=56` layers at d_hidden=512 ≈ same parameter count).

### Checkpoint Persistence

| File | Contents |
|------|----------|
| `.pt` | Model weights + optimiser + genome state + `_global_step` |
| `.mem` | Episodic buffer, consolidated memory, relational graph, causal rules, narrative, entity store |
| `.dna.json` | Human-readable evolved genome snapshot |

All three written every `save_every` steps and pushed to Git LFS immediately.

---

## 12  Open Questions & Future Work

- **No RL loop yet** — world/self/forward losses are supervised proxies; a
  real environment or RLHF signal would greatly strengthen them.
- **Neuromodulators are not grounded** — NT levels are initialised from zero
  and shaped only by the training signal.  A real sensory environment would
  make them semantically meaningful.
- **Φ estimation is approximate** — true IIT Φ is NP-hard.  The MIP lower
  bound via Gaussian MI is tractable and principled but not Φ_max.  For
  $n \le 8$ the enumeration is exact; for $n > 8$ the Fiedler bisection is
  a near-optimal heuristic.
- **CALM threshold is not trained** — adaptive calibration via policy gradient
  on exit-vs-accuracy is left for future work.
- **Vesicle migration is stochastic** — `torch.multinomial` introduces
  non-determinism at inference.  For reproducible evaluation, fix the seed or
  replace with argmax migration.
- **DNC link matrix is dense** — `_dnc_L: (capacity, capacity)` float32.
  At capacity=4096 → 64 MB; at 32768 → 4 GB.  Sparse COO/CSR storage needed
  for very large capacities.
- **gws_feedback_projs are zero-init** — they start inactive and must be
  learned from data.  Pre-training with auxiliary Φ-maximisation loss would
  accelerate activation of the backward loops.
- **Fiedler-gated BDNF requires warm Φ estimates** — in the first ~100 steps
  before Φ rises above zero, the homeostatic boost may over-strengthen random
  connections.  A warm-up schedule for `fiedler_boost` (e.g., ramp linearly
  over 200 steps) may be beneficial.

---

*Last updated: 2026-05-10.  Reflects commit `d963075` on branch `tpu`.*
