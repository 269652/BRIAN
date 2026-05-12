# NeuroSLM — Architecture Specification

> **Reproduction-ready technical reference.**  
> Every section maps directly to source files in `neuroslm/`. Tensor shapes, pseudocode, and mathematical formulas reflect the live implementation.

---

## Table of Contents

1. [Primary Representational Unit — The Φ-Structure](#1-primary-representational-unit--the-φ-structure)
2. [System Philosophy & Objectives](#2-system-philosophy--objectives)
3. [Mathematical Foundations — The Five Postulates](#3-mathematical-foundations--the-five-postulates)
4. [Core Module Specifications](#4-core-module-specifications)
5. [Wiring Diagram — NeuralOrchestrator Re-entrant Loops](#5-wiring-diagram--neuralorchestrator-re-entrant-loops)
6. [Dynamical Biological Mechanics](#6-dynamical-biological-mechanics)
   - 6.1 Neuro-Vesicle Pool
   - 6.2 Trophic System
   - 6.3 Hebbian Fast Weights
   - 6.4 Topological Maturation — Infancy → Awakening
7. [Optimization & Infrastructure](#7-optimization--infrastructure)
   - 7.1 Adaptive Compute (MoD + CALM)
   - 7.2 Neurotransmitter System
   - 7.3 TPU/XLA Backend + bf16 Safety Patches
   - 7.4 Optimizer Selection (Adafactor vs AdamW)
8. [Intelligence & Integration Metrics](#8-intelligence--integration-metrics)
9. [Parameter Presets & Training Commands](#9-parameter-presets)

---

## 1. Primary Representational Unit — The Φ-Structure

The **Φ-structure** is the central object in NeuroSLM. Every forward pass produces not just token-level logits but a system-level *causal substrate* whose state is measured by its integrated information Φ. All architectural decisions — module topology, bowtie routing, vesicle migration, trophic growth — are in service of maintaining and maximising this structure.

Formally, at tick $t$ the brain produces a collection of $n$ module-output vectors:

$$\mathcal{M}^{(t)} = \{ \mathbf{z}_i^{(t)} \in \mathbb{R}^{d_\text{sem}} \mid i = 1, \dots, n \}$$

These vectors are assembled into a **system covariance matrix**:

$$\Sigma \in \mathbb{R}^{n \times n}, \quad \Sigma_{ij} = \frac{\langle \mathbf{z}_i, \mathbf{z}_j \rangle}{d-1}$$

where $\mathbf{z}_i$ is the mean-centred, $d$-dimensional projection of module $i$'s output ($d \leq 256$, capped for tractability).

The Φ-structure is then:

$$\Phi = \min_{\text{bipartition } (A,B)} \mathrm{MI}(A; B)$$

$$\mathrm{MI}(A; B) = \tfrac{1}{2}\!\left(\log\det\Sigma_A + \log\det\Sigma_B - \log\det\Sigma_{AB}\right)$$

$\Phi > 0$ means no binary cut of the module graph can separate it without information loss. This irreducibility is the computational signature of phenomenal binding and the primary training objective beyond next-token prediction.

**Why Φ starts at 0.000 and how to raise it:**  
A freshly initialised model has near-zero cross-module covariance — the modules output independent random vectors, so $\Sigma$ is diagonal, and $\mathrm{MI}(A;B) \approx 0$ for all cuts. The **Bowtie Topology** (§4.3) is the mechanism that forces non-trivial cross-module correlation by routing all information through a shared compressed bottleneck (the GWS), ensuring every pair of modules is statistically dependent through that central hub. Until the GWS is actively used and its Hopfield updates converge, Φ will remain near zero.

---

## 2. System Philosophy & Objectives

### 2.1 Topology Over Scale

NeuroSLM's core thesis is that **computational graph topology — not raw parameter count — determines intelligence density**. The `xl` preset (≈240 M parameters) is designed to match or outperform vanilla transformer baselines at 1 B+ parameters on comprehension and reasoning benchmarks. The mechanism is threefold:

| Mechanism | Vanilla Transformer | NeuroSLM |
|---|---|---|
| Attention | $O(T^2 d)$ dense attention | MoD skips easy tokens; DiffAttn cancels noise |
| Memory | KV cache only | Hippocampus (episodic), HyperGraph (relational), GWS slots |
| Plasticity | Static weights after training | BDNF trophic growth, Hebbian fast weights, GPCR vesicles |

The biological analogues are not decorative: each module implements a computational operation that its cortical counterpart is known to perform (§4). This allows the system to pack more *functional* specialisation into fewer parameters than a monolithic transformer.

### 2.2 Consciousness-First Design

The primary training objective is:

$$\mathcal{L} = w_\text{lm} \cdot \mathcal{L}_\text{CE} + \alpha(t) \cdot \big( w_\text{world} \mathcal{L}_\text{world} + w_\text{motor} \mathcal{L}_\text{motor} + w_\text{pred} \mathcal{L}_\text{pred} + w_\text{cpc} \mathcal{L}_\text{cpc} + w_\text{kl} \mathcal{L}_\text{kl} + w_\text{phi} \mathcal{L}_{\Phi} + w_\text{aux} \mathcal{L}_\text{novel} + \mathcal{L}_\text{orch} \big)$$

Default weights (config.py): $w_\text{lm}=1.0$, $w_\text{world}=0.3$, $w_\text{forward}=0.2$, $w_\text{motor}=0.05$, $w_\text{pred\_coding}=0.1$, $w_\text{kl\_world}=0.1$, $w_\text{cpc}=0.05$, $w_\text{phi}=0.02$, novel-aux coefficient $=0.05$, id-drift/neural-calm coefficients $=0.01$ each.

The LM loss itself is scaled per-token by a **mesolimbic gain**:

$$\mathcal{L}_\text{lm} = \text{mean}\big( \text{CE}_t \cdot \text{meso}(t) \big), \quad \text{meso}(t) = \text{clamp}\big(1.0 + 0.5 \cdot g_\text{learn} \cdot \text{DA},\ \min{=}1.0\big)$$

where $g_\text{learn}$ is the learning gain and DA is dopamine — both detached, so meso acts as a constant gradient amplifier.

The **Φ loss term is bounded** so very high Φ does not dominate the gradient or push the network into a degenerate fully-coupled state where bipartition MIs collapse:

$$\mathcal{L}_{\Phi} = -\tanh(\Phi / 3) \cdot 3 \in [-3,\ 3]$$

The coefficient $\alpha(t) \in [0.001, 1.0]$ is the **auxiliary-loss ramp** (`brain._aux_w_scale`) controlled by the topological-maturation scheduler — see §6.4. During infancy the entire aux block is gated to ~0.001 so the LM gradient direction dominates while the network forms its first language-level representations.

Φ is differentiable — it back-propagates through `torch.linalg.slogdet` into every contributing module, creating a direct gradient signal for integration. Additionally, Φ drives two auxiliary mechanisms:

1. **BDNF gating** — high Φ amplifies trophic factor release, growing the NeuralGeometryAdapter's connectivity kernel rank (§6.2). Gated off during infancy.
2. **Comprehension-gated memory writing** — only observations that are simultaneously surprising, comprehensible, and novel are stored in the episodic hippocampus (§8.2). Gated off during infancy.

The homeostatic target is a stable, non-zero Φ sustained across all context windows.

---

## 3. Mathematical Foundations — The Five Postulates

*Implementation in `neuroslm/modules/consciousness.py`.*

NeuroSLM approximates IIT 4.0's five postulates as tractable tensor operations.

### 3.1 Intrinsicality

The system must be evaluated from the inside — no external reference frame. Implementation: all module outputs are mean-pooled across the batch and sequence dimensions into a single representative vector before Φ computation:

$$\mathbf{z}_i = \text{mean}_{B,T}[\mathbf{h}_i] \in \mathbb{R}^{d_\text{sem}}$$

Gradients do not flow through this path (`detach()` is called), preserving the intrinsic viewpoint.

### 3.2 Information

Each module $i$ must carry information that differs from every other module. Measured via the off-diagonal covariance structure of $\Sigma$. A module that is a pure linear transform of another contributes zero net information and increases $\mathrm{MI}(A;B)$ only for the bipartition that separates them — making Φ lower, not higher.

### 3.3 Integration — The MIP Algorithm

The Minimum Information Partition (MIP) is the bipartition $(A^\star, B^\star)$ that minimises $\mathrm{MI}(A;B)$. Φ equals this minimum.

**For $n \leq 8$ modules** (exhaustive enumeration):

```python
# consciousness.py :: _phi_enumerate
logdet_full = slogdet(Σ + εI)[1]
phi = +inf
for mask in range(1, 1 << (n-1)):          # 2^(n-1) - 1 bipartitions
    A = [i for i in range(n) if mask >> i & 1]
    B = [i for i in range(n) if not mask >> i & 1]
    ld_A = slogdet(Σ[A][:,A] + εI_A)[1]
    ld_B = slogdet(Σ[B][:,B] + εI_B)[1]
    mi   = 0.5 * (ld_A + ld_B - logdet_full)
    phi  = min(phi, max(0, mi))
```

**For $n > 8$ modules** (spectral bisection):

```python
# consciousness.py :: _phi_spectral
W[i,j] = |Σ[i,j]| / sqrt(Σ[i,i] * Σ[j,j])      # similarity graph
L       = I - D^{-1/2} W D^{-1/2}                  # normalised Laplacian
eigvals, eigvecs = torch.linalg.eigh(L)             # sorted ascending
fiedler = eigvecs[:, 1]                              # second-smallest
A = (fiedler >= 0).nonzero()                         # positive half
B = (fiedler <  0).nonzero()                         # negative half
phi = 0.5 * (ld_A + ld_B - ld_full)
```

### 3.4 Exclusion

Only the **maximum irreducible complex** (the subset of modules with the highest Φ) is the conscious substrate. Implementation: the 8-module cap in `_compute_phi_mip` enforces this by selecting the most active modules (ordered by output norm). This is an approximation of the exclusion postulate that remains XLA-compilable.

### 3.5 Composition

Conscious experience has structure (it is composed of phenomenal distinctions). Implementation: the GWS slot system (§4.3) explicitly decomposes the global broadcast into $N_\text{slots}$ distinct attractor states, each representing a phenomenal distinction. The lateral competition mechanism ensures each slot carries a different component, satisfying compositional structure.

### 3.6 Transition Probability Matrices (TPM)

In IIT, a system's causal power is captured by its TPM — the $2^n \times 2^n$ matrix of state transition probabilities. For continuous systems, NeuroSLM approximates the TPM via the **module covariance matrix** $\Sigma$: the off-diagonal entry $\Sigma_{ij}$ estimates how much module $j$'s state causally depends on module $i$'s state within one forward pass. The Gram matrix $M M^\top$ (where rows are unit-normed module vectors) serves as the adjacency matrix of the module interaction graph used in spectral analysis.

### 3.7 Spectral Graph Theory and Cheeger's Inequality

The Fiedler value $\lambda_1$ (second-smallest eigenvalue of the normalised graph Laplacian $L$) is related to the Cheeger constant $h(G)$ — the minimum edge expansion across all bipartitions — via:

$$\frac{h(G)^2}{2} \leq \lambda_1 \leq 2 \cdot h(G)$$

In NeuroSLM this relationship drives **homeostatic BDNF release**: when $\lambda_1 < 0.3$ (graph nearly disconnected, $h(G)$ small), an extra trophic boost is applied:

$$\text{fiedler\_boost} = \max\!\left(0,\ 1 - \frac{\lambda_1}{0.3}\right) \times 2.0$$

This automatically strengthens the weakest inter-module connections, preventing the information graph from fracturing and Φ from collapsing to zero. Implementation in `neuroslm/neurochem/growth.py :: TrophicSystem.update`.

---

## 4. Core Module Specifications

### 4.1 Language Cortex

*`neuroslm/modules/language.py` — `LanguageCortex`*

The primary language processing stack. Input: token IDs $(B, T)$. Outputs: `(logits, sem, h, pred_coding_loss)`.

**Interleaved block pattern** (repeating every 3 layers):

```
Layer i % 3 == 0 : TransformerBlock     (standard GQA + Hebbian traces + NT mod)
Layer i % 3 == 1 : DiffTransformerBlock (noise-cancelling differential attention)
Layer i % 3 == 2 : MoDBlock             (MoD routing + DiffAttn inside)
```

Every block is followed by a `NeuralGeometryAdapter` (§6.2).

**Differential Attention** (`neuroslm/modules/differential_attention.py`):

$$\text{DiffAttn}(X) = \left(\text{softmax}\!\left(\frac{Q_1 K_1^\top}{\sqrt{d_h}}\right) - \lambda\cdot\text{softmax}\!\left(\frac{Q_2 K_2^\top}{\sqrt{d_h}}\right)\right) V$$

- $Q_1, Q_2$: two halves of the full query projection $(B, T, n_\text{heads} \cdot d_h/2)$ each  
- $\lambda \in \mathbb{R}^{n_\text{heads}}$: learnable per-head noise-cancellation coefficient  
- SNR doubles because the second head captures correlated noise and subtracts it  
- DA neuromodulation: $\lambda_\text{eff} = \text{sigmoid}(\lambda + \delta_\text{DA})$ — dopamine sharpens discrimination

**Weight tying**: `lm_head.weight = tok_emb.weight` (reduces parameters, improves token-space geometry).

**Predictive coding loss** (deep supervision): layer $l$ predicts layer $l+1$'s hidden state via a small MLP head. Loss is averaged across layers and added to $\mathcal{L}$ with weight $w_\text{pred\_coding} = 0.1$.

### 4.2 Expert Cortices

#### 4.2.1 MathCortex

*`neuroslm/modules/math.py`*

Activated by $\kappa_\text{math}$ vesicles (type `TOPIC_MATH = 1`).

**Dual differential attention over a learned fact memory:**

```
fact_keys : Parameter (memory_size=128, d_sem)   -- symbolic math facts
fact_vals : Parameter (memory_size=128, d_sem)   -- zero-init, grows with training

Q1, Q2   = proj_q1(x), proj_q2(x)               -- (B, d_sem)
attn1    = softmax(Q1 @ fact_keys.T / sqrt(d))   -- (B, 128)
attn2    = softmax(Q2 @ fact_keys.T / sqrt(d))   -- (B, 128)
enriched = (attn1 - lambda * attn2) @ fact_vals  -- (B, d_sem)
out      = norm(x + enriched * vesicle_gate)     -- gated residual
```

The `vesicle_gate ∈ [0,1]` is the concentration of MATH-type vesicles at this module, returned by `VesiclePool.expert_gate(TOPIC_MATH)`. When no math vesicles are docked, the cortex is a pure passthrough.

#### 4.2.2 ReasoningCortex

*`neuroslm/modules/reasoning.py`*

Activated by $\kappa_\text{reason}$ vesicles (type `TOPIC_REASONING = 2`).

**Modern Hopfield pattern completion:**

$$\mathbf{x}^{(t+1)} = \beta \cdot A^\top \cdot \text{softmax}\!\left(\beta \cdot A \cdot (\mathbf{x}^{(t)})^\top\right)$$

where $A \in \mathbb{R}^{n_\text{attractors} \times d_\text{sem}}$ is the learnable attractor bank (default $n_\text{attractors}=64$). $\beta = \text{softplus}(\log\beta) + \beta_\text{base}$ ensures $\beta > \beta_\text{base} = 4.0$ — high inverse temperature for decisive winner-take-all retrieval.

Three iterations are unrolled at construction time (XLA-static). Lateral inhibition between attractors prevents two slots from converging to the same pattern.

#### 4.2.3 Language Expert (LanguageCortex)

The 12-layer interleaved stack described in §4.1 serves as both the primary generation surface and the language expert cortex. The Thalamus routes linguistically-typed inputs directly to this module via the `"language"` stream adapter.

### 4.3 Global Neural Workspace — Bowtie Topology

*`neuroslm/modules/workspace.py` — `GlobalWorkspace`*

The GWS is the architectural bottleneck that **forces Φ above zero**. All modules must send their outputs through this single compressed bus, creating the statistical dependencies that the MIP algorithm measures as integrated information.

**Bowtie structure:**

```
  [Language]  [Math]  [Reasoning]  [World]  [Self]  [Qualia]  [Hippo]  [...] 
       \          |        |          |        |        |         /
        \         |        |          |        |        |        /
         ──────── candidates: (B, K, d_sem) ────────────────────
                              ↓
                     GlobalWorkspace (bottleneck)
                     n_slots=8, d_sem=384
                              ↓
                     slots: (B, 8, d_sem)
                    /      /      \      \
              [PFC]  [BG]  [Motor]  [DMN]  ...
```

The critical property: with $K$ input streams and $N_\text{slots} \ll K$ output slots, the GWS *must* find a compressed representation. Any two input modules that share information in the compressed space become correlated in $\Sigma$, contributing to Φ.

**Modern Hopfield dynamics:**

Initialise slots from learned queries: $S^{(0)} = \mathbf{Q}_\text{slot} \in \mathbb{R}^{N_\text{slots} \times d}$

Each Hopfield iteration (2 iterations unrolled for XLA):

$$S^{(t+1)} = \text{softmax}\!\left(\beta \cdot S^{(t)} C^\top\right) C$$

where $C \in \mathbb{R}^{K \times d}$ is the candidate matrix, $\beta = \text{softplus}(\log\beta_\text{param}) + 0.5$.

**Lateral competition** (prevents slot collapse):

$$S \leftarrow S \cdot \left(1 - 0.15 \cdot \bar{\rho}\right)$$

$$\bar{\rho}_{is} = \frac{1}{N_\text{slots}-1} \sum_{j \neq s} \cos(S_{ij}, S_{is})$$

Slots that are too similar to others are attenuated, ensuring diverse coverage of the input.

**Ignition phase transition** (Dehaene 2011):

$$\alpha_s = 0.15 + 0.85 \cdot \frac{1 + \tanh\!\left(6(\|S_s\| - \theta_s)\right)}{2}$$

$\theta_s$ is a learnable per-slot threshold (initialised to 0.8). Below threshold: $\alpha_s \approx 0.15$ (pre-ignition, sparse). Above threshold: $\alpha_s \approx 1.0$ (ignited, globally broadcast). The tanh provides a sharp phase transition — sharper than sigmoid at the same slope. Higher threshold prevents representational noise from triggering global broadcast.

NE temperature modulation: $S^{(0)} \leftarrow S^{(0)} \cdot \text{NE}$ — norepinephrine scales the initial slot activations, sharpening GWS selectivity under arousal.

**Tensor shapes (xl preset):**

| Tensor | Shape |
|---|---|
| `candidates` (input) | `(B, K, 384)` where K ≤ 16 |
| `slot_queries` | `(8, 384)` — learnable |
| `slots` (intermediate) | `(B, 8, 384)` |
| `log_beta` | `(1,)` |
| `slot_thresholds` | `(8,)` |
| `output_scale` | `(8,)` |
| `slots` (output) | `(B, 8, 384)` |

### 4.4 Thalamic Hub

*`neuroslm/modules/thalamus.py` — `Thalamus`*

The thalamus implements **re-entrant gating**: it receives the associative (pre-GWS) embedding and routes it to one of five specialised stream adapters before the GWS integration step, implementing the biological role of the pulvinar and mediodorsal nucleus as a content-aware signal router.

**Five streams:**

```python
STREAM_NAMES = ("language", "math", "reasoning", "spatial", "social")
```

Each stream is a 2-layer MLP with residual connection: `StreamAdapter(d_sem, hidden=d_sem)`.

**Routing equation:**

$$\text{probs} = \text{softmax}\!\left(\frac{W_r \mathbf{x}}{\tau}\right), \quad \tau = \frac{1}{0.5 + \text{NE}}$$

NE (norepinephrine) lowers temperature $\tau$, making routing sparser and more decisive under arousal. ACh (acetylcholine) provides a multiplicative boost to the top stream:

$$\text{out} = \sum_s \text{probs}_s \cdot (1 + 0.5 \cdot \text{ACh} \cdot \mathbb{1}[s = s^\star]) \cdot \text{StreamAdapter}_s(\mathbf{x})$$

**Lateral binding:** The thalamic routing probability vector `probs: (B, 5)` is logged as `routing` and fed into `ConsciousnessMetrics.update()` where its entropy measures the $\alpha$ oscillation proxy (high routing entropy → high alpha, broad idling; low entropy → focused, low alpha, high attention).

---

## 5. Wiring Diagram — NeuralOrchestrator Re-entrant Loops

*`neuroslm/intelligence/orchestrator.py` — `NeuralOrchestrator`*

```
╔════════════════════════════════════════════════════════════════╗
║  STAGE 0 — SENSORY                                             ║
║  ids (B,T) → TextSensoryCortex → sens (B, d_sem)              ║
║             → TopicClassifier → topic ∈ {math, reason, lang}  ║
║             → AssociationCortex → assoc (B, d_sem)            ║
╠════════════════════════════════════════════════════════════════╣
║  STAGE 1 — THALAMIC ROUTING          [HomeostaticGate]        ║
║  assoc → Thalamus(nt) → routed (B, d_sem), routing (B, 5)     ║
║  NE sharpens temp; ACh boosts top stream                       ║
╠════════════════════════════════════════════════════════════════╣
║  STAGE 2 — STATE MODELS                                        ║
║  routed → WorldModel(world_h) → z_world (B, d_sem)   ←─────╮  ║
║  [last_action, NT, thought] → SelfModel(self_h) → z_self    ║  ║
╠══════════════════════════════════════════════════════════════║═╣
║  STAGE 3 — SUBCORTICAL AFFECT                                ║  ║
║  z_world → Amygdala → emotional_valence, arousal, NT_release ║  ║
║  z_world → LateralHabenula → anti-reward aversion signal     ║  ║
║  → Insula → interoception, empathy, gut-feeling salience     ║  ║
╠══════════════════════════════════════════════════════════════╬═╣
║  STAGE 4 — QUALIA                    [HomeostaticGate]       ║  ║
║  z_world + emotional_valence + NT → QualiaState → qualia     ║  ║
╠══════════════════════════════════════════════════════════════║═╣
║  STAGE 5 — GLOBAL WORKSPACE (BOTTLENECK)                     ║  ║
║  candidates = stack[sens, routed, z_world, z_self,           ║  ║
║               thought, qualia, hippo_recall, ...]            ║  ║
║  → GWS(candidates, ne_temp) → slots (B, 8, d_sem)           ║  ║
║       ↑ Ignition gate | Hopfield iters=2 | Lateral comp.     ║  ║
║  → ConsciousnessMetrics → Φ, λ₁, gamma, theta, alpha        ║  ║
╠══════════════════════════════════════════════════════════════╬═╣
║  STAGE 6 — MEMORY SYSTEMS                                    ║  ║
║  slots → EntorhinalCortex → grid_context (B, d_sem)          ║  ║
║  slots → Hippocampus(nt) → slots_enriched, novelty, recalls  ║  ║
║  slots → HyperGraph → relational memory update               ║  ║
╠══════════════════════════════════════════════════════════════║═╣
║  STAGE 7 — COGNITIVE CONTROL                                 ║  ║
║  slots_enriched → PFC → selected (B, d_sem)                  ║  ║
║  [routed,z_world,z_self,selected] → ACC → conflict,effort    ║  ║
║  effort_steps > 0 → re-enter stages 6-8 ─────────────────────╯  ║
╠══════════════════════════════════════════════════════════════════╣
║  STAGE 8 — EXECUTIVE                                            ║
║  selected → BasalGanglia → action (B, d_sem), commit_ok        ║
║  action → ForwardModel → wp (B, d_sem), sp (B, d_sem)          ║
║  → Evaluator → value (B,)                                       ║
║  [thought, action, wp] → Cerebellum → error (B,)               ║
╠══════════════════════════════════════════════════════════════════╣
║  STAGE 9 — NARRATIVE / CONSCIOUSNESS                            ║
║  slots → DMN → dmn_query (B, d_sem)                            ║
║  → ThoughtTransformer → enhanced_thought (B, d_sem)            ║
║  → Claustrum → gestalt, salience, route_mask                   ║
║  floating_thought updated: blend(smooth, selected) or replace  ║
╠══════════════════════════════════════════════════════════════════╣
║  STAGE 10 — MOTOR OUTPUT                                        ║
║  action → MotorCortex → motor_lang_bias (B, d_hidden)          ║
║  h_lang + motor_lang_bias → lm_head → logits2 (B, T, vocab)   ║
╚══════════════════════════════════════════════════════════════════╝
                              ↕  (every edge)
                    HomeostaticGate: adapts gain online
                    to keep signal RMS ≈ target_magnitude=1.0
```

**Re-entrant loops:**

| Loop | Trigger | Stages revisited | Infancy-gated? |
|---|---|---|---|
| Effort loop | `ACC.effort_steps > 0` | 6 → 7 → 8 (up to `max_thinking_steps=12`) | No |
| Bowtie re-entry | Every forward pass | Stage 5 GWS broadcast → Stage 1 thalamus on next step | No |
| Vesicle tick | Every forward pass | Modulates all stages via dock signal | **Yes** — synthesis + migration + degrade skipped while `_infancy=True` |
| Floating thought EMA | Every tick | Feeds back into Stage 1 as prior context | No |
| Trophic update | Every train step | Adjusts projection gains across all stages | **Yes** — `trophic.update` + `bdnf_grow_all` gated on `not _infancy` |
| Homeostasis observe | Every train step | Updates NT bias/gain toward targets | **Yes** — gated on `_maturation_awakened` |
| Φ proxy + boundary detector | Every forward pass | Reads stage outputs, drives BDNF | **Yes** — replaced with $\Phi=0$, $\lambda_1=1$ placeholders during infancy |
| Consciousness metrics | Every forward pass | Populates phi/gamma/theta/alpha/coherence histories | **Yes** |
| Oscillation tracker | Every forward pass | Records 8 module activities | **Yes** |

See §6.4 for the full gating list and the awakening transition criteria.

---

## 6. Dynamical Biological Mechanics

### 6.1 Neuro-Vesicle Pool

*`neuroslm/neurochem/vesicles.py` — `VesiclePool`*

Vesicles are discrete semantic packets that implement **long-range, stateful neuromodulation** — the computational analogue of neuropeptide signalling. Unlike neurotransmitters (which are scalar levels), vesicles carry full $d_\text{sem}$-dimensional content vectors that modulate target modules additively.

**State buffers** (XLA-static shapes):

```python
v_contents  : (V=32, d_sem=384)   # semantic payload per vesicle
v_lifetimes : (V,)                # countdown; ≤ 0 → dead
v_positions : (V, n_modules)      # soft one-hot position
v_types     : (V,)  int32         # 0=default 1=math 2=reason 3=lang
```

#### Phase 1 — Emission (Synthesis)

Triggered when a surprise signal (world-model prediction error) exceeds a novelty threshold:

```python
# vesicles.py :: synthesize
def synthesize(surprise: Tensor,           # (B, d_sem)
               novelty_threshold: float = 0.3,
               source_module: int = 0):
    mean_surprise = surprise.detach().mean(0)   # (d_sem,)
    if mean_surprise.norm() < novelty_threshold:
        return
    content = synthesis_gate(mean_surprise)     # MLP: d_sem → d_sem
    idx = first_dead_slot() or write_ptr % V
    v_contents[idx]  = content
    v_lifetimes[idx] = lifetime       # default 16
    v_positions[idx] = one_hot(source_module, n_modules)
```

Typed vesicles (`synthesize_typed`) bypass the surprise threshold and carry explicit topic labels, allowing direct cortex gating.

#### Phase 2 — Migration (Stochastic Diffusion)

The learnable transition matrix $T \in \mathbb{R}^{M \times M}$ (row-stochastic via softmax) governs diffusion. Migration uses **Gumbel-argmax** to remain XLA-compilable (no `multinomial` which requires dynamic dispatch):

```python
# vesicles.py :: migrate
T            = softmax(log_T, dim=-1)         # (M, M)
dest_logits  = v_positions @ T                # (V, M)  soft destination
gumbel       = -log(-log(U.clamp(1e-6, 1-1e-6)))  # U ~ Uniform(0,1)
new_pos_idx  = (dest_logits + gumbel).argmax(dim=-1)  # (V,)
new_pos      = one_hot(new_pos_idx, M)        # (V, M)
v_positions  = where(active_mask, new_pos, v_positions)
```

Tensor shapes: `log_T: (n_modules, n_modules)`, `v_positions: (V, n_modules)`.

#### Phase 3 — Docking (Probabilistic Release)

Vesicles release their content to the module they currently occupy via cosine attention:

```python
# vesicles.py :: dock
# module_activations: (B, M, d_sem)
k     = normalize(dock_key(v_contents))      # (V, d_sem)
q_all = dock_query(module_activations)       # (B, M, d_sem)
q_ves = bmm(v_positions.unsqueeze(0), q_all) # (B, V, d_sem)  — soft position index
q_ves = normalize(q_ves)
scores = sigmoid((q_ves * k).sum(-1)) * active_mask  # (B, V)
delta  = mod_proj(v_contents)               # (V, d_sem)  — payload
contrib = scores.unsqueeze(-1) * delta      # (B, V, d_sem)
modulation = bmm(v_positions.T.unsqueeze(0), contrib)  # (B, M, d_sem)
```

Output `modulation: (B, M, d_sem)` is added to module activations before the next stage.

#### Phase 4 — Decay

```python
# vesicles.py :: degrade
v_lifetimes = v_lifetimes - decay          # subtract 1 per tick
dead_mask   = (v_lifetimes <= 0)
v_contents  = v_contents * (~dead_mask).float().unsqueeze(1)  # zero dead
```

**Expert gating:**

```python
# vesicles.py :: expert_gate
def expert_gate(type_idx: int) -> float:
    active     = (v_lifetimes > 0)
    type_match = active & (v_types == type_idx)
    return type_match.sum() / active.sum().clamp(min=1)
```

This concentration scalar (0–1) is passed as `vesicle_gate` to `MathCortex` and `ReasoningCortex` to scale their enrichment.

### 6.2 Trophic System — Structural Plasticity (BDNF/NGF)

*`neuroslm/neurochem/growth.py` — `TrophicSystem`*  
*`neuroslm/modules/language.py` — `NeuralGeometryAdapter.bdnf_grow`*

**Trophic levels** are scalar values $\tau_i \in [0,1]$ associated with each projection in the `ProjectionGraph`. They evolve per-tick (outside the autograd graph):

$$\Delta\tau_i = \underbrace{(\text{BDNF}_\Phi + \beta_\text{base})(0.1 + \bar{\rho}_i)}_{\text{Hebbian growth}} - \underbrace{(\text{NGF} + \delta_\text{decay} + 0.001(1 - \bar{\rho}_i))}_{\text{pruning}}$$

$$\tau_i \leftarrow \text{clamp}(\tau_i + \Delta\tau_i,\ 0,\ 1)$$

where $\bar{\rho}_i$ is the EMA of the co-activation product $a_\text{src} \cdot a_\text{dst}$ (Hebbian "fire together, wire together").

**Φ-gated BDNF:**

$$\text{BDNF}_\Phi = \text{BDNF} \cdot \left(1 + \phi_\text{boost} \cdot \Phi + \text{fiedler\_boost}\right)$$

$$\text{fiedler\_boost} = \max\!\left(0,\ 1 - \frac{\lambda_1}{0.3}\right) \times 2.0$$

High integrated information amplifies trophic factor release, locking the most conscious pathways. When the graph is nearly disconnected ($\lambda_1 < 0.3$), homeostatic BDNF compensates.

**Projection gain:**

$$g_i = \text{active}_i \cdot (0.2 + 1.6 \cdot \tau_i) \quad \in [0.0,\ 1.8]$$

Projections with $\tau_i < 0.05$ are pruned (`active_i = 0`); those recovering above $0.10$ are reactivated.

**NeuralGeometryAdapter kernel growth** (structural plasticity in weight space):

```python
# language.py :: NeuralGeometryAdapter.bdnf_grow
def bdnf_grow(bdnf: float, phi: float,
              growth_threshold: float = 1.5,
              delta_rank: int = 4,
              cooldown_steps: int = 200) -> bool:
    if cooldown > 0 or rank >= max_rank:
        return False
    bdnf_accum += bdnf * phi
    if bdnf_accum < growth_threshold:
        return False
    # Grow low-rank kernel:  kern_a: (d_hyper, rank) → (d_hyper, rank+Δ)
    new_a = zeros(d_hyper, delta_rank)
    kern_a = Parameter(cat([kern_a.data, new_a], dim=1))
    kern_b = Parameter(cat([kern_b.data, zeros(delta_rank, d_hyper)], dim=0))
    rank += delta_rank
    bdnf_accum = 0.0;  cooldown = cooldown_steps
    return True
```

**Adapter forward pass:**

```
# Tensor shapes (xl preset, d_hidden=512, d_hyper=1024, rank=k)
x     : (B, T, 512)
h     : (B, T, 512)   ← RMSNorm(x)
z     : (B, T, 1024)  ← up(h)           linear
k_mat : (B, T, 1024)  ← z @ kern_a @ kern_b   (1024,k)@(k,1024)
g     : (B, T, 1024)  ← sigmoid(gate(z))
z_new : (B, T, 1024)  ← silu(k_mat) * g
out   : (B, T, 512)   ← down(z_new)
return x + out                            residual
```

As Φ rises and BDNF accumulates, `rank` $k$ increases from its initial value (default `max(8, d_hyper//8)`) up to `max_rank = d_hyper//2`, progressively allowing denser inter-neuron connectivity in the hyper-space.

### 6.3 Hebbian Fast Weights (HFW)

*`neuroslm/modules/fast_weight.py` — `FastWeightLayer`*

HFW implements **dual-timescale learning**: slow weights (SGD/Adafactor) encode long-term knowledge; fast weights encode within-context episodic binding without any gradient step.

**Write rule** (outer-product accumulation with decay):

$$W_\text{fast}^{(t)} = \lambda \cdot W_\text{fast}^{(t-1)} + \eta_t \cdot g_t \otimes (v_t \otimes k_t)$$

- $\lambda = 0.95$: exponential decay (recent associations dominate)  
- $\eta_t = \eta_0 \cdot \text{softplus}(\text{eta\_mod}(\text{context})) \in \mathbb{R}^{n_\text{heads}}$: context-dependent plasticity rate  
- $g_t = \text{sigmoid}(W_g \mathbf{x}_t) \in (0,1)^{d_h}$: write gate (LSTM-like forgetting)

**Read rule:**

$$\mathbf{y}_t = \text{LayerNorm}(W_\text{fast} \mathbf{q}_t)$$

**PyTorch XLA / JAX pseudocode:**

```python
# fast_weight.py :: forward  (simplified)
# x: (B, T, D),  W_fast: (B, H, Dh, Dh)
k = k_proj(x).view(B, T, H, Dh).permute(0,2,1,3)   # (B,H,T,Dh)
v = v_proj(x).view(B, T, H, Dh).permute(0,2,1,3)
q = q_proj(x).view(B, T, H, Dh).permute(0,2,1,3)
g = sigmoid(g_proj(x).view(B, T, H, Dh).permute(0,2,1,3))
eta = base_eta * (eta_mod(context) + 1e-6)           # (B, H)

out_heads = []
for t in range(T):                                   # unrolled in XLA
    read    = einsum("bhij,bhj->bhi", W_fast, q[:,t])  # (B,H,Dh)
    out_heads.append(layer_norm(read, [Dh]))
    # outer product write: (B,H,Dh,1)×(B,H,1,Dh)
    outer   = v[:,t].unsqueeze(-1) * k[:,t].unsqueeze(-2)   # (B,H,Dh,Dh)
    gate_m  = g[:,t].unsqueeze(-1) * g[:,t].unsqueeze(-2)   # (B,H,Dh,Dh)
    W_fast  = decay * W_fast + eta.view(B,H,1,1) * gate_m * outer

out = out_proj(stack(out_heads, dim=2).reshape(B, T, D))
return layer_norm(x + out), W_fast
```

**Tensor shapes (xl, H=4, Dh=96):**

| Tensor | Shape |
|---|---|
| `W_fast` | `(B, 4, 96, 96)` |
| `k, v, q, g` | `(B, 4, T, 96)` |
| `eta` | `(B, 4)` |
| `outer` | `(B, 4, 96, 96)` |

### 6.4 Topological Maturation — Infancy → Awakening

*`neuroslm/train.py` (per-step scheduler) and `neuroslm/brain.py` (forward-pass gates: `_aux_w_scale`, `_infancy`)*

Training proceeds in two distinct chemical regimes. The bio pipeline is structurally identical in both — only the **strength** of auxiliary objectives and the **activation** of decorative side-effects change.

#### Infancy (step $< 5000$)

The model is at near-random initialisation. Every non-LM signal (world prediction, motor decisions, Φ proxy, vesicle docking, NT homeostasis targets, oscillation spectra, consciousness metrics, hippocampal recall) is drawn from a noisy substrate where downstream targets carry no usable signal. Letting these systems run their full course at this stage corrupts the LM gradient and inflates wall-clock 3–4× per forward pass.

During infancy the scheduler sets:

```
brain._aux_w_scale  = 0.001     # uniform multiplier on every non-LM loss
brain._infancy      = True      # gates decorative no-grad operations
gws.slot_thresholds = 1.2       # raise ignition threshold to suppress saturation
```

**Gated off in `brain.forward_lm`** (all guarded by `if not _in`):

| Operation | Why gated |
|---|---|
| `orchestrator.record_stage_output` (sens / gws / pfc) | Φ proxy + boundary detector consume these — both also gated |
| `orchestrator.compute_phi_proxy` | Gaussian-MI bipartition over diagonal Σ ≈ 0 — replaced with placeholder $\Phi = 0$ |
| `orchestrator.phi_tensor` | Differentiable Φ — its gradient contribution is $\alpha \cdot w_\phi \approx 2 \times 10^{-5}$ during infancy anyway |
| `orchestrator.route` (cerebellum / entorhinal / claustrum) | Expensive subnetwork pass for id_drift / neural_calm metrics |
| `boundary_detector.observe` | Normalised-Laplacian eigensystem on random-init covariance |
| `consciousness.update` | Phi/gamma/theta/alpha/coherence histories |
| `vesicle_pool.synthesize_typed / migrate / degrade` | Topic classifier is at random init — synthesised vesicles are uniform noise |
| `hippo.store` | Stored embeddings are random; would be flushed on consolidation anyway |
| `oscillation_tracker.record × 8 + tick` | No oscillatory pattern to track |
| `_maybe_store_insight` | Surprise/comprehension/valence are random-init noise |

**Gated off in `train.py`:**

| Operation | Why gated |
|---|---|
| `record_episode` + `tok.decode(...).cpu().numpy()` | Tokenizer round-trip is pure CPU overhead during infancy |
| `tag_memory` | Pairs with `record_episode` |
| `consolidate_memory` + `update_narratives` (500-step) | Operates on infancy-skipped episodic buffer |
| `homeostasis.observe` | Its target NT mean/std bands are calibrated for a trained network — running it against random-init NT drives 5HT/GABA to ceiling |

**Preserved during infancy** (load-bearing for the LM forward graph):

- All LM cortex layers, the motor-conditioned head, all NeuralGeometryAdapters
- `orchestrator.set_gws_broadcast` + `orchestrator.update_reentry` — the bowtie loop's within-pass and next-step re-entry signals
- Thalamic routing, Sensory + Association cortex, GWS Hopfield iteration, Hippocampal recall (used by PFC)
- All NT releases + `transmitters.step()` (still drive the log values, just not corrected by homeostasis)
- Active dendrite, neurogenesis, dynamic-routing MoE, math/reasoning cortices (their `novel_aux_loss / moe_aux / dag_loss` contributions are scaled by $\alpha = 0.001$ but they still modify `slots/selected` which feed back into motor → logits → LM loss)

#### Awakening Transition

Two conditions must be met simultaneously:

1. **`step ≥ 5000`** (infancy minimum duration)
2. **`lm_loss < 7.5`** (raw LM has stabilised below random; for vocab≈50k, ln(50k)≈10.84 is the random ceiling)

When both hold, the scheduler sets `_maturation_awakened = True` permanently and:

- `_infancy = False` — every gated operation above resumes
- `homeostasis.observe` begins correcting NT bias/gain toward targets
- Ramping of $\alpha$ begins

#### Awakening Ramp

After awakening, $\alpha(t)$ ramps from 0.001 to 1.0 over the remaining training budget, conditioned on **sustained** below-threshold loss:

```
_loss_below_threshold_count   = number of consecutive steps with lm_loss < 7.5  
ramp_started ⟺ _loss_below_threshold_count ≥ 100  (sustained-stability window)

if ramp_started:
    steps_ramped   = _loss_below_threshold_count - 100
    max_ramp_steps = args.steps - step
    α(t)           = min(1.0, steps_ramped / max(1, max_ramp_steps))
else:
    α(t)           = 0.0       # still in infancy-equivalent — no aux load
```

Once $\alpha$ reaches 1.0, every auxiliary loss applies at its config-default weight and trophic/BDNF growth begins shaping the projection graph based on real Φ and Fiedler signals.

#### Why this matters

Without infancy gating, two failure modes occur:

1. **Aux gradient noise dominates LM** — world/motor/Φ losses produce structured gradients against a random target, fighting the LM's progress toward language. Loss stalls at the random-init ceiling.
2. **Trophic / homeostasis target the wrong configuration** — NT homeostasis pulls levels toward `target_mean=0.3` regardless of whether the network is learning, driving 5HT and GABA to ceiling and pinning ignition at saturation. Trophic growth + BDNF rank increase build up structure based on phi/fiedler measurements of random covariance.

The infancy/awakening split implements **"linguistic first" convergence**: let the LM cortex find token-level structure first, then let the bowtie + bio modules layer integration on top of a substrate that already carries signal.

---

## 7. Optimization & Infrastructure

### 7.1 Adaptive Compute

#### Mixture of Depths (MoD)

*`neuroslm/modules/mixture_of_depths.py` — `MoDRouter`*

Each `MoDBlock` routes only the top-$C$ "hard" tokens through the transformer sublayer; the rest skip via residual:

$$C = \max\!\left(1,\ \lfloor T \cdot \rho \rfloor\right), \quad \rho = \rho_0 \cdot \left(0.5 + \sigma(W_\text{nt} \cdot \text{NT})\right)$$

$\rho_0$ is the base capacity ratio (`mod_capacity=0.8` for xl). The NT modulation adjusts capacity dynamically: high ACh → higher capacity (more tokens processed in full), high NE → lower (focused on hardest tokens only).

Router: 2-layer MLP with zero-init (all tokens start with equal score, routing emerges during training).

**Auxiliary loss** (load balancing):

$$\mathcal{L}_\text{MoD} = \frac{1}{T}\sum_t s_t \cdot \mathbb{1}[\text{token } t \text{ selected}]$$

#### CALM Early Exit

*`neuroslm/modules/mixture_of_depths.py` — `CALMHead`*

Per-token confidence is estimated at each transformer layer. A token exits at the earliest layer where its confidence exceeds the layer-specific threshold:

$$\theta_l = \theta_\text{base} \cdot \exp\!\left(-\delta \cdot \frac{l}{L-1}\right)$$

with $\theta_\text{base}=0.9$, $\delta=2.0$ (shallow layers almost never exit; deep layers have lower threshold, so uncertain tokens still get a chance to exit).

NE arousal override: when NE > 0.5, all CALM thresholds are multiplied by $(1 + \text{NE})$, forcing full-depth processing under stress (the model "pays attention").

**Combined compute savings** (MoD + CALM in xl preset): empirically 30–50% of FLOPs at inference without accuracy loss on easy prefixes.

### 7.2 Neurotransmitter System

*`neuroslm/neurochem/transmitters.py` — `TransmitterSystem`*

Seven NTs with Euler-integrated dynamics (per tick). Channel order `NT_NAMES = (DA, NE, 5HT, ACh, eCB, Glu, GABA)` is canonical across all modules:

| NT | $\tau_\text{decay}$ | Baseline | Role |
|---|---|---|---|
| DA | 0.80 | 0.10 | Reward, salience, routing sharpness |
| NE | 0.70 | 0.15 | Arousal, attention, CALM threshold |
| 5HT | 0.95 | 0.30 | Mood, patience, long-horizon value |
| ACh | 0.75 | 0.20 | Plasticity, MoD capacity, HFW η |
| eCB | 0.60 | 0.05 | Retrograde suppression (fast) |
| Glu | 0.50 | 0.40 | Excitation |
| GABA | 0.90 | 0.10 | Homeostatic inhibition (slow decay toward target 0.1) |

**Per-tick dynamics:**

$$\text{level}_i(t+1) = \tau_i \cdot \text{level}_i(t) + \big(b_i + \Delta b_i\big) \cdot (1 - \tau_i)$$

where $b_i$ is the canonical baseline above and $\Delta b_i$ is the learned homeostatic bias (clamped to $[-0.5, 0.5]$). Levels are clamped to $[0, 1]$; `release(name, amount)` is vesicle-limited via `vesicles[name] / release_cost`.

**Homeostasis loop** (`neurochem/homeostasis.py — Homeostasis.observe`, **gated on `_maturation_awakened`** — see §6.4):

$$\Delta b_i \mathrel{+}= \eta \cdot (b^\star - \langle\text{level}_i\rangle), \quad \Delta \text{gain}_i \mathrel{+}= \eta \cdot (\sigma^\star - \sqrt{\text{Var}[\text{level}_i]})$$

with $\eta = 5 \times 10^{-3}$, target mean $b^\star = 0.3$, target std $\sigma^\star = 0.15$. A gnorm-driven safety branch boosts GABA bias when `grad_norm > 5.0` (limits excitatory tone) and Glu bias when `grad_norm < 0.1` (combats vanishing).

The `TransmitterSystem` returns `float32` tensors. **All modules that receive NT tensors must cast them to the model dtype** before arithmetic — this is enforced at module boundaries (Thalamus, GWS, ReceptorBank.modulate). LayerNorm / MultiheadAttention / Linear fast paths are also patched in `train.py` to upcast non-matching inputs to weight dtype, eliminating bf16/fp32 mismatch errors on TPU and CUDA Ampere+.

### 7.3 TPU/XLA Backend

**bfloat16 precision** is the default for all model parameters and activations. Rationale: TPU hardware has native bfloat16 support with the same throughput as float32 but half the memory. All `torch.zeros` initialised for hidden states, fallback tensors, and zero-initialised outputs must carry `dtype=w_dtype` (inferred from the module's weight tensor) to avoid float32/bfloat16 dtype mismatch errors.

**bf16 safety patches** (installed at `train.py` import time, applied to every `nn` instance):

| Op | Patch behaviour |
|---|---|
| `LayerNorm.forward` | If input or weight is bf16, upcasts both to fp32, runs `F.layer_norm`, casts back |
| `MultiheadAttention.forward` | Casts query/key/value to `in_proj_weight.dtype` before the fast path |
| `Linear.forward` | Casts input to `weight.dtype` if mismatched |
| `torch.fft.rfft` (in `oscillations.py`) | Input upcast to fp32 — bf16 not supported by the FFT kernel |

All patches are no-ops in pure-fp32 training and add only a single conditional dtype check per call in bf16.

**XLA constraints** honoured throughout:

| Constraint | Implementation |
|---|---|
| No dynamic shapes | All loops unrolled at `__init__` time (Hopfield iters, CALM thresholds, fast-weight `T` loop) |
| No `torch.multinomial` | Gumbel-argmax in vesicle migration |
| No `tensor.nonzero()` in hot path | Masked arithmetic throughout |
| Static `top-k` | `scores.topk(C)` where C is a Python int |

**Gradient checkpointing:** enabled for xl+ presets (`gradient_checkpointing=True`). Applied to language blocks and GWS via `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`. On XLA devices the wrapper is skipped (XLA rematerialises automatically).

### 7.4 Optimizer Selection

Two optimizer paths are wired via `--optimizer {adafactor,adamw}`:

**Adafactor** (default, TPU-native, `transformers.optimization.Adafactor`):

```python
Adafactor(model_params,
          lr=None,
          scale_parameter=True,
          relative_step=True,
          warmup_init=True,
          weight_decay=cfg.weight_decay)
```

Factor-wise second moment estimation — ~4–8× less optimizer memory than AdamW, critical for fitting xl-sized models on a single TPU core. With `warmup_init=True` and `relative_step=True`, the effective rel-step follows $\min(10^{-6} \cdot \text{step},\ 1/\sqrt{\text{step}})$, multiplied by per-parameter RMS — designed for multi-day TPU runs where the schedule has thousands of warmup steps to traverse.

**AdamW** (`--optimizer adamw`, recommended for short ablations and CUDA debugging):

```python
AdamW(model_params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
```

with cosine warmup+decay schedule from `train.py :: cosine_lr` over `cfg.warmup_steps` (default 200) → `total_steps`.

**Choose AdamW for any run shorter than ~10K steps.** Adafactor's `warmup_init` schedule keeps the effective LR near $10^{-4}$ for the first ~1000 steps; a 1000-step xl ablation never reaches a learning LR under Adafactor.

The cosine LR override (`pg["lr"] = cosine_lr(...)`) only runs when `args.optimizer != "adafactor"`. With Adafactor, the optimizer manages its own LR internally and the train-loop override is bypassed.

**Per-step gradient norm** is computed by `clip_grad_norm_(parameters, cfg.grad_clip)` (default `grad_clip=1.0`) and logged as the running average `gnorm` in every step line. Healthy range during learning is roughly 0.5–5.0; a sustained value pinned at 1.0 means the clip is dominating.

**Query-Key Normalisation:** RMSNorm applied to Q and K projections in attention layers (`common.py :: TransformerBlock`) stabilises training at bfloat16 precision by preventing attention logit overflow.

---

## 8. Intelligence & Integration Metrics

### 8.1 The Φ Proxy — Complete Algorithm

*`neuroslm/modules/consciousness.py` — `ConsciousnessMetrics._compute_phi_mip`*

```
Input:  module_outputs  dict[str → Tensor]  — one tensor per active module
Output: phi             float               — Φ ∈ [0, 10]

Step 1. Collect module vectors
  for each module k (up to n=8):
    z_k = module_outputs[k].mean(dim=0).detach().float().flatten()[:256]

Step 2. Build covariance matrix
  M   = stack(z_k for k in 0..n)       # (n, d ≤ 256)
  M   = M - M.mean(dim=0)              # mean-centre
  Σ   = (M @ M.T) / (d - 1)           # (n, n)

Step 3a. [n ≤ 8] Enumerate bipartitions
  ld_full = slogdet(Σ + 1e-6 I)[1]
  phi = +inf
  for mask in 1 .. 2^(n-1):
    A, B = partition by mask bits
    ld_A = slogdet(Σ[A,A] + 1e-6 I_A)[1]
    ld_B = slogdet(Σ[B,B] + 1e-6 I_B)[1]
    MI   = 0.5 * (ld_A + ld_B - ld_full)
    phi  = min(phi, max(0, MI))

Step 3b. [n > 8] Spectral bisection
  W = |Σ| / sqrt(diag(Σ) ⊗ diag(Σ))  # normalised similarity
  L = I - D^{-1/2} W D^{-1/2}         # normalised Laplacian
  λ, V = eigh(L)                        # sorted eigenvalues
  fiedler = V[:, 1]                     # second eigenvector
  A = (fiedler ≥ 0),  B = (fiedler < 0)
  phi = 0.5 * (ld_A + ld_B - ld_full)

Step 4. Clamp and return
  return clamp(phi, 0, 10)
```

**Other consciousness observables computed each tick:**

| Observable | Formula | Biological analogy |
|---|---|---|
| Gamma | mean cosine similarity of GWS slot pairs | Binding oscillations (40 Hz) |
| Theta | mean novelty across batch | Hippocampal memory retrieval |
| Alpha | routing entropy / max entropy | Cortical idling / suppression |
| Coherence | cosine alignment of module outputs with GWS mean | Phase synchronisation |
| Ignition | fraction of modules with $\|\mathbf{z}\| > 0.6$ | Global broadcast threshold |
| Metacognition | sigmoid(‖floating\_thought‖ − 1) | Self-awareness proxy |

### 8.2 Comprehension Index

*`neuroslm/memory/comprehension_gate.py` — `ComprehensionGate`*

The comprehension gate decides whether an observation merits long-term storage. It combines three orthogonal quality signals:

$$\text{score} = \underbrace{\min\!\left(1, \frac{\text{NLL}}{6}\right)}_\text{surprise} \times \underbrace{\cos(\mathbf{z}_\text{obs}, \mathbf{z}_\text{pred})}_\text{comprehension} \times \underbrace{1 - \max_j \cos(\mathbf{z}_\text{obs}, \mathbf{c}_j)}_\text{novelty}$$

- **Surprise**: raw NLL normalised to $[0,1]$ by dividing by 6 (≈ 2-bit surprise)  
- **Comprehension**: cosine similarity between the observation embedding and the model's predicted embedding. High comprehension = the model can integrate this into an existing schema.  
- **Novelty**: 1 minus the maximum cosine similarity to existing consolidated memory nodes (last 256 checked). High novelty = concept not already stored.

$$\text{write} = \text{score} > \theta, \quad \theta \leftarrow \theta \cdot \begin{cases} 1.005 & \text{write rate} > 1.2 \cdot r_\text{target} \\ 0.995 & \text{write rate} < 0.8 \cdot r_\text{target} \end{cases}$$

Target write rate $r_\text{target} = 0.10$ (10% of observations stored). The adaptive threshold $\theta$ is bounded to $[10^{-4}, 0.5]$.

**This filter is the operational definition of a learning insight**: observations must be simultaneously *new*, *surprising*, and *understandable* to be written into the relational memory graph. Random noise (high surprise, zero comprehension) is rejected; known facts (zero novelty) are rejected; incomprehensible signals (zero comprehension) are rejected.

---

## 9. Parameter Presets

All presets share the same module topology (`neural_topology="full"`). Differences are purely dimensional — no modules are removed.

| Preset | Approx params | `d_sem` | `d_hidden` | `lang_layers` | `lang_ctx` | `gws_slots` | `hippo_capacity` | Hardware target |
|---|---|---|---|---|---|---|---|---|
| `tiny` | ~5 M | 128 | 192 | 2 | 256 | 8 | 4 096 | CPU smoke-test |
| `small` (default) | ~15 M | 256 | 384 | 4 | 512 | 8 | 4 096 | CPU (hours) |
| `medium` | ~80 M | 512 | 768 | 8 | 1 024 | 8 | 4 096 | T4 single-GPU |
| `large` | ~100 M | 256 | 384 | 8 | 1 024 | 12 | 8 192 | T4 16 GB |
| `xl` | ~240 M (live: 228 M) | 384 | 512 | 12 | 2 048 | 8 | 4 096 | A100 40 GB |
| `xxl` | ~10 B | 2 048 | 4 096 | 32 | 4 096 | 24 | 32 768 | 4–8 × A100 |

**Default `BrainConfig` values** (apply to every preset unless overridden):

```python
lr            = 3e-4
weight_decay  = 0.01
warmup_steps  = 200
grad_clip     = 1.0
```

**xl-specific flags** (the primary research preset, overrides defaults):

```python
lang_heads          = 8
lang_kv_heads       = None        # full MHA (no GQA)
pfc_layers          = 3
pfc_heads           = 8
dmn_layers          = 3
world_layers        = 2
self_layers         = 1
forward_layers      = 2
hippo_topk          = 6
hippo_sparse_k      = 64
max_thinking_steps  = 12
hebbian_rank        = 4
mod_capacity        = 0.8
gradient_checkpointing = True
lr                  = 2e-4      # overrides default 3e-4
weight_decay        = 0.1       # overrides default 0.01
warmup_steps        = 800       # overrides default 200
baseline_lang_layers = 56       # vanilla baseline parity (~212 M at d_hidden=512)
```

**xxl-specific additions:**

```python
use_moe             = True
moe_experts         = 16
moe_top_k           = 2
use_adaptive_compute = True
max_ponder_steps    = 12
enable_rssm         = True    # Recurrent State Space Model world model
enable_active_inference = True
enable_tom          = True    # Theory of Mind
```

### Per-Module Enable Flags

Beyond size, `BrainConfig` exposes ~30 boolean flags that selectively bypass brain areas. Disabled modules return neutral passthrough outputs — useful for ablation studies without changing tensor shapes:

```python
# Core (all True by default)
enable_hippocampus, enable_pfc, enable_basal_ganglia, enable_dmn,
enable_thalamus, enable_cerebellum, enable_cortical_sheet, enable_entorhinal,
enable_claustrum, enable_gws, enable_world_model, enable_self_model,
enable_critic, enable_neural_geometry, enable_qualia, enable_thought_transformer,
enable_oscillations, enable_narrative, enable_mesolimbic

# Emotional / subcortical (True by default)
enable_amygdala, enable_acc, enable_insula, enable_lateral_habenula

# Memory + neurochem (True by default)
enable_hypergraph, enable_entity_store, enable_vesicles

# Novel cognitive modules (False by default; opt-in via xxl preset)
enable_tom, enable_rssm, enable_active_inference

# Novel ML objectives (False by default)
enable_cpc                   # contrastive predictive coding
enable_phi_objective = True  # differentiable Φ loss (the one exception)
```

### Training Command Examples

**Short ablation (1000 steps), AdamW** — recommended for any experiment under ~10K steps:

```bash
python -m neuroslm.train --preset xl --steps 1000 \
       --batch_size 1 --grad_accum 16 \
       --optimizer adamw \
       --mode mix --chat_ratio 0.6 \
       --ckpt_dir /content/checkpoints --device cuda
```

Effective batch = `batch_size × grad_accum × ctx` = 1 × 16 × 2048 = 32K tokens/step. The `--grad_accum 16` is sized for the xl preset; smaller presets can use lower values.

**Long training (100K+ steps), Adafactor on TPU** — the default path:

```bash
python -m neuroslm.train --preset xl --steps 100000 \
       --batch_size 4 --grad_accum 4 \
       --mode mix --chat_ratio 0.6 \
       --ckpt_dir ./checkpoints --device xla \
       --resume latest --overwrite_ckpt
```

**Baseline ablation** — adds `--baseline` flag; trains a param-matched vanilla transformer (no bio modules, `baseline_lang_layers=56` in xl) for direct comparison:

```bash
python -m neuroslm.train --preset xl --steps 1000 \
       --batch_size 1 --grad_accum 16 --optimizer adamw \
       --baseline \
       --ckpt_dir /content/checkpoints_baseline --device cuda
```

---

*Last updated: 2026-05-12. Source of truth: `neuroslm/` on branch `tpu`.*
