# NeuroSLM — Architecture Specification

> **Reproduction-ready technical reference.**
> Every section maps directly to source files in `neuroslm/`. Tensor shapes, pseudocode, and mathematical formulas reflect the live implementation.
>
> **As of 2026-05-18, the `xl` preset uses the SRC-TEH topology (§0).** Older sections describe the legacy bowtie-only flow that is preserved as the default for `tiny`/`small`/`medium`/`large` and as the inference path when `cfg.enable_src_teh=False`.

---

## Table of Contents

0. [SRC-TEH Topology (the new xl)](#0-src-teh-topology-the-new-xl)
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
10. [BRIAN — Narrative + Causal Memory Stack](#10-brian--narrative--causal-memory-stack)
    - 10.1 Contextual Sheaf F & H¹ Contradiction Detection
    - 10.2 Actual-Causation Head (IIT 4.0)
    - 10.3 ReasoningCortex Action → Reaction Predictor
    - 10.4 Narrative Engine (JSON stories) + EntityNarrative trust
    - 10.5 PersonalityVector → NT-baseline coupling
    - 10.6 NEMORI Predictive-Forgetting Gate
    - 10.7 Sleep-Cycle CLS (PC distillation + trophic renormalisation)
    - 10.8 DNC Temporal-Link Matrix L
    - 10.9 κ_cause Vesicles
11. [Cognitive Closure — Survival-Gated Action Loop](#11-cognitive-closure--survival-gated-action-loop)
    - 11.1 GridWorld Environment (10×10 SHRDLU)
    - 11.2 Sensory VAE Front-End
    - 11.3 Latent Qualia Manifold Q & Homeostatic Warp
    - 11.4 κ_neg Aversive Vesicles
    - 11.5 Basal Ganglia VQH + Expert Gating
    - 11.6 NAcc Reward-Prediction-Error
    - 11.7 SurvivalCausalHead (action → ΔS_{t+1})
    - 11.8 Homeostasis.step Tick

---

## 0. SRC-TEH Topology (the new xl)

**Shared Reading Cortex + Token-Level Expert Heads** — the post-RFC topology
adopted by the `xl` preset. Replaces the legacy "trunk → mean-pool → bowtie
→ tiny vesicle-gated cortices that touch only the last logit" pipeline with
a cleaner two-tier design where every expert receives token-level gradient
on every step. Detailed motivation in [`docs/RFC.md`](RFC.md).

### 0.1 Diagram

```
              ┌─────────────────────────────────────────────────────────┐
              │ TIER 1 — Shared Reading Cortex     (≈60% of params)    │
              │ neuroslm/modules/language.py :: LanguageCortex          │
              │ 10 deep blocks @ d_hidden=576 (Standard/DiffAttn/MoD)   │
              │ + NeuralGeometryAdapter after every block               │
              │ + MemoryCrossAttention on the LAST 2 blocks  ─────┐    │
              │   (RETRO-style K/V rows from bowtie EMA cache)    │    │
              │ produces:                                          │    │
              │   h ∈ (B, T, 576)                                  │    │
              │   tap_sem ∈ (B, d_sem) at layer 5 (mid-trunk tap) │    │
              │   sem     ∈ (B, d_sem) via AttentionPool          │    │
              └────────────────────────┬──────────────────────────┴────┘
                                       │
                                       ▼
              ┌─────────────────────────────────────────────────────────┐
              │ TIER 2 — Expert-Choice Routing  (capacity = T·1.5/3)   │
              │ neuroslm/modules/expert_router.py                        │
              │ each expert PULLS top-C tokens by affinity              │
              │                                                          │
              │   ┌─────────────┐ ┌────────────┐ ┌──────────────┐      │
              │   │LanguageExpert│ │MathCortex  │ │ReasoningCortex│     │
              │   │ 3 trunk blks │ │ 3 blocks + │ │ 3 blocks +    │     │
              │   │ @ d_hidden   │ │ fact-mem   │ │ Hopfield bank │     │
              │   │  zero-init   │ │ xattn      │ │ xattn         │     │
              │   └──────┬───────┘ └─────┬──────┘ └───────┬───────┘     │
              │          │ scatter-add residuals back into h            │
              │          ▼               ▼                ▼              │
              │       h_enriched ∈ (B, T, d_hidden)                     │
              └─────────────────────────┬───────────────────────────────┘
                                        │
                          re-norm + lm_head → logits (B,T,V)
                                        │
                                        ▼
              ┌─────────────────────────────────────────────────────────┐
              │ LAZY BOWTIE       (run every `bowtie_period`=4 steps)  │
              │ tap_sem  → Sensory/Assoc/Thalamus/World/Self/GWS/…     │
              │ slots EMA cached in `_bowtie_ema_slots`  ──────────────┘
              │   (feeds MemoryCrossAttention on next step's trunk)
              └─────────────────────────────────────────────────────────┘

              ┌─────────────────────────────────────────────────────────┐
              │ LATENT PROGRAM BUS    (across-step reasoning state)    │
              │ neuroslm/intelligence/latent_program_bus.py             │
              │ trunk writes:  bus ← EMA( write_head(h_pool) )         │
              │ trunk reads :  thought ← thought + bus_to_sem(bus)     │
              │  (zero-init bus_to_sem → identity at step 0)           │
              └─────────────────────────────────────────────────────────┘
```

### 0.2 Five mechanisms

| # | Mechanism | File / class | Purpose |
|---|---|---|---|
| A | **Expert-Choice Routing** | `modules/expert_router.py :: ExpertChoiceRouter` | Each expert pulls top-C tokens (C = T·cf/n_experts) by affinity. No dropped tokens; no load-balancing aux loss; XLA-static `topk`. |
| B | **AttentionPool** | `modules/language.py :: AttentionPool` | Replaces `h.mean(dim=1)` with a 1-token learnable cross-attention. Preserves positional structure → Φ proxy converges 2–3× faster. |
| C | **RETRO-style K/V injection** | `modules/common.py :: MemoryCrossAttention` (zero-init out) | Last 2 trunk blocks accept extra K/V rows: pooled bowtie EMA + (TODO) top-N consolidated memory entries. Identity at init. |
| D | **Mid-trunk tap** | `LanguageCortex.forward(return_tap=True)` | Layer 5 of 10 emits an AttentionPool summary that feeds the bowtie. Late layers receive bowtie back via memory xattn — the bowtie is now *used* by the trunk, not just biasing the last logit. |
| E | **Latent Program Bus** | `intelligence/latent_program_bus.py :: LatentProgramBus` | 16-dim across-step continuous "thought program". Trunk emits at end of pass; reads back at start of next pass via zero-init `bus_to_sem`. Replaces the routing role of vesicles. |

### 0.3 Lazy Bowtie

The full bowtie (Stages 0-10) is expensive. Under SRC-TEH it still runs
on every step for correctness (orchestrator state + re-entry loops are
load-bearing), but the **legacy d_sem expert enrichment** (MathCortex
and ReasoningCortex called on `_slots_mean`) is **skipped** — the
token-level experts already enriched the trunk at d_hidden. Each step's
post-GWS slots are written into `Brain._bowtie_ema_slots` via EMA
(`bowtie_ema_alpha`=0.4); the next forward pass uses that buffer as
RETRO-style K/V rows on the trunk's last two layers.

`Brain._bowtie_step` is a long-counter and `cfg.bowtie_period` controls
the cadence — currently informational, plumbed into the forward pass so
future revisions can fully short-circuit DMN/Hippo/PFC on off-steps.

### 0.4 Vesicle role transition

Vesicles previously *gated* which expert cortex fired (via `expert_gate`).
With SRC-TEH this role is taken over by the Expert-Choice Router and the
Latent Program Bus. Vesicles now serve **only** their plasticity and
homeostatic roles, per spec §6.1 and §10.9:

- **κ_cause vesicles** still stabilise causal-rule attractors after
  high-α actual-causation detection.
- **κ_neg vesicles** still drive escape/foraging attractor schemas under
  high aversive pressure (§11.4).
- **BDNF release** still scales with vesicle concentrations at the trophic
  step (§6.2).

### 0.5 Softened Trophic Gate

To prevent the historic "n_active: 0" graph collapse where a random-init
projection set self-prunes before learning, `TrophicSystem.update` now
accepts `maturity` and `prune_mat_threshold` arguments
(`neurochem/growth.py`). Pruning (`active[i]=0`) is suppressed while
`MAT < prune_mat_threshold` (default 0.3). Trophic levels still drift so
the learned plasticity state accumulates; only structural deactivation
is held off.

### 0.6 xl preset re-budget

| Component | Legacy xl | SRC-TEH xl |
|---|---|---|
| `d_hidden` | 512 | **576** |
| `lang_layers` (trunk depth) | 12 | **10** |
| LanguageCortex (trunk) | ~84M | ~100M |
| NeuralGeometryAdapter × N | ~48M | ~25M (10 sites, lower rank) |
| MathCortex | 0.6M | **~18M** (3 trunk blocks + fact xattn @ d_hidden) |
| ReasoningCortex | 0.7M | **~18M** (3 trunk blocks + attractor xattn @ d_hidden) |
| LanguageExpert (new) | – | **~18M** |
| ExpertChoiceRouter | – | 0.05M |
| MemoryCrossAttention (2×) | – | 1.5M |
| AttentionPool (sem + tap) | – | 1.5M |
| LatentProgramBus | – | 0.15M |
| Bowtie + bio modules | ~70M | ~30M (legacy d_sem expert path now skipped) |
| **Total** | ~228M | **~240M** |

### 0.7 Config flags (defaults are OFF; `xl()` flips them ON)

```python
enable_src_teh:        bool  = False     # master flag for §0
enable_memory_xattn:   bool  = False     # K/V injection on last N trunk blocks
n_memory_xattn_layers: int   = 2
n_memory_entries:      int   = 64        # cap on retrieved consolidated entries
mid_trunk_tap_layer:   int   = 0         # 0 = auto (= lang_layers // 2)
n_token_experts:       int   = 3         # Lang / Math / Reason
expert_capacity_factor: float = 1.5
expert_n_blocks:       int   = 3
expert_n_heads:        int   = 8
w_expert_aux:          float = 0.01      # routing entropy-collapse penalty
enable_latent_bus:     bool  = False
bus_dim:               int   = 16
bus_ema_alpha:         float = 0.5
bowtie_period:         int   = 4
bowtie_ema_alpha:      float = 0.4
trophic_prune_mat:     float = 0.3       # softened-pruning gate
```

### 0.8 Targeted-test guarantees

The SRC-TEH refactor is verified to keep the BRIAN narrative + causal
properties intact:

- `tests/test_narrative_memory.py::test_causal_generalization` — Gift→Joy
  generalisation on `ReasoningCortex.predict_reaction` after 120 epochs
  of `causal_aux_loss` training. PASSES.
- `tests/test_narrative_memory.py::test_predictive_forgetting_gain` — 100
  sleep-distill iterations on noisy episodes do not degrade held-out LM
  proxy fit. PASSES.
- All 129 tests in `tests/` pass.

### 0.9 Backwards compatibility

`enable_src_teh=False` (the default for `tiny`/`small`/`medium`/`large`)
preserves the exact legacy forward path; checkpoint round-trip works for
those presets unchanged. Switching `xl` to/from SRC-TEH changes parameter
shapes — checkpoints are NOT cross-compatible across the flag.

Checkpoint filenames carry the live param count
(`neuroslm_<preset>_<N>M_<optimizer>[_baseline]_<step>.pt`) so config
edits that change parameter shape (SRC-TEH on/off, d_hidden bumps,
expert resizes) are disambiguated on disk and the resume path can refuse
to cross-load mismatched shapes. The resume matcher accepts both the
new and the pre-tag naming so older checkpoints still resume.

### 0.10 Phased Maturation Engine

The first SRC-TEH training run (`commit c7cc8f7`) surfaced a regression
that was actually pre-existing but had been masked by infancy gates:
**awakening was a single switch**. At the moment MAT crossed 0.30 the
system simultaneously activated 8+ destabilising subsystems on top of a
model that had only barely beaten random — the LM stalled at
`lm_loss ≈ 7.0`, 5HT and GABA pinned at the 1.0 ceiling, Φ saturated at
~7 where `tanh(Φ/3)·3` has gradient ≈0.01, and 11-14 of 16 projections
were pruned within 100 awakened steps.

The fix replaces single-switch awakening with **per-subsystem MAT phase
gates** (smooth sigmoid `½(1 + tanh((MAT−c)/w))`). The aux-loss ramp
`α(t)` from `train.py` still multiplies on top, but each loss now has
its own onset window centred at the MAT level where that signal becomes
informative.

| Subsystem | Phase centre | Notes |
|---|---|---|
| `pred_coding` | 0.35 | Cheap internal supervision; engages earliest |
| `world` | 0.45 | World-model grounding once LM is past the random barrier |
| `motor` / `forward` | 0.50 | Action-space objectives need a working world model |
| `novel_aux` / `cpc` | 0.55 | Contrastive / novelty objectives |
| `kl_world` / `phi` | 0.60 | Heaviest objectives — last to engage |
| Mesolimbic CE gain | 0.55 | `learning_gain` is at random init below this MAT |
| Token-expert residual | 0.55 | Was pumping 31% noise into trunk at MAT 0.31 |
| Trophic structural pruning | 0.60 (`trophic_prune_mat`) | Suppresses `active[i]=0` through the post-awakening resolution window |

Implementation: `Brain._phase_gate(mat, center, width=0.10)` returns
0→1 smoothly across `[center−width, center+width]`. Loss assembly in
`forward_lm` multiplies each aux term by its phase factor; expert
dispatch passes `_mat_inner = phase_gate(MAT, 0.55, 0.15)` to
`forward_tokens(..., maturity=)`.

### 0.11 NT Saturation Scavenging

Independently of phasing, 5HT (`τ=0.95`) and GABA (`τ=0.90`) had a
structural problem: once a level pinned to 1.0, the maximum homeostatic
bias correction (`Δb = −0.5`) only dropped the level by ~1% per step
while every forward pass kept releasing fresh quanta. The level was
*physically incapable* of returning to the operating band.

`TransmitterSystem.step` now adds a **fast-reuptake scavenge**: whenever
`level > 0.9`, an extra 0.85× multiplicative drop is applied after the
normal `τ`-decay. Empirically 5HT/GABA at 1.0 return to ~0.72 / 0.54
within 5 steps. Models physiological auto-receptor inhibition and fast
extracellular reuptake triggered by transmitter excess.

```python
new_level = level * τ_decay + baseline * (1 − τ_decay)
sat_mask  = (new_level > 0.9).float()
new_level = new_level * (1 − 0.15 * sat_mask)            # scavenge
```

### 0.12 Φ-Loss Compression Widening

Previous: `L_Φ = −tanh(Φ/3) · 3`. At the observed Φ ≈ 7 this gives
gradient ≈ 0.01 of nominal — the Φ objective went silent exactly when
SRC-TEH started producing high-Φ states. New: `L_Φ = −tanh(Φ/8) · 8`,
which keeps the linear regime out to Φ ≈ 16 while still bounded.

| Φ | Old loss / grad | New loss / grad |
|---|---|---|
| 1 | −0.965 / −0.897 | −0.995 / −0.984 |
| 3 | −2.285 / −0.420 | −2.867 / −0.872 |
| 5 | −2.793 / −0.133 | −4.437 / −0.692 |
| **7** | **−2.944 / −0.037** | **−5.631 / −0.504** |
| 10 | −2.992 / −0.005 | −6.786 / −0.280 |

At the symptom value Φ = 7, the new compression returns a gradient
**13.6× stronger** than the old one — Φ becomes a useful objective
again.

### 0.13 Adaptive GWS Ignition

The training run that surfaced §0.10-0.12 also exposed a second
saturation pathology: the GWS ignition gate

$$\alpha_s = 0.3 + 0.7 \cdot \tfrac{1}{2}\bigl(1 + \tanh\bigl(6 (\|S_s\| - \theta_s)\bigr)\bigr)$$

was using a *static* per-slot threshold ($\theta_s \approx 0.8$). Under
SRC-TEH the trunk delivers candidates with much higher magnitudes
(post-Hopfield slot norms ≈ 2-8 vs. the original ≈ 0.8), so every slot
sat above $\theta_s$, $\alpha_s \to 1$, and the GWS stopped functioning
as a competitive bottleneck. With ignition saturated, Φ saturates
degenerately at ~7 (all slots correlated through the same broadcast)
and the bowtie's discriminative role vanishes — every module ends up
reading the same noise.

Fix: the threshold is now adaptive — it tracks the per-slot activity
scale via EMA so the ignition fraction stabilises near 50% regardless
of magnitude drift:

$$\theta^{\text{eff}}_s = \mathrm{EMA}[\|S_s\|] + \mathrm{margin} \cdot \sqrt{\mathrm{EMA}[\mathrm{Var}(\|S_s\|)]} + 0.1 \cdot |\theta^{\text{learn}}_s|$$

`workspace.py :: GlobalWorkspace.__init__` registers two persistent
buffers (`_activity_ema`, `_activity_var_ema`) and a step counter. A
two-phase EMA — α=0.5 for the first 20 steps (cold-start), α=0.05
afterwards (smooth tracking) — converges to the activity scale in ~10
steps. The learnable `slot_thresholds` is preserved as a small (×0.1)
residual bias so explicit per-slot discrimination can still emerge.

Empirically (random-init GWS, candidates from `randn(2,6,64)` with
post-Hopfield norm ≈ 8): ignition crashes 0.999 → 0.633 → 0.510 → 0.495
in 10 steps and stays at 0.495. The bottleneck is restored without any
hard cap.

### 0.14 MAT-Gated Memory-K/V Injection

The RETRO-style memory injection (§0.2 C) reads from `_bowtie_ema_slots`
and projects to d_hidden K/V rows on the trunk's last 2 blocks. At low
MAT the EMA is mostly random-init noise; feeding it as authoritative
retrieval pollutes the trunk. Mirroring the expert-residual policy,
`memory_kv` is now scaled by `phase_gate(MAT, 0.55, 0.15)`: 4% strength
at MAT 0.31, 50% at MAT 0.55, 97% at MAT 0.80. Below `1e-3` the entire
attention head is skipped (no compute) so early-awakening trunk passes
incur no memory-xattn cost.

### 0.15 Expert Residual Floor Removed

The legacy d_sem `MathCortex.forward` and `ReasoningCortex.forward`
methods clamped `m_eff = max(MAT, 0.05)` so a 5% residual always flowed
through. That 5% floor made sense for the d_sem path (small absolute
magnitude). At d_hidden under SRC-TEH it pumps unconditional noise into
the trunk's hidden state even when the inner phase gate (§0.10) says
"experts not ready yet". `forward_tokens` (used only by SRC-TEH) now
honours the caller's phase gate verbatim: below `m_eff < 1e-3` the
expert is a hard passthrough — no scatter-add, no router writes, no
gradient noise. Above 1e-3 it behaves as before.

### 0.16 Trophic Auto-Recovery

The softened prune gate (§0.5) prevents *new* deactivation while MAT <
0.6 but does not heal projections that were already pruned (the
checkpoint may carry an `active` mask from an earlier configuration, or
the SDNR-gated path may have aggressively pruned during a noisy
window). Observed symptom: `n_active: 5/16` stuck after the prune-mat
fix. `TrophicSystem.update` now ends with an **auto-recovery** step:
whenever the active fraction drops below `min_active_frac = 0.6`, the
top-(deficit) most-trophic *inactive* projections are reactivated and
their trophic level is lifted above `2.5·prune_threshold` so the next
update doesn't immediately re-deactivate them. The connectome can no
longer self-prune to a degenerate state.

### 0.17 Gradient-Accumulation Loss Display Fix

`train.py` was tracking the running LM loss across the full
grad-accum window but reporting only the *first* micro-batch's total
`loss`. This made the printed `loss` column oscillate ±0.5 nats step
to step (single-batch variance), while the `lm` column stayed smooth
(averaged variance ÷ √GA), creating a false "loss surging up" pattern
in the awakening logs. Both columns now average across the full GA
window — the displayed loss matches what's actually being optimised.

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

The **Φ loss term is bounded** so very high Φ does not dominate the gradient or push the network into a degenerate fully-coupled state where bipartition MIs collapse. The bound is `tanh(Φ/8)·8` (widened from the original `tanh(Φ/3)·3` per SRC-TEH §0.12 — the narrower form saturated at Φ≈6 and silenced the objective in the regime SRC-TEH reaches):

$$\mathcal{L}_{\Phi} = -\tanh(\Phi / 8) \cdot 8 \in [-8,\ 8]$$

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

The primary language processing stack — under SRC-TEH this is the **Tier-1
Shared Reading Cortex** that owns ~60% of total params and does LM,
comprehension, and knowledge extraction before any expert sees the stream.

Input: token IDs $(B, T)$.
Outputs (legacy): `(logits, sem, h, pred_coding_loss)`.
Outputs (SRC-TEH with `return_tap=True`): `(logits, sem, h, pred_coding_loss, tap_sem)`.

**New SRC-TEH knobs (see §0):**

- `enable_memory_xattn=True` adds a `MemoryCrossAttention` head (zero-init out)
  to the LAST `n_memory_xattn_layers` blocks. At forward time `memory_kv`
  carries the EMA of the bowtie's slot output (and, in future, top-N
  consolidated entries) — RETRO-style retrieval-augmented attention.
- `mid_trunk_tap_layer` (default = lang_layers/2) emits an `AttentionPool`
  pooled summary `tap_sem` that the brain feeds to the bowtie. Late
  trunk layers then receive the bowtie output back through the memory
  xattn — the trunk *uses* the bowtie, instead of merely biasing the last
  logit.
- `use_attention_pool=True` swaps the legacy `h.mean(dim=1)` for a learned
  1-token cross-attention into d_sem, preserving positional structure.

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

> **SRC-TEH note.** Under `cfg.enable_src_teh=True` both `MathCortex` and
> `ReasoningCortex` construct an *additional* 3-block transformer expert at
> `d_hidden` (constructor args `d_hidden`, `n_blocks`, `max_ctx`,
> `expert_n_heads`). The legacy d_sem fact-memory / Hopfield paths below
> are still built so existing tests pass, but they are **not** invoked in
> the SRC-TEH forward path — `Brain.forward_lm` calls
> `<cortex>.forward_tokens(routed_tokens)` instead. The token-level
> expert receives a `(B, C, d_hidden)` slice from the Expert-Choice
> router (§0.2 A) and applies: 3 transformer blocks → fact-memory or
> Hopfield-attractor cross-attention (zero-init out) → residual scaled by
> `max(MAT, 0.05)`. There is also a sibling `LanguageExpert`
> (`modules/expert_router.py`) which is a pure 3-block transformer +
> zero-init projection.

#### 4.2.1 MathCortex

*`neuroslm/modules/math.py`*

Activated by $\kappa_\text{math}$ vesicles (type `TOPIC_MATH = 1`) under
the legacy path, or by the Expert-Choice Router pulling its capacity
share of tokens under SRC-TEH.

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

Activated by $\kappa_\text{reason}$ vesicles (type `TOPIC_REASONING = 2`)
under the legacy path, or by the Expert-Choice Router under SRC-TEH (§0.2 A).

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

#### 4.4.1 Stochastic Thalamic Exploration

To prevent expert cortices from going dormant while the LM bootstraps on pure-text data, the router operates in a **stochastic ε-exploration** mode during training. With probability $\varepsilon$ the softmax distribution for a given batch item is replaced by a one-hot mass on a random **non-language** stream:

```python
# thalamus.py :: Thalamus.forward, training-only branch
explore = (rand(B, 1) < ε_eff).bool()
probs   = where(explore, one_hot(rand_int over {math, reasoning, spatial, social}), probs)
```

ε is **maturity-scaled** so young networks explore more aggressively:

$$\varepsilon_{\text{eff}} \;=\; \varepsilon\cdot(1 - M_t)
\qquad \varepsilon = 0.1\ \text{by default}$$

At $M=0$ a 10% slice of every batch is rerouted into an expert; at $M=1$ exploration is off (the router's learned policy takes over). This guarantees the MathCortex / ReasoningCortex / spatial / social streams receive non-zero gradient even during pure language training, so their parameters develop a baseline vocabulary and the Major Complex retains a non-trivial Φ across modules.

The choice of "non-language" pool (`EXPLORE_STREAMS = (1, 2, 3, 4)`) is deliberate: language already gets >99% of natural routing mass, so exploring into it would be a no-op. The rule is implemented as `torch.where` over a precomputed `one_hot` tensor → fully XLA-static, no Python branching on the trace.

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

### 6.4 Topological Maturation — The Maturity Index (MAT)

*`neuroslm/neurochem/transmitters.py::compute_mat`, `neuroslm/brain.py::update_maturity`, `neuroslm/train.py` (per-step scheduler)*

Training is governed by a continuous **Maturity Index** $M_t \in [0, 1]$ — a virtual "MAT protein" computed from the live LM loss:

$$M_t \;=\; \mathrm{clamp}\!\Bigl(1 - \frac{L_{\text{lm}}}{L_{\text{random}}},\ 0,\ 1\Bigr)
\qquad L_{\text{random}} \approx \log(\text{vocab}) \approx 10.84$$

The Brain stores a running EMA of $M_t$ as the buffer `brain.maturity`:

```
brain.maturity ← (1 - α)·brain.maturity + α·M_t        (α = 0.05)
brain._infancy ← (brain.maturity < 0.3)                 # legacy gate, auto-derived
```

`update_maturity(lm_loss)` is called once per training step in `train.py`. The EMA smooths transient bumps so the fade-in does not whiplash on noisy mini-batches.

#### Continuous "Fade-In" of expert cortices

Replaces the hard binary switch with **convex-combination residuals**:

$$h_{\text{out}} \;=\; (1 - m_{\text{eff}})\cdot h_{\text{in}} \;+\; m_{\text{eff}}\cdot \mathrm{Expert}(h_{\text{in}}),
\qquad m_{\text{eff}} = \max(M_t,\,0.05)$$

The 0.05 floor is a **noise broadcast** — at $M_t<0.1$ the network still "feels" the architectural boundary at 5% strength, so gradients reach the expert cortices long before they become load-bearing. `MathCortex.forward`, `ReasoningCortex.forward`, and the orchestrator's GWS-side integration all take `maturity` as a kwarg and apply this rule via simple arithmetic (XLA-static, no `if` branching).

#### Maturity-aware GABA homeostasis

A mature network is *allowed* to fire at higher variance because its activity is signal, not noise. The GABA PID controller (`neurochem/vesicles.py::PIDController.step`) shifts its target variance with $M_t$:

$$\sigma^2_{\text{target,eff}} \;=\; \sigma^2_{\text{target}}\cdot(1 + 1.5\,M_t)
\qquad \sigma^2_{\text{target}} = 0.5,\quad \text{range}\ 0.5\!\to\!1.25$$

So the dampening cuts in at low $M_t$ (suppress bowtie screech during random-init) and relaxes as $M_t$ climbs.

#### Legacy compute-skip gates (still useful)

The pre-MAT infancy flag is now derived as `_infancy = (M < 0.3)` and used **only** for FLOP-saving compute skips (vesicle synthesis, trophic update, episodic-buffer writes) — they no longer affect the forward graph's correctness, just its wall-clock cost while signals are still noise.

**Softened Trophic Gate (SRC-TEH §0.5).** Independent of the infancy
skip, `TrophicSystem.update` now also accepts `maturity` +
`prune_mat_threshold=0.3`. Pruning (setting `active[i]=0`) is suppressed
while `MAT < prune_mat_threshold`. Trophic levels still drift so the
learned plasticity state accumulates — only structural deactivation is
held off. Closes the historic failure mode where a random-init projection
graph self-pruned to `n_active: 0` before any learning could occur.

```
brain._aux_w_scale  = 0.001 → 1.0          # ramps after awakening (see below)
brain._infancy      = (M < 0.3)            # auto-derived, compute-skip only
gws.slot_thresholds = lerp(1.2, 0.8, M/0.3)  # ignition threshold relaxes with M
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

Two conditions must be met **simultaneously**:

1. **`brain.maturity > 0.3`** (the EMA-smoothed MAT has cleared the infancy band)
2. **`lm_loss < 7.5`** (raw LM has stabilised below random; for vocab≈50k, $\ln(50\text{k})\approx 10.84$ is the random ceiling)

When both hold, the scheduler sets `_maturation_awakened = True` permanently and:

- `homeostasis.observe` begins correcting NT bias/gain toward targets
- Auxiliary-loss ramping ($\alpha$) begins
- Trophic structural plasticity and vesicle dynamics fully engage

The step-count gate (formerly `step ≥ 5000`) is gone — awakening is now purely **performance-driven**, so a fast learner can awaken early and a slow learner can stay in linguistic-bootstrap mode for as long as it needs.

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

**Per-subsystem phase gates (SRC-TEH §0.10).** `α(t)` is now the *master*
scale; on top of it, each individual aux loss has its own MAT-keyed
phase gate so awakening is no longer a single switch. Effective weight
of aux loss $i$:

$$w_i^\text{eff}(t) = \alpha(t) \cdot \sigma_i(M_t) \cdot w_i, \qquad \sigma_i(M) = \tfrac{1}{2}\bigl(1 + \tanh\bigl((M - c_i)/w\bigr)\bigr)$$

with per-loss centres $c_i$ from §0.10 (pred_coding 0.35, world 0.45,
motor 0.50, novel/cpc 0.55, kl_world / phi 0.60). The mesolimbic CE
gain, the token-level expert residual, and the trophic structural
pruning all use the same gate shape with their own centres.

Once $\alpha$ reaches 1.0 *and* MAT clears every $c_i$ window, every auxiliary loss applies at its config-default weight and trophic/BDNF growth begins shaping the projection graph based on real Φ and Fiedler signals.

#### Why this matters

Without infancy gating, two failure modes occur:

1. **Aux gradient noise dominates LM** — world/motor/Φ losses produce structured gradients against a random target, fighting the LM's progress toward language. Loss stalls at the random-init ceiling.
2. **Trophic / homeostasis target the wrong configuration** — NT homeostasis pulls levels toward `target_mean=0.3` regardless of whether the network is learning, driving 5HT and GABA to ceiling and pinning ignition at saturation. Trophic growth + BDNF rank increase build up structure based on phi/fiedler measurements of random covariance.

The infancy/awakening split implements **"linguistic first" convergence**: let the LM cortex find token-level structure first, then let the bowtie + bio modules layer integration on top of a substrate that already carries signal.

---

## 7. Optimization & Infrastructure

### 7.1 Adaptive Compute (Maturity-aware)

#### Mixture of Depths (MoD)

*`neuroslm/modules/mixture_of_depths.py` — `MoDRouter`, `MoDBlock`*

Each `MoDBlock` routes only the top-$C$ "hard" tokens through the transformer sublayer; the rest skip via residual:

$$C = \max\!\left(1,\ \lfloor T \cdot \rho \rfloor\right), \quad \rho = \rho_0 \cdot \left(0.5 + \sigma(W_\text{nt} \cdot \text{NT})\right) \cdot s_{\text{mat}}$$

$\rho_0$ is the base capacity ratio (`mod_capacity=0.8` for xl). The NT modulation adjusts capacity dynamically: high ACh → higher capacity (more tokens processed in full), high NE → lower (focused on hardest tokens only).

**Maturity gating** ($s_{\text{mat}}$ via the per-block `_maturity_gate` attribute set by Brain before each forward):

| $M_t$ band | MoD behaviour | Why |
|---|---|---|
| $M_t < 0.2$ | **Hard skip** — block returns input as-is (zero FLOPs in attn+MLP) | Expert layers add ~noise while the LM head is still bootstrapping |
| $0.2 \le M_t < 1$ | $s_{\text{mat}} = 0.5 + 0.5\cdot\frac{M_t - 0.2}{0.8}$, scales router capacity 0.5×→1.0× | Gradual return of expert capacity as the network matures |
| $M_t = 1$ | Full base capacity | Steady-state behaviour matches the pre-MAT release |

The hard skip is a Python branch on a plain-Python `float` attribute (set outside the trace), so XLA does not retrace per-step — `_maturity_gate` is read once per forward and the chosen graph is reused.

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

**Saturation scavenging (SRC-TEH §0.11).** Slow-decay transmitters
(5HT τ=0.95, GABA τ=0.90) can pin to the 1.0 ceiling under sustained
release because even the maximum negative homeostatic bias (Δb = −0.5)
only drops the level ~1% per step. To prevent permanent saturation,
after the τ-decay update an extra **0.85× scavenge** is applied to any
channel whose updated level exceeds 0.9:

$$\text{sat\_mask} = \mathbf{1}[\text{level}_i(t+1) > 0.9],\qquad \text{level}_i \leftarrow \text{level}_i \cdot (1 - 0.15 \cdot \text{sat\_mask})$$

Empirically returns ceiling-pinned 5HT/GABA (1.0) to ~0.72/0.54 within
5 steps. Models physiological auto-receptor inhibition / fast
extracellular reuptake triggered by transmitter excess.

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
| `xl` | ~240 M (SRC-TEH, live ≈240 M) | 384 | **576** | **10** | 2 048 | 8 | 4 096 | A100 40 GB |
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
# Trunk (Tier 1) — SRC-TEH bumps d_hidden 512→576, drops lang_layers 12→10
d_hidden            = 576
lang_layers         = 10
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
baseline_lang_layers = 48       # vanilla baseline parity at d_hidden=576

# SRC-TEH topology (§0) — defaults ON for xl
enable_src_teh         = True
enable_memory_xattn    = True
n_memory_xattn_layers  = 2
n_memory_entries       = 64
mid_trunk_tap_layer    = 5      # tap at half-depth (layer 5 of 10)
n_token_experts        = 3
expert_capacity_factor = 1.5
expert_n_blocks        = 3
expert_n_heads         = 8
w_expert_aux           = 0.01
enable_latent_bus      = True
bus_dim                = 16
bus_ema_alpha          = 0.5
bowtie_period          = 4
bowtie_ema_alpha       = 0.4
trophic_prune_mat      = 0.3
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

---

## 10. BRIAN — Narrative + Causal Memory Stack

*Biologically Realistic Information Architecture Network — the codename for the post-awakening narrative + causal memory subsystem layered on top of the bowtie.*

Everything in this section is **gated by `_maturation_awakened`**: the structural primitives are constructed at brain init time (so checkpoints round-trip), but their effects on dynamics are inert during infancy. Awakening (§6.4) flips them on.

### 10.1 Contextual Sheaf F & H¹ Contradiction Detection

*`neuroslm/memory/sheaf.py`*

The relational hypergraph (§4.2 / `memory/hypergraph.py`) is overlaid with a **sheaf structure** F that assigns:

  • a local belief vector (the *section*) to each node U_i ∈ ℝ^{d_emb},
  • a learned linear *restriction map* R_{ij} : ℝ^{d_emb} → ℝ^{d_emb} to each edge (identity by default; learnable per-edge for non-trivial types).

The **1-cochain** on edge (i, j) is:

$$c_{ij} = R_{ij} \cdot v_i - v_j$$

Strict cohomological H¹ projects this cochain onto the orthogonal complement of im(δ⁰) — the part not explainable by a 0-coboundary correction. We compute that via a `lstsq` against the incidence matrix B ∈ ℝ^{|E| × |V|}.

**However**, two-node identity-restriction "contradictions" (the canonical "Alice likes coffee" / "Alice hates coffee" case) are 0-coboundaries in the strict sense — shifting both nodes equally resolves them. So for practical contradiction detection we additionally maintain a **raw pairwise inconsistency** signal:

$$\text{raw}(F) = \frac{\sum_{(i,j) \in E} w_{ij} \cdot \|R_{ij} v_i - v_j\|_2}{\sum_{(i,j) \in E} w_{ij}}$$

`SheafSection.h1_residual` holds this raw measure. `SheafConsistencyChecker.is_contradiction(section)` returns True when it exceeds the threshold (default 0.7). On contradiction, the **newer timestamp wins**: a `SUPERSEDES` edge is created from the newer node to the older one, and `is_superseded(older)` returns True for downstream gating.

**Global-section retrieval** runs damped Jacobi (4 iters, damping=0.5) over the node values to produce the maximum-consistency joint interpretation across multiple context patches. Retrieved via `MemoryHyperGraph.retrieve_global_section(query_emb)`.

### 10.2 Actual-Causation Head (IIT 4.0)

*`neuroslm/modules/actual_causation.py` — `ActualCausationHead`*

IIT 4.0 defines actual causation between source state $s_t$ and effect $e_{t+1}$ via the integrated information $\phi_c$ of the intervention $do(s = \text{counterfactual})$. The full enumeration is intractable; we use the standard tractable proxy:

$$\alpha(i \to j; t) = \big\| f_j(z_i^t) - f_j(z_i^{\text{baseline}}) \big\|_2^2 \cdot \sigma\!\left(\frac{\langle q(z_j^{t+1}),\ k(f_j(z_i^t)) \rangle}{\sqrt{d_h}}\right)$$

where $z_i^t$ is module i's mean output at time t, $f_j$ is a small shared MLP conditioned on (i, j) one-hots, and the **do(s = baseline)** reference is each module's running EMA of its own output. The attention term gates pairs where module j actually attended to module i.

Output: per-edge causal strength $\alpha \in [0, 1]^{n \times n}$, normalised across destinations per pass.

The head is trained passively via `aux_loss(prev, cur)` — an MSE that forces $f_{i \to j}(z_i^t) \approx z_j^{t+1}$, giving the do-intervention proxy its semantic grounding. Added to the total loss with weight `_aux_w_scale · w_causal = 0.05`, naturally suppressed during infancy.

A running EMA `alpha_ema` is the input to two downstream consumers:

  • **κ_cause vesicle emission** when any α ≥ `gate_threshold = 0.3` (§10.9)
  • **Trophic renormalisation** during the sleep cycle (§10.7)

### 10.3 ReasoningCortex Action → Reaction Predictor

*`neuroslm/modules/reasoning.py`*

The existing Hopfield-attractor cortex (§4.2.2) is extended with two new components:

**Low-rank recurrent dynamics** (causal attractor layer):

$$h_{t+1} = \tanh\!\big( A B \cdot h_t + W_{\text{in}} \cdot x_t \big), \quad A \in \mathbb{R}^{d \times r},\ B \in \mathbb{R}^{r \times d},\ r = 16$$

Two unrolled steps. Zero-init on B so the network starts as a pure passthrough of the Hopfield retrieval; the recurrent term learns low-rank fixed points that encode abstract relational schemas ("Insult → Offense").

**Action → Reaction predictor**:

```python
probs, completed = cortex.predict_reaction(action_emb)   # (B, n_action_types), (B, d_sem)
```

Two routes co-trained:

  1. Direct MLP classifier: `action_emb → logits / temperature`
  2. Modern Hopfield completion over a learnable prototype bank `reaction_prototypes ∈ ℝ^{T × d_sem}`

Logits are averaged across the two routes; `softmax` produces the categorical reaction distribution. `n_action_types = 14` to match `SocialMarkovMemory.ACTION_LABELS`.

Auxiliary cross-entropy loss `causal_aux_loss(action_emb, reaction_target_idx)` trains both routes simultaneously. **Test `test_causal_generalization`** confirms that after 120 epochs on 10 Gift→Joy and 10 Insult→Offense pairs, novel Gift inputs receive P(Joy) > 0.8.

### 10.4 Narrative Engine (JSON stories) + EntityNarrative trust

*`neuroslm/memory/narrative.py` — `NarrativeSystem`*

The existing autobiographical / world / entity narrative streams are extended with **structured JSON exports**:

```python
ns.self_summary(identity="Self") → {
    "identity": "Self",
    "tone": float, "coherence": float,
    "events": [
        {"t": int, "subject": "Self", "content": str, "valence": float, "salience": float},
        ...
    ]
}

ns.full_story(personality_vector=personality) → {
    "identity": "Self",
    "self": <self_summary>,
    "world": {"tone": ..., "coherence": ..., "n_events": int},
    "entities": [
        {
            "entity": "alice", "trust": 0.83, "confidence": 12.0,
            "nt_bias": {"DA": +0.08, "5HT": +0.05, "NE": -0.07},
            "events": [...]
        }
    ]
}
```

Trust and NT-bias fields are filled by `PersonalityVector` when passed in. Chronological order across events is preserved by sorting on `entry.timestamp`. **Test `test_autobiographical_coherence`** verifies the JSON structure stays consistent across three sequential events.

### 10.5 PersonalityVector → NT-baseline coupling

*`neuroslm/neurochem/personality.py` — `PersonalityVector`*

A slowly-evolving 5-dim trait vector P = (curiosity, agreeableness, vigilance, patience, hedonic_tone) and a per-entity **Beta-Bernoulli trust posterior**:

$$\alpha_e \leftarrow \alpha_e + \tfrac12(1 + v), \quad \beta_e \leftarrow \beta_e + \tfrac12(1 - v), \quad \text{trust}(e) = \frac{\alpha_e}{\alpha_e + \beta_e}$$

Personality drifts under a small learning rate $\eta_P = 5 \times 10^{-3}$ (≈ 200 consolidations to halve a trait) toward a drive vector supplied at consolidation time.

The vector and trust scores **bias the homeostasis baseline targets**:

| Personality dim | NT coefficients |
|---|---|
| curiosity     | +0.08 DA, +0.06 ACh, +0.02 NE |
| agreeableness | +0.06 5HT, +0.04 GABA |
| vigilance     | +0.10 NE, +0.03 ACh, −0.02 GABA |
| patience      | +0.08 5HT, +0.05 GABA, −0.02 DA |
| hedonic_tone  | +0.07 DA, +0.05 5HT |

Per-entity trust contributes additively when that entity is in the working set:

  • $\Delta b_{\text{DA}} = +0.10 \cdot w_e \cdot (\text{trust}(e) - 0.5)$ (trusted → DA up)
  • $\Delta b_{\text{5HT}} = +0.06 \cdot w_e \cdot (\text{trust}(e) - 0.5)$ (trusted → 5HT up)
  • $\Delta b_{\text{NE}} = +0.10 \cdot w_e \cdot (0.5 - \text{trust}(e))$ (distrusted → NE up)

All bias contributions sum and are then clamped to [−0.5, +0.5] inside `Homeostasis.observe`. **Test `test_theory_of_mind_consistency`** confirms Alice (positive valence history) accumulates higher DA bias and lower NE bias than Bob (negative valence history).

### 10.6 NEMORI Predictive-Forgetting Gate

*`neuroslm/memory/comprehension_gate.py`*

The existing surprise × comprehension × novelty gate is extended with the **NEMORI prior**: only the part of an observation that exceeds the model's anticipated surprise survives. The episode is rejected unless

$$\text{unpredicted\_surprise} = \text{surprise} - \text{anticipated\_surprise} \geq \text{nemori\_floor}$$

`nemori_floor = 0` is the default (back-compat); set positive to enforce predictive forgetting. The returned dict carries `unpredicted_surprise` and `nemori_kept` flags for telemetry.

### 10.7 Sleep-Cycle CLS

*`neuroslm/memory/sleep_cycle.py` — `SleepCycle`*

Every `sleep_period_steps = 5000` *awake* steps, the brain enters a brief sleep phase. Four operations:

  1. **Replay**: sample `replay_batch = 16` episodes per iteration (`n_iters = 4`) weighted by `salience × decay × |valence|`.
  2. **Bidirectional predictive coding distillation**: for each replayed episode, compute (a) the top-down predicted code `pc = predictor(emb)` and (b) the bottom-up reconstruction through the low-rank slow-weights adapter `bu = emb + emb @ slow_a @ slow_b`. The MSE between pc and bu is minimised, but **only on episodes where the NEMORI gate fires**.
  3. **Trophic renormalisation**: edges with `α_ema < 0.3` get their trophic level decayed by 0.05 (eventually pruned); edges with `α_ema ≥ 0.3` get boosted by 0.05.
  4. **Gaussian I(X;Z) proxy**: empirical MI between input embeddings and predictor output. Reported as `mi_reduction = post - pre` — negative values indicate compression.

A `SleepReport` is logged each cycle with `n_replays`, `distillation_loss`, `mi_reduction`, `pruned_edges`, `strengthened_edges`, and `duration_s`. **Test `test_predictive_forgetting_gain`** confirms 100 sleep iterations on a noisy buffer do not blow up I(X;Z) and do not degrade a held-out LM-proxy fit.

### 10.8 DNC Temporal-Link Matrix L

*`neuroslm/memory/hypergraph.py`*

The hypergraph maintains a **fixed-size DNC link matrix** L ∈ ℝ^{N × N} (N = 256 slots, recycled by oldest precedence). Each write updates:

$$L[\text{cur}, j] = \mathbf{1}[\text{prev\_slot} = j]$$

This records *write-order* transitions independently of wall-clock time. `temporal_link_neighbours(node_id, direction)` returns the top-k nodes typically written before or after a given node — the substrate for sequence-aware recall in the bowtie's hippocampal stage.

### 10.9 κ_cause Vesicles

*`neuroslm/neurochem/vesicles.py`*

New vesicle type `TOPIC_CAUSE = 4` (raising `N_VESICLE_TYPES = 5`). Emitted in `Brain.forward_lm` whenever `ActualCausationHead` reports any pair with α ≥ `gate_threshold = 0.3`. The vesicle carries the destination module's embedding as its content; on docking it stabilises causal-rule attractors in the ReasoningCortex by raising the activation of the corresponding prototype.

### 10.10 Forward-Pass Wiring

The BRIAN stack hooks into `brain.forward_lm` immediately after `_maybe_store_insight` and before the periodic consolidation block — all guarded by `if not _in:` so the entire stack is inert during infancy. The order of operations:

```text
1. Stack 8 canonical module outputs: (sem, selected, dmn_query, routed,
                                       action, dmn_query_mod, slots, motor_lang_bias)
2. ActualCausationHead(prev, cur)               # α (n, n), updates EMA
3. Emit κ_cause vesicle for strongest α edge   # type=TOPIC_CAUSE
4. PersonalityVector.apply_bias(transmitters,   # if any entity is in focus
                                  active_ents)
5. SleepCycle.maybe_sleep(step)                 # every 5000 awake steps
6. Cache cur as prev for next forward
```

Awakening is propagated via `Brain.set_awakened(True)`, called from `train.py` at the maturation transition. PersonalityVector and SleepCycle gate themselves internally on this flag.

---

---

## 11. Cognitive Closure — Survival-Gated Action Loop

*The embodied half of BRIAN. Where §10 covers post-awakening narrative + causal memory over text, §11 closes the loop on action: the model **acts to survive** in a latent grid manifold and uses the resulting survival signal to shape its own neurochemistry, attractor structure, and policy memory.*

All components in this section are constructed at brain init time (parameters round-trip in checkpoints) and are **idle for text-only training**. They only fire when the embodied loop is active — `Brain.ingest_grid_frame(frame)` and `Brain.tick_homeostasis(...)` are the entry points the embodied runner calls per env step.

### 11.1 GridWorld Environment (10×10 SHRDLU)

*`neuroslm/env/grid_world.py` — `GridWorld`, `GridFrame`*

A 10×10 cell grid populated with 7 tile types (`EMPTY`, `AGENT`, `FOOD`, `WATER`, `OBSTACLE`, `RUDE_USER`, `FRIENDLY`). The agent issues one of 6 discrete actions per tick: `NOOP / UP / DOWN / LEFT / RIGHT / INTERACT`.

Each tick emits a `GridFrame` carrying **three parallel sensory streams**:

| Stream | Shape | Content |
|---|---|---|
| `vision` | `(10, 10, 7)` one-hot | Spatial block positions, agent stamped on top |
| `affordance` | `dict[str → list]` | Typed adjacency relations (`above`, `below`, `left_of`, `right_of`, `next_to`, `on_top_of`) over occupied cells |
| `homeostatic` | `(3,)` ∈ `[0,1]³` | `(energy, hydration, integrity)` — the agent's internal state |

Plus the frame's `reward`, `survival_pressure`, `agent_pos`, `tick`, `caption`, and `done` flag.

**Dynamics:**

  • `INTERACT` on `FOOD` → +0.5 energy, tile cleared.
  • `INTERACT` on `WATER` → +0.5 hydration, tile cleared.
  • `INTERACT` on `RUDE_USER` → −0.20 integrity.
  • Adjacency to `RUDE_USER` → −0.02 integrity per tick.
  • Adjacency to `FRIENDLY` + `INTERACT` → +0.05 integrity.
  • Every tick: energy and hydration each decay by `decay_per_tick = 0.01`.
  • Starvation (E or H < 0.1) bleeds integrity at −0.005 per tick.
  • Episode terminates when integrity ≤ 0.

`GridWorld.stream(policy_fn)` is an infinite generator that calls the policy each tick and yields the resulting frame — `Brain.run_continuous` is meant to consume from it directly.

### 11.2 Sensory VAE Front-End

*`neuroslm/modules/sensory.py` — `SensoryVAE`*

A tiny β-VAE that compresses one `GridFrame` into the bowtie's `d_sem` manifold:

  • **Vision encoder**: 2-conv stack `(7, 10, 10) → (16, 10, 10) → (32, 5, 5)` → flatten → `d_sem / 2`.
  • **Homeostatic encoder**: 2-layer MLP `(3,) → d_sem / 4`.
  • **Affordance encoder**: 2-layer MLP on the 6-vec of relation counts → `d_sem / 4`.
  • **Mix + posterior heads** produce `(μ, log σ²) ∈ ℝ^{d_sem}`; reparameterise → `z`.
  • **Decoder** reconstructs only the vision channel (homeostat + affordance are too low-dim to need decode; KL pressure shapes their slice of z).

Loss: `MSE(decoded, vision) + β · KL(N(μ,σ²) || N(0,I))`. Trained as part of the auxiliary loss block, gated by `_aux_w_scale` like every other aux objective.

`Brain.ingest_grid_frame(frame)` calls `SensoryVAE.encode_frame` to produce the residual that the embodied loop injects into the bowtie.

### 11.3 Latent Qualia Manifold Q & Homeostatic Warp

*`neuroslm/modules/qualia.py` — `QualiaState.warp_broadcast`*

The Q-manifold biases the GWS broadcast mean by a learnable direction whose magnitude scales with homeostatic deficit:

$$\text{deficit}_c = \max(0, \tau_{\text{av}} - s_c), \quad \text{surplus}_c = \max(0, s_c - \tau_{\text{av}})$$

$$D = \sum_c g_c \cdot \text{deficit}_c, \quad S = \sum_c g_c \cdot \text{surplus}_c$$

$$\text{warp}(b) = b + D \cdot \hat{n}_{\text{av}} + \alpha_S \cdot S \cdot \hat{n}_{\text{ap}}$$

with aversive threshold $\tau_{\text{av}} = 0.20$, per-channel gains $g = (1.5, 1.0, 2.0)$ for energy/hydration/integrity, appetitive scale $\alpha_S = 0.04$. The aversive direction $\hat{n}_{\text{av}}$ and appetitive direction $\hat{n}_{\text{ap}}$ are learnable; aversive is initialised at 10× the magnitude of appetitive so the warp behaviour is asymmetric from step 0 (starvation → strong reinterpretation; satiety → mild bias).

`QualiaState.aversive_pressure()` exposes the scalar $D$ for downstream gating (§11.4). **Test `test_survival_imperative_qualia_shift`** confirms that setting energy to 0.05 produces a substantially higher aversive pressure than full-health, with a warp magnitude ≥ 2× the healthy bias.

### 11.4 κ_neg Aversive Vesicles

*`neuroslm/neurochem/vesicles.py`*

New vesicle type `TOPIC_NEG = 5` (raising `N_VESICLE_TYPES = 6`). Emitted in `Brain.ingest_grid_frame` whenever `QualiaState.aversive_pressure() > 0.10`. The vesicle carries the homeostat-warped sensory latent `z` as its content; it migrates from the GWS module through the vesicle-pool's learnable transition matrix `T` — by training-time these routes preferentially terminate at the **ReasoningCortex** to stabilise escape / foraging attractor schemas.

This is the inverse of the κ_cause vesicle (§10.9): κ_cause locks in *learned* high-causation pathways, κ_neg drives *novel* search when survival pressure rises.

### 11.5 Basal Ganglia VQH + Expert Gating

*`neuroslm/modules/basal_ganglia.py` — `BasalGanglia.select_option`*

The continuous action vector from the existing Go/NoGo selection is now additionally quantised onto a **discrete option lattice** (`n_options = 16` learnable codebook entries). Each option carries:

  • a key vector `k_i ∈ ℝ^{d_action}` (matched against the proposer via cosine similarity),
  • an **expert-routing logit** triplet `e_i ∈ ℝ^{n_experts}` with `n_experts = 3` covering `{Math, Reasoning, Motor}`.

VQ-VAE-style straight-through quantisation. The standard codebook + commitment loss:

$$\mathcal{L}_{VQ} = \|k_{i^*} - \text{sg}[a]\|^2 + \beta \cdot \|\text{sg}[k_{i^*}] - a\|^2, \quad \beta = 0.25$$

`select_option(action, da_bias)` returns `(option_idx, expert_probs, vq_loss, value_pred)`. The expert-routing distribution gates which downstream expert cortex actually drives the next motor output — the BG is now functioning as a **discrete cortical controller**, not just an action filter.

### 11.6 NAcc Reward-Prediction-Error

*`neuroslm/modules/basal_ganglia.py` — `BasalGanglia.nacc_rpe`, `update_option_value`*

A small 2-layer value head over `(thought, action) → ℝ` predicts the expected survival-aligned reward of the current (state, action) pair. After observing the env's actual survival outcome, the brain computes:

$$\text{RPE} = r_{\text{actual}} - \hat{V}(s, a)$$

Positive RPE → the brain releases a DA spike (read by `TransmitterSystem.release("DA", ...)`). The DA spike then:

  1. Boosts the **option's persistent policy value** via `update_option_value(idx, rpe, lr=0.1)`: EMA-blends the option's `option_da_value` toward the RPE.
  2. Amplifies trophic factor release on the active edges (the Φ-coupled BDNF mechanism, §6.2) — turning successful survival actions into structural plasticity (kernel rank increase in the NeuralGeometryAdapters that mediate the policy).
  3. Stabilises the current Φ-structure in the hippocampal hypergraph (§4.2) by raising the salience tag of recent episodes.

**Test `test_basal_ganglia_policy_adaptation`** confirms that 100 +0.8 RPE updates on a target option pull its DA-value above 0.5 while leaving the other 15 options near zero.

### 11.7 SurvivalCausalHead — action → ΔS_{t+1}

*`neuroslm/modules/survival_causal.py` — `SurvivalCausalHead`*

The sister of `ActualCausationHead`. Where §10.2 estimates module→module actual causation across the bowtie, this head estimates **action → next-step survival contribution**:

$$\alpha_{\text{surv}}(a_t) = \sigma\!\left(\|f(a_t) - f(a_{\text{baseline}})\|^2\right)$$

with $f : \mathbb{R}^{d_{\text{action}}} \to \mathbb{R}^3$ a small MLP that predicts $\Delta S_{t+1} = (S_{t+1} - S_t)$ from the action embedding, and $a_{\text{baseline}}$ an EMA of recent action embeddings.

Trained via the MSE auxiliary loss `loss(action, actual_delta_S)`. The output `α_surv` is the scalar that tells the brain "this action class meaningfully altered survival" — read by the trophic system to gate BDNF release on the active pathway. **Test `test_world_model_causal_predictivity`** confirms that 40 training epochs on a deterministic action → ΔS dataset drop the predictor's MSE by ≥ 50%.

### 11.8 Homeostasis.step Tick

*`neuroslm/neurochem/homeostasis.py` — `Homeostasis.step`*

Per-tick decay applied by the embodied loop:

  • Energy and hydration each decrement by `decay = 0.01`.
  • When `reward_action=True`, the channel that is currently most depleted is incremented by `reward_value` (clamped to 1.0).
  • Starvation (E or H < 0.10) bleeds integrity at −0.005 per tick.

This is invariant to the env: the env's own `step()` applies the same decay rule, so the agent's *internal* survival_state (held on the Brain as a buffer) stays in lockstep with the env's external one. `Brain.tick_homeostasis(reward_action, reward_value)` is the entry point.

### 11.9 Wiring Summary

```text
GridWorld.step(action)
   │
   ├── GridFrame {vision, affordance, homeostatic, reward, survival_pressure}
   │
   ▼
Brain.ingest_grid_frame(frame)
   │
   ├── SensoryVAE.encode_frame(frame)             → z ∈ ℝ^{d_sem}
   ├── QualiaState.warp_broadcast(z, homeostatic) → z_warped (aversive bias)
   ├── if aversive_pressure > 0.10:
   │     VesiclePool.synthesize_typed(z, TOPIC_NEG, source=GWS)
   │     → migrates → ReasoningCortex
   ├── Brain.survival_state ← frame.homeostatic
   │
   ▼
[BOWTIE FORWARD PASS — modified by z_warped + κ_neg vesicles]
   │
   ▼
BasalGanglia.forward(...)            → continuous action a
BasalGanglia.select_option(a)        → discrete option_idx + expert_probs
                                       + vq_loss
BasalGanglia.nacc_rpe(...)           → RPE; +RPE → DA release → BDNF amp
SurvivalCausalHead(action)           → α_surv → BDNF on active pathway
BasalGanglia.update_option_value(idx, rpe)   ← policy memory
   │
   ▼
Brain.tick_homeostasis(reward_action=bool(rpe>0), reward_value=rpe.clamp(0,1))
```

The closed loop runs at the env's tick rate (10 Hz default); the bowtie's forward pass is what shapes the next action.

---

*Last updated: 2026-05-19 (SRC-TEH §0, Phased Maturation §§0.10-0.12, Adaptive GWS §0.13, MAT-gated memory §0.14, Expert floor removal §0.15, Trophic recovery §0.16, GA loss-display fix §0.17 — see [`docs/RFC.md`](RFC.md)). Source of truth: `neuroslm/` on branch `master`.*
