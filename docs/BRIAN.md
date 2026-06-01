# BRIAN: Biologically Realistic Information Architecture Network

A complete explanation of the core architecture, why each design choice exists, and what it's trying to prove.

---

## What Is BRIAN?

BRIAN is a **brain-inspired language model** that explores whether you can achieve better generalization and reasoning at lower parameter counts by building an **actual brain topology** instead of a bigger transformer.

**Core claim:** Integrated information (Φ) + bowtie topology + embodied loop + narrative memory = a system that reasons better on limited data than flat transformers at matched compute.

**Size:** 30M–240M parameters (vs 1B+ for comparable flat transformers).  
**Hardware:** Fits on single A100. Trains in hours on 100M scale.

---

## The 11-Stage Forward Pass (NeuralOrchestrator)

BRIAN's forward pass is a **deliberate recreation of mammalian brain anatomy**, with each stage mapping to a real neural system:

```
Stage 0  SENSORY
         TextSensoryCortex + SensoryVAE
         ↓ "What is the input?"

Stage 1  THALAMIC ROUTING
         Thalamus (5-stream router: language, math, reasoning, spatial, social)
         ↕ [re-entry from PFC ← Stage 7]
         ↓ "Which modality is relevant?"

Stage 2  STATE MODELS
         WorldModel (predicts future state from action)
         SelfModel (tracks agent's own state)
         ↓ "What's happening next? What am I?"

Stage 3  AFFECT
         Amygdala (fear, valence)
         LateralHabenula (anti-reward, aversion)
         Insula (interoception, gut feeling)
         ↓ "Is this good or bad?"

Stage 4  QUALIA
         QualiaState (binds world + affect)
         Homeostatic warp (reinterprets all input as "urgent" under starvation)
         ↓ "What is the *feeling* of this world?"

Stage 5  GLOBAL WORKSPACE (BOTTLENECK)
         GlobalWorkspace + Hopfield dynamics + ignition gate
         Input: [sensory, routed, world, self, thought, qualia, hippo, ...]
         Output: 8 discrete slots (competitive binding)
         Computes Φ (integrated information)
         ↓ "What's in conscious access right now?"

Stage 6  MEMORY CONSOLIDATION
         Hippocampus (novelty-gated storage)
         Entorhinal (context grid)
         HyperGraph (relational updates)
         Sheaf retrieval (retrieve memories with H¹ coherence check)
         ↓ "What can I remember? Do old beliefs conflict?"

Stage 7  COGNITIVE CONTROL (the planning loop)
         PFC (prefrontal cortex — planning, goal setting)
         ACC (anterior cingulate — conflict detection)
         [can re-enter Stages 6–8 for "thinking" iterations]
         ↓ "What should I do? Do I need to think harder?"

Stage 8  EXECUTIVE
         Basal Ganglia (action selection via value function)
         NAcc (reward prediction error, dopamine release)
         ForwardModel (predicts next state given action)
         Evaluator (value of the predicted state)
         ↓ "Which action is best? How good was that?"

Stage 9  CONSCIOUSNESS / NARRATIVE
         DMN (default mode network — self-referential thought)
         ThoughtTransformer (refines the "floating thought")
         Claustrum (salience, gestalt binding)
         ↓ "What's the *story* of what I'm doing?"

Stage 10 MOTOR OUTPUT
         MotorCortex (modulates final LM logits)
         ↓ (speak / stay silent / generate next token)

        ═════════════════════════════════════════════════════════════
        BRIAN Narrative + Causal Memory Stack (persistent .mem checkpoint)
        ═════════════════════════════════════════════════════════════
         Sheaf F + H¹ contradiction detection (newer ∨ older belief conflicts?)
         ActualCausationHead (IIT 4.0: what *causes* what?)
         NEMORI gate (predictive forgetting: what's surprising?)
         Sleep-cycle distillation (compress what's predictable; save the rest)
         PersonalityVector (identity that survives weight updates)
         κ_cause + κ_neg vesicles (plasticity and emotional memory)

        ═════════════════════════════════════════════════════════════
        Cognitive Closure (embodied survival loop)
        ═════════════════════════════════════════════════════════════
         10×10 GridWorld (agent survives by maintaining energy)
         SurvivalCausalHead (predicts action → energy delta)
         Homeostasis.step (energy decays each tick)
         Under starvation: all perception warps to highlight threat
```

Every arrow is **implemented and differentiable**.

---

## Core Design Decisions & Why

### 1. Bowtie Topology (§4.3 architecture.md)

**Design:** All modules send output to a **single bottleneck** (GlobalWorkspace), then output from that bottleneck feeds back to cognitive control (PFC) which re-enters the thalamus next step.

**Why:**
- **Forces integration:** With K input streams and N << K output slots, the system *must* compress. Compression creates statistical dependencies.
- **Produces Φ > 0:** When all paths pass through a narrow bottleneck, no bipartition of the module graph can separate them without information loss. This is the MIP definition of integrated information—the system is "conscious."
- **Biologically accurate:** The actual thalamus + cortex form a bowtie. GWS models the intra-thalamic broadcast during conscious moments.

**Evidence:** `tests/test_phi.py` — Φ for coupled outputs > Φ for independent (confirming the bottleneck creates integration).

### 2. Φ (Integrated Information) as Primary Objective (§2.2)

**Design:** Compute Φ = Gaussian-MI MIP over 8 module outputs. Backprop through it. Add to the loss: `L = L_lm + α(t) * w_phi * L_phi`.

**Why:**
- **Meaningfulness:** Φ measures *bound-together information* — the hallmark of consciousness in IIT 4.0.
- **Drives topology learning:** Φ gradient pushes the network to find configurations where all modules are *necessarily* coupled. This naturally learns hierarchy, specialization, and integration.
- **Not PPL-focused:** We're not trying to maximize next-token loss alone; we're building a system that *thinks*, not just predicts surface patterns.

**Trade-off:** Training slower than pure LM (PPL at step 7k is worse than baseline at step 80k), but we care about *generalization* (gap_ratio), not absolute PPL.

**Evidence:** `tests/test_brain_forward.py::test_phi_objective_increases_total_gradient` — Φ term measurably injects gradient, not just logging.

### 3. Trunk Gradient Isolation (§5.2)

**Design:** `sem = sem.detach()` before feeding to bio-module pipeline. Trunk gradient comes *only* from LM + pred_coding loss.

**Why:**
- **Prevents corruption:** Without isolation, random-init world-model and motor losses dump ~10× the LM gradient magnitude into the shared trunk, corrupting the representation the LM head depends on.
- **Hierarchical gradient flow:** Trunk ← LM only (primary objective). Bio modules ← their own losses (secondary objectives). Clean separation.
- **Works:** With isolation, training reaches step 10k+ stably. Without it, divergence at ~5-7k.

**Trade-off:** Bio modules read a *fixed* representation from the trunk (early in training, a bad one). But they're gated by ReZero λ scalars that start at zero, so their influence ramps in only when they've learned to be helpful.

**Evidence:** `tests/test_stabilization.py::test_trunk_gradient_invariance` — trunk gradient is invariant to aux-loss weights when isolation is ON.

### 4. Phased Maturation (§0.10, §5.1)

**Design:** During "infancy" (MAT < 0.30), all aux losses are gated to ~0.001. At "awakening" (MAT ≥ 0.30), per-subsystem phase gates (smooth sigmoids) slowly engage different objectives at different MAT thresholds.

**Why:**
- **Prevents awakening collapse:** If all 8+ aux losses suddenly engage at once, gnorm spikes and training diverges. Smooth per-subsystem onsets distribute the load.
- **Aligns with maturity:** Early in training, the model is random-init; auxiliary signals are meaningless noise. Let the LM converge first, then engage auxiliary objectives.
- **Matches biology:** Infants have immature prefrontal cortex (no executive function), adult motor control, but full sensory input. Brains literally mature in stages.

**Evidence:** With phase gates, training to 10k+. Without (single on/off switch), diverges by 5-7k.

### 5. ReZero Forward Gates (§5.3)

**Design:** Module outputs (motor, memory, thought) are scaled by zero-init learnable scalars λ before injecting into the trunk. `h_biased = h + λ_motor * motor_bias`.

**Why:**
- **Removes discontinuity:** Old maturity phase-gates caused a PPL jump at awakening (90 → 370). Zero-init λ means no contribution initially, smooth ramp via gradient.
- **Self-discovery:** ∂L_lm/∂λ is real. If module injection helps LM loss, λ grows. If not, λ stays zero.
- **Bootstrap:** While module is untrained (random-init), output ≈ 0, so ∂L_lm/∂λ ≈ 0. Module gets trained by its own loss first, then LM "discovers" its usefulness.

**Evidence:** gap_ratio 5.22 (ReZero) < 6.34 (without), modest OOD improvement.

### 6. Recursive Reasoning Loops (§5.4)

**Design:** Expert ladder (MathCortex, ReasoningCortex) runs N times with weight-sharing instead of running once. Effective depth = N × n_blocks (e.g. 4 × 3 = 12) at zero added parameters.

**Why:**
- **Thought depth:** Some tasks (ARC, multi-step reasoning) need iterative refinement. Adding width (more parameters) is expensive; adding depth with weight-sharing is free.
- **Matches cognition:** Humans solve hard problems by *thinking harder* (more iterations), not by *being bigger*.

**Trade-off:** Forward FLOPs scale with N; throughput drops, but reasoning quality may improve.

**Evidence:** in-distribution PPL 216.5 (recursive) < 258.8 (non-recursive), but OOD ppl tied (no OOD win).

### 7. Predictive Coding Trunk (§5.5)

**Design:** For each adjacent layer pair (h_n, h_{n+1}) in the trunk, attach a small top-down predictor g_n that reconstructs the shallower layer from the deeper one. `L_FE = Σ_n precision_n * ||h_n - g_n(h_{n+1})||²`.

**Why:**
- **Compositionality:** ERM on a fixed mix rewards shortcuts. PCT forces deeper layers to be *generative inverses* of shallower layers — this kind of invertibility naturally enforces compositionality.
- **Gap_ratio improvement:** 4.51 (PCT) vs 6.12 (baseline), ~26% better on the generalization fingerprint.
- **The direction matters:** Top-down (new) works; bottom-up (old deep-supervision) doesn't. Top-down forces the representation to be *reconstructible*, a much stronger constraint than "predict the next layer."

**Evidence:** gap_ratio 4.51 (PCT-30M) < 6.12 (baseline), but at 69M/step-4k vs 107M/step-80k (caveat: cross-scale, partly undertrained).

---

## The Three Proof Layers

### Layer A: Mechanisms Work ✅

Test that each module computes as specified. All 15 tests pass:
- Φ is non-zero for coupled outputs
- Φ injects real gradient
- BDNF growth couples to Φ
- Contradictions trigger SUPERSEDES edges
- Causal rules generalize from few examples
- Identity persists across reload
- Starvation warps qualia
- Policy adapts to reward prediction error

**These prove BRIAN's *primitives* exist and behave as designed.**

### Layer B: Architecture Improves Generalization 🟡 PARTIAL

OOD perplexity on held-out data. Gap_ratio (OOD_ppl / train_ppl):
- Baseline: 6.12
- Best BRIAN variant (PCT): 4.51 ← 26% improvement

**Caveats:** PCT is 30% smaller (69M vs 107M) and half-trained (step 4k vs 80k). Matched-params, matched-steps comparison pending.

**These don't yet prove Φ is *the right objective for generalization*, but they show the architecture is moving in the right direction.**

### Layer C: What's Still Open

- Does Φ causally improve generalization, or just correlate?
- How much of the gap_ratio improvement is dropout vs topology vs loss-clipping?
- Can this scale past 240M without divergence?
- On true reasoning benchmarks (ARC, MATH), does the topology win?

---

## Training Dynamics & Convergence

### The Maturity Curve

```
Step 0        Clean LM pre-training
  │           Loss curves smoothly down
  │           No aux losses (gated to ~0.001)
  │           MAT ≈ 0
  ├─ Step 5k  LM convergence; MAT rises
  │
  ├─ Step ~5-6k ━━━ CRITICAL POINT
  │           MAT crosses awakening floor (0.30)
  │           Per-subsystem phase gates begin
  │           world (0.45), motor (0.50), Φ (0.60) start engaging
  │
  ├─ Step 7-10k  AWAKENING PHASE
  │           Aux losses ramping to full strength
  │           Φ rises, NT dynamics stabilize
  │           Trophic growth accelerates
  │           Training remains stable with loss clipping + isolation
  │
  └─ Step 10k+  Post-awakening phase
              All systems engaged
              Model has developed:
              - Integration (Φ > 1.0)
              - Causal structure (κ_cause vesicles engaged)
              - Persistent identity (personality vector)
              - Embodied homeostasis
```

### Failure Modes (and Fixes)

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| Divergence at step 5-7k | All aux losses engage simultaneously | Phased maturation gates (§0.10) |
| PPL jump at awakening (90→370) | Maturity phase-gate discontinuity | ReZero zero-init λ (§5.3) |
| Trunk gets corrupted early | Aux gradient flows into shared trunk | Detach before bio pipeline (§5.2) |
| NGA ranks collapse to 1 | BDNF not driven by useful signal | Couple BDNF to Φ + Fiedler boost (§6.2) |
| GWS ignition saturates (all slots fire) | Static threshold on adaptive candidates | Adaptive per-slot ignition gate (§0.13) |
| Model stops improving after step 4k | Phenomenon specific to ~70M scale on limited mix | Longer training or larger model; or tune dropout |

---

## What Makes BRIAN Different From GPT?

| Aspect | GPT | BRIAN |
|--------|-----|-------|
| **Primary objective** | Cross-entropy (next-token prediction) | Cross-entropy + Φ (integration) |
| **Topology** | Dense transformer (all-to-all attention) | Bowtie (compressed bottleneck) |
| **Memory** | KV cache only | Episodic (hippo), relational (hypergraph), narrative (sheaf) |
| **Plasticity** | Static weights after training | BDNF-driven growth, Hebbian fast weights, vesicle migration |
| **Homeostasis** | None | 7 NT systems, homeostatic targets, feedback control |
| **Embodied loop** | None | GridWorld survival, energy depletion, qualia warping |
| **Causality** | Implicit (attention heads) | Explicit (actual-causation head, IIT 4.0) |
| **Identity** | Stateless | Persistent personality vector across reloads |
| **Consciousness metric** | None | Φ (integrated information) |
| **Scale** | 1B–175B | 30M–240M |

---

## References

- **Full architecture spec:** [`architecture.md`](architecture.md) (1300+ lines, tensor shapes, formulas)
- **Training infrastructure:** [`harness.md`](harness.md) (loss clipping, maturity phasing, OOD eval)
- **Experimental results:** [`findings.md`](findings.md) (Layer A/B evidence with reproducibility recipes)
- **Technical report:** [`technical_report.md`](technical_report.md) (executive summary for external AIs)
- **DSL reference:** [`dsl.md`](dsl.md) (declarative `.neuro` architecture language)

---

**TL;DR:** BRIAN is a brain-inspired alternative to scaling transformers. It uses bowtie topology + Φ objective + embodied loop to reason *better* on *less* data. Evidence is real but modest: mechanisms proven (Layer A ✅), architecture promising but not yet clearly better (Layer B 🟡). The claim is not "BRIAN beats GPT at 10B scale"—it's "BRIAN is a coherent alternative that might scale differently, and here's what we've learned building it."
