# NeuroSLM — Architecture Reference

> Small language model (~100–350M params) whose architecture directly mirrors
> known neuroscience rather than being inspired by scaling laws.  Every module
> corresponds to a named brain structure with a documented biological function.

---

## Design Philosophy

NeuroSLM is built on three hypotheses:

1. **Topology matters more than scale** — a 250M model with the right
   computational graph (feedback, gating, memory consolidation) can match or
   exceed a 1B vanilla transformer on reasoning and few-shot tasks.
2. **Consciousness-like properties are trainable** — measurable proxies for
   integrated information (Φ), global workspace broadcast, and predictive coding
   can be embedded in training objectives and architectural constraints.
3. **Neurochemistry is a hyperparameter** — learned neuromodulator levels
   (DA, NE, 5HT, ACh) act as global gain signals that dynamically re-weight
   attention, memory writes, and learning rates without adding parameters per
   decision.

---

## Size Presets

| Preset | Params | GPU | VRAM | d_hidden | lang_layers | lang_ctx |
|--------|--------|-----|------|----------|-------------|----------|
| `tiny`   | ~5M   | CPU | —    | 192      | 2           | 256      |
| `small`  | ~15M  | CPU | —    | 384      | 4           | 512      |
| `medium` | ~80M  | T4  | 16GB | 768      | 8           | 1024     |
| `large`  | ~100M | T4  | 15GB | 384      | 8           | 1024     |
| `xl`     | ~258M | A100| 40GB | 512      | 12          | 2048     |
| `xxl`    | ~10B  | 4×A100|320GB| 4096   | 32          | 4096     |

All presets share the same module topology.  `neural_topology='baseline'`
reduces to vanilla transformer only (used for ablations).

---

## Top-Level Module Map

### Core cortical areas

| Attribute | Brain analog | File |
|-----------|-------------|------|
| `language` | Wernicke + Broca (language cortex) | `modules/language.py` |
| `sensory` | Primary sensory cortex + superior colliculus | `modules/sensory.py` |
| `association` | Multimodal association cortex | `modules/association.py` |
| `thalamus` | Thalamic relay + sensory gating | `modules/thalamus.py` |
| `cortical_sheet` | Cortical columns + minicolumns | `modules/cortical_column.py` |
| `entorhinal` | Entorhinal cortex / grid cells | `modules/entorhinal.py` |
| `neural_geometry` | Meta-trainable manifold reshaping | `modules/neural_geometry.py` |

### State models

| Attribute | Brain analog | File |
|-----------|-------------|------|
| `world` | Parietal / posterior cortex (or RSSM) | `modules/world_model.py` |
| `self_m` | Insula / TPJ (self-model) | `modules/self_model.py` |
| `forward_m` | Cerebellum (efference copy / prediction) | `modules/forward_model.py` |

### Global workspace & integration

| Attribute | Brain analog | File |
|-----------|-------------|------|
| `gws` | Frontoparietal global workspace | `modules/workspace.py` |
| `claustrum` | Claustrum (cross-modal binding / consciousness relay) | `modules/claustrum.py` |
| `thought_transformer` | Sustained recurrent thought / working memory | `modules/thought_transformer.py` |
| `qualia` | Phenomenal state representation | `modules/qualia.py` |
| `consciousness` | ConsciousnessMetrics (Φ, causal density, etc.) | `modules/consciousness.py` |

### Memory systems

| Attribute | Biological role | File |
|-----------|----------------|------|
| `hippo` | Dentate gyrus / CA3 / CA1 (fast episodic binding) | `modules/hippocampus.py` |
| `episodic` | Short-term episodic buffer | `memory/episodic.py` |
| `consolidated` | Long-term semantic / schema memory | `memory/consolidated.py` |
| `relational_memory` | Relational memory graph (knowledge graph) | `memory/relational_graph.py` |
| `hypergraph` | N-ary hyperedge memory (social + conceptual) | `memory/hypergraph.py` |
| `entity_store` | Per-entity style fingerprints + belief profiles | `memory/entity_store.py` |
| `causal` | Causal rule store (if A then B) | `memory/causal.py` |
| `narrative_system` | Narrative arc tracking (story coherence) | `memory/narrative.py` |
| `comprehension_gate` | Gated write filter (only surprising insights) | `memory/comprehension_gate.py` |
| `consolidator` | Episodic → semantic consolidation (runs every N steps) | `memory/consolidation.py` |
| `hippocampal` | Hippocampal enrichment layer | `memory/hippocampal.py` |
| `mesolimbic_tagger` | Reward / valence tagging for memories | `memory/mesolimbic.py` |

### Cognitive control

| Attribute | Brain analog | File |
|-----------|-------------|------|
| `pfc` | Dorsolateral PFC (working memory / executive) | `modules/pfc.py` |
| `dmn` | Default Mode Network (mind-wandering / self-referential) | `modules/dmn.py` |
| `bg` | Basal ganglia — striatal Go/NoGo action selection | `modules/basal_ganglia.py` |
| `evaluator` | ACC / OFC (value estimation) | `modules/evaluator.py` |
| `motor` | Primary motor cortex (action → token conditioning) | `modules/motor.py` |
| `critic` | Subconscious value critic | `modules/critic.py` |

### Emotional / subcortical

| Attribute | Brain analog | File |
|-----------|-------------|------|
| `amygdala` | Amygdala (fear conditioning, emotional tagging) | `modules/amygdala.py` |
| `acc` | Anterior cingulate cortex (conflict monitoring) | `modules/anterior_cingulate.py` |
| `insula` | Insula (interoception, gut feelings) | `modules/insula.py` |
| `lhb` | Lateral habenula (anti-reward, aversion learning) | `neurochem/lateral_habenula.py` |
| `cerebellum` | Cerebellum (prediction error, motor learning) | `modules/cerebellum.py` |

### Novel cognitive / ML modules (opt-in per config)

| Attribute | Mechanism | File |
|-----------|-----------|------|
| `tom` | Theory of Mind (belief/desire/intent, social prediction) | `modules/theory_of_mind.py` |
| `active_inference` | Free Energy Principle / canonical Friston predictive hierarchy | `intelligence/active_inference.py` |
| `vesicle_pool` | Neuro-vesicle content packets (migration, docking, degradation) | `neurochem/vesicles.py` |
| `active_dendrite` | Dendritic computation (context-dependent gating) | `modules/active_dendrite.py` |
| `dynamic_routing_moe` | Dynamic routing mixture-of-experts | `modules/dynamic_routing_moe.py` |
| `htm` | Hierarchical Temporal Memory (sparse temporal coding) | `modules/htm_layer.py` |
| `relational_attn` | Relational attention over memory graph | `modules/relational_attention.py` |
| `fast_weight` | Fast-weight associative memory (in-context learning) | `modules/fast_weight.py` |
| `diff_memory` | Differentiable external memory (NTM-style) | `modules/differentiable_memory.py` |
| `phase_attn` | Phase-modulated attention (oscillation-gated) | `modules/phase_modulated_attention.py` |
| `neurogenesis` | Dynamic neuron growth (adaptive capacity) | `modules/neurogenesis.py` |
| `pred_coding` | Inter-layer predictive coding loss | `modules/predictive_coding_loss.py` |
| `causal_module` | Causal inference (intervention / counterfactual) | `modules/causal_inference.py` |
| `cpc` | Contrastive predictive coding loss | `intelligence/contrastive_predictive_coding.py` |

---

## Language Cortex

`modules/language.py` — the token-in / logits-out backbone.

### Transformer stack

Each layer is one of three block types, interleaved:

- **`DiffTransformerBlock`** (Differential Attention) — uses two parallel
  softmax attention maps subtracted from each other.  This cancels common-mode
  noise and amplifies signal, effectively doubling SNR without extra params.
  Derived from Microsoft's 2024 Differential Transformer paper.

- **`MoDBlock`** (Mixture of Depths + CALM) — each token independently decides
  whether to skip the current layer or route through it.  A 2-layer MLP router
  assigns a per-token capacity score; only the top-k% tokens execute the full
  FFN.  Unused tokens get a residual passthrough.  Each `MoDBlock` also carries
  a **`CALMHead`** (Confident Adaptive Language Modeling, Schuster 2022): a
  small 2-layer MLP that estimates per-token confidence `∈ [0,1]` at every
  layer during inference.  Tokens whose confidence exceeds a layer-dependent
  threshold `θ_l = θ_base × exp(−decay × l/(L−1))` are frozen and skip all
  remaining layers — giving full-depth computation to hard tokens and early exit
  to easy tokens.  The threshold decays with depth: shallow layers are strict
  (almost never exit); deep layers are lenient (uncertain tokens still have a
  chance to resolve).  CALM operates inference-only and does not affect training
  gradients.

- **`TransformerBlock`** — standard pre-norm transformer block with RMSNorm +
  causal multi-head attention + SwiGLU FFN.  Used in `baseline` topology and
  as fallback.

### Neuroscience additions inside each block

- **NT-modulated attention temperature** — each attention head's softmax
  temperature is shifted by the current DA and NE neuromodulator levels.
  High NE → sharper attention (focused arousal); high DA → softer, exploratory.

- **Hebbian fast-weight traces** — a low-rank outer-product update to a
  fast-weight matrix after each token (implements Ba et al. 2016).
  `hebbian_rank` controls the rank (0 = disabled, 4–8 for xl).

- **Inter-layer predictive coding** — each layer generates a prediction of the
  next layer's output.  The prediction error is an auxiliary loss term
  (Whittington & Bogacz 2017), providing deep supervision that guides
  representations to be predictive at every level of abstraction.

- **`PredictiveCodingHead`** — inside `neuro_attention.py`, handles the
  per-layer residual prediction and error computation.

- **`NeuralGeometryAdapter`** — inserted after every transformer block,
  projects `d_hidden → 2×d_hidden` (hyperbolic-like expansion space), applies
  a learned low-rank connectivity kernel `kern_a @ kern_b` (virtual neural
  wiring that does not exist in the base transformer topology), then gates with
  a per-dimension sigmoid and projects back.  Zero-init on the down projection
  and `bias=-2` on the gate ensure it starts as a strict identity; the geometry
  of the activation manifold is discovered entirely during training.  Parameters
  are included in the meta-training set so the wiring topology itself is
  meta-learned.

### Outputs

`language(ids, thought, nt)` returns `(logits, sem, h, pred_coding_loss)`

- `logits`: `(B, T, vocab_size)` — next-token prediction
- `sem`: `(B, d_sem)` — comprehension embedding (last-position, projected)
- `h`: `(B, T, d_hidden)` — full hidden state sequence
- `pred_coding_loss`: scalar auxiliary loss from predictive coding heads

---

## Sensory & Association Pipeline

```
tokens → LanguageCortex → sem
sem   → TextSensoryCortex  (salience gating, novelty detection)
      → AssociationCortex  (multimodal fusion placeholder)
      → Thalamus           (relay + gating by NE levels)
```

`Thalamus` implements attentional gating: high NE → pass-through; low NE →
attenuate signal (sleep-like suppression).

---

## State Models

### WorldModel

Two implementations selectable via `enable_rssm`:

**Standard** — recurrent GRU-based world model.  Takes `(sem, action)` →
predicts next world state `h_world`.  Loss: MSE against actual next `sem`.

**RSSM** (Recurrent State Space Model, à la DreamerV3) — deterministic GRU
recurrence + stochastic categorical latent variables (`n_cats × d_cat`).
Posterior: encoder(`sem`) → categorical logits.
Prior: predicted from deterministic state.
Loss: KL divergence prior vs. posterior (straight-through Gumbel-Softmax).
Enables latent imagination rollouts without real tokens.

### SelfModel

Models the agent's own state: combines `sem`, last action vector, and NT
levels → predicts future self-state.  Analogous to the insula / TPJ
interoceptive self-model.

---

## Global Workspace (GWS)

`modules/workspace.py` — implements Baars / Dehaene Global Workspace Theory
via Modern Hopfield Networks (Ramsauer 2020) with Dehaene ignition dynamics.

### Hopfield iterative convergence

The attention mechanism is the Hopfield update rule:

```
slot^{t+1} = softmax(β × C × slot^t^T) × C
```

where `C` is the candidate set and `β = softplus(log_beta) + 0.5` is a
learned inverse temperature.  Iterating `hopfield_iters=2` times converges
toward the energy minimum — the attractor closest to each slot's initial query.
This corresponds to pattern completion, not merely pattern selection.

### Ignition phase transition (Dehaene 2011)

After Hopfield convergence:

1. **Lateral competition** — off-diagonal cosine similarity between slots drives
   15% inhibition of redundant patterns, ensuring each slot captures a distinct
   representation (winner-take-all in feature space).
2. **Ignition gate** — `activity = mean L2 norm of slots`.
   `ign_prob = sigmoid((activity − θ) × 4)`, giving a sharp phase transition
   around threshold `θ = 0.5`.
   `broadcast_scale = 0.3 + 0.7 × ign_prob`:
   pre-ignition ≈ 0.3 (local, sparse), post-ignition ≈ 1.0 (global broadcast).
3. **Per-slot learned scale** — `output_scale ∈ R^{n_slots}` (init 1.0).

The `_last_ignition` probability is detached and logged each step.

The Claustrum (`modules/claustrum.py`) acts as a binding relay: it cross-
attends over 8 modality streams and outputs a single binding signal added to
the GWS output.

---

## Memory Architecture

### Hippocampus (fast binding)

`modules/hippocampus.py` — sparse key-value episodic memory implementing
Complementary Learning Systems (CLS) theory with five recall streams.

#### Architecture

| Sub-region | Role | Implementation |
|------------|------|----------------|
| DG (Dentate Gyrus) | Sparse pattern separation / orthogonalisation | Fixed random projection → top-k winners → back-project |
| CA3 | Auto-associative pattern completion | Learned query/key/value attention over stored memories |
| CA1 | Mismatch / novelty detection | MLP(`expected`, `actual`) → scalar novelty ∈ [0,1] |

#### Multi-dimensional recall (five streams)

All streams are fused before GWS integration via cross-attention:

1. **Semantic** — cosine k-NN via chunked approximate kNN (see below).
2. **Temporal** — recency-weighted retrieval; write-order timestamps boost recent memories by +0.4 in the scoring.
3. **Mood/emotional** — weighted combination of semantic similarity (0.4), NT-state cosine similarity (0.4), and valence distance (0.2).  Implements mood-congruent memory bias.
4. **Associative chain** — 2-hop attention chaining: the best semantic recall becomes the query for a second retrieval, surfacing narratively linked memories.
5. **DNC temporal traversal** — forward write-order chains via the DNC link matrix `L` (see below).

#### DNC Temporal Link Matrix (Graves 2016)

`_dnc_L[i,j] ∈ [0,1]` approximates the probability that slot `j` was written
immediately before slot `i`.  Updated at every `store()` call:

```
outer = w_w[:, None] × p[None, :]         # current write × precedence
decay = (1 − w_w[:, None] − w_w[None, :]).clamp(0)
L     = decay × L + outer                  # update
L.fill_diagonal_(0)                        # no self-links
p     = (1 − Σw_w) × p + w_w             # precedence update
```

`_dnc_temporal_recall(query, topk, direction)` traverses write-order chains:

- `direction="forward"` → `w_blend = 0.3×w_r + 0.7×(w_r @ L^T)` — recalls what came *after* the best match (procedure steps, story continuation).
- `direction="backward"` → `w_blend = 0.3×w_r + 0.7×(w_r @ L)` — recalls what came *before* (causal antecedents).

#### Chunked approximate kNN

Direct `q @ K.T` with `N=65K` creates an `(B, N)` tensor that OOMs on 40 GB
GPUs.  `_approx_knn(query, keys, topk, chunk=2048)` processes keys in chunks
of 2048, maintaining a running top-k merge at `O(B × 2048 × d)` peak memory.
Scales to 65K+ stored memories without FAISS.

#### Capacity and write policy

- Stores up to `hippo_capacity` (4096–32768) memory vectors in a ring buffer.
- Writes: gated by ComprehensionGate (`lm_loss × comprehension_score`).
- Sparse coding: DG uses fixed orthogonal projection + top-`sparse_k` winners (encoding mode) or `2×sparse_k` winners (retrieval mode).
- ACh-gated: high ACh → stronger memory enrichment (encoding-boosted retrieval).

### Episodic → Semantic Consolidation

Every `consolidate_every` steps (default 500):

1. `EpisodicMemory` (ring buffer, 2048 entries) clusters recent events.
2. Cluster centroids become nodes in `ConsolidatedMemory` (semantic schemas).
3. `CausalRuleStore` extracts co-occurrence rules: if event A precedes B at
   rate > `min_support`, create rule A→B with confidence score.
4. `RelationalMemoryGraph` stores (subject, predicate, object) triples as a
   knowledge graph (max 8192 nodes).
5. After consolidation, high-surprise observations also written as semantic
   insight nodes in the relational graph.

### HyperGraph Memory

`memory/hypergraph.py` — N-ary hyperedges beyond binary relations.

- Stores arbitrary-arity hyperedges (e.g. `(agent, action, object, context,
  time)`).
- Dirichlet-Markov prior for social belief updating (tracks "who believes
  what" across entities).
- Deduplication via embedding hash; entity subgraph extraction.

### Entity Store

`memory/entity_store.py` — per-entity persistent profiles.

- Every named entity (person, place, concept) gets a style fingerprint
  (d_style = 64 dims).
- Tracks: preferences, beliefs, narrative role, last-seen context.
- The `TheoryOfMindModule` reads entity profiles to simulate their mental
  states.

### ComprehensionGate

`memory/comprehension_gate.py` — controls memory write rate.

- Computes a gate value from `lm_loss × comprehension_score`.
- Only observations that are both surprising (high loss) and understood
  (high comprehension) get written.
- Self-calibrating: adjusts threshold to maintain `target_write_rate = 10%`.

---

## Neurochemistry System

`neuroslm/neurochem/` — full neurotransmitter simulation.

### Neuromodulators tracked

| Symbol | Nucleus | Primary roles in model |
|--------|---------|------------------------|
| DA     | VTA + SNc | Reward prediction error, attention sharpness, BG action gating |
| NE     | Locus Coeruleus | Arousal, thalamic relay gain, attention focus |
| 5HT    | Raphe nuclei | Mood baseline, DMN suppression, patience / temporal discounting |
| ACh    | Nucleus Basalis Meynert | Memory encoding gain, PFC working memory, attention demand |
| Glu    | Cortical projections | Excitatory drive (PFC → BG) |
| GABA   | BG → Thalamus | Inhibitory gating of motor output |
| eCB    | Endocannabinoid | Language cortex disinhibition, memory erasure |

### Receptor banks

Each brain region has a `ReceptorBank` with signed receptor weights:

```
rcpt_pfc:   DA(+0.6), 5HT(+0.3), ACh(+0.4), GABA(-0.4)
rcpt_hippo: ACh(+0.5), Glu(+0.4)
rcpt_bg:    DA(+0.7), GABA(-0.5)
rcpt_thal:  NE(+0.5), GABA(-0.3)
rcpt_lang:  ACh(+0.3), eCB(-0.3)
rcpt_dmn:   5HT(-0.4), ACh(-0.2)
```

Receptor modulation applies to the embedding before it enters each region,
scaling the feature magnitudes by `1 + Σ(receptor_weight × NT_level)`.

### NT release pipeline

Each forward pass:

1. `VTA`, `LC`, `Raphe`, `NBM` nuclei compute NT demand from signals
   (novelty, reward-prediction-error, arousal, attention demand, mood).
2. `ProjectionGraph` releases NT along anatomically-correct pathways
   (VTA→NAcc, SNc→BG, LC→PFC, NBM→Hippo, etc.).
3. `ReuptakeSystem` decays NT levels each step.
4. `ReceptorAdaptation` up/down-regulates receptor sensitivity after
   sustained NT elevation (tolerance / sensitisation).
5. `PlasticityGate` modulates learning rate based on NT milieu.
6. `TrophicSystem` models axon growth / pruning based on co-activation.
7. `LateralHabenula` drives anti-reward signal: spikes when expected reward
   is not delivered, suppressing DA release (learned aversion).

### VesiclePool (neuromodulatory content packets)

`neurochem/vesicles.py` — discrete synaptic-vesicle-like packets providing
slow, long-range neuromodulation complementary to the fast NT scalar signals.

Each vesicle carries:
- **content vector** `c ∈ R^{d_sem}` — semantic payload
- **lifetime τ** — ticks until degradation
- **position** — which of the `n_modules` brain modules the vesicle is at
- **active flag** — whether the vesicle is alive

#### Life cycle (one tick per `cognitive_step`)

| Phase | Mechanism |
|-------|-----------|
| **Synthesis** | High-surprise events (`novelty × floating_thought`) create new vesicles via a learned synthesis gate MLP; vesicle content encodes the surprise signal. |
| **Migration** | Each active vesicle stochastically diffuses to a new module via a learned row-stochastic transition matrix `T = softmax(log_T)` — the migration dynamics are themselves trained. |
| **Docking** | Cosine attention between vesicle content key and module activation query produces a dock score ∈ [0,1]; docked content is projected to a modulation delta `Δ ∈ R^{d_sem}`. |
| **Degradation** | Lifetime decreases by 1 each tick; vesicles at τ ≤ 0 die and their slot becomes available for new synthesis. |

The modulation output `(B, n_modules, d_sem)` is applied to GWS slots after
the hippocampal enrichment step.  Enabled via `enable_vesicles=True`
(`n_vesicles=32`, `vesicle_lifetime=16` by default).

---

## Intelligence Layer

`neuroslm/intelligence/`

### NeuralOrchestrator

`intelligence/orchestrator.py` — learned routing across all brain areas.

Implements a staged feed-forward topology:

```
SENSORY → THALAMUS → STATE_MODELS → SUBCORTICAL → QUALIA
→ GWS → MEMORY → COGNITIVE_CTL → EXECUTIVE → CONSCIOUSNESS → MOTOR
```

At each stage, a small attention router decides which registered modules
execute and with what weight.  Gain is tracked per module; stability metrics
(gain mean/std) are logged every step with `.detach()` to avoid leaking
gradients into the metrics path.

### IntelligenceMetrics

`intelligence/metrics.py` — computes at every step:

- `causal_density` — ratio of causal rules to total observations
- `narrative_coherence` — cosine similarity of narrative buffer embeddings
- `memory_utilization` — fraction of episodic buffer occupied
- `consciousness_index` — proxy for Φ (integrated information), computed as
  mutual information between GWS output and individual module outputs
- `identity_drift` — KL divergence of self-model states across time
- `curiosity` — novelty-weighted attention entropy

### SpontaneousReflection

`intelligence/reflection.py` — fires during low-activity windows (DMN
period).  Re-plays recent episodic memories, generates counterfactual
continuations via the forward model, and writes insights to the relational
graph.  Analogous to hippocampal replay during quiet wakefulness.

### NeuralOscillationTracker

`intelligence/oscillations.py` — tracks band-power in 8 regions over a
sliding window of 64 steps.  Outputs delta/theta/alpha/beta/gamma power
estimates.  Used to gate phase-modulated attention and DMN periodicity.

---

## Consciousness-Specific Mechanisms

### QualiaState

`modules/qualia.py` — maintains a phenomenal state vector `q ∈ R^{d_sem}`.
Updated each step as a running average of GWS broadcast weighted by NT
levels.  Represents the "what it is like" quality of the current experience.

### ConsciousnessMetrics

`modules/consciousness.py` — computes measurable proxies each tick:

| Metric | Formula / Source | Meaning |
|--------|-----------------|---------|
| **γ (gamma)** | Mean off-diagonal cosine sim of GWS slots | Binding coherence |
| **θ (theta)** | Mean CA1 novelty signal | Memory retrieval activity |
| **α (alpha)** | Normalised routing entropy | Idling / suppression |
| **Φ (phi)** | MIP lower bound (see below) | Integrated information |
| **Coherence** | Cosine alignment of module outputs with GWS mean | Cross-module synchrony |
| **Ignition** | Fraction of modules with norm > 0.6 | GWS broadcast reach |
| **Metacognition** | `sigmoid(||thought|| − 1)` | Self-monitoring proxy |
| **Binding** | γ × coherence | Phenomenal binding strength |

#### IIT Φ via Minimum Information Partition (MIP)

Φ is estimated as the **minimum mutual information over all bipartitions**
`(A, B)` of the module output set — the weakest link in the system's
integration.  High Φ means no bipartition can disconnect the system without
losing mutual information (IIT postulate 5: irreducibility).

Gaussian MI estimator (tractable, O(n²) per bipartition):

```
MI(A; B) = 0.5 × (log det Σ_A + log det Σ_B − log det Σ_AB)
```

where `Σ` is estimated from the Gram matrix of module output vectors.

Two code paths:

- **n ≤ 8 modules** — `_phi_enumerate`: exhaustive search over all
  `2^(n−1) − 1` bipartitions (≤ 127 partitions); returns exact MIP lower bound.
- **n > 8 modules** — `_phi_spectral`: Fiedler vector of the normalised
  Laplacian provides the spectral bisection cut in `O(n³)`; returns a single
  near-optimal partition.

Φ is clamped to `[0, 10]` and logged to `_phi_history` (rolling 64-tick buffer).

### Theory of Mind (ToM)

`modules/theory_of_mind.py` — enabled for `xxl`; optional for `xl`.

- Maintains a learned belief-desire-intent representation for each tracked
  entity (from `entity_store`).
- Counterfactual simulation: given agent A's style fingerprint, predict what A
  would say/do in the current context.
- Social prediction error: MSE between predicted and actual entity embeddings
  is a training signal (`w_social = 0.1`).

### Active Inference / Free Energy

`intelligence/active_inference.py` — canonical Friston (2017) hierarchical
message passing with explicit superficial/deep pyramidal cell separation.

#### PredictiveLayer — canonical microcircuit

Each layer implements the Bastos et al. (2012) canonical microcircuit:

| Cell type | Variable | Computation |
|-----------|----------|-------------|
| Superficial pyramidal (error units) | `ε^l` | `Π^l ⊙ (x^l − pred^l)` where `Π = exp(log_precision)` |
| Deep pyramidal (state units) | `μ^l` | `recognition(x^l, ε^l)` — integrates evidence + error |
| Generative model (top-down) | `pred^{l-1}` | `generative(μ^l)` — predicts level below; zero-init |

Precision `Π^l` is a learned per-feature vector (inverse variance).
High precision → that feature's prediction error dominates learning.
Zero-initialised generative projection ensures the network starts with
uninformative priors and learns non-trivial predictions from data.

#### Two-pass forward

```
Pass 1 — Bottom-up (recognition):
  for l = 0..L-1:  μ^l = recognition(x^l, Π^l ⊙ (x^l − 0))
                   x^{l+1} = μ^l

Pass 2 — Top-down (generation):
  pred = None   (top layer has no prior from above)
  for l = L-1..0:
    ε^l, pred_down = layer_l(x^l, prior_pred=pred)
    pred = pred_down      ← this layer's prediction for level l−1
```

The top-down pass recomputes all `ε^l` with the generative prior, giving
each layer an error signal that accounts for both bottom-up evidence and
top-down predictions — the defining property of predictive coding.

#### Free energy

```
F = Σ_l ||ε^l||² + 0.05 × Σ_l ||μ^l||²
      accuracy          complexity prior
```

`F` is returned as a scalar auxiliary loss (`w_fe ≈ 0.1`).

#### Value signals

- **Epistemic value** — `EpistemicValueEstimator`: expected reduction in
  posterior uncertainty from taking an action; drives exploration.
- **Pragmatic value** — `pragmatic(μ^L)`: expected reward from current beliefs;
  drives exploitation.
- NT-modulated balance: high DA → pragmatic dominates; high NE → epistemic
  dominates.  Both signals pass to BG action selection.

---

## Epigenetic Genome / DNA System

`neuroslm/dna/`

The genome encodes module *algorithms* — not weights, but the computational
primitives each module uses.

### Components

- **`ModuleGenome`** (`dna/structural_genome.py`) — a tensor of alleles where
  each allele encodes an opcode (attention/memory/gating operation) and
  operands (temperature, learning rate, connectivity).
- **`GenomeCompiler`** (`dna/compiler.py`) — decompiles alleles into a Lisp-
  like DSL, extracts parameter values, pushes them into the corresponding
  module attributes.
- **`EpigeneticOptimizer`** (`dna/epigenetics.py`) — runs a population-based
  evolutionary search over genome alleles.  Fitness = negative validation loss
  + intelligence metric bonuses.  Mutations include: allele substitution,
  transposition, promoter strength change.
- **`LatentProgramEvolver`** (`dna/latent_program.py`) — encodes the genome in
  a continuous latent space for gradient-based evolution.

Genome state is checkpointed in `.pt` files (`module_genomes` key) and exported
as human-readable `.dna.json` snapshots.

---

## Training

### Data pipeline

`neuroslm/data.py` — interleaved streaming from:

- **Text**: FineWeb-Edu, Cosmopedia, TinyStories (10B+ tokens)
- **Chat**: OpenHermes-2.5, UltraChat-200k, WildChat-1M, SlimOrca, hh-rlhf,
  Dolly-15k
- **Mode `mix`**: `chat_ratio` fraction chat, remainder text (default 60/40)

### Loss function

```
L = w_lm       × CE(next-token, chunked over T=128 slices)
  + w_world    × MSE(world state prediction)          [0.3]
  + w_self     × MSE(self-model prediction)           [0.1]
  + w_forward  × MSE(forward model prediction)        [0.2]
  + w_value    × MSE(evaluator value)                 [0.1]
  + w_motor    × CE(action selection)                 [0.05]
  + w_pred_coding × inter-layer prediction error      [0.1]
  + w_cpc      × contrastive predictive coding loss   [0.05, optional]
  + w_kl_world × RSSM KL divergence                  [0.1, if RSSM]
  + w_social   × ToM social prediction error          [0.1, if ToM]
```

### Memory-efficient training

- **Chunked cross-entropy**: computes CE in T=128 token slices, avoids
  materialising `(B × T × vocab)` tensor (825 MB at xl/B=1).
- **Gradient checkpointing**: recomputes activations during backward pass
  instead of caching; saves ~50% activation memory.  Forced on when
  `device == "cuda"`.
- **Gradient accumulation**: `--grad_accum N` runs N micro-batches before
  `optimizer.step()`, giving effective batch = `batch_size × N` at 1/N peak
  activation memory.
- **`del logits`** after computing logits_motor frees one `(B,T,vocab)` tensor
  from the autograd graph before the large CE computation.
- **`torch.cuda.amp`** (mixed precision) with `GradScaler`.

### Optimiser

AdamW with:
- `lr = 2e-4` (xl), cosine decay with `warmup_steps = 800` linear warmup
- `weight_decay = 0.1`, `grad_clip = 1.0`
- NT levels modulate effective LR per-step via `PlasticityGate` (soft
  multiplicative scaling, not a replacement for AdamW).

---

## Inference Pipeline (`Brain.cognitive_step`)

```
token → LanguageCortex (with floating_thought + NT) [CALM early exit at MoDBlocks]
      → Sensory → Association → Thalamus
      → World + Self state update (RSSM if enable_rssm)
      → Amygdala, Insula (emotional colouring)
      → QualiaState update
      → GlobalWorkspace broadcast [Hopfield iters → lateral competition → ignition]
      → Hippocampus recall (5 streams: semantic/temporal/mood/associative/DNC)
          + EntorhinalCortex grid encoding
      → Cerebellum prediction error update
      → PFC thought selection + replace-gate
      → ACC conflict check
      → BasalGanglia action selection (DA-modulated)
      → ForwardModel + Evaluator
      → DMN (every dmn_period steps: spontaneous reflection)
      → ThoughtTransformer + Claustrum
      → ToM update (if enable_tom)
      → RSSM world state (if enable_rssm)
      → ActiveInference / Free Energy (if enable_active_inference)
          [two-pass: bottom-up recognition → top-down error recomputation]
      → VesiclePool.tick() (if enable_vesicles)
          [synthesize from surprise → migrate → dock → degrade]
          → apply vesicle modulation to GWS slots
      → ConsciousnessMetrics snapshot [Φ-MIP, gamma/theta/alpha, ignition]
      → MotorCortex → LanguageCortex.from_sem (action-conditioned token bias)
      → NT release via nuclei + projection graph
      → Episodic memory write (gated by ComprehensionGate)
      → NarrativeSystem update
      → EntityStore update (if entities detected)
      → HyperGraph update (if social hyperedge)
```

`floating_thought` is a running exponential average (`thought_alpha = 0.3`)
of the PFC output — it persists across tokens as a "held thought" that biases
the next language cortex forward pass.

---

## Topology Modes & Ablation

```python
cfg.neural_topology = 'full'      # all modules active (default)
cfg.neural_topology = 'baseline'  # vanilla transformer only

cfg.baseline = True               # builds only LanguageCortex, skips all modules

brain.disable_module('hippo')     # bypass hippocampus at runtime
brain.enable_module('tom')        # re-enable ToM
brain.module_status()             # dict of enabled/disabled per module
```

Per-module `enable_*` flags in `BrainConfig` control construction; the
`BrainModule.enabled` flag controls runtime routing.  A disabled module
returns a neutral zero-tensor passthrough without computing its forward pass.

### Checkpoint persistence

- `.pt` — full model + optimiser + genome state + `_global_step`
- `.mem` — memory checkpoint: episodic buffer, consolidated memory, relational
  graph, causal rules, narrative, entity store
- `.dna.json` — human-readable evolved genome snapshot

All three are written every `save_every` steps and pushed to Git LFS
immediately after creation.

---

## Key Design Decisions (open questions / future work)

- **No RL loop yet** — world/self/forward losses are supervised proxies; a
  real environment or RLHF signal would greatly strengthen them.
- **Neuromodulators are not grounded** — NT levels are initialised from zero
  and shaped only by the training signal.  A real sensory environment
  (hunger, fatigue, social reward) would make them meaningful.
- **Hebbian updates are offline** — fast-weight traces update within the
  forward pass but are not a true online local learning rule; they still
  require backprop.
- **Φ estimation is approximate** — true IIT Φ computation is NP-hard; the
  MIP lower bound via Gaussian mutual information is tractable and principled
  but not the full Φ_max.  For n ≤ 8 modules the enumeration is exact over
  all bipartitions; for n > 8 the Fiedler bisection is a near-optimal heuristic.
- **CALM threshold is not trained** — the base threshold and decay are fixed
  hyperparameters; adaptive calibration (e.g. via policy gradient on exit-vs-
  accuracy trade-off) is left for future work.
- **Vesicle migration is stochastic at inference** — `torch.multinomial` in
  `VesiclePool.migrate()` introduces non-determinism during inference.  For
  reproducible evaluation, seed the RNG or replace with argmax migration.
- **DNC link matrix is dense** — `_dnc_L: (capacity, capacity)` is stored as
  a float32 dense buffer.  At `capacity=4096` this is 64 MB; at 32768 it is
  4 GB.  The xl preset caps capacity at 4096.  Sparse storage (COO or CSR)
  would be needed for larger capacities.
- **Genome evolution is slow** — the epigenetic optimiser requires many
  evaluations; in practice only genome checkpointing is used during training,
  with evolution reserved for post-training search.
