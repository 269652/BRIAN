# RFC — NeuroSLM Architectural Redesign

> **Status:** Draft / proposal. No code changes yet.
> **Goal:** 2–3× efficiency boost while removing param-starvation of expert cortices and giving the model a substantial shared "reading" trunk before expert specialization.
> **Source basis:** `docs/architecture.md` (spec) ↔ live `neuroslm/` source as of branch `master`, 2026-05-18.

---

## 1. Spec ↔ Implementation Crosscheck

| Area | Spec says | Code does | Verdict |
|---|---|---|---|
| GWS Hopfield iters | 2 (§4.3) | `n_iters=3` in `ReasoningCortex` ([reasoning.py:67](../neuroslm/modules/reasoning.py#L67)); GWS itself is OK | Minor doc drift |
| Thalamus streams | 5 StreamAdapters: language/math/reasoning/spatial/social (§4.4) | Exists, BUT these are **not** the `MathCortex`/`ReasoningCortex` modules — they're tiny 2-layer MLPs sharing names | Naming confusion in spec — easy to misread |
| Expert cortex placement | Implied: pre-bowtie expert routing (§4.2) | `MathCortex`/`ReasoningCortex` actually run **after** GWS on `_slots_mean` (B, d_sem) ([brain.py:958-973](../neuroslm/brain.py#L958-L973)) and only modulate `slots`, never logits directly | Spec under-describes; experts are post-hoc residuals, not branches |
| Expert influence on LM | "scale the enrichment" (§6.1) | Enrichment is broadcast back as `slots + 0.2·_slots_mean` then flows to motor → **last token only** via `motor_bias` ([language.py:300-304](../neuroslm/modules/language.py#L300-L304)) | Experts touch only the final logit, not the sequence |
| Expert param share | Implied substantial | `MathCortex` ≈ 4·d_sem² ≈ 0.6M; `ReasoningCortex` ≈ 0.7M. Combined < **0.6%** of xl's 228M | Confirms concern: experts are vestigial |
| BRIAN stack (§10) | ActualCausationHead, PersonalityVector, SleepCycle wired in forward_lm | All present, gated by `_maturation_awakened` ([brain.py:244-255](../neuroslm/brain.py#L244-L255)) | Match |
| Sem pooling | "mean-pooled across B,T" (§3.1) | `sem = self.to_sem(h.mean(dim=1))` ([language.py:307](../neuroslm/modules/language.py#L307)) | Match — but **lossy bottleneck** (see §3-C) |

**Headline finding:** the experts in the spec read as "deep specialized cortices the trunk routes into," but in code they are additive flavor tweaks on a mean-pooled vector that touches **one logit per step**. The trunk *is* the only path that does language modeling, comprehension, and knowledge extraction. The fix isn't to add a trunk before the experts — it's to **promote the experts to be token-level continuations of the trunk** while keeping the trunk dominant in param count.

---

## 2. Proposed Topology — "Shared Reading Cortex + Token-Level Expert Heads" (SRC-TEH)

Two tiers, fully differentiable, no vesicle gating required for the LM path (vesicles become a **modulation** signal, not an on/off switch).

```
              ┌────────────────────────────────────────────────────────┐
              │  TIER 1 — Shared Reading Cortex  (≈ 60% of params)     │
              │  10 deep blocks @ d_hidden=576                          │
              │  • interleaved Standard / DiffAttn / MoD                │
              │  • NeuralGeometryAdapter (reversible) after every block │
              │  • produces:  h ∈ (B, T, 576)                           │
              │              fp ∈ (B, T, 384)  — per-token fingerprint  │
              └──────────────────┬─────────────────────────────────────┘
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
        token-level router (top-1 per token, 3-way; load-balanced; soft during early MAT)
                │                │                │
        ┌───────▼────────┐ ┌─────▼────────┐ ┌─────▼─────────┐
        │ LangExpert     │ │ MathExpert   │ │ ReasonExpert  │
        │ 3 blocks       │ │ 3 blocks +   │ │ 3 blocks +    │
        │ @ d_hidden     │ │ DiffAttn fact│ │ Hopfield bank │
        │ + stylistic    │ │ memory       │ │ + low-rank    │
        │ FFN            │ │ (d=576)      │ │ recurrent     │
        └───────┬────────┘ └─────┬────────┘ └─────┬─────────┘
                └────────────────┼────────────────┘
                       gather → h' ∈ (B, T, 576)
                                 │
                       ┌─────────▼─────────┐
                       │ TIER-1 final norm │
                       │ + tied lm_head    │
                       └─────────┬─────────┘
                                 │
                                 ▼
              bowtie (GWS/PFC/BG/...) consumes attention-pooled fp
              vesicles now MODULATE expert routing instead of GATING cortices
```

### Why this fixes the param starvation

- Tier 1 still owns the LM weight (≈60%). Coherent language, comprehension, world-knowledge all live there.
- Each expert is a *real* transformer-style head at `d_hidden`, not a 2-layer MLP at `d_sem`. Roughly 8-10% each (combined 25-30%).
- The 3 experts together get the param budget that `NeuralGeometryAdapter`s and the proliferating bio modules currently absorb without participating in LM gradient.
- Math/Reasoning now get **token-level gradient on every step**, not just when a vesicle fires.

### xl-sized param budget sketch (target ≈ 240M, isokinetic with today)

| Slot | Current xl | Proposed SRC-TEH |
|---|---|---|
| Token emb (tied) | 25.7M | 28.9M (d_hidden 512→576) |
| Trunk transformer blocks | 12 × ~7M = 84M | 10 × ~10M = 100M (shared) |
| NeuralGeometryAdapter | 12 × ~4M = 48M | 10 × ~2.5M = 25M (reversible, lower rank) |
| MathCortex | 0.6M | 18M (3 blocks, d=576, with fact memory) |
| ReasoningCortex | 0.7M | 18M (3 blocks + Hopfield bank) |
| LanguageExpert (new) | 0 | 18M (style/finishing) |
| Bowtie + bio modules | ~70M | ~30M (consolidate — see §3-G) |
| **Total** | **~228M** | **~238M** |

---

## 3. Novel mechanics (the 2–3× efficiency bets)

Each item: *what*, *where the speedup comes from*, *risk*. Ranked by expected ROI.

### A. Token-Level Expert Routing inside the trunk (the centrepiece)

Top-1 routing per token over {Lang, Math, Reason} after the trunk. Use **expert-choice routing** (Zhou et al. 2022) instead of token-choice: each expert pulls its capacity (T/3 + slack) of the highest-affinity tokens. No load-balancing aux loss needed; no dropped tokens.

- **Speedup:** each expert sees ~⅓ the sequence so wall-clock is ~1.3× faster than running all experts, while *effective* capacity is ~3× bigger than today's residual cortices.
- **Risk:** routing collapse early in training — mitigate with soft top-2 during MAT < 0.5, hard top-1 thereafter.

### B. Reversible NeuralGeometryAdapter

Current adapters store full activations across 12 sites — that's the dominant non-attention activation cost. RevNet-style coupling (Gomez et al. 2017) lets you reconstruct on the backward pass.

- **Speedup:** frees ~30% activation memory → permits 1.5× larger trunk at the same TPU memory budget.
- **Risk:** numerical drift in bf16; require fp32 master copies of the coupling parameters.

### C. Attention-Pool sem instead of mean-pool

`sem = self.to_sem(h.mean(dim=1))` throws away positional structure. Replace with a 1-token learnable query that cross-attends over h with a tiny head (≤0.5M extra params).

- **Speedup:** downstream bowtie quality jumps; Φ has more discriminating module signals so the **Φ loss converges 2–3× faster** in the logged metric. Trickle-down: less time at low-Φ regime → BDNF + trophic system kicks in earlier.
- **Risk:** negligible.

### D. Mid-trunk Bowtie Tap + Lazy Bowtie

The bowtie currently runs *every* forward pass on the *final* hidden state — the most expensive possible placement. Instead:

1. Tap the bowtie input from the **mid-trunk** layer (layer 5 of 10), so the trunk's late layers can integrate the bowtie's output via cross-attention on the upper layers.
2. Run heavy bowtie stages (Hippocampus, HyperGraph, Sleep, ToM) only every K steps (K=4 default; K=1 under high Φ instability). On off-steps, use EMA of last bowtie outputs.

- **Speedup:** ~3× on the bowtie path; trunk now actually **uses** bowtie output instead of just biasing the last logit.
- **Risk:** memory staleness — mitigate by re-running the bowtie on contradiction detection (Sheaf H¹).

### E. Speculative Self-Decoding using own shallow layers

Use Tier 1 layers 1-4 as a draft model; verify with full 10. Standard speculative decoding (Leviathan et al. 2023).

- **Speedup:** 1.8-2.2× inference latency on most prefixes, zero distribution change.
- **Risk:** none (provably equivalent output).

### F. Replace vesicles-as-router with Latent Program Bus

Vesicles are a clever model of neuropeptide signalling but they gate cortices on/off with discrete topic indices and add significant orchestration cost. Replace the *routing* role with a **16-dim latent program token** that the trunk emits, experts read/write, and the trunk reads back next step. (Vesicles can keep their plasticity / structural-growth role.)

- **Speedup:** learned, continuous chain-of-thought channel — iterative reasoning depth without extra forward passes. Often the largest single quality lift in similar work (Hao et al. 2024 "Coconut").
- **Risk:** training stability — initialize the read-back projection to zero so trunk starts identical to today's behaviour.

### G. Bowtie consolidation — fewer, bigger modules

There are ~40 named brain modules; many overlap functionally (HyperGraph + RelationalMemoryGraph + ConsolidatedMemory + Episodic; CorticalSheet + EntorhinalCortex + Claustrum; Insula + Amygdala + ACC + LHb). Param spend per useful gradient signal is low.

- **Speedup:** merging into ≤12 modules with shared embeddings frees ~40M params for §A's experts and ~25% wall-clock per step.
- **Risk:** large cross-file refactor; do last.

### H. Φ-gated MoD capacity (replace fixed `mod_capacity=0.8`)

Make capacity adaptive: `ρ = clip(0.3 + 0.7 · (1 − Φ_stable), 0.3, 1.0)`. When Φ is stable & high, fewer tokens need full attention; when Φ is collapsing, push capacity up.

- **Speedup:** drop avg FLOPs ~20% with no quality regression.
- **Risk:** none, conservative bounds.

### I. Memory as Attention Keys (retrieval-augmented Tier 1)

Today Hippocampus retrieval is a separate stage that feeds the GWS. Instead, expose the top-N consolidated memory entries directly as extra K/V rows in the *last two trunk layers* (RETRO / Memformer style). Each retrieved fact gets used by every token's attention.

- **Speedup:** saves the bowtie a stage and gives the trunk direct episodic recall — typically a 1.5-2× factual-recall gain.
- **Risk:** KV-cache management complexity; cap retrieved set at 64 entries.

### J. Token-level Φ proxy for early exit

CALM currently exits on per-token confidence. Add a per-token Φ contribution estimate (cheap: norm of the token's slot-projected representation). Exit when **both** confidence is high *and* this token isn't carrying Φ-load.

- **Speedup:** lifts CALM's 30-50% savings closer to 60% with no quality loss.
- **Risk:** none.

---

## 4. Combined efficiency targets

Summed expected wins (multiplicative where independent, additive where overlapping):

| Axis | Today (xl baseline) | With SRC-TEH + A-J |
|---|---|---|
| Training wall-clock / step | 1.0× | **2.5–3.0×** faster |
| Activation memory at fixed batch | 1.0× | **0.6–0.7×** |
| Effective inference latency | 1.0× | **2.0–2.5×** faster |
| Comprehension/LM quality at fixed params | baseline | **+15–25%** (token-level expert routing + retrieval-augmented trunk + better sem pool) |

---

## 5. Recommended sequencing

1. **A + C + I** together — this is the SRC-TEH topology change; expect the biggest comprehension/LM jump and lets us measure the rest against the new baseline.
2. **B** — reversible adapters, low-risk memory win.
3. **D + F** — bowtie lazy-tap + latent program bus; biggest wall-clock win on the full forward.
4. **E + H + J** — pure inference/efficiency.
5. **G** — last, because it's a giant cross-file refactor and §1-4 items don't depend on it.

---

## 6. Open questions

- Should `LangExpert` (new) absorb the current `from_sem` thought-conditioning, or keep that on Tier 1?
- For §A token routing, do we keep MoD inside the trunk (compute-shape diversity) or rely entirely on token-level expert routing to handle dynamic compute?
- The vesicle system's *plasticity* role (κ_cause, κ_neg, BDNF amplification) is still load-bearing — confirm §F only retires the *routing* role.
- xxl preset assumes 32 lang_layers; SRC-TEH at xxl scale would be ~24 trunk + 6 per expert. Worth designing now so we don't paint ourselves into a corner.

---

*Authored 2026-05-18. Discussion: open. No code changes pending until a slice is approved.*
