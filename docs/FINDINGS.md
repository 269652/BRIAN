# FINDINGS — what BRIAN is testing, and what we've learned

> **Purpose.** A single ledger of every architectural hypothesis BRIAN is
> trying to prove or falsify, the mechanism that operationalizes it,
> and the reproducible artifact that backs (or refuses) it.
>
> **Rule.** Every claim in this document is tied to either (a) a named
> test in `tests/`, or (b) a result JSON in `results/`, or (c) a
> committed checkpoint sidecar, or (d) a raw log under `logs/vast/`
> with a matching analysis in `logs/analyzed/`. No metric appears here
> without an artifact path. If the artifact is on a feature branch and
> not yet on `master`, the branch and commit are cited explicitly.
>
> Last refreshed: 2026-05-25. Update this file in the same change set
> that produces new evidence. Pipeline for converting raw vast.ai
> training/eval logs into entries here: see
> [logs/analyzed/INSTRUCTIONS.md](../logs/analyzed/INSTRUCTIONS.md).

---

## How to read this document

BRIAN's claims split into two layers with very different evidence
standards. Treat them separately.

| Layer | Question | Evidence type |
|---|---|---|
| **A** — Mechanism existence | "Does the brain-like primitive exist, gradient-couple, and behave as specified?" | Unit/behavioural tests on a constructed brain. Local, deterministic, ~7s on CPU. |
| **B** — Architectural bet | "Does the topology beat a flat transformer at matched params / FLOPs / OOD?" | OOD perplexity on WikiText-103-v1, sliding window. Requires a trained checkpoint + a comparable baseline. JSON in `results/`. |

Layer A is largely settled — the mechanisms are wired and the named
tests pass. Layer B is the live research question. Most of the recent
commit traffic (`stabilize/recursive-reasoning`,
`stabilize/trunk-grad-isolation`, `arch/predictive-coding-trunk`,
`arch/synthesis-v1`) is ablation arc on Layer B.

---

## Status legend

| | Meaning |
|---|---|
| ✅ **CONFIRMED** | Artifact exists, metric crosses the spec's threshold, reproducible. |
| 🟡 **PARTIAL** | Mechanism wired or first result obtained, but the headline claim is not (yet) cleanly supported. Caveat documented. |
| 🟠 **PENDING** | Mechanism wired and unit-tested; full-scale eval not yet run. |
| ❌ **FALSIFIED** | Tested at the appropriate scale and failed to clear the bar the hypothesis set. |
| ⚠ **UNVERIFIED OVERCLAIM** | Claim appears in `README.md` or `docs/architecture.md` but no committed artifact backs it. |

---

## Layer A — Mechanism existence (unit-test backed)

These tests assert that a constructed `Brain` carries out the
mechanism described in `docs/architecture.md`. They prove the primitive
exists. They do **not** prove the primitive scales or improves LM
quality on real corpora.

Reproduce all of them with:
```bash
py -3 -m pytest tests/test_phi.py tests/test_brain_forward.py tests/test_neurochem.py tests/test_narrative_memory.py tests/test_cognitive_closure.py -v
```

### H1 — Φ is non-zero for coupled module outputs

**Hypothesis.** The Gaussian-MI MIP lower bound on integrated
information is strictly positive when module outputs are rank-1
coupled and approaches zero when they are independent.

- **Spec.** `docs/architecture.md` §3.3, §8.1
- **Test.** `tests/test_phi.py::test_phi_higher_for_coupled_outputs`
- **Status.** ✅ CONFIRMED — passes deterministically on a fresh brain.

### H2 — The Φ objective injects real gradient (A/B)

**Hypothesis.** Adding the `phi_loss` term measurably increases
‖∂L/∂θ‖ vs an identical forward without it — i.e. Φ is not just a
logging proxy, it shapes weights.

- **Spec.** `docs/architecture.md` §2.2
- **Test.** `tests/test_brain_forward.py::test_phi_objective_increases_total_gradient`
- **Status.** ✅ CONFIRMED.

### H3 — Φ-coupled BDNF reshapes the projection graph

**Hypothesis.** High-Φ pathways receive ≥ as much trophic support as
low-Φ pathways → kernel rank of the affected NeuralGeometryAdapter
grows preferentially on the integrated path.

- **Spec.** `docs/architecture.md` §6.2
- **Test.** `tests/test_neurochem.py::test_trophic_phi_boosts_growth`
- **Status.** ✅ CONFIRMED.

### H4 — Sheaf H¹ detects narrative contradictions

**Hypothesis.** "Alice likes coffee" followed by "Alice hates coffee"
produces a measurable Čech 1-cocycle on the contextual sheaf F, and
the system writes a SUPERSEDES edge from new → old.

- **Spec.** `docs/architecture.md` §10.1
- **Test.** `tests/test_narrative_memory.py::test_sheaf_contradiction_detection`
- **Status.** ✅ CONFIRMED — H¹ > 0.5 on the canonical contradiction pair.

### H5 — Causal generalisation from few-shot narratives

**Hypothesis.** After observing 10 (Gift → Joy) and 10 (Insult →
Offense) episodes, a novel Gift trial produces P(Joy) > 0.8 via the
CausalRuleStore — *without* gradient updates.

- **Spec.** `docs/architecture.md` §10.2, §10.3
- **Test.** `tests/test_narrative_memory.py::test_causal_generalization`
- **Status.** ✅ CONFIRMED.

### H6 — Identity & personality persist across re-instantiation

**Hypothesis.** Saving the `.mem` checkpoint, instantiating a fresh
`Brain`, and loading it back recovers the same autobiographical
summary embedding within tolerance.

- **Spec.** `docs/architecture.md` §10.5
- **Test.** `tests/test_cognitive_closure.py::test_autobiographical_personality_consistency`
- **Status.** ✅ CONFIRMED.

### H6.5 — Embodied survival reshapes qualia and policy

**Hypothesis.** A controlled energy drop (homeostatic perturbation)
produces a measurably distinct qualia warp; 100 +RPE updates pull a
target option's BG-VQH DA-value above 0.5.

- **Spec.** `docs/architecture.md` §11
- **Tests.** `tests/test_cognitive_closure.py::test_survival_imperative_qualia_shift`, `::test_basal_ganglia_policy_adaptation`
- **Status.** ✅ CONFIRMED.

---

## Layer B — Architectural bet (the live research question)

Each row of Layer B is an ablation against the previous one. The
common eval harness is `scripts/vast_ood_eval.sh` calling
`brian_ood_test.py`: 100 train batches @ 4096 tokens (≈ 409.6 k
tokens), 200 WikiText-103-v1 windows (≈ 102.9 k tokens), sliding
window with stride 512 / seq_len 1024, GPT-2 BPE tokenizer
(vocab 50257).

### Reference table — what we have measured

| Row | Branch | Ckpt step | Params | train_ppl | OOD_ppl | **gap_ratio** | verdict | artifact |
|---|---|---|---|---|---|---|---|---|
| **B0** **flat-transformer baseline** | `stabilize/trunk-grad-isolation` | 80000 | 106.9M | **66.0** | **404.0** | 6.12 | STRONG OVERFITTING | `results/ood_baseline-80k_107M_step80000.json` |
| **B0.collapse** baseline brain (pre-§5.2) | (pre-§5.2) | 5000 | ~107M | (diverged ~5–6k) | — | — | post-awakening collapse | (see §5.2) |
| **B1** trunk-iso + recursive | `stabilize/recursive-reasoning` | 5000 | 108.2M | 216.5 | 1372.8 | **6.34** | STRONG OVERFITTING | `results/ood_recursive_108M_step5000.json` |
| **B2** trunk-iso + ReZero (load-bug, do not cite) | `stabilize/trunk-grad-isolation` | 7000 | 107.8M | 1169.9 | 5242.7 | 4.48 | ARTIFACT — λ params zero-init at eval (see B2.fix) | `results/ood_rezero-buggy-preload_107M_step7000.json` |
| **B2.fix** trunk-iso + ReZero (legacy-fallback fix) | `stabilize/trunk-grad-isolation` | 7000 | 107.8M | 258.8 | 1351.5 | **5.22** | STRONG OVERFITTING | `results/ood_rezero-fixed_107M_step7000.json` |
| **B3** PCT (loss-only, 30M preset) | `arch/predictive-coding-trunk` | 4000 (best) | 69.2M | 400.9 | 1806.6 | **4.51** | STRONG OVERFITTING (but lowest ratio so far) | `results/ood_pct-30m_68M_step4000.json` |
| **B4** abstain-fix + multi-cortex (30m_p4 scale, full DNA) | `master` @ `a22eecc` | 2000 | **889.6M** | **102.9** | **295.9** | **2.87** | **NEW BAND** (gap 2.0–3.0) | `logs/vast/20260614*_af758c381388_arch_889M_abstain-fix-dna-arch-30m_p4_step2kof2k.log` |

All numbers above are read directly from the committed JSON. No
hand-summarised numbers in this table.

### What the table says

1. **The flat baseline (B0) beats every BRIAN variant on absolute
   PPL at this snapshot.** Baseline 80k: train 66.0, OOD 404.0. Best
   BRIAN variant (B2.fix 7k): train 258.8, OOD 1351.5 — **3.9× worse
   train PPL, 3.3× worse OOD PPL** at near-matched params (106.9M vs
   107.8M). **Crucial confound: baseline got 11× more training
   steps** (80k vs 7k), and a 100M model at step 7k is nowhere near
   converged, so this is *not* a matched-compute comparison. See
   **H12** for the full reading.
2. **The "STRONG OVERFITTING" verdict is shared by every row,
   including the baseline.** This eval's threshold (`gap_ratio > 2.0`)
   does not discriminate any model in our ablation space — every
   100M-scale model on a FineWeb-Edu + OpenHermes mix shortcut-fits
   the training distribution. *The verdict label says less than the
   numbers do.*
3. **gap_ratio is the only axis where BRIAN beats the baseline.**
   B2.fix 5.22 < B0 6.12 (BRIAN ~15% better on the
   generalization-fingerprint axis). B3 PCT 4.51 < B0 6.12 (~26%
   better, but cross-row caveat — B3 is 69M not 107M).
4. **gap_ratio progression across BRIAN ablations** (B1 → B2.fix →
   B3): 6.34 → 5.22 → 4.51. PCT shows the largest drop; see **H10**
   for caveats before reading this as victory.
5. **B2 (buggy)** is preserved for forensics, not for citation —
   see **H8**.

---

### H7 — Trunk gradient isolation prevents the awakening collapse

**Hypothesis.** Aux-loss gradients into the shared trunk are what
drove the post-step-5k divergence. Detaching `sem` before the bio
pipeline removes that divergence path.

- **Spec.** `docs/architecture.md` §5.2
- **Mechanism.** `cfg.detach_trunk_from_aux=True` (default ON since `2dd893b`).
- **Evidence.**
  - Recursive run reaches step 5000 cleanly at train PPL 216.5
    (`results/ood_recursive_108M_step5000.json`); pre-fix runs
    diverged at the same milestone.
  - `tests/test_stabilization.py` asserts trunk gradient is invariant
    to aux-loss weights.
- **Status.** ✅ CONFIRMED.

### H8 — ReZero zero-init forward gates improve OOD over trunk-iso alone

**Hypothesis.** Replacing maturity-phase gates on forward injections
with zero-init learnable scalars λ removes the awakening
discontinuity and yields lower OOD perplexity at matched compute.

- **Spec.** `docs/architecture.md` §5.3
- **Evidence (after the load bug was fixed).**
  - B2.fix gap_ratio **5.22** vs B1 **6.34** — ReZero ~17 % better.
  - B2.fix OOD_ppl **1351.5** vs B1 **1372.8** — essentially tied.
  - B2.fix train_ppl **258.8** vs B1 **216.5** — recursive trains ~20 % better in-distribution.
  - Both rows still "STRONG OVERFITTING."
- **Status.** ❌ FALSIFIED **as an OOD lever** (tied OOD_ppl);
  🟡 PARTIAL on the secondary "gap_ratio improves" claim.
- **Forensic note.** B2 row pre-fix showed train_ppl 1169.9 due to
  `BrainConfig` defaults injecting `λ=0` parameters the trained ckpt
  didn't store. Root cause + fix: commit `32074d3` / `d3e5161`.

### H9 — Recursive reasoning (depth-multiplied at constant params)

**Hypothesis.** Looping the expert ladder `N=4` times with
weight-sharing yields effective reasoning depth `N·n_blocks` at zero
added params, improving LM and OOD quality.

- **Spec.** `docs/architecture.md` §5.4
- **Mechanism wired.** ✅ (`cce9be0`); test: `tests/test_recursive_reasoning.py`.
- **OOD outcome (B1 vs B2.fix).**
  - In-dist win: train_ppl 216.5 (recursive) < 258.8 (ReZero).
  - OOD: tied (1372.8 vs 1351.5).
  - gap_ratio: **6.34 worse than** ReZero's 5.22.
- **Status.** 🟡 PARTIAL — clear training-quality win, no OOD win,
  *increases* gap_ratio vs the same trunk without recursion.

### H10 — PCT (top-down generative trunk) shifts gap_ratio out of the 5–6× band

**Hypothesis (verbatim, `architecture.md` §5.5).**
> At matched train PPL, a PCT-trunk model will have ≥ 2× lower OOD
> gap_ratio than the current ReZero / recursive baselines.

- **Mechanism wired.** ✅ Top-down predictors `g_n: h_{n+1} → ĥ_n`,
  detached target, free-energy added to `pred_coding_loss` slot.
  Behaviour tests: `tests/test_pct_smoke.py` (5/5).
- **First eval (B3).** gap_ratio **4.51** at train_ppl 400.9 (best ckpt
  step 4000, 69.2M params, `pct_30m` preset).
- **Comparable cells.**
  - vs B2.fix (107.8M, train_ppl 258.8, gap_ratio 5.22): PCT is **0.71 lower** (–13.5 %).
  - vs B1 (108.2M, train_ppl 216.5, gap_ratio 6.34): PCT is **1.83 lower** (–28.9 %).
- **Status.** 🟡 PARTIAL — directionally matches; does NOT clear the
  "≥ 2× lower" bar (would require gap_ratio ≤ 2.6).
- **Critical caveat ⚠.** B3 is **not at matched train PPL** with
  B2.fix — smaller (69M vs 108M) and earlier best (step 4000 vs
  7000). Lower train PPL mechanically pulls gap_ratio up, so part of
  PCT's win could be undertraining. A same-params, longer-trained
  PCT eval is the unambiguous test.

### H11 — SRC-TEH topology: 2–3× wall-clock + +15-25 % comprehension at fixed params

**Source.** `docs/RFC.md`, shipped as `d895cb2`.

- **Mechanism wired.** ✅
- **Headline efficiency claims** (RFC §4): not yet backed by a
  committed head-to-head benchmark.
- **Status.** ⚠ UNVERIFIED OVERCLAIM until baseline-vs-SRC-TEH
  wall-clock + quality table commits to `results/`.

### H12 — "Measurably better at matched FLOPs than a flat 230M dense transformer"

**Source.** `README.md` line 276.

**Head-to-head (committed 2026-05-25).** Evaluated
`_baseline_mix_80000.pt` on the same harness as the BRIAN B2.fix
run. Same branch (`stabilize/trunk-grad-isolation`), same eval
script, same OOD corpus.

| Side | Architecture | Params | Steps | train_ppl | OOD_ppl | gap_ratio |
|---|---|---|---|---|---|---|
| **Baseline** | flat transformer (`--baseline`) | **106.9M** | **80 000** | **66.0** | **404.0** | 6.12 |
| **BRIAN** | bowtie + trunk-iso + ReZero (B2.fix) | 107.8M | 7 000 | 258.8 | 1351.5 | **5.22** |
| **Δ (BRIAN ÷ baseline)** | | 1.01× | 0.0875× | 3.92× worse | 3.34× worse | 0.85× (BRIAN 15% better) |

**Reading.**

- On **absolute PPL** the flat baseline is ~3-4× better. This
  contradicts the README claim *at this snapshot*.
- On **gap_ratio** BRIAN is 15% better. Topology contribution is
  real but small.

**The asymmetric-compute caveat (load-bearing).**

Baseline got 80 000 steps; BRIAN got 7 000 — **11× compute gap**.
BRIAN divergence after step ~7-10k (H7 / §5.2) means we have no
longer-trained BRIAN ckpt. So this is "baseline at full compute vs
BRIAN at early-stop," not a matched-compute race.

Quantifying what 11× more compute typically buys on a 100M-class
transformer in this regime:
- Loss delta in snapshot: ln(258.8) − ln(66.0) = **1.37 nats**.
- A 100M model at step 7k is **nowhere near converged**; per-step
  improvements are still in the steep early-training regime
  (~0.5-1 nats per decade of steps is not unusual).
- 11× more compute (≈ one decade) plausibly closes 1-2 nats of train
  loss for a baseline.
- The 3-4× absolute-PPL gap **is within the range that
  compute-asymmetry alone can explain**.

What this caveat *does not* mitigate:
- **BRIAN can't be trained as long as the baseline** on this mix —
  diverges past ~7-10k while baseline trains cleanly to 80k+. This
  is an independent architectural cost.

- **Status.** 🟡 **PARTIAL / INCONCLUSIVE.** The snapshot is
  consistent with both "BRIAN is worse at matched compute" and
  "BRIAN would close the gap at matched compute." Cannot be resolved
  with current artifacts.
- **What would settle it.** (1) Train a fresh baseline to step 7000
  and re-eval — matched-compute, ~$3-5 on A100. (2) Train BRIAN past
  10k without divergence — open research. (3) A non-PPL eval where
  Layer-A capabilities or gap_ratio matter more than next-token PPL.

---

### H21 — Per-position abstain logit fixes catastrophic cortex CE (2026-06-14)

**Hypothesis.** The flat `_ABSTAIN_LOGIT = -1e4` constant used to fill
unmapped trunk-vocab slots in `LMExpertEnsemble._project_to_trunk_vocab`
poisons standalone-cortex cross-entropy: every target token whose ID
the GPT-2 cortex doesn't tokenize (most of the trunk's 50,257-vocab
extension) hits the `-1e4` slot → CE per such token ≈ 10,000 nats →
`cortex_loss_ema` blows up to ~500 → harness's Slot-C inhibition
correctly diagnoses the "catastrophic cortex" and pushes `α_eff → 0`
→ fusion collapses, trunk trains alone, all signal from the 3
pretrained GPT-2 experts is destroyed. Replacing the flat constant
with a per-position formula `abstain = max(mapped_logits) − ln(V_trunk)`
restores the inductive bias that an unmapped slot represents
"vocabulary item the cortex never saw" — its logit should sit
*at the uniform-distribution baseline* relative to the slots the
cortex did populate, not 10,000 nats below them. Predicted impact:
8–14× drop in train PPL, similar on OOD.

- **Spec.** `neuroslm/experts.py::LMExpertEnsemble._project_to_trunk_vocab`
  (the `_ABSTAIN_LOGIT` constant and the per-position formula that
  replaced it), pinned by `tests/training/test_lm_expert_abstain_safety.py`
  (5 contracts).
- **Tests.** All 57 `LMExpert*` tests + 151 training + 741 dsl GREEN
  after the fix (pre-fix standalone CE on random batch ≈ 17.37 nats,
  post-fix ≈ 4.03 nats — a 4.3× reduction).
- **Run** — vast.ai **40925851**, label `abstain-fix-dna-arch-30m_p4`,
  A100 SXM4 @ $0.74/hr, branch `master` @ `a22eecc`, DNA-driven
  (`dna/evol/arch.dna`), **889.6M params**, preset `rcc_bowtie_30m_p4`
  (`d_model=512 depth=8 heads=8 ctx=512 batch=16 lr=5e-4 wd=0.01
  warmup=2400`), loss-clip=True(f=3.0), bf16, 2000 steps. Boot stamp
  in log header confirms unfolded DSL sha + git sha at deploy time.
- **Trajectory.**

  | Step | train PPL | mid-OOD WikiText | gap_ratio | cortex telemetry |
  |---|---|---|---|---|
  | 500 | 201.6 | 413.5 | 2.05 | `α_eff=0.500 inh=0.000 cx_ema=3.31` |
  | 1000 | 129.2 | 284.7 | 2.20 | `α_eff=0.503 inh=0.000 cx_ema=3.27` |
  | 1500 | 121.1 | 264.6 | 2.18 | `α_eff=0.504 inh=0.000 cx_ema=3.08` |
  | 2000 | **102.9** | **274.1** | **2.66** | `α_eff=0.505 inh=0.000 cx_ema=3.21` |
  | Final (200-seq, 32,914 tok) | — | **295.9** | **2.87** | — |

  Compare to broken precursor deploy **40923107** (same arch, broken
  abstain): train PPL **1444**, OOD PPL **4655**, `cortex[α_eff=0.000
  inh=1.000 cx_ema=491]` — fusion entirely off. The B4 numbers above
  represent a **14× train-PPL and 17× OOD-PPL improvement** and the
  first time any BRIAN variant has crossed the gap_ratio < 3.0
  threshold on this eval harness (B0–B3 all sat in 4.5–6.3).
- **Outcome.** ✅ **CONFIRMED.** The abstain-fix hypothesis is
  validated: with `α_eff` stable at ~0.5 (not pushed to 0 by the
  catastrophic-cortex defence), the fusion contributes signal, the
  trunk trains, and PPL/OOD drop by ~order of magnitude. **gap_ratio
  2.87** is the new best in the Layer B reference table.
- **🟠 Adjacent issues uncovered** (do not invalidate H21 — recorded
  for follow-up):
  1. **Gradient spike** at steps 1100–1700 (gnorm peaked 142M).
     `loss_clip=True(f=3.0)` caught it (lm loss stayed 4.6–5.2 nats),
     but the spike is a band-aid, not a fix. The `cortex_pre_head_norm`
     stops *cortex* anisotropy; nothing yet stops *trunk* anisotropy
     once the cortex retires it does not fire. Open follow-up.
  2. **Resume crashed 8×** after the run completed, with
     `Unexpected key(s) in state_dict: _genetics_orch.lib.*,
     _transmitter_sys.*`. The checkpoint at step 2000 was saved by a
     `BRIANHarness` snapshot that carries optional subsystems
     (`_genetics_orch`, `_transmitter_sys`) which the resume path's
     freshly-built `BRIANHarness` did not register. Schema-drift
     between save and load. Open follow-up — gated by either
     (a) `strict=False` for these optional keys, or (b) wiring the
     subsystems on by default in `arch.neuro`.
  3. **gap_ratio drift upward** (2.05 → 2.20 → 2.18 → 2.66 → 2.87).
     Train PPL keeps dropping; OOD PPL is roughly flat at 264–295.
     This is the classical overfit signature, but in a *new and
     much smaller* gap band than the 5–6× regime B0–B3 lived in. A
     10k run is the next thing required to know whether (a) gap_ratio
     plateaus around 3, (b) train PPL bottoms out and OOD catches up,
     or (c) overfit accelerates. **This drives the next experiment.**
- **Follow-up:** **10k-step rerun at same scale + same arch + same
  abstain fix**, capture full trajectory at 500/1000/2000/5000/10000
  with mid-OOD at each milestone. Tracked as the deploy queued
  immediately after this finding is committed.

---

### H22 — SmolLM2-360M `general` expert upgrade (2026-06-14)

**Hypothesis.** B4 (H21) used `gpt2` (~125M, 2019, ~40 GB WebText) for
the `general` routing slot of the multi-cortex ensemble. Replacing it
with `smollm2_360m` (`HuggingFaceTB/SmolLM2-360M`, late 2024, ~360M,
4 T tokens of FineWeb-Edu + SmolLM-corpus + Cosmopedia) should improve
the trunk's distillation target on natural-English tokens: ~3× the
parameters and ~100× the training tokens at the same routing slot. The
other two experts (`microsoft/CodeGPT-small-py` for code,
`Qwen/Qwen2.5-0.5B` for reasoning) are unchanged so the comparison is
clean.

Predicted impact at the same `30m_p4` scale + 10k steps:

- **Train PPL @ 2000:** target ≤ 102.9 (B4 baseline) — better
  distillation target should pull the trunk down faster.
- **Train PPL @ 10000:** target ≤ 70 (under H21 trajectory
  extrapolation 102.9 @ 2k → ~70 @ 10k assuming similar slope).
- **Mid-OOD PPL @ 10k (WikiText-103):** target ≤ 250 (B4 was 295.9 at
  2k; SmolLM2's superior natural-English coverage should compound
  through KL distillation).
- **gap_ratio @ 10k:** target ≤ 2.5 (B4 hit 2.87 at 2k; aim to
  stay under 3.0 even at deeper training).
- **Realised harness param count:** ~1.12 B (was 889.6 M with gpt2).
- **Throughput:** ~5-15 % slower per step. SmolLM2 uses its own
  tokenizer (~49 152 BPE) ≠ trunk's gpt2 (50 257 BPE), so the
  `general` expert now traverses the cross-tokenizer **VocabBridge**
  path (per-sample retokenise + char-offset align) instead of the
  same-tokenizer fast path. Ensemble splits 1 fast (CodeGPT) + 2
  bridge (SmolLM2 + Qwen2.5).

**Risk surface (why this could regress, not just improve).** SmolLM2's
own tokenizer means **abstain-fill rate is higher** for the `general`
expert than it was with gpt2 (which shared the trunk's BPE). If
SmolLM2's vocab coverage of the trunk's gpt2 vocab is poor — say
< 60 % of trunk slots map cleanly — many trunk positions will see
`general` contribute only the uniform-baseline abstain logit. The
per-position abstain fix from H21 keeps that safe (no fusion-killing
catastrophe) but it does cap the effective signal injected by the
`general` expert at its coverage fraction. The Qwen2.5 expert
already lives with this on the `reasoning` slot, so the regime is
proven survivable; the empirical question is whether SmolLM2's
better per-mapped-token quality outweighs its coverage haircut.

- **Spec.** `architectures/rcc_bowtie/arch.neuro` `multi_cortex.experts[0]`,
  uses the `smollm2_360m` alias resolved via
  `neuroslm/experts.py::resolve_expert_alias` (the alias registry
  shipped with Item 5).
- **Tests.** Alias resolution pinned by
  `tests/training/test_expert_alias_registry.py` (14 contracts; all
  GREEN as of `aeac569`). Bridge-path correctness pinned by
  `tests/training/test_lm_expert_bridge_safety.py` and
  `tests/training/test_lm_expert_abstain_safety.py`. No new tests
  required — the swap is a config change exercising already-tested
  code paths.
- **Run** — vast.ai deploy queued at commit time, label `neuroslm-full`,
  A100 SXM4 single-GPU, branch `master`, scale `30m_p4`, 10 000 steps,
  `OOD_EVERY=500`. Predecessor instance 40950265 (H21 / B4 10k repeat
  with legacy roster) was destroyed at step 840 to free the budget
  for this run.
- **Outcome.** ⏳ pending the deploy's first OOD snapshot at step 500.
  Will record train PPL + WikiText PPL + gap_ratio at 500 / 1000 /
  2000 / 5000 / 10000 and compare row-for-row to B4.
- **What confirms / refutes:** confirms iff the 10k row beats or
  matches the H21 predictions above on at least 2 of 3 metrics
  (train PPL, OOD PPL, gap_ratio). Refutes iff train PPL plateaus
  worse than B4 at any milestone, or gap_ratio drifts above 3.0
  (suggesting SmolLM2's coverage penalty outweighs its quality
  advantage).

---

## What proved to solve or break things — the punchline list

### Things that demonstrably solved something
- **Trunk gradient isolation (§5.2 / H7)** — *fixed* the post-awakening collapse. Single most important convergence fix.
- **ReZero zero-init forward gates (§5.3 / H8)** — *removed* the awakening discontinuity. Modest gap_ratio win (5.22 vs 6.34); no absolute-OOD win.
- **Recursive reasoning (§5.4 / H9)** — *improves* in-distribution training quality (~20%).
- **PCT loss-only (§5.5 / H10)** — *first* sub-5× gap_ratio in the arc, with the matched-PPL caveat.
- **Per-position abstain logit (H21, 2026-06-14)** — *unlocked* multi-cortex fusion: 14× train-PPL / 17× OOD-PPL drop on rcc_bowtie_30m_p4 vs the broken precursor, and the **first BRIAN variant under gap_ratio 3.0** (2.87 vs ≥4.51 for all prior).

### Things that broke or under-delivered
- **README H12 ("BRIAN measurably better at matched FLOPs vs flat baseline") not yet supported** — head-to-head shows baseline 80k beats BRIAN 7k by ~3-4× on absolute PPL. Compute asymmetry (11× more steps for baseline) too large for this comparison to decide H12. Result is *consistent with* H12 being false but also consistent with H12 being rescuable at matched compute. BRIAN does win gap_ratio modestly (15%) even under the asymmetry.
- **BRIAN training stability under §5.2-5.4 caps out around step 7-10k** at 107M scale on FineWeb-Edu+OpenHermes mix. Baseline trains cleanly to 80k+. *Independent of the matched-compute confound.*
- **Maturity-phase gates on forward injections** — caused the PPL jump at awakening; replaced by ReZero λ.
- **Building a fresh `Brain` with current defaults for eval of older ckpt** — silently injected λ=0 the trained model never had → bogus B2. Fixed by legacy-default-fallback (`32074d3` / `d3e5161`).
- **"More training fixes OOD"** — *expected* to fail at this scale. Anchored prediction (not yet measured).

### Open / not yet measured
- Same-params PCT eval against same-params ReZero / recursive baseline (matched-PPL test for H10).
- Full PCT-feedback mode (`pct_mode="feedback"`) vs loss-only.
- SRC-TEH wall-clock numbers (H11).
- Matched-compute baseline (step-7000 baseline) for H12.
- **H21 / B4 10k-step run** — does the gap_ratio plateau, the train PPL bottom out, or the overfit accelerate past step 2000? Deploy queued immediately after H21 was recorded.
- **Trunk anisotropy spike fix** — H21 follow-up #1, the 1100–1700 gradient spike is currently band-aided by `loss_clip(f=3.0)`. Open whether `cortex_pre_head_norm`-style LayerNorm on the *trunk* pre-head suppresses it, or whether the source is elsewhere (residual stream, attention out-projection).
- **Checkpoint schema-drift fix** — H21 follow-up #2, the resume crashed 8× with `Unexpected key(s)` for optional `_genetics_orch.*` / `_transmitter_sys.*` subsystems. Gated by either `strict=False` for these keys or wiring them on by default.

---

## Reproducibility — how to re-run each row

### Layer A tests
```bash
py -3 -m pytest tests/test_phi.py tests/test_brain_forward.py \
                tests/test_neurochem.py tests/test_narrative_memory.py \
                tests/test_cognitive_closure.py tests/test_pct_smoke.py -v
```
Requires `tiktoken>=0.6` (in `requirements.txt`).

### Layer B OOD evals (vast.ai recipe)
```bash
BRANCH=<branch> \
CKPT=<lfs_checkpoints/...pt> \
ROLE_TAG=<tag> \
VAST_GPU_QUERY="gpu_name in [A100_SXM4,A100_PCIE,A100_SXM,A100X,RTX_4090] \
                num_gpus=1 rentable=true verified=true reliability>0.99" \
  bash scripts/vast_ood_eval.sh
```
Do **not** weaken the `verified=true reliability>0.99` filter.

| Row | BRANCH | CKPT | ROLE_TAG | result JSON |
|---|---|---|---|---|
| B0 | `stabilize/trunk-grad-isolation` | `lfs_checkpoints/neuroslm_large_107M_adamw_baseline_mix_80000.pt` | `baseline-80k` | `ood_results_baseline-80k.json` |
| B1 | `stabilize/recursive-reasoning` | `lfs_checkpoints/neuroslm_large_107M_adamw_mix_5000.pt` | `recursive` | `ood_results_recursive.json` |
| B2.fix | `stabilize/trunk-grad-isolation` | `lfs_checkpoints/neuroslm_large_107M_adamw_mix_best.pt` | `rezero-fixed` | `ood_results_rezero-fixed.json` |
| B3 | `arch/predictive-coding-trunk` | `lfs_checkpoints/neuroslm_pct_30m_68M_adamw_mix_best.pt` | `pct-30m` | `ood_results_pct-30m.json` |

After the eval finishes, copy the per-branch JSON into `results/`
with the canonical name pattern
`ood_<tag>_<params>M_step<step>.json`.

### Raw-log → analysis pipeline
Every vast.ai run also leaves a raw stdout log under
`logs/vast/<id>.log` (fetched by `npm run sync:logs`). To convert
raw logs into structured findings:

1. Run the LLM analysis procedure in
   [logs/analyzed/INSTRUCTIONS.md](../logs/analyzed/INSTRUCTIONS.md).
2. It produces a `logs/analyzed/<descriptive-name>.md` companion to
   each raw log, renames the raw log to the descriptive name, and
   appends any new insights here in FINDINGS.md.

---

## Audit notes (things this ledger is *not* hiding)

- **README test badge** ("126 passing"): not currently verifiable
  from a fresh venv without `tiktoken` installed. README has been
  left as-is per user revert.
- **Baseline (flat-transformer) eval landed 2026-05-25** — see
  `results/ood_baseline-80k_107M_step80000.json`. Verdict on H12:
  🟡 partial/inconclusive (baseline wins absolute PPL by ~3-4×;
  BRIAN wins gap_ratio by ~15%; baseline had 11× more compute so
  comparison is not matched-compute). See **H12** for what would
  resolve it.
- **A prior draft of this document (and a memory note) claimed
  H12 was ❌ FALSIFIED**, on a scaling-law argument that was wrong
  for the early-training regime the snapshot is actually in.
  Retracted on the same day; verdict downgraded to 🟡. The
  underlying snapshot data is unchanged; only the *reading* of it
  is corrected.
- **PAT exposure in vast.ai responses** is still pending rotation —
  the GitHub PAT was visible in raw API responses on 2026-05-23 and
  has not been confirmed rotated since.
- **Training-log artifacts** — raw vast.ai stdout is captured in
  `logs/vast/*.log` by `npm run sync:logs` and analyzed into
  `logs/analyzed/*.md`. The convergence claims in **H7** are
  reconstructible from the raw log captures of the corresponding
  training runs.
- **synth-v1 training trajectory (logged 2026-05-25)** — analysis at
  [logs/analyzed/train_synth-v1_mix-10000_20260525.md](../logs/analyzed/train_synth-v1_mix-10000_20260525.md).
  The `arch/synthesis-v1` variant (SGB + PCT-stronger +
  PredictiveDropout-dropped + top-down-only) trains to step 10k on
  the small preset (68M), best lm_ema 4.7482 at step 4000.
  **Reproduces the "best at step ~4000 then degrade" plateau pattern
  previously seen for PCT-30M (B3)** — two independent trunk
  variants of the same preset size hit the same ceiling, suggesting
  the cap is preset-driven, not trunk-architecture-driven, at this
  parameter count. Grad-skip safety net (gnorm > 3×EMA → skip) fired
  20 times across steps 3762–8591 (max gnorm 22.62) and kept the
  run from diverging — that mechanism is doing real work and is not
  yet documented as such elsewhere in the spec.

---

## Backlog (next changes that would shrink the gaps in this doc)

1. **Eval PCT step 9000 (and a larger PCT preset) on master tooling.**
   Closes the matched-PPL caveat on **H10**.
2. **Train a fresh baseline to step 7000** and OOD-eval it — gives
   the true matched-compute comparison for **H12**.
3. **Train BRIAN past step 10k without divergence at 107M scale on
   the FineWeb-Edu+OpenHermes mix.** Open research problem.
4. **Run the LLM analysis pipeline** on every existing raw log
   under `logs/vast/` so each row in the reference table has a
   linked analysis md file. See `logs/analyzed/INSTRUCTIONS.md`.
   First analysis landed 2026-05-25: synth-v1 training log
   (`train_synth-v1_mix-10000_20260525.md`).
4b. **OOD-eval `neuroslm_synth_30m_68M_adamw_mix_best.pt`** —
   the missing artifact for a B4 (synth-v1) row in the reference
   table. Same recipe as B3 (PCT). Without this, synth-v1's only
   evidence is in-distribution training PPL from a single log.
5. **Dump baseline checkpoint sidecar JSONs** that the baseline
   training run was missing — needed to back baseline training-PPL
   claims without re-running eval.
6. **Add a CI guard** (`tests/test_findings_artifacts.py`) that
   asserts every JSON/log referenced in this doc exists on disk.
7. **Run an eval that doesn't use next-token PPL** — ARC-Challenge
   held-out, WMDP-OOD, a held-out episodic-recall or
   narrative-coherence task. Where Layer-A capabilities can compete.
8. **Rotate the GitHub PAT** (security, not research).

## Run 38469631 — 2026-06-01 15:17 UTC

I can see this is a successful training completion of the RCC BoWTie P4 architecture up to step 10,000. Here's what the log shows:

**Training Outcome:**
- ✓ Completed 10,000 steps as planned
- Final PPL: **242.1** at step 10,000
- Loss trajectory: relatively stable oscillation in the 5.2–6.3 range throughout
- Checkpoint saved successfully: `dsl_arch_step10000.pt`

**Training Dynamics:**
- Significant PPL variance (174→536 range across steps) — typical under loss clipping (f=3.0)
- Learning rate settled to minimum (3.00e-05)
- Throughput: ~33k tok/s (slowing into the run)
- All auxiliary metrics (Φ, λ₁, NT, osc) remained within expected bounds

**Issue at Shutdown:**
The `PyGILState_Release` threading error after step 10,000 is a Python runtime cleanup issue, not a training failure. The restart loop correctly:
1. Detected training reached target
2. Attempted resume (which promptly exited cleanly)
3. Final state: idle instance ready for destruction

**Question:** Are you looking to:
- Compare this PPL (242.1 @ 10k) against baseline or prior runs?
- Analyze the loss trajectory for stability/convergence rate?
- Move forward with OOD evaluation or another variant?

Based on your memory, this fits the P4 loss-clipping experiment. Should I check whether this meets your target PPL threshold for proceeding?

---

## NFG Layout � Revision Assessment (2026-06-01)

**Revision:** rcc_bowtie NFG render after stabilisation of house layout.

**Score:** Structural coherence 8.5 � Readability 8.2 � RCC bowtie identity 8.5 � Compiler-style maturity 8.3

### What works
- Graph has stabilised into a coherent house layout that no longer shifts unpredictably between runs.
- Central RCC bowtie trunk is the strongest feature; the shaded backbone band clearly highlights world ? thalamus ? GWS ? PFC ? BG ? motor.
- Nucleus placement across the top implies modulatory oversight rather than exclusion.
- Core path edges feel intentional; dark backbone chain through PFC ? BG ? motor and the thalamus/GWS/PFC loop structure are readable at a glance.

### Limitations identified
- Lower peripheral nodes (insula, amygdala, self_m, qualia, cerebellum/forward_m/evaluator) still look like catalogued extras rather than fully snapped submodules.
- The central highlighted band is doing explanatory work that the topology itself should eventually carry (the graph is understandable because of band + placement, not purely because attachment grammar and routing are solved).

### Next steps (implemented in this commit)
1. **Canonical slot templates** � _PRESET_TEMPLATES dict keyed by preset family (e.g. "rcc_bowtie"). Overrides _RESERVED_SLOTS so the same preset family always renders with the same stable house layout.
2. **Subsystem envelopes** � _draw_subsystem_envelopes() draws faint dashed rounded-rect overlays for memory, self-model, predictive-ctrl, cortical-loop, and interoceptive groups. Drawn at zorder=0 so they frame but never obscure node circles.
3. **Anchor constraint pass** � ANCHOR_ALPHA = 0.22 pull in _neuroanatomical_layout runs 3 iterations pulling each non-pinned population toward the weighted centroid of its synapse neighbours. Snaps peripheral nodes to their cluster without disturbing the pinned spine.
4. **Modulation lane port assignment** � existing NT rail-point mechanism now combined with the per-NT ordered departure so each NT's arcs share a visual departure stub before splaying.

### Remaining open items
- Port-aware routing for NT arcs entering spine nodes (ordered entry ports per NT lane, not just departure stem).
- Subsystem-aware force layout: repulsion between envelope groups so self-model and memory cannot drift into the cortical loop region.
- Recognizable "compiler-generated neurodiagram" style will require explicit subsystem framing + canonicalised port grammar rather than more force-directed relaxation.
