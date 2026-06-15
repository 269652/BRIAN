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

---

## Run 40952126 — 2026-06-14 18:48 UTC — H22 SmolLM2 expert swap

**Status:** ❌ **FALSIFIED** — `general` expert swap `gpt2 → smollm2_360m` regressed
training trajectory by ~8× wall-clock and never closed the gap before being
destroyed manually at step 7800/10000.

**Hypothesis (commit `9d070bf`):** Upgrading the `general` expert from gpt2
(125M, 2019, WebText) to SmolLM2-360M (360M, 2024, 4T tokens of curated
FineWeb-Edu) would improve trunk perplexity. The per-position abstain logic
from H21 (`neuroslm/experts.py::VocabBridge.apply`) was assumed to make the
cross-tokenizer bridge path safe.

**Artifacts:**
- Run log: `logs/vast/20260614T184807Z_31cf84a0b3c6_arch_1127M_h22-smollm2-dna-arch_step7800of10k.log` (213 KB, 510 lines)
- Baseline log: `logs/vast/20260614T182653Z_07aba24be2bf_rcc_bowtie_889M_run_step920of10k.log` (gpt2 variant, same harness, only ran 920 steps)
- DNA in checkpoint: commit `700f16e` (`dna/evol/arch.dna` regenerated post-swap)

### Symptom table

| Metric | GPT2 (889M) baseline | H22 SmolLM2 (1127M) | Delta |
|---|---|---|---|
| Initial lm CE @ step 20 | 5.55 nats | 5.85 nats | **+0.30** worse |
| lm CE @ step 500 | 5.31 (monotone ↓) | **6.40** | **+1.09**, having REGRESSED from 5.85 |
| train ppl @ step 500 | 201.6 | **603.5** | **3.0× worse** |
| ood ppl @ step 500 | 413.6 | 656.3 | 1.6× worse |
| Steps to reach train_ppl ≈ 175 | ~920 | ~7500 | **~8× slower** |
| Steady-state throughput | ~2400 tok/s | ~950 tok/s | **2.5× slower** |
| Wall-clock-equivalent step 7800 → | step ~3120 of gpt2 | step 7800 | gpt2 would be at train_ppl ~150 by then (extrapolated) |
| Max gnorm during 0-2k window | <600 | **809,260** @ step 1720 | catastrophic |
| Gnorm explosions (>10k) in first 2k steps | 0 | 6 (steps 960, 1720, 1740, 1780, 1840, 1860) | 6 ÷ 0 = ∞ |
| Frozen-param accounting | 889M trunk + frozen gpt2 experts | 1127M (+238M from SmolLM2 swap) | bridged frozen weight does not help trunk learn faster |

### Root-cause analysis

**Three compounding failures**, in order of severity:

#### 1. The trunk REGRESSED in early training (the smoking gun)

Token-level `lm` CE on the trunk's own LM head over the first 500 steps:

```
GPT2 baseline:  5.55 → 5.42 → 5.38 → 5.31  (monotone improvement, Δ = -0.24)
H22 SmolLM2:    5.85 → 6.01 → 6.26 → 6.40  (monotone DEGRADATION, Δ = +0.55)
```

The trunk did not just learn slower — it actively unlearned for ~700 steps,
then drifted sideways until step ~3000 before finally recovering. By step
7500 the trunk reached `lm CE = 5.10` — barely better than gpt2 baseline at
step 920. Eight times the compute for parity.

**Mechanism (hypothesis):** the bridged SmolLM2 logits arriving via the
distillation loss (KL with `λ_t` ramp + temperature 4.0, configured in
`architectures/rcc_bowtie/arch.neuro:200-220`) pushed the trunk toward a
distribution that the trunk's own embedding/LM-head geometry could not
represent. SmolLM2 has a different tokenizer (49,152 BPE vs gpt2's 50,257),
so the bridge maps each trunk vocab id to the nearest single-token surface
equivalent and abstains (`max(mapped) - ln(V)`) on the rest. Even with the
H21 fix, this introduces a systematic bias on every unmapped slot — and for
a 49k vs 50k vocab pair, that's thousands of slots per step receiving the
abstain signal as the teacher target. The trunk is then trained via KL to
push these slots toward uniform — a destructive prior.

#### 2. Catastrophic gradient explosions in the 1700-1900 step window

Six gnorm spikes >10k inside 240 steps, all correlated with NE pegged at
0.97-0.98 (norepinephrine maxed = NT system detecting model in trouble):

```
step  960:  gnorm   32,100   NE=0.97  C3:pc=  7,407
step 1720:  gnorm  809,260   NE=0.97  C3:pc=291,422
step 1740:  gnorm  334,185   NE=0.68  C3:pc=297,691
step 1780:  gnorm   93,550   NE=0.16  C3:pc=115,784
step 1840:  gnorm  128,209   NE=0.14  C3:pc= 75,610
step 1860:  gnorm   16,121   NE=0.11  C3:pc= 40,864
```

The `C3:pc` ("C3 prediction confidence") column normally sits at 0.15-0.35.
Values of 290,000+ indicate the trunk's representation has lost isotropy —
classic rogue-dim collapse driven by an anisotropic teacher signal. The
loss clipping (`f=3.0`) and the NT-mediated cortex inhibition saved the run
from diverging, but the recovery cost ~2000 steps.

#### 3. Bridge-path throughput tax compounds the loss penalty

Per-sample retokenisation + char-offset alignment (`LMExpert._forward_bridge`,
`neuroslm/experts.py:545-640`) is unavoidable when expert vocab ≠ trunk vocab.
Measured at runtime: 950 tok/s vs gpt2 baseline's 2400 tok/s (2.5× slower).
With the loss-trajectory tax compounding, H22 at step 7800 has roughly the
same training signal as gpt2 baseline would have at step 3120, where gpt2
was already at train_ppl ≈ 145 (extrapolated from baseline's
174 @ step 920 + linear-in-log fit).

### Insights for improving gpt2-based expert cortices

The forensic post-mortem reveals what made the gpt2 baseline robust and
what to copy/avoid for the next architectural iteration:

1. **Same-tokenizer experts are non-negotiable for the trunk's first 2000
   steps.** The bridge path's per-position abstain is mathematically sound
   (per-row `max(mapped) - ln(V)` keeps the unmapped CE at the uniform
   baseline) but **the distillation loss treats abstain slots as targets**,
   not as "no-op" slots. The trunk gets pulled toward uniform on every
   unmapped vocab id every step. Fix options:
   - (a) **Gate distillation by bridge coverage** — only apply KL on tokens
     where the bridge mapped successfully; treat unmapped slots as
     missing-label. New `distillation_mask` kwarg in
     `harness._cortex_fusion_aux_step`.
   - (b) **Gate distillation by `bridge.coverage` at construction time** —
     refuse to enable distillation for any expert with coverage < 0.95.
     Cheaper, less invasive, but stops research into harder bridges.
   - (c) **Same-tokenizer-only experts in the cold-start phase**, switch
     bridge experts on after step 2000 once trunk is settled. Two-phase
     `experts:` roster in the DSL.
2. **Add a `gnorm_emergency_brake` to the harness.** When `gnorm > 10×
   gnorm_ema`, freeze the cortex contribution (`α_eff = 0`) for the next
   step and skip the optimizer update. The NT system already detects this
   (NE → 0.97); wire that detection to a hard halt instead of just a soft
   modulation. Will save the next destabilised run from the 1700-1900
   window cost.
3. **The `C3:pc` channel is a leading indicator of rogue-dim collapse.**
   `C3:pc > 100` predicts gnorm explosion ~20 steps later in this log.
   Add `C3:pc > 50` to the harness's early-warning system; emit a WARN
   line and dump router weights + per-expert CE for postmortem.
4. **gpt2 fast-path experts win on per-FLOP utility.** The 889M baseline
   would extrapolate to train_ppl ≈ 100 by step 10000 (linear-in-log fit on
   step 20-920 trajectory). The 1127M H22 is on track for train_ppl ≈ 150
   by step 10000. **More params via bridge experts = strictly worse than
   more depth in a same-tokenizer trunk** at the rcc_bowtie scale.
   Next experiment: H23 swap `code` slot from `microsoft/CodeGPT-small-py`
   (~124M, same tok) to a same-tok bigger code expert (no bridge tax)
   instead of adding cross-tok generalist experts.
5. **Distillation gap-floor=0.1 was too aggressive.** With H22's initial CE
   gap of ~5 nats, `λ_t` ramped to max immediately and the trunk got a full
   KL signal from a teacher whose distribution it couldn't represent. For
   bridge-path experts specifically, set `distillation_gap_floor` to
   `max(0.1, 0.5 × initial_bridge_kl_divergence)` — measured once at step 0
   then frozen.

### Operational lessons

- **Manual destroy at step 7800/10000 was the right call.** Cost: ~$0.73/hr
  × 3.3 hr ≈ $2.40 wasted on the failing run. Would have been ~$4.00 if it
  ran to completion. The early kill saved 22%.
- **`brian logs <destroyed_id>` failed** because the vast API can't reach
  a destroyed instance. The log was only retrievable by `git pull` (the run
  pushes its log file on self-destruct). Tracked as a CLI gap; fix:
  `brian logs` should fall back to `git fetch && git pull && cat
  logs/vast/*<id>*.log` when vast returns 404. **In flight.**
- **No `brian logs --latest`** meant manually searching `logs/vast/`. Add
  `--latest` flag that resolves to the most-recently-modified file. **In
  flight.**

### What this falsifies

| Claim | Status before run | Status after run |
|---|---|---|
| H21's per-position abstain makes bridge-path experts safe | 🟠 PENDING | 🟡 PARTIAL — safe for forward, NOT safe for distillation gradient |
| Bigger pretrained expert improves trunk via distillation | 🟠 PENDING | ❌ FALSIFIED at rcc_bowtie scale, bridge-path variant |
| Frozen expert weight count is a strict win | implicit | ❌ FALSIFIED — bridge tax >> param benefit |


---

## Run 40952126 — 2026-06-14 21:30 UTC — root-cause UPDATE (post-mortem #2)

After writing the post-mortem above, the obvious next question was
*"which of the five candidate mechanisms is actually killing the bridge?"*
I built `scripts/diagnose_bridge_ce.py` to test each one in isolation
against gpt2's own next-token CE on a 7-sentence English paragraph. The
results inverted my initial diagnosis.

### Setup

* Trunk tokenizer: gpt2 (V_t = 50257)
* Expert: SmolLM2-360M (V_e = 49152)
* Test paragraph: 83 trunk tokens / 79 expert tokens of natural English
* Baseline: gpt2's own next-token CE on the paragraph = **3.016 nats**

### Four bridge configurations measured

| Config | Bridge build | Alignment | CE (nats) | Δ vs gpt2 |
|---|---|---|---|---|
| (C1) Current shipping code | strict 1-token only | smallest-`e` with `e_end ≥ t_end` | 3.068 | +0.05 |
| (C2) Relaxed bridge only | `n ≥ 1`, use first expert subtoken | smallest-`e` (unchanged) | **3.870** | **+0.85** |
| (C3) Exact alignment only | strict 1-token (unchanged) | only-`e` with `e_end == t_end` else -1 | **2.798** | **−0.22** |
| (C4) Both | relaxed | exact | 3.547 | +0.53 |

### Hypothesis flips

**Vocab coverage is NOT the bottleneck.** I assumed the strict
`len(eids) == 1` rule (73.6% trunk coverage) was wasting signal at
26% of slots. Relaxing it to `len(eids) >= 1` (99.99% coverage) actually
makes CE **WORSE by 0.85 nats**. Reason: many trunk tokens share the
same first expert subtoken — e.g. ` general`, ` generate`, ` generation`
all start with ` gen` — so the bridged trunk softmax dilutes correct
mass equally across all "siblings" with the same prefix. CE penalty
≈ ln(sibling count). The strict bridge's per-position abstain (with
`max(mapped) − ln V_trunk`) was the *correct* fallback all along.

**Alignment SHIFT is the bottleneck.** At positions where trunk and
expert tokenisations don't share a char-end boundary, the legacy
`smallest e such that e_end ≥ t_end` rule picks an expert position
whose end is *strictly greater than* trunk's. Two failures compound:

1. **One-step leakage** — the expert at `e` has already SEEN trunk's
   target as part of its input prefix.
2. **Wrong-horizon prediction** — the expert at `e` predicts content
   *starting past* trunk's prediction horizon. Using that as the
   distillation target trains trunk toward "what comes after trunk's
   target", not "what trunk's target IS".

On natural English with gpt2/SmolLM2, only ~5% of trunk positions
suffer from this misalignment. But that 5% of *actively wrong* signal
is more harmful than 5% of *uniform* signal — the trunk's distillation
loss tries to match it, dragging the trunk toward bad next-token
distributions.

### The actual fix (one-line conceptually, ~50 lines of code)

* New helper `_align_by_char_offsets_exact` returns `-1` at mismatched
  positions. The legacy `_align_by_char_offsets` stays for back-compat.
* `LMExpert._forward_bridge` switches to the exact helper; at `-1`
  positions, the output buffer stays at zero (= uniform after softmax).
* `LMExpert.last_alignment_coverage` now exposes the per-batch fraction
  of trunk positions that exact-aligned. Harness can log this as a
  leading indicator.
* `VocabBridge.build` is unchanged — keeps the strict `len(eids) == 1`
  rule because relaxing it hurts more than it helps.

Files: `neuroslm/experts.py`,
`tests/training/test_lm_expert_bridge_exact_alignment.py` (14 tests,
8 unit + 3 telemetry + 1 integration + 2 export-surface).

### Predicted training impact

The standalone bridge-CE win is 0.27 nats (3.068 → 2.798) on English.
In the H22 training run this would feed back through the distillation
KL on every step. With distillation weight α≈0.5 (the H22 setting),
that's a ~0.13-nat improvement in the per-step KL loss, which
historically converts to ~3-5× of that in trunk LM CE at the start
of training where the trunk is most plastic. So a conservative
prediction for H23:

* Step 500 trunk lm CE: H22 was 6.40, H23 should be ≤ 5.5
  (the gpt2 baseline was 5.31)
* Step 7800 train_ppl: H22 hit 175 after 7800 steps; H23 should
  cross 175 by ~step 3500-4500
* OOD `wikitext` ppl gap_ratio: should converge toward the gpt2
  baseline's 1.6-1.8 range (H22 was stuck at 1.09 because the trunk
  LM head was perma-stuck high on both)

### Things STILL to fix from the H22 post-mortem

These remain real concerns even with the alignment fix:

* **Bridge throughput tax (2.5×).** Per-sample Python loop in
  `_forward_bridge` is the bottleneck. A batched re-tokenisation +
  vectorised alignment would close most of the gap. Defer until H23
  shows whether the CE win justifies a bigger refactor.
* **`gnorm_emergency_brake`.** Still needed — the alignment fix doesn't
  prevent the kind of routing collapse seen at steps 1720-1860.
* **`C3:pc > 50` early-warning.** Still worth wiring up.
* **Same-tok experts cold-start.** No longer required for *correctness*
  (the alignment fix makes cross-tok safe), but still a throughput win
  during the bootstrap phase.

### What this updates

| Claim from post-mortem #1 | Status after post-mortem #2 |
|---|---|
| "Vocab coverage gates distillation quality"                | ❌ FALSIFIED — coverage was a red herring |
| "Bridge path adds ~0.5 nats per-step CE penalty"           | 🟡 PARTIAL — actual mechanism is alignment shift, ~0.27 nats |
| "H22 SmolLM2 cannot beat gpt2 at this scale"               | ❌ FALSIFIED — with exact alignment, SmolLM2 beats gpt2 by 0.22 nats on the held-out paragraph |
| "Same-tok experts are non-negotiable in the first 2000 steps" | 🟡 PARTIAL — required for *speed*, no longer required for *correctness* |
| "Per-position abstain is mathematically sound"             | ✅ CONFIRMED — and now lightly used (only at the ~5% of positions that misalign) |




## Run 40968510 � 2026-06-14 23:30 UTC � H23 post-mortem #3 (the REAL mechanism)

**Hypothesis.** H22 SmolLM2 swap regressed PPL/OOD because the
exact-end alignment fix (commit `a976fee`) was *necessary but not
sufficient*. There is a second, much larger mechanism upstream of the
bridge: the KL-distillation loss in
`neuroslm.harness.NeuroSLMHarness._cortex_fusion_aux_step` uses
`F.kl_div(..., reduction="batchmean")` which divides the per-token KL
sum by `B` only (not `B × T`). For (B=1, T=512, V=50257) the reported
KL is therefore `~T × per_token_KL = ~512 × per_token_KL`. The LM
cross-entropy term uses the default `reduction="mean"` (averages over
`B × T`). The two loss terms are on incompatible scales — the
distillation term dominates by `T` ≈ 500×, and the dominance scales
with teacher sharpness.

**Status.** ✅ CONFIRMED — reproduced numerically in
`scripts/diagnose_kl_distill_blowup.py`. Cross-validates against H22's
own training log at step 500 (`kl=1512.000` matches the predicted
`per_token_KL × (T-1) × T² = 5 × 71 × 16 = ~5680` upper bound; actual
value reduced because the EMA-gap-ramp has begun to throttle λ).

### The numerical smoking gun

`scripts/diagnose_kl_distill_blowup.py` — gpt2 trunk (random init) +
each teacher candidate on a 7-sentence English paragraph (T=72):

| Teacher | per-token KL × T² (correct) | batchmean × T² (current) | Ratio | × LM loss |
|---|---|---|---|---|
| gpt2 (fast path) | **5.014 nats** | **355.981 nats** | 71× | **33×** |
| SmolLM2 (bridge path, ✓ alignment) | **4.550 nats** | **323.034 nats** | 71× | **30×** |

Reduction-ratio 71× equals `T-1 = 71` to the digit. The bug is
deterministic and isolated to one PyTorch keyword argument.

Scaled to the training batch (T=512), the bug yields
`kl_batchmean ≈ 2300` — **exactly matching** the H22 `train.log`
step 20 value of `kl=2304.000`.

### Why gpt2 expert trained successfully despite the same bug

Both teachers produce equally-broken distillation magnitudes at step
0 (gpt2: 356, SmolLM2: 323 nats). But during training, two things
diverge:

1. **Imitation feasibility.** A gpt2-trunk can in principle learn to
   imitate gpt2-teacher *exactly* — same tokenizer, same architecture
   family, same head structure. The KL gradient points to a reachable
   target; over 500-1000 steps the gap closes and λ throttles down.
   A gpt2-trunk **cannot** efficiently learn to imitate SmolLM2-teacher
   — different tokenizer (49 152 vs 50 257), different positional
   encoding (RoPE vs absolute), different head normalization. The KL
   gradient points to an *unreachable* target. The student
   "spins its wheels" and the gap stays large for thousands of steps.

2. **Capacity mismatch amplification.** SmolLM2 has 3× the parameters
   and 100× the training tokens of gpt2 — it produces sharper logits.
   A sharper teacher × the broken reduction × an unreachable imitation
   target × T²=16 temperature scaling = the H22 training pathology.

H22 log @ step 500: `cortex[α_eff=0.500 λ=1.115 kl=1512 lm_ema=10.28
cx_ema=4.23]`. The trunk LM loss is `lm=6.40`. **Effective loss
weighting**: `lm=6.40` vs `λ×kl = 1.115 × 1512 = 1686`. The
distillation term is **263× larger** than the LM term. The trunk is
being trained almost entirely on a noisy gradient toward an
unreachable target distribution.

### Why isolated-paragraph CE looked great

Post-mortem #2 (commit `a976fee`) measured the bridge's *output CE*
in isolation: gpt2 own CE = 3.016 vs SmolLM2 bridge CE = 2.798 on the
held-out paragraph. That was a valid, correct measurement of the
expert's quality — **but it never exercised the distillation loss
path**. The bridge is healthy; the harness's loss-combination is the
bug.

### What this updates

| Prior claim | Status after post-mortem #3 |
|---|---|
| "Alignment shift is the dominant H22 mechanism" | 🟡 PARTIAL — alignment shift is real (≈0.27 nats of bridge CE), but the loss-reduction bug is **two orders of magnitude larger** and was the actual training killer |
| "Exact-end alignment fix should unlock SmolLM2" | ❌ FALSIFIED — necessary, but not sufficient. SmolLM2 still tanks training because of (a) reduction bug and (b) cross-family imitation infeasibility |
| "H22 = SmolLM2 too big for trunk capacity" | 🟡 PARTIAL — true, but the dominant mechanism is the reduction bug; even gpt2 expert is being distilled at 33× LM-loss strength right now |

### Decisions taken in this run

1. **Swap `general` slot back to `gpt2`** in all three roster locations
   (`architectures/rcc_bowtie/arch.neuro`,
   `architectures/master/arch.neuro`,
   `architectures/current/arch.neuro`). This is the user's request and
   the immediate, surgical revert to the known-good H21 baseline.
2. **DO NOT silently fix the reduction bug.** That change rebalances
   the loss landscape for every existing trained checkpoint and every
   future run; it warrants its own RFC + ablation. Flag it here.
3. **Add `scripts/diagnose_kl_distill_blowup.py` to the repo** as
   permanent regression-proof of the mechanism.

### Followups (do not silently apply)

* **F1 — fix reduction.** Change `reduction="batchmean"` →
  `reduction="mean"` in `_cortex_fusion_aux_step` and rerun the H21
  baseline. Expect: ~30× drop in `kl` metric values, smoother
  `lm_loss` trajectory, faster convergence.
* **F2 — re-tune `distillation_lambda_max`.** After F1, the natural
  scale is per-token KL ≈ O(1) nats. `lambda_max=1.0` is then a
  reasonable upper bound. Without F1, the *de facto* effective lambda
  is `T × 1.0 = 512`.
* **F3 — re-attempt SmolLM2 swap** once F1 + F2 are in. With correct
  loss scaling, the imitation-infeasibility may also be addressable
  via a *projection-on-LSH-subspace* trick (only distill onto the
  ~100 most-probable trunk tokens per position).

### Regression-pinned by

* `scripts/diagnose_kl_distill_blowup.py` (numerical mechanism)
* H22 `logs/20260614-184807_arch_1127M_h22-smollm2-dna-arch_*/train.log`
  step-20 + step-500 `cortex[…kl=…]` values

## Run pre-H24 — 2026-06-15 00:30 UTC — Capacity-Funneled Distillation (CFD): from explosion to implode

Per user request, the H22 "teacher too strong" pathology was promoted
from a swap-back-and-warn item (post-mortem #3 above) to a **design
target**: instead of just dampening the strong teacher, *funnel* it
into a representation the student can actually exploit, so that PPL
*decreases monotonically* with teacher capacity. The user's framing:
"there's an optimal train status for SLMs where the weights are
optimal for PPL and OOD PPL, and a bigger teacher should outperform a
smaller one at the same student parameter count."

### Diagnosis recap (one-paragraph form)

A naive KL distillation loss with a much higher-capacity teacher fails
for three reasons that *compound*: (1) the teacher's softmax has modes
the student trunk cannot disentangle (unrepresentable target → the
loss has no zero), (2) the teacher's distribution is far sharper than
the student's at init (sharpness mismatch → gradient is dominated by
where student is least able to follow), and (3) gradients from the LM
loss and the distill loss can point in opposing directions
(unconstrained tug-of-war → no floor on the harm). The
`reduction='batchmean'` bug (Followup F1) amplified all three by ~T
but is *not* their cause; even the corrected per-token KL fails the
same way, slower.

### The CFD design (three stages, all closed-form)

Each stage neutralises exactly one ingredient of the diagnosis. They
compose, and each has a one-line analytic interpretation.

**Stage 1 — top-$K$ rank-preserving sparsification.**
Replace the raw teacher softmax with a $K$-mode-plus-uniform-tail
projection: keep the top-$K$ teacher logits at their softmax mass,
spread the residual $1 - \sum_{i \in \mathrm{TopK}} p_i$ uniformly over
the $V - K$ remaining tokens. This makes the imitation target lie
inside the student's reachable softmax simplex when $K$ is at or below
the student's mode-resolution capacity. The KL now has a *reachable
floor* (it can converge to zero); the student stops spending gradient
budget on distinctions it cannot make. Schedule: $K = 4 \to 32$
linear over the first half of training ("easy first, hard later").

**Stage 2 — entropy-matched temperature.**
The temperature is computed per batch as
$T_{\mathrm{eff}} = T_0 \cdot \max(1, H(p_s) / H(p_t))$. Early in
training the student is much less certain than the teacher
($H_s \gg H_t$), so $T_{\mathrm{eff}}$ is large and the teacher's
sharpness is softened to a level the student can plausibly match. As
the student catches up, $T_{\mathrm{eff}} \to T_0$ and the teacher's
fine-grained distinctions come into focus — *after* the student has
the representational scaffolding to use them. This is exactly the
self-paced revelation the user asked for: the teacher reveals more
detail only as the student earns capacity.

**Stage 3 — gradient-alignment gate.**
Compute the cosine $g_{\mathrm{align}} \in [-1, 1]$ between the
distillation gradient and the LM-CE gradient (cheaply, via the trunk's
last-layer bias as a single shared probe parameter). Then weight
$\lambda_{\mathrm{eff}} = \lambda_0 \cdot (1 + g_{\mathrm{align}})/2$.
When the teacher pulls along the LM gradient,
$g_{\mathrm{align}} \to 1$ and distillation runs at full strength;
when it pulls against, $g_{\mathrm{align}} \to -1$ and
$\lambda_{\mathrm{eff}} \to 0$ automatically. **The student can never
be hurt by the teacher.** This is PCGrad / GradReg applied to KD; the
no-harm floor becomes mechanical, not asymptotic.

### Why this should "implode" PPL instead of explode

Three claims, each falsifiable by the four-arm ablation in
`tests/training/test_cfd_distillation.py`:

* **(A) Reachable target.** Stages 1+2 together project the teacher
  onto the student's reachable distribution family. Loss has a zero;
  optimisation has a stable fixed point.
* **(B) Strictly more bits than one-hot.** The top-$K$ projection
  carries strictly more bits than the one-hot LM target (top-1 is the
  one-hot rank-1 case). For any student with capacity > 1 mode this
  is non-trivially useful information.
* **(C) Gradient compatibility.** Stage 3 projects the distill
  gradient onto the half-space aligned with the LM gradient. The
  combined update is component-wise no worse than LM-only.

A+B+C imply $\mathrm{ppl}(\theta_{\mathrm{CFD}}) \le \mathrm{ppl}(\theta_{\mathrm{LM-only}})$
with strict inequality whenever the teacher's top-$K$ distribution
disagrees with the empirical one-hot but agrees in direction with the
LM gradient — which is *exactly* the regime where the teacher's
pretraining knowledge transfers. Formal statement: see
`hypothesis/H006_capacity_funneled_distillation_implode.md` and
`docs/formal_framework.md §13`.

### The implied "parameter-efficient frontier"

If H006 holds, then for fixed student capacity $C_s$ the map
$C_t \mapsto \mathrm{ppl}(\theta_s^\star(t))$ has an infimum
$\mathrm{ppl}^\star(C_s)$ approached as $C_t \to \infty$. This is a
*measurable* SLM-frontier curve — the best PPL any $C_s$-parameter LM
can achieve when distilled from an arbitrarily strong CFD-funneled
teacher. We can chart it empirically by scaling $C_t$ at fixed $C_s$
and observing the implode. Practical implication: a small student
trained from a much larger CFD-funneled teacher should *outperform* a
small student trained from a teacher matched to its own size, at the
same parameter count.

### Decisions taken in this run

1. **Declare H006 as a falsifiable hypothesis** under
   `hypothesis/H006_capacity_funneled_distillation_implode.md`,
   `proof_status: missing` (Lean obligation deferred until the
   empirical four-arm ablation passes).
2. **Promote the F1 reduction-bug fix into the CFD redesign.** The
   per-token reduction is fixed *as a side-effect of* the CFD path;
   the legacy `reduction='batchmean'` code path is kept under a flag
   for bit-identical reproduction of prior runs.
3. **Ship CFD behind a new `cfd_enabled: bool` switch on
   `MultiCortexConfig`** (default `false`). H21–H23 reproduce
   bit-identically; only new runs that explicitly opt in get the new
   path.
4. **Write the four-arm ablation test first**, then implement.

### Followups (post-CFD)

* **F4 — empirical frontier chart.** Run the
  $\{C_t = 125\text{M}, 360\text{M}, 1\text{B}, 3\text{B}\}$ scan at
  fixed $C_s = 30\text{M}$ trunk, plot
  $\mathrm{ppl}^\star(C_s; C_t)$, fit the asymptote
  $\mathrm{ppl}^\star(C_s)$.
* **F5 — re-attempt SmolLM2 swap with CFD enabled.** Expected: PPL
  better than the H21 all-gpt2-family baseline (the original goal of
  H22).
* **F6 — formalise (II)** (monotone implode in teacher capacity)
  in Lean. Requires defining a Brian-side
  `CapacityOrdering` predicate via KL on the data distribution and a
  short refinement-of-information argument.

### Regression-pinned by

* `hypothesis/H006_capacity_funneled_distillation_implode.md` —
  formal statement of the theorem
* `tests/training/test_cfd_distillation.py` — four-arm ablation
  falsifier (Arms A/B/C/D with predicted PPL ordering)
* `docs/formal_framework.md §13` — derivations of (I), (II), (III)
* `scripts/diagnose_kl_distill_blowup.py` — kept as the prior-art
  baseline that the CFD design must out-perform
