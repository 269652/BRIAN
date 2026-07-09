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
| **B5** H21 10k rerun (same GPT-2 roster, h24-cfd label) | `master` @ `8d7140c` | 3000 (mid-run, in progress) | 889.6M | **45.0** | **130.1** | 2.89 | COMPARABLE to B4 (dramatically better abs PPL; gap stable) | `logs/vast/20260615T092922Z_cd3a9493b050_arch_889M_h24-cfd-10k-dna-arch_step3540of10k.log` |
| **B6** H22 SmolLM2 upgrade (1.12B total, DNA-arch 10k) | `master` @ `c19bf62` | 10000 | **1127M** (146.9M trainable) | **23.6** | **155.0** | 6.55 | GAP REGRESSION vs B4 (better abs PPL; overfit worsened) | `logs/20260615/neuroslm-full-dna-arch/175931_0_10000.log` |

B5 and B6 numbers are read directly from the training logs. B5 is a
mid-run snapshot (step 3000 of 10k); final B5 numbers pending. B6 is
the completed H22 run.

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
6. **B5 (step 3000, in progress):** Dramatically better absolute PPL
   than B4 at a comparable step count (train 45.0 vs 102.9 at step
   2000, OOD 130.1 vs 295.9 final). gap_ratio stable at 2.89 —
   neither improving nor regressing. Training longer helps absolute
   quality; gap is near a floor.
7. **B6 (SmolLM2 upgrade, 10k):** Extraordinary train PPL (23.6 —
   best seen in this arc), but gap_ratio REGRESSES to 6.55. The
   larger/better-trained general expert provides a stronger distillation
   target → faster in-distribution convergence → stronger overfit.
   The absolute OOD PPL (155.0) is also the best seen, but the trunk
   memorises distribution faster than the OOD corpus rewards.

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
- **Run** — vast.ai **41084160**, label `neuroslm-full-dna-arch`,
  A100, branch `master` @ `c19bf62`, 10 000 steps.
- **Result** (B6 in Layer B table):

  | Step | train PPL | OOD PPL (WikiText-103) | gap_ratio |
  |---|---|---|---|
  | 10000 (final) | **23.6** | **155.0** | **6.55** |

- **Canonical checkpoint:** `hf://moritzroessler/BRIAN/checkpoints/20260615-175931_c19bf629_neuroslm-full-dna-arch/step10000.pt`
- **Log:** `logs/20260615/neuroslm-full-dna-arch/175931_0_10000.log`
- **Outcome.** ❌ **FALSIFIED on gap_ratio.** SmolLM2 upgrade dramatically
  improves absolute train PPL (23.6 at 10k vs 102.9 for B4 at 2k;
  even vs B5's 45.0 at 3k), but gap_ratio REGRESSES from 2.87 (B4)
  to 6.55 (B6), reversing all the H21 gains and returning to the B0–B3
  regime (4.5–6.3). The larger, better-trained `general` expert provides
  a stronger distillation target → trunk trains faster → BUT the trunk
  memorises the training distribution more aggressively. SmolLM2's
  quality advantage over gpt2 manifests as deeper in-distribution fit,
  not better generalisation. The coverage haircut (bridge-path via
  SmolLM2's own tokeniser) does not save the gap_ratio.
- **⚠ Post-run restart artifact:** after reaching step 10000, the
  restart loop continued and saved checkpoint clones under timestamps
  20260615-184943, 20260615-185105, 20260615-185222, each showing
  wikitext_ppl=142.1 and train_ppl=nan. These are zero-step resumption
  artifacts, not real training steps. Canonical result is step 10000
  at train_ppl=23.6.
- **What would confirm H22:** a run where the SmolLM2 upgrade holds
  gap_ratio ≤ 2.5 at 10k steps alongside improved OOD PPL. Needs a
  stronger regularisation strategy to counteract the faster-fit
  dynamic (higher dropout, stronger flooding, lower lr after warmup).

---

### H23 — H21 10k rerun: gap_ratio trajectory past step 2000 (2026-06-15)

**Hypothesis.** B4 / H21 at step 2000 showed gap_ratio drifting
2.05 → 2.87. Rerunning the same architecture for 10k steps
distinguishes: (a) gap_ratio plateaus ~3 (natural floor), (b) OOD
catches up as train PPL bottoms out (convergence), (c) overfit
accelerates.

- **Spec.** Same as H21: `architectures/master/arch.neuro`, DNA-mode,
  preset `rcc_bowtie_30m_p4`, GPT-2 roster, abstain-fix active.
- **Run** — vast.ai **cd3a9493b050**, label `h24-cfd-10k-dna-arch`,
  branch `master` @ `8d7140c`, 10 000 steps (in progress when log
  was last captured at step 3540).
- **Mid-run trajectory (step 3000 mid-OOD):**

  | Step | train PPL | OOD PPL (WikiText-103, 50-seq) | gap_ratio |
  |---|---|---|---|
  | 3000 | **45.0** | **130.1** | **2.89** |

- **Mid-run checkpoint:** `/workspace/brian/lfs_checkpoints/20260615-092625_7fdc3ccd_neuroslm-full-h24-cfd-10k-dna-arch/step3000.pt`
- **Log:** `logs/vast/20260615T092922Z_cd3a9493b050_arch_889M_h24-cfd-10k-dna-arch_step3540of10k.log`
- **Outcome.** 🟡 **PARTIAL / IN PROGRESS (step 3000 of 10k).**
  Early evidence:
  1. Both train PPL (45.0) and OOD PPL (130.1) improved dramatically
     from B4 final (102.9 / 295.9). Longer training helps absolute
     quality on both axes simultaneously.
  2. gap_ratio 2.89 ≈ B4's 2.87 — essentially unchanged over an extra
     1000 steps. Points to scenario (a): the gap is at a floor near
     2.9, not drifting up nor converging down.
  3. Most importantly: with B6 (SmolLM2, same steps to 10k) showing
     gap_ratio 6.55, the comparison isolates the expert upgrade as the
     source of the regression. GPT-2 roster → gap stable 2.89;
     SmolLM2 roster → gap regresses 6.55. **The general-expert quality
     vs coverage tradeoff is the active research question.**
- **Follow-up:** full 10k result needed to confirm plateau vs continued
  drift. Compare row B5 final vs B6 (SmolLM2) at matched steps.

---

### H24 — Pontryagin / Hopfion-lite topological-charge diagnostic (2026-06-23)

**Hypothesis.** Each trained attention head, projected per-token onto
S² via a learnable `Linear(head_dim, 3)`, traces a discrete map T→S².
Its Berg-Lüscher signed solid-angle sum (`Q_h`) and inter-layer
orientation decorrelation (`ε_ortho`) are *non-degenerate* observables
of routing structure. We predict that across a healthy 10k run:

1. **`Q_h` distribution per head is non-trivial** — at least ⅓ of heads
   accumulate |Q_h| > 0.1 by step 1000 (sliding-triangle winding above
   random-walk noise floor of ≈ 1/√T per head).
2. **`ε_ortho` grows with training** — early layers and late layers
   develop distinct projections; expect ε_ortho > 0.1 in the trained
   model and a measurable monotone climb from step 0.
3. **Pruning low-|Q_h| heads is safer** than pruning by random-or-norm —
   the head-pruning robustness eval should show ≥ 30% fewer
   degradations when low-|Q_h| heads are removed first vs random.

- **Spec.** `architectures/master/arch.neuro` line 869 +
  `architectures/SmolLM/arch.neuro` (synced) — block
  `regularization { pontryagin_topo_charge: { enabled: true,
  alpha: 0.0, gamma: 0.0, Q_target: 0.0, weight_init_std: 0.02 } }`.
  Active in **DIAGNOSTIC MODE** only — Q_h and ε_ortho are logged
  every step but zero is added to the loss budget. Penalty mode
  (`alpha` or `gamma > 0`) is a follow-up experiment after the
  baseline distribution of Q_h across heads is characterised.
- **Mechanism.** `neuroslm/mechanisms/topo_charge.py` —
  van Oosterom-Strang signed spherical-triangle area (atan2
  formulation, stable across the full sphere; Berg-Lüscher 1981);
  per-block forward hook installed by
  `LanguageCortex.enable_topo_charge_capture_now()`; consumed by
  `RegularizationController.collect_topo_charge_aux()`; auto-fired
  by `BRIANHarness._topo_charge_aux_step()` after the cortex-fusion
  compose site.
- **TDD evidence.** 61 GREEN tests across 6 files: math contracts
  (`tests/dsl/test_topo_charge.py`), §14 stub-detection meta-tests
  (`tests/dsl/test_topo_charge_stub_audit.py`), DSL parse
  (`tests/dsl/test_topo_charge_dsl_parse.py`), LanguageCortex hook
  (`tests/dsl/test_topo_charge_attn_capture.py`), RegController glue
  (`tests/dsl/test_topo_charge_regcontroller.py`), end-to-end inert-
  gate (`tests/dsl/test_topo_charge_harness_integration.py` — pins
  the load-bearing `torch.equal` zero-contribution invariant when
  alpha=gamma=0).
- **Outcome.** ⏳ **PENDING DEPLOY.** No metrics captured yet — diagnostic
  is active in arch but the next training run is needed to gather
  trajectories. Per CLAUDE.md §1e the deploy requires explicit user
  authorisation; not initiated by this session.

---

### H25 — Liouville Symplectic Residual: Noether-conserved hidden channels (2026-06-23)

**Hypothesis.** Splitting `d_model` into canonical coordinates `(q, p)`
and running one Stoermer-Verlet leapfrog step of a learned Hamiltonian
`H = KE(p) + V(q) + W(q)` provides two durable inductive biases:

1. **Symplectic structure (det J = 1 exactly).** Each leapfrog substep is
   a triangular shear; their composition preserves phase-space volume by
   construction. We predict that models trained with `noether_strength > 0`
   develop lower rank-collapse risk (erank stays higher) and smoother
   loss trajectories compared to the baseline.
2. **Noether residual signal.** `L_Noether = (H_final − H_initial)²` is
   identically zero for a perfectly-conserved Hamiltonian. With
   `noether_strength > 0` the optimizer is rewarded for organizing the
   hidden state so that phase-space energy is preserved across the layer.
   We predict `noether_H_diff < 0.1` after 5k steps (current value on
   random inputs ≈ 0.5–2.0).
3. **Long-context stability.** A symplectic channel that conserves energy
   along the sequence axis provides a stable propagation path for
   long-range dependencies. We predict gap_ratio improvement of ≥ 10% at
   T ≥ 1024 vs a matched baseline without the mechanism.

- **Spec.** `architectures/master/arch.neuro` + `architectures/SmolLM/arch.neuro`
  — block `regularization { liouville_symplectic: { enabled: true,
  noether_strength: 0.0, dtau_init: 0.1, potential_kind: quadratic,
  w_rank: 4 } }`. Active in **DIAGNOSTIC MODE** — `noether_loss` and
  `noether_H_diff` are logged every step; zero added to the loss budget.
  Penalty mode (`noether_strength > 0`) is a follow-up experiment.
- **Mechanism.** `neuroslm/mechanisms/liouville_symplectic.py` —
  `LiouvilleSymplecticBlock` wraps one explicit Stoermer-Verlet step
  (half-kick → drift → half-kick) with `torch.autograd.grad(create_graph=True)`
  for `∂_q H`. `QOnlyPotential` abstract base enforces the `forward(q)`
  type signature at construction time (type-level det(J)=1 guarantee).
  `LowRankPairwise(W)` is the token-interaction term (q-only by construction).
  Noether residual stashed as `_last_noether`. Block runs on the final
  hidden state exposed by `language_model._last_hidden`; consumed by
  `RegularizationController.collect_symplectic_aux()`; auto-fired by
  `BRIANHarness._symplectic_aux_step()` after `_topo_charge_aux_step`.
- **TDD evidence.** 39 GREEN tests across 4 files:
  - `tests/dsl/test_liouville_symplectic.py` (14 tests) — math contracts:
    `QOnlyPotential` type rejection, `QuadraticPotential.energy` vs hand-
    computed, fp64 `det(J)=1` invariant (atol=1e-9), HLW H-oscillation
    bound for 1-D harmonic oscillator over 100 steps, mass positivity,
    Noether = (H_final−H_initial)², FD vs autograd grad on dtau.
  - `tests/dsl/test_liouville_symplectic_dsl_parse.py` (9 tests) — DSL
    parse surface: defaults, full block, field validation.
  - `tests/dsl/test_liouville_symplectic_regcontroller.py` (10 tests) —
    disabled/diagnostic/active paths, lazy-build reuse, odd-d_model grace.
  - `tests/dsl/test_liouville_symplectic_harness.py` (6 tests) — inert-
    gate `torch.equal` zero-contribution (noether_strength=0), keys set
    when enabled, backward from total does not crash.
- **Outcome.** ⏳ **PENDING DEPLOY.** No metrics captured yet — diagnostic
  active in arch, next training run needed. Per CLAUDE.md §1e the deploy
  requires explicit user authorisation; not initiated by this session.

### H26 — KJPLA-v2: Kuramoto-Josephson Phase Lattice Attention (2026-06-23)

**Hypothesis.** Equipping each attention head with a per-(head, layer, token)
phase φ and coupling phases across heads (intra-layer Kuramoto) and across
layers (inter-layer Josephson) provides a calibration signal that:

1. **Head diversity (Kuramoto sync).** Phases converge toward a shared mean
   when η > 0, but distinct phases survive when w_h encodes divergent content
   signals. We predict that PLV (Phase Locking Value) per head remains
   heterogeneous (std > 0.1) across heads after 2k steps — not collapsed.
2. **Josephson inter-layer coupling.** When K_h > 0, the optimizer is rewarded
   for maintaining a layer-to-layer phase stride of Δ_h = 2πh/(H·L). We predict
   the order parameter R_ℓ climbs toward 0.8+ over 5k steps (random init ≈ 0.0).
3. **OOD gap_ratio.** With josephson_strength > 0, we predict gap_ratio ≤ 4.9
   (H22 baseline: 6.55) at matched step count, motivated by the phase ordering
   inducing more structured cross-layer information routing.

- **Spec.** `architectures/master/arch.neuro` + `architectures/SmolLM/arch.neuro`
  — block `regularization { kjpla_phase_lattice: { enabled: true,
  josephson_strength: 0.0, entropy_strength: 0.0, eps_H: 0.5 } }`. Active in
  **DIAGNOSTIC MODE** — `josephson_loss` (= 0 at K_h=0) logged every step.
  Penalty mode (`josephson_strength > 0`) is a follow-up experiment.
- **Mechanism.** `neuroslm/mechanisms/kjpla.py` — `KJPLAttention` replaces
  `CausalSelfAttention` in the trunk. Phases init to zero (w_h=0, η=β_h=K_h=0)
  so step-0 loss is bit-identical to vanilla (`torch.equal`, not allclose). Phase
  stash is bfloat16 (memory discipline). delta_h is a non-persistent buffer
  (deterministic from n_heads·n_layers, not saved in state_dict).
  `josephson_loss` function computes L_J = −(1/L)Σ K̄_h · R_ℓ standalone
  (testable independent of forward). Consumed by
  `RegularizationController.collect_kjpla_aux()`; auto-fired by
  `BRIANHarness._kjpla_aux_step()` after `_symplectic_aux_step`.
- **TDD evidence.** 65 GREEN tests across 4 files:
  - `tests/dsl/test_kjpla_attention.py` (35 tests) — bit-identity at zero init
    (`torch.equal`), bfloat16 phi stash, delta_h non-persistent, Josephson
    R=1 when phi stride matches delta_h, FD vs autograd on beta_h (atol=1e-3),
    Kuramoto: eta=0 gives phi1==phi0, backward reaches K_h.grad, T=1/GQA
    boundary cases.
  - `tests/dsl/test_kjpla_dsl_parse.py` (18 tests) — DSL parse: defaults,
    full block, josephson_strength/entropy_strength/eps_H validation.
  - `tests/dsl/test_kjpla_regcontroller.py` (6 tests) — disabled/diagnostic/
    active paths, no-key-leak when disabled, backward from total.
  - `tests/dsl/test_kjpla_harness.py` (6 tests) — _kjpla_aux_step wiring,
    no-key-leak when disabled, backward from total.
- **Outcome.** ⏳ **PENDING DEPLOY.** Mechanism wired and diagnostic active;
  first training run needed to observe R_ℓ trajectory. Per CLAUDE.md §1e the
  deploy requires explicit user authorisation; not initiated by this session.

---

## DSL v2 — `model { }` block (2026-06-23, commit `0b65c00`)

**Type:** Infrastructure capability (not a falsifiable hypothesis; no Layer B artifact needed).

**What shipped.**

The DSL gained a first-class `model { ... }` block that declares any standard causal LM
independent of the BRIAN brain subsystem. The block is parsed by
`neuroslm/dsl/model_spec.py` into a `ModelSpec` + `SheafConfig` dataclass pair, and
`neuroslm/models/__init__.py::build_model(spec)` instantiates the corresponding
`nn.Module`.

| `kind` value | Maps to | Architecture |
|---|---|---|
| `gpt2` | `neuroslm/models/gpt2.py::GPT2Model` | fused QKV/Conv1D→Linear, learned pos_embed, LayerNorm, GELU |
| `llama` | `neuroslm/models/llama.py::LlamaModel` | RoPE, GQA, SwiGLU, RMSNorm (SmolLM2/LLaMA-family) |
| `qwen` | `neuroslm/models/llama.py::LlamaModel` | same architecture family as llama |
| `mistral` | `neuroslm/models/llama.py::LlamaModel` | same |
| `brian` | (reserved — BRIAN trunk with THSD mechanisms) | KJPLA, Noether, topo charge, etc. |

HF weight loading: each model module exports `hf_to_model_state_dict(hf_sd)` that
remaps HuggingFace parameter names to the canonical internal scheme.

**THSD framing.** Every LM is a cellular sheaf F where token hidden states are stalks
and the attention+FFN block is the coboundary operator δ: C⁰(F) → C¹(F). GPT-2 and
LLaMA-family models are *trivial* H¹ sheaves (no conservation laws, no THSD mechanisms).
BRIAN is a *non-trivial* sheaf with Noether residuals (H25), topo-charge diagnostics
(H24), and phase-lattice coupling (H26).

**New arch.neuro files.**

- `architectures/gpt2/arch.neuro` — GPT-2 124M (trivial H¹ sheaf)
- `architectures/smollm2-135m/arch.neuro` — SmolLM2 135M (trivial H¹ sheaf, LLaMA family)
- `architectures/qwen2.5-0.5b/arch.neuro` — Qwen2.5-0.5B (trivial H¹ sheaf, LLaMA family)

**Tests.** See `tests/dsl/test_model_spec*.py` (parse + round-trip) and
`tests/models/test_gpt2*.py` / `tests/models/test_llama*.py` (forward shape + HF loading).

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
- **SmolLM2 upgrade (H22 / B6) REGRESSED gap_ratio from 2.87 → 6.55** — larger, better-trained general expert makes the trunk train faster in-distribution but memorise more aggressively. The quality upgrade is real (train PPL 23.6 vs 45.0 at B5 step 3000) but the generalisation win is gone. Needs stronger regularisation to counteract the faster-fit dynamic.

### Open / not yet measured
- Same-params PCT eval against same-params ReZero / recursive baseline (matched-PPL test for H10).
- Full PCT-feedback mode (`pct_mode="feedback"`) vs loss-only.
- SRC-TEH wall-clock numbers (H11).
- Matched-compute baseline (step-7000 baseline) for H12.
- ~~**H21 / B4 10k-step run**~~ — **PARTIALLY RESOLVED** by H23 / B5: step 3000 shows gap_ratio 2.89 (stable, not accelerating). Full 10k result still pending.
- **Trunk anisotropy spike fix** — H21 follow-up #1, the 1100–1700 gradient spike is currently band-aided by `loss_clip(f=3.0)`. Open whether `cortex_pre_head_norm`-style LayerNorm on the *trunk* pre-head suppresses it, or whether the source is elsewhere (residual stream, attention out-projection).
- **Checkpoint schema-drift fix** — H21 follow-up #2, the resume crashed 8× with `Unexpected key(s)` for optional `_genetics_orch.*` / `_transmitter_sys.*` subsystems. Gated by either `strict=False` for these keys or wiring them on by default.
- **SmolLM2 + stronger regularisation** — H22 FALSIFIED on gap_ratio but the quality improvement is real. Try: higher dropout (0.20 → 0.25), stronger flooding (4.0 → 5.0), lower peak LR (5e-4 → 2e-4) to counteract the faster-fit dynamic introduced by the better expert.

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

### H27 — Rebalance distillation so the trunk learns standalone (2026-06-24)

**Status:** 🟠 **PENDING** — wired + arch retuned; full run not yet measured.

**Hypothesis.** On the 100m/seq-2048 run (vast 42397874, commit `2e2a5a08`)
the *fused* model trained well (WikiText OOD ppl ~80–93, < 100) but the
**standalone trunk did not learn** — trunk-only `ppl = exp(lm_loss_ema)`
stayed ~9k–12k and even *rose* (7.6 → 9.4 nats). Mechanism: in
`fusion_mode=additive_correction` the *detached* cortex (full coverage
after [[h27-bridge-coverage]] / commit `2e2a5a08`) carried the fused
output alone at `α=0.1`, so the trunk got ~no fused-loss gradient, and
the CFD teacher at `T=4` was soft enough that matching it left the trunk
blurry at `T=1`. Net: the trunk free-rode.

Predicted fix (this commit): raising the trunk's fusion weight and
sharpening the teacher should force the trunk to internalise the cortex's
knowledge, so **trunk-only ppl falls toward the fused OOD (~80)** rather
than diverging from it.

**Spec (this change set).**
- `architectures/SmolLM/arch.neuro` `multi_cortex.fusion_init`: `0.1 → 0.5`
  (5× the trunk's fused-loss gradient).
- `multi_cortex.distillation_temperature` + `LanguageCortex.cfd_temperature`:
  `4.0 → 2.0` (sharper teacher; transfers confident predictions, not just
  high-entropy mass).
- `neuroslm/train_dsl.py` `_ood_eval_logits`: the mid-OOD probe now
  evaluates the **standalone trunk** (`harness.language_model`, cortex
  dropped) so OOD ppl is consistent with the trunk-only train ppl and
  `gap_ratio` is meaningful again.

This is a **stack** (α + temperature changed together). Follow-up: single-
knob ablation (α-only vs T-only) once the stack shows direction.

**Run.** (pending — to be filled with vast id + trajectory after deploy.)

**Watch.** trunk-only `ppl` (should fall, not plateau ~9k); trunk-only
WikiText OOD (new metric); `gap_ratio` (should land O(1)); `cx_ema`
(teacher should stay ~4 — if it climbs, the sharper T is hurting the
target). Trajectory at steps 500/1000/2000/5000.

[EVIDENCE: tests/test_mid_ood_uses_lm_only_loss.py::TestOodEvalLogitsAreTrunkOnly]

### H28 — logits_mixture (not additive_correction) for a standalone trunk (2026-06-24)

**Status:** 🟡 **PARTIAL** — trunk now LEARNS (✓) but OVERFITS hard (✗). See result below.

**Falsified premise (H27).** H27 raised α (0.1→0.5) under
`fusion_mode=additive_correction` to make the trunk learn. Run **43125941**
(commit `ca709250`, A100, α_eff=0.501, T=2.0) showed the rebalance was
*active and the distillation healthy* (`kl=0.43`, `cx_ema≈4` — a good
teacher) yet the **trunk-only OOD ppl ROSE 24k→88k** (steps 500→1000) while
train trunk-ppl sat ~10k. The trunk diverged on held-out data.

**Diagnosis.** additive_correction is `fused = cortex.detach() + α·trunk`,
so the LM loss drives `α·trunk → target − cortex`, i.e. the trunk learns a
**residual** `(target−cortex)/α`, not a distribution. Standalone it emits
`(target−cortex)/α` = garbage (→ OOD 88k). Distillation simultaneously
pulls `trunk → cortex`; the two objectives **conflict**, and raising α only
sharpens the residual. additive_correction therefore *cannot* yield a
droppable trunk, at any α/T.

**Hypothesis.** `fusion_mode=logits_mixture`
(`fused = (1-α)·trunk + α·cortex`, α = cortex weight) makes the trunk own
`1-α` of the output → it learns the **full** prediction, and distillation
**reinforces** the same direction (`trunk → cortex ≈ target`). So the
trunk-only OOD should **fall** toward the teacher's quality (cx_ema≈4 → ppl
~50), and `inhibition_enabled` anneals α_eff→0 as the trunk catches up (the
automatic cortex-drop).

**Spec (this change set).**
- `architectures/SmolLM/arch.neuro` `multi_cortex.fusion_mode`:
  `additive_correction → logits_mixture`.
- `multi_cortex.fusion_init`: `0.5 → 0.3` (now CORTEX weight; trunk owns 0.7).
- Distillation T stays 2.0 (H27), inhibition stays on (drives the drop).

**Run.** vast 43133274 (A100, commit `887451b6`, α_eff=0.300, T=2.0, dropout=0).

**Result (MIXED).** The fusion fix worked: train trunk-ppl collapsed
4173 → 268 (steps 500→7500), lm_ema ~9 → ~5 (per-step ppl hit 145) — the
standalone trunk finally learns, which it never did under
additive_correction. BUT the trunk-only WikiText OOD EXPLODED 10.9k →
175k (gap_ratio 2.6 → 358); at 175k the OOD CE ≈ 12 nats > uniform 10.8,
i.e. the trunk is *confidently wrong* off-distribution — catastrophic
overfitting. Distillation transferred the teacher's training-batch outputs
(lm_ema → cx_ema≈4) by MEMORISATION, not its generalising function.
Contributing: dropout=0, wd=0.01, OOD regularisers inert (reg Σ≈0.001),
inhibition never engaged (inh=0 — lm_ema≈cx_ema ⇒ zero gap), 60% chat vs
prose OOD. Verdict: logits_mixture ✅ for learning; generalisation is now
the live blocker → H29.

[EVIDENCE: tests/training/test_expert_correction_fusion.py::TestSmolLMUsesLogitsMixture]

### H29 — Regularise the over-fitting standalone trunk (2026-06-24)

**Status:** 🟠 **PENDING** — first single knob (dropout) wired.

**Hypothesis.** H28's trunk overfits (train ppl 268 / OOD 175k). Standard
capacity regularisation should close the gap. Single knob first to isolate:
`dropout 0.0 → 0.1`. If insufficient, escalate wd (0.01→0.05),
stochastic_depth (0.0→0.1), and engage the PR2 OOD controller
(DAR/PCC/isotropy, currently Σ≈0.001).

**Spec.** `architectures/SmolLM/arch.neuro` `dropout: 0.0 → 0.1`.

**Watch.** trunk-only OOD ppl must come DOWN below uniform (CE < 10.8 /
ppl < 50k, ideally toward the teacher's ~30–50) while train ppl stays
reasonable; gap_ratio should fall from ~300 toward O(1–5).

> **Elegant-fix track (parallel).** dropout is the empirical stop-gap. The
> principled fix — transferring the teacher's *function* (its
> generalisation), not its training-point values — is being designed:
> Jacobian/Sobolev-matched distillation + logit-norm calibration. Will land
> as H30 once specced.

[EVIDENCE: architectures/SmolLM/arch.neuro (dropout=0.1)]

---

### H30 — Transfer the teacher's *function*, not its values (2026-06-30)

**Status:** 🟠 **SPLIT** — LogitNorm ❌ FALSIFIED-as-wired (run 43247602,
below), now DISABLED. Consistency-only 🟢 ARMED for the next deploy
(`logit_norm_tau=0.0`, `consistency_weight=1.0`, σ=0.1, subsample 1×512) —
the clean single-mechanism test per §10. Record the vast id +
step-{500,1k,2k,5k,full} trunk-only OOD trajectory here when it runs.

> **Run 43247602 (2026-06-30) — LogitNorm FALSIFIED as wired; disabled.**
> First deploy that trained (OOM fixes held). But stacking LogitNorm +
> consistency violated §10 (one mechanism/run) and LogitNorm broke learning:
> by step 500 the raw trunk CE was *stuck at uniform* (`lm` 10.61→10.05 vs
> ln 50257 = 10.82; H28 was ~5.6 here), `gnorm` 2.6→20.7, `Φ` 0.94→0.52, OOD
> ppl 95.9k (gap 16.3). Root cause: `_compute_loss_from_logits` normalises the
> TRAIN CE to `f/(τ‖f‖)` (magnitude-free), but `lm`/`ppl`/OOD are read off the
> RAW pre-fusion logits whose magnitude LogitNorm no longer constrains → raw
> softmax stays ~uniform, and the `1/‖f‖` gradient factor blows up as raw
> `‖f‖` stays small. So LogitNorm (a) makes the perplexity telemetry
> meaningless and (b) destabilises training. Not a drop-in guardrail — needs
> eval-side normalisation + a vocab-tuned τ (0.04→norm 25 is wrong for V=50k)
> + an inference temperature before it can be reused. Implementation kept
> (default-off, `logit_norm_tau=0.0`); contracts in
> `tests/test_logit_norm_calibration.py` still pass. The **consistency** term
> was untestable through the LogitNorm damage → re-run it ALONE next.

> **Run 43244973 (2026-06-30) — first armed deploy OOM'd; fixed, not yet
> re-run.** Booted clean on commit `55163b18` but died at step 0 with
> `CUDA OutOfMemoryError: Tried to allocate 1.54 GiB` inside
> `consistency_distill_loss`'s `log_softmax`. Two defects, both now fixed:
> (1) the loss built `softmax`/`log_softmax` over the whole `(B=4,T=2048,
> V=50257)` block — a 1.54 GiB fp32 spike, twice (teacher+student); fixed by
> chunking the token dim to `(chunk,V)`, mirroring
> `BRIANHarness._chunked_flat_ce`. (2) `kl_div(reduction="batchmean")` on the
> 3-D tensor divided by `B` only → loss `T`× (2048×) too large, would have
> diverged even if it fit; fixed to the per-token mean (÷ `B·T`). Now
> `consistency_weight=1.0` is comparable to `distillation_lambda_max=1.0`.
> Contracts: `tests/test_consistency_distill.py::TestConsistencyMemorySafety`
> (per-token-mean, chunk-invariant value + gradient).

> **Run 43245905 (2026-06-30) — second OOM, now in `backward`; fixed.**
> With the loss chunked, the forward fit but `scaled.backward()` died at the
> same 1.54 GiB: the consistency pass ran a *second* trunk forward over the
> whole `(B=4,T=2048,V=50257)`, whose logit GRADIENT is a 1.54 GiB fp32
> tensor that won't fit on top of the main step. Fix: probe a cheap
> `(batch, prefix)` subsample (`consistency_batch=1`,
> `consistency_max_tokens=512` → 1 seq × 512 tok/step) — a Jacobian-
> consistency estimator is unbiased on a subsample, reducing batch keeps each
> probe a real full-context sequence, capping tokens bounds the second
> forward's logits+grad to ~0.1 GiB. New pure helper
> `BRIANHarness._consistency_subsample`; contracts in
> `tests/test_consistency_distill.py::TestConsistencySubsample` (batch/token
> clamping, prefix semantics). Safe to redeploy.

**Hypothesis.** H28's catastrophe (train ppl 268 ✓ but OOD 175k, CE ≈ 12 >
uniform ln(50257)=10.82 — *confidently wrong* off-distribution) has a
precise mechanical cause: pointwise KL distillation transfers the teacher's
training-point **values** but not its generalising **function**, and CE on
raw logits actively *rewards* unbounded confidence. Two orthogonal,
principled fixes should each independently pull OOD CE back below uniform:

1. **Jacobian-consistency distillation** (Srinivas & Fleuret, ICML 2018).
   Add `L_consist = T²·KL(softmax(teacher(x)/T) ‖ softmax(student(x+δ)/T))`
   with δ Gaussian noise on the trunk's input embedding. Matching the
   teacher under input perturbation is the first-order equivalent of
   matching its input-Jacobian → it transfers the teacher's *local
   function*, so the student cannot spike to confidently-wrong values in the
   neighbourhood of a training point. This directly attacks the
   memorisation→OOD-explosion failure.
2. **LogitNorm calibration** (Wei et al., ICML 2022). Train CE on
   `f/(τ·‖f‖)` instead of `f`. The mechanism is **scale-invariance**:
   `logit_norm(c·f)=logit_norm(f)`, so the network can no longer lower its
   loss by inflating `‖f‖`. It stops manufacturing overconfidence → OOD CE
   is *capped near* (never above) uniform. A guardrail, complementary to (1).

**Spec.**
- `neuroslm/regularizers.py`: `logit_norm(logits, tau)` (Part 1, committed
  `69ddd36f`) and `consistency_distill_loss(teacher, student, T)` (Part 2,
  this commit).
- `neuroslm/dsl/nn_lang.py`: `DSLLanguageCortex.forward(ids,
  embed_noise_std=0.0)` — additive Gaussian embedding perturbation (σ=0 is
  an exact no-op).
- `neuroslm/harness.py`: `_compute_loss_from_logits` applies LogitNorm when
  `logit_norm_tau>0`; `_cortex_fusion_aux_step` runs a stash-preserving
  noised trunk forward and adds `consistency_weight·L_consist` when
  `consistency_weight>0` (both default-off).
- `neuroslm/dsl/training_config.py`: `logit_norm_tau: float = 0.0`,
  `consistency_weight: float = 0.0`, `consistency_noise_std: float = 0.1`.

**Watch (next deploy).** Enable in `architectures/SmolLM/arch.neuro`
(`logit_norm_tau ≈ 0.04`, `consistency_weight ≈ 1.0`, σ=0.1) and re-run the
H28 config. Trunk-only OOD ppl must fall below uniform (CE < 10.8) — target
gap_ratio O(1–5) — while train ppl stays reasonable. Single-mechanism
ablation backlog: deploy ① and ② separately to attribute the gain (per §10
stack-finding rule).

[EVIDENCE: tests/test_logit_norm_calibration.py (10 contracts: scale-invariance, norm=1/τ, argmax-preserved, wired-into-loss)]
[EVIDENCE: tests/test_consistency_distill.py (10 contracts: T²-KL teacher-detached, embed-noise hook, stash-preserving forward, off→noop/on→positive aux-step)]

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

---

## H-STE — Semantic Turbulence Engine (2026-06-21)

### Hypothesis

BRIAN's single-scale attention lacks the multi-resolution structure that
physical systems use to move information efficiently across scales. Three
physics-inspired mechanisms, wired together, can push the trunk beyond
the current gap_ratio floor:

1. **RG Cascade (H-STE-RG):** Partitioning the sequence into G groups
   at token scales 2^g and coupling them with Kolmogorov 5/3-law weights
   λ_g ∝ 2^{-5g/6} creates a turbulence-like multi-scale enrichment of
   the hidden states. Cost: ≈30% more attention compute; gain: 3× richer
   scale structure per token.

2. **GPE Phase Field (H-STE-GPE):** Encoding the VBB hidden state as a
   complex superfluid ψ ∈ ℂ^{d/2} and running imaginary-time GPE steps
   produces a semantic coherence order parameter ρ ∈ [0,1]. ρ→1 means
   the model has condensed onto a single meaning; ρ→0 means genuinely
   ambiguous context. ρ is the ideal signal for the P3 context gate
   (high ρ → trust experts more → low α).

3. **NT Criticality (H-STE-C):** Tracking the branching ratio σ = layer-
   to-layer Frobenius norm ratio and steering toward σ=1 (Beggs & Plenz
   critical point) maximises dynamic range and information transmission.
   NT signals generated: GABA (σ>1), NE (σ<1), DA reward (σ≈1).

**Predicted gains (conservative, assuming 0.5 compounding factor):**

| Stage | Mechanism | Gain |
|---|---|---|
| STE-A | NT criticality only | 1.5× OOD PPL |
| STE-B | Criticality + RG cascade | 2.0× OOD PPL |
| STE-C | Full STE (all three) | 2–4× OOD PPL |

### Spec

- **Commit:** `HEAD` (2026-06-21, STE initial implementation)
- **DSL block:** `architectures/SmolLM/arch.neuro` `semantic_turbulence { }` (line ~411)
- **Config:** `neuroslm/dsl/training_config.py::SemanticTurbulenceConfig`
- **Implementation:** `neuroslm/emergent/semantic_turbulence.py`
- **Harness wiring:** `neuroslm/harness.py::_build_semantic_turbulence()` + forward pass

### Tests (Layer A — CONFIRMED GREEN)

| Test file | Tests | Contract |
|---|---|---|
| `tests/dsl/test_semantic_turbulence_dsl.py` | 29 | DSL parser round-trips all fields |
| `tests/training/test_rg_cascade.py` | 16 | Kolmogorov λ_g ratios; coarse-grain shape; perfect reconstruction; Frobenius non-expansion; differentiable forward |
| `tests/training/test_gpe_phase_field.py` | 15 | Complex encode/decode lossless; GPE step reduces free energy; ρ∈[0,1]; ρ→1 for aligned phases; ρ→0 for random phases |
| `tests/training/test_criticality_control.py` | 18 | σ=1 for identity; σ>1 for amplification; EMA tracking; GABA↑ when σ>1; NE↑ when σ<1; DA↑ when σ≈1; criticality loss = weight×(σ-1)² |

**All 78 Layer-A tests confirmed GREEN on 2026-06-21.**

### Run (Layer B — PENDING)

Not yet deployed. Ablation protocol:

- **STE-A:** `enabled: true, criticality_weight: 0.01` (zero new params — pure loss term)
- **STE-B:** `enabled: true, n_rg_groups: 3` (adds ~5% params in RG projections)
- **STE-C:** `enabled: true` (full config — all three modules)

**Baseline:** B5 H21 row (`gap_ratio=2.89, train_ppl=45.0, ood_ppl=130.1` at step 3000).
**Target:** STE-C gap_ratio < 2.0 at matched training steps and parameter count.

### Mathematical grounding

**Module 1 — RG Cascade.** Kolmogorov (1941) showed that in isotropic
turbulence, kinetic energy spectrum follows E(k) ∝ k^{-5/3}. The coupling
λ_g ∝ 2^{-5g/6} mirrors this: coarser scales (smaller k in wavenumber
space) carry more energy, finer scales (larger k) carry less. The
Kolmogorov length `η = (ν³/ε)^{1/4}` sets the cutoff below which viscosity
dissipates energy — here n_rg_groups sets the analogous cutoff.

**Module 2 — GPE Phase Field.** The Gross-Pitaevskii equation for a
trapped Bose-Einstein condensate:

    iℏ ∂ψ/∂t = [-ℏ²∇²/2m + V(r) + g|ψ|²]ψ

In imaginary time (t → -iτ), this becomes gradient descent on the
Ginzburg-Landau free energy F[ψ] = ∫(|∇ψ|²/2m + V|ψ|² + g|ψ|⁴/2)dr.
The equilibrium superfluid (condensate ground state) has |ψ|² = const
and uniform phase — exactly the "semantically unambiguous context" we want.

**Module 3 — Neural Criticality.** Beggs & Plenz (2003) showed that
avalanche size distributions in rat cortex follow power-laws (P(s) ∝ s^{-3/2})
only near the branching ratio σ=1, where each neuron triggers on average
one other neuron. At σ=1: maximal dynamic range, longest correlation
length, maximum information transmission. The NT loop (GABA/NE/DA)
mirrors the biological neuromodulator control of cortical excitability.

### Follow-up required

1. Deploy STE-A (criticality only) as zero-cost ablation.
2. Compare STE-A vs H21 baseline at same training budget.
3. If STE-A confirms σ-drift, deploy STE-B and STE-C.
4. Measure ρ trajectory — should climb as training progresses (semantic condensation).

### Outcome

🔵 **PENDING** — awaiting first training run.

### H31 — NGL: a Turing-complete evolvable language that discovers ML algorithms on CPU (2026-07-07)

**Status:** 🟢 **CONFIRMED (optimizer discovery)** / 🟠 **INCONCLUSIVE
(flow-modulation/EI)** — the language, its exact-match optimizer library, the
genetic operators, and the CPU discovery harness all land green (35 contracts,
`tests/genetic/`). A cold-start search rediscovers-and-tunes SGD on a convex
problem, and a seeded search *selects the adaptive-normalization structure* and
beats SGD by 75% on non-convex parity. The effective-information-driven
flow-modulation search runs end-to-end but did not find a high-synergy rule in a
small CPU budget — recorded as a negative result with a follow-up.

**Hypothesis.** The architecture DSLs (Layers A–D) cannot express an ML
*algorithm* — they have no persistent state or control — so "search the language
space for a novel mechanism" is not tractable on them. A typed **register-machine
language** (NGL, the AutoML-Zero / Lion-discovery substrate) can express
optimizers, learning rules and flow-modulation as evolvable programs, and a
Pareto GA over that space, benchmarked with tiny CPU models, will recover the
*structure* a task needs (not just tuned scalars). Expected signal: discovered
update rules reach ≤ SGD final loss on a held tiny task; on a task where SGD
provably plateaus (parity), the search selects an adaptive rule that clears it.

**Spec.** New package `neuroslm/genetic/` (design in
`docs/dsl_subsystem_roadmap.md` §NGL):
- `language.py` — register machine: scalar+tensor banks, ~35 total-semantics ops
  (`REGISTRY`), `Program`, `semantic_vector()`. Execution is total (eps-guarded
  div, abs-folded sqrt/log, shape-fallback matmul) and memory-capped
  (`_MAX_ELEMS`) so blind mutation never crashes or OOMs.
- `optimizer.py` — SGD / Momentum / RMSProp / Adam / Lion encoded as NGL programs
  + `NGLOptimizer` torch adapter. Each reproduces its reference **bit-for-bit**
  (`tests/genetic/test_optimizer.py`, atol 1e-6/1e-5) — the proof NGL spans the
  update-rule grammar.
- `evolve.py` — `mutate` / `crossover` / `random_program`, all-maximised
  `Objective` + `pareto_front`, `auto_evolve` GA (tournament + elitism + optional
  novelty in semantic space).
- `discovery.py` — `benchmark_optimizer` (trains a tiny MLP), `run_optimizer_
  discovery`, `run_flow_modulation_discovery` (EI proxy via
  `information.net_synergy`).
- CLI: `brian discover optimizer|flow` (`neuroslm/cli.py::cmd_discover`).

**Runs (CPU, seconds each; artifacts under `results/discovery/`).**

| Run | seeds | task | SGD base | discovered | Δ | winning structure |
|---|---|---|---|---|---|---|
| `optimizer_from_scratch_s0` | SGD+random | regression (convex) | 0.4551 | **0.1577** | **−65.3%** | scaled-grad, lr≈0.31, cost 1 |
| `optimizer_from_scratch_parity_s1` | SGD+random | parity (non-convex) | 0.6982 | 0.6845 | −2.0% | scaled-grad, lr≈2.14, cost 1 (plateau) |
| `optimizer_seeded_parity_s1` | SOTA+random | parity (non-convex) | 0.6982 | **0.1726** | **−75.3%** | **RMSProp-family adaptive norm, lr≈0.027, cost 9** |
| `flow_modulation_s0` | identity+random | parity | (SGD 0.69) | 0.6409 | — | degenerate constant update, synergy≈0.0 (inconclusive) |

**Outcome.**
- ✅ The bit-exact optimizer library confirms NGL is expressive enough to *be*
  the SOTA optimizers — the precondition for searching their neighbourhood.
- ✅ The cold-start regression run shows genuine search: with only SGD@lr=0.02 +
  random seeds it found the cheapest possible rule (`update = −0.31·g`) and cut
  loss 65% — it discovered the optimal step size, not a memorised constant.
- ✅ The **key structural result**: on parity, *no* single scaled-gradient rule
  escapes the ~0.69 (random-guess) plateau (best cold-start = −2%), but when the
  adaptive-normalization structure is in reach the search selects and tunes it to
  0.173 (−75%). The mechanism parity needs — per-coordinate gradient
  *modulation* — is exactly what the language search recovers. This is the
  "find modulation that outperforms" claim, demonstrated in miniature.
- 🟠 The EI-driven flow-modulation search is a working scaffold but did not
  surface a high-synergy rule in 12 generations; the winner was a degenerate
  scalar-broadcast update (synergy ≈ 0). Negative result per §10.7.

**Follow-up.**
1. Flow-modulation: larger budget + a denser EI signal (per-layer synergy,
   `pid_synergy` unique/redundant atoms) + seed with real modulation motifs
   (`mechanics/nfo.neuro`, divisive normalization) before concluding.
2. Promote a confirmed discovered rule into `learned_opt.py` / the training loop
   and measure OOD-ppl / gap_ratio on a real `brian deploy` run — the bridge
   from CPU discovery to the GPT-2-param-matched goal.
3. Extend the benchmark from tiny MLP to a 1–2 layer DSL `LanguageCortex` so the
   discovery objective can include throughput (tok/s) and Φ directly.

[EVIDENCE: tests/genetic/ — 35 contracts green (language/optimizer/evolve/discovery/cli)]
[ARTIFACT: results/discovery/optimizer_seeded_parity_s1.json (RMSProp structure, −75% vs SGD)]
[ARTIFACT: results/discovery/optimizer_from_scratch_s0.json (cold-start lr discovery, −65%)]

### H32 — NGL grows an arch bridge, a verified algebraic simplifier, and a neuroanatomic auto-evolve (2026-07-07)

**Status:** 🟢 **CONFIRMED** — three capabilities land on the NGL substrate (H31),
all TDD, 71 contracts green in `tests/genetic/`:

1. **Arch → NGL compilation.** `neuroslm/genetic/compile_arch.py` lowers an
   `nn_lang` forward graph (Layer A SSA tensor DAG) into an NGL `Program`: each
   SSA value → a register, each op/binop → one instruction, params → pre-bound
   tensor registers. The composite NN ops (`linear`, `rmsnorm`, `layernorm`,
   `swiglu`, `gelu`, `embedding`) were added to the NGL registry, delegating to
   the canonical `nn_ops` atoms, so lowering is near 1:1. **Byte-equivalence** is
   the contract: an FFN block (`rmsnorm → swiglu → residual`) run as an NGL
   program equals the compiled `nn_lang` module's forward, atol 1e-6. Scalar-config
   ops (attention's `n_heads`) raise `UnsupportedLowering` rather than
   miscompile. Now discovery/simplification can run on the *actual architecture*.

2. **Verified algebraic simplifier.** `neuroslm/genetic/rewrite.py` builds an
   expression DAG from a program (forward symbolic eval, register-reuse-correct),
   applies value-preserving algebraic identities to a fixpoint (`add-0`, `sub-0`,
   `mul-1`, `neg-neg`, `transpose²`, `(a+b)-b → a`, `cscale` constant folding,
   like-term combination `a·x + b·x → (a+b)·x`), and lowers back with CSE.
   **Every accepted rewrite is globally probe-verified**, so a shape-dependent
   identity that doesn't hold is rejected, never miscompiled. Integrated into
   `simplify()` (DCE → peephole → algebra → try-delete). Demonstrated end-to-end:
   a bloated compiled FFN (`h=rmsnorm; z=h+h; h2=z-h; m=swiglu; dead=gelu(m);
   return x+m`) simplifies **6 → 3 instructions** — dead code removed AND
   `(h+h)-h` collapsed to `h` — behaviour preserved on probes
   (`results/discovery/simplify_bloated_ffn.json`).

3. **Neuroanatomically-constrained trunk auto-evolve.**
   `neuroslm/genetic/neuro_evolve.py` evolves an NGL neuromodulation injected into
   a tiny CPU LM's residual stream (`h ← h · g(h)`), fitness = Pareto
   `(−val_PPL, +neuroanatomic_plausibility)`. The realism prior is grounded in
   canonical neuromodulation: divisive normalization (Carandini & Heeger 2012),
   multiplicative gain (Salinas & Thier 2000), saturating/homeostatic
   dose-response (bounded ops), metabolic economy (program length) — and it scores
   the *dead-code-eliminated* program so vestigial ops earn no credit. Run
   (`results/discovery/trunk_modulation_s0.json`): a bounded `tanh` gain reduces
   validation PPL **8.556 → 7.755 (−9.4%)** at plausibility 0.60, and the Pareto
   front makes the realism/PPL trade explicit (ppl 7.72 @ plaus 0.55; 7.76 @ 0.60;
   8.83 @ 0.60). CLI: `brian discover trunk`.

**Hypothesis (trunk).** SmolLM's trunk PPL is too high; the neuroscience lever is
to reshape the residual stream with a biologically-motivated gain (neuromodulation)
rather than scale parameters. A Pareto search over NGL modulations, constrained by
a neuroanatomic-realism prior, will lower a tiny-LM's val PPL without leaving the
realistic-mechanism region. **Confirmed in miniature** (−9.4% on the tiny CPU LM,
bounded-gain motif selected).

**Honest scope.** Competitive SmolLM PPL is a **GPU** claim, not a CPU one — the
tiny-LM result is the *engine* working, not the trunk fixed. The path to cash it:
promote the discovered gain law into the real trunk's residual stream and measure
train/OOD PPL on a `brian deploy` run. Recorded as follow-up.

**Follow-up.**
1. Wire the best trunk modulation into `neuroslm/harness.py` / the SmolLM trunk
   behind a default-off flag; A/B on a real deploy (§10 loop) — the CPU→GPU bridge.
2. Lower attention (scalar-config ops) into NGL so a *full* TransformerBlock, not
   just the FFN, is compilable/simplifiable — needs config-arg support in
   `compile_arch`.
3. Extend the algebraic rule set toward an e-graph (equality saturation) for
   distributivity / factoring that greedy rewriting misses.

[EVIDENCE: tests/genetic/ — 71 contracts green (adds compile_arch/rewrite/neuro_evolve)]
[ARTIFACT: results/discovery/trunk_modulation_s0.json (bounded-tanh gain, −9.4% val PPL)]
[ARTIFACT: results/discovery/simplify_bloated_ffn.json (compiled FFN 6→3 instructions)]

### H33 — All mechanics are lowerable/simplifiable/evolvable; modulations persist as .neuro (2026-07-07)

**Status:** 🟢 **CONFIRMED** — 81 contracts green in `tests/genetic/`. Three
additions close the loop from "NGL can express optimizers" to "NGL is the
substrate every model mechanic lives in, and discovered modulations are managed
artifacts."

1. **Full-mechanic lowering (attention → NGL).** NGL `Instruction`s gained an
   optional `config` tuple and `OpSpec.uses_config`, so scalar-config ops lower
   as opaque nodes. `causal_self_attention` is registered (delegating to
   `nn_ops`), and `compile_layer_to_ngl(source, bindings=…)` splits tensor vs
   config args. The **entire `TRANSFORMER_BLOCK_DSL`** (attention + residuals +
   swiglu) now compiles to NGL **byte-equivalent** to the reference module
   (atol 1e-5). The block is therefore simplifiable and evolvable, not just the
   FFN. A bug where the DSL's `x = x + a` reassignment made `input_regs` point at
   an interior temp was fixed (snapshot inputs before lowering); `config` is
   preserved through `to_source`, the peephole pass, and the algebraic
   rewriter's `to_expr`/`from_expr` (attention stays intact through simplify).

2. **Shape-correct simplification verification.** `compile_arch.make_probes`
   instantiates the reference layer to get real param shapes + a valid input, and
   `simplify(..., probes=…)` verifies every rewrite against those **non-degenerate**
   values (the generic all-zero probes made any rewrite look equivalent on an arch
   program). The attention op correctly survives simplification (opaque, not
   deletable); residual simplification around it is verified real.

3. **Modulation store — `modulations/*.neuro`, managed by `brian`.**
   `neuroslm/genetic/modulation_store.py`: an NGL `Program` serializes to a
   `.neuro` `modulation { program { … } }` block (round-trip exact, incl. config
   ops), and `ModulationStore` gives save / list / show / drop / **merge**
   (compose gains sequentially, `g₂(g₁(h))`). `brian discover trunk --save NAME`
   persists the discovered gain law; `brian modulation {list,show,drop,merge}`
   manages the store. So a discovered neuromodulation is a versionable artifact
   that can be merged or thrown away, not a log line.

4. **Novelty + efficiency search.** `run_optimizer_discovery(novelty_weight=…)`
   (CLI `--novelty`) adds semantic-space distance to the objective, hunting
   *novel* rules rather than seed variations; the cost objective and the trunk
   prior's metabolic-economy term already reward *more efficient* mechanics, and
   the simplifier turns any program into its cheapest equivalent. "Search for
   novel or more efficient algorithms" is thus first-class.

**GPU guardrail (explicit).** The CPU path discovers *candidate* mechanics,
modulations, and update rules and proves them on tiny models. A **param-matched
GPT-2 competitor comes ONLY from GPU exploration + extensive GPU training**
(`brian deploy`). `brian discover trunk` prints this, and `--save` is the bridge:
persist the gain law on CPU, wire it into the trunk, cash it on GPU. No CPU run
claims a competitive model.

**Follow-up.**
1. `brian deploy --modulation NAME` — inject a stored modulation into the SmolLM
   trunk's residual stream for a real GPU A/B (the CPU→GPU bridge; needs explicit
   deploy authorization per CLAUDE.md §1e).
2. Lower a whole stacked `LanguageCortex` (embedding → N blocks → head), not just
   one block, so end-to-end arch search/simplification runs.

[EVIDENCE: tests/genetic/ — 81 contracts green (adds compile_attention/modulation_store/cli_modulation)]

### H34 — Abstraction (macros), searchable attention, compiler passes, prior-art gate (2026-07-07)

**Status:** 🟢 **CONFIRMED** — 120 contracts green in `tests/genetic/` (chunked;
the full suite OOMs in one process on a 16 GB box — an env limit, not a defect).
Four extensions raise the ceiling on *what* the search can discover:

1. **Macros / ADFs** (`neuroslm/genetic/macros.py`). Reusable sub-programs called
   via a `call` op; `expand_macros` inlines them (fresh temps, copy-in input
   isolation so a macro writing its input never clobbers the caller, cycle guard).
   `Program.library` makes execution transparent (auto-flatten). `mutate(library=)`
   grafts a whole macro as one gene; `auto_evolve(macro_library=)` threads it
   through the GA. This is the abstraction lever for building complex algorithms
   from chunks. `discover optimizer --macros`.
2. **Attention as primitives** (`attention_primitives.py`). New axis-aware ops
   (`softmax_last`, `l2norm_last`, `causal_mask`) let single-head causal attention
   be written as an NGL program that matches a torch reference **bit-for-bit** — so
   the attention *mechanism* becomes mutable/searchable (drop QK-norm, swap the
   score fn, add a gate), not just rewireable from outside.
3. **Compiler passes** (`rewrite.py`): explicit `cse`, `constant_fold`, and a
   unified `optimize` pipeline on top of DCE + the algebraic rewriter.
4. **Prior-art gate** (`known.py`). Known algorithms (the SOTA optimizers + the
   trivial gradient/backprop rule) matched in hyperparameter-invariant semantic
   space; `discover optimizer --avoid-known` penalizes rediscovery so budget goes
   to genuine novelty. This directly answers "don't rediscover backprop/Adam."

**Outcome.** The engine now has the four things that gate *complexity + novelty*:
abstraction (macros), mechanism-level search (attention primitives), clean IR
optimization (compiler passes), and a novelty gate (prior art). Honest limit:
these raise the ceiling; certifying a discovered algorithm as research-grade still
needs a multi-task validation ladder + GPU-scale search (H32/H33 boundary).

[EVIDENCE: tests/genetic/ — test_macros(8), test_attention_primitives(5), test_compiler_passes(8), test_known(6) green]

### H35 — Flow+compute heat, geometric topology analysis, modulation auto-push (2026-07-07)

**Status:** 🟢 **CONFIRMED** — the "record/measure information flow, find bottlenecks,
search low-hanging first" spine (15 new contracts). Note: the repo's existing
`neuroslm/evolution/` subsystem (TrainingHeatmap + grad-norm heat + propose→gate
keep-if-better loop, wired into `train_dsl.py`/`harness.py`) already does online
mutation-during-training; these add the *flow/compute* signal + *geometry* it lacked.

- **Flow+compute profiler** (`profile.py`). A `recorder` hook on `Program.execute`
  captures, per op, **information flow** (output norm) and **compute** (estimated
  FLOPs). `ExecutionProfile` ranks `heavy_compute`, `hot_flow`, and — the search
  signal — `low_hanging` (high flow / low compute: cheap edges with big effect).
  Serialisable (`to_dict`) for visualization.
- **Geometric topology** (`topology.py`). Projects the profile into a weighted
  DiGraph and runs graph theory: betweenness (bottleneck routing), articulation
  points (cut vertices), max-flow/min-cut (tightest info bottleneck), algebraic
  connectivity (spectral integration). `propose_edits` turns geometry into
  structural suggestions (bypass/parallelise a bottleneck, prune a
  high-compute/low-flow edge). This is the high-leverage alternative to a literal
  fluid-flow simulation (deliberately *not* built — graph theory gives the same
  bottleneck/flow signal at a fraction of the cost).
- **Modulation auto-push** (`modulation_pusher.py`). `push_modulations` commits +
  pushes *only* `modulations/*.neuro` (scoped, best-effort, never raises) so a long
  Colab/vast run streams discoveries to git. `discover trunk --save --push`; the
  Colab explore cell exposes `PUSH`.
- CLI: `brian discover profile --layer-file X --binding D=16 …` prints the
  compute/flow/low-hanging rankings + bottlenecks + proposed edits (JSON via `--out`).

**Worked example** (`discover profile` on an FFN block): swiglu flagged as the
betweenness bottleneck → "parallelise"; rmsnorm flagged heavy-compute/low-flow →
"prune candidate"; low-hanging = the residual add (high flow, cheap).

**Follow-up (unbuilt, by choice).** Wiring NGL modulation discovery in as the
`EvolutionLoop` proposer (online evolve-during-training with the flow/compute heat
steering *which* pathway to search) is the natural next step — the gate + heatmap
already exist; this batch built the flow/geometry signal that would make it smart.

[EVIDENCE: tests/genetic/ — test_profile(6), test_topology(6), test_modulation_pusher(3) green]

### H36 — Exploration wired into training + a persistent search ledger (2026-07-07)

**Status:** 🟢 **CONFIRMED** — the online search loop + cross-run dedup (10 new
contracts). Complements the repo's existing `neuroslm/evolution/` loop by adding
an NGL-modulation explorer with a keep-if-better A/B gate and a *persistent*
ledger.

- **Persistent SearchLedger** (`ledger.py`). Each searched program gets a
  hyperparameter-invariant **semantic signature** (hash of its quantized semantic
  vector), stored to JSON. `has_searched`/`is_dud`/`record`/`stats`; a fresh run
  loads the accumulated history and **skips duds** (patterns already tried that
  didn't help) instead of re-searching them. Records dedup by signature (latest
  outcome wins, count accumulates).
- **TrainingExplorer** (`training_explorer.py`). Fires every `explore_every` steps;
  runs a short NGL modulation search (skipping ledger duds, counting them),
  A/B-tests the winner against the identity baseline via a caller-supplied
  `score_fn`, keeps it only if the metric improves, and records every attempt to
  the ledger. `run_training_with_exploration` is the runnable miniature: a tiny CPU
  LM whose residual-stream modulation is searched every N steps and installed only
  if it lowers validation ppl. Model-agnostic — the same explorer attaches to the
  real trunk by supplying a `score_fn` that applies a modulation to the trunk and
  returns a val metric.
- CLI: `brian discover explore --explore-every 500 --ledger PATH` runs the loop and
  prints per-exploration keep/reject + `skipped_duds`; `brian discover ledger`
  inspects/clears the ledger.

**Worked run.** Run 1: explorations at steps 500/1000 (one KEPT), ledger grew to 46
patterns. Run 2 (same ledger): loaded 46 prior patterns and **skipped 2 duds** it
had already searched, no re-work. Honest caveat: the tiny-LM metric is noisy
(modulations are A/B'd on the live model without retraining), so the *mechanism* is
what's demonstrated; the clean improvement signal needs the GPU trunk.

**Follow-up.** The real-trunk integration is one call in the training loop
(`explorer.maybe_explore(step, trunk_score_fn)`) + a `score_fn` that injects a
modulation into the SmolLM residual stream and evaluates a val batch — validated on
a `brian deploy` run.

[EVIDENCE: tests/genetic/ — test_ledger(6), test_training_explorer(4) green]

### H37 — Seed from the arch's baseline, push all artifacts, illuminate the manifold (2026-07-07)

**Status:** 🟢 **CONFIRMED** — 17 new contracts. Three additions: start the search
from the model's *current* algorithm, stream *all* artifacts during runs, and
quality-diversity search over the semantic manifold.

- **Baselines + seed-from** (`baselines.py`). The standard optimizers as NGL
  programs with explicit tradeoffs (per-step cost, optimizer-state memory,
  stability): sgd(cost 1/mem 0), momentum(3/1), rmsprop(9/1), **adam(27/2, the
  trunk default)**, lion(8/1). `run_optimizer_discovery(seed_from=["adam", ...])`
  starts the population from the arch's current algorithm(s) and searches outward.
  CLI `discover optimizer --seed-from adam,lion`; `discover baselines` prints the
  tradeoff table.
- **Artifact auto-push** (`modulation_pusher.push_artifacts`). Generalized from the
  modulation pusher: commits+pushes an explicit set of paths (modulation store +
  `.neuro/search_ledger.json` + run JSONs), scoped so it never sweeps unrelated
  changes. `discover explore --push` streams the ledger + modulations back to git
  during a Colab/vast run.
- **Quality-diversity (MAP-Elites)** (`qd_search.py`). The honest, computable core
  of "let algorithms emerge from the mathematical shapes of the semantic manifold":
  each program projects to a low-dim structural **descriptor** (length × op-family
  diversity — the manifold coordinates); MAP-Elites keeps the best performer per
  cell and iterates, *illuminating* the space into a diverse zoo of high-performing
  algorithms across shapes (cheap-simple in one region, deep-adaptive in another).
  `discover qd --task parity --iters N` illuminated 11 shape-cells on a smoke run.

**On the grander framing.** The idea of "discovering undiscovered mathematics at a
semantical manifold via fluid-flow perturbations" is aspirational; its *computable*
realization is (a) MAP-Elites illuminating the program manifold (built here) and
(b) the flow/topology perturbation analysis from H35 (`profile.py`+`topology.py`).
Together they let novel, efficient algorithms emerge across the geometry and let us
measure how perturbing neural flow changes emergent metrics — without a literal,
low-ROI fluid simulation.

[EVIDENCE: tests/genetic/ — test_baselines(6), test_qd(5), test_modulation_pusher(+2)=5 green]

### H38 — Per-arch/preset run heatmaps + rebase-robust artifact push (2026-07-07)

**Status:** 🟢 **CONFIRMED** — 12 new contracts. Two concrete additions plus a
live-bug fix.

- **Per-arch/preset heatmap** (`heatmap_store.py`). Every training checkpoint now
  records where gradient heat concentrated, namespaced to `heatmaps/<arch>/
  <preset>.json` (latest run wins). `HeatmapStore` + `record_training_run` (reuses
  `evolution.grad_heat.parameter_grad_norms`), wired into `train_dsl.train()` at
  the save point behind a guarded try/except (can never break training). The
  summary ranks the hottest pathways — the map that shows *where* a wild gnorm
  lives.
- **Rebase-robust pushers** (`modulation_pusher.push_artifacts`). The Colab
  log/artifact pusher was failing with `! [rejected] (fetch first)` whenever
  master advanced under it (concurrent code/log/artifact pushes). `push_artifacts`
  now fetch+rebase+retries on rejection; the Colab notebook log-pusher got the same
  fix and also streams `heatmaps/` + `modulations/` + `.neuro/search_ledger.json`
  alongside logs during a run.

[EVIDENCE: tests/genetic/ — test_heatmap_store(6), test_modulation_pusher(+1 concurrent)=7 green]

### H39 — Seed the ledger with known algorithms + optimize commonly-used mechanics (2026-07-07)

**Status:** 🟢 **CONFIRMED** — 10 new contracts. The explorer skips known spaces,
and the compiler passes are turned on the common mechanics to find reductions.

- **Prior-art seeding** (`known.seed_ledger_with_known`). Records every known
  algorithm/mechanic as an NGL program in the persistent ledger with
  `outcome="known"` → `SearchLedger.is_dud` returns True, so the training explorer
  and discovery search treat them as already-explored dead space and spend budget
  only on *novel* mechanics. Covers the SOTA optimizers, the macro building blocks
  (divisive_norm, rms_scale, sign_interp, bounded_gain), and canonical modulation
  motifs (identity/tanh/sigmoid gain). Idempotent (dedups by signature), persists
  across runs. CLI: `discover ledger --seed-known`; `discover explore` auto-seeds
  by default (`--no-seed-known` to disable).
- **Mechanic optimizer** (`mechanic_optimizer.py`). Runs the full compiler
  pipeline (DCE → CSE → constant-fold → algebraic → probe-verified try-delete) on
  commonly-used mechanics and reports whether each can be reduced. Also detects
  subexpressions **shared across** mechanics (compute-once targets). CLI:
  `discover optimize-mechanics`.

**Worked run** (`discover optimize-mechanics`): momentum 3→2, rmsprop 9→7,
**adam 27→22, lion 8→4** — the superoptimizer found real redundancy in the
hand-written encodings, behaviour probe-verified. Shared subexpressions:
`cscale(t2,c=0.9)` across adam/lion/momentum, `square(t0)` across adam/rmsprop,
and the β₁ interpolation shared by adam/lion — genuine factor-once opportunities.

[EVIDENCE: tests/genetic/ — test_seed_known(5), test_mechanic_optimizer(5) green]

### H40 — Semantic-description language, full mechanic catalog, shared-macro lift (2026-07-07)

**Status:** 🟢 **CONFIRMED** — 42 new contracts (23 semantics + 11 catalog + 8
shared-macro). NGL programs now carry a *machine-checked meaning*, the whole
research-mechanic corpus is enumerable, and a shared computation is factored once
and reused everywhere.

- **Static semantic analysis** (`semantics.py`). Abstract interpretation over a
  boolean value lattice (`bounded / nonneg / normalized / sign_only / mixes`):
  each op has a transfer function, and `analyze()` folds them into a
  `SemanticSummary` — `role`, `bounded`, `normalizing`, `elementwise`,
  `sign_based`, `stateful`, plus the inferred `inputs`/`state` registers. Role and
  state fall out of the analysis rather than being declared: a read-modify-write
  buffer ⇒ `stateful` ⇒ `optimizer_update`; `softmax_last`+`matmul` ⇒ `attention`;
  a lone bounded nonlinearity ⇒ `activation`. `describe()` renders the summary as
  the human-facing "what it does / when to use it" language; `interchangeable(a,b)`
  is the substitution gate — same role **and** matching abstract output contract
  (a bounded activation is *not* interchangeable with an unbounded one, even
  though both are activations). This is the formal layer that guides CSE /
  mechanic-reuse: only interchangeable mechanics may be swapped. CLI:
  `discover semantics --known adam` → "Role: optimizer_update … stateful (buffers:
  t2, t3, s0)".
- **Full mechanic catalog** (`catalog.py`). `MechanicCatalog` loads **all 74**
  `mechanics/` + `dynamics/` + `structures/` `*.neuro` specs through the existing
  `mechanic_parser` — the rich `summary` / `when_to_use` / `not_for` / `properties`
  blocks already in the repo *are* the semantic-description surface, so "all
  currently existing research mechanics" is a live enumeration (12 categories:
  attention, normalization, position, sequence_mixer, routing, feedforward,
  physics, …), not a hand-kept list of 13. `catalog_names()` feeds the novelty
  gate. CLI: `discover mechanics [--category attention] [--describe rope]`.
- **Shared-macro lift** (`shared_macros.py`). `extract_shared_as_macros(mechanics)`
  finds multi-op subexpressions common to ≥2 mechanics, lifts each into a `Macro`,
  and rewrites every mechanic to `call` it — CSE promoted from *inside one program*
  to *across the whole mechanic set*, so an improvement discovered in one algorithm
  is reused in **all** algorithms that share the subexpression. Every rewrite is
  probe-verified (expand → `programs_equivalent`); anything unsafe to factor (a
  register re-written by an optimizer's state buffer, or an intermediate read
  outside the cone) is left untouched. `promote_modulation(store, name)` stamps a
  modulation validated through training/ablation as the new reference
  implementation (`is_reference`). CLI: `discover extract-shared`.

Answers the "13 algorithms is too low — we need **all** research mechanics" ask
(74 enumerated), the "semantical description language … based on static semantic
analysis / abstract interpretation" ask (semantics.py + the catalog's when_to_use
surface), and the "extract improvements into reusable subexpressions reused in all
algorithms … validated modulations become the reference implementation" ask
(shared_macros.py).

[EVIDENCE: tests/genetic/ — test_semantics(23), test_catalog(11), test_shared_macros(8) green]

### H41 — Semantic normalization: canonicalize equivalent expressions before search (2026-07-07)

**Status:** 🟢 **CONFIRMED** — 13 new contracts. Syntactically-different but
semantically-identical NGL programs are reduced to one canonical form and
substituted, so exploration never re-searches a rewrite of something already seen.

- **Normalization pass** (`normalize.py`). `canonical_form()` reduces a program to
  its normal form under the repo's convergent rewrite system (`optimize` +
  `simplify` to a fixpoint); `semantic_signature()` is that form's structural key
  (identical key ⇒ rewrite-provably equal — the decidable core). `normalize_
  semantics(programs, counts, prefer)` clusters by two verified layers — exact
  canonical-signature (rewrite-equal) then probe-equal merges — and substitutes one
  canonical representative per class: the most-used (`prefer="frequency"`) or the
  lowest-complexity (`prefer="simplest"`, where complexity = `(size, branches,
  distinct_ops)`, a straight-line-DAG proxy for cyclomatic complexity).
- **Runs after semantic labels, before exploration.** Probe-merges are gated by the
  `semantics.analyze` role label **and** statefulness. Wired into `TrainingExplorer`
  (`ExploreConfig.normalize=True`): each candidate is canonicalized before the
  ledger's dud-skip / signature dedup, so all syntactic variants collapse to one
  ledger entry. CLI: `discover normalize [--prefer simplest]`.
- **Soundness fix caught in the loop.** The first run collapsed `sgd ⇐ momentum`:
  momentum's state buffer reads as zero on a single-shot probe, so it *looks* like
  sgd on one step though they diverge across steps. Fix: never probe-merge a
  program `semantics.analyze` labels `stateful` — single-shot observational
  equivalence is only a sound equality witness for pure programs. Regression test
  `TestStatefulSoundness` locks it in; the corpus now collapses only the genuinely
  equal `tanh_gain ⇐ bounded_gain` (both `tanh(x)`).

**Theory (the honest boundary).** A normal form that is topologically identical for
*all* semantically-equal programs regardless of algorithm is impossible in general:
program equivalence is undecidable (Rice's theorem; reduction from halting), and
the minimal program for an intent is uncomputable (Kolmogorov complexity). What is
possible — and what this pass is — is canonicalization *relative to a fixed theory*:
a terminating + confluent rewrite system gives unique normal forms (Church–Rosser),
and e-graph-style extraction picks the lowest-cost member of an equivalence class.
On NGL's total, side-effect-free, analytic fragment we additionally get behavioural
equivalence by finite probing (polynomial identity testing / Schwartz–Zippel). So
the pass is **sound** (never merges what it can't verify) and practically strong,
but complete only w.r.t. its rewrite theory and probe budget — not a universal
intent-minimizer. That boundary is the design, not a limitation to paper over.

[EVIDENCE: tests/genetic/test_normalize.py(13) green; explorer integration in test_normalize.py::TestExplorerIntegration]

### H42 — Prepopulate the catalog with 22 web-verified 2024–2026 research mechanics (2026-07-07)

**Status:** 🟢 **CONFIRMED** — catalog grows 74 → **96** mechanics; every entry's
arXiv id was web-verified before writing (not model-recalled). Since
`catalog_names()` feeds the novelty gate, the discovery loop now treats all of
these as prior art automatically.

New `mechanics/*.neuro` specs (name — arXiv):
- **attention:** native_sparse_attention (2502.11089), moba (2502.13189),
  selective_attention (2410.02703), forgetting_attention (2503.02130),
  softpick (2504.20966)
- **position:** yarn (2309.00071), longrope (2402.13753)
- **sequence_mixer:** deltanet (2406.06484), gated_deltanet (2412.06464),
  titans (2501.00663), xlstm (2405.04517), rwkv7 (2503.14456)
- **normalization:** ngpt (2410.01131)
- **routing:** loss_free_balancing (2408.15664), fine_grained_experts (2401.06066)
- **optimizer:** soap (2409.11321), schedule_free (2405.15682),
  adam_mini (2406.16793), galore (2403.03507)
- **training_dynamics:** qk_clip / MuonClip (2507.20534), grokfast (2405.20233),
  mup (2203.03466)

Each spec carries `summary`, `equation` (mechanism), `when_to_use`, `not_for`,
`empirical_evidence{source,result,caveat}`, `maturity`, `novelty_vs_baseline`,
and a `references` citation — the same schema the existing 74 use. `impl` is left
empty: these are **prior-art catalog entries**, not wired Python mechanisms
(honest per §14 — they document what exists so the search doesn't rediscover it,
they are not stubbed implementations claiming behaviour they lack). Provenance:
every arXiv id confirmed by web search this session (WebSearch tool), not recalled.

[EVIDENCE: tests/genetic/test_catalog.py::TestPrepopulated2024_2026 (3 contracts) green; mechanics/ +22 files parse via mechanic_parser]

### H43 — softpick as an evolvable NGL primitive + deeper Colab discover (2026-07-07)

**Status:** 🟢 **CONFIRMED** — 14 new contracts. One of the prepopulated mechanics
(softpick) is now a real, evolvable NGL op — not a black box the search can only
route around, but an atom it can mutate into.

- **`softpick_last` NGL op** (`language.py`). Exact formula, verified against the
  paper (arXiv:2504.20966) via WebFetch of the HTML source:
  `softpick(x)_i = ReLU(e^{x_i} − 1) / (Σ_j |e^{x_j} − 1| + eps)` over the last
  axis, implemented with the paper's max-subtraction (algebraically identical: the
  e^{−m} factor cancels between numerator and denominator). Real math, CPU,
  gradient-flowing (§14 — not a stub): tests pin non-negativity, **true zeros** for
  x≤0 (the no-attention-sink property), sum→1 only when all logits are positive
  (else <1, i.e. not sum-to-one), finiteness on extreme logits, and gradient flow
  through negative entries via the abs() denominator.
- **Evolvable, not opaque.** Because it is a registered `nonlin` op it enters the
  GA's mutation vocabulary (`evolve._OP_NAMES`) automatically. `attention_
  primitives.single_head_attention_softpick` is the one-op mutation of the softmax
  attention program (`softmax_last`→`softpick_last`) — the exact edit the search can
  now make on its own to discover sink-free attention. `semantics.analyze` labels it
  a bounded, normalizing, element-mixing **attention** mechanism (the role detector
  now recognizes a softpick-over-matmul score as attention).
- **Deeper Colab discover cell.** The T4 discover cell now defaults to the
  `explore` mode (training loop + persistent ledger), with the prior-art gate ON
  (seeds the ledger so only novel mechanics are searched), **semantic normalization
  ON** (syntactic variants collapse — validated: a deep run skipped 13 duds at step
  1000), novelty pressure, `--macros`, and artifact push. Validated end-to-end on
  CPU: `discover explore` kept improvements and persisted 35 ledger patterns.

Honest scope note: of the three candidates offered (softpick / forgetting_attention
/ selective_attention), only softpick is cleanly expressible in NGL's straight-line
DAG. Forgetting-Transformer and Selective-Attention both need a **sequence-axis
cumulative scan** (cumulative log-forget-gate / accumulated selection) that NGL has
no primitive for; adding a `cumsum`-style scan op is the prerequisite and is logged
as future work rather than faked.

[EVIDENCE: tests/genetic/test_softpick.py (14 contracts) green; deep `discover explore` run persisted a 35-pattern ledger with dud-skipping active]

### H44 — `discover explore` made sound: live progress + non-diverging baseline (2026-07-07)

**Status:** 🟢 **CONFIRMED** — 3 new contracts. Two operational defects and one
*scientific* defect fixed, so the deep-explore run produces interpretable numbers.

- **Live progress** (operational). The run only *looked* frozen — `explore` logged
  nothing until each 500-step summary. Threaded a progress callback through
  `run_training_with_exploration → maybe_explore → explore → auto_evolve.on_
  generation`: explore-step header, throttled per-generation `best_ppl` with
  evaluated/skipped-dud counts, and a training heartbeat between searches.
- **Artifact push on fresh runtimes** (operational). `commit failed` was a fresh
  Colab clone with no `git config user.name/email`. `modulation_pusher` now supplies
  a fallback bot identity for the commit (respecting a configured one), and the CLI
  surfaces the failure detail instead of hiding it.
- **Non-diverging baseline** (scientific — the important one). The A/B gate installs
  whatever eval'd best on the current model, but a modulation that eval's well can
  wreck training dynamics (the residual gain `h·g(h)` clamps to ±8; an 8× gain
  compounds over steps). Left unchecked the reported baseline ran away (127k → 24M
  over 4k steps) so every "KEPT" was a win over a **collapsing reference** —
  meaningless. Fix: (1) a fresh modulator `Memory` per forward (the shared one leaked
  state across every training step — an unbounded-state divergence source); (2) a
  **checkpoint-and-restore guard** — track the unmodulated ("identity") val ppl as
  the model's health, snapshot weights at each new best, and when an install pushes
  ppl past 3× the best seen, **roll the weights back** to the last healthy checkpoint,
  drop the modulation, and reset the optimizer moments. Swapping the modulation alone
  wasn't enough — the damage is baked into the weights. Validated: baseline now
  bounded (127k → 88k, stable), the guard visibly restores on divergence, and the
  discovered modulation consistently beats the no-mod baseline (34k < 88k). The toy's
  absolute ppl is still just a tiny-Markov toy — but the *comparison is now sound*.

[EVIDENCE: tests/genetic/test_training_explorer.py::TestStability::test_baseline_does_not_diverge, ::TestProgress; test_modulation_pusher.py::TestFreshRuntime]

### H45 — `discover explore` produces auditable artifacts: persisted winners + reproducible search (2026-07-07)

**Status:** 🟢 **CONFIRMED** — 2 new contracts. The explorer now leaves durable,
inspectable measurements instead of throwaway telemetry.

- **Winners persist** (`run_training_with_exploration(store=…)`). Each *installed*
  winner is saved to `modulations/<run>-step<N>.neuro` with its measured Δ
  (`baseline_ppl`, `best_ppl`, `delta_ppl`, `step`). The Δ is honest — `res.baseline`
  is measured *after* the divergence guard runs, i.e. on the restored **healthy**
  model, not the drifting one. CLI `discover explore` now writes to `modulations/`
  and reports the durable count.
- **Drop-on-revert.** An install the guard later reverts (it destabilized training)
  is proven bad, so its persisted file is removed — the store ends with only the
  survivors. Contract: persisted ≤ installs.
- **Reproducible search.** `explore()` seeded its RNG with builtin `hash()`, which is
  salted per process (`PYTHONHASHSEED`) → runs weren't reproducible, which is fatal
  for a *measurement*. Replaced with a blake2b digest of `(run_id, step)` —
  deterministic across processes. (Also loosened the stability test to the guard's
  real guarantee: bounded, not within 5× — a single bad window can spike before the
  next health check, but never runs away to millions.)

**First honest measurements** (tiny-Markov toy — weak model, but sound comparison):
- seed 0 (1000 steps): **1 durable winner** — `t0 = l2norm_last(t0); t0 = div(t0, t4)`,
  baseline 87 855 → 8 697, Δ≈79 158, survived to the end. The search **rediscovered
  residual-stream normalization** (L2-norm) as a helpful modulation — a real,
  known-good mechanism, not op-salad.
- seed 3 (3000 steps): **0 durable winners** — all 4 installs destabilized training
  and were dropped (4 reverts). An honest null result.

The toy's absolute ppl (~9k–120k) is just a tiny-Markov toy; what's now sound is
that a persisted modulation carries a real Δ against a healthy baseline and has
survived continued training. Note the semantic-analysis role label is imprecise
here (an in-place `h→f(h)` modulation that rewrites `t0` is tagged
`optimizer_update` because "rewrites its input" trips the stateful heuristic) — a
labeling nuance, not a measurement error.

[EVIDENCE: tests/genetic/test_training_explorer.py::TestPersistence (2 contracts) green; determinism via _stable_seed]

**Addendum — persist the *minimal* mechanism.** The GA emits bloated winners (a
real Colab discovery was 12 instructions, 8 of them dead code). Winners are now run
through `_minimal_equivalent` (DCE + probe-verified peephole superopt, falling back
to raw if not probe-equivalent) *before* persisting and installing, so the saved
`modulations/*.neuro` **is** the mechanism. The pushed `run_0_step4000` (Δ
4425.84→784.43, durable) reduces 12→3 ops: `g = |h|`, i.e. the modulation
`h → h·|h|` (a sign-preserving squared gain / contrast-sharpening nonlinearity).
Honest limit: the semantic-role label is *still* `optimizer_update` even minimized,
because the discovered program reads an uninitialized register (`t3` read before
write) which trips the stateful heuristic — a **search-quality** artifact (the GA
emits programs that read undefined registers), not a persistence one. Simplifying
the artifact doesn't fix a mislabel rooted in the program's own structure.

[EVIDENCE: tests/genetic/test_training_explorer.py::TestPersistence::test_persisted_programs_are_minimal green; run_0_step4000 minimizes 12→3 ops]

**Addendum 2 — penalize undefined-register reads (search quality).** The root of the
op-salad / stateful-mislabel is that NGL reads of an unwritten register return
zeros, so the GA freely emits programs reading undefined registers.
`evolve.undefined_reads(program, inputs)` counts those; `TrainingExplorer._fitness_
penalty` applies a multiplicative bump (`1 + wellformed_penalty·count`, default 0.05)
to the search **objective only** — the reported ppl stays true — so the GA prefers
clean mechanics. CLI: `discover explore --wellformed-penalty`. Honest empirical note:
across a 4-seed A/B (penalty 0 vs 0.05) the effect on the *persisted* winners'
cleanliness was marginal (avg undefined reads ~0.5 either way) — because DCE-on-
persist already strips dead undefined reads, so most survivors were already clean,
and their roles now label correctly (`generic` / `normalization`, not the spurious
`optimizer_update`). The penalty's real value is steering eval budget away from
ill-formed candidates during the search, not the final artifact; it is unit-verified
and tunable, not claimed to dominate on this toy.

[EVIDENCE: tests/genetic/test_wellformed.py (9 contracts) green; A/B persisted-winner roles = generic/normalization]

### H46 — Exploration wired into the *real* trunk: read-only discovery probe (2026-07-08)

**Status:** 🟢 **CONFIRMED** — 6 new contracts. A `--explore_every N` flag makes a
real training run gather first discovery data on the actual trunk, safely.

- **Why not an installed hook.** `BRIANHarness` has no simple residual-`modulate=`
  seam like the toy `_TinyLM`; its modulations are architectural (NT/orchestrator
  fusion in the 1.1B stack). Adding a hook into the training forward is a genuinely
  risky change to install mid-flight on a live run, so H46 ships the **safe half**
  first: a *read-only probe* that measures whether a residual modulation of the
  trunk's final hidden state would lower next-token CE — without ever touching the
  training forward or weights.
- **`probe_hidden_modulation(hidden, head_fn, targets, …)`** (`training_explorer.py`).
  Baseline = CE of the LM head on the (detached) final hidden; then the GA searches a
  modulation, re-projecting the head on `modulate(hidden)` — all under `torch.no_grad`
  (never builds a graph, so it can't perturb the run and is cheap). Keep-if-better,
  persist the minimal winner with its Δ. Floored at identity (best ≤ baseline).
- **`train_dsl.py`:** `_run_trunk_probe` runs a fresh no-grad forward to fetch
  `language_model._last_hidden`, probes it, logs `baseline_ce / best_ce / Δ`, and
  pushes `modulations/` + the ledger. Wired at the `--explore_every` cadence,
  wrapped in try/except (a probe failure can never crash training). Colab training
  cell defaults `--explore_every 500`.
- **Explorer core hardened (bugs found + fixed via TDD).** (1) The winner is now the
  best *raw* candidate by the *clean* metric, floored at identity — the wellformed
  penalty had made the GA's penalized-objective genome disagree with the true best,
  and scoring the *canonical* form was unsound (canonicalization is non-deterministic
  and can produce a program that throws as a live modulator). (2) A masked
  `NameError` — `_make_modulator` wasn't imported at module scope — made the probe's
  try/except silently score *every* candidate as `baseline×10`; the missing import is
  the whole reason an early probe looked broken. Ledger dedup now keys on the
  canonical form consistently (record + is_dud agree).

Honest scope: this is a *probe*, not an installed mechanism. It measures whether the
trunk's final representation has structure a fixed modulation can exploit (a real,
if narrow, signal) and banks candidate winners — it does **not** train the modulation
into the model. Installing a searched modulation into the forward (the higher-risk,
higher-signal step) remains a deliberate opt-in for later, once the probe shows
something worth installing.

[EVIDENCE: tests/genetic/test_trunk_probe.py (4), tests/training/test_trunk_probe_wiring.py (2) green]

### H47 — `mutate()` crashed on `call` (macro) instructions: `KeyError: 'call'` (2026-07-08)

**Status:** 🟢 **FIXED** — 3 new contracts, found by a real Colab T4 run.

- **Symptom.** `brian discover optimizer --pop 64 --generations 40 --steps 80
  --seed 0 --task parity --novelty 0.3 --avoid-known --macros` crashed at
  generation 2/40 with `KeyError: 'call'` inside `evolve.py::mutate()`.
- **Root cause.** `call` instructions (macro invocations, grafted via
  `--macros` → `insert_call`) are deliberately absent from `REGISTRY` —
  `Instruction.__post_init__` special-cases `op == "call"` and skips the
  registry-membership check, since a macro's arity/behaviour lives in the
  `MacroLibrary`, not `REGISTRY[op].fn`. `mutate()`'s `point_reg` and
  `point_const` branches didn't know this: both did an unconditional
  `spec = REGISTRY[old.op]`, which is fine for every ordinary op but throws
  the moment a population member (legitimately, once `--macros` is on)
  contains a `call` instruction and the GA rolls `point_reg`/`point_const`
  on it.
- **Fix.** `point_reg` now reads arity as `len(old.ins)` when `old.op ==
  "call"` (mirroring the macro's own input count) instead of
  `REGISTRY[old.op].n_in`, and reconstructs the `Instruction` carrying
  `old.config`/`old.macro` through (previously dropped even for non-call
  ops — a latent bug that would have silently discarded `config`/`macro`
  on any `point_reg` mutation). `point_const` is a no-op for `call`
  instructions (they carry no const — it's resolved via the library, not
  `REGISTRY[op].uses_const`).
- **Regression sweep.** `tests/genetic/test_evolve.py` (10, incl. 3 new
  `TestMutationOfCallInstructions` cases reproducing the exact crash before
  the fix) and `tests/genetic/test_macros.py` (8) green. Full
  `tests/genetic/` sweep: 270 passed, 1 pre-existing unrelated failure
  (`test_baselines.py::TestSeededDiscovery::test_discovery_can_start_from_adam`,
  confirmed failing identically on the pre-fix commit — a seeded-Adam
  quality gap, not a `mutate()` regression).

[EVIDENCE: tests/genetic/test_evolve.py::TestMutationOfCallInstructions (3) green; reproduces + fixes live Colab crash]

### H48 — `auto_evolve`'s tracked "best" regressed under novelty pressure (2026-07-08)

**Status:** 🟢 **FIXED** — 1 new contract, found from the H47-fixed live Colab
run's own progress log.

- **Symptom.** The freshly-unblocked `discover optimizer --novelty 0.3 --macros`
  run's per-generation `best_loss` was not monotonic:
  `gen6=0.4014 → gen7=0.4014 → … → gen10=0.7035` — the reported "best so far"
  got *worse*, which should be structurally impossible in an elitist GA.
- **Root cause.** `run_optimizer_discovery` passes `weights=[1.0, 1.0, 0.5]`
  when `novelty_weight > 0`, and `auto_evolve` tracked `best_prog`/`best_obj`
  (and `history`, and the value hand to `on_generation` for the printed
  `best_loss`) using that *novelty-inclusive* scalar. Novelty is a distance-
  to-the-current-population bonus — recomputed fresh every generation, so a
  program's novelty score is only meaningful *within* the generation it was
  scored in. Comparing an old `best_obj` (novelty computed against an earlier,
  more-diverse population) against a new generation's objectives (novelty
  computed against a now-more-converged population) is comparing two different
  quantities that happen to share a name. A new individual with a strictly
  worse loss but a bigger novelty bonus (large relative to a converging
  population) could out-score the old champion and overwrite it — so the
  *displayed* "best" (and the actual `result.best_program` returned to the
  caller!) could regress on the metric that actually matters (loss/cost).
- **Fix.** `auto_evolve.scored()` now returns `(raw, sel)` — `raw` is exactly
  `evaluate(p)` per program (comparable across generations), `sel` is `raw`
  plus the novelty bonus (population-relative, selection-only). Elitism carry-
  forward and tournament selection still use `sel` (so novelty still drives
  exploration pressure) — but `best_prog`/`best_obj`/`history`/the
  `on_generation` callback now track strictly by `raw` scored with
  `weights[:len(raw_dims)]`. The tracked best is now guaranteed non-decreasing
  regardless of novelty pressure, and `EvolveResult.best_objective` always has
  the caller's real objective arity (2-tuple for `run_optimizer_discovery`),
  never a novelty-inflated 3-tuple.
- **Why this matters beyond cosmetics.** `run_optimizer_discovery` returns
  `best_final_loss = -result.best_objective.values[0]` as *the* headline
  number for a discovery run. Before this fix, that number could be a worse
  loss than one actually visited earlier in the same search — a silent
  regression in the one metric the whole harness exists to report faithfully.

[EVIDENCE: tests/genetic/test_evolve.py::TestAutoEvolve::test_history_is_monotonic_even_with_novelty_pressure green]

### H49 — `best_loss` progress line still looked non-monotonic: it's a (loss, cost) trade-off (2026-07-08)

**Status:** 🟢 **CLARIFIED + FIXED THE DISPLAY** — 1 new contract.

- **Symptom.** Even after H48, a live Colab T4 re-run of `discover optimizer
  --novelty 0.3 --macros` still showed `best_loss` going the wrong way:
  `gen0=0.6415 → gen1=0.7035`. Reproduced locally with the H48 fix already
  applied — so this is *not* a regression of H48, it's a second, distinct
  source of the same visual symptom.
- **Root cause.** `run_optimizer_discovery`'s own documented objective is
  **`(-final_loss, -cost_weight*n_instructions)`** — two dimensions, both
  real, both tracked by `auto_evolve` post-H48 (correctly, via the raw
  scalar). A program with meaningfully fewer instructions can out-score a
  lower-loss one on the *combined* scalar and legitimately become the new
  champion — the search is doing exactly what it's told (trade quality for
  efficiency). The bug was never in the tracking; it was that the progress
  line printed only `best_loss=` (one of the two tracked dimensions), so a
  genuine efficiency-for-quality trade came across as an impossible
  regression in an elitist GA.
- **Fix.** The `optimizer` progress line now prints `cost=` alongside
  `best_loss=`, reconstructed from `-o.values[1] / cost_weight` (both are the
  *raw*, H48-tracked dimensions — no novelty ever leaks into this line
  post-H48). Reading the two together, a "regression" in loss is now legible
  as "the champion got cheaper."
- **Honest scope.** This is a display fix, not a search change — `--novelty`
  still trades pure loss-greediness for efficiency + diversity pressure by
  design; that trade-off is real and was always the intended behaviour of
  the multi-objective Pareto search (see `front_stats`).

[EVIDENCE: tests/genetic/test_discovery.py::TestProgressReporting::test_progress_line_shows_cost_alongside_loss green]

### H50 — `EvolveResult` gains a `primary_*` champion: the headline "best" is now truly monotonic (2026-07-08)

**Status:** 🟢 **FIXED** — 1 new contract at the `auto_evolve` level, plus every
downstream caller repointed.

- **Symptom.** A live Colab T4 re-run of `discover optimizer --novelty 0.3
  --macros` *still* showed `best_loss` going backwards after both H48 and H49:
  `gen0=0.5602 cost=9 → gen1=0.6430 cost=2`. H49's `cost=` annotation made the
  trade-off legible but didn't fix the actual complaint — the user's (correct)
  expectation is that "best score" should never get worse, full stop.
- **Root cause, precisely.** `run_optimizer_discovery`'s declared objective is
  genuinely 2-D: `(-final_loss, -cost_weight*n_instructions)`. H48 fixed
  `auto_evolve` to track its champion by this *raw combined scalar* — correctly
  monotonic *on the scalar* — but the scalar itself blends loss and cost, so a
  much-cheaper, slightly-worse-loss program can legitimately raise the combined
  scalar while *lowering* the loss component alone. That's not a bug in the
  search; it's the declared multi-objective design working as intended. The bug
  was reporting only the loss component of that blended champion as if it were
  a standalone, independently-tracked metric.
- **Fix.** `EvolveResult` gains `primary_program` / `primary_objective`: a
  *second*, independent champion tracked purely by `values[0]` (the caller's
  named primary metric — loss, ppl, …), with all other dimensions (cost,
  novelty, plausibility, EI) completely ignored for this tracker. It's a plain
  running-argmax over one number, so it is monotonic by construction — no
  scalar blending, no population-relative bonus, nothing to trade off against.
  `on_generation` now receives `(gen, total, best_obj, primary_obj)`; every
  progress line and every `*Outcome.best_program`/`best_*_loss` (`discovery.py`
  optimizer + flow-modulation, `neuro_evolve.py` trunk) now reports
  `primary_*`, not the combined-scalar champion. The combined champion
  (`best_program`/`best_objective`) still exists and still drives
  elitism/tournament/the Pareto `front` — multi-objective exploration is
  unchanged, only *what gets reported as "the best"* changed.
- **Verified live** (not just unit tests): a 10-generation CPU run with
  `--novelty 0.3 --macros` on the parity task now prints
  `gen0..5 best_loss=0.6931 → gen6..10 best_loss=0.6507` — strictly
  non-increasing loss end to end, with `cost` jumping 6→54 at the same
  generation (the new champion is far more expensive — a genuine trade
  the search made, now truthfully reported instead of looking like noise).

[EVIDENCE: tests/genetic/test_evolve.py::TestAutoEvolve::test_primary_metric_is_monotonic_even_when_cost_trades_off green; tests/genetic/{test_progress,test_known,test_evolve}.py (23) + {test_discovery,test_cli_discover,test_device,test_training_explorer,test_neuro_evolve,test_cli_trunk,test_trunk_probe}.py (39) all green, 62 total]

### H51 — progress line names the algorithm, not just a number (2026-07-08)

**Status:** 🟢 **FIXED** — 2 new contracts.

- **Complaint (verbatim).** "Still doesn't output much useful progress info,
  and I have no idea which algorithms it actually explores atm" — on a live
  Colab T4 run where H50 had already made `best_loss` correctly monotonic
  (`0.5602 → 0.0311`, held for 13+ generations). The numbers were now honest,
  but a bare `best_loss=0.0311 cost=54` still doesn't say *what* was found.
- **Fix.** `auto_evolve`'s `on_generation` callback gains a 5th argument,
  `primary_prog` — the actual `Program` behind `primary_objective`, not just
  its score. `discovery.py::_make_progress` gains a `describe` hook: whenever
  the champion program's rendered description changes, it prints one extra
  line — `    champion: <neuroslm.genetic.semantics.describe(prog)>` — reusing the
  abstract-interpretation machinery from H33/H40 (role, boundedness,
  normalization, statefulness, op families) that already existed for exactly
  this purpose but had never been wired into a progress stream. Wired into
  optimizer discovery, flow-modulation discovery, and trunk evolution
  (`neuro_evolve.py`) — all three `_make_progress` call sites.
- **Verified live**, gen 0 of a `--novelty 0.3 --macros` run now prints:
  `champion: Role: gating. Use to modulate a signal by a learned gate. Traits:
  bounded output, elementwise, sign-based. Families: arith, nonlin; 6
  instruction(s). Notes: output is magnitude-bounded …; sign-based update
  (scale-free, Lion-like); pointwise — no cross-element mixing.` — directly
  answers "which algorithm", not just "how good".
- **Fails safe.** `_describe_champion` catches any exception from
  `semantics.describe` (e.g. an unexpanded `call` op semantics can't yet
  reason about) and renders `<undescribable: ...>` rather than crashing a
  live discovery/training run over a cosmetic feature.

[EVIDENCE: tests/genetic/test_progress.py::TestAutoEvolveCallback::test_on_generation_called_each_generation_plus_gen0, TestDiscoveryProgress::test_optimizer_progress_describes_the_champion_algorithm green]

### H52 — Multi-site probe: discovery on *optimizable regions* of the real trunk (2026-07-09)

**Status:** 🟢 **CONFIRMED** — 13 new contracts + a live smoke run showing Δ>0
at an intermediate layer where the terminal site (H46) gave Δ=0.

- **Hypothesis.** H46's Δ=0 was a *site* problem, not a search problem: the
  terminal hidden is the single most end-to-end-optimized point in the network.
  Intermediate block outputs are shaped only by indirect pressure, so real
  slack should survive there — and a probe that measures per-layer headroom
  under the TRUE loss should find nonzero Δ exactly at the layers a headroom
  scan flags, while skipping converged ones.
- **`DSLLanguageCortex.forward_from_layer(k, hidden)`** (`nn_lang.py`) — re-run
  the real tail (remaining blocks + adapters + NT gain + PCT + NFO + final norm
  + LM head) from a possibly-modulated block-k output. Bit-exact against
  `forward()` in eval mode for every k (incl. `pct_trunk>0` via the stashed
  lower-layer outputs, and cosine head); strictly read-only (weights, buffers,
  stashes pinned by test). The NT-gain block was extracted into
  `_compute_block_gain` so both paths share one code path (no drift).
- **`neuroslm/genetic/layer_probe.py`** — the three-stage probe:
  `headroom_scan` (deterministic perturbation battery per layer → sensitivity
  = loss leverage, improvement = measured slack, i.e. a trivial perturbation
  already beating the trained forward), `select_sites` (slack first, never
  insensitive sites, speculative fallback to the most-promising sensitive
  site), `probe_optimizable_regions` (NGL modulation search at the chosen
  sites, every candidate scored by TRUE next-token CE through the real tail;
  eval-mode + no_grad + mode-restore; winners persist with Δ and site in the
  name: `<run>_L<k>_step<n>`).
- **`train_dsl._run_trunk_probe`** now takes the multi-site path whenever the
  trunk exposes `forward_from_layer` (i.e. always, for real `--model dsl_lm`
  training); the legacy terminal-hidden re-projection remains as fallback for
  models without the layer stash. New `--explore_sites` flag (default 2).
  Colab cells 5+6: `EXPLORE_EVERY=500` default ON for GPU training — every 500
  steps the log shows the per-layer slack table
  (`L0: sens=… improve=+… ←slack | L1: … tight | …`) followed by the search
  result, answering "which parts of my model are still optimizable" inline.
- **Smoke evidence (real cortex, 60 real train steps, uneven optimization):**
  headroom table flagged L1 `tight` (skipped) and L0/L2–L5 `←slack`; the probe
  searched L4+L0 and found a genuine winner at L4 — `Δ_ce=0.0029` **through
  the real tail** — persisted as `smoke_L4_step500`. Same machinery, right
  site, nonzero result where H46's terminal probe measured exactly zero.
- **Honest scope.** The probe *finds and banks* mechanisms that measurably
  lower the true loss of the live model at a specific site and step; it does
  not install them into the training forward (that remains the deliberate
  opt-in follow-up, now with site-tagged candidates worth installing). A Δ
  measured at step N on one batch is a candidate, not a confirmed mechanism —
  confirmation = recurrence across probes + an install A/B.

[EVIDENCE: tests/dsl/test_forward_from_layer.py (5), tests/genetic/test_layer_headroom.py (5), tests/training/test_trunk_probe_wiring.py (3, incl. multi-site path) green; forward-parity suites (test_loss_parity_n8, test_pct_trunk, test_dsl_language_cortex_equivalence, test_cosine_head, test_novel_topology — 47) green]

### H53 — Close the loop: explore-only discovery + evidence-gated install into training (2026-07-09)

**Status:** 🟢 **CONFIRMED** — 20 new contracts + an end-to-end 3-stage live run.

- **Motivation (user workflow).** On a slow T4, training and discovery compete
  for the same GPU-hours. H52's probe only fired *inside* training, and its
  banked winners never took effect. Two additions close the loop:
  bank winners cheaply WITHOUT training, then have the NEXT training run
  install the ones that survive an evidence gate.
- **`--explore_only --explore_rounds N`** (`train_dsl.run_probe_only`) —
  discovery without training: probe the current model state (combine with
  `--resume` to probe the latest checkpoint) over N *fresh batches*; no
  optimizer, no backward, weights pinned untouched by test. Winners bank to
  `modulations/` site-tagged; a per-site search tally prints at the end.
  Colab: `EXPLORE_ONLY=True` in cell 4's config runs cell 5 in this mode.
- **`DSLLanguageCortex._layer_modulations`** (`nn_lang.py`) — install point:
  block index → callable applied at exactly the probe site (block output,
  post adapter/gain — what `_last_layer_outputs[k]` stashes). Empty dict is
  bit-identical to baseline (parity suites green); `forward_from_layer` stays
  bit-exact with installs active (tail re-applies deeper sites only);
  gradients flow through installed modulations (`h·g(sg[h])`, the same
  `_make_modulator` semantics the probe scored); a throwing modulation is
  bypassed AND auto-uninstalled — a banked winner can never crash a run.
- **`--use_modulations`** (`genetic/modulation_install.py`) — at training
  start, load `modulations/`, group winners by (site, program semantics)
  (recurrence pools Δ evidence), then a **count-aware live gate** on a fresh
  batch of the CURRENT model: recurring winners (≥2 probes) must not get
  worse; single-shot winners must STRICTLY improve — probe batch + install
  batch on the same checkpoint weights = 2-fold cross-batch validation. Every
  install/reject decision prints with live before→after CE.
- **End-to-end evidence (real cortex, 60 pretrain steps, then the exact user
  workflow):** Stage 1 — 6 explore-only rounds banked 12 winners (weights
  untouched). Stage 2 — the gate installed exactly ONE: `trunk_p_L3_step2`,
  whose improvement generalized to a fresh batch (ce 5.5978→5.5491,
  Δ=0.049); the other four were rejected as batch-specific (fresh-batch
  ce unchanged). Stage 3 — training proceeded with the install active,
  gradients flowing. The gate discriminates real mechanisms from noise —
  which is the entire point.
- **Expert-cortex probing (queued next).** The frozen pretrained expert LMs
  (PPL≈50) are the *stronger* discovery target: they were optimized for their
  pretraining distribution, not BRIAN's mixture, so domain-shift slack is
  well-defined and durable (frozen weights ⇒ winners never go stale). v1 =
  expert final hidden → own head via the expert-side tokenization path in
  `experts.py`; H46's null-site argument does not apply to frozen models on
  shifted data — the headroom scan will measure it.

[EVIDENCE: tests/dsl/test_layer_modulations.py (5), tests/genetic/test_modulation_install.py (11), tests/training/test_trunk_probe_wiring.py::test_probe_only_loop_runs_without_training green; tests/dsl full sweep 1254 passed (2 pre-existing mechanics-index failures unrelated)]

### H54 — Expert-cortex probe: `brian discover experts` on the frozen pretrained LMs (2026-07-09)

**Status:** 🟢 **CONFIRMED (machinery)** — 10 new contracts + live runs on a real
pretrained model with real streamed data. First roster-scale measurement is the
user's Colab run.

- **Why experts (task #40, follow-up to H52/H53).** The frozen expert cortices
  (SmolLM2-360M, CodeGPT-small-py, Qwen2.5-0.5B — PPL≈50 territory) were
  optimized for THEIR pretraining distributions, not BRIAN's data mixture, so
  domain-shift slack is well-defined; frozen weights make every banked winner
  durable (no per-checkpoint staleness); and a modulation that lowers an
  expert's CE sharpens the KL-distillation teacher signal the whole arch
  consumes. Scoring is the expert's OWN next-token CE in its OWN token space —
  the vocab bridge is a distillation concern, not a discovery one.
- **`neuroslm/genetic/expert_probe.py`** — `ProbedExpert.load` (reuses
  `experts.py`'s `_load_lm_cached`/`_split_lm`/alias resolution; no VocabBridge
  built — probing needs no trunk), `expert_batch` (expert-tokenizer windows,
  capped at the model's hard ctx), `probe_expert` (headroom line + NGL search
  on the final hidden through the expert's own head), `run_expert_discovery`
  (multi-round × multi-expert, winners bank site-tagged `expert_<alias>_step<r>`),
  `make_texts_provider` (stateful stream). CLI: `brian discover experts
  --models … --rounds …`; Colab cell 13 `MODE="experts"` is the new default.
- **Two real bugs caught by the live smoke, fixed via TDD:**
  1. *bf16 quantization*: experts load in bf16; CE computed through the bf16
     head was quantized to 1/32-nat steps (live: `baseline_ce=4.1875,
     improve=+0.03125` — all n/32), coarser than the Δs the probe hunts.
     Fixed: fp32 hidden + fp32 functional head view (`_fp32_head`, no
     mutation of the process-wide bf16 cache). Post-fix the same probe reads
     `baseline_ce=4.1698, improve=+0.001912` — smooth, honest.
  2. *repeated texts*: re-opening the HF stream every round returned the same
     first-N texts (two rounds with bit-identical baselines), voiding the
     fresh-batch recurrence evidence. Fixed: one stateful provider iterator
     across rounds — live baselines now advance (4.17 → 3.98 → 3.63).
- **First honest measurements (distilgpt2 on FineWeb-Edu, CPU smoke):**
  headroom at the final hidden is real but small (+0.0019 to +0.0022 nats in
  2 of 3 rounds; third measured tight) — expected, since FineWeb-Edu is close
  to that model's pretraining data. The GA (≤57 evals) didn't bank a winner:
  honest nulls at smoke budget. The roster experts on BRIAN's chat/mix data
  are where the domain shift — and the expected slack — is larger.
- **Known follow-up:** seed the probe GA with const-gain programs at the
  battery's best perturbation (the battery twice found trivial slack the
  57-eval GA missed — starting FROM the measured slack instead of identity
  would let the search refine rather than rediscover it).

[EVIDENCE: tests/genetic/test_expert_probe.py (10) green; live `brian discover experts --models distilgpt2` runs pre/post-fix logs above]

### H55 — `brian deploy-discover`: run discovery (not just training) on vast.ai (2026-07-10)

**Status:** 🟢 **CONFIRMED** — 20 new contracts + a bash-syntax-verified live
onstart script; human-confirmation gate verified live to block a non-
interactive invocation.

- **Motivation.** `brian deploy` only ever launches training. Discovery runs
  (H52-H54: multi-site trunk probe, expert-cortex probe) previously had no
  path off a live Colab session — a long `experts`/`trunk`/`explore` search
  either ran on the free (but session-tied, and on a free-tier T4 sometimes
  slow) local GPU, or not at all. The ask: rent a vast.ai instance for a
  discover run, unattended, with its log + modulations + search ledger
  reaching git WHILE it runs (not just at the end, since an interrupted
  instance would otherwise lose everything since the last internal push).
- **`neuroslm/connectors/vast_discover.py`** (new, sibling to `vast.py` —
  not a modification of it, same reasoning `vast_train.sh`'s own docstring
  gives for not touching `vast_deploy.sh`: a discover job has a genuinely
  different shape than arch/scale/steps training). `DiscoverDeployConfig`
  restricts to `DEPLOYABLE_MODES = ("experts", "trunk", "explore")` — the
  only modes that can run long and produce artifacts worth streaming back;
  `optimizer`/`flow`/`qd`/`simplify` already finish in seconds-to-minutes on
  the free local Colab GPU, so renting a paid instance for them would spend
  money for no benefit. `build_discover_onstart()` builds the container-side
  script (clone, `vast_bootstrap.sh` reused verbatim for pip deps, then a
  **mode-agnostic background pusher** — independent of whatever a discover
  mode pushes internally — that calls the existing `push_artifacts()` every
  `push_interval` seconds, scoped to the run's log + `modulations/` + the
  search ledger + `heatmaps/`), runs `brian discover <mode> ... --push`,
  final push, then self-destroys (mirrors the training onstart's self-destroy
  block, keyed to a distinct `neuroslm-discover` label so it never touches a
  concurrent training instance).
- **`scripts/vast_discover.sh`** (new, sibling to `vast_train.sh`) — offer
  search + instance create, no ARCH/hardware-block lookup (discover has no
  arch scale). Bash-syntax-verified (`bash -n`) both for the launcher and for
  a fully Python-substituted onstart script. **Default GPU tier revised to
  A100** (`--gpu-query`/`GPU_QUERY` overridable) after the first live run
  (instance 44315230, `phone-discover`) landed on a cheaper RTX_3090 host
  whose image pull alone ate several minutes — a discover job's total wall
  time is dominated by rental-host luck at that tier, not GPU compute, so the
  "cheap tier is fine, discover workloads are light" reasoning this section
  originally gave was true for compute but not for time-to-first-log. The
  Colab cell's `GPU_QUERY` knob defaults to A100 too.
- **`brian deploy-discover <mode> [...]`** (`cli.py::cmd_deploy_discover`) —
  routes through the SAME `_require_human_confirmation` gate `brian deploy`
  uses (no separate, weaker path); verified live that a piped/non-interactive
  invocation is blocked exactly like `brian deploy` is. Mode validated (both
  by argparse `choices` and the connector's own check) before the gate is
  ever reached, so an invalid mode never prompts.
- **Colab**: two new cells (14 markdown, 15 code) mirroring the existing
  phone-deploy pattern — `cli.main(["deploy-discover", ...])` run IN-PROCESS
  so the confirmation prompt renders in the live cell (same reason the
  training deploy cell does this: a subprocess has no interactive channel).

[EVIDENCE: tests/test_vast_discover_deploy.py (20) green; tests/test_connectors.py + test_deploy_confirmation.py + test_deploy_safety_gate.py (91 total) unaffected; live CLI: `brian deploy-discover optimizer` rejected before the gate, `echo "" | brian deploy-discover experts --rounds 5` BLOCKED by the human-confirmation gate exactly as `brian deploy` is]

### H55.1 — `deploy-discover experts` ran on CPU on a rented A100 (real cost bug) (2026-07-09)

**Status:** 🔴 **CONFIRMED BUG, FIXED** — 2 new contracts, caught by the
user's own second live deploy (instance 44317528, A100 SXM4 @ $0.61/hr).

- **Symptom.** Round 1 of `deploy-discover experts` took ~90 minutes on a
  rented A100 — the log showed `[discover:experts] device=cpu`. At that rate
  the configured 30 rounds would have run ~45 hours and ~$27 while the GPU
  sat idle.
- **Root cause.** `cmd_deploy_discover`'s per-mode arg construction added
  `--device auto` to the `trunk` branch (H55) but the `experts` branch never
  got the same line — `discover experts` defaults to `--device cpu` when the
  flag is absent. A straight omission, not a `_resolve_device` bug (that
  function was already correct — H54's `--device auto` works fine when
  actually passed).
- **Fix.** One line: `discover_args += ["--device", "auto"]` added to the
  `experts` branch, mirroring `trunk`.
- **Silver lining.** Round 1 still produced a genuine result even while
  CPU-bound: `microsoft/CodeGPT-small-py` (pretrained on Python, probed on
  FineWeb-Edu general text) showed real domain-shift slack —
  `baseline_ce=6.5992 → best_ce=6.2368`, Δ=0.36 nats — already pushed to
  `origin/master` as `modulations/expert_codegpt_small_py_step1.neuro`
  before the code fix landed, so the finding is preserved regardless of what
  happens to that instance. `smollm2_360m`/`qwen2_5_0_5b` measured `tight`
  (no slack) on round 1 — consistent with H54: general-purpose models on
  general web text have little room, code/domain-specialized ones on
  shifted data do.
- **Status of the live instance.** `44317528` was still running on CPU (~90
  min into round 1) at the time this fix landed — the fix does not apply
  retroactively to an already-running onstart script. Getting the GPU
  speedup on this run requires destroying and redeploying with the fixed
  code (user's call — deploy is human-gated, see H55).

[EVIDENCE: tests/test_vast_discover_deploy.py::TestDiscoverArgsUseTheRentedGpu (2) green, RED-confirmed against the pre-fix commit via `git stash`]

### H56 — Explore-during-training: wire the H52/H53 multi-site probe into `brian deploy` itself (2026-07-09)

**Status:** 🟢 **CONFIRMED** — 9 new contracts (5 in `test_connectors.py`, 4 in
`test_vast_train_dsl_loop_explore.py`); full `test_connectors.py` +
`test_deploy_safety_gate.py` (72 total) green; the one unrelated failure
surfaced by `brian test quick` (`test_baselines.py::TestSeededDiscovery::
test_discovery_can_start_from_adam`, a pre-existing seed-variance flake in
the optimizer-discovery GA) reproduces identically with this diff stashed
out — confirmed pre-existing, not a regression.

- **Motivation.** The user asked: why does discovery only touch the frozen
  pretrained experts (H54) or a synthetic `_TinyLM` proxy (`discover trunk`'s
  actual implementation, not the real SmolLM trunk), when the goal is to
  optimize *our own* trunk's topology and how fast/well it learns from the
  teachers? Two concrete asks followed: (1) probe the REAL trunk from a
  loaded checkpoint (Mode A — `deploy-discover` checkpoint mode, not yet
  built), and (2) run the existing H52/H53 real-trunk multi-site probe
  *during* a live `brian deploy` training run, not just as a separate
  Colab/vast discover job. This entry covers (2).
- **The gap.** `train_dsl.py` already has `--explore_every`/`--explore_pop`/
  `--explore_gens`/`--explore_len`/`--explore_sites`/`--use_modulations`
  (H52/H53) and the Colab **local-GPU** training cell (cell 4/5) already
  wires them through. But `grep`ing `neuroslm/connectors/vast.py`,
  `scripts/vast_train_dsl_loop.sh`, and `neuroslm/connectors/base.py` for
  those flag names returned nothing — the actual `brian deploy` → vast.ai
  path (the one that survives a Colab disconnect) silently never exercised
  the real-trunk probe at all.
- **Fix — same env-var-substitution pipeline OOD_EVERY already uses:**
  - `DeployConfig` (`connectors/base.py`) gains `explore_every: int = 0`
    (off by default, matching `train_dsl.py`'s own default) plus
    `explore_pop`/`explore_gens`/`explore_len`/`explore_sites`/
    `use_modulations`.
  - `cli.py::cmd_deploy` gains `--explore-every`/`--explore-pop`/
    `--explore-gens`/`--explore-len`/`--explore-sites`/`--use-modulations`
    flags, forwarded into `DeployConfig` via `getattr(args, ..., default)`
    (defensive against the existing `_deploy_ns()` test fixture that
    predates these fields).
  - `VastConnector._build_env` (`connectors/vast.py`) only sets the
    `EXPLORE_*`/`USE_MODULATIONS` env vars when `config.explore_every > 0` —
    a bare `brian deploy` stays byte-for-byte unaffected.
  - `_build_onstart`'s substitution dict + `_ONSTART_TEMPLATE`'s `USE_DSL=1`
    branch thread `EXPLORE_EVERY`/`EXPLORE_POP`/`EXPLORE_GENS`/
    `EXPLORE_LEN`/`EXPLORE_SITES`/`USE_MODULATIONS` through to
    `vast_train_dsl_loop.sh`'s environment, same `__PLACEHOLDER__`
    `str.replace()` mechanism every other onstart var uses (avoids the bash
    heredoc pipe-buffer deadlock the template's own comments warn about).
  - `scripts/vast_train_dsl_loop.sh` reads the six `EXPLORE_*`/
    `USE_MODULATIONS` env vars with the same `${VAR:-default}` pattern as
    every other tunable, builds an `EXPLORE_ARGS` bash array
    (`--use_modulations` appended conditionally — it's a boolean flag, never
    unconditional, since that would force-install banked winners on every
    deploy including ones that never asked for it), and forwards
    `"${EXPLORE_ARGS[@]}"` into the `python -u -m neuroslm.train_dsl`
    invocation.
  - Colab cell 10 (phone-deploy) gains the same `EXPLORE_*`/
    `USE_MODULATIONS` knobs as cell 4's local-GPU block, defaulted off,
    forwarded as `--explore-every`/etc. only when `EXPLORE_EVERY` is
    nonzero.
- **Net effect.** A live vast.ai training run launched via `brian deploy
  --explore-every 2000 --explore-sites 3` (or the equivalent Colab
  phone-deploy cell knobs) now periodically runs the real multi-site probe
  against the actual model being trained, banking winners to
  `modulations/`, exactly like a local Colab GPU run already could — no
  behavior change for anyone who doesn't pass the new flags.

### H57 — `brian discover checkpoint`: probe the REAL trunk from a loaded checkpoint, no training (2026-07-09)

**Status:** 🟢 **CONFIRMED** — 17 new contracts in `test_discover_checkpoint.py`
+ 8 new contracts in `test_vast_discover_deploy.py` (checkpoint deployability
+ arg construction) + 4 new contracts pinning `cmd_deploy`'s `--latest`
resolution before/after a refactor into a shared helper; full
`test_vast_discover_deploy.py` (28) + `test_discover_checkpoint.py` (17) +
`test_deploy_safety_gate.py` (19) green.

- **Motivation.** The second half of the user's "Do both" request — Mode A,
  "Load Checkpoints in Discover". The user's underlying question: why does
  discovery only ever touch frozen pretrained experts (H54) or a synthetic
  proxy (`discover trunk`/`discover explore`'s actual implementation is
  `_TinyLM`, confirmed by reading `neuro_evolve.py` — never the real SmolLM
  trunk), instead of measuring headroom on the model actually being trained?
- **Key finding: the real-trunk, checkpoint-loading, probe-only machinery
  already existed** — `train_dsl.py`'s `--explore_only` + `--resume_from`
  (H52/H53) builds the exact-match DSL harness, loads a real checkpoint
  (local path or `hf://...` URI), and runs the multi-site probe with no
  optimizer/backward. It just had no CLI entry point outside a full
  training deploy. Rather than re-implementing harness construction in
  `cli.py` (CLAUDE.md §1b: reuse before reinventing), `discover checkpoint`
  is a thin wrapper: resolve a checkpoint URI, shell out to
  `python -m neuroslm.train_dsl --explore_only --resume_from ... <explore
  knobs>`, propagate its exit code.
- **`_resolve_checkpoint_uri()`** (new, `cli.py`) — extracted from
  `cmd_deploy`'s existing `--resume`/`--latest` block (unchanged behavior;
  pinned by 4 new tests in `test_deploy_safety_gate.py` before the
  refactor, still green after) so `discover checkpoint --checkpoint/--latest`
  resolves checkpoints identically to a training deploy — same
  `find_latest_checkpoint()` HF Hub lookup, same `hf://repo/path` URI
  shape, same error messages.
- **`brian discover checkpoint`** (`cli.py::cmd_discover`, new `checkpoint`
  mode) — flags: `--checkpoint PATH_OR_URI` / `--latest` (+ `--hf-repo` /
  `--hf-prefix`), `--arch` / `--preset` (default from `brian.toml
  [current].arch` / `rcc_bowtie_30m_p4`), `--rounds` / `--pop` /
  `--generations` / `--length` / `--sites` (the same H52/H53 probe knobs),
  `--device`, `--push` (belt-and-braces final push; the underlying
  `run_probe_only()` already pushes every round it saves a winner).
  Rejects with a clear message (no subprocess spawned) when neither
  `--checkpoint` nor `--latest` is given, or when `--latest` finds nothing.
- **`deploy-discover checkpoint`** — added to
  `vast_discover.DEPLOYABLE_MODES` (now `experts`/`trunk`/`explore`/
  `checkpoint`). No onstart-template change needed: `checkpoint` is a
  normal `brian discover <mode>` invocation like the other three, so the
  existing `python3 -m neuroslm.cli discover __MODE__ __DISCOVER_ARGS__
  --push` command line in `vast_discover.py`'s onstart script works
  unmodified — `cmd_deploy_discover` just gained a `checkpoint` branch that
  forwards `--checkpoint`/`--latest`/`--hf-repo`/`--hf-prefix`/`--arch`/
  `--preset`/`--rounds`/`--pop`/`--generations`/`--length`/`--sites` and
  (per the H55.1 cost-bug lesson) always adds `--device auto`. Rejects a
  deploy missing both `--checkpoint` and `--latest` BEFORE the human-
  confirmation gate, matching the existing "reject invalid mode before
  confirming" pattern.
- **Colab**: cell 12/13 (free local-GPU discover) and cell 14/15
  (vast.ai deploy-discover) both gained `checkpoint` as a selectable mode
  with its own knobs block (`CKPT_LATEST`/`CKPT_URI`/`CKPT_ARCH`/
  `CKPT_PRESET`/`CKPT_ROUNDS`/`CKPT_SITES`).

[EVIDENCE: tests/test_discover_checkpoint.py (17) green; tests/test_vast_discover_deploy.py (28, 8 new) green; tests/test_deploy_safety_gate.py (19, 4 new `TestCmdDeployLatestResolution`) green]

[EVIDENCE: tests/test_connectors.py::test_T1_build_env_propagates_explore_fields_when_set / test_T2_build_env_skips_explore_when_off / test_T3_onstart_substitutes_explore_placeholders / test_T4_onstart_defaults_explore_off / test_T5_cmd_deploy_forwards_explore_flags_to_deploy_config (5) green; tests/test_vast_train_dsl_loop_explore.py (4) green; tests/test_connectors.py + test_vast_train_dsl_loop_explore.py + test_deploy_safety_gate.py (72 total) green]
