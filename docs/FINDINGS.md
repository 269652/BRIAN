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

## What proved to solve or break things — the punchline list

### Things that demonstrably solved something
- **Trunk gradient isolation (§5.2 / H7)** — *fixed* the post-awakening collapse. Single most important convergence fix.
- **ReZero zero-init forward gates (§5.3 / H8)** — *removed* the awakening discontinuity. Modest gap_ratio win (5.22 vs 6.34); no absolute-OOD win.
- **Recursive reasoning (§5.4 / H9)** — *improves* in-distribution training quality (~20%).
- **PCT loss-only (§5.5 / H10)** — *first* sub-5× gap_ratio in the arc, with the matched-PPL caveat.

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
