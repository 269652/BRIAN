# Emergent Topology — RCC-Bowtie v2 design spec

**Status:** Draft 1, 2026-06-03.
**Author:** GitHub Copilot agent (architect role) + S. Morro.
**Scope:** Six mechanisms (C1–C6) that turn the currently-decorative biology of
`architectures/rcc_bowtie` into measurable substrate dynamics, and add a
topological-charge field that supports genuine *emergent* phenomena (the
skyrmion analog for sequence models).

---

## 0. Why this document exists

The 2026-06-03 baseline trajectory (40k-step run, all interventions OFF) made
two facts inescapable:

1. **The trunk transformer is healthy.** Train PPL fell 50132 → 552 over 2040
   steps, wiki PPL fell 9166 → 4081, `gap_ratio` *oscillates* in
   [4.4, 7.8] rather than monotonically widening. The earlier "widening gap"
   panic was an artefact of the failed PR2/PR-A/PR-B interventions, not a real
   defect in the substrate.
2. **Most of the "biology" is dead.** Of the bio readouts:
   - `NT[DA=0.15 NE=0.20 5HT=0.35 ACh=0.25 eCB=0.10 Glu=0.45 GABA=0.15]` is
     **constant for 2040 steps**. The metric observer's `NTSystem` is a tiny
     drift-back-to-baseline ODE driven only by global activation magnitude,
     which is itself stable after warmup. Receptors see noise.
   - `ign ≈ 0.97 ± 0.02` (saturated). GWS ignition fires every step. It is not
     a workspace; it is a wire.
   - `mesoLG ≈ 0.50 ± 0.03` (chance). Mesoscopic phase-locking is reporting
     pure noise.
   - `λ₁ = 0.075` (clamped, never moves). Fiedler is a graph property, not a
     dynamical observable.
   - `osc[δ θ γ]` is a softmax over uncorrelated activation-power bands — no
     cross-frequency coupling, no binding.

Only `Φ` (0.04 → 0.11, +170%) is honestly growing. That is the *one* signal
that the bowtie graph is doing dynamical work.

This document specifies six mechanisms that, together, make every dead
observable alive and add a new conserved-charge field on top.

## 1. Design principle: emergence from conserved topological charge

Skyrmions emerge from three ingredients in a continuum spin model:

1. A continuous unit-norm field $\vec m(x), |\vec m|=1$
2. A symmetry-breaking term (Dzyaloshinskii–Moriya: chiral, $\vec m\cdot(\nabla\times\vec m)$)
3. A conserved topological charge $Q\in\mathbb Z$

The discrete sequence analog is exact:

- **Field:** hidden states $h_t/\|h_t\| \in S^{d-1}$ along the token axis.
- **Symmetry-breaking:** a skew-symmetric coupling $R\in\mathfrak{so}(d)$ that
  splits each pair $(h_t, h_{t+1})$ into an in-plane and out-of-plane
  component.
- **Charge:** the integer winding number
  $$Q = \frac{1}{2\pi}\sum_t \arg\!\left(\langle h_t, h_{t+1}\rangle + i\,\langle h_t, R h_{t+1}\rangle\right) \in \mathbb Z$$

$Q$ cannot be smoothly destroyed by gradient descent without crossing a
*domain wall* (a token at which $\langle h_t, h_{t+1}\rangle \le 0$). This
gives the model **discrete, perturbation-robust long-range memory** the way
skyrmion lattices give matter robust magnetic memory.

We do not need to introduce $Q$ as a loss term to observe it: $Q(t)$ can be
computed post-hoc from the trunk's existing residual stream. If the model
spontaneously develops discrete plateaus of $Q$ aligned with discourse units,
that is *emergence*. If it does not, $Q$ is a candidate auxiliary signal we
can encourage with a bounded loss (Phase 8+, out of scope here).

## 2. Falsifiable predictions

Every phase below has at least one prediction that can be checked against
real training logs:

| Phase | Predicted observable behaviour |
|---|---|
| C1 | NT values move with training state (gnorm spikes → NE spike; surprise → DA phasic) |
| C2 | `ign_rate` distribution becomes bimodal (mostly 0.0–0.2, occasional spikes ≥ 0.7), not the saturated 0.97 ± 0.02 |
| C3 | `pc_residual` falls monotonically as the motor→sensory reentry learns; correlation with `train_loss` decay should be ≥ 0.5 |
| C4 | `Q(t)` shows step-function plateaus that align with paragraph/topic boundaries (test on WikiText-103 eval); wall count ≈ #paragraphs |
| C5 | Specialisation index of K parallel workspaces rises monotonically (one expert per token type) |
| C6 | `pac` (gamma amplitude ↔ theta phase coupling) rises with `Φ`; correlation ≥ 0.3 by step 4000 |

A phase that fails its prediction is **deleted**, not patched.

## 3. The six mechanisms

### C1 — Driven neuromodulators (`emergent/driven_nt.py`)

Replace the constant-drift `MetricObserver.NTSystem` with a closed-loop
controller that reads scalar statistics from the trunk every step:

| NT | Driver | Closed-form |
|---|---|---|
| DA | per-step surprise = $\bar L_\text{recent} - L_t$ | $\mathrm{DA}_t = \sigma(\text{surprise}/\sigma_L)$ |
| NE | grad-norm EMA | $\mathrm{NE}_t = \tanh(\|\nabla\|_2/\tau_\text{NE})$ |
| 5HT | long-window loss EMA | $\mathrm{5HT}_t = 1 - \sigma(L_\text{slow}-L_\text{ref})$ |
| ACh | attention entropy mean (or activation peakiness if attention unavailable) | $\mathrm{ACh}_t = 1 - H(\text{attn})/\log T$ |
| eCB | activation-norm/dim (retrograde inhibition) | $\mathrm{eCB}_t = \tanh(\|h\|_2/(\sqrt d\cdot\tau_\text{eCB}))$ |
| Glu | mean forward magnitude | $\mathrm{Glu}_t = \tanh(\bar{|h|}/\tau_\text{Glu})$ |
| GABA | $1-\overline{\mathrm{ign}}$ (inhibits when workspace saturates) | $\mathrm{GABA}_t = 1-\mathrm{ign}_t$ |

All bounded in [0,1], all functions of *training state*, not constants.
Implementation: one new class `DrivenNTSystem`, drop-in replacement for
`metrics.NTSystem` selected by `enable_emergent=True`.

**Tests:** monotonicity, bounds, response to synthetic step-input on each
driver, baseline-recovery when drivers are constant.

### C2 — Metastable ignition (`emergent/metastable_ignition.py`)

Current `gws_ignition()` returns a continuous peakiness ∈ [0,1] that
saturates. Replace with an **event detector**:

$$g_t = \sigma\!\left(\frac{\|x_t\|_2 - \theta_t - \beta\cdot\mathrm{NE}_t}{\tau}\right)\cdot\mathbb 1[\max p > p_*]$$

with an EMA-tracked adaptive threshold $\theta_t$ targeting a desired
ignition rate $\rho^* = 0.2$ (Dehaene-correct: ignition is *rare*).

Telemetry: `ign_rate` (fraction of steps with $g_t > 0.5$), `ign_strength`
(mean over events), `ign_threshold` (current $\theta_t$).

**Tests:** rate converges to target on random input, threshold adapts under
distribution shift, NE coupling lowers threshold under high arousal.

### C3 — Predictive-coding reentry residual (`emergent/pc_reentry.py`)

The `motor → sensory` synapse already exists in the bowtie. We *observe*
without changing the forward pass: at each step, given cached
`h_motor[t-1]` and `h_sensory[t]`, compute

$$e_t = h_{\text{sensory},t} - W_\text{pred}\, h_{\text{motor},t-1}$$

with $W_\text{pred}$ a learnable diagonal+low-rank predictor (small param
budget, registered on the harness as a buffer-side module that does **not**
gradient-flow into the trunk in this phase).

Telemetry: `pc_residual_norm`, `pc_explained_var`.

**Tests:** predictor improves on synthetic linear-dynamics data; residual
is unchanged when motor and sensory are uncorrelated noise; predictor
gradient does not leak into trunk (`assert h.grad is None after backward`).

### C4 — Topological charge $Q(t)$ (`emergent/topological_charge.py`)

The core of this proposal. Given the trunk's per-block hidden trajectory
$h_t\in\mathbb R^d$ (from `_layer_acts[-1]`), normalise to the sphere,
compute the running winding number with a *fixed* random skew-symmetric
$R$ (no parameters), and report:

- `Q_total` = total signed winding over the sequence
- `Q_walls` = number of sign flips of $\langle h_t, h_{t+1}\rangle$
  (domain walls)
- `Q_plateau_len` = mean length of constant-$Q$ runs

$R$ is initialised once (seed-stable, lives in the observer) so that runs
are comparable. A learnable $R$ is a Phase 8 option.

**Tests:** Q is integer-valued on synthetic spiral data, walls count
matches paragraph boundaries on a controlled prompt, sign-flip under
reversed sequence, invariance under global rotation.

### C5 — Bowtie lattice specialisation probe (`emergent/bowtie_lattice.py`)

The current architecture has *one* GWS at $d=256$. Splitting it into K
parallel narrow workspaces with lateral inhibition is a big surgery (Phase
8+). What we ship now is the **specialisation probe**: given the existing
GWS activations sliced into K=4 contiguous chunks, compute the
mutual-information–normalised specialisation index

$$S = \frac{1}{K}\sum_k \max_c \frac{p(c\,|\,k)}{p(c)}$$

over a token-class taxonomy (`dialogue` / `prose` / `code` / `other`)
inferred from the chat/prose mixture indicator.

Telemetry: `lattice_spec` ∈ [1, ∞), `lattice_active_k`.

If `lattice_spec` stays at 1.0, the existing single workspace is not even
*latently* specialising — confirming the C5 surgery is needed. If it rises,
the architecture is doing K-bowtie work for free and we instead refactor
the GWS to expose those K lanes explicitly.

**Tests:** index = 1 when slices are random, = K when slices are
perfectly disjoint over classes, monotonic under interpolation.

### C6 — Phase-amplitude coupling (`emergent/pac_binding.py`)

Currently `osc[δ θ γ]` is a power-band softmax over the
`OscillationTracker` signal. We add the *Tort modulation index* over the
same buffer:

$$\mathrm{PAC} = \frac{H_{\max} - H(P)}{H_{\max}}, \quad P_j = \frac{\bar A^\gamma_j}{\sum_j \bar A^\gamma_j}$$

where $\bar A^\gamma_j$ is the mean gamma-band envelope amplitude in theta
phase bin $j$, and $H(\cdot)$ is Shannon entropy of the resulting
distribution over phase bins. Bounded in [0, 1]. PAC = 0 ⇔ amplitude is
uniform across phase bins (no coupling). PAC > 0 ⇔ gamma envelope locks
to theta phase (cross-frequency binding).

Telemetry: `pac`, `pac_pref_phase`.

**Tests:** PAC ≈ 0 on uncorrelated noise, PAC ≈ 1 on a synthetic
modulated signal $A = (1 + \cos\theta)\cdot \gamma$, robust to scale
and DC offset.

## 4. Integration contract

A single switch enables *all* phases at once: `MetricObserver(... ,
enable_emergent=True)`. The new metric keys are:

```
nt_driven:        same shape as old `nt`, alive
ign_rate:         float in [0,1]
ign_strength:     float in [0,1]
ign_threshold:    float, current θ
pc_residual:      float ≥ 0
pc_explained:     float in [0,1]
Q_total:          int (signed)
Q_walls:          int ≥ 0
Q_plateau_len:    float ≥ 0
lattice_spec:     float ≥ 1
pac:              float in [0,1]
pac_pref_phase:   float in [-π, π]
```

When `enable_emergent=False` (default), `observe()` returns the same dict
it returned before this PR — byte-identical for back-compat with logs,
checkpoints, and downstream eval scripts.

Log-line gating: a new column block appears only when at least one
emergent value is non-trivial in the current window, mirroring the
existing `reg_str` convention in `train_dsl._format_log_line`.

## 5. What is explicitly NOT in this PR

- **No trunk forward-pass changes.** The transformer blocks are
  untouched. We *do* read training-state scalars (loss, grad-norm,
  activation magnitude) to drive the NTs, but the values produced are
  observation-only in this PR — they are not yet plugged back into the
  modulation matrix in `arch.neuro`. (That promotion is Phase 8.)
- **No new loss terms.** No aux loss is registered. The trunk's
  gradient signal is unchanged.
- **No changes to checkpoint format.** Observer state is rebuilt from
  scratch on resume.
- **Math-first DSL surface.** All six mechanisms now have canonical
  declarative equations in `architectures/rcc_bowtie/lib/emergent.neuro`,
  imported by `arch.neuro`. The Python in `neuroslm/emergent/` is the
  *lowering*, not the spec.
- **Active by default.** `MetricObserver(enable_emergent=True)` is the
  new default so every training run produces the C1–C6 telemetry
  needed to evaluate which mechanism to promote next. Pass
  `enable_emergent=False` to recover the pre-2026-06-03 log shape.

## 6. Phase order and acceptance criteria

| Phase | Acceptance |
|---|---|
| 1 (C1) | All `tests/emergent/test_driven_nt.py` pass; baseline run with `--emergent` shows NT values that move ≥ 0.05 in the first 500 steps |
| 2 (C2) | `tests/emergent/test_metastable_ignition.py` pass; `ign_rate` distribution in baseline run is *not* dominated by the [0.9, 1.0] bin |
| 3 (C3) | `tests/emergent/test_pc_reentry.py` pass; `pc_residual` is monotonically non-increasing over a 1000-step EMA |
| 4 (C4) | `tests/emergent/test_topological_charge.py` pass; `Q_walls` correlates with paragraph count on a held-out WikiText sample at $r ≥ 0.3$ |
| 5 (C5) | `tests/emergent/test_bowtie_lattice.py` pass; `lattice_spec` is reported with sane defaults |
| 6 (C6) | `tests/emergent/test_pac_binding.py` pass; `pac` > 0 on synthetic modulated input |
| Integration | Full `pytest tests/` passes (≥433 + new). Baseline log line is byte-identical when `--emergent` is OFF. |

## 7. After landing

Phase 8 (out of scope) is the *interventional* layer: once telemetry shows
which mechanisms actually carry signal, we promote the corresponding
mechanism into a forward-pass change (gated ignition, real PC residual
gradient, bounded $-\lambda_Q \tanh|Q|$ loss, K-lane GWS surgery, PAC
reward). Each such promotion is a separate PR, each gated behind a switch,
each A/B'd against the baseline established in this PR.

This is the discipline that the failed PR2/PR-A/PR-B arc lacked:
**telemetry before intervention, every time.**
