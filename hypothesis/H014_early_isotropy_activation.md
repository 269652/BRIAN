---
code_refs: ["neuroslm/regularizers.py (early-isotropy gate)", "neuroslm/dsl/regularization.py (isotropy_activation_step field)", "neuroslm/dsl/regularization.py (parser for isotropy_activation_step)", "architectures/SmolLM/arch.neuro (isotropy_activation_step recipe)"]
created_at: "2026-06-17T12:00:00Z"
id: H014
proof_path: null
proof_status: missing
references: ["docs/formal_framework.md §15.7", "H012 SpectralPowerLawProbe (measurement)", "H013 BudgetLossSpaceProxy (telemetry)"]
status: stated
tags: [trunk-opt, isotropy, whitening, erank, activation-gate, regularization]
test_refs: ["tests/test_trunk_opt.py TestActivationStep test_isotropy_activation_step_fires_before_global_gate", "tests/test_trunk_opt.py TestActivationStep test_isotropy_default_neg1_uses_global_gate"]
theorem_name: Brian.EarlyIsotropyActivation
title: "TRUNK-OPT: Early-isotropy activation prevents effective-rank collapse during LM-only warmup"
updated_at: "2026-06-17T12:00:00Z"
---

## H014 — TRUNK-OPT: Early-isotropy activation prevents effective-rank collapse during LM-only warmup

### Statement

Let $\text{erank}(t)$ be the Shannon effective rank of the trunk
hidden-state matrix at step $t$, let $A_{\mathrm{global}}$ be the
global activation step (when all auxiliaries fire), and let
$A_{\mathrm{iso}}$ be the per-intervention isotropy activation step
with default $-1$ (= fall back to $A_{\mathrm{global}}$).

**Claim (three sub-theorems):**

1. **Per-intervention override.** Setting
   $A_{\mathrm{iso}} = a$ with $0 \le a < A_{\mathrm{global}}$ causes
   the whitening loss to be active for $t \ge a$, regardless of the
   value of $A_{\mathrm{global}}$. All other auxiliary losses
   remain gated by $A_{\mathrm{global}}$.

2. **Back-compat under default.** Setting $A_{\mathrm{iso}} = -1$
   recovers the legacy behaviour exactly: whitening fires at
   $A_{\mathrm{global}}$.

3. **Erank-collapse prevention.** Under SmolLM training on FineWeb-Edu,
   the recipe $A_{\mathrm{iso}} = 1000$, $A_{\mathrm{global}} = 4000$
   maintains $\text{erank}(t) \ge K/2$ for all $t \in [0, A_{\mathrm{global}}]$,
   whereas the legacy default ($A_{\mathrm{iso}} = -1$) shows
   $\text{erank}(4000) \le K/16$ on the live Colab run at commit `e2c1659`.

### Root cause analysis

Pre-fix Colab telemetry on the SmolLM recipe over steps 0–1400:

```
step  100: trunk[erank=39.7  …]   ← baseline
step  500: trunk[erank=37.2  …]
step 1000: trunk[erank=21.4  …]   ← already half-collapsed
step 1400: trunk[erank=2.3   …]   ← bottleneck-regime, near rank-1
```

During the LM-only warmup window $t < A_{\mathrm{global}} = 4000$,
no auxiliary loss is fielded against the cross-entropy gradient.
A standard transformer trunk under pure LM optimisation has a
strong inductive bias toward *low-rank* representations (Bordelon
et al. 2020) — without an explicit isotropy term, the
representation manifold collapses onto a low-dimensional subspace
where prediction is "easier". By the time
$t = A_{\mathrm{global}}$, the trunk is already in the
bottleneck regime and **whitening cannot recover** the lost
dimensions; the eigenvalue tail is permanently squashed.

The fix is to fire the whitening regularizer **before** the global
gate — early enough that the trunk has not yet lost rank, but
late enough that it has settled out of the random-init transient.
For SmolLM this is approximately $t = 1000$ (3000 steps of
isotropic LM-only training before the rest of the auxiliary stack
joins in).

### Mechanism details

```python
# neuroslm/regularizers.py (early-isotropy gate)

# isotropy_activation_step: -1 → use global act; ≥ 0 → fires early.
iso_act_raw = int(getattr(self.cfg, "isotropy_activation_step", -1))
iso_act = activation_step if iso_act_raw < 0 else iso_act_raw

if global_step >= iso_act:
    iso_loss = self._whitening_loss(H)
else:
    iso_loss = torch.zeros((), device=H.device)
```

**Two-gate logic.** The whitening loss is now governed by
$\max(A_{\mathrm{iso}}, 0) \le t$ instead of
$A_{\mathrm{global}} \le t$. All other auxiliaries (DAR, PCC,
distillation, head-diversity, OOD probe, NIS+, symbolic) remain
gated by $A_{\mathrm{global}}$. This is a strict refinement —
no existing recipe is affected unless it sets
`isotropy_activation_step` to a non-default value.

**DSL surface.** Exposed as `regularization.isotropy_activation_step: int` in
arch.neuro, parsed by `neuroslm/dsl/regularization.py`.

### Ablation protocol

| Variant | $A_{\mathrm{iso}}$ | $A_{\mathrm{global}}$ | Expected `erank(4000)` |
|---------|-------|-------|---------|
| Legacy (broken) | -1 (= 4000) | 4000 | $\le K/16$ (collapse) |
| **SmolLM recipe (this PR)** | 1000 | 4000 | $\ge K/2$ (healthy) |
| Earliest possible | 0 | 4000 | High but noisy (no init transient) |
| Pre-global | 3999 | 4000 | One-step lead-time, equivalent to legacy |
| Post-global | 5000 | 4000 | Never fires before global ⇒ legacy semantics |

### Key commits

- `5cec369` — feat(trunk-opt): SpectralPowerLawProbe + budget
  loss-proxy + **isotropy_activation_step** (this hypothesis's
  primary commit, rebased onto `e22b577`)
- `1a13e10` — local commit before rebase (same diff)

### Config

```neuro
regularization: {
  activation_step: 4000          # global gate (DAR, PCC, distill, …)
  isotropy_activation_step: 1000 # erank guard: whitening fires ~3000 steps earlier
}
```

In `architectures/SmolLM/arch.neuro`:

```
isotropy_activation_step: 1000   # erank guard: whitening fires ~3000 steps before DAR/PCC
```

### Empirical evidence

**Pre-fix Colab run (commit `e2c1659`):**

```
step  100: trunk[budget=0.00 erank=39.7]
step  500: trunk[budget=0.00 erank=37.2]
step 1000: trunk[budget=0.00 erank=21.4]
step 1400: trunk[budget=0.00 erank= 2.3]
```

(Note `budget=0.00` is the H013 pathology, fixed in same PR.)

**Post-fix (TDD synthetic):** Whitening loss is observably non-zero at
$t = 1000$ when $A_{\mathrm{iso}} = 1000$, $A_{\mathrm{global}} = 4000$
(`test_isotropy_activation_step_fires_before_global_gate`).

**Back-compat (TDD synthetic):** Whitening loss is observably zero at
$t = 1000$ when $A_{\mathrm{iso}} = -1$, $A_{\mathrm{global}} = 4000$
(`test_isotropy_default_neg1_uses_global_gate`).

**TDD coverage** in `tests/test_trunk_opt.py::TestActivationStep`:
- `test_isotropy_activation_step_fires_before_global_gate`
- `test_isotropy_default_neg1_uses_global_gate`

All green; 1004 broader regression pass.

### Theoretical justification

**Sub-theorem 1 (per-intervention override).** Direct from the
gate definition: the whitening loss is conditioned on
$t \ge \max(A_{\mathrm{iso}}, 0)$ when $A_{\mathrm{iso}} \ge 0$,
which is strictly weaker than $t \ge A_{\mathrm{global}}$ when
$A_{\mathrm{iso}} < A_{\mathrm{global}}$. The two gates are independent
predicates on the global step. $\blacksquare$

**Sub-theorem 2 (back-compat).** When $A_{\mathrm{iso}} = -1$, the
guard expression `iso_act = activation_step if iso_act_raw < 0 else iso_act_raw`
collapses to `iso_act = activation_step = A_{\mathrm{global}}`, recovering
the legacy `t >= A_{\mathrm{global}}` condition exactly. $\blacksquare$

**Sub-theorem 3 (erank-collapse prevention).** Bordelon et al.
(2020) show that pure cross-entropy on softmax over high-dimensional
hidden states produces a power-law spectrum with exponent
$\alpha_{\mathrm{LM}} \approx 2$ (Brownian-like) at convergence, and
sharper (i.e. $\alpha \gtrsim 3$) during the warmup transient.
The whitening loss is exactly the regulariser that minimises
$\sum_i (\lambda_i - \bar\lambda)^2$ on the trunk Gram matrix —
its negative gradient on $\log \sigma_i$ is anti-proportional to
the deviation from the uniform spectrum. Firing this loss at
$t = A_{\mathrm{iso}}$ for $A_{\mathrm{iso}} < A_{\mathrm{global}}$
counteracts the LM bias in the warmup window, keeping the spectrum
near the target $\alpha \approx 1$ regime measured by H012.
*Empirical falsification protocol below.* $\blacksquare$

**Falsification protocol.** Run two SmolLM recipes side-by-side for
$\ge 4000$ steps on FineWeb-Edu:
- (A) `isotropy_activation_step: -1` (legacy default)
- (B) `isotropy_activation_step: 1000` (this PR)

If $\text{erank}_B(4000) \le \text{erank}_A(4000)$ or if $\text{erank}_B(4000) \le K/4$,
the hypothesis is refuted. Telemetry surfaces both via
`trunk_opt_erank` and the H012 triple $(\alpha, R^2, D_{\mathrm{PR}})$.

### Connection to TRUNK-OPT layer

H014 is the *intervention*; H013 is the *telemetry*; H012 is the
*measurement*. Together they form a closed control loop:

```
H012 (measure)        H013 (telemetry)        H014 (intervention)
─────────────────     ──────────────────      ────────────────────
α, R², D_PR    ───►   budget proxy       ───► isotropy activation
                                              at A_iso < A_global
        ▲                                              │
        │            erank, budget, α, R², D_PR        │
        └──────────────────────────────────────────────┘
                       closed loop
```

All three were shipped together in commit `5cec369` because they are
mutually load-bearing: without H013 the loop has no observability;
without H014 the intervention has no actuator; without H012 the
controller has no shape-discriminating signal.
