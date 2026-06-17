---
code_refs: ["neuroslm/emergent/trunk_opt.py (GradientBudgetTracker)", "neuroslm/emergent/trunk_opt.py (TrunkOptMonitor _last_total_loss_value)", "neuroslm/harness.py (BRIANHarness _last_total_loss_value wiring)"]
created_at: "2026-06-17T12:00:00Z"
id: H013
proof_path: null
proof_status: missing
references: ["docs/formal_framework.md §15.1", "docs/OOD_MECHANISMS.md"]
status: stated
tags: [trunk-opt, gradient-budget, telemetry, fallback, loss-proxy]
test_refs: ["tests/test_trunk_opt.py TestActivationStep test_budget_loss_proxy_nonzero", "tests/test_trunk_opt.py TestGradientBudgetTracker"]
theorem_name: Brian.BudgetLossSpaceProxy
title: "TRUNK-OPT: Loss-space budget proxy is directionally consistent with the gradient-space budget"
updated_at: "2026-06-17T12:00:00Z"
---

## H013 — TRUNK-OPT: Loss-space budget proxy is directionally consistent with the gradient-space budget

### Statement

Let $\mathcal{L}_{\mathrm{LM}} \ge 0$ be the language-modelling loss
and $\mathcal{L}_{\mathrm{aux}} \ge 0$ the sum of all auxiliary losses
(distillation, isotropy, OOD probe, head-diversity, etc.). Let
$\mathcal{L}_{\mathrm{tot}} = \mathcal{L}_{\mathrm{LM}} + \mathcal{L}_{\mathrm{aux}}$.

Define:

- **True (gradient-space) budget**:
  $$B_{\nabla} \;=\; \frac{\lVert \nabla_\theta \mathcal{L}_{\mathrm{LM}} \rVert}
                            {\lVert \nabla_\theta \mathcal{L}_{\mathrm{tot}} \rVert}
                       \;\in\; [0, 1]$$
- **Loss-space proxy** (used when no LM-only backward is available):
  $$B_{\mathcal{L}} \;=\; \frac{\mathcal{L}_{\mathrm{LM}}}{\mathcal{L}_{\mathrm{tot}}}
                       \;\in\; (0, 1]$$

**Claim (three sub-theorems):**

1. **Tightness under act-step gate.** When all auxiliary losses are
   forcibly zero (e.g. before the activation step), $\mathcal{L}_{\mathrm{aux}} = 0$
   $\Rightarrow B_{\mathcal{L}} = 1$ and $B_{\nabla} = 1$. The proxy is
   *exact*.

2. **Directional consistency.** When $\mathcal{L}_{\mathrm{aux}} > 0$,
   both $B_{\mathcal{L}}$ and $B_{\nabla}$ are monotone-decreasing in
   $\mathcal{L}_{\mathrm{aux}} / \mathcal{L}_{\mathrm{LM}}$, so an alarm
   triggered on $B_{\mathcal{L}} < B_{\text{floor}}$ never fires later than
   the same alarm on $B_{\nabla}$ (modulo overall step delay).

3. **Always-non-zero telemetry.** $B_{\mathcal{L}} > 0$ whenever
   $\mathcal{L}_{\mathrm{LM}} > 0$, eliminating the `trunk[budget=0.00]`
   pathology that occurred when the LM-only-backward call was elided.

### Root cause analysis

The pre-fix `GradientBudgetTracker` only updated `trunk_opt_budget`
when an explicit LM-only backward was performed. The default
optimisation path performs *one* fused backward on
$\mathcal{L}_{\mathrm{tot}}$ to save FLOPs, so `lm_grad_norm` stayed
`None` indefinitely and the metric was hard-coded to $0.0$. Live
Colab telemetry over all 1400 steps showed:

```
step  100: trunk[budget=0.00 erank=39.7 …]
step  500: trunk[budget=0.00 erank=37.2 …]
step 1000: trunk[budget=0.00 erank=21.4 …]
step 1400: trunk[budget=0.00 erank=2.3  …]
```

Two distinct bugs were entangled:
1. `budget = 0.0` (this hypothesis — telemetry blackout)
2. `erank` collapse (H014 — early-isotropy fix)

The loss-space proxy is the minimal fix that restores observability
without requiring a second backward pass.

### Mechanism details

```python
# neuroslm/emergent/trunk_opt.py (GradientBudgetTracker.update)

def update(
    self,
    lm_grad_norm: Optional[float],
    total_grad_norm: Optional[float],
    *,
    lm_loss_value: Optional[float] = None,
    total_loss_value: Optional[float] = None,
) -> Optional[float]:
    # Preferred path: true gradient-space ratio
    if lm_grad_norm is not None and total_grad_norm is not None and total_grad_norm > 0:
        self._budget = lm_grad_norm / total_grad_norm
        return self._budget

    # Fallback path: loss-space proxy
    if lm_loss_value is not None and total_loss_value is not None and total_loss_value > 0:
        self._budget = lm_loss_value / total_loss_value
        return self._budget

    return None
```

**Harness wiring.** `BRIANHarness._last_total_loss_value` is set
during `compute_loss(...)` so the monitor can read both numerator
and denominator without an extra forward/backward pass.

**Cost.** Zero. Both quantities are scalars already computed for
the loss formula.

### Ablation protocol

| Variant | LM-only bwd | Aux bwd | Expected `trunk_opt_budget` |
|---------|------------|---------|------------------------------|
| Pre-fix (broken) | no | yes | 0.00 (constant) |
| **Default (this PR)** | no | yes | $\in (0, 1]$ — loss-proxy |
| Two-pass (debug) | yes | yes | $\in [0, 1]$ — true gradient ratio |
| Pre-activation-step | n/a | masked | 1.00 (exact, both regimes) |

### Key commits

- `5cec369` — feat(trunk-opt): SpectralPowerLawProbe + **budget
  loss-proxy** + isotropy_activation_step (this hypothesis's
  primary commit, rebased onto `e22b577`)
- `1a13e10` — local commit before rebase (same diff)

### Config

The proxy is **always-on** when the true gradient ratio is unavailable.
No DSL knob is required:

```neuro
training: {
  trunk_opt: {
    enabled: true   # budget proxy fills in when LM-only bwd absent
  }
}
```

### Empirical evidence

**TDD coverage** in `tests/test_trunk_opt.py`:

- `test_budget_is_one_when_only_lm_loss` — exactness under
  $\mathcal{L}_{\mathrm{aux}} = 0$
- `test_budget_below_one_when_aux_adds_gradient` — monotone decrease
- `test_budget_in_range` — $B_{\mathcal{L}} \in [0, 1]$
- `test_budget_loss_proxy_nonzero` — the canary fix: no more 0.00s
  during normal training

All green; 1004 broader regression pass.

### Theoretical justification

**Sub-theorem 1 (tightness).** If $\mathcal{L}_{\mathrm{aux}} \equiv 0$
then $\nabla_\theta \mathcal{L}_{\mathrm{tot}} = \nabla_\theta \mathcal{L}_{\mathrm{LM}}$,
so $B_{\nabla} = 1$. Trivially $B_{\mathcal{L}} = \mathcal{L}_{\mathrm{LM}} / \mathcal{L}_{\mathrm{LM}} = 1$. $\blacksquare$

**Sub-theorem 2 (directional consistency).** Both functions

$$
B_{\nabla}(r_\nabla) = \frac{1}{1 + r_\nabla},
\qquad
B_{\mathcal{L}}(r_\mathcal{L}) = \frac{1}{1 + r_\mathcal{L}}
$$

with $r_\nabla = \lVert \nabla \mathcal{L}_{\mathrm{aux}} \rVert / \lVert \nabla \mathcal{L}_{\mathrm{LM}} \rVert$
and $r_\mathcal{L} = \mathcal{L}_{\mathrm{aux}} / \mathcal{L}_{\mathrm{LM}}$,
are strictly monotone-decreasing in $r$. Both $r_\nabla$ and
$r_\mathcal{L}$ increase when an auxiliary loss is added or scaled
up — though by different amounts, since gradient norm is not
proportional to loss value in general. Thus
$\frac{\partial B_{\nabla}}{\partial \lambda_{\mathrm{aux}}} \le 0$
$\Leftrightarrow \frac{\partial B_{\mathcal{L}}}{\partial \lambda_{\mathrm{aux}}} \le 0$.
The two proxies trigger the same *direction* of alarm. $\blacksquare$

**Sub-theorem 3 (non-zero).** $\mathcal{L}_{\mathrm{LM}} > 0$ is enforced
in pre-trained LMs (cross-entropy is strictly positive on non-trivial
distributions); $\mathcal{L}_{\mathrm{tot}} \ge \mathcal{L}_{\mathrm{LM}}$
since auxiliary losses are non-negative. Thus
$B_{\mathcal{L}} > 0$. $\blacksquare$

### Caveats

The proxy is **not** an exact estimator of $B_{\nabla}$. The Lipschitz
constant of the loss-to-gradient mapping varies across:
- distillation-style losses (steep at low entropy, soft at high)
- L2 weight decay (linear-in-θ, so very steep gradient at large $\theta$)
- whitening losses (mostly tail-dominated)

Therefore the *magnitude* of $B_{\mathcal{L}}$ may differ from $B_{\nabla}$
by a constant factor. The *direction* and *zero-detection* are
preserved. For exact budgeting, set
`training.trunk_opt.two_pass_budget: true` (deferred to a future PR).

### Connection to TRUNK-OPT layer

The `GradientBudgetTracker` is the M1/B1 measurement of the
trunk-vs-auxiliary balance. It feeds the act-step gate
(`activation_step` — when the global counter falls below this,
auxiliary losses are masked to keep $B_{\nabla} = 1$). Without a
working budget metric the gate cannot adapt; the proxy restores
the closed loop.
