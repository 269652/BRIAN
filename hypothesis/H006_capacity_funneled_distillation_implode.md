---
code_refs:
  - neuroslm/harness.py::_cortex_fusion_aux_step
  - neuroslm/harness.py::_cfd_topk_target
  - neuroslm/harness.py::_cfd_effective_temperature
  - neuroslm/harness.py::_cfd_grad_alignment_gate
  - neuroslm/dsl/training_config.py::MultiCortexConfig.cfd_*
created_at: "2026-06-15T00:30:00Z"
id: H006
proof_path: null
proof_status: missing
references:
  - "formal_framework.md §13"
  - "FINDINGS.md Run pre-H24 — CFD design"
status: draft
tags: [distillation, capacity, implode, slm, kl, teacher-student, optimal-training]
test_refs:
  - tests/training/test_cfd_distillation.py
theorem_name: Brian.CapacityFunneledDistillationImplode
title: Capacity-Funneled Distillation produces monotone-implode PPL in teacher capacity
updated_at: "2026-06-15T00:30:00Z"
---

## Statement

Let $\theta_s$ be a student LM with effective capacity $C_s$ (parameter
count, or more precisely the dimension of the reachable softmax simplex
under the student's architecture). Let $\mathcal{T} = \{t_1, t_2, \dots\}$
be a family of teacher LMs with capacities $C_{t_1} < C_{t_2} < \dots$,
all aligned to the same LM objective on the same data distribution
$\mathcal{D}$.

Define the **Capacity-Funneled Distillation** loss

$$
\mathcal{L}_{\mathrm{CFD}}(\theta_s; t)
  \;=\; \mathcal{L}_{\mathrm{LM}}(\theta_s)
       + \lambda_{\mathrm{eff}}(\theta_s, t) \cdot T_{\mathrm{eff}}^2 \cdot
         \mathrm{KL}\!\left(\tilde{p}_t^{(K, T_{\mathrm{eff}})} \;\Big\|\; p_{\theta_s}^{(T_{\mathrm{eff}})}\right)
$$

where, with $p_t = \mathrm{softmax}(t(x))$ the raw teacher distribution,

1. **(Stage 1 — top-$K$ rank-preserving sparsification.)**
   $\tilde{p}_t^{(K, T)}$ keeps the top-$K$ teacher modes (at temperature $T$)
   and redistributes the remaining mass uniformly over the $V{-}K$ tail.
2. **(Stage 2 — entropy-matched temperature.)**
   $T_{\mathrm{eff}} = T_0 \cdot \max(1, H(p_{\theta_s}) / H(p_t))$ is
   computed per batch so the teacher's entropy is rescaled to a value the
   student can match.
3. **(Stage 3 — gradient-alignment gate.)**
   $\lambda_{\mathrm{eff}} = \lambda_0 \cdot \frac{1 + g_{\mathrm{align}}}{2}$
   where $g_{\mathrm{align}} = \cos(\nabla_\theta \mathcal{L}_{\mathrm{distill}},
   \nabla_\theta \mathcal{L}_{\mathrm{LM}}) \in [-1, 1]$ is a one-scalar
   gradient-alignment probe on the trunk's last-layer bias.

Let $\theta_s^\star(t) = \arg\min_\theta \mathcal{L}_{\mathrm{CFD}}(\theta; t)$
and $\mathrm{ppl}(\theta) = \exp(\mathcal{L}_{\mathrm{LM}}(\theta))$.

**Then:**

(I) **(No-harm floor.)** For every teacher $t$:
$$
\mathrm{ppl}\bigl(\theta_s^\star(t)\bigr) \;\le\; \mathrm{ppl}\bigl(\theta_s^\star(\varnothing)\bigr)
$$
where $\theta_s^\star(\varnothing)$ is the LM-only optimum.

(II) **(Monotone implode in teacher capacity.)** If $t_2$ contains all
the LM-aligned information of $t_1$ and strictly more (formally:
$\mathrm{KL}(p_{\mathcal{D}} \| p_{t_2}) < \mathrm{KL}(p_{\mathcal{D}} \| p_{t_1})$),
then
$$
\mathrm{ppl}\bigl(\theta_s^\star(t_2)\bigr) \;\le\; \mathrm{ppl}\bigl(\theta_s^\star(t_1)\bigr)
$$
with strict inequality whenever the top-$K$ projection of $t_2$ disagrees
with $t_1$ on $x \sim \mathcal{D}$ on a set of positive measure.

(III) **(Optimal-training existence.)** For fixed student capacity $C_s$,
the map $C_t \mapsto \mathrm{ppl}(\theta_s^\star(t))$ has a unique
infimum $\mathrm{ppl}^\star(C_s)$ attained in the limit
$C_t \to \infty$, and $\mathrm{ppl}^\star(C_s)$ is strictly decreasing
in $C_s$.

## Why it matters

This is the formal restatement of the user's intuition that **there
exists an optimal training oracle whose information density matches the
student's capacity**. Three consequences:

* **The teacher-too-strong bug is structurally impossible under CFD.**
  Naive KL with reduction='batchmean' and a high-capacity teacher
  produced PPL explosion (H22, run 40952126: PPL 603 at step 500). The
  three CFD stages each kill one ingredient of that explosion (Stage 1
  removes unrepresentable tail-mass gradient; Stage 2 removes
  sharpness mismatch; Stage 3 makes the floor (I) mechanical, not
  asymptotic).
* **SLMs have a well-defined "implode optimum."** Part (III) says that
  for fixed parameter budget, there is a measurable
  $\mathrm{ppl}^\star(C_s)$ — the best PPL any LM with $C_s$ parameters
  can achieve when distilled from an arbitrarily strong, CFD-funneled
  teacher. This is a *parameter-efficient frontier*. We can chart it
  empirically by scaling $C_t$ at fixed $C_s$ and observing the implode.
* **The MoE-cortex picture is unified.** The current trunk + cortex
  fusion (`MultiCortexConfig.experts`) becomes a single instance of
  CFD where the teacher is the cortex ensemble. Stage 1+2+3 give the
  fusion a principled λ schedule that's currently hand-tuned through
  `distillation_gap_floor` / `_ceiling`.

## Falsifier

`tests/training/test_cfd_distillation.py` runs the four-arm ablation on
a 1M-param trunk over 200 steps:

| Arm | Teacher | Distill mode | Predicted final PPL |
|---|---|---|---|
| A | none | — | $P_{\mathrm{LM}}$ |
| B | gpt2 | naive KL | $\le P_{\mathrm{LM}}$ |
| C | SmolLM2-360M | naive KL | $\gg P_{\mathrm{LM}}$ (explosion) |
| D | SmolLM2-360M | **CFD** | $< P_B \le P_{\mathrm{LM}}$ |

The hypothesis is **refuted** if any of:
* (I) $P_D > P_A + \tau$ (no-harm violation; $\tau = 5\%$ tolerance)
* (II) $P_D \ge P_B$ (monotone-implode violation under capacity
  comparison)
* The grad-alignment scalar $g_{\mathrm{align}}$ stays negative on
  more than 50% of steps with CFD-enabled SmolLM2 (Stage 3 inert).

## Proof sketch

Parts (I) and (III) reduce to convex-analysis arguments on the
Bregman-divergence landscape of CFD; the Lean stub at
`hypothesis/proofs/H006_capacity_funneled_distillation_implode.lean`
will formalise (I) first (the no-harm floor) since (II) needs an
information-theoretic hypothesis on $t_2 \succeq t_1$ that is itself
non-trivial to discharge.
