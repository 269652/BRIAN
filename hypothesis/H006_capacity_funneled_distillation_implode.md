---
code_refs: [neuroslm/harness.py, neuroslm/dsl/training_config.py]
created_at: "2026-06-15T00:30:00Z"
id: H006
proof_path: null
proof_status: missing
references: [formal_framework.md §13, FINDINGS.md Run pre-H24]
status: draft
tags: [distillation, capacity, implode, slm, kl, teacher-student]
test_refs: [tests/training/test_cfd_distillation.py]
theorem_name: Brian.CapacityFunneledDistillationImplode
title: Capacity-Funneled Distillation produces monotone-implode PPL in teacher capacity
updated_at: "2026-06-15T22:00:00Z"
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

## v2 — Generalisation-Funneled Distillation (GFD)

The H22/B6 run falsified the implicit assumption that part (II) of
the theorem (monotone-implode in teacher capacity, measured on the
training distribution) implies a *corresponding* implode on
out-of-distribution data. With SmolLM2-360M replacing GPT-2 as the
expert, otherwise identical config:

| variant         | train PPL | OOD PPL  | gap_ratio (OOD/train) |
|-----------------|-----------|----------|-----------------------|
| GPT-2 expert    | 38.4      | 110.2    | 2.87                  |
| SmolLM2 expert  | **23.6**  | **155.0** | **6.55** (REGRESSED) |

A stronger teacher accelerates *memorisation* of corpus-specific
patterns (the unigram marginal carries most of this signal). CFDv1's
top-K projection treats all positions identically — there is no
mechanism to down-weight frequency-driven peaks vs context-driven peaks.

GFD adds two mechanisms (docs/formal_framework.md §14):

* **M2 (prior-residual sparsification)**: subtract $\gamma \log p_{\mathrm{uni}}$
  from teacher logits before Stage 1. $\gamma = 0$ is bit-identical
  to CFDv1; $\gamma = 1$ removes the unigram floor entirely so the
  distillation channel only carries PMI signal.
* **M4 (pointwise-K from teacher PMI)**: $K(t) = \mathrm{clip}(K_{\max} \cdot \exp(-\mathrm{PMI}(t)/\sigma), K_{\min}, K_{\max})$
  per position. High-PMI positions (sharp peaks on rare-prior tokens)
  get small K → concentrate the signal; low-PMI positions (peaks on
  common tokens) get large K → soft regulariser.

### v2 falsifier — extended four-arm ablation

| Arm | Teacher        | Distill mode      | γ    | pointwise-K | Predicted (vs Arm D) |
|-----|----------------|-------------------|------|-------------|----------------------|
| A   | none           | LM-only           | —    | —           | $P_{\mathrm{LM}}$    |
| C   | adversarial    | naive KL          | —    | —           | explosion            |
| D   | adversarial    | CFDv1             | 0.0  | off         | $\le P_A$ (no-harm)  |
| E   | adversarial    | CFDv1 + **M2**    | 0.5  | off         | $\le P_D$ on training, $<$ OOD gap |
| F   | adversarial    | CFDv1 + **M2+M4** | 0.5  | on          | $\le P_E$, smallest train↔OOD gap   |

Theorems (IV) (M2 preserves no-harm floor) and (V) (M2 strictly
reduces the marginal-imitation gradient) are stated in
`docs/formal_framework.md §14.2`. The v2 contract tests
(`TestCFDv2*` in `tests/training/test_cfd_distillation.py`, 15
cases) verify:

* M2 is identity at γ=0 (bit-identical back-compat)
* M2 downweights common tokens and boosts contextual peaks on rare tokens
* M4's K(t) is monotone-non-increasing in PMI(t) and respects $[K_{\min}, K_{\max}]$
* Variable-K top-K projection collapses to scalar top-K when K is constant

Arms E and F require a real H22/B6-style sweep with the GPT-2/SmolLM2
expert pair to confirm. Deferred to a CDGA experiment.

### v2 hypothesis is **refuted** if any of:

* (IV) $P_E^{(\text{train})} > P_D^{(\text{train})} \cdot 1.05$
  (M2 violates the no-harm floor in practice)
* (V) On the H22/B6 setup, $\mathrm{gap}_E \ge \mathrm{gap}_D$
  (M2 fails to close the train↔OOD gap)
* (VI) $\mathrm{gap}_F > \mathrm{gap}_E$ (M4 makes things worse)
