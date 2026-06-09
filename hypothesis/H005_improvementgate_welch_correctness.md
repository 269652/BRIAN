---
code_refs: [neuroslm/verification/improvement_gate.py]
created_at: "2026-06-09T18:00:16Z"
id: H005
proof_path: hypothesis/proofs/H005_improvementgate_welch_correctness.lean
proof_status: stub
references: [formal_framework.md §9, formal_framework.md §10.2]
status: stated
tags: [improvement-gate, welch, statistical, admission]
test_refs: [tests/verification/test_improvement_gate.py]
theorem_name: Brian.ImprovementGateWelch
title: ImprovementGate Welch correctness
updated_at: "2026-06-09T18:00:16Z"
---

**Statement.** Given samples $X \sim \mathcal{D}_{\text{before}}$, $Y \sim \mathcal{D}_{\text{after}}$ and a one-sided Welch's $t$-test $T(X, Y)$, ``ImprovementGate.admit(X, Y, direction='decrease', alpha, min_effect)`` returns ``True`` iff

$$T(X, Y) < t_\alpha \;\wedge\; |\mathbb{E}[Y] - \mathbb{E}[X]| / |\mathbb{E}[X]| \ge \mathrm{min\_effect}$$

and the direction-of-effect predicate holds.

**Why it matters.** The statistical admission criterion is the current default backend behind the gate; the Lean proof formalises that the Python implementation matches the textbook one-sided Welch test up to a (provable) threshold-comparison wrapper.