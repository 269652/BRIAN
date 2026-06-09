---
code_refs: [neuroslm/thsd/symbolic.py]
created_at: "2026-06-09T18:00:16Z"
id: H003
proof_path: hypothesis/proofs/H003_symbolic_sparsity_collapse.lean
proof_status: verified
references: [formal_framework.md §10.2, formal_framework.md §3]
status: stated
tags: [thsd, symbolic, gumbel-softmax, sparsity]
test_refs: [tests/thsd/test_symbolic.py]
theorem_name: Brian.SymbolicSparsity
title: Symbolic sparsity collapse
updated_at: "2026-06-09T19:45:22Z"
---

**Statement.** Let $U_\tau(x)$ be the Gumbel-Softmax output of a ``SymbolicHyperNeuron`` at temperature $\tau > 0$. As $\tau \to 0^+$,

$$\|U_\tau(x)\|_0 \;\longrightarrow\; 1 \quad \text{a.s.}$$

i.e. every symbolic unit collapses to a single active operator (a discrete expression).

**Why it matters.** Justifies the Gumbel-Softmax annealing schedule used in the THSD discovery operator. Without this guarantee the symbolic substrate could stay diffuse and never produce a clean discrete program.