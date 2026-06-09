---
code_refs: [neuroslm/thsd/phi.py, neuroslm/verification/triple_guard.py]
created_at: "2026-06-09T18:00:16Z"
id: H001
proof_path: hypothesis/proofs/H001_phi_monotone_under_coupling_addition.lean
proof_status: stub
references: [formal_framework.md §10.2, architecture.md §5.5]
status: stated
tags: [phi, monotonicity, structural, iit]
test_refs: [tests/thsd/test_phi.py, tests/training/test_rcc_bowtie_triple_guard.py]
theorem_name: Brian.PhiMonotone
title: Phi monotone under coupling addition
updated_at: "2026-06-09T18:05:55Z"
---

**Statement.** Let $\theta' = \theta \oplus \alpha$ be a mutation that adds a non-negative coupling $\alpha \ge 0$ to the architecture's sheaf Laplacian $L = \delta^{0\top}\delta^0$. Then

$$\Phi(\theta') \;\ge\; \Phi(\theta)$$

where $\Phi(\theta)$ is the IIT 4.0 integrated-information proxy (``neuroslm/thsd/phi.py``).

**Why it matters.** A mutation that strictly *adds* a non-negative projection cannot reduce $\Phi$. The evolutionary loop relies on this to admit additive mutations cheaply (no full $\Phi$ recomputation required when the structural delta is purely additive).