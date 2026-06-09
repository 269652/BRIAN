---
code_refs: [neuroslm/verification/triple_guard.py]
created_at: "2026-06-09T18:00:16Z"
id: H004
proof_path: hypothesis/proofs/H004_triple_guard_soundness.lean
proof_status: verified
references: [formal_framework.md §6.4, formal_framework.md §10.2]
status: stated
tags: [triple-guard, soundness, structural]
test_refs: [tests/training/test_rcc_bowtie_triple_guard.py]
theorem_name: Brian.TripleGuardSound
title: Triple-Guard soundness
updated_at: "2026-06-09T19:45:23Z"
---

**Statement.** Let $g_\Phi$, $g_{H^1}$, $g_\lambda$ be the three sub-guards ($\Phi$ minimum, $H^1$ maximum, Fiedler $\lambda_1$ minimum). Then for every mutation $m$:

$$\mathrm{TripleGuard.admit}(m) \;\iff\; g_\Phi(m) \wedge g_{H^1}(m) \wedge g_\lambda(m)$$

(soundness — the gate returns ``admitted = True`` iff every sub-guard passes).

**Why it matters.** Soundness is the audit guarantee for the structural admission boundary. If this proof goes through, every verifier downstream can trust that a ``Verdict(admitted=True)`` means all three invariants survived the mutation.