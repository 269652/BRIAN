# -*- coding: utf-8 -*-
"""Seed ``hypothesis/`` with the 5 canonical Lean obligations named
in ``docs/formal_framework.md`` §10.2.

Idempotent: re-running overwrites any existing ``H001..H005`` records
with the canonical content. New hypotheses (``H006+``) are untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from neuroslm.discoveries import (
    HypothesisRecord, HypothesisStore, emit_hypothesis_proof,
)


def main() -> int:
    store = HypothesisStore(REPO / "hypothesis")
    seeds = _build_seeds()
    for h in seeds:
        store.save(h)
        emit_hypothesis_proof(h, REPO / "hypothesis")
        store.save(h)        # persist proof_status="stub"
    print(f"Seeded {len(seeds)} canonical hypotheses → hypothesis/")
    for r in store.list_all():
        print(f"  {r.id}  {r.status:8s} {r.proof_status:8s}  {r.title}")
    return 0


def _build_seeds() -> list:
    return [
        HypothesisRecord(
            id="H001",
            title="Phi monotone under coupling addition",
            statement_md=(
                "**Statement.** Let $\\theta' = \\theta \\oplus \\alpha$ be a "
                "mutation that adds a non-negative coupling $\\alpha \\ge 0$ "
                "to the architecture's sheaf Laplacian "
                "$L = \\delta^{0\\top}\\delta^0$. Then\n"
                "\n"
                "$$\\Phi(\\theta') \\;\\ge\\; \\Phi(\\theta)$$\n"
                "\n"
                "where $\\Phi(\\theta)$ is the IIT 4.0 integrated-information "
                "proxy (``neuroslm/thsd/phi.py``).\n"
                "\n"
                "**Why it matters.** A mutation that strictly *adds* a "
                "non-negative projection cannot reduce $\\Phi$. The "
                "evolutionary loop relies on this to admit additive mutations "
                "cheaply (no full $\\Phi$ recomputation required when the "
                "structural delta is purely additive)."
            ),
            theorem_name="Brian.PhiMonotone",
            status="stated",
            references=["formal_framework.md §10.2", "architecture.md §5.5"],
            code_refs=["neuroslm/thsd/phi.py",
                       "neuroslm/verification/triple_guard.py"],
            test_refs=["tests/thsd/test_phi.py",
                       "tests/training/test_rcc_bowtie_triple_guard.py"],
            tags=["phi", "monotonicity", "structural", "iit"],
        ),

        HypothesisRecord(
            id="H002",
            title="OOD gap decrease under CDGA",
            statement_md=(
                "**Statement.** Let $L_{\\text{base}}$ be the base training "
                "loss and $\\mathrm{CDGA}$ the Cross-Distribution Gradient "
                "Alignment term (``docs/CDGA.md``, $\\lambda \\ge 0$). Then\n"
                "\n"
                "$$\\Delta_{\\mathrm{OOD}}(\\theta + \\lambda\\cdot"
                "\\mathrm{CDGA}) \\;\\le\\; \\Delta_{\\mathrm{OOD}}(\\theta)$$\n"
                "\n"
                "where $\\Delta_{\\mathrm{OOD}} = L_{\\text{OOD}} - "
                "L_{\\text{ID}}$ is the OOD generalisation gap.\n"
                "\n"
                "**Why it matters.** Adds a free generalisation guarantee: "
                "turning on the CDGA term never widens the OOD gap, even when "
                "it doesn't help. The empirical version is exercised in "
                "``test_cdga_smoke.py``."
            ),
            theorem_name="Brian.OodGapDecrease",
            status="stated",
            references=["formal_framework.md §10.2", "docs/CDGA.md"],
            code_refs=["neuroslm/regularizers.py"],
            test_refs=["tests/test_cdga_smoke.py"],
            tags=["ood", "cdga", "monotonicity", "regularisation"],
        ),

        HypothesisRecord(
            id="H003",
            title="Symbolic sparsity collapse",
            statement_md=(
                "**Statement.** Let $U_\\tau(x)$ be the Gumbel-Softmax output "
                "of a ``SymbolicHyperNeuron`` at temperature $\\tau > 0$. As "
                "$\\tau \\to 0^+$,\n"
                "\n"
                "$$\\|U_\\tau(x)\\|_0 \\;\\longrightarrow\\; 1 \\quad "
                "\\text{a.s.}$$\n"
                "\n"
                "i.e. every symbolic unit collapses to a single active "
                "operator (a discrete expression).\n"
                "\n"
                "**Why it matters.** Justifies the Gumbel-Softmax annealing "
                "schedule used in the THSD discovery operator. Without this "
                "guarantee the symbolic substrate could stay diffuse and "
                "never produce a clean discrete program."
            ),
            theorem_name="Brian.SymbolicSparsity",
            status="stated",
            references=["formal_framework.md §10.2", "formal_framework.md §3"],
            code_refs=["neuroslm/thsd/symbolic.py"],
            test_refs=["tests/thsd/test_symbolic.py"],
            tags=["thsd", "symbolic", "gumbel-softmax", "sparsity"],
        ),

        HypothesisRecord(
            id="H004",
            title="Triple-Guard soundness",
            statement_md=(
                "**Statement.** Let $g_\\Phi$, $g_{H^1}$, $g_\\lambda$ be the "
                "three sub-guards ($\\Phi$ minimum, $H^1$ maximum, Fiedler "
                "$\\lambda_1$ minimum). Then for every mutation $m$:\n"
                "\n"
                "$$\\mathrm{TripleGuard.admit}(m) \\;\\iff\\; "
                "g_\\Phi(m) \\wedge g_{H^1}(m) \\wedge g_\\lambda(m)$$\n"
                "\n"
                "(soundness — the gate returns ``admitted = True`` iff every "
                "sub-guard passes).\n"
                "\n"
                "**Why it matters.** Soundness is the audit guarantee for the "
                "structural admission boundary. If this proof goes through, "
                "every verifier downstream can trust that a "
                "``Verdict(admitted=True)`` means all three invariants "
                "survived the mutation."
            ),
            theorem_name="Brian.TripleGuardSound",
            status="stated",
            references=["formal_framework.md §6.4", "formal_framework.md §10.2"],
            code_refs=["neuroslm/verification/triple_guard.py"],
            test_refs=["tests/training/test_rcc_bowtie_triple_guard.py"],
            tags=["triple-guard", "soundness", "structural"],
        ),

        HypothesisRecord(
            id="H005",
            title="ImprovementGate Welch correctness",
            statement_md=(
                "**Statement.** Given samples $X \\sim "
                "\\mathcal{D}_{\\text{before}}$, "
                "$Y \\sim \\mathcal{D}_{\\text{after}}$ and a one-sided "
                "Welch's $t$-test $T(X, Y)$, ``ImprovementGate.admit(X, Y, "
                "direction='decrease', alpha, min_effect)`` returns ``True`` "
                "iff\n"
                "\n"
                "$$T(X, Y) < t_\\alpha \\;\\wedge\\; |\\mathbb{E}[Y] - "
                "\\mathbb{E}[X]| / |\\mathbb{E}[X]| \\ge "
                "\\mathrm{min\\_effect}$$\n"
                "\n"
                "and the direction-of-effect predicate holds.\n"
                "\n"
                "**Why it matters.** The statistical admission criterion is "
                "the current default backend behind the gate; the Lean proof "
                "formalises that the Python implementation matches the "
                "textbook one-sided Welch test up to a (provable) threshold-"
                "comparison wrapper."
            ),
            theorem_name="Brian.ImprovementGateWelch",
            # Empirically discharged by tests/verification/test_improvement_gate.py
            # but the Lean obligation is still ``stated`` until the .lean file
            # has no ``sorry`` left — verifier promotes proof_status, not status.
            status="stated",
            references=["formal_framework.md §9", "formal_framework.md §10.2"],
            code_refs=["neuroslm/verification/improvement_gate.py"],
            test_refs=["tests/verification/test_improvement_gate.py"],
            tags=["improvement-gate", "welch", "statistical", "admission"],
        ),
    ]


if __name__ == "__main__":
    raise SystemExit(main())
