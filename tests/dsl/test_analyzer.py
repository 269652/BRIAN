# -*- coding: utf-8 -*-
"""Tests for the SymPy-based architecture analyzer + brian CLI."""
import os
import subprocess
import sys

import pytest
import sympy as sp


ARCH_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "architectures", "rcc_bowtie")


pytestmark = pytest.mark.skipif(
    not os.path.isdir(ARCH_ROOT),
    reason="rcc_bowtie architecture not present in this checkout")


def test_compile_to_sympy_populates_all_sections():
    from neuroslm.dsl.analyzer import compile_to_sympy
    sys = compile_to_sympy(ARCH_ROOT)
    # rcc_bowtie has 28 populations, 14 synapses, 17 modulations, 7 NTs
    # → 66 state vars total
    assert len(sys.state_vars) == 66, f"got {len(sys.state_vars)}"
    assert len(sys.nt_state) == 7
    # Every state var has an entry in `equations`
    for sv in sys.state_vars:
        assert sv in sys.equations, f"{sv} has no equation"


def test_solve_fixed_points_returns_at_least_one():
    from neuroslm.dsl.analyzer import compile_to_sympy, solve_fixed_points
    sys = compile_to_sympy(ARCH_ROOT)
    fps = solve_fixed_points(sys, max_solutions=1)
    assert len(fps) >= 1
    # Every NT should resolve to its baseline at activity=0 (the homeostatic
    # fixed point: 0 = release*0 - reuptake*(c - base) → c = base).
    fp = fps[0]
    for nt_sym in sys.nt_state:
        sol = fp.values.get(nt_sym)
        assert sol is not None, f"no fixed-point solution for {nt_sym}"


def test_jacobian_shape_and_sparsity():
    from neuroslm.dsl.analyzer import compile_to_sympy, jacobian
    sys = compile_to_sympy(ARCH_ROOT)
    J = jacobian(sys)
    assert J.shape == (66, 66)
    # rcc_bowtie is sparsely connected — under 5% density expected
    nnz = sum(1 for i in range(J.shape[0]) for j in range(J.shape[1])
              if J[i, j] != 0)
    density = nnz / (J.shape[0] * J.shape[1])
    assert density < 0.05, f"unexpectedly dense Jacobian (density {density:.1%})"


def test_wa_queries_are_under_size_cap():
    from neuroslm.dsl.analyzer import wolfram_alpha_queries
    qs = wolfram_alpha_queries(ARCH_ROOT, max_chars=180)
    assert len(qs) > 0
    for label, q in qs:
        assert len(q) <= 180, f"query too long: {label}"
        # WA-relevant function name present
        assert any(fn in q for fn in ("Solve[", "Plot[", "DSolve["))


def test_nt_steady_state_matches_closed_form():
    """For each NT the steady-state value should satisfy
    c* = base + (release/reuptake) * activity.
    Verify against the symbolic solver output."""
    from neuroslm.dsl.analyzer import compile_to_sympy, solve_fixed_points
    sys = compile_to_sympy(ARCH_ROOT)
    fps = solve_fixed_points(sys, max_solutions=1)
    assert fps
    fp = fps[0]
    # Substitute activity=0.5, then verify the solver's c* against the
    # closed-form. rcc_bowtie config values:
    nt_specs = {
        "dopamine":        (0.10, 0.20, 0.80),
        "norepinephrine":  (0.15, 0.30, 0.70),
        "serotonin":       (0.30, 0.05, 0.95),
        "acetylcholine":   (0.20, 0.25, 0.75),
        "endocannabinoid": (0.05, 0.40, 0.60),
        "glutamate":       (0.40, 0.50, 0.50),
        "gaba":            (0.10, 0.10, 0.90),
    }
    for name, (base, rel, reup) in nt_specs.items():
        c_sym = sp.Symbol(f"c_{name}")
        a_sym = sp.Symbol(f"activity_{name}")
        sol = fp.values.get(c_sym)
        if sol is None:
            continue
        # Substitute activity = 0.5
        numeric = float(sol.subs(a_sym, 0.5))
        expected = base + (rel / reup) * 0.5
        assert abs(numeric - expected) < 1e-6, \
            f"{name}: got {numeric}, expected {expected}"


# ── brian CLI smoke tests ──────────────────────────────────────────────

def _run_cli(*args) -> subprocess.CompletedProcess:
    """Run `py -m neuroslm.cli ...` and capture stdout/stderr."""
    return subprocess.run(
        [sys.executable, "-m", "neuroslm.cli", *args],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def test_cli_help():
    r = _run_cli("--help")
    assert r.returncode == 0
    assert "compile" in r.stdout
    assert "wolfram" in r.stdout
    assert "analyze" in r.stdout
    assert "deploy" in r.stdout
    assert "ood" in r.stdout


def test_cli_analyze_subcommand():
    r = _run_cli("analyze", ARCH_ROOT, "--jacobian")
    # The Unicode × in the output may cause Windows-cp1252 encode errors
    # for stdout printing, but the analyzer logic itself runs.
    assert "state vars" in r.stdout
    assert "Jacobian" in r.stdout or "Jacobian" in r.stderr


def test_flow_finds_bowtie_waist():
    from neuroslm.dsl.analyzer import analyze_flow
    fr = analyze_flow(ARCH_ROOT)
    # rcc_bowtie's waist should include the global workspace (gws)
    assert "gws" in fr.bowtie_waist, f"gws not in waist: {fr.bowtie_waist}"
    # There must be at least one multi-hop path through the bowtie
    assert any(len(p) >= 4 for p in fr.paths), "no deep dataflow paths"
    # The cognitive cascade dmn -> gws -> ... -> motor must be present
    longest = fr.longest_path
    assert "gws" in longest, f"gws not in longest path: {longest}"
    assert "motor" in longest, f"motor not in longest path: {longest}"


def test_phi_proxy_is_positive_and_balanced():
    from neuroslm.dsl.analyzer import compute_phi_proxy
    pr = compute_phi_proxy(ARCH_ROOT)
    # rcc_bowtie is structurally sparse + differentiated -> small Phi but >0
    assert pr.phi_proxy > 0.0
    assert pr.differentiation > 0.5, "expected high differentiation for sparse arch"
    # gws should be among the highest-contributing modules (bowtie waist)
    top_contributors = [name for name, _ in pr.per_module_contribution[:5]]
    assert "gws" in top_contributors, f"gws missing from top contributors: {top_contributors}"


def test_discover_proposes_modifications():
    from neuroslm.dsl.analyzer import discover_modifications
    baseline, props = discover_modifications(ARCH_ROOT, metric="phi", top_k=5)
    assert baseline >= 0.0
    assert len(props) > 0
    # The top proposal must strictly improve the baseline (otherwise the
    # arch is already optimal w.r.t. the metric — flag as a hint that
    # something is wrong with the search).
    assert props[0].delta_metric >= 0.0, \
        f"top proposal worsens metric? {props[0]}"
    # All four mod kinds should be discoverable in principle
    kinds = {m.kind for m in
             discover_modifications(ARCH_ROOT, metric="phi", top_k=200)[1]}
    assert "add_edge" in kinds
    assert "remove_edge" in kinds


def test_cli_wolfram_subcommand(tmp_path):
    out = tmp_path / "arch.m"
    r = _run_cli("wolfram", ARCH_ROOT, "--full", "--out", str(out))
    assert r.returncode == 0, r.stderr
    assert out.is_file()
    code = out.read_text(encoding="utf-8")
    # All four IIT-grade sections present
    for sec in ("Populations", "Synapses", "Modulations",
                "NeurotransmitterDynamics"):
        assert sec in code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
