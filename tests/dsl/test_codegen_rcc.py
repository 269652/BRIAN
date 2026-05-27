# -*- coding: utf-8 -*-
"""End-to-end demo: codegen on the full RCC bowtie architecture.

This is the "whole arch stays semantically identical" guarantee in
action. We:

  1. Compile rcc_bowtie.neuro through the new codegen and run forward.
  2. Programmatically rewrite five populations to use explicit `equation:`
     fields instead of the `dynamics:` enum.
  3. Assert the rewritten circuit produces byte-equal outputs to the
     enum-only version, after syncing random-initialized buffers.

If the codegen ever drifts from semantic equivalence between the enum
path and the explicit-equation path, this test will catch it.
"""
import re
import pytest
import torch
import torch.nn as nn

from neuroslm.dsl.codegen import CodeGenerator
from neuroslm.dsl.compiler import NeuroMLCompiler
from neuroslm.dsl.equations import DYNAMICS_EQUATIONS


# The five populations we rewrite. Chosen to exercise every algebraic
# dynamics in the macro table: rate_code, gated, winner_take_all,
# attractor_network, attention_pool. (`static` is skipped — `neural_geometry`
# uses it but it's a trivial identity, less interesting to demo.)
POPS_TO_REWRITE = {
    "sensory":              "rate_code",
    "thalamus":             "gated",
    "gws":                  "winner_take_all",
    "hippo":                "attractor_network",
    "thought_transformer":  "attention_pool",
}


# ── Helpers ────────────────────────────────────────────────────────────

def _load_rcc_source() -> str:
    """Read the rcc_bowtie .neuro source from its repo location."""
    from pathlib import Path
    here = Path(__file__).resolve()
    # tests/dsl/this_file → repo root → neuroslm/dsl/rcc_bowtie.neuro
    rcc_path = here.parent.parent.parent / "neuroslm" / "dsl" / "rcc_bowtie.neuro"
    return rcc_path.read_text(encoding="utf-8")


def _inject_equations(source: str, pops: dict) -> str:
    """Append `equation: "..."` to each named population's block.

    `pops` is `{population_name: dynamics_name}`. The canonical equation
    is pulled from DYNAMICS_EQUATIONS — same string the enum-path codegen
    would resolve to internally, so byte-equal source emerges from both
    paths after compilation.
    """
    out = source
    for pop_name, dyn in pops.items():
        eq = DYNAMICS_EQUATIONS.get(dyn)
        assert eq is not None, f"no canonical equation for {dyn!r}"

        # Match `population <name> {  ... }` then append `equation: "..."`
        # to its body just before the closing brace.
        pattern = re.compile(
            rf'(population\s+{re.escape(pop_name)}\s*\{{)([^}}]*)(\}})',
            re.DOTALL,
        )

        def _add_eq(m, _eq=eq):
            head, body, tail = m.group(1), m.group(2), m.group(3)
            body = body.rstrip().rstrip(",")
            body = body + f',\n    equation: "{_eq}"\n'
            return head + body + tail

        new_out, n = pattern.subn(_add_eq, out, count=1)
        assert n == 1, f"could not find population block {pop_name!r}"
        out = new_out
    return out


def _sync(target: nn.Module, source: nn.Module) -> None:
    """Copy every shared (name, shape) tensor from source to target."""
    src = dict(source.state_dict())
    tgt = dict(target.state_dict())
    for name, t in tgt.items():
        if name in src and src[name].shape == t.shape:
            t.copy_(src[name])


# ── Compile-and-run rcc_bowtie ─────────────────────────────────────────

class TestRCCBowtieCompiles:
    def test_codegen_emits_valid_python(self):
        source = _load_rcc_source()
        ir = NeuroMLCompiler.compile(source)
        # generate() runs ast.parse internally — raises on syntax errors
        CodeGenerator(ir, module_name="RCCBowtie").generate()

    def test_compile_to_module(self):
        source = _load_rcc_source()
        ir = NeuroMLCompiler.compile(source)
        Cls = CodeGenerator(ir, module_name="RCCBowtie").compile_to_module()
        assert issubclass(Cls, nn.Module)

    def test_forward_returns_all_populations(self):
        source = _load_rcc_source()
        ir = NeuroMLCompiler.compile(source)
        Cls = CodeGenerator(ir, module_name="RCCBowtie").compile_to_module()
        circuit = Cls(d_sem=128)

        x = torch.randn(2, 128)
        out = circuit(x)
        expected_names = {p.name for p in ir.populations}
        assert set(out.keys()) == expected_names
        for name, t in out.items():
            assert t.shape == (2, 128), f"{name}: unexpected shape {t.shape}"
            assert not torch.isnan(t).any(), f"{name}: NaN in output"

    def test_nt_modulation_applied(self):
        source = _load_rcc_source()
        ir = NeuroMLCompiler.compile(source)
        Cls = CodeGenerator(ir, module_name="RCCBowtie").compile_to_module()
        circuit = Cls(d_sem=128)

        x = torch.abs(torch.randn(2, 128))  # nonneg input drives clearer differences
        out_a = circuit(x)
        out_b = circuit(x, nt_levels={"dopamine": 1.5, "serotonin": 0.8})

        # At least one population must respond to NT levels — they're
        # multiplicative gains so for a nonneg dopamine target the
        # output should differ.
        any_diff = any(
            not torch.allclose(out_a[p], out_b[p])
            for p in out_a
        )
        assert any_diff, "no population responded to NT level change"


# ── Whole-arch semantic-equivalence demo ──────────────────────────────

class TestRCCBowtieEquationParity:
    """The headline test: rewrite five populations, prove identical output.

    Identical = `torch.allclose` after `_sync` copies the randomly-init'd
    synapse weights and parameters from one module to the other.
    """

    def test_macro_vs_equation_identical_outputs(self):
        # 1) Macro-only source
        source_macro = _load_rcc_source()
        ir_macro = NeuroMLCompiler.compile(source_macro)
        Cls_macro = CodeGenerator(ir_macro, module_name="RCCMacro").compile_to_module()
        circuit_macro = Cls_macro(d_sem=128)

        # 2) Source with explicit equations injected for five populations
        source_eq = _inject_equations(source_macro, POPS_TO_REWRITE)
        ir_eq = NeuroMLCompiler.compile(source_eq)
        Cls_eq = CodeGenerator(ir_eq, module_name="RCCEq").compile_to_module()
        circuit_eq = Cls_eq(d_sem=128)

        # 3) Sync random params/buffers so the comparison is fair
        _sync(circuit_eq, circuit_macro)

        # 4) Run identical forward passes
        torch.manual_seed(0)
        x = torch.randn(2, 128)
        out_m = circuit_macro(x)
        out_e = circuit_eq(x)

        # 5) Every rewritten population must produce identical output
        for pop_name in POPS_TO_REWRITE:
            diff = (out_m[pop_name] - out_e[pop_name]).abs().max().item()
            assert diff < 1e-6, (
                f"{pop_name}: macro vs equation diverged "
                f"(max abs diff {diff})"
            )

    def test_injection_actually_changes_source(self):
        """Sanity check that the rewrite is doing something visible."""
        source = _load_rcc_source()
        rewritten = _inject_equations(source, {"sensory": "rate_code"})
        assert 'equation: "y = ReLU(x)"' in rewritten
        # And the IR sees the field
        ir = NeuroMLCompiler.compile(rewritten)
        sensory = next(p for p in ir.populations if p.name == "sensory")
        assert sensory.equation == "y = ReLU(x)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
