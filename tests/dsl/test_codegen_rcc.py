# -*- coding: utf-8 -*-
"""End-to-end demo: codegen on the full RCC bowtie architecture.

After the Stage-6 migration, `architectures/rcc_bowtie/` is the canonical
source for this architecture. This test exercises the full pipeline:

    folder  →  compile_folder  →  ProgramIR  →  CodeGenerator  →  nn.Module

and confirms that the resulting module is well-formed, runs forward on
a synthetic input, and responds to neurotransmitter modulation.

The equation-vs-macro parity guarantee is covered by `test_codegen.py`
(parameterised over every algebraic dynamics) and the back-to-back
folder/legacy comparison once lived in `test_rcc_bowtie_migration.py`.
"""
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from neuroslm.dsl.codegen import CodeGenerator
from neuroslm.dsl.multifile import compile_folder


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARCH_ROOT = REPO_ROOT / "architectures" / "rcc_bowtie"


def _compile_arch(module_name: str = "RCCBowtie"):
    ir = compile_folder(ARCH_ROOT)
    Cls = CodeGenerator(ir, module_name=module_name).compile_to_module()
    return ir, Cls


# ── Compile-and-run rcc_bowtie ─────────────────────────────────────────

class TestRCCBowtieCompiles:
    def test_codegen_emits_valid_python(self):
        ir = compile_folder(ARCH_ROOT)
        # generate() runs ast.parse internally — raises on syntax errors
        CodeGenerator(ir, module_name="RCCBowtie").generate()

    def test_compile_to_module(self):
        _, Cls = _compile_arch()
        assert issubclass(Cls, nn.Module)

    def test_forward_returns_all_populations(self):
        ir, Cls = _compile_arch()
        circuit = Cls(d_sem=128)

        x = torch.randn(2, 128)
        out = circuit(x)
        expected_names = {p.name for p in ir.populations}
        assert set(out.keys()) == expected_names
        for name, t in out.items():
            assert t.shape == (2, 128), f"{name}: unexpected shape {t.shape}"
            assert not torch.isnan(t).any(), f"{name}: NaN in output"

    def test_nt_modulation_applied(self):
        _, Cls = _compile_arch()
        circuit = Cls(d_sem=128)

        x = torch.abs(torch.randn(2, 128))
        out_a = circuit(x)
        out_b = circuit(x, nt_levels={"dopamine": 1.5, "serotonin": 0.8})

        any_diff = any(
            not torch.allclose(out_a[p], out_b[p])
            for p in out_a
        )
        assert any_diff, "no population responded to NT level change"

    def test_explicit_equations_throughout(self):
        """Sanity check that the folder is genuinely math-first: every
        non-passthrough population carries an explicit `equation:` or
        `ode:` field, not just an enum dynamics."""
        ir = compile_folder(ARCH_ROOT)
        missing = [
            p.name for p in ir.populations
            if not (p.equation or p.ode)
        ]
        assert not missing, (
            f"populations without explicit equation/ode: {missing}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
