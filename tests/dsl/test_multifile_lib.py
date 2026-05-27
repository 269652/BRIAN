# -*- coding: utf-8 -*-
"""Stage 4 — user-defined `dynamics` and `function` blocks from lib files.

Lib files contribute reusable mechanics that any module can import:

    # lib/dynamics.neuro
    export dynamics lif_neuron {
        ode: "tau * dV/dt = -(V - V_rest) + R * x",
        state: { V: "torch.zeros(1, d_sem)" },
        constants: { tau: 0.05, V_rest: 0.0, R: 1.0, dt: 0.01 }
    }

    export function decay(x, alpha) {
        equation: "(1 - alpha) * x"
    }

When a module imports such a dynamics, it can then say
    population pyramidal { count: 256, dynamics: "lif_neuron" }
and the codegen treats `lif_neuron` like a built-in macro.

This stage delivers:
  * parse_dynamics_block(body)   → DynamicsDecl   (from equations.py)
  * parse_function_block(body)   → FunctionDecl
  * Resolver.resolve() now populates program.user_dynamics + user_functions
  * program.lookup_dynamics(file, name) — searches imports first, then
    user-defined locals, then nothing (built-in lookup is codegen's job).
"""
import pytest
from pathlib import Path

from neuroslm.dsl.multifile import (
    Resolver,
    ResolverError,
    parse_dynamics_block,
    parse_function_block,
    FunctionDecl,
)
from neuroslm.dsl.equations import DynamicsDecl


def _write(root: Path, rel: str, content: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# ── parse_dynamics_block ───────────────────────────────────────────────

class TestParseDynamicsBlock:
    def test_algebraic_dynamics(self):
        body = 'equation: "y = ReLU(W @ x + b)"'
        decl = parse_dynamics_block(body)
        assert decl.equation == "y = ReLU(W @ x + b)"
        assert decl.ode is None

    def test_ode_dynamics(self):
        body = '''
            ode: "tau * dV/dt = -(V - V_rest) + R * x",
            state: { V: "torch.zeros(1, d_sem)" },
            constants: { tau: 0.05, V_rest: 0.0, R: 1.0, dt: 0.01 }
        '''
        decl = parse_dynamics_block(body)
        assert decl.ode == "tau * dV/dt = -(V - V_rest) + R * x"
        assert decl.equation is None
        assert decl.state == {"V": "torch.zeros(1, d_sem)"}
        assert decl.constants["tau"] == 0.05
        assert decl.constants["V_rest"] == 0.0

    def test_dynamics_with_params(self):
        body = '''
            equation: "y = ReLU(x) * sigmoid(gate)",
            params: { gate: "torch.zeros(1)" }
        '''
        decl = parse_dynamics_block(body)
        assert decl.params == {"gate": "torch.zeros(1)"}


# ── parse_function_block ───────────────────────────────────────────────

class TestParseFunctionBlock:
    def test_function_with_args_and_equation(self):
        fn = parse_function_block("decay", "(x, alpha)",
                                  'equation: "(1 - alpha) * x"')
        assert isinstance(fn, FunctionDecl)
        assert fn.name == "decay"
        assert fn.args == ["x", "alpha"]
        assert fn.equation == "(1 - alpha) * x"

    def test_function_single_arg(self):
        fn = parse_function_block("squared", "(x)", 'equation: "x * x"')
        assert fn.args == ["x"]


# ── Resolver collects user_dynamics + user_functions ───────────────────

class TestResolverLibCollection:
    def test_collects_dynamics_from_lib(self, tmp_path):
        _write(tmp_path, "arch.neuro",
               'architecture x { d_sem: 256 }\n'
               'import { lif } from "@/lib/dyn"')
        _write(tmp_path, "lib/dyn.neuro",
               'export dynamics lif {\n'
               '  ode: "tau * dV/dt = -V + x",\n'
               '  state: { V: "torch.zeros(1, d_sem)" },\n'
               '  constants: { tau: 0.05, dt: 0.01 }\n'
               '}')

        program = Resolver(tmp_path).resolve()
        lib_path = (tmp_path / "lib" / "dyn.neuro").resolve()

        # user_dynamics is keyed by (file, name) → DynamicsDecl
        assert (lib_path, "lif") in program.user_dynamics
        decl = program.user_dynamics[(lib_path, "lif")]
        assert decl.ode == "tau * dV/dt = -V + x"
        assert decl.constants["tau"] == 0.05

    def test_collects_functions_from_lib(self, tmp_path):
        _write(tmp_path, "arch.neuro",
               'architecture x { d_sem: 256 }')
        _write(tmp_path, "lib/util.neuro",
               'export function decay(x, alpha) {\n'
               '  equation: "(1 - alpha) * x"\n'
               '}')

        program = Resolver(tmp_path).resolve()
        lib_path = (tmp_path / "lib" / "util.neuro").resolve()
        assert (lib_path, "decay") in program.user_functions
        fn = program.user_functions[(lib_path, "decay")]
        assert fn.args == ["x", "alpha"]


# ── lookup_dynamics traverses imports ─────────────────────────────────

class TestDynamicsLookup:
    def test_lookup_finds_imported_dynamics(self, tmp_path):
        _write(tmp_path, "arch.neuro",
               'architecture x { d_sem: 256 }')
        # Module imports a dynamics from lib
        idx = _write(tmp_path, "modules/pfc.neuro",
                     'import { lif } from "@/lib/dyn"\n'
                     'export population p { count: 256, dynamics: "lif" }')
        _write(tmp_path, "lib/dyn.neuro",
               'export dynamics lif {\n'
               '  ode: "tau * dV/dt = -V + x",\n'
               '  state: { V: "torch.zeros(1, d_sem)" },\n'
               '  constants: { tau: 0.05, dt: 0.01 }\n'
               '}')

        program = Resolver(tmp_path).resolve()
        idx_path = idx.resolve()
        decl = program.lookup_dynamics(idx_path, "lif")
        assert decl is not None
        assert decl.ode == "tau * dV/dt = -V + x"

    def test_lookup_returns_none_for_unknown(self, tmp_path):
        _write(tmp_path, "arch.neuro",
               'architecture x { d_sem: 256 }')
        idx = _write(tmp_path, "modules/pfc.neuro",
                     'export population p { count: 256 }')

        program = Resolver(tmp_path).resolve()
        decl = program.lookup_dynamics(idx.resolve(), "does_not_exist")
        assert decl is None

    def test_lookup_aliased_dynamics(self, tmp_path):
        # `import { lif as my_neuron }` — looking up by alias works
        _write(tmp_path, "arch.neuro",
               'architecture x { d_sem: 256 }')
        idx = _write(tmp_path, "modules/pfc.neuro",
                     'import { lif as my_neuron } from "@/lib/dyn"\n'
                     'export population p { count: 256, dynamics: "my_neuron" }')
        _write(tmp_path, "lib/dyn.neuro",
               'export dynamics lif {\n'
               '  ode: "tau * dV/dt = -V + x",\n'
               '  state: { V: "torch.zeros(1, d_sem)" },\n'
               '  constants: { tau: 0.05, dt: 0.01 }\n'
               '}')

        program = Resolver(tmp_path).resolve()
        decl = program.lookup_dynamics(idx.resolve(), "my_neuron")
        assert decl is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
