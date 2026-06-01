# -*- coding: utf-8 -*-
"""DSL → PyTorch code generator.

Takes a compiled `ProgramIR` and emits an `nn.Module` subclass whose
forward pass runs the circuit. Each population becomes its own inner
class; their algebraic equations are lowered to torch ops via
`equations.lower_to_torch`. Synapses are linear projections between
populations; neurotransmitter modulations are post-hoc multiplicative
gains.

Semantic contract (Phase 7 Stage 1):
    - Population specified via `dynamics: "rate_code"` (enum macro) must
      produce identical output to one specified via `equation: "y = ReLU(x)"`.
    - Both must match a hand-written reference implementation up to float
      tolerance (see tests/dsl/test_codegen.py).
    - Synapse and modulation behavior is preserved from the legacy
      template-based codegen.

Cycle handling: populations are evaluated in declaration order. A synapse
whose source is later in declaration order than its target reads the
source's *previous-step* output from a state buffer (`self.last_{name}`),
giving deterministic one-step-delayed feedback for re-entry loops.
"""
from __future__ import annotations
import ast
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .compiler import (
    ProgramIR,
    PopulationIR,
    SynapseIR,
    ModulationIR,
)
from .equations import (
    DYNAMICS_DECLS,
    DynamicsDecl,
    equation_for_population,
    find_decl_for_equation,
    find_decl_for_ode,
    lower_to_torch,
    parse_equation,
    parse_ode,
)


# Symbols that are *always* available in the population forward() scope
# and therefore should not be treated as learnable parameters.
_RESERVED_SYMBOLS = {"x", "y", "d_sem"}


class CodeGenerator:
    """Compile a `ProgramIR` to executable nn.Module Python source."""

    def __init__(self, ir: ProgramIR, module_name: str = "GeneratedCircuit"):
        self.ir = ir
        self.module_name = self._sanitize(module_name)

    # ── Public API ──────────────────────────────────────────────────

    def generate(self) -> str:
        """Return the full Python source for the generated module."""
        parts = [self._gen_header(), self._gen_imports()]
        for pop in self.ir.populations:
            parts.append(self._gen_population_class(pop))
        parts.append(self._gen_circuit_class())
        src = "\n\n".join(parts) + "\n"
        # Sanity-check syntax before handing it to the caller.
        ast.parse(src)
        return src

    def compile_to_module(self) -> type:
        """Execute the generated source and return the circuit class."""
        src = self.generate()
        ns: Dict = {"torch": torch, "nn": nn, "F": F}
        exec(compile(src, f"<generated:{self.module_name}>", "exec"), ns)
        cls = ns.get(self.module_name)
        if cls is None:
            raise RuntimeError(f"generated source did not define {self.module_name}")
        return cls

    def save_to_file(self, path: str) -> None:
        """Write the generated source to a file (useful for debugging)."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.generate())

    @staticmethod
    def from_architecture(arch_name: str) -> "CodeGenerator":
        """Convenience: load a registered architecture and build a generator."""
        from neuroslm.architectures.loader import load_architecture_spec
        spec = load_architecture_spec(arch_name)
        return CodeGenerator(spec["ir"], module_name=f"{arch_name}_Generated")

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _sanitize(name: str) -> str:
        out = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
        if out and out[0].isdigit():
            out = "_" + out
        return out or "GeneratedCircuit"

    @staticmethod
    def _population_class_name(pop_name: str) -> str:
        """e.g. `sensory_cortex` → `Pop_sensory_cortex`."""
        return f"Pop_{pop_name}"

    # ── Header / imports ────────────────────────────────────────────

    def _gen_header(self) -> str:
        return (
            f'"""Auto-generated circuit module: {self.module_name}\n'
            f'\n'
            f'    populations: {len(self.ir.populations)}\n'
            f'    synapses:    {len(self.ir.synapses)}\n'
            f'    modulations: {len(self.ir.modulations)}\n'
            f'"""'
        )

    def _gen_imports(self) -> str:
        return (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F"
        )

    # ── Per-population class ────────────────────────────────────────

    def _resolve_decl(self, pop: PopulationIR
                      ) -> Tuple[Optional[DynamicsDecl], Optional[str], bool]:
        """Decide which (decl, body, is_ode) triple drives this population.

        Returns:
          decl    — params/state/constants carrier (may be None when the
                    user wrote a bare equation/ode that doesn't match any
                    macro; in that case unbound symbols default to scalar
                    learnable parameters initialized to zero).
          body    — the equation or ODE source string to lower. None means
                    we have no algebraic/ODE form (passthrough fallback).
          is_ode  — True if the body is an ODE (`dvar/dt = …`).

        Precedence (most specific wins):
          pop.ode      → ODE form (Stage 2)
          pop.equation → algebraic form (Stage 1)
          pop.dynamics → macro lookup (either form, depending on decl)
        """
        if pop.ode:
            ode_str = self._resolve_equation_ref(pop.ode)
            decl = find_decl_for_ode(ode_str)
            return decl, ode_str, True

        if pop.equation:
            eq_str = self._resolve_equation_ref(pop.equation)
            decl = find_decl_for_equation(eq_str)
            return decl, eq_str, False

        decl = DYNAMICS_DECLS.get(pop.dynamics)
        if decl is None:
            return None, None, False
        return decl, decl.body, decl.is_ode

    def _gen_population_class(self, pop: PopulationIR) -> str:
        decl, body, is_ode = self._resolve_decl(pop)
        if body is None:
            # No equation and no ODE form known → passthrough fallback.
            # Users with a custom dynamics not yet in the registry can
            # supply `equation:` or `ode:` to override this.
            return self._gen_passthrough_class(pop)

        if is_ode:
            return self._gen_ode_population_class(pop, decl, body)

        eq = parse_equation(body)
        free = set(eq.free_symbols)

        decl_params = dict(decl.params) if decl else {}
        decl_state = dict(decl.state) if decl else {}
        decl_consts = dict(decl.constants) if decl else {}

        declared = set(decl_params) | set(decl_state) | set(decl_consts)
        unbound = free - _RESERVED_SYMBOLS - declared

        # Auto-promote any unbound free symbol to a scalar nn.Parameter
        # initialized to zero. Users can override by supplying a matching
        # macro decl or (Stage 1.x) by adding explicit `params:` blocks.
        for name in sorted(unbound):
            decl_params[name] = "torch.zeros(1)"

        body_expr = lower_to_torch(eq)
        cls_name = self._population_class_name(pop.name)

        # __init__
        init = ["    def __init__(self, d_sem: int):",
                "        super().__init__()",
                "        self.d_sem = d_sem"]
        for name, init_expr in decl_params.items():
            init.append(f"        self.{name} = nn.Parameter({init_expr})")
        for name, init_expr in decl_state.items():
            init.append(f'        self.register_buffer("{name}", {init_expr})')

        # forward
        fwd = ["    def forward(self, x):",
               "        if x is None:",
               "            x = torch.zeros(1, self.d_sem)",
               "        d_sem = self.d_sem"]
        for name in decl_params:
            fwd.append(f"        {name} = self.{name}")
        for name in decl_state:
            fwd.append(f"        {name} = self.{name}")
        for name, val in decl_consts.items():
            fwd.append(f"        {name} = {val!r}")
        fwd.append(f"        return {body_expr}")

        doc = f'    """Population {pop.name}: {body}"""'
        return "\n".join([f"class {cls_name}(nn.Module):", doc, ""]
                         + init + [""] + fwd)

    def _gen_ode_population_class(self,
                                  pop: PopulationIR,
                                  decl: Optional[DynamicsDecl],
                                  ode_str: str) -> str:
        """Emit a population class that integrates an ODE one step per call.

        The generated forward implements the canonical Euler step:

            V_new = V_old + dt * rhs(V_old, x, params, constants)
            self.V := mean(V_new, dim=0)    # batch-collapsed for storage
            return V_new

        State is registered as a buffer named after the ODE's state variable.
        `dt` and any constants must be supplied by `decl.constants` — for the
        canonical macros we set them; for user-supplied ODEs without a decl
        we fall back to dt=0.01 and any unbound symbols default to scalar
        learnable params zero-initialised.
        """
        ode = parse_ode(ode_str)
        cls_name = self._population_class_name(pop.name)

        decl_params = dict(decl.params) if decl else {}
        decl_state = dict(decl.state) if decl else {}
        decl_consts = dict(decl.constants) if decl else {}

        # Ensure the state variable has a buffer declared. If the decl
        # didn't declare it (user-supplied bare ODE), register a default
        # (1, d_sem) zero buffer.
        if ode.state_var not in decl_state:
            decl_state[ode.state_var] = "torch.zeros(1, d_sem)"

        # dt must exist as a constant for the Euler step. Default 0.01 if
        # the decl didn't set one.
        if "dt" not in decl_consts:
            decl_consts["dt"] = 0.01

        # Auto-promote any free symbol not in {state, dt, constants, x,
        # d_sem} to a scalar learnable param.
        free = set(ode.free_symbols)
        already_bound = (
            {ode.state_var, "x", "d_sem", "dt"}
            | set(decl_params) | set(decl_state) | set(decl_consts)
        )
        for name in sorted(free - already_bound):
            decl_params[name] = "torch.zeros(1)"

        # Lower the rhs by wrapping it in an EquationExpr-like façade so we
        # can reuse the existing lowering. Simplest: build a fake "y = rhs"
        # equation and lower it.
        from .equations import EquationExpr
        fake = EquationExpr(
            source=str(ode.rhs),
            lhs=__import__("sympy").Symbol("__rhs__"),
            rhs=ode.rhs,
            free_symbols=set(ode.free_symbols),
        )
        rhs_expr = lower_to_torch(fake)

        # __init__
        init = ["    def __init__(self, d_sem: int):",
                "        super().__init__()",
                "        self.d_sem = d_sem"]
        for name, init_expr in decl_params.items():
            init.append(f"        self.{name} = nn.Parameter({init_expr})")
        for name, init_expr in decl_state.items():
            init.append(f'        self.register_buffer("{name}", {init_expr})')

        # forward — Euler step
        fwd = ["    def forward(self, x):",
               "        if x is None:",
               "            x = torch.zeros(1, self.d_sem)",
               "        d_sem = self.d_sem"]
        for name in decl_params:
            fwd.append(f"        {name} = self.{name}")
        for name in decl_state:
            fwd.append(f"        {name} = self.{name}")
        for name, val in decl_consts.items():
            fwd.append(f"        {name} = {val!r}")

        sv = ode.state_var
        fwd.append(f"        _rhs = {rhs_expr}")
        fwd.append(f"        {sv}_new = {sv} + dt * _rhs")
        # Persist the new state, batch-collapsed so the buffer keeps its
        # (1, d_sem) shape for next-step broadcasting.
        fwd.append(
            f"        self.{sv}.copy_({sv}_new.detach().mean(dim=0, keepdim=True))"
        )
        fwd.append(f"        return {sv}_new")

        doc = f'    """Population {pop.name}: {ode_str}"""'
        return "\n".join([f"class {cls_name}(nn.Module):", doc, ""]
                         + init + [""] + fwd)

    # ── Synapse / modulation equation resolution ────────────────────

    # Canonical legacy form for a synapse with `weight: w`: every legacy
    # synapse becomes `y = weight * (x_pre @ W)` where W is the
    # randomly-initialised weight buffer. Explicit `equation:` overrides.
    _SYNAPSE_CANONICAL = "y = weight * (x_pre @ W)"

    # Canonical legacy forms for modulations:
    #   multiplicative: y = output * (c * gain)    — preserves Phase-5 behavior
    #   additive:       y = output + (c * gain)
    _MOD_CANONICAL = {
        "multiplicative": "y = output * (c * gain)",
        "additive":       "y = output + (c * gain)",
    }

    def _resolve_equation_ref(self, eq_str: str) -> str:
        """Resolve equation reference (@name) to its formula.

        If eq_str starts with @, look it up in equation_decls.
        Otherwise, return eq_str unchanged.
        """
        if eq_str and eq_str.startswith("@"):
            eq_name = eq_str[1:]
            for eq_def in self.ir.equation_decls:
                if eq_def.name == eq_name:
                    return eq_def.formula
            raise ValueError(f"undefined equation reference: {eq_str!r}")
        return eq_str

    def _synapse_contribution_expr(self, i: int,
                                   syn: SynapseIR,
                                   src_expr: str) -> str:
        """Lower a synapse to the Python expression evaluating its contribution.

        Resolves the synapse's equation (explicit or canonical legacy),
        binds runtime symbols to the right Python expressions, and returns
        a single inline expression suitable for use as part of the target
        population's input sum.
        """
        eq_str = syn.equation or self._SYNAPSE_CANONICAL
        eq_str = self._resolve_equation_ref(eq_str)
        eq = parse_equation(eq_str)
        weight = syn.weight if syn.weight is not None else 1.0
        name_map = {
            "x_pre":  src_expr,
            "W":      f"self.syn_{i}_w",
            "weight": repr(float(weight)),
        }
        return lower_to_torch(eq, name_map=name_map)

    def _modulation_expr(self, mod: ModulationIR) -> str:
        """Lower a modulation to the Python expression for the new target output."""
        if mod.equation:
            eq_str = mod.equation
        else:
            effect = mod.effect or "multiplicative"
            if effect not in self._MOD_CANONICAL:
                raise ValueError(
                    f"unsupported modulation effect {effect!r}; "
                    f"either use equation: or set effect to one of "
                    f"{list(self._MOD_CANONICAL)}"
                )
            eq_str = self._MOD_CANONICAL[effect]

        eq_str = self._resolve_equation_ref(eq_str)
        eq = parse_equation(eq_str)
        gain = mod.gain if mod.gain is not None else 1.0
        name_map = {
            "output": f"outputs[{mod.target_population!r}]",
            "c":      f"nt_levels[{mod.source_nt!r}]",
            "gain":   repr(float(gain)),
        }
        return lower_to_torch(eq, name_map=name_map)

    def _gen_passthrough_class(self, pop: PopulationIR) -> str:
        """Stub class for dynamics that Stage 1 doesn't lower yet.

        Returns the input unchanged. Stage 2 replaces this with the real
        ODE-integrating implementation.
        """
        cls_name = self._population_class_name(pop.name)
        return (
            f"class {cls_name}(nn.Module):\n"
            f'    """Population {pop.name}: passthrough (no Stage-1 equation)"""\n'
            f"    def __init__(self, d_sem: int):\n"
            f"        super().__init__()\n"
            f"        self.d_sem = d_sem\n"
            f"\n"
            f"    def forward(self, x):\n"
            f"        if x is None:\n"
            f"            x = torch.zeros(1, self.d_sem)\n"
            f"        return x"
        )

    # ── Top-level circuit class ─────────────────────────────────────

    def _gen_circuit_class(self) -> str:
        cls_name = self.module_name

        # __init__
        init = [
            "    def __init__(self, d_sem: int = 256):",
            "        super().__init__()",
            "        self.d_sem = d_sem",
        ]

        # Instantiate per-population modules
        for pop in self.ir.populations:
            pcls = self._population_class_name(pop.name)
            init.append(f"        self.{pop.name} = {pcls}(d_sem)")

        # Synapse weights as fixed buffers (random init; not learnable in
        # Stage 1 — that's the codegen contract today, can be lifted later
        # by emitting Linear layers instead).
        for i, syn in enumerate(self.ir.synapses):
            init.append(
                f'        self.register_buffer("syn_{i}_w", '
                f'torch.randn(d_sem, d_sem) * 0.1)'
            )

        # Previous-step output buffers for back-edges. Created for every
        # population so we don't have to enumerate back-edges separately —
        # cheap memory cost and keeps the generated code simple.
        for pop in self.ir.populations:
            init.append(
                f'        self.register_buffer("last_{pop.name}", '
                f'torch.zeros(1, d_sem))'
            )

        # forward
        fwd = [
            "    def forward(self, sensory_input, nt_levels=None):",
            "        outputs = {}",
            "        batch = sensory_input.shape[0]",
        ]

        # Compute population evaluation order (declaration order) and
        # classify each incoming synapse as forward-edge or back-edge.
        order_index = {p.name: i for i, p in enumerate(self.ir.populations)}
        for pop in self.ir.populations:
            incoming = [
                (i, s) for i, s in enumerate(self.ir.synapses)
                if s.target == pop.name
            ]
            if not incoming:
                fwd.append(
                    f"        outputs[{pop.name!r}] = "
                    f"self.{pop.name}(sensory_input)"
                )
                continue

            # Each incoming synapse: resolve its equation (explicit or the
            # canonical legacy form `y = weight * (x_pre @ W)`), lower it,
            # binding free symbols to runtime expressions.
            terms = []
            for i, syn in incoming:
                src_idx = order_index.get(syn.source, -1)
                tgt_idx = order_index[pop.name]
                if src_idx < tgt_idx:
                    src_expr = f"outputs[{syn.source!r}]"
                else:
                    src_expr = f"self.last_{syn.source}.expand(batch, -1)"

                terms.append(self._synapse_contribution_expr(i, syn, src_expr))

            input_expr = " + ".join(terms)
            fwd.append(
                f"        outputs[{pop.name!r}] = "
                f"self.{pop.name}({input_expr})"
            )

        # NT modulations — equation-driven
        if self.ir.modulations:
            fwd.append("        if nt_levels is not None:")
            for mod in self.ir.modulations:
                target = mod.target_population
                if target not in order_index:
                    continue
                fwd.append(f"            if {mod.source_nt!r} in nt_levels:")
                fwd.append(
                    f"                outputs[{target!r}] = "
                    + self._modulation_expr(mod)
                )

        # Update last-step buffers (detached, in-place to preserve buffer
        # identity for the next call).
        for pop in self.ir.populations:
            fwd.append(
                f"        self.last_{pop.name}.copy_("
                f"outputs[{pop.name!r}].detach().mean(dim=0, keepdim=True))"
            )

        fwd.append("        return outputs")

        doc = (
            f'    """Top-level circuit: {len(self.ir.populations)} '
            f'populations, {len(self.ir.synapses)} synapses, '
            f'{len(self.ir.modulations)} modulations."""'
        )
        return "\n".join([f"class {cls_name}(nn.Module):", doc, ""]
                         + init + [""] + fwd)


# ── Module-level convenience ────────────────────────────────────────────────

def generate_module(ir: "ProgramIR", module_name: str = "GeneratedCircuit") -> str:
    """Convenience wrapper: compile a ProgramIR to Python source.

    Equivalent to ``CodeGenerator(ir, module_name).generate()``.
    """
    return CodeGenerator(ir, module_name=module_name).generate()
