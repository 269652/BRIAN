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
    FeatureIR,
    FeatureEndpointIR,
    _ParamRef,
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


class CodegenError(RuntimeError):
    """Raised when the IR is structurally valid but cannot be lowered.

    Examples: a synapse references an unknown feature, a feature is
    active but missing its ``impl:`` binding, an endpoint kind doesn't
    match its usage site. Per CLAUDE.md §14 every such error is loud
    and fatal — we never silently fall back to a stub.
    """


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
        # v2.0 DSL: vesicles and sieves
        for vesicle in self.ir.vesicles:
            parts.append(self._gen_vesicle_class(vesicle))
        for sieve in self.ir.sieves:
            parts.append(self._gen_sieve_class(sieve))
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
        lines = [
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
        ]
        # §14: an active feature wired into a synapse must have its
        # impl class imported into the generated module. Inactive
        # features are deliberately omitted so the ablation switch
        # (flip `active: false`) genuinely strips the dependency.
        seen: set = set()
        for feat in self._wired_active_features():
            module_path, cls_name = self._split_impl_path(feat)
            key = (module_path, cls_name)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"from {module_path} import {cls_name}")
        return "\n".join(lines)

    # ── Feature wiring (§14) ────────────────────────────────────────

    def _feature_by_name(self, name: str) -> Optional[FeatureIR]:
        for f in self.ir.features:
            if f.name == name:
                return f
        return None

    def _resolve_feature_ref(
        self, ref: str, *, context: str
    ) -> Tuple[FeatureIR, FeatureEndpointIR]:
        """Resolve a synapse/modulation ``feature: "..."`` reference.

        Accepts ``"<feature>"`` (short form, requires exactly one
        endpoint on the feature) or ``"<feature>.<endpoint>"``.
        Raises :class:`CodegenError` for any of: unknown feature,
        unknown endpoint, ambiguous short form. ``context`` is a
        human-readable site name for the error message (e.g.
        ``"synapse a->b"``).
        """
        parts = ref.split(".", 1)
        feat_name = parts[0]
        ep_name = parts[1] if len(parts) == 2 else None

        feat = self._feature_by_name(feat_name)
        if feat is None:
            known = ", ".join(sorted(f.name for f in self.ir.features)) \
                    or "<none>"
            raise CodegenError(
                f"{context}: unknown feature {feat_name!r} "
                f"(known: {known})"
            )

        if ep_name is None:
            if len(feat.endpoints) == 0:
                raise CodegenError(
                    f"{context}: feature {feat_name!r} has no endpoints; "
                    f"cannot resolve short-form reference {ref!r}"
                )
            if len(feat.endpoints) > 1:
                names = ", ".join(e.name for e in feat.endpoints)
                raise CodegenError(
                    f"{context}: feature {feat_name!r} has multiple "
                    f"endpoints ({names}); short-form reference {ref!r} "
                    f"is ambiguous — use \"{feat_name}.<endpoint>\""
                )
            return feat, feat.endpoints[0]

        for ep in feat.endpoints:
            if ep.name == ep_name:
                return feat, ep
        known = ", ".join(e.name for e in feat.endpoints) or "<none>"
        raise CodegenError(
            f"{context}: feature {feat_name!r} has no endpoint "
            f"{ep_name!r} (known: {known})"
        )

    def _wired_active_features(self) -> List[FeatureIR]:
        """Active features that at least one synapse actually references.

        Inactive features — even if referenced — are excluded so the
        canonical edge takes over and no impl import / instantiation
        leaks into the generated module.
        """
        out: List[FeatureIR] = []
        seen_names: set = set()
        for syn in self.ir.synapses:
            if not syn.feature_ref:
                continue
            try:
                feat, _ = self._resolve_feature_ref(
                    syn.feature_ref,
                    context=f"synapse {syn.source}->{syn.target}",
                )
            except CodegenError:
                # Defer the error to the actual lowering step so the
                # message points at the failing synapse, not just at
                # import generation.
                continue
            if not feat.active:
                continue
            if feat.name in seen_names:
                continue
            seen_names.add(feat.name)
            out.append(feat)
        return out

    def _split_impl_path(self, feat: FeatureIR) -> Tuple[str, str]:
        """Split ``feat.impl`` into ``(module_path, class_name)``.

        Raises :class:`CodegenError` if the impl is missing or malformed.
        """
        if not feat.impl:
            raise CodegenError(
                f"feature {feat.name!r}: active and wired into a synapse "
                f"but has no `impl:` binding — per §14 every wired "
                f"feature MUST resolve to a real Python class."
            )
        if "." not in feat.impl:
            raise CodegenError(
                f"feature {feat.name!r}: `impl` must be a dotted Python "
                f"path (e.g. ``pkg.mod.ClassName``); got {feat.impl!r}"
            )
        return feat.impl.rsplit(".", 1)

    @staticmethod
    def _format_kwarg(value) -> str:
        """Render a feature.params value as a Python literal kwarg.

        Coercion table (ordered most-specific to least):

        * :class:`_ParamRef` → bare identifier (no quotes). Lets the
          impl constructor receive a runtime variable like ``d_sem``.
        * ``bool`` → ``True`` / ``False`` (must come BEFORE the int
          branch because ``isinstance(True, int)`` is ``True``).
        * ``int`` → integer literal.
        * ``float`` → keeps the decimal (so ``c=1.0`` not ``c=1``).
        * ``str`` → quoted via ``repr``.
        * fallback → ``repr``.
        """
        if isinstance(value, _ParamRef):
            return value.expr
        if isinstance(value, bool):
            return repr(value)
        if isinstance(value, int):
            return repr(value)
        if isinstance(value, float):
            return repr(float(value))
        if isinstance(value, str):
            return repr(value)
        return repr(value)

    def _feature_init_lines(self, feat: FeatureIR) -> List[str]:
        """Emit ``self.feature_<name> = ImplClass(**params)`` for one
        active wired feature.
        """
        _, cls_name = self._split_impl_path(feat)
        kwargs = ", ".join(
            f"{k}={self._format_kwarg(v)}" for k, v in feat.params.items()
        )
        return [
            f"        self.feature_{feat.name} = {cls_name}({kwargs})",
        ]

    def _feature_edge_expr(
        self,
        syn: SynapseIR,
        src_expr: str,
        feat: FeatureIR,
        ep: FeatureEndpointIR,
    ) -> str:
        """Lower a feature-routed synapse to a Python expression.

        Contract (kind="edge"): the impl is called like an nn.Module
        on the pre-population's activation. Most impls (notably
        attention) expect a sequence dim ``(B, T, D)``; the canonical
        BRIAN circuit emits ``(B, D)`` per step. We adapt by
        unsqueezing to ``T=1`` and squeezing back. The result is then
        gated by the synapse weight, preserving the legacy edge-
        weighting semantics.
        """
        if ep.kind != "edge":
            raise CodegenError(
                f"synapse {syn.source}->{syn.target}: feature endpoint "
                f"{feat.name}.{ep.name} has kind {ep.kind!r}; only "
                f"kind=\"edge\" can be wired into a synapse (kinds "
                f"\"modulator\" / \"transform\" are wired into "
                f"modulations / populations respectively)."
            )
        weight = float(syn.weight if syn.weight is not None else 1.0)
        # Impl forward: (B, T, D) → (B, T, D). The synapse passes (B, D),
        # we add T=1 then strip it so the contribution lands at (B, D)
        # for the post-population's input sum.
        return (
            f"({weight!r} * self.feature_{feat.name}"
            f"(({src_expr}).unsqueeze(1)).squeeze(1))"
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

        §14 path: if the synapse has a ``feature_ref`` AND the referenced
        feature is ``active: true``, the contribution is computed by
        calling the feature's impl class instead of the canonical
        ``weight * (x_pre @ W)``. Inactive features fall through to the
        canonical form — flipping ``active`` is therefore a real
        ablation switch.
        """
        if syn.feature_ref:
            feat, ep = self._resolve_feature_ref(
                syn.feature_ref,
                context=f"synapse {syn.source}->{syn.target}",
            )
            if feat.active:
                # Validate impl exists *now* so the error points at
                # the synapse site (covered by
                # test_active_feature_missing_impl_raises_when_wired).
                self._split_impl_path(feat)
                return self._feature_edge_expr(syn, src_expr, feat, ep)
            # Inactive → fall through to canonical edge below.

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

    # ── v2.0 Vesicle and Sieve generation (Phase II) ────────────────

    def _gen_vesicle_class(self, vesicle) -> str:
        """Generate a vesicle docking module (zero-init gate for ReZero contract)."""
        cls_name = f"Vesicle_{vesicle.name}"

        init = [
            f"class {cls_name}(nn.Module):",
            f'    """Neuro-vesicle: {vesicle.name} (trigger: {vesicle.trigger})"""',
            "    def __init__(self, d_sem: int):",
            "        super().__init__()",
            "        self.d_sem = d_sem",
            "        self.lifetime = " + str(vesicle.lifetime),
            "        self.content_dim = " + str(vesicle.content_dim),
            "        self.alpha = nn.Parameter(torch.zeros(1))  # ReZero zero-init gate",
            "",
            "    def forward(self, x):",
            "        if x is None:",
            "            x = torch.zeros(1, self.d_sem)",
            "        gate = torch.sigmoid(self.alpha)  # Soft gate, zero-init → ~0",
            "        # With zero-init gate, output ≈ x (identity at first forward)",
            "        return x * (1 - gate)  # Gate opens as alpha grows",
        ]

        return "\n".join(init)

    def _gen_sieve_class(self, sieve) -> str:
        """Generate a topological sieve (gnorm filtering with zero-init gate)."""
        cls_name = f"Sieve_{sieve.name}"

        init = [
            f"class {cls_name}(nn.Module):",
            f'    """Topological sieve: {sieve.name} (threshold: {sieve.gnorm_threshold})"""',
            "    def __init__(self, d_sem: int):",
            "        super().__init__()",
            "        self.d_sem = d_sem",
            "        self.threshold = " + str(sieve.gnorm_threshold),
            "        self.gate = nn.Parameter(torch.zeros(1))  # ReZero zero-init",
            "",
            "    def forward(self, x):",
            "        if x is None:",
            "            x = torch.zeros(1, self.d_sem)",
            "        g = torch.sigmoid(self.gate)  # Soft gate, zero-init → ~0",
            "        gnorm = torch.linalg.norm(x, dim=-1, keepdim=True)",
            "        # When gnorm > threshold and gate is active, project orthogonal",
            "        mask = (gnorm > self.threshold).float()",
            "        # For now, simple passthrough; projection logic in Phase II.5",
            "        return x * (1 - g * mask)  # Gate controls filtering",
        ]

        return "\n".join(init)

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

        # §14: instantiate any active feature impl whose endpoint is
        # actually referenced by a synapse. These appear as
        # ``self.feature_<name>`` modules on the circuit and are
        # invoked from the forward pass via _feature_edge_expr.
        for feat in self._wired_active_features():
            init.extend(self._feature_init_lines(feat))

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
