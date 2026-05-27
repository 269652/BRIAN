# -*- coding: utf-8 -*-
"""Mathematical equation layer for the NeuroML DSL.

Stage 1 of the math-first DSL: algebraic equations of the form `y = f(x)`.
Each population/synapse/modulation may carry an `equation:` field. The
parser turns the equation into a SymPy expression; the lowerer emits the
equivalent PyTorch op sequence; the solver answers fixed-point/stability
questions about the equation.

Enum dynamics (`dynamics: "rate_code"`) are still supported — they expand
to canonical equations via `expand_dynamics_macro()`.

Variable conventions
    Populations:     x = input, y = output, s = persistent state (Stage 2+)
    Synapses:        x_pre = presynaptic activation, y = contribution to target
    Modulations:     c = NT concentration, gain = output gain factor
    Parameters:      any other free symbol resolved from params dict

Supported algebraic operators
    +  -  *  /         arithmetic (elementwise, broadcast)
    @                  matrix multiply
    **                 power
    ReLU sigmoid tanh  pointwise nonlinearities
    softmax            normalized exponential (last dim)
    exp log sqrt       elementary functions
    max(a, b) min      pointwise extrema
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

import sympy as sp


# ── Canonical equations + declarations for enum dynamics ──────────────
#
# `DYNAMICS_EQUATIONS` exposes the equation string for backwards-compat;
# `DYNAMICS_DECLS` is the structured form the codegen consumes — it says
# what learnable params, persistent state, and built-in constants each
# macro implies.
#
# Why this matters: the equation `y = ReLU(x) * sigmoid(gate)` is fine to
# read, but to *generate* a working nn.Module the codegen also needs to
# know that `gate` is `nn.Parameter(torch.zeros(1))`. That information
# lives here and is keyed on the dynamics name. When the user writes the
# canonical equation explicitly we reverse-look-up the same decl, so
# enum-form and explicit-equation-form produce byte-identical modules.

@dataclass(frozen=True)
class DynamicsDecl:
    """Everything codegen needs to emit a population class.

    Exactly one of `equation` (algebraic, Stage 1) or `ode` (Stage 2)
    must be set. The other helpers (`is_ode`, `body`) are read by the
    codegen to pick the right lowering path.

    equation:  algebraic equation `y = f(...)` — closed-form output
    ode:       ODE `dV/dt = g(...)` — integrated by Euler step in forward
    params:    {name → init expression}  — emitted as nn.Parameter
    state:     {name → init expression}  — emitted as register_buffer
    constants: {name → Python literal}   — bound as local in forward()
    """
    equation: Optional[str] = None
    ode: Optional[str] = None
    params: Dict[str, str] = field(default_factory=dict)
    state: Dict[str, str] = field(default_factory=dict)
    constants: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if (self.equation is None) == (self.ode is None):
            raise ValueError("DynamicsDecl needs exactly one of equation= or ode=")

    @property
    def is_ode(self) -> bool:
        return self.ode is not None

    @property
    def body(self) -> str:
        return self.equation if self.equation is not None else self.ode


DYNAMICS_DECLS: Dict[str, Optional[DynamicsDecl]] = {
    "rate_code": DynamicsDecl(
        equation="y = ReLU(x)",
    ),
    "winner_take_all": DynamicsDecl(
        equation="y = softmax(x / 0.1) * d_sem",
    ),
    "gated": DynamicsDecl(
        equation="y = ReLU(x) * sigmoid(gate)",
        params={"gate": "torch.zeros(1)"},
    ),
    "attractor_network": DynamicsDecl(
        equation="y = (1 - alpha) * s + alpha * ReLU(x)",
        state={"s": "torch.zeros(1, d_sem)"},
        constants={"alpha": 0.1},
    ),
    "attention_pool": DynamicsDecl(
        equation="y = softmax(x) * ReLU(x)",
    ),
    "static": DynamicsDecl(
        equation="y = x",
    ),
    # Stage 2 — leaky integrate-and-fire (sub-threshold form). Spiking
    # events (threshold + reset) are deferred to Stage 3 Brian2 grammar.
    "integrate_and_fire": DynamicsDecl(
        ode="dV/dt = (-V + x) / tau",
        state={"V": "torch.zeros(1, d_sem)"},
        constants={"tau": 0.05, "dt": 0.01},
    ),
}

DYNAMICS_EQUATIONS: Dict[str, Optional[str]] = {
    name: (decl.equation if decl is not None and not decl.is_ode else None)
    for name, decl in DYNAMICS_DECLS.items()
}


def find_decl_for_equation(equation_str: str) -> Optional[DynamicsDecl]:
    """Reverse lookup: given an equation string, return the matching decl.

    Used by codegen when the user wrote an explicit `equation:` field — if
    it's the canonical form of a known algebraic macro, we want the same
    params/state/constants applied. Returns None if no macro matches.
    """
    target = equation_str.strip()
    for name, decl in DYNAMICS_DECLS.items():
        if decl is not None and not decl.is_ode and decl.equation.strip() == target:
            return decl
    return None


def find_decl_for_ode(ode_str: str) -> Optional[DynamicsDecl]:
    """Reverse lookup for ODE-form macros — parallel of `find_decl_for_equation`."""
    target = ode_str.strip()
    for name, decl in DYNAMICS_DECLS.items():
        if decl is not None and decl.is_ode and decl.ode.strip() == target:
            return decl
    return None


# ── Operator → SymPy function map ──────────────────────────────────────
#
# SymPy doesn't ship neural-net nonlinearities, so we declare them as
# undefined functions. The lowerer recognizes them by name and emits the
# corresponding torch op. Anyone reading the equation also gets a faithful
# symbolic form (e.g. `ReLU(W @ x + b)`).

_RELU      = sp.Function("ReLU")
_SIGMOID   = sp.Function("sigmoid")
_TANH      = sp.Function("tanh")
_SOFTMAX   = sp.Function("softmax")
_MATMUL    = sp.Function("matmul")  # used for `@` after pre-parse rewrite

# Functions SymPy already knows: exp, log, sqrt, Max, Min — we route them
# through sp.* directly.

_LOCAL_DICT: Dict[str, Any] = {
    "ReLU":    _RELU,
    "relu":    _RELU,
    "sigmoid": _SIGMOID,
    "tanh":    _TANH,
    "softmax": _SOFTMAX,
    "exp":     sp.exp,
    "log":     sp.log,
    "sqrt":    sp.sqrt,
    "max":     sp.Max,
    "min":     sp.Min,
    "Max":     sp.Max,
    "Min":     sp.Min,
    # Override SymPy's built-in `I` (imaginary unit) and `E` so neural-
    # current and other single-letter symbols stay as plain Symbols.
    # Users writing complex math should spell it out as sp.I explicitly.
    "I":       sp.Symbol("I"),
    "E":       sp.Symbol("E"),
}


# ── Equation IR ────────────────────────────────────────────────────────

@dataclass
class EquationExpr:
    """Parsed algebraic equation.

    The DSL form `y = ReLU(W @ x + b)` becomes:
        lhs   = sp.Symbol('y')
        rhs   = ReLU(matmul(W, x) + b)
        free  = {'W', 'x', 'b'}
    """
    source: str                                  # original equation string
    lhs: sp.Symbol                               # left-hand-side (output) symbol
    rhs: sp.Expr                                 # right-hand-side expression
    free_symbols: Set[str] = field(default_factory=set)

    def __str__(self) -> str:
        return f"{self.lhs} = {self.rhs}"


# ── Parser ─────────────────────────────────────────────────────────────

def parse_equation(source: str) -> EquationExpr:
    """Parse `lhs = rhs` algebraic equation string into an EquationExpr.

    The parser rewrites `@` (matrix multiply) to a `matmul(a, b)` function
    call before handing the string to SymPy, because `@` isn't a SymPy
    operator. All other operators (`+ - * / **`) are SymPy-native.

    Raises ValueError on malformed input or unsupported tokens.
    """
    if not source or "=" not in source:
        raise ValueError(f"equation must contain '=': {source!r}")

    lhs_str, _, rhs_str = source.partition("=")
    lhs_str, rhs_str = lhs_str.strip(), rhs_str.strip()

    if not lhs_str.isidentifier():
        raise ValueError(f"equation LHS must be a single identifier, got: {lhs_str!r}")

    # Rewrite `a @ b` → `matmul(a, b)` so SymPy can parse it. Left-associative;
    # `a @ b @ c` becomes `matmul(matmul(a, b), c)`.
    rhs_rewritten = _rewrite_matmul(rhs_str)

    try:
        rhs_expr = sp.sympify(rhs_rewritten, locals=_LOCAL_DICT)
    except (sp.SympifyError, SyntaxError, TypeError) as e:
        raise ValueError(f"could not parse equation RHS {rhs_str!r}: {e}") from e

    lhs_sym = sp.Symbol(lhs_str)
    free = {str(s) for s in rhs_expr.free_symbols}

    return EquationExpr(source=source, lhs=lhs_sym, rhs=rhs_expr, free_symbols=free)


def _rewrite_matmul(expr_str: str) -> str:
    """Rewrite `a @ b` → `matmul(a, b)`. Handles nested parentheses correctly."""
    # Walk the string, find each `@`, and wrap the two operands.
    # An "operand" is either a parenthesized group or an identifier (possibly
    # with attribute access / function call). This is intentionally simple —
    # if it fails, the user can always use `matmul(...)` explicitly.
    s = expr_str
    while "@" in s:
        idx = s.index("@")
        left_end = idx
        right_start = idx + 1

        # Walk left to find start of left operand
        i = left_end - 1
        while i >= 0 and s[i] == " ":
            i -= 1
        if i < 0:
            raise ValueError(f"missing left operand for @ in {expr_str!r}")
        if s[i] == ")":
            depth = 1
            j = i - 1
            while j >= 0 and depth > 0:
                if s[j] == ")": depth += 1
                elif s[j] == "(": depth -= 1
                j -= 1
            left_start = j + 1
        else:
            j = i
            while j >= 0 and (s[j].isalnum() or s[j] in "_."):
                j -= 1
            left_start = j + 1
        left = s[left_start:left_end].strip()

        # Walk right to find end of right operand
        i = right_start
        while i < len(s) and s[i] == " ":
            i += 1
        if i >= len(s):
            raise ValueError(f"missing right operand for @ in {expr_str!r}")
        if s[i] == "(":
            depth = 1
            j = i + 1
            while j < len(s) and depth > 0:
                if s[j] == "(": depth += 1
                elif s[j] == ")": depth -= 1
                j += 1
            right_end = j
        else:
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] in "_."):
                j += 1
            right_end = j
        right = s[i:right_end].strip()

        s = s[:left_start] + f"matmul({left}, {right})" + s[right_end:]
    return s


# ── Macro expansion (enum → equation) ──────────────────────────────────

def expand_dynamics_macro(dynamics_name: str) -> Optional[str]:
    """Return the canonical equation string for an enum dynamics name.

    Returns None for dynamics that don't have an algebraic form yet
    (e.g. `integrate_and_fire` — that needs Stage 2 ODE support).
    """
    return DYNAMICS_EQUATIONS.get(dynamics_name)


def equation_for_population(dynamics: Optional[str],
                            equation: Optional[str]) -> Optional[EquationExpr]:
    """Resolve a population's effective equation.

    Precedence:
        1. Explicit `equation:` field wins
        2. Otherwise expand the `dynamics:` enum macro
        3. None → caller should fall back to template-based codegen
    """
    if equation:
        return parse_equation(equation)
    if dynamics:
        canonical = expand_dynamics_macro(dynamics)
        if canonical is not None:
            return parse_equation(canonical)
    return None


# ── Lowering to PyTorch source ─────────────────────────────────────────

def lower_to_torch(expr: EquationExpr,
                   tensor_vars: Optional[Set[str]] = None) -> str:
    """Render the equation's RHS as a Python expression using torch ops.

    Args:
        expr: parsed EquationExpr
        tensor_vars: names of free symbols that are runtime tensors (the
            rest are treated as scalars or parameters). When None, all
            free symbols are assumed to be tensors.

    Returns:
        A Python expression string that, evaluated in a scope where the
        free symbols are bound to torch tensors / scalars, produces the
        equation's output.
    """
    return _lower_node(expr.rhs, tensor_vars)


def _lower_node(node: sp.Expr, tensor_vars: Optional[Set[str]]) -> str:
    """Recursively lower a SymPy expression to a torch-op Python string."""
    # Symbols → plain identifier
    if isinstance(node, sp.Symbol):
        return node.name

    # Numbers → literal
    if node.is_Number:
        # Render integers without ".0", floats with full precision
        if node.is_Integer:
            return str(int(node))
        return repr(float(node))

    # Named functions
    if isinstance(node, sp.Function):
        fn = node.func.__name__
        args = [_lower_node(a, tensor_vars) for a in node.args]
        return _emit_function(fn, args)

    # Arithmetic
    if isinstance(node, sp.Add):
        return "(" + " + ".join(_lower_node(a, tensor_vars) for a in node.args) + ")"
    if isinstance(node, sp.Mul):
        return "(" + " * ".join(_lower_node(a, tensor_vars) for a in node.args) + ")"
    if isinstance(node, sp.Pow):
        base = _lower_node(node.base, tensor_vars)
        expn = _lower_node(node.exp, tensor_vars)
        return f"({base} ** {expn})"

    raise NotImplementedError(f"can't lower SymPy node {type(node).__name__}: {node}")


def _emit_function(fn: str, args: List[str]) -> str:
    """Emit the torch-op equivalent of a named function call."""
    if fn in ("ReLU", "relu"):
        assert len(args) == 1
        return f"F.relu({args[0]})"
    if fn == "sigmoid":
        assert len(args) == 1
        return f"torch.sigmoid({args[0]})"
    if fn == "tanh":
        assert len(args) == 1
        return f"torch.tanh({args[0]})"
    if fn == "softmax":
        assert len(args) == 1
        return f"F.softmax({args[0]}, dim=-1)"
    if fn == "matmul":
        assert len(args) == 2
        return f"({args[0]} @ {args[1]})"
    if fn == "exp":
        return f"torch.exp({args[0]})"
    if fn == "log":
        return f"torch.log({args[0]})"
    if fn == "sqrt":
        return f"torch.sqrt({args[0]})"
    if fn in ("Max", "max"):
        return f"torch.maximum({args[0]}, {args[1]})"
    if fn in ("Min", "min"):
        return f"torch.minimum({args[0]}, {args[1]})"
    raise NotImplementedError(f"no torch lowering for function {fn!r}")


# ── Symbolic analysis (fixed-point / Jacobian) ─────────────────────────
#
# Our nonlinearities (ReLU, sigmoid, tanh, softmax) are declared as
# undefined SymPy functions so they render nicely in pretty-printed
# equations. But SymPy can't differentiate or numerically evaluate
# undefined functions, which breaks fixed-point and Jacobian work.
#
# The workaround: before any symbolic analysis, expand placeholders into
# their closed forms. The parsed equation stays human-readable; only the
# analysis path sees the expanded form.

def _expand_for_analysis(expr: sp.Expr) -> sp.Expr:
    """Rewrite ReLU/sigmoid/tanh placeholders into differentiable forms.

    softmax is not handled here — it's multi-dimensional and its analysis
    belongs in a tensor-aware Stage. matmul → ordinary product (scalar
    analysis only); the caller is responsible for noticing if their
    "fixed point" question involves matrix vars.
    """
    return expr.replace(
        lambda e: getattr(e.func, "__name__", "") == "sigmoid",
        lambda e: 1 / (1 + sp.exp(-e.args[0]))
    ).replace(
        lambda e: getattr(e.func, "__name__", "") == "tanh",
        lambda e: sp.tanh(e.args[0])
    ).replace(
        lambda e: getattr(e.func, "__name__", "") == "ReLU",
        lambda e: sp.Max(0, e.args[0])
    ).replace(
        lambda e: getattr(e.func, "__name__", "") == "matmul",
        lambda e: e.args[0] * e.args[1]
    )


def find_fixed_point(expr: EquationExpr,
                     input_symbol: str = "x",
                     param_bindings: Optional[Dict[str, float]] = None,
                     guess: float = 0.0,
                     max_iter: int = 50) -> Optional[float]:
    """Find x* such that x* = f(x*) for the equation y = f(x).

    Useful for recurrent populations: given `y = ReLU(W*x + b)` and a
    recurrence `x = y`, the fixed point is the steady-state activation.

    Returns None if the solver doesn't converge (e.g. for ReLU(x) the
    fixed-point equation `x = ReLU(x)` has every nonneg point as a fixed
    point — SymPy returns NaN).
    """
    param_bindings = param_bindings or {}

    rhs = _expand_for_analysis(expr.rhs)
    for name, val in param_bindings.items():
        rhs = rhs.subs(sp.Symbol(name), val)

    input_sym = sp.Symbol(input_symbol)
    fixed_point_eq = rhs - input_sym

    try:
        sol = sp.nsolve(fixed_point_eq, input_sym, guess, maxsteps=max_iter)
        return float(sol)
    except (ValueError, ZeroDivisionError, sp.PolynomialError):
        return None


def jacobian_at(expr: EquationExpr,
                input_symbol: str,
                point: float,
                param_bindings: Optional[Dict[str, float]] = None) -> Optional[float]:
    """Compute df/dx at the given point — used for stability analysis.

    For 1D the eigenvalue is just the derivative; |df/dx| < 1 at a fixed
    point means the fixed point is locally stable under the recurrence
    x_{t+1} = f(x_t).
    """
    param_bindings = param_bindings or {}
    rhs = _expand_for_analysis(expr.rhs)
    for name, val in param_bindings.items():
        rhs = rhs.subs(sp.Symbol(name), val)

    input_sym = sp.Symbol(input_symbol)
    try:
        derivative = sp.diff(rhs, input_sym)
        return float(derivative.subs(input_sym, point))
    except (ValueError, TypeError):
        return None


# ── ODE layer (Phase 7 Stage 2) ─────────────────────────────────────────
#
# Syntax: `dvar/dt = rhs` or `coef * dvar/dt = rhs`. Higher-order
# derivatives (d2y/dt2) are intentionally rejected — they decompose into
# a system of first-order ODEs, which is a Stage-3 concern.
#
# The parser:
#   1. splits on `=`
#   2. matches the LHS against `[coef *] dvar/dt`
#   3. divides the parsed RHS by the coefficient so the canonical form
#      `dvar/dt = (rhs / coef)` falls out
#
# Downstream consumers (codegen Euler step, fixed-point solver, Jacobian
# stability) treat the normalized rhs uniformly.

import re

@dataclass
class ODEExpr:
    """Parsed ODE: `dvar/dt = rhs` after normalization.

    rhs is post-normalized — the original coefficient (if any) has
    already been divided out, so callers always see canonical form.
    """
    source: str
    state_var: str                  # e.g. "V"
    rhs: sp.Expr                    # canonical: dvar/dt = rhs
    coefficient: Optional[str]      # original coef expression, or None if 1
    free_symbols: Set[str] = field(default_factory=set)

    def __str__(self) -> str:
        return f"d{self.state_var}/dt = {self.rhs}"


# `d<var>/dt` on LHS; everything before it is the optional coefficient
_ODE_LHS_RE = re.compile(
    r'^\s*(?P<coef>.*?)\s*d(?P<var>[A-Za-z_]\w*)\s*/\s*dt\s*$'
)
# Higher-order derivatives — `d2V/dt2` style. Reject explicitly.
_HIGHER_ORDER_RE = re.compile(r'd\d+[A-Za-z_]\w*\s*/\s*dt\d+')


def parse_ode(source: str) -> ODEExpr:
    """Parse `[coef *] dvar/dt = rhs` into an `ODEExpr`.

    Raises ValueError on malformed input or higher-order derivatives.
    """
    if not source or "=" not in source:
        raise ValueError(f"ODE must contain '=': {source!r}")

    lhs_str, _, rhs_str = source.partition("=")
    lhs_str, rhs_str = lhs_str.strip(), rhs_str.strip()

    if _HIGHER_ORDER_RE.search(lhs_str):
        raise ValueError(
            f"higher-order ODEs not supported (decompose to a first-order "
            f"system instead): {lhs_str!r}"
        )

    m = _ODE_LHS_RE.match(lhs_str)
    if not m:
        raise ValueError(
            f"could not parse ODE LHS {lhs_str!r}; "
            f"expected `dvar/dt` or `coef * dvar/dt`"
        )

    coef_str = m.group("coef").strip()
    state_var = m.group("var")

    # `coef * dvar/dt` — the trailing `*` belongs to the coefficient
    if coef_str.endswith("*"):
        coef_str = coef_str[:-1].strip()
    if not coef_str:
        coef_str = "1"

    rhs_rewritten = _rewrite_matmul(rhs_str)
    try:
        rhs_expr = sp.sympify(rhs_rewritten, locals=_LOCAL_DICT)
    except (sp.SympifyError, SyntaxError, TypeError) as e:
        raise ValueError(f"could not parse ODE RHS {rhs_str!r}: {e}") from e

    if coef_str != "1":
        try:
            coef_expr = sp.sympify(coef_str, locals=_LOCAL_DICT)
        except (sp.SympifyError, SyntaxError, TypeError) as e:
            raise ValueError(f"could not parse ODE coefficient {coef_str!r}: {e}") from e
        rhs_expr = rhs_expr / coef_expr

    free = {str(s) for s in rhs_expr.free_symbols}

    return ODEExpr(
        source=source,
        state_var=state_var,
        rhs=rhs_expr,
        coefficient=coef_str if coef_str != "1" else None,
        free_symbols=free,
    )


def ode_for_population(ode: Optional[str],
                       dynamics: Optional[str]) -> Optional[ODEExpr]:
    """Resolve a population's ODE the way `equation_for_population` does."""
    if ode:
        return parse_ode(ode)
    if dynamics:
        decl = DYNAMICS_DECLS.get(dynamics)
        if decl is not None and decl.is_ode:
            return parse_ode(decl.ode)
    return None


def ode_fixed_point(ode: ODEExpr,
                    param_bindings: Optional[Dict[str, float]] = None,
                    guess: float = 0.1) -> Optional[float]:
    """Find V* such that dV/dt = 0 at V = V*.

    For `dV/dt = (-V + I) / tau`, the fixed point is V* = I. The solver
    substitutes all named params, then numerically solves `rhs == 0`.
    """
    param_bindings = param_bindings or {}
    rhs = _expand_for_analysis(ode.rhs)
    for name, val in param_bindings.items():
        rhs = rhs.subs(sp.Symbol(name), val)

    state_sym = sp.Symbol(ode.state_var)
    try:
        sol = sp.nsolve(rhs, state_sym, guess)
        # nsolve can return complex roots even for real-valued problems.
        # Accept only solutions with negligible imaginary part.
        if hasattr(sol, "is_complex") and sol.is_complex:
            if abs(complex(sol).imag) > 1e-9:
                return None
            return float(complex(sol).real)
        return float(sol)
    except (ValueError, ZeroDivisionError, sp.PolynomialError, TypeError):
        return None


def ode_stable_at(ode: ODEExpr,
                  point: float,
                  param_bindings: Optional[Dict[str, float]] = None) -> Optional[bool]:
    """Linearized stability of the ODE at `point`.

    Stable iff Re(d rhs / d state) < 0 at the equilibrium — i.e. the
    linearized system contracts under time evolution. None on failure.
    """
    param_bindings = param_bindings or {}
    rhs = _expand_for_analysis(ode.rhs)
    for name, val in param_bindings.items():
        rhs = rhs.subs(sp.Symbol(name), val)

    state_sym = sp.Symbol(ode.state_var)
    try:
        derivative = sp.diff(rhs, state_sym)
        slope = float(derivative.subs(state_sym, point))
        return slope < 0
    except (ValueError, TypeError):
        return None
