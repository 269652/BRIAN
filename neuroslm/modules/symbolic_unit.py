# -*- coding: utf-8 -*-
"""Symbolic Hyper-Neuron — discoverable mathematical equations as a layer.

This module provides the **mathematical-invention primitive** of the
Multi-Objective-Fitness work order:

    "Spezialisierte Hyper-Neuronen drücken ihre interne Logik in Form
     von expliziten mathematischen Gleichungen aus.  Ein interner
     AlphaEvolve-Loop könnte neue Operatoren entdecken."

A `SymbolicHyperNeuron` is a small differentiable layer that, per output
unit, selects exactly:

    1.  an operator from a fixed bank   (e.g. `add`, `mul`, `exp`, `sin`)
    2.  an input feature for slot A      (e.g. `phi`)
    3.  an input feature for slot B      (e.g. `metabolic_demand`)

All three selections use Gumbel-Softmax with annealable temperature
``tau``.  At high ``tau`` the layer behaves like a soft mixture
(allowing gradient to flow through every operator and feature); at low
``tau`` it hardens to a one-hot choice and the unit becomes a single
extractable expression, e.g. ``"exp(phi)"`` or ``"phi * metabolic_demand"``.

Mathematical specification
--------------------------
Let :math:`\\mathbf{x} \\in \\mathbb{R}^{n_F}` be the input feature vector,
:math:`\\pi^{a}_i,\\,\\pi^{b}_i \\in \\Delta^{n_F-1}` the Gumbel-softmax
selections for slots A and B of unit :math:`i`, and
:math:`\\pi^{op}_i \\in \\Delta^{n_O-1}` the operator selection.  Then

.. math::
    x^a_i = \\sum_{j} \\pi^{a}_{i,j}\\, x_j, \\qquad
    x^b_i = \\sum_{j} \\pi^{b}_{i,j}\\, x_j, \\qquad
    y_i = \\sum_{o} \\pi^{op}_{i,o}\\, \\mathrm{op}_o(x^a_i, x^b_i).

Sparsity regulariser
--------------------
The entropy of the three softmax distributions is exposed as
``sparsity_loss``.  Driving it to zero via an auxiliary loss term
hardens the selections into one-hot ⇒ the layer becomes a discrete
formula.  This is the contract the optimisation loop uses to discover
*specific* mathematical structures rather than soft blends.

Why a fixed operator bank
-------------------------
The bank is intentionally small (≤ 8 ops) so that

    * the Gumbel-Softmax search space is tractable,
    * every op is numerically stable (`exp` is clamped),
    * extracted strings remain human-readable.

To explore a larger op space, a curriculum can swap banks at runtime
(`unit.operator_bank = OperatorBank(...)`) — typically widening from
`{id, add, mul}` to the full bank as training matures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# Operator bank
# ──────────────────────────────────────────────────────────────────────

# Per-op symbol used for human-readable extraction.  Falls back to the
# bare name when the symbol is None (e.g. `exp(a)`).
_OP_SYMBOLS = {
    "identity": ("id", "{a}"),                  # symbol-fmt, expression-fmt
    "add":      ("+",  "({a} + {b})"),
    "sub":      ("-",  "({a} - {b})"),
    "mul":      ("*",  "({a} * {b})"),
    "exp":      ("exp", "exp({a})"),
    "sin":      ("sin", "sin({a})"),
    "tanh":     ("tanh", "tanh({a})"),
    "neg":      ("-",  "(-{a})"),
}


# Clamp protects `exp` from overflow.  20 ⇒ exp(20) ≈ 4.85e8, well
# inside fp32 dynamic range; -20 protects `exp(-x)` from underflow to
# subnormals that would create NaN gradients downstream.
_EXP_CLAMP = 20.0


def _safe_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:    # noqa: ARG001
    return torch.exp(a.clamp(-_EXP_CLAMP, _EXP_CLAMP))


def _identity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:    # noqa: ARG001
    return a


def _add(a, b):    return a + b
def _sub(a, b):    return a - b
def _mul(a, b):    return a * b
def _sin(a, b):    return torch.sin(a)    # noqa: ARG001
def _tanh(a, b):   return torch.tanh(a)   # noqa: ARG001
def _neg(a, b):    return -a              # noqa: ARG001


@dataclass
class OperatorBank:
    """A finite set of binary differentiable operators.

    Each entry in ``ops`` must accept two ``(N,)`` tensors and return a
    same-shape tensor.  Unary ops ignore the second argument.

    Use ``OperatorBank.default()`` for the standard
    ``{identity, add, sub, mul, exp, sin, tanh}`` bank.
    """
    names: List[str]
    ops: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.ops):
            raise ValueError(
                f"OperatorBank: names ({len(self.names)}) and "
                f"ops ({len(self.ops)}) must have the same length"
            )
        if len(self.names) == 0:
            raise ValueError("OperatorBank must contain at least one op")

    @property
    def n_ops(self) -> int:
        return len(self.names)

    @classmethod
    def default(cls) -> "OperatorBank":
        """The standard bank.  Mix of linear, multiplicative and
        non-linear ops so the symbolic unit can discover both
        polynomial and transcendental relationships."""
        return cls(
            names=["identity", "add", "sub", "mul", "exp", "sin", "tanh"],
            ops=[_identity, _add, _sub, _mul, _safe_exp, _sin, _tanh],
        )

    def apply_all(
        self, x_a: torch.Tensor, x_b: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate every op at every position and stack on the last axis.

        Parameters
        ----------
        x_a, x_b : torch.Tensor, same shape (..., N)

        Returns
        -------
        torch.Tensor with shape ``(..., N, n_ops)``
        """
        if x_a.shape != x_b.shape:
            raise ValueError(
                f"OperatorBank.apply_all: x_a {tuple(x_a.shape)} and "
                f"x_b {tuple(x_b.shape)} must have the same shape"
            )
        outs = [op(x_a, x_b) for op in self.ops]
        return torch.stack(outs, dim=-1)


# ──────────────────────────────────────────────────────────────────────
# Symbolic hyper-neuron layer
# ──────────────────────────────────────────────────────────────────────

class SymbolicHyperNeuron(nn.Module):
    """Layer of `n_units` learnable, extractable symbolic formulas.

    Parameters
    ----------
    n_units : int
        Number of symbolic units (output dimensionality).
    n_features : int
        Input feature dimensionality.  Each unit picks two features
        from this set as the slots A and B of its operator.
    operator_bank : OperatorBank, optional
        Bank of binary operators.  Defaults to ``OperatorBank.default()``.
    tau : float, optional
        Initial Gumbel-Softmax temperature.  High (≥ 1.0) ⇒ soft
        mixture (exploration); low (≤ 0.1) ⇒ near-discrete selection
        (extractable formulas).  Default 1.0.
    feature_names : Sequence[str], optional
        Names used by ``expression_strings``.  Defaults to ``x0..x{n-1}``.

    Forward
    -------
    Input  : ``x : torch.Tensor`` of shape ``(..., n_features)``
    Output : ``y : torch.Tensor`` of shape ``(..., n_units)``

    The forward is fully differentiable in both inputs *and* the
    per-unit selection logits, so the LM-loss gradient can shape what
    each unit discovers.
    """

    def __init__(
        self,
        n_units: int,
        n_features: int,
        operator_bank: Optional[OperatorBank] = None,
        tau: float = 1.0,
        feature_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        if n_units <= 0:
            raise ValueError(f"n_units must be > 0, got {n_units}")
        if n_features <= 0:
            raise ValueError(f"n_features must be > 0, got {n_features}")
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")

        self.n_units = n_units
        self.n_features = n_features
        self.operator_bank = operator_bank or OperatorBank.default()
        self.tau = float(tau)

        if feature_names is None:
            feature_names = [f"x{i}" for i in range(n_features)]
        if len(feature_names) != n_features:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) must match "
                f"n_features ({n_features})"
            )
        self.feature_names: List[str] = list(feature_names)

        # Per-unit selection logits.  Initialised with small noise so the
        # initial softmax is approximately uniform but symmetry is broken
        # — without the noise every unit would collapse to the same
        # operator under any sharpening pressure.
        self.input_a_logits = nn.Parameter(
            0.01 * torch.randn(n_units, n_features)
        )
        self.input_b_logits = nn.Parameter(
            0.01 * torch.randn(n_units, n_features)
        )
        self.op_logits = nn.Parameter(
            0.01 * torch.randn(n_units, self.operator_bank.n_ops)
        )

    # ── temperature control ─────────────────────────────────────────

    def set_tau(self, tau: float) -> None:
        """Anneal the Gumbel-Softmax temperature in-place."""
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        self.tau = float(tau)

    # ── selection sampling ──────────────────────────────────────────

    def _sample_selection(self, logits: torch.Tensor) -> torch.Tensor:
        """Return a (possibly stochastic) softmax selection.

        * In training mode, ``F.gumbel_softmax`` adds Gumbel noise so
          the optimiser can explore different operators / inputs.
        * In eval mode, a pure ``softmax(logits / tau)`` is used — no
          randomness, so forward passes are reproducible.
        """
        if self.training:
            # `hard=False` keeps the relaxation differentiable; we do
            # straight-through hardening *only* on request via the
            # `expression_strings` extractor.
            return F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1)
        return F.softmax(logits / self.tau, dim=-1)

    # ── forward ─────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(..., n_features) -> (..., n_units)``."""
        if x.shape[-1] != self.n_features:
            raise ValueError(
                f"SymbolicHyperNeuron: input has {x.shape[-1]} features, "
                f"expected {self.n_features}"
            )

        # Selections: (n_units, n_features) and (n_units, n_ops).
        sel_a = self._sample_selection(self.input_a_logits)   # (U, F)
        sel_b = self._sample_selection(self.input_b_logits)   # (U, F)
        sel_op = self._sample_selection(self.op_logits)       # (U, O)

        # Pre-flatten the leading dims so we can broadcast cleanly.
        leading_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.n_features)               # (N, F)

        # Broadcast inputs through the per-unit feature mixers.
        # Result: (N, U).
        x_a = x_flat @ sel_a.t()                              # (N, U)
        x_b = x_flat @ sel_b.t()                              # (N, U)

        # Apply every operator at every (N, U) position.  Shape: (N, U, O).
        all_ops = self.operator_bank.apply_all(x_a, x_b)

        # Mix the per-op outputs by the per-unit operator selection.
        # sel_op: (U, O) → broadcast across N.
        y_flat = (all_ops * sel_op.unsqueeze(0)).sum(dim=-1)  # (N, U)

        return y_flat.reshape(*leading_shape, self.n_units)

    # ── regularisation ──────────────────────────────────────────────

    @staticmethod
    def _entropy(logits: torch.Tensor) -> torch.Tensor:
        """Shannon entropy of softmax(logits) along the last axis."""
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        return -(probs * log_probs).sum(dim=-1)

    def sparsity_loss(self) -> torch.Tensor:
        """Entropy of all three selection distributions, averaged over units.

        Mathematically:

        .. math::
            \\mathcal{L}_\\text{sparsity} = \\frac{1}{3 n_U} \\sum_i
                \\bigl[ H(\\pi^{a}_i) + H(\\pi^{b}_i) + H(\\pi^{op}_i)\\bigr].

        Driven toward zero by an auxiliary loss term, this collapses
        each unit to a one-hot operator + one-hot input pair — i.e.
        a discrete formula — without forcing the choice.
        """
        h_a = self._entropy(self.input_a_logits).mean()
        h_b = self._entropy(self.input_b_logits).mean()
        h_o = self._entropy(self.op_logits).mean()
        return (h_a + h_b + h_o) / 3.0

    # ── expression extraction ───────────────────────────────────────

    def _argmax_selections(self) -> tuple:
        """Hard one-hot choice per unit (uses argmax, ignores tau)."""
        idx_a = self.input_a_logits.argmax(dim=-1).tolist()
        idx_b = self.input_b_logits.argmax(dim=-1).tolist()
        idx_o = self.op_logits.argmax(dim=-1).tolist()
        return idx_a, idx_b, idx_o

    def expression_strings(self) -> List[str]:
        """Return one printable formula per unit.

        Examples
        --------
        ``["(phi * surprise)", "exp(metabolic_demand)", "(x0 + x2)"]``
        """
        idx_a, idx_b, idx_o = self._argmax_selections()
        out: List[str] = []
        for u in range(self.n_units):
            a_name = self.feature_names[idx_a[u]]
            b_name = self.feature_names[idx_b[u]]
            op_name = self.operator_bank.names[idx_o[u]]
            # Use the canonical expression format if known, else fall
            # back to a name(args) form.
            if op_name in _OP_SYMBOLS:
                _, fmt = _OP_SYMBOLS[op_name]
                out.append(fmt.format(a=a_name, b=b_name))
            else:
                out.append(f"{op_name}({a_name}, {b_name})")
        return out

    # ── repr ────────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"n_units={self.n_units}, n_features={self.n_features}, "
            f"n_ops={self.operator_bank.n_ops}, tau={self.tau}"
        )
