# -*- coding: utf-8 -*-
"""Op registry — DSL-driven dispatch for HypergraphExecutor.

This module is the single point of truth that maps an edge's or node's
``attrs`` dict (lifted from the ``.neuro`` DSL) to the actual PyTorch
operation that runs at every forward pass.  The hypergraph IR carries
**what** every element does; this registry says **how** it runs.

Dispatch rules
--------------
For **edges** (``resolve_edge_op``), attrs are inspected in priority order:

  1. ``feature``      — explicit feature-flag op (e.g. ``hyperbolic_attention.edge``).
                        Feature attrs take priority over equation attrs so a
                        feature-flagged mechanism can override the default
                        equation for the same topological slot.
  2. ``equation``     — canonical equation reference (e.g. ``@standard_synapse``).
                        Strips the leading ``@`` and looks up in the equation
                        registry.
  3. fallback         — ``StandardSynapseOp`` (weight=1.0 linear projection).
                        Used when both ``feature`` and ``equation`` are absent
                        or unknown.

For **nodes** (``resolve_node_op``):

  * ``act`` attr      — one of ``"relu"`` (default), ``"silu"``, ``"tanh"``.
                        Unknown → ``relu``.

The three canonical equations (``@standard_synapse``,
``@multiplicative_modulation``, ``@additive_modulation``) and every
feature-flagged op live in pure data structures: ``_EDGE_REGISTRY`` for
equations, ``_FEATURE_REGISTRY`` for feature flags.  Adding a new op is
a one-line registration call — no executor changes needed.

This file is the dispatch layer; the actual math lives in
:mod:`neuroslm.modules` (for parametric feature ops) and inline below
(for the three non-parametric base ops).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# Op base classes
# ──────────────────────────────────────────────────────────────────────

class EdgeOp:
    """Abstract dispatch unit for a hyperedge.

    Subclasses opt into two flavours of behaviour:

    * **Synapse-style ops** (parametric or not) — implement
      :meth:`build_module` to provide the ``nn.Module`` that gets stored
      in ``HypergraphExecutor.edge_projections`` and :meth:`forward` to
      compute the edge's contribution to the downstream node's incoming
      sum.

    * **Modulation-style ops** (always non-parametric) — return ``None``
      from :meth:`build_module` (so the edge does NOT appear in
      ``edge_projections``) and implement :meth:`apply_modulation` to
      transform the downstream node's output in-place after the node
      activation has been applied.

    The two flavours are dispatched by the executor based on
    ``edge.kind``: ``"synapse"`` calls :meth:`forward`, ``"modulation"``
    calls :meth:`apply_modulation`.  An op may implement both if it
    semantically fits either role.
    """

    is_modulation: bool = False  # subclasses set True for modulation ops

    def build_module(self, d_model: int) -> Optional[nn.Module]:
        """Create the ``nn.Module`` for this edge, or return ``None``.

        ``None`` signals the edge has no parametric state and the
        executor must NOT add it to ``edge_projections``.
        """
        return None

    def forward(
        self,
        src_state: torch.Tensor,
        module: Optional[nn.Module],
        attrs: Dict[str, str],
    ) -> torch.Tensor:
        """Compute this edge's contribution to dst's incoming sum.

        ``src_state`` is the upstream population's most recent activation,
        shape ``(B, d_model)``.  ``module`` is whatever this op's
        :meth:`build_module` returned (``None`` for non-parametric ops).
        Returns a tensor of shape ``(B, d_model)``.
        """
        raise NotImplementedError(f"{type(self).__name__}.forward not implemented")

    def apply_modulation(
        self,
        node_out: torch.Tensor,
        attrs: Dict[str, str],
        nt_level: float,
    ) -> torch.Tensor:
        """For modulation ops: transform ``node_out`` and return new value.

        Default is identity — synapse-style ops simply don't override this
        and the executor skips the call.
        """
        return node_out


class NodeOp:
    """Activation dispatch unit for a population node.

    Currently a thin wrapper around a torch activation function; kept as
    a class for symmetry with :class:`EdgeOp` and so future activations
    can carry parameters (e.g. PReLU) without changing the executor.
    """

    name: str = "relu"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)


# ──────────────────────────────────────────────────────────────────────
# The three canonical equations
# ──────────────────────────────────────────────────────────────────────

class StandardSynapseOp(EdgeOp):
    """``@standard_synapse`` — weight × Linear(d_model, d_model) projection.

    Equation: ``y = weight × (W · x + b)``.

    Attrs:
        weight (str, default "1.0"): scalar multiplier applied to the
            projection output.  ``weight="0"`` disconnects the edge
            (output is identically zero, identical to the "no edge"
            baseline up to numerical noise).
    """

    def build_module(self, d_model: int) -> nn.Linear:
        return nn.Linear(d_model, d_model)

    def forward(self, src_state, module, attrs):
        weight = _safe_float(attrs.get("weight"), default=1.0)
        # module is the nn.Linear created by build_module — never None for synapse ops
        out = module(src_state) if module is not None else src_state
        return weight * out


class MultiplicativeModulationOp(EdgeOp):
    """``@multiplicative_modulation`` — node_out × (gain × nt_level).

    Equation: ``y = x × (gain × c)`` where ``c`` is the neurotransmitter
    level. When ``c`` is absent from ``nt_levels`` the modulation is
    skipped entirely (identity on ``node_out``).

    Attrs:
        gain (str, default "1.0"): scalar coefficient applied to ``nt_level``.
        effect (str, default "multiplicative"): redundant tag; kept for
            backward-compat with the legacy DSL syntax.
    """

    is_modulation = True

    def build_module(self, d_model):
        return None  # non-parametric

    def apply_modulation(self, node_out, attrs, nt_level):
        gain = _safe_float(attrs.get("gain"), default=1.0)
        return node_out * (gain * nt_level)


class AdditiveModulationOp(EdgeOp):
    """``@additive_modulation`` — node_out + (gain × nt_level).

    Equation: ``y = x + (gain × c)``.  ``gain × c`` is a scalar that
    broadcasts across every element of ``node_out``.  Skipped when
    ``c`` is absent from ``nt_levels``.

    Attrs:
        gain (str, default "1.0"): scalar coefficient applied to ``nt_level``.
        effect (str, default "additive"): redundant tag.
    """

    is_modulation = True

    def build_module(self, d_model):
        return None

    def apply_modulation(self, node_out, attrs, nt_level):
        gain = _safe_float(attrs.get("gain"), default=1.0)
        return node_out + (gain * nt_level)


# ──────────────────────────────────────────────────────────────────────
# Feature-flagged ops (each defined in a single .neuro file)
# ──────────────────────────────────────────────────────────────────────

class _FeatureSeqWrapperOp(EdgeOp):
    """Common scaffold for feature ops whose module expects ``(B, T, D)``.

    The base executor passes ``(B, D)`` tensors, so we unsqueeze to add a
    pseudo-time axis of length 1, run the module, and squeeze it back
    out.  ``weight`` from the edge attrs scales the result, identical to
    :class:`StandardSynapseOp`.
    """

    def forward(self, src_state, module, attrs):
        weight = _safe_float(attrs.get("weight"), default=1.0)
        if module is None:
            return weight * src_state
        if src_state.dim() == 2:
            x = src_state.unsqueeze(1)               # (B, D) → (B, 1, D)
            out = module(x)
            if isinstance(out, tuple):
                out = out[0]
            out = out.squeeze(1)                     # (B, 1, D) → (B, D)
        else:
            out = module(src_state)
            if isinstance(out, tuple):
                out = out[0]
        return weight * out


class HyperbolicAttentionOp(_FeatureSeqWrapperOp):
    """Feature: ``hyperbolic_attention.edge`` — Poincaré-disc multi-head attn."""

    def build_module(self, d_model: int) -> nn.Module:
        from neuroslm.modules.hyperbolic_attention import HyperbolicMultiHeadAttention
        n_heads = _pick_n_heads(d_model)
        return HyperbolicMultiHeadAttention(d_model=d_model, n_heads=n_heads)


class RopeTorusOp(_FeatureSeqWrapperOp):
    """Feature: ``rope_torus.edge`` — torus-rotated positional encoding."""

    def build_module(self, d_model: int) -> nn.Module:
        from neuroslm.modules.rope_torus import RoPETorus
        if d_model % 2 != 0:
            # RoPETorus requires even d_model. Fall back to a Linear to keep
            # the executor differentiable; documented behaviour.
            return nn.Linear(d_model, d_model)
        return RoPETorus(d_model=d_model, max_seq_len=8, learnable_periods=True)


class HRRMemoryOp(_FeatureSeqWrapperOp):
    """Feature: ``hrr_memory.edge`` — Holographic Reduced Representation memory."""

    def build_module(self, d_model: int) -> nn.Module:
        from neuroslm.modules.hrr_memory import HRRMemory
        # d_memory keyed to d_model so the test-time shapes line up cleanly
        return HRRMemory(d_model=d_model, d_memory=max(64, d_model))


class PredictiveCodingResidualOp(_FeatureSeqWrapperOp):
    """Feature: ``predictive_coding_residual.edge`` — PC residual correction."""

    def build_module(self, d_model: int) -> nn.Module:
        from neuroslm.modules.predictive_coding_residual import PredictiveCodingResidual
        return PredictiveCodingResidual(d_model=d_model)


class SurpriseGatedMoEOp(_FeatureSeqWrapperOp):
    """Feature: ``surprise_gated_moe.edge`` — surprise-modulated mixture of experts."""

    def build_module(self, d_model: int) -> nn.Module:
        from neuroslm.modules.surprise_gated_moe import SurpriseGatedMoE
        return SurpriseGatedMoE(
            d_model=d_model,
            n_experts=4,
            d_hidden=2 * d_model,
            k_min=1,
            k_max=2,
        )


# ──────────────────────────────────────────────────────────────────────
# Node activation ops
# ──────────────────────────────────────────────────────────────────────

class ReLUNodeOp(NodeOp):
    name = "relu"

    def __call__(self, x):
        return F.relu(x)


class SiLUNodeOp(NodeOp):
    name = "silu"

    def __call__(self, x):
        return F.silu(x)


class TanhNodeOp(NodeOp):
    name = "tanh"

    def __call__(self, x):
        return torch.tanh(x)


# ──────────────────────────────────────────────────────────────────────
# Registries — declarative dispatch tables
# ──────────────────────────────────────────────────────────────────────

# Equation name (without the leading "@") → EdgeOp factory.
# To add a new equation, register here AND add the matching .neuro def
# in lib/equations.neuro (or lib/features/<name>.neuro for features).
_EDGE_REGISTRY: Dict[str, type] = {
    "standard_synapse":         StandardSynapseOp,
    "multiplicative_modulation": MultiplicativeModulationOp,
    "additive_modulation":       AdditiveModulationOp,
}

# Feature flag name → EdgeOp factory.  Feature attrs look like
# "<name>.edge" or "<name>.node"; only the "<name>" prefix is consulted.
_FEATURE_REGISTRY: Dict[str, type] = {
    "hyperbolic_attention":         HyperbolicAttentionOp,
    "rope_torus":                   RopeTorusOp,
    "hrr_memory":                   HRRMemoryOp,
    "predictive_coding_residual":   PredictiveCodingResidualOp,
    "surprise_gated_moe":           SurpriseGatedMoEOp,
}

# Activation name → NodeOp factory.
_NODE_REGISTRY: Dict[str, type] = {
    "relu": ReLUNodeOp,
    "silu": SiLUNodeOp,
    "tanh": TanhNodeOp,
}


# ──────────────────────────────────────────────────────────────────────
# Resolver entry points
# ──────────────────────────────────────────────────────────────────────

def resolve_edge_op(
    attrs: Dict[str, str],
    kind: str = "synapse",
) -> EdgeOp:
    """Return the :class:`EdgeOp` instance for this edge's attrs.

    Priority order:
      1. ``attrs["feature"]`` — split on '.' and use the first component
         to look up ``_FEATURE_REGISTRY``.  Wins over ``equation``.
      2. ``attrs["equation"]`` — strip a leading ``@`` and look up in
         ``_EDGE_REGISTRY``.
      3. Default — :class:`StandardSynapseOp` (or one of the modulation
         ops if *kind* implies it).

    *kind* nudges the default when both ``feature`` and ``equation``
    are missing: ``kind="modulation"`` defaults to multiplicative (the
    historical legacy behaviour) so modulation edges without an
    explicit equation don't silently become linear projections.
    """
    feature = attrs.get("feature") or ""
    if feature:
        feature_name = feature.split(".", 1)[0]
        cls = _FEATURE_REGISTRY.get(feature_name)
        if cls is not None:
            return cls()

    equation = (attrs.get("equation") or "").lstrip("@")
    if equation:
        cls = _EDGE_REGISTRY.get(equation)
        if cls is not None:
            return cls()
        # Unknown equation → fall through to default.  Documented and
        # tested (OD-15): we never crash on a typo'd equation reference.

    if kind == "modulation":
        return MultiplicativeModulationOp()
    return StandardSynapseOp()


def resolve_node_op(attrs: Dict[str, str]) -> NodeOp:
    """Return the :class:`NodeOp` instance for this node's attrs.

    Reads ``attrs["act"]`` (default ``"relu"``).  Unknown values fall
    back to ReLU silently — the executor must never crash on a typo.
    """
    act = (attrs.get("act") or "relu").lower()
    cls = _NODE_REGISTRY.get(act, ReLUNodeOp)
    return cls()


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _safe_float(value: Any, *, default: float) -> float:
    """Coerce *value* to float; return *default* on None or parse failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pick_n_heads(d_model: int) -> int:
    """Pick a reasonable n_heads divisor for *d_model* (powers of 2 first)."""
    for n in (8, 4, 2, 1):
        if d_model % n == 0:
            return n
    return 1


__all__ = [
    "EdgeOp",
    "NodeOp",
    "StandardSynapseOp",
    "MultiplicativeModulationOp",
    "AdditiveModulationOp",
    "HyperbolicAttentionOp",
    "RopeTorusOp",
    "HRRMemoryOp",
    "PredictiveCodingResidualOp",
    "SurpriseGatedMoEOp",
    "ReLUNodeOp",
    "SiLUNodeOp",
    "TanhNodeOp",
    "resolve_edge_op",
    "resolve_node_op",
    "_EDGE_REGISTRY",
    "_FEATURE_REGISTRY",
    "_NODE_REGISTRY",
]
