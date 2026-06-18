# -*- coding: utf-8 -*-
"""HypergraphExecutor — runtime tensor routing through the hypergraph IR.

The compiler's traditional path is:
  DSL → HypergraphIR → CodeGenerator → exec() → static nn.Module

This module provides the alternative:
  DSL → HypergraphIR → HypergraphExecutor (live nn.Module)

The IR persists at runtime. Every operation that runs is dispatched from
:mod:`neuroslm.compiler.op_registry` based on each element's ``attrs``
dict — nothing is hardcoded in the executor.  The hypergraph IS the
program; PyTorch IS the interpreter.

Edge dispatch chain (in priority order):
  1. ``edge.attrs["feature"]`` → ``_FEATURE_REGISTRY`` (overrides equation)
  2. ``edge.attrs["equation"]`` → ``_EDGE_REGISTRY`` (strip leading ``@``)
  3. Fallback to ``StandardSynapseOp`` (linear projection, weight=1.0)

Node dispatch:
  ``node.attrs["act"]`` → ``relu`` | ``silu`` | ``tanh`` (default ``relu``)

Modulation edges (``edge.kind == "modulation"``) are NEVER added to
``edge_projections`` (they are non-parametric).  They run AFTER the
node activation has been applied, transforming the node output in
place when the corresponding NT level is present in the forward call's
``nt_levels`` dict.

This is the foundation for trainable graph structure: because the
routing is differentiable, gradient information about which edges
matter is already present in the computation graph. Growing or
pruning edges becomes a matter of adding/removing nn.Module parameters
and re-running the traversal.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroslm.compiler.hypergraph_ir import HypergraphIR, HyperEdge, HyperNode
from neuroslm.compiler.op_registry import (
    EdgeOp,
    NodeOp,
    resolve_edge_op,
    resolve_node_op,
)


class HypergraphExecutor(nn.Module):
    """Execute a HypergraphIR as a live, differentiable nn.Module.

    Each population node gets an ``nn.Linear(d_model, d_model)`` plus a
    NodeOp-selected activation. Each synapse edge gets an
    :class:`EdgeOp`-defined nn.Module projection. Each modulation edge
    gets a non-parametric EdgeOp that runs after the node activation.

    Forward pass:
      1. Topological traversal over synapse edges.
      2. Each node sums: ``node_layer(x)`` plus every upstream synapse's
         ``EdgeOp.forward(src_state, projection, attrs)`` contribution.
      3. Apply node activation (``NodeOp(__call__)``).
      4. Apply every modulation edge whose NT is present in
         ``nt_levels`` via ``EdgeOp.apply_modulation(node_out, attrs, nt)``.
      5. Store the result in ``states[name]``.

    Returns a dict ``{population_name: tensor (B, d_model)}``.
    """

    def __init__(self, ir: HypergraphIR, d_model: int) -> None:
        super().__init__()
        self._ir = ir
        self.d_model = d_model

        pop_nodes = [n for n in ir.nodes if n.kind == "population"]
        all_edges = list(ir.hyperedges)
        syn_edges = [e for e in all_edges if e.kind == "synapse"]
        mod_edges = [e for e in all_edges if e.kind == "modulation"]

        # ── Per-node activation ops (NodeOp instances) ─────────────────
        self._node_ops: Dict[str, NodeOp] = {
            n.name: resolve_node_op(n.attrs) for n in pop_nodes
        }

        # ── Per-node Linear transformations ────────────────────────────
        self.node_layers: nn.ModuleDict = nn.ModuleDict({
            self._safe_key(n.name): nn.Linear(d_model, d_model)
            for n in pop_nodes
        })

        # ── Per-edge ops + parametric modules ──────────────────────────
        # _edge_ops: edge.id → EdgeOp instance (one per IR edge)
        # edge_projections: nn.ModuleDict of EdgeOp.build_module() results,
        #                   keyed by safe edge id.  Modulation ops return
        #                   None from build_module → NOT added here.
        self._edge_ops: Dict[str, EdgeOp] = {}
        edge_modules: Dict[str, nn.Module] = {}
        for e in syn_edges:
            op = resolve_edge_op(e.attrs, kind=e.kind)
            self._edge_ops[e.id] = op
            module = op.build_module(d_model)
            if module is not None:
                edge_modules[self._safe_key(e.id)] = module
        for e in mod_edges:
            op = resolve_edge_op(e.attrs, kind=e.kind)
            self._edge_ops[e.id] = op
            # Modulation ops MUST return None from build_module — if they
            # don't (custom user op), we still add them to be safe, but
            # the canonical AdditiveModulationOp / MultiplicativeModulationOp
            # return None as documented.
            module = op.build_module(d_model)
            if module is not None:
                edge_modules[self._safe_key(e.id)] = module

        self.edge_projections: nn.ModuleDict = nn.ModuleDict(edge_modules)

        # Cache for the forward pass — avoid re-computing each call
        self._pop_names: List[str] = [n.name for n in pop_nodes]
        self._syn_edges: List[HyperEdge] = syn_edges
        self._mod_edges: List[HyperEdge] = mod_edges
        self._topo_order: List[str] = self._topological_sort()

        # Per-element activation RMS from the most recent forward pass.
        # Keyed by IR element ID: "population:{name}" and "{edge.id}".
        # Updated in-place on every forward; empty until the first call.
        self._last_act_norms: Dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        nt_levels: Dict[str, float] | None = None,
        **kwargs,  # absorbs harness kwargs we don't use
    ) -> Dict[str, torch.Tensor]:
        """Route x through the hypergraph and return per-population outputs.

        Args:
            x: Input tensor of shape ``(B, d_model)``.
            nt_levels: Optional dict mapping neurotransmitter name → scalar
                level; consulted by modulation edges.  ``None`` or
                missing key → modulation edge is skipped (identity).

        Returns:
            Dict mapping population name → output tensor ``(B, d_model)``.
        """
        states: Dict[str, torch.Tensor] = {}
        act_norms: Dict[str, float] = {}
        nt_levels = nt_levels or {}

        for name in self._topo_order:
            key = self._safe_key(name)
            node_layer = self.node_layers[key]
            node_op = self._node_ops.get(name)
            if node_op is None:
                # Population added after construction (shouldn't happen)
                # — fall back to ReLU to keep the graph differentiable.
                from neuroslm.compiler.op_registry import ReLUNodeOp
                node_op = ReLUNodeOp()

            # Always include the node's own transformation of the raw input.
            # This ensures source nodes (no in-edges) are not identity ops.
            incoming = [node_layer(x)]

            # Add a projected contribution from each upstream neighbour,
            # recording the RMS activation of each edge's projection.
            for edge in self._syn_edges:
                if edge.members[1] != name:
                    continue
                src_name = edge.members[0]
                src_state = states.get(src_name, x)
                op = self._edge_ops[edge.id]
                proj_key = self._safe_key(edge.id)
                # nn.ModuleDict has no .get() — use __contains__ + index
                module = self.edge_projections[proj_key] if proj_key in self.edge_projections else None
                edge_out = op.forward(src_state, module, edge.attrs)
                # RMS of the edge contribution (detached — must not break grad graph)
                act_norms[edge.id] = float(
                    edge_out.detach().norm().item()
                    / math.sqrt(max(edge_out.numel(), 1))
                )
                incoming.append(edge_out)

            # Sum all incoming signals, then apply node activation
            agg = torch.stack(incoming, dim=0).sum(dim=0)
            node_out = node_op(agg)

            # Apply modulation edges that target this node
            for mod_edge in self._mod_edges:
                if mod_edge.members[1] != name:
                    continue
                nt_name = mod_edge.members[0]
                if nt_name not in nt_levels:
                    # NT not present this forward → modulation is identity
                    continue
                nt_level = float(nt_levels[nt_name])
                mod_op = self._edge_ops[mod_edge.id]
                node_out = mod_op.apply_modulation(
                    node_out, mod_edge.attrs, nt_level
                )

            states[name] = node_out
            # RMS of the node's output (detached)
            act_norms[f"population:{name}"] = float(
                node_out.detach().norm().item()
                / math.sqrt(max(node_out.numel(), 1))
            )

        self._last_act_norms = act_norms
        return states

    def to_ir(self) -> HypergraphIR:
        """Return the HypergraphIR that this executor was built from.

        Structural roundtrip: node count, edge count, names, and members
        are all preserved. This is the basis for IR → executor → IR
        roundtrip compilability.
        """
        return HypergraphIR(
            nodes=list(self._ir.nodes),
            hyperedges=list(self._ir.hyperedges),
            source_map=self._ir.source_map,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_key(s: str) -> str:
        """Sanitise a string for use as an nn.ModuleDict key.

        nn.ModuleDict keys must not contain '.'; '->' and ':' are also
        problematic in attribute-access contexts.
        """
        return s.replace("->", "__").replace(".", "_").replace(":", "_")

    def _topological_sort(self) -> List[str]:
        """Kahn's algorithm over synapse edges among population nodes.

        Nodes with no incoming synapse edges are processed first. Any
        remaining nodes (in a cycle or unreachable) are appended in
        declaration order so the forward pass always covers every node.
        """
        pop_set = set(self._pop_names)
        in_degree: Dict[str, int] = {n: 0 for n in self._pop_names}
        adj: Dict[str, List[str]] = {n: [] for n in self._pop_names}

        for edge in self._syn_edges:
            src, dst = edge.members[0], edge.members[1]
            if src in pop_set and dst in pop_set:
                in_degree[dst] += 1
                adj[src].append(dst)

        # Sources first (stable: preserve declaration order within a tier)
        queue: List[str] = [
            n for n in self._pop_names if in_degree[n] == 0
        ]
        order: List[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbour in adj[node]:
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)

        # Append any nodes not reached (cycles or disconnected)
        seen = set(order)
        for name in self._pop_names:
            if name not in seen:
                order.append(name)

        return order


def executor_grad_norms(executor: HypergraphExecutor) -> Dict[str, float]:
    """Per-IR-element L2 grad norms from a HypergraphExecutor.

    Returns a dict keyed by IR element IDs ready for TrainingHeatmap.update:
      "population:{name}"    for each population node
      "synapse:{src}->{dst}" for each synapse edge (i.e., edge.id)

    Bypasses the generic parameter_grad_norms + alias path — the executor
    names its parameters node_layers.{key}.* and edge_projections.{key}.*
    which the alias splitter can't resolve to node names. This function
    reads the executor's internal lookup tables directly.

    Returns an empty dict when no backward pass has been done (all .grad
    are None), so callers don't need to guard against it.
    """
    out: Dict[str, float] = {}

    for name in executor._pop_names:
        key = executor._safe_key(name)
        layer = executor.node_layers[key]
        sumsq = sum(
            float(p.grad.detach().norm(2).item()) ** 2
            for p in layer.parameters()
            if p.grad is not None
        )
        if sumsq > 0.0:
            out[f"population:{name}"] = math.sqrt(sumsq)

    for edge in executor._syn_edges:
        key = executor._safe_key(edge.id)
        proj = executor.edge_projections[key]
        sumsq = sum(
            float(p.grad.detach().norm(2).item()) ** 2
            for p in proj.parameters()
            if p.grad is not None
        )
        if sumsq > 0.0:
            out[edge.id] = math.sqrt(sumsq)

    return out


def executor_activation_norms(executor: HypergraphExecutor) -> Dict[str, float]:
    """Per-IR-element RMS activation norms from the most recent forward pass.

    Returns a dict keyed by IR element IDs ready for TrainingHeatmap.update:
      "population:{name}"    for each population node (RMS of its output)
      "synapse:{src}->{dst}" for each synapse edge (RMS of its projection)

    Returns an empty dict until the first forward() call.

    These are information-throughput signals — how much signal flows through
    each node/edge — as opposed to executor_grad_norms which measures training
    pressure. Use this as the heat source for visualising hot/cold paths.
    """
    return dict(executor._last_act_norms)
