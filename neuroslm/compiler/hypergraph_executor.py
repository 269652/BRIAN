# -*- coding: utf-8 -*-
"""HypergraphExecutor — runtime tensor routing through the hypergraph IR.

The compiler's traditional path is:
  DSL → HypergraphIR → CodeGenerator → exec() → static nn.Module

This module provides the alternative:
  DSL → HypergraphIR → HypergraphExecutor (live nn.Module)

The IR persists at runtime. Each population node owns an nn.Linear; each
synapse edge owns a projection nn.Linear. The forward pass does a
topological traversal so every tensor step is differentiable and gradients
flow naturally through the full graph topology — no exec(), no string gen.

This is the foundation for trainable graph structure: because the routing
is differentiable, gradient information about which edges matter is already
present in the computation graph. Growing or pruning edges becomes a matter
of adding/removing nn.Module parameters and re-running the traversal.
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroslm.compiler.hypergraph_ir import HypergraphIR, HyperEdge, HyperNode


class HypergraphExecutor(nn.Module):
    """Execute a HypergraphIR as a live, differentiable nn.Module.

    Each population node gets an nn.Linear(d_model, d_model) + ReLU.
    Each synapse edge gets an nn.Linear(d_model, d_model) projection.

    Forward pass:
      1. All source nodes (no incoming synapses) transform the input x.
      2. Each subsequent node in topological order sums: its own
         transformation of x plus the projected output of every upstream
         neighbour that has a synapse edge pointing to it.
      3. Returns a dict {population_name: tensor (B, d_model)}.

    This matches the output contract of the exec()-compiled circuits in
    BRIANHarness (_pick_sink_output reads from a dict of pop outputs).
    """

    def __init__(self, ir: HypergraphIR, d_model: int) -> None:
        super().__init__()
        self._ir = ir
        self.d_model = d_model

        pop_nodes = [n for n in ir.nodes if n.kind == "population"]
        syn_edges = [e for e in ir.hyperedges if e.kind == "synapse"]

        # One learnable transformation per population node
        self.node_layers: nn.ModuleDict = nn.ModuleDict({
            self._safe_key(n.name): nn.Linear(d_model, d_model)
            for n in pop_nodes
        })

        # One learnable projection per synapse edge
        self.edge_projections: nn.ModuleDict = nn.ModuleDict({
            self._safe_key(e.id): nn.Linear(d_model, d_model)
            for e in syn_edges
        })

        # Cache for the forward pass — avoid re-computing each call
        self._pop_names: List[str] = [n.name for n in pop_nodes]
        self._syn_edges: List[HyperEdge] = syn_edges
        self._topo_order: List[str] = self._topological_sort()

        # Per-element activation RMS from the most recent forward pass.
        # Keyed by IR element ID: "population:{name}" and "{edge.id}".
        # Updated in-place on every forward; empty until the first call.
        self._last_act_norms: Dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        **kwargs,  # absorbs nt_levels and other harness kwargs
    ) -> Dict[str, torch.Tensor]:
        """Route x through the hypergraph and return per-population outputs.

        Args:
            x: Input tensor of shape (B, d_model).

        Returns:
            Dict mapping population name → output tensor (B, d_model).
        """
        states: Dict[str, torch.Tensor] = {}
        act_norms: Dict[str, float] = {}

        for name in self._topo_order:
            key = self._safe_key(name)
            node_layer = self.node_layers[key]

            # Always include the node's own transformation of the raw input.
            # This ensures source nodes (no in-edges) are not identity ops.
            incoming = [node_layer(x)]

            # Add a projected contribution from each upstream neighbour,
            # recording the RMS activation of each edge's projection.
            for edge in self._syn_edges:
                if edge.members[1] == name:
                    src_name = edge.members[0]
                    proj_key = self._safe_key(edge.id)
                    proj = self.edge_projections[proj_key]
                    src_state = states.get(src_name, x)
                    edge_out = proj(src_state)
                    # RMS of the edge projection (detached — must not break grad graph)
                    act_norms[edge.id] = float(
                        edge_out.detach().norm().item()
                        / math.sqrt(max(edge_out.numel(), 1))
                    )
                    incoming.append(edge_out)

            # Sum all incoming signals, then apply non-linearity
            agg = torch.stack(incoming, dim=0).sum(dim=0)
            node_out = F.relu(agg)
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
