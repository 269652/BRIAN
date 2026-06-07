# -*- coding: utf-8 -*-
"""Tensor HyperGraph IR (THG-IR) — serializable architecture checkpoint.

The THG-IR is a graph-based representation of the model's topology that can be:
  - Saved to disk (JSON) and restored
  - Converted to/from ProgramIR (DSL AST)
  - Mutated in-place (node embeddings, edge weights)
  - Used as the substrate for evolutionary algorithms

Key invariants:
  - Nodes carry operator_embedding (latent DNA vector)
  - Edges represent synaptic connections and plasticity rules
  - Round-trip conversion ProgramIR → THG → ProgramIR is stable
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class THGNode:
    """A node in the THG (e.g., a population or complex)."""
    id: str
    kind: str  # "population", "complex", "workspace", "gene", etc.
    operator_embedding: List[float]  # Latent DNA vector (d_pay dims)
    metadata: Dict = field(default_factory=dict)


@dataclass
class THGEdge:
    """An edge in the THG (e.g., a synapse or modulation)."""
    id: str
    src: str
    dst: str
    kind: str  # "synapse", "modulation", "vesicle_dock", etc.
    weight: float = 1.0
    plasticity: str = "fixed"  # "fixed", "hebb", "stdp", etc.


@dataclass
class THGCheckpoint:
    """Serializable snapshot of the model's architecture (topology + genes)."""
    version: str  # e.g., "2.0"
    nodes: Dict[str, THGNode]
    edges: Dict[str, THGEdge]
    gene_state: Dict  # GeneticOrchestrator snapshot
    step: int = 0
    metadata: Dict = field(default_factory=dict)

    def save(self, path: str) -> None:
        """Serialize checkpoint to JSON file."""
        data = {
            "version": self.version,
            "step": self.step,
            "metadata": self.metadata,
            "gene_state": self.gene_state,
            "nodes": {
                nid: asdict(node) for nid, node in self.nodes.items()
            },
            "edges": {
                eid: asdict(edge) for eid, edge in self.edges.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "THGCheckpoint":
        """Deserialize checkpoint from JSON file."""
        with open(path) as f:
            data = json.load(f)

        nodes = {
            nid: THGNode(**node_data)
            for nid, node_data in data.get("nodes", {}).items()
        }
        edges = {
            eid: THGEdge(**edge_data)
            for eid, edge_data in data.get("edges", {}).items()
        }

        return cls(
            version=data.get("version", "2.0"),
            nodes=nodes,
            edges=edges,
            gene_state=data.get("gene_state", {}),
            step=data.get("step", 0),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_program_ir(cls, ir) -> "THGCheckpoint":
        """Convert ProgramIR (DSL AST) to THGCheckpoint."""
        nodes: Dict[str, THGNode] = {}
        edges: Dict[str, THGEdge] = {}

        # Create nodes for populations
        for pop in ir.populations:
            nodes[pop.id] = THGNode(
                id=pop.id,
                kind="population",
                operator_embedding=[0.0] * 16,  # d_pay=16 default
                metadata={"count": pop.count, "dynamics": pop.dynamics}
            )

        # Create nodes for complexes
        for cx in ir.complexes:
            nodes[cx.id] = THGNode(
                id=cx.id,
                kind="complex",
                operator_embedding=[0.0] * 16,
                metadata={"trunk": cx.trunk}
            )

        # Create edges for synapses
        for i, syn in enumerate(ir.synapses):
            edge_id = f"syn_{i}_{syn.id}"
            edges[edge_id] = THGEdge(
                id=edge_id,
                src=syn.source,
                dst=syn.target,
                kind="synapse",
                weight=syn.weight or 1.0,
                plasticity="fixed"
            )

        # Create edges for modulations
        for i, mod in enumerate(ir.modulations):
            edge_id = f"mod_{i}_{mod.id}"
            edges[edge_id] = THGEdge(
                id=edge_id,
                src=mod.source_nt,
                dst=mod.target_population,
                kind="modulation",
                weight=mod.gain or 1.0,
                plasticity="fixed"
            )

        return cls(
            version="2.0",
            nodes=nodes,
            edges=edges,
            gene_state={},
            step=0,
            metadata={"source": "ProgramIR"}
        )

    def to_program_ir(self):
        """Convert THGCheckpoint back to ProgramIR (best-effort)."""
        from neuroslm.dsl.compiler import (
            ProgramIR, PopulationIR, SynapseIR, ModulationIR,
            ComplexSubstrateIR
        )

        populations: List[PopulationIR] = []
        complexes: List[ComplexSubstrateIR] = []
        synapses: List[SynapseIR] = []
        modulations: List[ModulationIR] = []

        # Reconstruct populations from nodes
        for nid, node in self.nodes.items():
            if node.kind == "population":
                pop = PopulationIR(
                    name=nid,
                    id=nid,
                    count=node.metadata.get("count", 256),
                    dynamics=node.metadata.get("dynamics", "rate_code"),
                )
                populations.append(pop)
            elif node.kind == "complex":
                cx = ComplexSubstrateIR(
                    name=nid,
                    id=nid,
                    trunk=node.metadata.get("trunk", "")
                )
                complexes.append(cx)

        # Reconstruct synapses from edges
        for eid, edge in self.edges.items():
            if edge.kind == "synapse":
                syn = SynapseIR(
                    source=edge.src,
                    target=edge.dst,
                    id=eid,
                    weight=edge.weight,
                )
                synapses.append(syn)
            elif edge.kind == "modulation":
                mod = ModulationIR(
                    source_nt=edge.src,
                    target_population=edge.dst,
                    id=eid,
                    gain=edge.weight,
                )
                modulations.append(mod)

        return ProgramIR(
            id="restored_from_thg",
            populations=populations,
            complexes=complexes,
            synapses=synapses,
            modulations=modulations,
        )

    def mutate_node(self, node_id: str, delta_embedding: List[float]) -> None:
        """Update a node's operator_embedding in-place (additive mutation)."""
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found in checkpoint")

        node = self.nodes[node_id]
        # Ensure dimensions match
        if len(delta_embedding) != len(node.operator_embedding):
            raise ValueError(
                f"delta_embedding dim {len(delta_embedding)} != "
                f"node embedding dim {len(node.operator_embedding)}"
            )

        # In-place addition
        node.operator_embedding = [
            e + d for e, d in zip(node.operator_embedding, delta_embedding)
        ]
