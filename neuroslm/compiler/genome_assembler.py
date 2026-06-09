# -*- coding: utf-8 -*-
"""GenomeAssembler — Layer 4b, the full DSL <-> DNA pipeline.

Runs every layer end-to-end:

    DSL source
      -> HypergraphIR + SourceMap                 (Layer 3)
      -> coding region : codon gene stream         (Layer 4a, degenerate)
       + payload       : IR(JSON) -> bases         (Layer 1)
                          protected (Hamming) + duplex (Layer 2)
      -> Genome  (the .dna content; pure nucleotides + codons)
      -> disassemble -> bit-identical DSL

A Genome is a real nucleotide structure, not stored DSL text:

    coding_region     codon-encoded gene stream over the IR's nodes/edges
                      (NODE / EDGE / KIND_* / MEMBER ... STOP) — the
                      evolvable, mutation-robust "coding" strand.
    payload_sense     Hamming-protected IR bytes, encoded to bases.
    payload_antisense Watson-Crick complement of the protected payload,
                      checksummed — the repair template.

The payload carries the serialized HypergraphIR, whose SourceMap holds
the original source. Disassembly error-corrects the duplex, decodes the
IR, and renders the SourceMap — byte-for-byte identical to the input.
Mutating the IR (via render_with_overrides) changes only the mutated
region; everything else still round-trips exactly.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from neuroslm.compiler.nucleotide_codec import NucleotideCodec
from neuroslm.compiler.dna_error_correction import DNAErrorCorrection
from neuroslm.compiler.codon_table import CodonTable
from neuroslm.compiler.hypergraph_ir import (
    HypergraphIR, lift_dsl_to_hypergraph,
)

# IR kind -> codon-table KIND symbol.
_KIND_SYMBOL = {
    "architecture": "KIND_ARCH",
    "population": "KIND_POP",
    "neurotransmitter": "KIND_NT",
    "synapse": "KIND_SYN",
    "modulation": "KIND_MOD",
}

_GENOME_VERSION = "dna-genome/1.0"


@dataclass
class Genome:
    """The nucleotide content of a .dna file."""
    coding_region: str                 # codon gene stream (bases)
    payload_sense: str                 # Hamming-protected IR payload (bases)
    payload_antisense: str             # complementary repair template (bases)
    meta: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "version": _GENOME_VERSION,
            "coding_region": self.coding_region,
            "payload_sense": self.payload_sense,
            "payload_antisense": self.payload_antisense,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Genome":
        return cls(
            coding_region=d["coding_region"],
            payload_sense=d["payload_sense"],
            payload_antisense=d["payload_antisense"],
            meta=dict(d.get("meta", {})),
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "Genome":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class GenomeAssembler:
    """Assemble DSL into a Genome and disassemble it back, bit-identically."""

    def __init__(self) -> None:
        self._codec = NucleotideCodec()
        self._ec = DNAErrorCorrection()
        self._table = CodonTable()

    # ── assemble ─────────────────────────────────────────────────────────

    def assemble(self, source: str) -> Genome:
        ir = lift_dsl_to_hypergraph(source)

        coding_region = self._table.encode_symbols(self._symbol_stream(ir))

        payload_json = json.dumps(ir.to_dict()).encode("utf-8")
        payload_bases = self._codec.encode_bytes(payload_json)
        protected = self._ec.protect(payload_bases)
        sense, antisense = self._ec.make_duplex(protected)

        meta = {
            "n_nodes": len(ir.nodes),
            "n_edges": len(ir.hyperedges),
            "source_bytes": len(source.encode("utf-8")),
        }
        return Genome(coding_region, sense, antisense, meta)

    # ── disassemble ──────────────────────────────────────────────────────

    def disassemble(self, genome: Genome) -> str:
        protected = self._ec.read_duplex(genome.payload_sense,
                                         genome.payload_antisense)
        payload_bases = self._ec.recover(protected)
        payload_json = self._codec.decode_bytes(payload_bases)
        ir = HypergraphIR.from_dict(json.loads(payload_json.decode("utf-8")))
        return ir.source_map.render()

    # ── multi-file bundle (each file is a chromosome) ────────────────────

    def assemble_bundle(self, files: Dict[str, str], main: str) -> Genome:
        """Assemble an entire modular architecture into one genome.

        ``files`` maps a relative path (e.g. "lib/equations.neuro") to its
        source. Every file is lifted to a HypergraphIR; their genes form a
        single coding region, and the per-file IRs (which carry each file's
        SourceMap) are the error-corrected payload — so the whole modular
        arch disassembles back byte-for-byte.
        """
        file_irs = {rel: lift_dsl_to_hypergraph(src)
                    for rel, src in files.items()}

        symbols: List[str] = []
        for rel in sorted(file_irs):  # deterministic gene order
            symbols += ["START", "DATA"]            # chromosome (file) marker
            symbols += self._symbol_stream(file_irs[rel])
            symbols += ["SEP"]
        coding_region = self._table.encode_symbols(symbols) if symbols else ""

        payload = {
            "main": main,
            "files": {rel: ir.to_dict() for rel, ir in file_irs.items()},
        }
        payload_json = json.dumps(payload).encode("utf-8")
        payload_bases = self._codec.encode_bytes(payload_json)
        protected = self._ec.protect(payload_bases)
        sense, antisense = self._ec.make_duplex(protected)

        meta = {
            "n_files": len(files),
            "main": main,
            "n_nodes": sum(len(ir.nodes) for ir in file_irs.values()),
            "n_edges": sum(len(ir.hyperedges) for ir in file_irs.values()),
        }
        return Genome(coding_region, sense, antisense, meta)

    def disassemble_bundle(self, genome: Genome) -> Tuple[str, Dict[str, str]]:
        """Disassemble a bundle genome back to (main_path, {relpath: source})."""
        protected = self._ec.read_duplex(genome.payload_sense,
                                         genome.payload_antisense)
        payload_bases = self._ec.recover(protected)
        payload = json.loads(self._codec.decode_bytes(payload_bases).decode("utf-8"))
        if "files" not in payload:
            raise ValueError("genome payload is not a multi-file bundle")
        files = {
            rel: HypergraphIR.from_dict(d).source_map.render()
            for rel, d in payload["files"].items()
        }
        return payload.get("main", "arch.neuro"), files

    def disassemble_with_overrides(self, genome: Genome,
                                   overrides: Dict[str, str]) -> str:
        """Disassemble, re-rendering mutated nodes (evolved genome)."""
        protected = self._ec.read_duplex(genome.payload_sense,
                                         genome.payload_antisense)
        payload_bases = self._ec.recover(protected)
        payload_json = self._codec.decode_bytes(payload_bases)
        ir = HypergraphIR.from_dict(json.loads(payload_json.decode("utf-8")))
        return ir.source_map.render_with_overrides(overrides)

    # ── coding-region gene stream ────────────────────────────────────────

    def _symbol_stream(self, ir: HypergraphIR) -> List[str]:
        """Build the codon-table symbol stream for the IR's genes."""
        symbols: List[str] = []
        for n in ir.nodes:
            symbols += ["START", "NODE", _KIND_SYMBOL.get(n.kind, "DATA"), "STOP"]
        for e in ir.hyperedges:
            symbols += ["START", "EDGE", _KIND_SYMBOL.get(e.kind, "DATA")]
            symbols += ["MEMBER"] * len(e.members)
            symbols += ["STOP"]
        return symbols
