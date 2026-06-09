# -*- coding: utf-8 -*-
"""Discoveries — formal-insight ledger + Lean proof pipeline.

This package is the engine behind the two top-level data folders:

* ``hypothesis/``  — human-authored formal claims about the architecture.
                     Each file is YAML-front-matter + Markdown body
                     (KaTeX equations live in the body). The sibling
                     ``hypothesis/proofs/<id>_*.lean`` file is the formal
                     proof obligation, optionally discharged by Lean.

* ``discoveries/`` — engine-authored discoveries from the evolutionary
                     loop. Same on-disk format. ``DiscoveryRecord``
                     additionally tracks the parent genome, mutation
                     chain, fitness delta, and the
                     ``dna_integrated`` bit that controls whether the
                     mutation has been promoted back into the lineage.

The intent is that:

  1. Every claim about BRIAN — whether a human conjecture or an engine
     discovery — has a stable id, a single canonical Markdown file, and
     an optional Lean proof under the same id.
  2. The evolutionary loop writes a :class:`DiscoveryRecord` for every
     mutation the gates admit, emits a Lean proof stub, optionally
     runs the Lean kernel, and (only if verified) calls
     :func:`splice.splice_discovery_into_dna` to promote the mutation
     into the genome.
  3. The whole flow is auditable: ``hypothesis/index.json`` and
     ``discoveries/index.json`` give a machine-readable view; the
     ``.md`` files are git-friendly source of truth.

See ``hypothesis/README.md`` and ``discoveries/README.md`` for the
on-disk schema, and ``docs/formal_framework.md`` §10 for the Lean
backend specification.
"""
from neuroslm.discoveries.records import HypothesisRecord, DiscoveryRecord
from neuroslm.discoveries.store import HypothesisStore, DiscoveryStore
from neuroslm.discoveries.lean import (
    LeanVerdict, emit_hypothesis_proof, emit_discovery_proof,
    verify_lean_proof,
)
from neuroslm.discoveries.splice import SpliceResult, splice_discovery_into_dna

__all__ = [
    "HypothesisRecord", "DiscoveryRecord",
    "HypothesisStore", "DiscoveryStore",
    "LeanVerdict", "emit_hypothesis_proof", "emit_discovery_proof",
    "verify_lean_proof",
    "SpliceResult", "splice_discovery_into_dna",
]
