# -*- coding: utf-8 -*-
"""Promote a verified :class:`DiscoveryRecord` into the genome.

The DNA splice path is the *last mile* of the discovery pipeline:

  evolutionary loop → admission gates → record → Lean stub →
  Lean verifier → ``splice_discovery_into_dna`` → ``arch.neuro``

Only ``proof_status == "verified"`` discoveries may splice — this is the
single audit boundary between *"the engine suggested this"* and
*"this is now part of the lineage"*.

The splice is **declarative**: we parse the discovery's
``mutation_args_json`` (a list of mutation-op dicts, e.g.
``[{"op": "add_modulation", "nt": "dopamine", "target": "pfc", ...}]``)
and append a DSL block expressing each op to the architecture's
``arch.neuro``. The block carries a comment header tagging the
discovery id so a human can backtrack from genome → ledger → proof.

Idempotency: once a record's ``dna_integrated`` bit is set we refuse
to splice it again — the bit is the splice ledger entry.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from neuroslm.discoveries.records import DiscoveryRecord


# ── result ─────────────────────────────────────────────────────────


@dataclass
class SpliceResult:
    """Outcome of a single :func:`splice_discovery_into_dna` call."""
    success: bool
    touched_files: List[str] = field(default_factory=list)
    block_appended: str = ""
    reason: Optional[str] = None


# ── mutation → DSL renderers ───────────────────────────────────────


def _render_add_modulation(args: Dict[str, Any], discovery_id: str) -> str:
    """``add_modulation`` op → a DSL ``modulation nt -> target { ... }``
    block, tagged with the discovery id."""
    nt     = args.get("nt")
    target = args.get("target")
    effect = args.get("effect", "multiplicative")
    gain   = args.get("gain", 0.3)
    if not nt or not target:
        raise ValueError(
            f"add_modulation requires nt + target, got {args!r}"
        )
    return (
        f'modulation {nt} -> {target} {{ '
        f'effect: "{effect}", gain: {gain} '
        f'}}    # autodiscovered ({discovery_id})\n'
    )


def _render_add_feedback(args: Dict[str, Any], discovery_id: str) -> str:
    src    = args.get("src")
    tgt    = args.get("tgt") or args.get("dst")
    weight = args.get("weight", 0.1)
    if not src or not tgt:
        raise ValueError(f"add_feedback requires src + tgt, got {args!r}")
    return (
        f'synapse {tgt} -> {src} {{ '
        f'weight: {weight} '
        f'}}    # autodiscovered feedback ({discovery_id})\n'
    )


def _render_add_gating(args: Dict[str, Any], discovery_id: str) -> str:
    nt  = args.get("nt")
    pop = args.get("pop") or args.get("target")
    if not nt or not pop:
        raise ValueError(f"add_gating requires nt + pop, got {args!r}")
    return (
        f'modulation {nt} -> {pop} {{ '
        f'effect: "multiplicative", gain: 0.3 '
        f'}}    # autodiscovered gating ({discovery_id})\n'
    )


# Single dispatch table — adding a new mutation op only needs a new
# entry here; the splice contract is otherwise unchanged.
_MUTATION_RENDERERS = {
    "add_modulation": _render_add_modulation,
    "add_feedback":   _render_add_feedback,
    "add_gating":     _render_add_gating,
}


# ── public ─────────────────────────────────────────────────────────


def splice_discovery_into_dna(record: DiscoveryRecord,
                              arch_root: Union[str, Path]) -> SpliceResult:
    """Append the DSL block(s) for a verified discovery to ``arch.neuro``.

    Args:
        record:    the discovery to promote. Must have
                   ``proof_status == "verified"`` and an unset
                   ``dna_integrated`` bit. The record will be mutated:
                   ``dna_integrated`` and ``dna_integrated_at`` are set
                   on success.
        arch_root: directory containing ``arch.neuro``.

    Returns:
        :class:`SpliceResult` describing the outcome — never raises for
        already-integrated discoveries (those are a no-op), but raises
        :class:`RuntimeError` for unverified proofs (the proof gate)
        and :class:`FileNotFoundError` if the architecture is missing.
    """
    if record.proof_status != "verified":
        raise RuntimeError(
            f"splice refused: discovery {record.id} has "
            f"proof_status={record.proof_status!r}, must be 'verified' "
            f"before promotion into the genome"
        )

    # Idempotency: already in DNA → no-op.
    if record.dna_integrated:
        return SpliceResult(
            success=False,
            reason=f"discovery {record.id} already integrated "
                   f"at {record.dna_integrated_at}",
        )

    arch_root = Path(arch_root)
    arch_file = arch_root / "arch.neuro"
    if not arch_file.is_file():
        raise FileNotFoundError(f"arch.neuro not found at {arch_file}")

    # Parse the mutation argument list (engine-encoded JSON).
    raw_args = record.mutation_args_json or "[]"
    try:
        mutations: List[Dict[str, Any]] = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"discovery {record.id}: mutation_args_json is not valid JSON"
        ) from exc
    if not isinstance(mutations, list):
        raise ValueError(
            f"discovery {record.id}: mutation_args_json must be a list "
            f"of mutation-op dicts"
        )

    # Render each op into its DSL form. We fail loudly on unknown ops
    # — silent drops would break the audit trail.
    rendered: List[str] = []
    for op_args in mutations:
        op = op_args.get("op")
        renderer = _MUTATION_RENDERERS.get(op)
        if renderer is None:
            raise ValueError(
                f"discovery {record.id}: no renderer for mutation op {op!r}"
            )
        rendered.append(renderer(op_args, record.id))

    # Assemble a single appended block with a discovery-id header so a
    # human can grep for the provenance straight from the genome.
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = (
        "\n"
        "# ============================================================\n"
        f"# Autodiscovered + Lean-verified: {record.id}\n"
        f"# Title: {record.title}\n"
        f"# Parent: {record.parent_dna_id}\n"
        f"# Generation: {record.generation}\n"
        f"# Spliced at: {timestamp}\n"
        f"# Theorem: {record.theorem_name}\n"
        f"# Proof: {record.proof_path or '<emitted at promote time>'}\n"
        "# ============================================================\n"
    )
    block = header + "".join(rendered)

    # Append, preserving the original file byte-for-byte. We don't
    # touch any existing declaration — this matches the SourceMap
    # round-trip guarantee of HypergraphIR.
    existing = arch_file.read_text(encoding="utf-8")
    # Ensure a trailing newline before our block for clean diffs.
    if existing and not existing.endswith("\n"):
        existing = existing + "\n"
    arch_file.write_text(existing + block, encoding="utf-8")

    # Mark the discovery as integrated.
    record.promote_to_dna(at=timestamp)

    return SpliceResult(
        success=True,
        touched_files=[str(arch_file.resolve())],
        block_appended=block,
    )
