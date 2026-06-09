# -*- coding: utf-8 -*-
"""Nucleotide codec — the biological base layer of the DNA compiler.

Real DNA stores information in four bases (A, C, G, T) read in codons
(triplets). This module is the lossless foundation everything else sits
on:

    bytes  <->  nucleotide sequence (2 bits/base, 4 bases/byte)
    sequence  <->  codons (triplets)  with a start/stop reading frame

Base assignment (2 bits each):
    A = 00, C = 01, G = 10, T = 11

The byte layer packs 4 bases per byte (8 bits). The codon layer groups
bases into triplets — the genetic reading frame — independently of byte
boundaries, exactly as biology reads 3-base codons regardless of where a
"machine word" would fall.

Higher layers (error correction, hypergraph-IR encoding) treat this
module as a black box: give it bytes, get a strand; give it a strand,
get the bytes back, bit-for-bit.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List

# ── Base alphabet ────────────────────────────────────────────────────────
# Order defines the 2-bit value: A=0, C=1, G=2, T=3.
BASES = ("A", "C", "G", "T")
_BASE_TO_BITS = {b: i for i, b in enumerate(BASES)}
_BITS_TO_BASE = {i: b for i, b in enumerate(BASES)}

# Watson-Crick complement (used by the error-correction layer).
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}

# Canonical reading-frame markers, as in the standard genetic code.
START_CODON = "ATG"
STOP_CODONS = ("TAA", "TAG", "TGA")


@dataclass(frozen=True)
class Codon:
    """A triplet of bases — the unit of the genetic reading frame."""
    b0: str
    b1: str
    b2: str

    def __post_init__(self):
        for b in (self.b0, self.b1, self.b2):
            if b not in _BASE_TO_BITS:
                raise ValueError(f"invalid base in codon: {b!r}")

    def __len__(self) -> int:
        return 3

    def __str__(self) -> str:
        return f"{self.b0}{self.b1}{self.b2}"

    @classmethod
    def from_str(cls, s: str) -> "Codon":
        if len(s) != 3:
            raise ValueError(f"codon must be 3 bases, got {s!r}")
        return cls(s[0], s[1], s[2])


class NucleotideCodec:
    """Lossless bytes <-> bases and bases <-> codons translation."""

    # ── byte layer (4 bases per byte) ────────────────────────────────────

    def encode_bytes(self, payload: bytes) -> str:
        """Encode bytes to a nucleotide sequence (4 bases per byte)."""
        out: List[str] = []
        for byte in payload:
            # Most-significant base first so the strand reads left-to-right
            # in the same order as the bits.
            out.append(_BITS_TO_BASE[(byte >> 6) & 0b11])
            out.append(_BITS_TO_BASE[(byte >> 4) & 0b11])
            out.append(_BITS_TO_BASE[(byte >> 2) & 0b11])
            out.append(_BITS_TO_BASE[byte & 0b11])
        return "".join(out)

    def decode_bytes(self, sequence: str) -> bytes:
        """Decode a nucleotide sequence back to the exact bytes."""
        if len(sequence) % 4 != 0:
            raise ValueError(
                f"sequence length {len(sequence)} is not a multiple of 4 "
                "(4 bases encode 1 byte) — strand is truncated or corrupt"
            )
        out = bytearray()
        for i in range(0, len(sequence), 4):
            byte = 0
            for j in range(4):
                base = sequence[i + j]
                if base not in _BASE_TO_BITS:
                    raise ValueError(f"invalid base {base!r} at position {i + j}")
                byte = (byte << 2) | _BASE_TO_BITS[base]
            out.append(byte)
        return bytes(out)

    # ── codon layer (triplets / reading frame) ───────────────────────────

    def to_codons(self, sequence: str) -> List[Codon]:
        """Group a base sequence into codons (triplets)."""
        if len(sequence) % 3 != 0:
            raise ValueError(
                f"sequence length {len(sequence)} is not a multiple of 3 "
                "(codons are triplets)"
            )
        return [
            Codon.from_str(sequence[i:i + 3])
            for i in range(0, len(sequence), 3)
        ]

    def from_codons(self, codons: List[Codon]) -> str:
        """Join codons back into a base sequence."""
        return "".join(str(c) for c in codons)

    def wrap_reading_frame(self, payload: str, pad: bool = False) -> str:
        """Frame a payload with a start codon and a stop codon.

        If ``pad`` is set, the payload is padded so the whole framed strand
        is codon-aligned. The pad length (0..2) is recorded by encoding it
        in the choice of stop codon's middle bases is overkill — instead we
        prepend a single pad-count base after the start codon, which
        ``unwrap_reading_frame(unpad=True)`` consumes.
        """
        body = payload
        if pad:
            pad_len = (3 - (len(payload) % 3)) % 3
            # Record pad_len (0,1,2) as one base after START, then pad body.
            pad_base = _BITS_TO_BASE[pad_len]
            body = pad_base + payload + ("A" * pad_len)
        else:
            if len(payload) % 3 != 0:
                raise ValueError(
                    "payload not codon-aligned; pass pad=True to auto-pad"
                )
        return START_CODON + body + STOP_CODONS[0]

    def unwrap_reading_frame(self, framed: str, unpad: bool = False) -> str:
        """Strip the start/stop frame, recovering the original payload."""
        if not framed.startswith(START_CODON):
            raise ValueError("framed strand does not begin with a start codon")
        if framed[-3:] not in STOP_CODONS:
            raise ValueError("framed strand does not end with a stop codon")
        body = framed[len(START_CODON):-3]
        if unpad:
            if not body:
                raise ValueError("padded frame is missing its pad-count base")
            pad_len = _BASE_TO_BITS[body[0]]
            body = body[1:]
            if pad_len:
                body = body[:-pad_len]
        return body
