# -*- coding: utf-8 -*-
"""DNA error correction + redundancy — Layer 2 of the DNA compiler.

Makes a nucleotide strand robust to corruption with two biologically
grounded mechanisms:

1. Watson-Crick duplex (``make_duplex`` / ``read_duplex``)
   A sense strand and its complementary antisense strand are stored
   together. Each carries a checksum; the reader trusts the strand whose
   checksum still validates and, if the sense strand is damaged, rebuilds
   the payload from the antisense template — exactly how a cell repairs
   one DNA strand from the other.

2. Hamming(7,4) parity (``protect`` / ``recover``)
   Parity bases are interleaved so any single bit-flip inside a 7-bit
   block is detected and corrected.

The two compose: ``protect`` first (per-block bit correction), then
``make_duplex`` (whole-strand template repair).
"""
from __future__ import annotations
import struct
import hashlib
from typing import List, Tuple

from neuroslm.compiler.nucleotide_codec import (
    NucleotideCodec, BASES, COMPLEMENT,
)

_BASE_TO_VAL = {b: i for i, b in enumerate(BASES)}

# Duplex per-strand checksum length, in bases (8 bases = 16 bits).
_CHECKSUM_BASES = 8
# Hamming length header, in bases (16 bases = 32-bit base count).
_LEN_HEADER_BASES = 16


class DNAErrorCorrection:
    """Complementary-strand redundancy + Hamming(7,4) parity."""

    def __init__(self) -> None:
        self._codec = NucleotideCodec()

    # ── Watson-Crick complement ──────────────────────────────────────────

    def complement(self, strand: str) -> str:
        try:
            return "".join(COMPLEMENT[b] for b in strand)
        except KeyError as exc:  # pragma: no cover - message clarity only
            raise ValueError(f"invalid base in strand: {exc.args[0]!r}")

    def reverse_complement(self, strand: str) -> str:
        return self.complement(strand)[::-1]

    # ── Duplex (double strand) redundancy ────────────────────────────────

    def make_duplex(self, payload: str) -> Tuple[str, str]:
        """Return (sense, antisense) — a checksummed double strand."""
        sense = self._checksum(payload) + payload
        anti_body = self.complement(payload)
        antisense = self._checksum(anti_body) + anti_body
        return sense, antisense

    def read_duplex(self, sense: str, antisense: str) -> str:
        """Recover the payload, trusting whichever strand validates."""
        # Prefer the sense strand when its checksum still holds.
        chk, body = sense[:_CHECKSUM_BASES], sense[_CHECKSUM_BASES:]
        if self._checksum(body) == chk:
            return body
        # Fall back to the antisense template.
        chk, body = antisense[:_CHECKSUM_BASES], antisense[_CHECKSUM_BASES:]
        if self._checksum(body) == chk:
            return self.complement(body)
        raise ValueError("both strands corrupt — cannot recover payload")

    def _checksum(self, body: str) -> str:
        digest = hashlib.sha256(body.encode("ascii")).digest()
        # 2 bytes -> 8 bases.
        return self._codec.encode_bytes(digest[: _CHECKSUM_BASES // 4])

    # ── Hamming(7,4) parity ──────────────────────────────────────────────

    def protect(self, payload: str) -> str:
        """Add Hamming(7,4) parity, returning an all-bases protected strand."""
        base_len = len(payload)
        header = self._codec.encode_bytes(struct.pack(">I", base_len))
        if base_len == 0:
            return header

        data_bits = _bases_to_bits(payload)
        # Pad data bits up to a multiple of 4 (nibble per codeword).
        while len(data_bits) % 4 != 0:
            data_bits.append(0)

        code_bits: List[int] = []
        for i in range(0, len(data_bits), 4):
            code_bits.extend(_hamming_encode(data_bits[i:i + 4]))

        body = _bits_to_bases(code_bits)  # pads to even internally
        return header + body

    def recover(self, protected: str) -> str:
        """Correct single-bit-per-block errors and return the payload."""
        if not protected:
            return ""
        header = protected[:_LEN_HEADER_BASES]
        body = protected[_LEN_HEADER_BASES:]
        (base_len,) = struct.unpack(">I", self._codec.decode_bytes(header))
        if base_len == 0:
            return ""

        data_bits_len = 2 * base_len
        padded_data_len = ((data_bits_len + 3) // 4) * 4
        n_blocks = padded_data_len // 4
        code_bits_needed = n_blocks * 7

        body_bits = _bases_to_bits(body)[:code_bits_needed]

        data_bits: List[int] = []
        for i in range(0, code_bits_needed, 7):
            data_bits.extend(_hamming_decode(body_bits[i:i + 7]))

        data_bits = data_bits[:data_bits_len]
        return _bits_to_bases(data_bits)


# ── bit <-> base helpers ─────────────────────────────────────────────────

def _bases_to_bits(strand: str) -> List[int]:
    bits: List[int] = []
    for b in strand:
        v = _BASE_TO_VAL[b]
        bits.append((v >> 1) & 1)
        bits.append(v & 1)
    return bits


def _bits_to_bases(bits: List[int]) -> str:
    bits = list(bits)
    if len(bits) % 2 != 0:
        bits.append(0)
    out = []
    for i in range(0, len(bits), 2):
        out.append(BASES[(bits[i] << 1) | bits[i + 1]])
    return "".join(out)


# ── Hamming(7,4) over GF(2) ──────────────────────────────────────────────
# Codeword layout (1-indexed): [p1, p2, d1, p3, d2, d3, d4]
#   p1 covers positions 1,3,5,7   p2 covers 2,3,6,7   p3 covers 4,5,6,7

def _hamming_encode(nibble: List[int]) -> List[int]:
    d1, d2, d3, d4 = nibble
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4
    return [p1, p2, d1, p3, d2, d3, d4]


def _hamming_decode(block: List[int]) -> List[int]:
    c = list(block)
    if len(c) < 7:
        c = c + [0] * (7 - len(c))
    s1 = c[0] ^ c[2] ^ c[4] ^ c[6]
    s2 = c[1] ^ c[2] ^ c[5] ^ c[6]
    s3 = c[3] ^ c[4] ^ c[5] ^ c[6]
    syndrome = s1 | (s2 << 1) | (s3 << 2)
    if syndrome != 0 and syndrome <= 7:
        c[syndrome - 1] ^= 1  # correct the flipped bit
    return [c[2], c[4], c[5], c[6]]  # d1, d2, d3, d4
