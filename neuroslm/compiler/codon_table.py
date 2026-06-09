# -*- coding: utf-8 -*-
"""Degenerate codon table — Layer 4a of the DNA compiler.

The IR's structural symbol stream (NODE, EDGE, KIND_POP, ATTR, ...) is
encoded as codons through a *degenerate* table, mirroring the real
genetic code where 64 codons map onto a smaller symbol set. Degeneracy
means many single-base mutations are synonymous (silent): the decoded
symbol is unchanged. That robustness is what lets evolution explore the
genome without constantly breaking it — a smooth, self-improvement-
friendly landscape.

START is the canonical ATG; STOP owns the three stop codons (so STOP is
itself degenerate, exactly as in biology). The remaining 60 codons are
partitioned in quads across the structural symbols, which preserves a
"wobble" effect at the third base position.
"""
from __future__ import annotations
from typing import Dict, List

from neuroslm.compiler.nucleotide_codec import (
    BASES, START_CODON, STOP_CODONS,
)

# Structural alphabet of the IR coding region (amino-acid analogues).
_OTHER_SYMBOLS: List[str] = [
    "NODE", "EDGE",
    "KIND_ARCH", "KIND_POP", "KIND_NT", "KIND_SYN", "KIND_MOD",
    "ATTR", "MEMBER", "NAME", "DATA", "SEP", "INT", "FLOAT", "STR",
]  # 15 symbols -> 60 free codons / 15 = 4 synonyms each


def _codon_of(value: int) -> str:
    return BASES[value // 16] + BASES[(value // 4) % 4] + BASES[value % 4]


class CodonTable:
    """A degenerate, biologically-styled codon <-> symbol mapping."""

    def __init__(self) -> None:
        all_codons = [_codon_of(v) for v in range(64)]
        reserved = {START_CODON} | set(STOP_CODONS)
        free = [c for c in all_codons if c not in reserved]  # 60 codons

        self.symbol_to_codons: Dict[str, List[str]] = {
            "START": [START_CODON],
            "STOP": list(STOP_CODONS),
        }
        for i, sym in enumerate(_OTHER_SYMBOLS):
            self.symbol_to_codons[sym] = free[i * 4:(i + 1) * 4]

        self.codon_to_symbol: Dict[str, str] = {}
        for sym, codons in self.symbol_to_codons.items():
            for c in codons:
                self.codon_to_symbol[c] = sym

        self.symbols: List[str] = ["START", "STOP"] + list(_OTHER_SYMBOLS)

    # ── encode / decode ──────────────────────────────────────────────────

    def encode_symbols(self, symbols: List[str]) -> str:
        """Encode a symbol stream using each symbol's canonical codon."""
        out: List[str] = []
        for s in symbols:
            codons = self.symbol_to_codons.get(s)
            if not codons:
                raise ValueError(f"unknown symbol: {s!r}")
            out.append(codons[0])
        return "".join(out)

    def decode_symbols(self, strand: str) -> List[str]:
        """Decode a codon strand back into the symbol stream."""
        if len(strand) % 3 != 0:
            raise ValueError(
                f"strand length {len(strand)} is not codon-aligned (multiple of 3)"
            )
        out: List[str] = []
        for i in range(0, len(strand), 3):
            codon = strand[i:i + 3]
            sym = self.codon_to_symbol.get(codon)
            if sym is None:
                raise ValueError(f"invalid codon: {codon!r}")
            out.append(sym)
        return out

    # ── degeneracy helpers ───────────────────────────────────────────────

    def synonymous_codons(self, symbol: str) -> List[str]:
        return list(self.symbol_to_codons.get(symbol, []))

    def silent_mutation_fraction(self) -> float:
        """Fraction of all single-base substitutions that stay synonymous."""
        total = 0
        silent = 0
        for codon, sym in self.codon_to_symbol.items():
            for pos in range(3):
                for b in BASES:
                    if b == codon[pos]:
                        continue
                    mutated = codon[:pos] + b + codon[pos + 1:]
                    total += 1
                    if self.codon_to_symbol.get(mutated) == sym:
                        silent += 1
        return silent / total if total else 0.0
