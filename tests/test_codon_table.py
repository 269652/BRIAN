# -*- coding: utf-8 -*-
"""TDD: Degenerate codon table (Layer 4a).

Real DNA maps 64 codons onto ~20 amino acids + stop. That redundancy
(degeneracy) is why most single-base mutations are synonymous (silent),
which makes the genome robust and gives evolution a smooth landscape to
climb. This table gives the IR's structural symbol stream the same
property.

Tested contract:
  - exactly 64 codons, fully assigned
  - the code is degenerate (some symbol has >1 codon)
  - START -> ATG, STOP -> one of the stop codons
  - encode/decode round-trips a symbol stream
  - swapping a codon for a synonym preserves the decoded symbol
  - a measurable fraction of single-base mutations are synonymous (>0)
"""
import pytest

from neuroslm.compiler.codon_table import CodonTable
from neuroslm.compiler.nucleotide_codec import (
    BASES, START_CODON, STOP_CODONS,
)


class TestTableStructure:
    def test_there_are_64_codons(self):
        table = CodonTable()
        assert len(table.codon_to_symbol) == 64

    def test_all_codons_are_base_triplets(self):
        table = CodonTable()
        for codon in table.codon_to_symbol:
            assert len(codon) == 3
            assert all(b in BASES for b in codon)

    def test_code_is_degenerate(self):
        """At least one symbol is reachable by more than one codon."""
        table = CodonTable()
        assert any(len(cs) > 1 for cs in table.symbol_to_codons.values())

    def test_start_symbol_maps_to_atg(self):
        table = CodonTable()
        assert table.encode_symbols(["START"]) == START_CODON

    def test_stop_symbol_maps_to_a_stop_codon(self):
        table = CodonTable()
        assert table.encode_symbols(["STOP"]) in STOP_CODONS

    def test_every_symbol_has_at_least_one_codon(self):
        table = CodonTable()
        for sym in table.symbols:
            assert len(table.symbol_to_codons[sym]) >= 1


class TestEncodeDecode:
    def test_roundtrip_symbol_stream(self):
        table = CodonTable()
        stream = ["START", "NODE", "KIND_POP", "ATTR", "EDGE",
                  "KIND_SYN", "MEMBER", "STOP"]
        strand = table.encode_symbols(stream)
        assert table.decode_symbols(strand) == stream

    def test_encoded_stream_is_codon_aligned(self):
        table = CodonTable()
        strand = table.encode_symbols(["NODE", "EDGE", "ATTR"])
        assert len(strand) % 3 == 0

    def test_decode_rejects_unknown_codon_length(self):
        table = CodonTable()
        with pytest.raises(ValueError):
            table.decode_symbols("ATGA")  # not a multiple of 3


class TestDegeneracyAndRobustness:
    def test_synonymous_codons_decode_to_same_symbol(self):
        table = CodonTable()
        for sym, codons in table.symbol_to_codons.items():
            for codon in codons:
                assert table.codon_to_symbol[codon] == sym

    def test_synonym_substitution_is_silent(self):
        """A symbol with synonyms: swapping codons leaves the decode intact."""
        table = CodonTable()
        sym = next(s for s, cs in table.symbol_to_codons.items() if len(cs) > 1)
        codons = table.symbol_to_codons[sym]
        # Two different codons for the same symbol decode identically.
        assert table.codon_to_symbol[codons[0]] == table.codon_to_symbol[codons[1]]

    def test_silent_mutation_fraction_is_positive(self):
        """A nonzero share of single-base mutations are synonymous."""
        table = CodonTable()
        frac = table.silent_mutation_fraction()
        assert 0.0 < frac <= 1.0

    def test_synonymous_codons_listed_for_symbol(self):
        table = CodonTable()
        sym = next(s for s, cs in table.symbol_to_codons.items() if len(cs) > 1)
        syns = table.synonymous_codons(sym)
        assert len(syns) >= 2
        assert all(table.codon_to_symbol[c] == sym for c in syns)
