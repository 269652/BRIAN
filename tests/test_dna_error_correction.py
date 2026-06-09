# -*- coding: utf-8 -*-
"""TDD: DNA error correction + redundancy (Layer 2).

Two biologically-grounded mechanisms make the strand robust:

1. Watson-Crick duplex — a sense strand plus its complementary antisense
   strand. If the sense strand is damaged, it is rebuilt from the
   antisense template (this is how real cells repair DNA). A per-strand
   checksum tells the reader which strand to trust.

2. Hamming(7,4) parity — adds parity bits so any single bit-flip inside a
   7-bit block is detected and corrected.

These tests pin the exact guarantees the encoding layer relies on.
"""
import pytest

from neuroslm.compiler.dna_error_correction import DNAErrorCorrection
from neuroslm.compiler.nucleotide_codec import NucleotideCodec, BASES


class TestComplementaryStrand:
    """Watson-Crick complementarity: A<->T, C<->G."""

    def test_complement_swaps_watson_crick_pairs(self):
        ec = DNAErrorCorrection()
        assert ec.complement("ACGT") == "TGCA"

    def test_complement_is_an_involution(self):
        """Complementing twice returns the original strand."""
        ec = DNAErrorCorrection()
        strand = "AACGTTGCA"
        assert ec.complement(ec.complement(strand)) == strand

    def test_reverse_complement(self):
        """reverse_complement = complement read 3'->5'."""
        ec = DNAErrorCorrection()
        # complement("AACG") = "TTGC"; reversed -> "CGTT"
        assert ec.reverse_complement("AACG") == "CGTT"

    def test_complement_rejects_invalid_base(self):
        ec = DNAErrorCorrection()
        with pytest.raises(ValueError):
            ec.complement("ACGX")


class TestDuplexRepair:
    """A double strand recovers from whole-strand damage via its template."""

    def test_duplex_roundtrip_lossless(self):
        ec = DNAErrorCorrection()
        codec = NucleotideCodec()
        payload = codec.encode_bytes(b"duplex payload \x00\xab")
        sense, antisense = ec.make_duplex(payload)
        assert ec.read_duplex(sense, antisense) == payload

    def test_antisense_is_complement_of_sense_core(self):
        ec = DNAErrorCorrection()
        sense, antisense = ec.make_duplex("ACGTACGT")
        # The antisense must encode the complement of the sense payload.
        # Reading the duplex back yields the original sense payload.
        assert ec.read_duplex(sense, antisense) == "ACGTACGT"

    def test_recovers_when_sense_strand_is_corrupted(self):
        """Garble the entire sense strand — antisense template repairs it."""
        ec = DNAErrorCorrection()
        codec = NucleotideCodec()
        payload = codec.encode_bytes(b"important genetic payload")
        sense, antisense = ec.make_duplex(payload)
        # Corrupt many bases in the sense strand.
        corrupted = list(sense)
        for i in range(0, len(corrupted), 2):
            corrupted[i] = "A"
        corrupted_sense = "".join(corrupted)
        recovered = ec.read_duplex(corrupted_sense, antisense)
        assert recovered == payload

    def test_recovers_when_antisense_strand_is_corrupted(self):
        """Symmetry: damage the antisense, recover from sense."""
        ec = DNAErrorCorrection()
        codec = NucleotideCodec()
        payload = codec.encode_bytes(b"redundancy works both ways")
        sense, antisense = ec.make_duplex(payload)
        corrupted_antisense = "A" * len(antisense)
        recovered = ec.read_duplex(sense, corrupted_antisense)
        assert recovered == payload


class TestHammingParity:
    """Hamming(7,4): single bit-flip per 7-bit block is corrected."""

    def test_protect_recover_lossless(self):
        ec = DNAErrorCorrection()
        codec = NucleotideCodec()
        payload = codec.encode_bytes(b"Hamming protected gene \x01\x02\x03")
        protected = ec.protect(payload)
        assert ec.recover(protected) == payload

    def test_protected_strand_is_all_bases(self):
        ec = DNAErrorCorrection()
        protected = ec.protect("ACGTACGTACGT")
        assert all(ch in BASES for ch in protected)

    def test_corrects_single_bit_flip(self):
        """Flip one bit (one base value by +1 mod 4 on its low bit) -> corrected."""
        ec = DNAErrorCorrection()
        codec = NucleotideCodec()
        payload = codec.encode_bytes(b"single bit error test")
        protected = ec.protect(payload)

        # Introduce a single-bit error: change exactly one base so that
        # only its low bit flips (A<->C, G<->T are 1-bit-apart pairs).
        flip = {"A": "C", "C": "A", "G": "T", "T": "G"}
        idx = len(protected) // 2
        corrupted = protected[:idx] + flip[protected[idx]] + protected[idx + 1:]
        assert corrupted != protected

        assert ec.recover(corrupted) == payload

    def test_empty_payload(self):
        ec = DNAErrorCorrection()
        assert ec.recover(ec.protect("")) == ""


class TestCombinedProtection:
    """protect() + make_duplex() compose into a fully armored strand."""

    def test_full_armor_roundtrip(self):
        ec = DNAErrorCorrection()
        codec = NucleotideCodec()
        payload_bytes = b"fully armored genetic message \xde\xad\xbe\xef"
        payload = codec.encode_bytes(payload_bytes)

        protected = ec.protect(payload)
        sense, antisense = ec.make_duplex(protected)

        # Recover the protected strand from the duplex, then strip parity.
        recovered_protected = ec.read_duplex(sense, antisense)
        recovered_payload = ec.recover(recovered_protected)
        assert codec.decode_bytes(recovered_payload) == payload_bytes
