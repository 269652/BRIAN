# -*- coding: utf-8 -*-
"""TDD: Nucleotide codec — bytes <-> ATCG bases <-> codons.

Biological foundation of the DNA compiler. Real DNA uses 4 bases
(A, C, G, T) and reads them in codons (triplets). This codec:

1. Encodes arbitrary bytes into a nucleotide sequence (2 bits/base)
2. Decodes a nucleotide sequence back to the exact bytes (lossless)
3. Groups bases into codons (3 bases) with a reading frame
4. Provides start (ATG) and stop (TAA/TAG/TGA) codons

These tests pin the lossless byte<->base contract and the codon
structure that higher layers (error correction, IR encoding) rely on.
"""
import pytest

from neuroslm.compiler.nucleotide_codec import (
    NucleotideCodec, Codon, BASES, START_CODON, STOP_CODONS,
)


class TestBaseEncoding:
    """Lossless bytes <-> nucleotide-base roundtrip."""

    def test_four_canonical_bases_exist(self):
        """The alphabet is exactly A, C, G, T."""
        assert set(BASES) == {"A", "C", "G", "T"}
        assert len(BASES) == 4

    def test_single_byte_encodes_to_four_bases(self):
        """One byte (8 bits) = four 2-bit bases."""
        codec = NucleotideCodec()
        seq = codec.encode_bytes(b"\x00")
        assert seq == "AAAA"  # 0x00 -> 00 00 00 00 -> A A A A

    def test_byte_0xff_encodes_to_four_ts(self):
        """0xFF -> 11 11 11 11 -> TTTT."""
        codec = NucleotideCodec()
        assert codec.encode_bytes(b"\xff") == "TTTT"

    def test_byte_roundtrip_is_lossless(self):
        """encode then decode returns the original bytes exactly."""
        codec = NucleotideCodec()
        payload = b"Hello, DNA! \x00\x01\x02\xfe\xff"
        seq = codec.encode_bytes(payload)
        assert codec.decode_bytes(seq) == payload

    def test_unicode_text_roundtrip(self):
        """UTF-8 (including non-ASCII) survives the roundtrip."""
        codec = NucleotideCodec()
        text = "architecture rcc_bowtie — Φ > 0 → thalamus"
        payload = text.encode("utf-8")
        seq = codec.encode_bytes(payload)
        assert codec.decode_bytes(seq).decode("utf-8") == text

    def test_encoded_sequence_only_contains_bases(self):
        """Every character of an encoded strand is a valid base."""
        codec = NucleotideCodec()
        seq = codec.encode_bytes(b"arbitrary payload 12345")
        assert all(ch in BASES for ch in seq)

    def test_empty_bytes_roundtrip(self):
        codec = NucleotideCodec()
        assert codec.encode_bytes(b"") == ""
        assert codec.decode_bytes("") == b""

    def test_decode_rejects_non_base_characters(self):
        codec = NucleotideCodec()
        with pytest.raises(ValueError):
            codec.decode_bytes("ACGTX")

    def test_decode_rejects_length_not_multiple_of_four(self):
        """4 bases per byte — a partial byte is a corruption signal."""
        codec = NucleotideCodec()
        with pytest.raises(ValueError):
            codec.decode_bytes("ACG")


class TestCodons:
    """Codon (triplet) structure and the genetic reading frame."""

    def test_codon_is_three_bases(self):
        c = Codon("A", "T", "G")
        assert len(c) == 3
        assert str(c) == "ATG"

    def test_start_codon_is_atg(self):
        """ATG is the canonical start codon (as in real biology)."""
        assert START_CODON == "ATG"

    def test_stop_codons_are_canonical(self):
        """TAA, TAG, TGA are the three stop codons."""
        assert set(STOP_CODONS) == {"TAA", "TAG", "TGA"}

    def test_split_into_codons(self):
        """A base sequence groups into codons left-to-right."""
        codec = NucleotideCodec()
        codons = codec.to_codons("ATGAAATAA")
        assert [str(c) for c in codons] == ["ATG", "AAA", "TAA"]

    def test_to_codons_requires_multiple_of_three(self):
        codec = NucleotideCodec()
        with pytest.raises(ValueError):
            codec.to_codons("ATGA")

    def test_codons_join_back_to_sequence(self):
        codec = NucleotideCodec()
        seq = "ATGCCCGGGTAA"
        codons = codec.to_codons(seq)
        assert codec.from_codons(codons) == seq

    def test_wrap_in_reading_frame_adds_start_and_stop(self):
        """A coding region is framed by a start and a stop codon."""
        codec = NucleotideCodec()
        framed = codec.wrap_reading_frame("AAACCC")
        assert framed.startswith(START_CODON)
        assert framed[-3:] in STOP_CODONS

    def test_unwrap_reading_frame_recovers_payload(self):
        codec = NucleotideCodec()
        payload = "AAACCCGGG"
        framed = codec.wrap_reading_frame(payload)
        assert codec.unwrap_reading_frame(framed) == payload

    def test_framed_payload_survives_byte_roundtrip(self):
        """bytes -> bases -> frame -> unframe -> bases -> bytes."""
        codec = NucleotideCodec()
        payload = b"gene payload \x10\x20"
        seq = codec.encode_bytes(payload)
        # pad to multiple of 3 for codon framing, tracked by codec
        framed = codec.wrap_reading_frame(seq, pad=True)
        recovered_seq = codec.unwrap_reading_frame(framed, unpad=True)
        assert codec.decode_bytes(recovered_seq) == payload
