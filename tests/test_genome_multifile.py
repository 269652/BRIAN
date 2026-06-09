# -*- coding: utf-8 -*-
"""TDD: multi-file genome — the whole modular arch lives in the genome.

A real architecture is many files (arch.neuro + modules/*.neuro +
lib/*.neuro). The genome must encode ALL of them (each file is a
chromosome) and disassemble back to every file byte-for-byte, so a full
modularized DSL arch can be unfolded from a single .dna.
"""
import pytest

from neuroslm.compiler.genome_assembler import GenomeAssembler
from neuroslm.compiler.codon_table import CodonTable
from neuroslm.compiler.nucleotide_codec import BASES


FILES = {
    "arch.neuro": (
        "architecture main { d_sem: 256, dt: 0.01 }\n"
        'import { x } from "@/lib/equations"\n'
        'import { cortex } from "@/modules/cortex"\n'
        "population gws { count: 512, dynamics: \"rate_code\" }\n"
    ),
    "lib/equations.neuro": (
        "# shared equations — Φ aware\n"
        'export equation x { params: [a], formula: "a * 2" }\n'
    ),
    "modules/cortex.neuro": (
        'export population cortex { count: 1024, dynamics: "rate_code" }\n'
        "synapse cortex -> gws { weight: 0.5 }\n"
    ),
}


class TestBundleRoundtrip:
    def test_every_file_roundtrips_bit_identical(self):
        asm = GenomeAssembler()
        genome = asm.assemble_bundle(FILES, main="arch.neuro")
        main, files = asm.disassemble_bundle(genome)
        assert main == "arch.neuro"
        assert files == FILES  # all files, byte-for-byte

    def test_bundle_meta_counts_files(self):
        asm = GenomeAssembler()
        genome = asm.assemble_bundle(FILES, main="arch.neuro")
        assert genome.meta["n_files"] == 3
        assert genome.meta["main"] == "arch.neuro"

    def test_payload_is_nucleotides_not_text(self):
        asm = GenomeAssembler()
        genome = asm.assemble_bundle(FILES, main="arch.neuro")
        assert all(ch in BASES for ch in genome.payload_sense)
        assert "population" not in genome.payload_sense


class TestBundleCodingRegion:
    def test_coding_region_covers_all_files_populations(self):
        asm = GenomeAssembler()
        genome = asm.assemble_bundle(FILES, main="arch.neuro")
        symbols = CodonTable().decode_symbols(genome.coding_region)
        # gws (arch) + cortex (module) = 2 populations across the bundle.
        assert symbols.count("KIND_POP") == 2

    def test_coding_region_is_codon_aligned_bases(self):
        asm = GenomeAssembler()
        genome = asm.assemble_bundle(FILES, main="arch.neuro")
        assert all(ch in BASES for ch in genome.coding_region)
        assert len(genome.coding_region) % 3 == 0


class TestBundleErrorCorrection:
    def test_recovers_bundle_from_corrupted_sense_strand(self):
        asm = GenomeAssembler()
        genome = asm.assemble_bundle(FILES, main="arch.neuro")
        genome.payload_sense = "A" * len(genome.payload_sense)
        _, files = asm.disassemble_bundle(genome)
        assert files == FILES


class TestSingleFileStillWorks:
    """The single-source assemble()/disassemble() path is unchanged."""

    def test_single_source_roundtrip(self):
        asm = GenomeAssembler()
        src = 'architecture s { d_sem: 256 }\npopulation p { count: 8 }\n'
        assert asm.disassemble(asm.assemble(src)) == src
