# -*- coding: utf-8 -*-
"""TDD: GenomeAssembler (Layer 4b) — the full DSL <-> DNA pipeline.

Ties every layer together:

    DSL source
      -> HypergraphIR + SourceMap            (Layer 3)
      -> coding region : codon gene stream   (Layer 4a, degenerate)
       + payload       : IR bytes -> bases   (Layer 1)
                         protected (Hamming) + duplex (Layer 2)
      -> Genome  (the .dna content)
      -> disassemble -> bit-identical DSL

The genome is a real nucleotide representation (coding codons + an
error-corrected double-stranded payload), not stored DSL text. Yet the
unmutated genome disassembles to the original source byte-for-byte.
"""
import json
import tempfile
from pathlib import Path

import pytest

from neuroslm.compiler.genome_assembler import GenomeAssembler, Genome
from neuroslm.compiler.codon_table import CodonTable
from neuroslm.compiler.nucleotide_codec import BASES


SAMPLE = """architecture demo { d_sem: 256, dt: 0.01 }

neurotransmitter dopamine { base_concentration: 0.5 }

population cortex { count: 512, dynamics: "rate_code" }
population striatum { count: 256, dynamics: "rate_code" }

synapse cortex -> striatum { weight: 0.5 }
modulation dopamine -> striatum { gain: 1.2 }
"""


class TestAssembleDisassemble:
    def test_roundtrip_is_bit_identical(self):
        asm = GenomeAssembler()
        genome = asm.assemble(SAMPLE)
        assert asm.disassemble(genome) == SAMPLE

    def test_roundtrip_unicode_and_comments(self):
        src = (
            "# Φ > 0 — re-entry sensory → thalamus\n"
            'architecture u { d_sem: 256 }\n'
            'population p { count: 8, dynamics: "rate_code" }  # tail\n'
        )
        asm = GenomeAssembler()
        assert asm.disassemble(asm.assemble(src)) == src

    def test_genome_is_not_stored_dsl_text(self):
        """The payload strand is nucleotides, not the raw DSL."""
        asm = GenomeAssembler()
        genome = asm.assemble(SAMPLE)
        assert all(ch in BASES for ch in genome.payload_sense)
        assert "population" not in genome.payload_sense  # it's ACGT, not text

    def test_empty_like_minimal_source(self):
        asm = GenomeAssembler()
        src = "architecture m { d_sem: 256 }\n"
        assert asm.disassemble(asm.assemble(src)) == src


class TestCodingRegion:
    """The coding region is a degenerate codon gene stream over the IR."""

    def test_coding_region_is_codon_aligned_bases(self):
        asm = GenomeAssembler()
        genome = asm.assemble(SAMPLE)
        assert all(ch in BASES for ch in genome.coding_region)
        assert len(genome.coding_region) % 3 == 0

    def test_coding_region_has_one_gene_kind_per_population(self):
        asm = GenomeAssembler()
        genome = asm.assemble(SAMPLE)
        symbols = CodonTable().decode_symbols(genome.coding_region)
        assert symbols.count("KIND_POP") == 2  # cortex, striatum

    def test_coding_region_encodes_edges(self):
        asm = GenomeAssembler()
        genome = asm.assemble(SAMPLE)
        symbols = CodonTable().decode_symbols(genome.coding_region)
        assert symbols.count("KIND_SYN") == 1
        assert symbols.count("KIND_MOD") == 1


class TestErrorCorrection:
    """The double-stranded payload survives strand corruption."""

    def test_recovers_from_corrupted_sense_strand(self):
        asm = GenomeAssembler()
        genome = asm.assemble(SAMPLE)
        # Garble the whole sense strand; antisense template must repair it.
        genome.payload_sense = "A" * len(genome.payload_sense)
        assert asm.disassemble(genome) == SAMPLE

    def test_recovers_from_corrupted_antisense_strand(self):
        asm = GenomeAssembler()
        genome = asm.assemble(SAMPLE)
        genome.payload_antisense = "A" * len(genome.payload_antisense)
        assert asm.disassemble(genome) == SAMPLE


class TestGenomeSerialization:
    def test_genome_roundtrips_through_dict(self):
        asm = GenomeAssembler()
        genome = asm.assemble(SAMPLE)
        g2 = Genome.from_dict(genome.to_dict())
        assert asm.disassemble(g2) == SAMPLE

    def test_genome_saves_and_loads(self):
        asm = GenomeAssembler()
        genome = asm.assemble(SAMPLE)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "g.dna"
            genome.save(str(path))
            loaded = Genome.load(str(path))
        assert asm.disassemble(loaded) == SAMPLE


class TestRealArchitecture:
    """The real rcc_bowtie main source round-trips bit-identically."""

    def test_rcc_bowtie_main_source_bit_identical(self):
        arch_root = Path(__file__).parent.parent / "architectures" / "master"
        if not (arch_root / "arch.neuro").exists():
            pytest.skip("master arch not found")

        from neuroslm.compiler.module_bundler import ModuleBundler
        bundled = ModuleBundler(arch_root).bundle(arch_root / "arch.neuro")
        source = bundled.main_source

        asm = GenomeAssembler()
        genome = asm.assemble(source)
        assert asm.disassemble(genome) == source
