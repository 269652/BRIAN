# -*- coding: utf-8 -*-
"""TDD: RibosomeCompiler stores a nucleotide genome and unfolds from it.

Proves the end-to-end pipeline is wired into the real .dna path:
  - compile_file embeds a Genome (codon coding region + error-corrected
    double-stranded payload) in the DNA — not just stored DSL text
  - unfold_file reconstructs the DSL from that genome, bit-identically
"""
import json
import tempfile
from pathlib import Path

import pytest

from neuroslm.compiler.ribosome import RibosomeCompiler
from neuroslm.compiler.genome_assembler import Genome
from neuroslm.compiler.nucleotide_codec import BASES


def _write_arch(arch_dir: Path, body: str) -> None:
    arch_dir.mkdir(parents=True, exist_ok=True)
    (arch_dir / "arch.neuro").write_text(body, encoding="utf-8")


class TestGenomeEmbedded:
    def test_dna_contains_nucleotide_genome(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            _write_arch(td / "arch", (
                "architecture demo { d_sem: 256, dt: 0.01 }\n"
                'population cortex { count: 512, dynamics: "rate_code" }\n'
                "synapse cortex -> cortex { weight: 0.5 }\n"
            ))
            dna_file = td / "demo.dna"
            RibosomeCompiler().compile_file(str(td / "arch"), str(dna_file))

            blob = json.loads(dna_file.read_text())
            assert "genome" in blob["invariants"], "DNA must embed a genome"

            genome = Genome.from_dict(blob["invariants"]["genome"])
            # Coding region + payload are nucleotides, not DSL text.
            assert all(ch in BASES for ch in genome.coding_region)
            assert all(ch in BASES for ch in genome.payload_sense)
            assert "population" not in genome.payload_sense

    def test_unfold_through_genome_is_bit_identical(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            body = (
                "architecture demo { d_sem: 256, dt: 0.01 }\n"
                'population cortex { count: 512, dynamics: "rate_code" }\n'
                'population striatum { count: 256, dynamics: "rate_code" }\n'
                "synapse cortex -> striatum { weight: 0.5 }\n"
            )
            _write_arch(td / "arch", body)
            dna_file = td / "demo.dna"
            out = td / "out.neuro"

            comp = RibosomeCompiler()
            comp.compile_file(str(td / "arch"), str(dna_file))
            comp.unfold_file(str(dna_file), str(out))

            assert out.read_text(encoding="utf-8") == body

    def test_unfold_survives_payload_strand_corruption(self):
        """Corrupt the sense payload strand in the .dna; unfold still works."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            body = (
                "architecture demo { d_sem: 256 }\n"
                'population p { count: 8, dynamics: "rate_code" }\n'
            )
            _write_arch(td / "arch", body)
            dna_file = td / "demo.dna"
            out = td / "out.neuro"

            comp = RibosomeCompiler()
            comp.compile_file(str(td / "arch"), str(dna_file))

            # Damage the sense strand on disk; antisense template must repair.
            blob = json.loads(dna_file.read_text())
            g = blob["invariants"]["genome"]
            g["payload_sense"] = "A" * len(g["payload_sense"])
            dna_file.write_text(json.dumps(blob), encoding="utf-8")

            comp.unfold_file(str(dna_file), str(out))
            assert out.read_text(encoding="utf-8") == body


class TestRealArchRoundtrip:
    def test_rcc_bowtie_roundtrip_bit_identical(self):
        arch_root = Path(__file__).parent.parent / "architectures" / "master"
        if not (arch_root / "arch.neuro").exists():
            pytest.skip("master arch not found")

        original = (arch_root / "arch.neuro").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            dna_file = td / "rcc.dna"
            out = td / "rcc.neuro"
            comp = RibosomeCompiler()
            comp.compile_file(str(arch_root), str(dna_file))
            comp.unfold_file(str(dna_file), str(out))
            assert out.read_text(encoding="utf-8") == original
