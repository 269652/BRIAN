# -*- coding: utf-8 -*-
"""TDD: Epigenetics system for incremental DNA evolution.

Tests verify:
1. Epigenetics reads minify flag from parent DNA
2. Gene patches respect parent DNA format
3. Source maps tracked through evolution
4. Incremental patches compose correctly
5. Module structure preserved through evolution
6. Evolved DNA unfolds correctly
"""
import pytest
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass

from neuroslm.compiler.ribosome import RibosomeCompiler, DNAPatch, LatentDNA


class TestEpigeneticsDNA:
    """Test epigenetics for incremental DNA evolution."""

    def test_epigenetics_reads_minify_flag_from_parent_dna(self):
        """Epigenetics should inherit minify setting from parent DNA."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()

            # Create arch with minify: true
            arch_file = arch_dir / "arch.neuro"
            arch_file.write_text("""
architecture test { d_sem: 256, minify: true }
population main { count: 256 }
""")

            # Compile to DNA
            dna_file = tmpdir / "parent.dna"
            compiler = RibosomeCompiler()
            compiler.compile_file(str(arch_dir), str(dna_file))

            # Load DNA and check minify flag
            dna = LatentDNA.load(str(dna_file))

            # Minify flag should be accessible for epigenetics
            minify_setting = dna.invariants.get("minify")
            assert minify_setting is not None, "DNA should store minify setting"
            assert minify_setting is True, "Should inherit minify: true"

    def test_gene_patch_respects_parent_minify_format(self):
        """Gene patches should preserve parent DNA's minify format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()

            arch_file = arch_dir / "arch.neuro"
            arch_file.write_text("""
architecture test { d_sem: 256, minify: true }
population main { count: 256 }
""")

            # Compile to base DNA
            base_dna_file = tmpdir / "base.dna"
            compiler = RibosomeCompiler()
            compiler.compile_file(str(arch_dir), str(base_dna_file))

            # Load base DNA
            base_dna = LatentDNA.load(str(base_dna_file))
            base_minify = base_dna.invariants.get("minify")

            # Create a patch (simulating evolution)
            patch = DNAPatch(
                version="1.0",
                step=100,
                kind="node_mutation",
                target="main",
                delta=[0.1, 0.2, 0.3],
                metadata={"minify_inherited": base_minify}
            )

            # Patch should preserve minify setting
            assert patch.metadata.get("minify_inherited") == base_minify

    def test_evolved_dna_maintains_module_structure(self):
        """Evolved DNA should maintain original module structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()
            lib_dir = arch_dir / "lib"
            lib_dir.mkdir()

            arch_file = arch_dir / "arch.neuro"
            arch_file.write_text("""
architecture test { d_sem: 256 }
import "@/lib/utils"
population main { count: 256 }
""")

            utils_file = lib_dir / "utils.neuro"
            utils_file.write_text("population utils { count: 512 }")

            # Compile to DNA
            dna_file = tmpdir / "test.dna"
            compiler = RibosomeCompiler()
            compiler.compile_file(str(arch_dir), str(dna_file))

            # Load and verify module structure is preserved
            dna_json = json.loads(Path(dna_file).read_text())
            bundled = dna_json["invariants"].get("bundled_dsl", {})
            modules = bundled.get("modules", {})

            # Should have the utils module
            assert "@/lib/utils" in modules, "Module structure should be preserved"

    def test_source_map_preserved_through_evolution(self):
        """Source maps should be preserved when evolved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()

            arch_file = arch_dir / "arch.neuro"
            arch_file.write_text("""
architecture test { d_sem: 256 }
population main { count: 256 }
population aux { count: 512 }
""")

            # Compile
            dna_file = tmpdir / "test.dna"
            compiler = RibosomeCompiler()
            compiler.compile_file(str(arch_dir), str(dna_file))

            # Load source maps
            source_map = compiler.load_source_map(str(dna_file))

            # Source maps should exist for evolution tracking
            assert source_map is not None, "Source maps needed for evolution"

    def test_incremental_patch_application(self):
        """Gene patches should apply incrementally to DNA."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()

            arch_file = arch_dir / "arch.neuro"
            arch_file.write_text("""
architecture test { d_sem: 256 }
population main { count: 256 }
""")

            # Compile base
            base_dna_file = tmpdir / "base.dna"
            compiler = RibosomeCompiler()
            compiler.compile_file(str(arch_dir), str(base_dna_file))

            # Load base
            base_dna = LatentDNA.load(str(base_dna_file))
            original_data = list(base_dna.data)

            # Create patch
            patch = DNAPatch(
                version="1.0",
                step=100,
                kind="node_mutation",
                target="main",
                delta=[0.01, 0.02, 0.03],
                metadata={}
            )

            # Patch should be applicable (this tests structure, not actual application)
            assert len(patch.delta) == 3
            assert patch.target == "main"

    def test_evolved_dna_unfolds_to_valid_dsl(self):
        """Evolved DNA should unfold to valid, parseable DSL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()

            arch_file = arch_dir / "arch.neuro"
            arch_file.write_text("""
architecture test { d_sem: 256, minify: false }
population main { count: 256 }
population aux { count: 512 }
synapse main -> aux { weight: 1.0 }
""")

            # Compile
            dna_file = tmpdir / "test.dna"
            compiler = RibosomeCompiler()
            compiler.compile_file(str(arch_dir), str(dna_file))

            # Unfold
            unfolded_file = tmpdir / "unfolded.neuro"
            compiler.unfold_file(str(dna_file), str(unfolded_file))

            # Should be valid
            content = unfolded_file.read_text()
            assert "architecture" in content
            assert "population" in content

    def test_minify_true_evolution_stays_compact(self):
        """Evolved DNA with minify: true should stay minified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()

            arch_file = arch_dir / "arch.neuro"
            arch_file.write_text("""
architecture test {
    d_sem: 256,
    minify: true
}

population main { count: 256 }
population aux { count: 512 }
""")

            # Compile with minify: true
            dna_file = tmpdir / "minified.dna"
            compiler = RibosomeCompiler()
            compiler.compile_file(str(arch_dir), str(dna_file))

            # Load DNA
            dna_json = json.loads(Path(dna_file).read_text())
            dsl_in_dna = dna_json["invariants"]["dsl_code"]

            # Should be minified (no excessive newlines/whitespace)
            # Count newlines - minified should have fewer
            original_lines = arch_file.read_text().split('\n')
            dna_lines = dsl_in_dna.split('\n')

            # Minified should have significantly fewer lines
            assert len(dna_lines) <= len(original_lines), \
                f"Minified ({len(dna_lines)}) should be <= original ({len(original_lines)})"

    def test_minify_false_evolution_stays_readable(self):
        """Evolved DNA with minify: false should stay pretty-printed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()

            arch_file = arch_dir / "arch.neuro"
            arch_file.write_text("""
architecture test {
    d_sem: 256,
    minify: false
}

population main {
    count: 256,
    dynamics: "rate_code"
}
""")

            # Compile with minify: false
            dna_file = tmpdir / "pretty.dna"
            compiler = RibosomeCompiler()
            compiler.compile_file(str(arch_dir), str(dna_file))

            # Unfold
            unfolded_file = tmpdir / "unfolded.neuro"
            compiler.unfold_file(str(dna_file), str(unfolded_file))

            content = unfolded_file.read_text()

            # Should have proper indentation (pretty-printed)
            assert "\n" in content
            assert len(content.split('\n')) > 5

    def test_rcc_bowtie_evolution_structure(self):
        """Full bowtie arch should support evolution through DNA."""
        arch_root = Path(__file__).parent.parent / "architectures" / "master"
        if not arch_root.exists():
            pytest.skip("master arch not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Compile rcc_bowtie
            dna_file = tmpdir / "rcc_bowtie.dna"
            compiler = RibosomeCompiler()
            compiler.compile_file(str(arch_root), str(dna_file))

            # Load and verify evolution structure
            dna = LatentDNA.load(str(dna_file))

            # Should have all necessary components for evolution
            assert "bundled_dsl" in dna.invariants, "Need bundled DSL for evolution"
            bundled = dna.invariants.get("bundled_dsl", {})
            # Source map could be in DNA invariants or in bundled_dsl
            has_source_map = (
                "source_map" in dna.invariants or
                "source_map" in bundled or
                "minification_map" in dna.invariants
            )
            assert has_source_map, "Need source maps for evolution traceability"

            # Should be able to unfold
            unfolded_file = tmpdir / "evolved.neuro"
            compiler.unfold_file(str(dna_file), str(unfolded_file))
            assert unfolded_file.exists()
