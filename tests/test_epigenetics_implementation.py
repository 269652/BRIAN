# -*- coding: utf-8 -*-
"""TDD: Full epigenetics implementation with minification and source maps.

Tests verify:
1. Gene patches are created correctly
2. Evolution context tracks minify settings
3. Patches compose into evolved DNA
4. Evolved DNA maintains format (minified or pretty)
5. Source maps enable attribution through evolution
6. Unfold respects evolved DNA settings
"""
import pytest
import tempfile
import json
from pathlib import Path

from neuroslm.compiler.epigenetics import (
    EvolutionContext, GeneticEpigenetics, PatchComposer
)
from neuroslm.compiler.ribosome import RibosomeCompiler, DNAPatch, LatentDNA


class TestEpigeneticsImplementation:
    """Test full epigenetics implementation."""

    def test_create_gene_patch(self):
        """Gene patches should be created with proper structure."""
        patch = GeneticEpigenetics.create_patch(
            step=100,
            target="cortex",
            delta=[0.1, 0.2, 0.3],
            kind="node_mutation",
            metadata={"reason": "evolution"}
        )

        assert patch.version == "1.0"
        assert patch.step == 100
        assert patch.target == "cortex"
        assert len(patch.delta) == 3
        assert patch.metadata["reason"] == "evolution"

    def test_evolution_context_from_dna(self):
        """Evolution context should initialize from DNA."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()

            arch_file = arch_dir / "arch.neuro"
            arch_file.write_text("""
architecture test { d_sem: 256, minify: true }
population main { count: 256 }
""")

            # Compile to DNA
            dna_file = tmpdir / "test.dna"
            compiler = RibosomeCompiler()
            compiler.compile_file(str(arch_dir), str(dna_file))

            # Create evolution context
            ctx = GeneticEpigenetics.create_evolution_context(str(dna_file))

            # Should inherit minify setting
            assert ctx.minify_setting is True
            assert ctx.parent_dna is not None

    def test_apply_patch_to_context(self):
        """Patches should be applied to evolution context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()

            arch_file = arch_dir / "arch.neuro"
            arch_file.write_text("""
architecture test { d_sem: 256 }
population main { count: 256 }
""")

            dna_file = tmpdir / "test.dna"
            compiler = RibosomeCompiler()
            compiler.compile_file(str(arch_dir), str(dna_file))

            ctx = GeneticEpigenetics.create_evolution_context(str(dna_file))

            # Apply patch
            patch = GeneticEpigenetics.create_patch(
                step=100,
                target="main",
                delta=[0.1, 0.2],
            )
            ctx.apply_patch(patch)

            assert len(ctx.patches) == 1
            assert ctx.patches[0].target == "main"

    def test_compose_patches_into_evolved_dna(self):
        """Patches should compose into evolved DNA."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()

            arch_file = arch_dir / "arch.neuro"
            arch_file.write_text("""
architecture test { d_sem: 256 }
population main { count: 256 }
""")

            dna_file = tmpdir / "test.dna"
            compiler = RibosomeCompiler()
            compiler.compile_file(str(arch_dir), str(dna_file))

            ctx = GeneticEpigenetics.create_evolution_context(str(dna_file))

            # Apply patches
            patch1 = GeneticEpigenetics.create_patch(
                step=100, target="main", delta=[0.1]
            )
            patch2 = GeneticEpigenetics.create_patch(
                step=200, target="main", delta=[0.2]
            )
            ctx.apply_patch(patch1)
            ctx.apply_patch(patch2)

            # Compose
            evolved = ctx.compose()

            # Should preserve minify setting
            assert evolved.invariants.get("minify") == ctx.minify_setting
            # Should track patches
            assert "_applied_patches" in evolved.invariants
            assert len(evolved.invariants["_applied_patches"]) == 2

    def test_patch_composer_chaining(self):
        """Patches should compose via chaining API."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()

            arch_file = arch_dir / "arch.neuro"
            arch_file.write_text("""
architecture test { d_sem: 256, minify: false }
population main { count: 256 }
population aux { count: 512 }
""")

            dna_file = tmpdir / "test.dna"
            compiler = RibosomeCompiler()
            compiler.compile_file(str(arch_dir), str(dna_file))

            # Load base DNA
            base_dna = LatentDNA.load(str(dna_file))

            # Compose patches using chaining
            evolved = (
                PatchComposer(base_dna)
                .add_patch(GeneticEpigenetics.create_patch(
                    step=100, target="main", delta=[0.1, 0.2]
                ))
                .add_patch(GeneticEpigenetics.create_patch(
                    step=200, target="aux", delta=[0.05]
                ))
                .compose()
            )

            # Evolved DNA should have patches
            assert "_applied_patches" in evolved.invariants
            assert len(evolved.invariants["_applied_patches"]) == 2

    def test_evolved_dna_respects_minify_setting(self):
        """Evolved DNA should maintain original minify setting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()

            # Base with minify: true
            arch_file = arch_dir / "arch.neuro"
            arch_file.write_text("""
architecture test { d_sem: 256, minify: true }
population main { count: 256 }
""")

            base_file = tmpdir / "base.dna"
            compiler = RibosomeCompiler()
            compiler.compile_file(str(arch_dir), str(base_file))

            # Apply patches
            evolved_file = tmpdir / "evolved.dna"
            evolved = GeneticEpigenetics.apply_patches_to_dna(
                str(base_file),
                [GeneticEpigenetics.create_patch(
                    step=100, target="main", delta=[0.1]
                )],
                output_file=str(evolved_file)
            )

            # Evolved should maintain minify: true
            assert evolved.invariants.get("minify") is True

    def test_unfold_evolved_dna_pretty_prints(self):
        """Unfolding evolved DNA should pretty-print by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()

            arch_file = arch_dir / "arch.neuro"
            arch_file.write_text("""
architecture test { d_sem: 256, minify: true }
population main { count: 256 }
population aux { count: 512 }
synapse main -> aux { weight: 1.0 }
""")

            dna_file = tmpdir / "test.dna"
            compiler = RibosomeCompiler()
            compiler.compile_file(str(arch_dir), str(dna_file))

            # Load, apply patch, evolve
            evolved = GeneticEpigenetics.apply_patches_to_dna(
                str(dna_file),
                [GeneticEpigenetics.create_patch(
                    step=100, target="main", delta=[0.1]
                )]
            )

            # Unfold with pretty-printing
            unfolded_file = tmpdir / "unfolded.neuro"
            GeneticEpigenetics.unfold_evolved_dna(
                evolved, str(unfolded_file), pretty_print=True
            )

            content = unfolded_file.read_text()
            # Should have proper formatting
            assert "architecture" in content
            assert "\n" in content  # Multiple lines

    def test_evolved_dna_maintains_modules(self):
        """Evolved DNA should maintain module structure."""
        arch_root = Path(__file__).parent.parent / "architectures" / "master"
        if not arch_root.exists():
            pytest.skip("master arch not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Compile rcc_bowtie
            dna_file = tmpdir / "rcc_bowtie.dna"
            compiler = RibosomeCompiler()
            compiler.compile_file(str(arch_root), str(dna_file))

            # Apply patches
            evolved = GeneticEpigenetics.apply_patches_to_dna(
                str(dna_file),
                [
                    GeneticEpigenetics.create_patch(
                        step=100, target="gws", delta=[0.1, 0.2]
                    ),
                    GeneticEpigenetics.create_patch(
                        step=200, target="pfc", delta=[0.05, 0.15]
                    ),
                ]
            )

            # Modules should be preserved
            bundled = evolved.invariants.get("bundled_dsl", {})
            modules = bundled.get("modules", {})
            assert len(modules) > 5, "Should maintain module structure"

    def test_source_maps_track_evolution(self):
        """Source maps should track attribution through evolution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()

            arch_file = arch_dir / "arch.neuro"
            arch_file.write_text("""
architecture test { d_sem: 256 }
population cortex { count: 256 }
population hippocampus { count: 512 }
""")

            dna_file = tmpdir / "test.dna"
            compiler = RibosomeCompiler()
            compiler.compile_file(str(arch_dir), str(dna_file))

            # Create context with patches
            ctx = GeneticEpigenetics.create_evolution_context(str(dna_file))
            ctx.apply_patch(GeneticEpigenetics.create_patch(
                step=100, target="cortex", delta=[0.1],
                metadata={"reason": "improved_recall"}
            ))
            ctx.apply_patch(GeneticEpigenetics.create_patch(
                step=200, target="hippocampus", delta=[0.2],
                metadata={"reason": "enhanced_consolidation"}
            ))

            evolved = ctx.compose()

            # Source maps should track patches
            source_maps = evolved.invariants.get("_evolution_source_maps", {})
            assert "cortex" in source_maps or "_evolution_source_maps" in evolved.invariants
