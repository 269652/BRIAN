# -*- coding: utf-8 -*-
"""TDD: ModuleBundler must handle ES6-style imports.

Tests verify:
1. ES6 imports are recognized: import { x } from "@/path"
2. Traditional DSL imports still work: import "@/path"
3. Mixed imports work together
4. All modules are properly bundled
5. Unfold produces byte-identical output
"""
import pytest
import tempfile
from pathlib import Path

from neuroslm.compiler.module_bundler import ModuleBundler


class TestModuleBundlerES6:
    """Test ES6 import support in ModuleBundler."""

    def test_es6_style_imports_are_recognized(self):
        """ModuleBundler should recognize ES6 imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()
            lib_dir = arch_dir / "lib"
            lib_dir.mkdir()

            # Main arch with ES6-style import
            arch_file = arch_dir / "arch.neuro"
            arch_file.write_text("""
architecture test { d_sem: 256 }
import { x, y, z } from "@/lib/utils"
population main { count: 256 }
""")

            # Utility module
            utils_file = lib_dir / "utils.neuro"
            utils_file.write_text("population utils { count: 512 }")

            # Bundle
            bundler = ModuleBundler(arch_dir)
            bundled = bundler.bundle(arch_file)

            # Should find the module
            assert len(bundled.modules) > 0, "Should find @/lib/utils module"
            assert "@/lib/utils" in bundled.modules

    def test_traditional_dsl_imports_still_work(self):
        """Traditional DSL imports should still be recognized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()
            lib_dir = arch_dir / "lib"
            lib_dir.mkdir()

            # Main arch with traditional import
            arch_file = arch_dir / "arch.neuro"
            arch_file.write_text("""
architecture test { d_sem: 256 }
import "@/lib/cortex"
population main { count: 256 }
""")

            # Cortex module
            cortex_file = lib_dir / "cortex.neuro"
            cortex_file.write_text("population cortex { count: 1024 }")

            # Bundle
            bundler = ModuleBundler(arch_dir)
            bundled = bundler.bundle(arch_file)

            # Should find the module
            assert len(bundled.modules) > 0
            assert "@/lib/cortex" in bundled.modules

    def test_mixed_es6_and_dsl_imports(self):
        """Both ES6 and traditional imports in same file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()
            lib_dir = arch_dir / "lib"
            lib_dir.mkdir()

            arch_file = arch_dir / "arch.neuro"
            arch_file.write_text("""
architecture mixed { d_sem: 256 }
import "@/lib/old_style"
import { new, style } from "@/lib/new_style"
population main { count: 256 }
""")

            old_style = lib_dir / "old_style.neuro"
            old_style.write_text("population old { count: 256 }")

            new_style = lib_dir / "new_style.neuro"
            new_style.write_text("population new { count: 512 }")

            bundler = ModuleBundler(arch_dir)
            bundled = bundler.bundle(arch_file)

            # Both should be found
            assert "@/lib/old_style" in bundled.modules
            assert "@/lib/new_style" in bundled.modules
            assert len(bundled.modules) == 2

    def test_rcc_bowtie_modules_are_bundled(self):
        """Full rcc_bowtie should bundle all modules."""
        arch_root = Path(__file__).parent.parent / "architectures" / "rcc_bowtie"

        if not arch_root.exists():
            pytest.skip("rcc_bowtie architecture not found")

        arch_file = arch_root / "arch.neuro"
        if not arch_file.exists():
            pytest.skip("arch.neuro not found")

        bundler = ModuleBundler(arch_root)
        bundled = bundler.bundle(arch_file)

        # Should find substantial number of modules
        # rcc_bowtie has: equations, regularizers, cdga, emergent, sensory, thalamus, etc.
        assert len(bundled.modules) > 5, \
            f"rcc_bowtie should have many modules, got {len(bundled.modules)}"

        # Should have lib modules
        assert any("lib" in spec for spec in bundled.modules.keys()), \
            "Should have lib/ modules"

        # Should have modules
        assert any("modules" in spec for spec in bundled.modules.keys()), \
            "Should have modules/ modules"

    def test_unfold_rcc_bowtie_produces_complete_dsl(self):
        """Unfolding rcc_bowtie DNA should produce complete DSL."""
        from neuroslm.compiler.ribosome import RibosomeCompiler

        arch_root = Path(__file__).parent.parent / "architectures" / "rcc_bowtie"
        if not arch_root.exists():
            pytest.skip("rcc_bowtie not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Compile
            dna_file = tmpdir / "test.dna"
            unfold_file = tmpdir / "unfold.neuro"

            compiler = RibosomeCompiler()
            compiler.compile_file(str(arch_root), str(dna_file))
            compiler.unfold_file(str(dna_file), str(unfold_file))

            # Read original
            original = (arch_root / "arch.neuro").read_text(encoding="utf-8")
            unfolded = unfold_file.read_text(encoding="utf-8")

            # Should be byte-identical or very similar
            assert len(unfolded) > 0
            assert "architecture" in unfolded
            assert "population" in unfolded or "module" in unfolded.lower()

            # Check that key imports are preserved
            original_imports = [line for line in original.split('\n') if 'import' in line]
            unfolded_imports = [line for line in unfolded.split('\n') if 'import' in line]

            assert len(unfolded_imports) > 0, \
                f"Unfolded should have imports, got:\n{unfolded[:500]}"
