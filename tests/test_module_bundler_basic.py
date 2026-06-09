# -*- coding: utf-8 -*-
"""Basic tests for module bundler without pytest runner dependency."""
import tempfile
import json
from pathlib import Path

from neuroslm.compiler.module_bundler import ModuleBundler, SourceMap


def test_module_bundler_simple():
    """Test bundler can resolve and collect imports."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create architecture
        arch_dir = tmpdir / "arch"
        arch_dir.mkdir()
        lib_dir = arch_dir / "lib"
        lib_dir.mkdir()

        # Main file
        main_file = arch_dir / "arch.neuro"
        main_file.write_text("""
architecture test {
    d_sem: 256
}

import "@/lib/utils"

population main { count: 256 }
""")

        # Library file
        lib_file = lib_dir / "utils.neuro"
        lib_file.write_text("""
population utils { count: 512 }
""")

        # Bundle
        bundler = ModuleBundler(arch_dir)
        bundled = bundler.bundle(main_file)

        # Verify
        assert bundled.main_source is not None
        assert "main" in bundled.main_source
        assert len(bundled.modules) > 0
        assert "@/lib/utils" in bundled.modules

        print("✓ Module bundler collects imports correctly")


def test_source_map_generation():
    """Test that source maps track module origins."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        arch_dir = tmpdir / "arch"
        arch_dir.mkdir()
        lib_dir = arch_dir / "lib"
        lib_dir.mkdir()

        main_file = arch_dir / "arch.neuro"
        main_file.write_text("""
architecture test { d_sem: 256 }
import "@/lib/cortex"
population entry { count: 256 }
""")

        cortex_file = lib_dir / "cortex.neuro"
        cortex_file.write_text("population cortex { count: 1024 }")

        bundler = ModuleBundler(arch_dir)
        bundled = bundler.bundle(main_file)

        # Check source map exists
        assert bundled.source_map is not None

        # Convert to dict (like it would be stored in DNA)
        sm_dict = bundled.source_map.to_dict()
        assert "line_to_module" in sm_dict
        assert "module_to_lines" in sm_dict
        assert "section_offsets" in sm_dict

        print("✓ Source maps generated correctly")


def test_bundled_dsl_serialization():
    """Test that bundled DSL can be serialized and deserialized."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        arch_dir = tmpdir / "arch"
        arch_dir.mkdir()
        lib_dir = arch_dir / "lib"
        lib_dir.mkdir()

        main_file = arch_dir / "arch.neuro"
        main_file.write_text("""
architecture main { d_sem: 256 }
import "@/lib/dynamics"
population gws { count: 256 }
""")

        dyn_file = lib_dir / "dynamics.neuro"
        dyn_file.write_text("neurotransmitter dopamine { base_concentration: 0.5 }")

        bundler = ModuleBundler(arch_dir)
        bundled = bundler.bundle(main_file)

        # Serialize
        bundled_dict = bundled.to_dict()
        assert "main_source" in bundled_dict
        assert "modules" in bundled_dict

        # Deserialize
        from neuroslm.compiler.module_bundler import BundledDSL
        bundled2 = BundledDSL.from_dict(bundled_dict)

        assert bundled2.main_source == bundled.main_source
        assert len(bundled2.modules) == len(bundled.modules)

        print("✓ Bundled DSL serialization works")


def test_inline_imports():
    """Test that imports can be inlined."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        arch_dir = tmpdir / "arch"
        arch_dir.mkdir()
        lib_dir = arch_dir / "lib"
        lib_dir.mkdir()

        main_file = arch_dir / "arch.neuro"
        main_file.write_text("""
architecture main { d_sem: 256 }
import "@/lib/extra"
population main { count: 256 }
""")

        extra_file = lib_dir / "extra.neuro"
        extra_file.write_text("neurotransmitter serotonin { base_concentration: 0.3 }")

        bundler = ModuleBundler(arch_dir)
        bundled = bundler.bundle(main_file)

        # Inline
        inlined = bundled.inline_imports()
        assert "serotonin" in inlined or "extra" in inlined.lower()

        print("✓ Import inlining works")


if __name__ == "__main__":
    test_module_bundler_simple()
    test_source_map_generation()
    test_bundled_dsl_serialization()
    test_inline_imports()
    print("\nAll basic tests passed! ✓")
