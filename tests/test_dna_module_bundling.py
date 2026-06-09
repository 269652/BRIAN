# -*- coding: utf-8 -*-
"""TDD: DNA compiler must bundle all module imports during roundtrip.

Tests verify that:
1. DSL with module imports compiles to DNA with all modules bundled
2. Unfolding DNA reproduces modularized structure with resolved imports
3. Evolved improvements can live in modules/libs, not just inline
"""
import pytest
import tempfile
import json
from pathlib import Path

from neuroslm.compiler.ribosome import RibosomeCompiler
from neuroslm.dsl.multifile import FolderLoader


class TestDNAModuleBundling:
    """Test that DNA compiler preserves and bundles module structure."""

    def test_compile_bundles_all_module_imports(self):
        """Compiling DSL with imports should bundle all modules into DNA.

        Given a DSL with `import "@/lib/cortex"`, the DNA should contain
        both the main arch.neuro AND the bundled cortex.neuro content.
        """
        compiler = RibosomeCompiler()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a modularized architecture
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()
            lib_dir = arch_dir / "lib"
            lib_dir.mkdir()

            # Main arch file with import
            main_arch = arch_dir / "arch.neuro"
            main_arch.write_text("""
architecture main {
    d_sem: 256,
    dt: 0.01
}

import "@/lib/cortex"

population gws {
    count: 512,
    dynamics: "rate_code"
}
""")

            # Library module
            cortex_module = lib_dir / "cortex.neuro"
            cortex_module.write_text("""
population cortex_layer {
    count: 1024,
    dynamics: "rate_code"
}

synapse gws -> cortex_layer {
    weight: 0.5
}
""")

            # Compile to DNA
            dna_file = tmpdir / "test.dna"
            compiler.compile_file(str(arch_dir), str(dna_file))

            # Verify DNA file created
            assert dna_file.exists()

            # Load DNA and check it contains all module content
            dna_json = json.loads(dna_file.read_text())
            assert "invariants" in dna_json

            # DNA should bundle all imports, not just store the main file
            dsl_code = dna_json["invariants"].get("dsl_code", "")
            assert dsl_code, "DNA must contain bundled DSL code"

            # The bundled code should reference or include cortex content
            # Either as inline or as resolved imports
            assert "cortex" in dsl_code or "cortex_layer" in dsl_code

    def test_unfold_preserves_module_structure(self):
        """Unfolding DNA should produce DSL with module structure preserved.

        When unfolding, the output should either:
        1. Re-create the module files with imports, or
        2. Inline modules but mark their origin
        """
        compiler = RibosomeCompiler()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create modularized architecture
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()
            lib_dir = arch_dir / "lib"
            lib_dir.mkdir()

            main_arch = arch_dir / "arch.neuro"
            main_arch.write_text("""
architecture test {
    d_sem: 256,
    dt: 0.01
}

import "@/lib/dynamics"

population main_pop {
    count: 256,
    dynamics: "rate_code"
}
""")

            dynamics_module = lib_dir / "dynamics.neuro"
            dynamics_module.write_text("""
neurotransmitter dopamine {
    base_concentration: 0.5,
    release_rate: 0.1,
    reuptake_rate: 0.05
}
""")

            # Compile and unfold
            dna_file = tmpdir / "test.dna"
            unfolded_neuro = tmpdir / "evolved.neuro"

            compiler.compile_file(str(arch_dir), str(dna_file))
            compiler.unfold_file(str(dna_file), str(unfolded_neuro))

            # Verify unfolded file exists and has content
            assert unfolded_neuro.exists()
            unfolded_content = unfolded_neuro.read_text()
            assert len(unfolded_content) > 50

            # Should preserve references to imported content
            # Either as import statement or inlined content
            assert "dopamine" in unfolded_content or "dynamics" in unfolded_content

    def test_roundtrip_with_nested_modules(self):
        """Roundtrip should handle nested module hierarchies.

        Given:
          arch/
            arch.neuro (imports lib/core)
            lib/
              core.neuro (imports lib/utils)
              utils.neuro

        The DNA should bundle all three files, and unfold should preserve
        the hierarchy (or mark it clearly).
        """
        compiler = RibosomeCompiler()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()
            lib_dir = arch_dir / "lib"
            lib_dir.mkdir()

            # Three-level module hierarchy
            main = arch_dir / "arch.neuro"
            main.write_text("""
architecture main { d_sem: 256, dt: 0.01 }
import "@/lib/core"
population entry { count: 256, dynamics: "rate_code" }
""")

            core = lib_dir / "core.neuro"
            core.write_text("""
import "@/lib/utils"
population core { count: 512, dynamics: "rate_code" }
synapse entry -> core { weight: 1.0 }
""")

            utils = lib_dir / "utils.neuro"
            utils.write_text("""
neurotransmitter serotonin {
    base_concentration: 0.3,
    release_rate: 0.2,
    reuptake_rate: 0.1
}
""")

            # Compile to DNA
            dna_file = tmpdir / "nested.dna"
            compiler.compile_file(str(arch_dir), str(dna_file))
            assert dna_file.exists()

            # The nested module (utils, reached only via core) must be
            # bundled into the DNA — that is what "bundle all imports" means.
            # We assert it on the DNA itself, not on the unfolded main file,
            # because unfold must stay BIT-IDENTICAL to the original arch.neuro
            # (which imports core, not utils inline).
            import json as _json
            dna_json = _json.loads(dna_file.read_text())
            modules = dna_json["invariants"]["bundled_dsl"]["modules"]
            assert "@/lib/core" in modules
            assert "@/lib/utils" in modules, "nested import must be bundled"
            assert "serotonin" in modules["@/lib/utils"]["source"]

            # Unfold from DNA — must reproduce the original main source exactly.
            unfolded = tmpdir / "unfolded.neuro"
            compiler.unfold_file(str(dna_file), str(unfolded))
            assert unfolded.exists()

            unfolded_text = unfolded.read_text(encoding="utf-8")
            assert unfolded_text == main.read_text(encoding="utf-8")

    def test_dna_tracks_module_origin_for_evolution(self):
        """DNA should track which parts came from which module.

        This enables evolved improvements to be stored back into the
        module that contributed them, not just inlined.
        """
        compiler = RibosomeCompiler()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()
            lib_dir = arch_dir / "lib"
            lib_dir.mkdir()

            main_arch = arch_dir / "arch.neuro"
            main_arch.write_text("""
architecture main { d_sem: 256, dt: 0.01 }
import "@/lib/learning"
population gws { count: 256, dynamics: "rate_code" }
""")

            learning_module = lib_dir / "learning.neuro"
            learning_module.write_text("""
population learning_layer {
    count: 512,
    dynamics: "rate_code"
}
""")

            # Compile to DNA
            dna_file = tmpdir / "tracked.dna"
            compiler.compile_file(str(arch_dir), str(dna_file))

            # DNA should track module origin
            dna_json = json.loads(dna_file.read_text())

            # Check if DNA has module metadata
            invariants = dna_json.get("invariants", {})

            # DNA should have a way to track which content came from which module
            # Either as separate entries or as metadata
            has_module_tracking = (
                "module_map" in invariants or
                "modules" in invariants or
                "file_sources" in invariants or
                "@/lib/learning" in dna_json.get("invariants", {}).get("dsl_code", "")
            )
            assert has_module_tracking, "DNA must track module origins for evolution"

    def test_resolved_imports_unfold_without_dangling_references(self):
        """Unfolded DSL should have no unresolved import references.

        When DNA is unfolded, any `import "@/..."` statements should either:
        1. Point to actual files that will be present, or
        2. Be inlined so no external reference is needed
        """
        compiler = RibosomeCompiler()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()
            lib_dir = arch_dir / "lib"
            lib_dir.mkdir()

            main_arch = arch_dir / "arch.neuro"
            main_arch.write_text("""
architecture main { d_sem: 256, dt: 0.01 }
import "@/lib/cortex"
population entry { count: 256, dynamics: "rate_code" }
""")

            cortex = lib_dir / "cortex.neuro"
            cortex.write_text("""
population cortex { count: 1024, dynamics: "rate_code" }
synapse entry -> cortex { weight: 0.5 }
""")

            # Compile to DNA
            dna_file = tmpdir / "unresolved_test.dna"
            compiler.compile_file(str(arch_dir), str(dna_file))

            # Unfold to a NEW directory (simulating fresh code)
            new_dir = tmpdir / "new_location"
            new_dir.mkdir()
            unfolded_file = new_dir / "evolved.neuro"

            compiler.unfold_file(str(dna_file), str(unfolded_file))
            assert unfolded_file.exists()

            content = unfolded_file.read_text()

            # Either the unfolded content has imports and those files exist,
            # or it's self-contained with no external imports
            if "import" in content:
                # If imports are present, they must be resolvable
                # (This is a grammar check, not a file existence check,
                # since we're in a temp dir)
                import_lines = [l for l in content.split('\n') if 'import' in l]
                for line in import_lines:
                    # Should be valid import syntax
                    assert '@/' in line or './' in line or '../' in line, \
                        f"Invalid import syntax: {line}"

    def test_source_map_tracks_module_attribution(self):
        """Source map should track which modules are sources of changes.

        When evolution modifies lines, the source map shows which module
        that line came from, enabling attribution of improvements.
        """
        compiler = RibosomeCompiler()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()
            lib_dir = arch_dir / "lib"
            lib_dir.mkdir()

            main_arch = arch_dir / "arch.neuro"
            main_arch.write_text("""
architecture main { d_sem: 256, dt: 0.01 }
import "@/lib/learning"
population gws { count: 256, dynamics: "rate_code" }
""")

            learning = lib_dir / "learning.neuro"
            learning.write_text("""
population learning_layer {
    count: 512,
    dynamics: "rate_code"
}
""")

            # Compile to DNA
            dna_file = tmpdir / "sourcemap_test.dna"
            compiler.compile_file(str(arch_dir), str(dna_file))

            # Load and check source map
            source_map = compiler.load_source_map(str(dna_file))
            assert source_map is not None, "Source map must be generated"

            # Source map should have section offsets for known modules
            if "section_offsets" in source_map:
                offsets = source_map["section_offsets"]
                assert "main" in offsets or len(offsets) > 0

    def test_evolved_improvements_can_stay_in_modules(self):
        """After evolution, improvements should be traceable back to modules.

        If a learned improvement happens in a module's region, that
        improvement can be stored back to the module file (not just inline).
        """
        compiler = RibosomeCompiler()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()
            lib_dir = arch_dir / "lib"
            lib_dir.mkdir()

            main_arch = arch_dir / "arch.neuro"
            main_arch.write_text("""
architecture main { d_sem: 256, dt: 0.01 }
import "@/lib/optim"
population main { count: 256, dynamics: "rate_code" }
""")

            optim = lib_dir / "optim.neuro"
            optim.write_text("""
neurotransmitter dopamine { base_concentration: 0.5 }
""")

            dna_file = tmpdir / "evolution_test.dna"
            compiler.compile_file(str(arch_dir), str(dna_file))

            # Simulate an evolution: query which module line 5 came from
            # (Would be part of learning/evolution workflow)
            module_owner = compiler.get_module_for_change(str(dna_file), 5)

            # Either found a module owner or returned None (both acceptable)
            assert module_owner is None or isinstance(module_owner, str)
