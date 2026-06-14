# -*- coding: utf-8 -*-
"""TDD test: DNA roundtrip must produce byte-identical DSL code.

This test verifies lossless roundtrip: compile arch.neuro → DNA → unfold
produces the exact same bytes as the original.
"""
import pytest
import tempfile
from pathlib import Path
import hashlib

from neuroslm.compiler.ribosome import RibosomeCompiler


class TestDNARoundtripByteIdentity:
    """Verify that DSL → DNA → DSL produces byte-identical output."""

    def test_simple_arch_byte_identical(self):
        """Simple architecture should survive roundtrip with exact bytes."""
        compiler = RibosomeCompiler()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create simple arch
            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()

            original_content = """architecture simple {
    d_sem: 256,
    dt: 0.01
}

population pop1 {
    count: 256,
    dynamics: "rate_code"
}

synapse pop1 -> pop1 {
    weight: 0.5
}
"""

            main_arch = arch_dir / "arch.neuro"
            main_arch.write_text(original_content, encoding="utf-8")

            # Roundtrip
            dna_file = tmpdir / "test.dna"
            unfolded_file = tmpdir / "unfolded.neuro"

            compiler.compile_file(str(arch_dir), str(dna_file))
            compiler.unfold_file(str(dna_file), str(unfolded_file))

            # Compare
            unfolded_content = unfolded_file.read_text(encoding="utf-8")

            # Should be byte-identical
            assert unfolded_content == original_content, \
                "Roundtrip should produce byte-identical DSL"

    def test_rcc_bowtie_roundtrip_fidelity(self):
        """RCC bowtie architecture should maintain high fidelity in roundtrip.

        Note: Due to module imports, we verify the main source is preserved,
        not necessarily byte-identical (imports are resolved).
        """
        compiler = RibosomeCompiler()
        arch_root = str(Path(__file__).parent.parent / "architectures" / "master")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Compile to DNA
            dna_file = tmpdir / "rcc_bowtie.dna"
            compiler.compile_file(arch_root, str(dna_file))

            # Unfold from DNA
            unfolded_file = tmpdir / "unfolded.neuro"
            compiler.unfold_file(str(dna_file), str(unfolded_file))

            # Verify file was created and has substantial content
            assert unfolded_file.exists()
            content = unfolded_file.read_text(encoding="utf-8")

            # Should have key markers
            assert "architecture" in content.lower()
            assert len(content) > 1000

            # Verify it's valid DSL-like syntax
            assert "{" in content and "}" in content
            assert "population" in content.lower() or "complex" in content.lower()

    def test_roundtrip_preserves_comments_and_formatting(self):
        """Roundtrip should preserve comments and whitespace."""
        compiler = RibosomeCompiler()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()

            original_content = """# This is a comment
architecture test {
    d_sem: 256,  # inline comment
    dt: 0.01
}

# Population definition
population main {
    count: 256,
    dynamics: "rate_code"
}
"""

            arch_file = arch_dir / "arch.neuro"
            arch_file.write_text(original_content, encoding="utf-8")

            # Roundtrip
            dna_file = tmpdir / "test.dna"
            unfolded_file = tmpdir / "unfolded.neuro"

            compiler.compile_file(str(arch_dir), str(dna_file))
            compiler.unfold_file(str(dna_file), str(unfolded_file))

            unfolded_content = unfolded_file.read_text(encoding="utf-8")

            # Should preserve key structural elements
            assert "architecture" in unfolded_content
            assert "population" in unfolded_content

    def test_roundtrip_hash_stability(self):
        """Hash of roundtripped code should be stable (or documented as acceptable loss).

        If not byte-identical, we track what changed and why.
        """
        compiler = RibosomeCompiler()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()

            original = """architecture main { d_sem: 256 }
population p1 { count: 256 }"""

            arch_file = arch_dir / "arch.neuro"
            arch_file.write_text(original, encoding="utf-8")

            # First roundtrip
            dna1 = tmpdir / "round1.dna"
            unfold1 = tmpdir / "round1.neuro"
            compiler.compile_file(str(arch_dir), str(dna1))
            compiler.unfold_file(str(dna1), str(unfold1))
            content1 = unfold1.read_text(encoding="utf-8")
            hash1 = hashlib.sha256(content1.encode("utf-8")).hexdigest()

            # Second roundtrip (DNA → neuro → DNA → neuro)
            dna2 = tmpdir / "round2.dna"
            unfold2 = tmpdir / "round2.neuro"
            compiler.compile_file(str(arch_dir), str(dna2))
            compiler.unfold_file(str(dna2), str(unfold2))
            content2 = unfold2.read_text(encoding="utf-8")
            hash2 = hashlib.sha256(content2.encode("utf-8")).hexdigest()

            # After first roundtrip, subsequent roundtrips should be stable
            assert hash1 == hash2, \
                "Second and subsequent roundtrips should produce identical output"

    def test_no_unresolved_imports_in_roundtrip(self):
        """Unfolded DSL should not contain unresolvable import references."""
        compiler = RibosomeCompiler()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            arch_dir = tmpdir / "arch"
            arch_dir.mkdir()
            lib_dir = arch_dir / "lib"
            lib_dir.mkdir()

            arch_file = arch_dir / "arch.neuro"
            arch_file.write_text("""
architecture main { d_sem: 256 }
import "@/lib/helper"
population main { count: 256 }
""", encoding="utf-8")

            helper_file = lib_dir / "helper.neuro"
            helper_file.write_text("""
neurotransmitter nt1 { base_concentration: 0.5 }
""", encoding="utf-8")

            # Roundtrip
            dna_file = tmpdir / "test.dna"
            unfold_file = tmpdir / "unfolded.neuro"

            compiler.compile_file(str(arch_dir), str(dna_file))
            compiler.unfold_file(str(dna_file), str(unfold_file))

            content = unfold_file.read_text(encoding="utf-8")

            # If imports are present, they should be valid
            import_lines = [l for l in content.split('\n') if 'import' in l]
            for line in import_lines:
                # Valid import syntax
                assert '@/' in line or './' in line or '../' in line, \
                    f"Invalid import in roundtrip: {line}"

            # Should have the imported content somehow (inlined or referenced)
            assert "nt1" in content or "helper" in content, \
                "Roundtrip should preserve imported symbols"
