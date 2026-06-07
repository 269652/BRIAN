# -*- coding: utf-8 -*-
"""Tests for DNA CLI commands (brian dna compile/unfold)."""
import pytest
import tempfile
from pathlib import Path
import sys
import argparse

from neuroslm.compiler.ribosome import RibosomeCompiler


class TestDNACompile:
    """Test DNA compilation from arch.neuro to binary."""

    def test_dna_compile_creates_file(self):
        """Compiling arch should create a .dna file."""
        compiler = RibosomeCompiler()
        arch_root = str(Path(__file__).parent.parent / "architectures" / "rcc_bowtie")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dna = str(Path(tmpdir) / "test.dna")
            compiler.compile_file(arch_root, output_dna)

            # File should exist
            assert Path(output_dna).exists()
            assert Path(output_dna).stat().st_size > 0

    def test_dna_compile_roundtrip(self):
        """Compile → unfold → should recover original (approximately)."""
        compiler = RibosomeCompiler()
        arch_root = str(Path(__file__).parent.parent / "architectures" / "rcc_bowtie")

        with tempfile.TemporaryDirectory() as tmpdir:
            dna_path = str(Path(tmpdir) / "test.dna")
            neuro_path = str(Path(tmpdir) / "test.neuro")

            # Compile
            compiler.compile_file(arch_root, dna_path)
            assert Path(dna_path).exists()

            # Unfold
            compiler.unfold_file(dna_path, neuro_path)
            assert Path(neuro_path).exists()

            # Read recovered DSL (with UTF-8 encoding)
            recovered = Path(neuro_path).read_text(encoding='utf-8')
            assert "architecture" in recovered or "population" in recovered or "complex" in recovered


class TestDNAUnfold:
    """Test DNA unfolding from binary back to DSL."""

    def test_dna_unfold_creates_file(self):
        """Unfolding DNA should create a .neuro file."""
        compiler = RibosomeCompiler()
        arch_root = str(Path(__file__).parent.parent / "architectures" / "rcc_bowtie")

        with tempfile.TemporaryDirectory() as tmpdir:
            dna_path = str(Path(tmpdir) / "test.dna")
            neuro_path = str(Path(tmpdir) / "test.neuro")

            compiler.compile_file(arch_root, dna_path)
            compiler.unfold_file(dna_path, neuro_path)

            # File should exist and be readable
            assert Path(neuro_path).exists()
            content = Path(neuro_path).read_text(encoding='utf-8')
            assert len(content) > 0

    def test_dna_unfold_produces_valid_dsl(self):
        """Unfolded DSL should parse (approximately) as valid architecture."""
        compiler = RibosomeCompiler()
        arch_root = str(Path(__file__).parent.parent / "architectures" / "rcc_bowtie")

        with tempfile.TemporaryDirectory() as tmpdir:
            dna_path = str(Path(tmpdir) / "test.dna")
            neuro_path = str(Path(tmpdir) / "test.neuro")

            compiler.compile_file(arch_root, dna_path)
            compiler.unfold_file(dna_path, neuro_path)

            recovered_dsl = Path(neuro_path).read_text(encoding='utf-8')

            # Should have some basic structure markers
            has_architecture = "architecture" in recovered_dsl
            has_population = "population" in recovered_dsl
            has_complex = "complex" in recovered_dsl

            # At least one of these should be present
            assert has_architecture or has_population or has_complex


class TestDNACLIIntegration:
    """Test DNA CLI commands via CLI parser."""

    def test_dna_compile_via_cli(self):
        """Test `brian dna compile` command."""
        from neuroslm.cli import cmd_dna

        arch_root = str(Path(__file__).parent.parent / "architectures" / "rcc_bowtie")

        with tempfile.TemporaryDirectory() as tmpdir:
            dna_path = str(Path(tmpdir) / "test.dna")

            # Create a mock argparse.Namespace for compile
            args = argparse.Namespace(
                dna_cmd="compile",
                arch=arch_root,
                output=dna_path
            )

            # Should return 0 (success)
            result = cmd_dna(args)
            assert result == 0
            assert Path(dna_path).exists()

    def test_dna_unfold_via_cli(self):
        """Test `brian dna unfold` command."""
        from neuroslm.cli import cmd_dna

        arch_root = str(Path(__file__).parent.parent / "architectures" / "rcc_bowtie")

        with tempfile.TemporaryDirectory() as tmpdir:
            dna_path = str(Path(tmpdir) / "test.dna")
            neuro_path = str(Path(tmpdir) / "test.neuro")

            # First compile
            compile_args = argparse.Namespace(
                dna_cmd="compile",
                arch=arch_root,
                output=dna_path
            )
            assert cmd_dna(compile_args) == 0

            # Then unfold
            unfold_args = argparse.Namespace(
                dna_cmd="unfold",
                dna=dna_path,
                output=neuro_path
            )
            result = cmd_dna(unfold_args)
            assert result == 0
            assert Path(neuro_path).exists()

    def test_dna_compile_arch_shortname(self):
        """Test `brian dna compile` with short arch name."""
        from neuroslm.cli import cmd_dna

        with tempfile.TemporaryDirectory() as tmpdir:
            dna_path = str(Path(tmpdir) / "test.dna")

            # Use short name "rcc_bowtie" instead of full path
            args = argparse.Namespace(
                dna_cmd="compile",
                arch="rcc_bowtie",
                output=dna_path
            )

            result = cmd_dna(args)
            assert result == 0
            assert Path(dna_path).exists()


class TestDNAUnfoldDirectoryDestination:
    """Regression: `brian dna unfold evol.dna --output some/dir/` must treat
    a trailing-separator path (or an existing directory) as a directory
    destination and write `<dna_stem>.neuro` inside it.

    Original failure:
        brian dna unfold .\\dna\\evol.dna --output .\\architectures\\evol\\
        ✗ Unfold failed: [Errno 22] Invalid argument: '.\\architectures\\evol\\'

    Root cause: the CLI passed the trailing-slash path straight to
    `open()` which on Windows fails with Errno 22 because the path
    looks like a file but ends with a separator.
    """

    def _make_dna(self, tmpdir: Path) -> Path:
        """Compile a small DNA file we can repeatedly unfold."""
        compiler = RibosomeCompiler()
        arch_root = str(Path(__file__).parent.parent / "architectures" / "rcc_bowtie")
        dna_path = tmpdir / "evol.dna"
        compiler.compile_file(arch_root, str(dna_path))
        assert dna_path.exists()
        return dna_path

    def test_unfold_to_trailing_separator_path_treats_as_directory(self):
        """`--output some/dir/` (trailing sep, does not exist) ⇒ create
        the directory and write `<dna_stem>.neuro` inside it."""
        from neuroslm.cli import cmd_dna

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            dna_path = self._make_dna(tmpdir)

            # Path-with-trailing-separator (the exact failure mode from
            # the user's command line). Use os.sep so the test works
            # on both Windows (`\`) and POSIX (`/`).
            import os
            out_dir_str = str(tmpdir / "evol") + os.sep
            args = argparse.Namespace(
                dna_cmd="unfold",
                dna=str(dna_path),
                output=out_dir_str,
            )

            rc = cmd_dna(args)
            assert rc == 0, "CLI must not crash on directory destination"
            # Expected file: <out_dir>/<dna_stem>.neuro
            expected = tmpdir / "evol" / "evol.neuro"
            assert expected.exists(), (
                f"unfolded file not written to expected path {expected}; "
                f"contents of {tmpdir / 'evol'}: "
                f"{list((tmpdir / 'evol').iterdir()) if (tmpdir / 'evol').exists() else 'directory missing'}"
            )
            assert expected.read_text(encoding="utf-8"), \
                "unfolded .neuro file is empty"

    def test_unfold_to_existing_directory_treats_as_directory(self):
        """`--output some/existing/dir` (no trailing sep, but already a
        directory) ⇒ also treat as directory destination, same result."""
        from neuroslm.cli import cmd_dna

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            dna_path = self._make_dna(tmpdir)

            out_dir = tmpdir / "evol_existing"
            out_dir.mkdir()
            args = argparse.Namespace(
                dna_cmd="unfold",
                dna=str(dna_path),
                output=str(out_dir),   # no trailing separator
            )

            rc = cmd_dna(args)
            assert rc == 0
            expected = out_dir / "evol.neuro"
            assert expected.exists(), \
                f"unfolded file not written to expected path {expected}"

    def test_unfold_to_explicit_file_path_still_works(self):
        """Regression guard for the file path: `--output some/dir/foo.neuro`
        (explicit `.neuro` extension, parent dir exists) must continue
        to land at that exact file — no directory-coercion regression."""
        from neuroslm.cli import cmd_dna

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            dna_path = self._make_dna(tmpdir)

            sub = tmpdir / "sub"
            sub.mkdir()
            out_file = sub / "custom_name.neuro"
            args = argparse.Namespace(
                dna_cmd="unfold",
                dna=str(dna_path),
                output=str(out_file),
            )

            rc = cmd_dna(args)
            assert rc == 0
            assert out_file.exists(), "explicit file destination broken"
            # Confirm we did NOT also create a file named `custom_name.neuro/`
            # inside (i.e. didn't misclassify as directory).
            assert out_file.is_file()
