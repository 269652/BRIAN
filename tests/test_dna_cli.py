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

            # Read recovered DSL
            recovered = Path(neuro_path).read_text()
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
            content = Path(neuro_path).read_text()
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

            recovered_dsl = Path(neuro_path).read_text()

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
