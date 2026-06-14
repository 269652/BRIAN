# -*- coding: utf-8 -*-
"""TDD: CLI commands should show help when called with missing args, not crash.

Tests verify that all CLI entry points handle missing arguments gracefully.
"""
import pytest
import subprocess
import sys
from pathlib import Path


class TestCLICommands:
    """Test that CLI commands behave correctly with missing/invalid args."""

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "neuroslm" / "cli.py").exists(),
        reason="CLI module not found"
    )
    def test_brian_compile_with_no_args_shows_help(self):
        """brian compile (no args) should show help, not crash."""
        result = subprocess.run(
            [sys.executable, "-m", "neuroslm.cli", "compile"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        # Should either show help or exit gracefully with error message
        assert result.returncode != 0, "Should fail with missing required arg"
        output = result.stdout + result.stderr
        assert ("usage" in output.lower() or "required" in output.lower() or
                "error" in output.lower()), \
            "Should show help/usage when arg is missing, not crash"

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "neuroslm" / "cli.py").exists(),
        reason="CLI module not found"
    )
    def test_brian_dna_with_no_subcommand_shows_help(self):
        """brian dna (no subcommand) should show help, not crash."""
        result = subprocess.run(
            [sys.executable, "-m", "neuroslm.cli", "dna"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        # Should either show help or exit gracefully with error message
        assert result.returncode != 0, "Should fail with missing subcommand"
        output = result.stdout + result.stderr
        assert ("usage" in output.lower() or "required" in output.lower() or
                "error" in output.lower()), \
            "Should show help/usage when subcommand is missing"

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "neuroslm" / "cli.py").exists(),
        reason="CLI module not found"
    )
    def test_brian_dna_compile_requires_arch(self):
        """brian dna compile (no arch) should show help."""
        result = subprocess.run(
            [sys.executable, "-m", "neuroslm.cli", "dna", "compile"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode != 0, "Should fail with missing arch arg"
        output = result.stdout + result.stderr
        assert ("usage" in output.lower() or "required" in output.lower() or
                "error" in output.lower() or "arch" in output.lower()), \
            "Should show help/error about missing arch"

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "neuroslm" / "cli.py").exists(),
        reason="CLI module not found"
    )
    def test_brian_dna_unfold_requires_dna_file(self):
        """brian dna unfold (no file) should show help."""
        result = subprocess.run(
            [sys.executable, "-m", "neuroslm.cli", "dna", "unfold"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode != 0, "Should fail with missing dna arg"
        output = result.stdout + result.stderr
        assert ("usage" in output.lower() or "required" in output.lower() or
                "error" in output.lower() or "dna" in output.lower()), \
            "Should show help/error about missing DNA file"

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "architectures" / "master").exists(),
        reason="master architecture not found"
    )
    def test_brian_dna_compile_rcc_bowtie_succeeds(self):
        """brian dna compile master should work without crashing."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "test.dna")

            result = subprocess.run(
                [sys.executable, "-m", "neuroslm.cli", "dna", "compile",
                 "master", "--output", output_file],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent),
                timeout=30,
            )

            assert result.returncode == 0, \
                f"DNA compile failed:\n{result.stderr}"
            assert Path(output_file).exists(), f"Output file not created: {output_file}"
            assert Path(output_file).stat().st_size > 0, "DNA file is empty"

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "dna" / "evol" / "arch.dna").exists(),
        reason="dna/evol/arch.dna not found"
    )
    def test_brian_dna_unfold_succeeds(self):
        """brian dna unfold should work without crashing."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            dna_file = str(Path(__file__).parent.parent / "dna" / "evol" / "arch.dna")
            output_file = os.path.join(tmpdir, "unfolded.neuro")

            result = subprocess.run(
                [sys.executable, "-m", "neuroslm.cli", "dna", "unfold",
                 dna_file, "--output", output_file],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent),
                timeout=30,
            )

            assert result.returncode == 0, \
                f"DNA unfold failed:\n{result.stderr}"
            assert Path(output_file).exists(), f"Output file not created: {output_file}"
            assert Path(output_file).stat().st_size > 0, "Unfolded file is empty"

    def test_brian_compile_rcc_bowtie_succeeds(self):
        """brian compile master should succeed (compile DSL to Python)."""
        result = subprocess.run(
            [sys.executable, "-m", "neuroslm.cli", "compile", "master"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            timeout=30,
        )

        assert result.returncode == 0, \
            f"Compile failed:\n{result.stderr}"
        # Should output generated code
        assert len(result.stdout) > 100, "Should output generated code"
