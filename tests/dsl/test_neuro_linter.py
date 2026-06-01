# -*- coding: utf-8 -*-
"""Tests for the .neuro DSL linter."""
import pytest
from pathlib import Path
from neuroslm.dsl.neuro_linter import NeuroLinter, Severity, lint_file


@pytest.fixture
def tmp_neuro_file(tmp_path):
    """Create a temporary .neuro file."""
    def _make(content: str):
        f = tmp_path / "test.neuro"
        f.write_text(content)
        return f
    return _make


class TestBraceMatching:
    """Test brace/bracket/paren matching."""

    def test_valid_braces(self, tmp_neuro_file):
        """Valid nested braces."""
        content = """
architecture test {
    d_sem: 256
}
"""
        f = tmp_neuro_file(content)
        diags = lint_file(f)
        assert not any(d.severity == Severity.ERROR for d in diags)

    def test_unmatched_closing_brace(self, tmp_neuro_file):
        """Unmatched closing brace detected."""
        content = """
architecture test {
    d_sem: 256
}}
"""
        f = tmp_neuro_file(content)
        diags = lint_file(f)
        errors = [d for d in diags if d.severity == Severity.ERROR and "unmatched" in d.code]
        assert len(errors) > 0

    def test_unclosed_brace(self, tmp_neuro_file):
        """Unclosed brace detected."""
        content = """
architecture test {
    d_sem: 256
"""
        f = tmp_neuro_file(content)
        diags = lint_file(f)
        errors = [d for d in diags if d.severity == Severity.ERROR and "unclosed" in d.code]
        assert len(errors) > 0


class TestDeclarationExtraction:
    """Test extraction of declarations."""

    def test_extract_population(self, tmp_neuro_file):
        """Extract population declarations."""
        content = """
population pfc {
    count: 256
}
"""
        f = tmp_neuro_file(content)
        linter = NeuroLinter(f)
        linter.lint()
        assert "pfc" in linter.populations

    def test_extract_exported_population(self, tmp_neuro_file):
        """Extract exported population."""
        content = """
export population motor {
    count: 128
}
"""
        f = tmp_neuro_file(content)
        linter = NeuroLinter(f)
        linter.lint()
        assert "motor" in linter.populations
        assert "motor" in linter.exports

    def test_extract_dynamics(self, tmp_neuro_file):
        """Extract dynamics declarations."""
        content = """
dynamics rate_code {
    equation: "y = ReLU(x)"
}
"""
        f = tmp_neuro_file(content)
        linter = NeuroLinter(f)
        linter.lint()
        assert "rate_code" in linter.dynamics_decls

    def test_extract_import(self, tmp_neuro_file):
        """Extract import declarations (ES6-style)."""
        content = """
import { rate_code, integrate_and_fire } from "@/lib/dynamics"
"""
        f = tmp_neuro_file(content)
        linter = NeuroLinter(f)
        linter.lint()
        assert "lib/dynamics" in linter.imports
        # Imported names should be added to populations
        assert "rate_code" in linter.populations
        assert "integrate_and_fire" in linter.populations


class TestReferenceValidation:
    """Test reference checking."""

    def test_undefined_population_in_synapse(self, tmp_neuro_file):
        """Warn about undefined population in synapse."""
        content = """
synapse pfc -> undefined_pop {
    weight: 0.5
}
"""
        f = tmp_neuro_file(content)
        diags = lint_file(f)
        warnings = [d for d in diags if "undefined-population" in d.code]
        assert len(warnings) >= 1

    def test_valid_synapse_reference(self, tmp_neuro_file):
        """Valid synapse references."""
        content = """
population pfc { count: 256 }
population bg { count: 128 }

synapse pfc -> bg {
    weight: 0.5
}
"""
        f = tmp_neuro_file(content)
        diags = lint_file(f)
        undefined_warnings = [d for d in diags if "undefined-population" in d.code]
        assert len(undefined_warnings) == 0


class TestEquationValidation:
    """Test equation variable validation."""

    def test_builtin_variables_in_equation(self, tmp_neuro_file):
        """Built-in variables should not trigger warnings."""
        content = """
population pfc {
    count: 256,
    equation: "y = ReLU(x)"
}
"""
        f = tmp_neuro_file(content)
        diags = lint_file(f)
        # Should not warn about x, y, ReLU
        undefined = [d for d in diags if "undefined" in d.code]
        assert len(undefined) == 0

    def test_math_functions_recognized(self, tmp_neuro_file):
        """Math functions should be recognized."""
        content = """
population pfc {
    equation: "y = sin(x) + cos(x) + exp(x)"
}
"""
        f = tmp_neuro_file(content)
        diags = lint_file(f)
        undefined = [d for d in diags if "potentially-undefined" in d.code]
        # Should not warn about sin, cos, exp
        assert len(undefined) == 0


class TestComments:
    """Test comment handling."""

    def test_ignore_comments(self, tmp_neuro_file):
        """Comments should be ignored."""
        content = """
# This is a comment
architecture test { } # inline comment
# synapse undefined -> also_undefined { }
"""
        f = tmp_neuro_file(content)
        diags = lint_file(f)
        undefined = [d for d in diags if "undefined" in d.code]
        assert len(undefined) == 0


class TestImportResolution:
    """Test import path resolution."""

    def test_unresolved_import(self, tmp_path):
        """Warn about unresolved imports."""
        f = tmp_path / "test.neuro"
        f.write_text('import { foo } from "@/nonexistent/path"\n')
        diags = lint_file(f)
        unresolved = [d for d in diags if "unresolved-import" in d.code]
        assert len(unresolved) > 0

    def test_resolved_import(self, tmp_path):
        """Valid import paths."""
        # Create a lib file
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "dynamics.neuro").write_text("dynamics test { }")

        # Create main file that imports it
        main = tmp_path / "main.neuro"
        main.write_text('import { test } from "@/lib/dynamics"\n')

        diags = lint_file(main)
        unresolved = [d for d in diags if "unresolved-import" in d.code]
        assert len(unresolved) == 0


class TestIntegration:
    """Integration tests with realistic .neuro content."""

    def test_valid_small_architecture(self, tmp_neuro_file):
        """Valid small architecture."""
        content = """
architecture test_arch {
    d_sem: 128,
    dt: 0.01
}

export population sensory {
    count: 64,
    dynamics: "rate_code"
}

export population motor {
    count: 32,
    equation: "y = ReLU(x)"
}

synapse sensory -> motor {
    weight: 0.7,
    neurotransmitter: "glutamate"
}
"""
        f = tmp_neuro_file(content)
        diags = lint_file(f)
        errors = [d for d in diags if d.severity == Severity.ERROR]
        # Should have no structural errors
        assert len(errors) == 0
        # Should extract declarations
        linter = NeuroLinter(f)
        linter.lint()
        assert "sensory" in linter.populations
        assert "motor" in linter.populations
