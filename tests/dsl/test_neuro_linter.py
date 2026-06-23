# -*- coding: utf-8 -*-
"""Tests for the .neuro DSL linter."""
import pytest
from pathlib import Path
from neuroslm.dsl.neuro_linter import NeuroLinter, Severity, lint_file
from neuroslm.dsl.compiler import NeuroMLCompiler, NeuroMLError


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


class TestCompilerIntegration:
    """Test compiler integration with linter validation."""

    def test_compiler_validates_with_linter(self, tmp_path):
        """Compiler should run linter and reject files with structural errors."""
        # Create a file with unmatched braces
        f = tmp_path / "bad.neuro"
        f.write_text("population test {\n    count: 256\n")

        with pytest.raises(NeuroMLError) as exc_info:
            NeuroMLCompiler.compile_file(str(f))

        assert "Linting failed" in str(exc_info.value)
        assert "error" in str(exc_info.value).lower()

    def test_compiler_accepts_valid_file(self, tmp_path):
        """Compiler should accept valid .neuro files."""
        f = tmp_path / "valid.neuro"
        f.write_text("""
population test {
    count: 256,
    dynamics: "rate_code"
}
""")
        # Should not raise
        result = NeuroMLCompiler.compile_file(str(f))
        assert result is not None

    def test_compiler_lists_linting_errors(self, tmp_path):
        """Compiler error message should list specific linting errors."""
        f = tmp_path / "bad.neuro"
        f.write_text("population test { count: 256 ")  # Missing closing brace

        try:
            NeuroMLCompiler.compile_file(str(f))
            assert False, "Should have raised NeuroMLError"
        except NeuroMLError as e:
            # Error message should reference the problematic file
            assert "Linting failed" in str(e)


class TestEnumStyleDetection:
    """Test detection of enum-style declarations."""

    def test_enum_style_warning(self, tmp_neuro_file):
        """Enum-style constant block should trigger warning."""
        content = """
neurotransmitter_types {
    DOPAMINE: 1,
    SEROTONIN: 2,
    ACETYLCHOLINE: 3
}
"""
        f = tmp_neuro_file(content)
        diags = lint_file(f)
        warnings = [d for d in diags if d.code == "enum-style-declaration"]
        assert len(warnings) == 1
        assert "DOPAMINE" in warnings[0].message or "enum-style" in warnings[0].message

    def test_dsl_native_no_warning(self, tmp_neuro_file):
        """DSL native constants block should not trigger warning."""
        content = """
dynamics rate_code {
    constants: {
        tau: 0.01,
        threshold: 1.0
    }
}
"""
        f = tmp_neuro_file(content)
        diags = lint_file(f)
        warnings = [d for d in diags if d.code == "enum-style-declaration"]
        assert len(warnings) == 0

    def test_enum_single_key_no_warning(self, tmp_neuro_file):
        """Single-key object should not trigger enum warning."""
        content = """
config {
    VALUE: 42
}
"""
        f = tmp_neuro_file(content)
        diags = lint_file(f)
        warnings = [d for d in diags if d.code == "enum-style-declaration"]
        assert len(warnings) == 0


class TestBlockColonSyntax:
    """Test detection and autofix of bare `key {` block syntax."""

    def test_nested_bare_block_warns(self, tmp_neuro_file):
        """Nested `key {` without colon emits missing-block-colon warning."""
        content = """
model {
    kind: gpt2
    sheaf {
        dim: 768
    }
}
"""
        f = tmp_neuro_file(content)
        diags = lint_file(f)
        codes = [d.code for d in diags]
        assert "missing-block-colon" in codes

    def test_colon_syntax_no_warning(self, tmp_neuro_file):
        """`key: {` colon syntax does not trigger warning."""
        content = """
model {
    kind: gpt2
    sheaf: {
        dim: 768
    }
}
"""
        f = tmp_neuro_file(content)
        diags = lint_file(f)
        codes = [d.code for d in diags]
        assert "missing-block-colon" not in codes

    def test_top_level_block_no_warning(self, tmp_neuro_file):
        """Top-level blocks like `model {` and `training {` don't warn."""
        content = """
model {
    kind: gpt2
}
training {
    optimizer: adamw
}
"""
        f = tmp_neuro_file(content)
        diags = lint_file(f)
        codes = [d.code for d in diags]
        assert "missing-block-colon" not in codes

    def test_multiple_bare_blocks_all_warned(self, tmp_neuro_file):
        """All bare nested blocks are flagged individually."""
        content = """
model {
    sheaf {
        dim: 768
    }
    extra {
        x: 1
    }
}
"""
        f = tmp_neuro_file(content)
        diags = lint_file(f)
        mc = [d for d in diags if d.code == "missing-block-colon"]
        assert len(mc) >= 2

    def test_autofix_adds_colon(self, tmp_neuro_file):
        """autofix_block_colon_syntax rewrites bare `key {` → `key: {`."""
        from neuroslm.dsl.neuro_linter import autofix_block_colon_syntax
        original = "model {\n    sheaf {\n        dim: 768\n    }\n}\n"
        fixed = autofix_block_colon_syntax(original)
        assert "sheaf: {" in fixed

    def test_autofix_preserves_colon_syntax(self, tmp_neuro_file):
        """autofix is idempotent: `key: {` is unchanged."""
        from neuroslm.dsl.neuro_linter import autofix_block_colon_syntax
        original = "sheaf: {\n    dim: 768\n}\n"
        fixed = autofix_block_colon_syntax(original)
        assert fixed == original

    def test_autofix_idempotent(self, tmp_neuro_file):
        """Applying autofix twice gives the same result."""
        from neuroslm.dsl.neuro_linter import autofix_block_colon_syntax
        original = "model {\n    sheaf {\n        dim: 768\n    }\n}\n"
        once = autofix_block_colon_syntax(original)
        twice = autofix_block_colon_syntax(once)
        assert once == twice

    def test_autofix_skips_strings(self, tmp_neuro_file):
        """Strings containing `word {` are not rewritten."""
        from neuroslm.dsl.neuro_linter import autofix_block_colon_syntax
        original = 'model {\n    comment: "use sheaf { } syntax"\n}\n'
        fixed = autofix_block_colon_syntax(original)
        assert '"use sheaf { } syntax"' in fixed
