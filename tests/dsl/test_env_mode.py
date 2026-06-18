# -*- coding: utf-8 -*-
"""TDD: BRIAN_ENV environment-gated DSL compilation.

The system lets declarations in .neuro files be tagged with @env=MODE or
placed inside [env=MODE] section blocks. The compiler reads BRIAN_ENV and
omits declarations that don't match the active mode.

Modes
-----
training   — full model with frozen cortex experts + KL-distillation (default)
dev        — same as training but with extra debug wiring
prod       — trunk only; no expert forward passes, no KL loss

Design contracts pinned here:

  1. @env=MODE decorator — attached to the next declaration
  2. [env=MODE] section — all declarations inside inherit that mode
  3. No tag → included in every mode ("*")
  4. BRIAN_ENV default → "training" when env var is absent
  5. Compiler gate — declarations filtered before hypergraph construction
  6. Expert declarations get @env=training; excluded in prod mode
"""
from __future__ import annotations

import os
import textwrap
from pathlib import Path
import pytest


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse(source: str, path: str = "test.neuro"):
    from neuroslm.dsl.multifile import parse_module
    return parse_module(source, Path(path))


def _env_tags(source: str) -> dict[str, str | None]:
    """Return the env_tags dict from a parsed module."""
    ast = _parse(source)
    return ast.env_tags


# ── 1. @env=MODE decorator on individual declarations ────────────────────────

class TestEnvDecorator:

    def test_decorated_declaration_stored_with_env_tag(self):
        src = textwrap.dedent("""\
            @env=training
            population experts {
                count: 3
            }
        """)
        tags = _env_tags(src)
        assert tags.get("experts") == "training"

    def test_undecorated_declaration_has_no_tag(self):
        src = textwrap.dedent("""\
            population trunk {
                count: 64
            }
        """)
        tags = _env_tags(src)
        assert tags.get("trunk") is None

    def test_multiple_declarations_different_tags(self):
        src = textwrap.dedent("""\
            population trunk {
                count: 64
            }
            @env=training
            population experts {
                count: 3
            }
            @env=dev
            population debug_monitor {
                count: 8
            }
        """)
        tags = _env_tags(src)
        assert tags.get("trunk") is None
        assert tags.get("experts") == "training"
        assert tags.get("debug_monitor") == "dev"

    def test_prod_tag(self):
        src = textwrap.dedent("""\
            @env=prod
            population lightweight {
                count: 32
            }
        """)
        assert _env_tags(src).get("lightweight") == "prod"

    def test_decorator_before_export_declaration(self):
        src = textwrap.dedent("""\
            @env=training
            export population experts {
                count: 3
            }
        """)
        tags = _env_tags(src)
        assert tags.get("experts") == "training"


# ── 2. [env=MODE] section blocks ─────────────────────────────────────────────

class TestEnvSection:

    def test_section_block_tags_all_enclosed_declarations(self):
        src = textwrap.dedent("""\
            population trunk { count: 64 }

            [env=training]
            population experts { count: 3 }
            population distill_head { count: 16 }
        """)
        tags = _env_tags(src)
        assert tags.get("trunk") is None          # before section → untagged
        assert tags.get("experts") == "training"
        assert tags.get("distill_head") == "training"

    def test_section_closed_by_next_section(self):
        src = textwrap.dedent("""\
            [env=training]
            population experts { count: 3 }

            [env=dev]
            population debug { count: 8 }
        """)
        tags = _env_tags(src)
        assert tags.get("experts") == "training"
        assert tags.get("debug") == "dev"

    def test_decorator_overrides_section_tag(self):
        """@env=prod inside [env=training] wins."""
        src = textwrap.dedent("""\
            [env=training]
            population experts { count: 3 }
            @env=prod
            population prod_only { count: 4 }
        """)
        tags = _env_tags(src)
        assert tags.get("experts") == "training"
        assert tags.get("prod_only") == "prod"


# ── 3. No tag → all modes ("*") ──────────────────────────────────────────────

class TestNoTag:

    def test_untagged_declaration_included_in_all_modes(self, tmp_path, monkeypatch):
        from neuroslm.compiler.hypergraph_ir import filter_module_for_env
        src = textwrap.dedent("""\
            population trunk { count: 64 }
        """)
        ast = _parse(src)
        for mode in ("dev", "training", "prod"):
            monkeypatch.setenv("BRIAN_ENV", mode)
            filtered = filter_module_for_env(ast, mode)
            all_names = set(filtered.exports) | set(filtered.private)
            assert "trunk" in all_names, f"untagged 'trunk' must survive mode={mode}"


# ── 4. BRIAN_ENV default = "training" ────────────────────────────────────────

class TestBrianEnvDefault:

    def test_default_is_training_when_env_not_set(self, monkeypatch):
        from neuroslm.compiler.hypergraph_ir import current_env_mode
        monkeypatch.delenv("BRIAN_ENV", raising=False)
        assert current_env_mode() == "training"

    def test_reads_brian_env_var(self, monkeypatch):
        from neuroslm.compiler.hypergraph_ir import current_env_mode
        monkeypatch.setenv("BRIAN_ENV", "prod")
        assert current_env_mode() == "prod"

    def test_invalid_env_raises(self, monkeypatch):
        from neuroslm.compiler.hypergraph_ir import current_env_mode
        monkeypatch.setenv("BRIAN_ENV", "banana")
        with pytest.raises(ValueError, match="BRIAN_ENV"):
            current_env_mode()


# ── 5. Compiler gate — filter_module_for_env ─────────────────────────────────

class TestCompilerFilter:

    def _make_ast(self, source: str):
        return _parse(source)

    def test_training_mode_excludes_prod_only_declaration(self, monkeypatch):
        from neuroslm.compiler.hypergraph_ir import filter_module_for_env
        src = textwrap.dedent("""\
            population trunk { count: 64 }
            @env=prod
            population prod_only { count: 4 }
        """)
        ast = self._make_ast(src)
        filtered = filter_module_for_env(ast, "training")
        all_names = set(filtered.exports) | set(filtered.private)
        assert "trunk" in all_names
        assert "prod_only" not in all_names

    def test_prod_mode_excludes_training_only_declaration(self, monkeypatch):
        from neuroslm.compiler.hypergraph_ir import filter_module_for_env
        src = textwrap.dedent("""\
            population trunk { count: 64 }
            @env=training
            population experts { count: 3 }
        """)
        ast = self._make_ast(src)
        filtered = filter_module_for_env(ast, "prod")
        all_names = set(filtered.exports) | set(filtered.private)
        assert "trunk" in all_names
        assert "experts" not in all_names

    def test_training_mode_includes_training_declaration(self):
        from neuroslm.compiler.hypergraph_ir import filter_module_for_env
        src = textwrap.dedent("""\
            @env=training
            population experts { count: 3 }
        """)
        ast = self._make_ast(src)
        filtered = filter_module_for_env(ast, "training")
        all_names = set(filtered.exports) | set(filtered.private)
        assert "experts" in all_names

    def test_dev_includes_training_declarations(self):
        """dev mode is a superset of training — training-tagged decls also included."""
        from neuroslm.compiler.hypergraph_ir import filter_module_for_env
        src = textwrap.dedent("""\
            @env=training
            population experts { count: 3 }
        """)
        ast = self._make_ast(src)
        filtered = filter_module_for_env(ast, "dev")
        all_names = set(filtered.exports) | set(filtered.private)
        assert "experts" in all_names

    def test_prod_excludes_dev_declarations(self):
        from neuroslm.compiler.hypergraph_ir import filter_module_for_env
        src = textwrap.dedent("""\
            @env=dev
            population debug_monitor { count: 8 }
        """)
        ast = self._make_ast(src)
        filtered = filter_module_for_env(ast, "prod")
        all_names = set(filtered.exports) | set(filtered.private)
        assert "debug_monitor" not in all_names

    def test_exported_declarations_preserve_export_status_after_filter(self):
        from neuroslm.compiler.hypergraph_ir import filter_module_for_env
        src = textwrap.dedent("""\
            @env=training
            export population experts { count: 3 }
        """)
        ast = self._make_ast(src)
        filtered = filter_module_for_env(ast, "training")
        assert "experts" in filtered.exports


# ── 6. Expert declarations are @env=training by default ──────────────────────

class TestExpertDefaultEnvTag:

    def test_expert_block_without_decorator_is_training_by_default(self):
        """expert declarations are implicitly @env=training unless overridden."""
        src = textwrap.dedent("""\
            expert gpt2 {
                domain: "general"
                freeze: true
            }
        """)
        tags = _env_tags(src)
        assert tags.get("gpt2") == "training"

    def test_expert_block_can_be_overridden_to_prod(self):
        src = textwrap.dedent("""\
            @env=prod
            expert gpt2 {
                domain: "general"
                freeze: true
            }
        """)
        tags = _env_tags(src)
        assert tags.get("gpt2") == "prod"
