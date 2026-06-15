# -*- coding: utf-8 -*-
"""Regression tests for the ``module_bundler.ModuleBundler.resolve_import``
spec alignment with the runtime ``PathResolver``.

The bundler powers ``RibosomeCompiler.compile_file`` (DNA snapshotting),
which means any specifier shape it can't resolve gets silently dropped
on compile and re-appears as a phantom import on unfold. The runtime
resolver in ``neuroslm/dsl/multifile.py`` supports four prefixes:

    * ``@brian/<path>``  →  ``<repo_root>/<path>``
    * ``@lib/<path>``    →  ``<repo_root>/lib/<path>``
    * ``@/<path>``       →  ``<arch_root>/<path>``
    * ``./<path>`` / ``../<path>``  →  relative to importing file

…but the bundler historically only handled ``@brian/`` (and got even
that wrong — pointing at ``<repo_root>/architectures/lib/`` instead of
``<repo_root>/``). Result:

    * ``import "@lib/equations"`` → resolved → None → silently dropped
    * ``ModuleBundler(architectures/master).bundle(...)`` → 0 modules
    * downstream bundler-driven tests (DNA roundtrip, NFG IR lift)
      all fail in different ways

This file pins the resolver contract by re-exercising every prefix
shape that production ``arch.neuro`` files use, with the actual
on-disk layout (``lib/`` at repo root, ``architectures/master/modules/``
inside the arch).
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MASTER_ARCH = REPO_ROOT / "architectures" / "master"


# ──────────────────────────────────────────────────────────────────────
# Contract 1 — every prefix used by master/arch.neuro resolves to a file
# ──────────────────────────────────────────────────────────────────────


class TestResolveImportPrefixAlignment:
    """``ModuleBundler.resolve_import`` must return a real path for
    every specifier shape that the canonical ``master/arch.neuro``
    uses, with no fallthrough to ``None``."""

    def test_at_lib_equations_resolves(self):
        from neuroslm.compiler.module_bundler import ModuleBundler
        b = ModuleBundler(MASTER_ARCH)
        resolved = b.resolve_import(
            "@lib/equations", MASTER_ARCH / "arch.neuro"
        )
        assert resolved is not None, (
            "@lib/equations failed to resolve — bundler is missing "
            "the @lib/ prefix handler (runtime resolver supports it; "
            "bundler must match or DNA roundtrip drops the import)"
        )
        assert resolved == (REPO_ROOT / "lib" / "equations.neuro").resolve()

    def test_at_lib_modules_subpath_resolves(self):
        from neuroslm.compiler.module_bundler import ModuleBundler
        b = ModuleBundler(MASTER_ARCH)
        resolved = b.resolve_import(
            "@lib/modules/sensory", MASTER_ARCH / "arch.neuro"
        )
        assert resolved is not None
        assert resolved == (
            REPO_ROOT / "lib" / "modules" / "sensory.neuro"
        ).resolve()

    def test_at_lib_features_subpath_resolves(self):
        from neuroslm.compiler.module_bundler import ModuleBundler
        b = ModuleBundler(MASTER_ARCH)
        resolved = b.resolve_import(
            "@lib/features/hyperbolic_attention",
            MASTER_ARCH / "arch.neuro",
        )
        assert resolved is not None
        assert resolved == (
            REPO_ROOT / "lib" / "features" / "hyperbolic_attention.neuro"
        ).resolve()

    def test_at_brian_anchors_at_repo_root_not_architectures_lib(self):
        """``@brian/<path>`` must resolve relative to the REPO root,
        matching ``neuroslm.dsl.multifile.PathResolver``. The old bundler
        anchored it at ``<repo>/architectures/lib/`` which is the wrong
        directory and breaks ``@brian/lib/equations``."""
        from neuroslm.compiler.module_bundler import ModuleBundler
        b = ModuleBundler(MASTER_ARCH)
        resolved = b.resolve_import(
            "@brian/lib/equations", MASTER_ARCH / "arch.neuro"
        )
        assert resolved is not None, (
            "@brian/lib/equations failed to resolve — bundler is "
            "anchoring @brian/ at the wrong directory"
        )
        assert resolved == (REPO_ROOT / "lib" / "equations.neuro").resolve()

    def test_at_slash_arch_local_still_works(self):
        """Back-compat: ``@/<path>`` anchored at arch_root must still
        work (this is what ``architectures/evol/arch.neuro`` uses)."""
        from neuroslm.compiler.module_bundler import ModuleBundler
        # Use a tmp arch with its own @/lib/ tree (don't depend on the
        # production layout — production uses @lib/ now).
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            (td_path / "lib").mkdir()
            target = td_path / "lib" / "x.neuro"
            target.write_text("population x { count: 1 }")
            arch = td_path / "arch.neuro"
            arch.write_text('architecture test { d_sem: 1 }\nimport "@/lib/x"')
            b = ModuleBundler(td_path)
            resolved = b.resolve_import("@/lib/x", arch)
            assert resolved == target.resolve()


# ──────────────────────────────────────────────────────────────────────
# Contract 2 — bundling the production master arch picks up all modules
# ──────────────────────────────────────────────────────────────────────


class TestBundleProductionArch:
    """End-to-end: ``ModuleBundler(architectures/master).bundle(arch.neuro)``
    must collect every imported module file. With ``@lib/`` broken, the
    result was 0 modules; the fix must yield at least the modules and
    lib files the arch actually imports."""

    def test_bundles_at_least_lib_modules_and_features(self):
        from neuroslm.compiler.module_bundler import ModuleBundler
        b = ModuleBundler(MASTER_ARCH)
        bundled = b.bundle(MASTER_ARCH / "arch.neuro")
        # Master arch imports 13 modules under @lib/modules, 5 under
        # @lib/features, plus @lib/equations + @lib/regularizers +
        # @lib/cdga + @lib/emergent + @lib/constraints. 24 total.
        # Lower-bound to >5 so the test stays robust to legitimate
        # module additions / removals.
        assert len(bundled.modules) > 5, (
            f"expected > 5 modules to be bundled, got "
            f"{len(bundled.modules)}; bundler's @lib/ resolver still "
            f"broken or arch.neuro lost its imports"
        )

    def test_bundled_modules_include_lib_subtree(self):
        from neuroslm.compiler.module_bundler import ModuleBundler
        b = ModuleBundler(MASTER_ARCH)
        bundled = b.bundle(MASTER_ARCH / "arch.neuro")
        lib_specs = [s for s in bundled.modules if "lib" in s]
        assert lib_specs, (
            f"no @lib/* modules bundled; got keys: "
            f"{sorted(bundled.modules.keys())[:10]}"
        )

    def test_bundled_modules_include_modules_subtree(self):
        from neuroslm.compiler.module_bundler import ModuleBundler
        b = ModuleBundler(MASTER_ARCH)
        bundled = b.bundle(MASTER_ARCH / "arch.neuro")
        mod_specs = [s for s in bundled.modules if "modules" in s]
        assert mod_specs, (
            f"no @lib/modules/* bundled; got keys: "
            f"{sorted(bundled.modules.keys())[:10]}"
        )
