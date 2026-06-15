# -*- coding: utf-8 -*-
"""Cross-architecture shared library resolution via the ``@brian/`` and
``@lib/`` prefixes.

The resolver supports four import scopes (2026-06-15 layout — lib
hoisted from ``architectures/lib/`` up to ``<repo>/lib/``):

  ``@/...``     — anchored at the architecture root (e.g. ``@/lib/x``
                  from ``architectures/master/arch.neuro`` resolves to
                  ``architectures/master/lib/x.neuro``).
  ``@brian/...`` — anchored at the **repository root** itself, so
                   ``@brian/lib/equations`` resolves to
                   ``<repo>/lib/equations.neuro``,
                   ``@brian/architectures/master/arch`` resolves to
                   that file, etc. Useful for cross-folder references
                   that aren't necessarily in ``lib/``.
  ``@lib/...``   — shorthand for ``@brian/lib/...``; resolves at
                   ``<repo>/lib/``. Preferred for shared modules /
                   equations / features / blocks. This is the clean
                   public surface; ``@brian/lib/...`` is the explicit
                   long form.
  ``./``, ``../`` — relative to the importing file.

Repo root is auto-discovered by walking up from the architecture root
until a ``pyproject.toml`` is found. The path may also be passed
explicitly to ``PathResolver`` for tests + future tooling.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuroslm.dsl.multifile import PathResolver


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def repo_with_arch(tmp_path: Path):
    """Builds a fake repo:

        <tmp>/
            pyproject.toml
            lib/
                equations.neuro
                features/
                    hyperbolic_attention.neuro
                    torus_rope.neuro
            architectures/
                master/
                    arch.neuro

    Returns ``(repo_root, arch_root)``.
    """
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='fake-brian'\n", encoding="utf-8"
    )

    # Shared lib lives at the repo root (2026-06-15 layout).
    lib = tmp_path / "lib"
    (lib / "features").mkdir(parents=True)
    (lib / "equations.neuro").write_text(
        '# shared\nexport equation eq_a { params:[x], formula:"y=x" }\n',
        encoding="utf-8",
    )
    (lib / "features" / "hyperbolic_attention.neuro").write_text(
        "# hyperbolic feature\n", encoding="utf-8"
    )
    (lib / "features" / "torus_rope.neuro").write_text(
        "# torus rope feature\n", encoding="utf-8"
    )

    # Arch
    arch = tmp_path / "architectures" / "master"
    arch.mkdir(parents=True)
    (arch / "arch.neuro").write_text("# arch root", encoding="utf-8")

    return tmp_path, arch


# ──────────────────────────────────────────────────────────────────────
# 1. Repo-root auto-discovery
# ──────────────────────────────────────────────────────────────────────


class TestRepoRootDiscovery:
    def test_resolver_finds_pyproject_by_walking_up(self, repo_with_arch):
        repo_root, arch = repo_with_arch
        r = PathResolver(arch)
        assert r.repo_root == repo_root.resolve()

    def test_explicit_repo_root_overrides_discovery(self, tmp_path):
        # No pyproject at all; passing repo_root explicitly must work
        arch = tmp_path / "arch_only"
        arch.mkdir()
        (arch / "arch.neuro").write_text("", encoding="utf-8")
        custom_repo = tmp_path / "custom_repo"
        (custom_repo / "lib").mkdir(parents=True)
        (custom_repo / "lib" / "x.neuro").write_text("", encoding="utf-8")

        r = PathResolver(arch, repo_root=custom_repo)
        assert r.repo_root == custom_repo.resolve()

    def test_no_pyproject_anywhere_leaves_repo_root_none(self, tmp_path):
        arch = tmp_path / "orphan_arch"
        arch.mkdir()
        (arch / "arch.neuro").write_text("", encoding="utf-8")
        r = PathResolver(arch)
        assert r.repo_root is None


# ──────────────────────────────────────────────────────────────────────
# 2. ``@brian/`` resolves at the **repo root** itself
# ──────────────────────────────────────────────────────────────────────


class TestBrianPrefixResolution:
    """``@brian/<path>`` is the long-form, root-anchored alias.

    ``@brian/lib/equations`` resolves to ``<repo>/lib/equations.neuro``.
    ``@brian/architectures/master/arch`` (if it existed) would resolve
    to that file. The prefix points at the entire repo, not just the
    lib — use ``@lib/...`` (below) for the common case of sharing
    library files.
    """

    def test_brian_lib_equations_resolves_to_repo_lib_equations(
        self, repo_with_arch
    ):
        repo_root, arch = repo_with_arch
        r = PathResolver(arch)
        resolved = r.resolve(
            "@brian/lib/equations", from_file=arch / "arch.neuro"
        )
        assert resolved == (repo_root / "lib" / "equations.neuro").resolve()

    def test_brian_lib_features_subdir_resolves(self, repo_with_arch):
        repo_root, arch = repo_with_arch
        r = PathResolver(arch)
        resolved = r.resolve(
            "@brian/lib/features/hyperbolic_attention",
            from_file=arch / "arch.neuro",
        )
        assert resolved == (
            repo_root / "lib" / "features" / "hyperbolic_attention.neuro"
        ).resolve()

    def test_brian_explicit_neuro_suffix_works(self, repo_with_arch):
        repo_root, arch = repo_with_arch
        r = PathResolver(arch)
        resolved = r.resolve(
            "@brian/lib/features/torus_rope.neuro",
            from_file=arch / "arch.neuro",
        )
        assert resolved == (
            repo_root / "lib" / "features" / "torus_rope.neuro"
        ).resolve()

    def test_brian_folder_falls_back_to_index_neuro(
        self, repo_with_arch
    ):
        repo_root, arch = repo_with_arch
        # Create a folder-module under <repo>/lib/
        pkg = repo_root / "lib" / "plasticity"
        pkg.mkdir()
        (pkg / "index.neuro").write_text("# pkg index", encoding="utf-8")

        r = PathResolver(arch)
        resolved = r.resolve(
            "@brian/lib/plasticity", from_file=arch / "arch.neuro"
        )
        assert resolved == (pkg / "index.neuro").resolve()

    def test_brian_unknown_path_raises_file_not_found(self, repo_with_arch):
        _, arch = repo_with_arch
        r = PathResolver(arch)
        with pytest.raises(FileNotFoundError):
            r.resolve(
                "@brian/lib/does_not_exist", from_file=arch / "arch.neuro"
            )

    def test_brian_can_reach_arbitrary_repo_files(self, repo_with_arch):
        """``@brian/`` is the repo-root alias, so it can reference
        files outside ``lib/`` (e.g. another arch's modules). This is
        what makes it the long-form for the ``@lib/`` shortcut."""
        repo_root, arch = repo_with_arch
        # Drop a file outside lib/ but inside the repo.
        target = repo_root / "architectures" / "shared.neuro"
        target.write_text("# shared\n", encoding="utf-8")
        r = PathResolver(arch)
        resolved = r.resolve(
            "@brian/architectures/shared", from_file=arch / "arch.neuro"
        )
        assert resolved == target.resolve()


# ──────────────────────────────────────────────────────────────────────
# 3. ``@lib/`` shorthand — anchored at ``<repo>/lib/``
# ──────────────────────────────────────────────────────────────────────


class TestLibPrefixResolution:
    """``@lib/<path>`` is the preferred alias for shared-library imports.

    It is exactly equivalent to ``@brian/lib/<path>`` and is what
    arch.neuro files should use day-to-day.
    """

    def test_lib_equations_resolves_to_repo_lib_equations(
        self, repo_with_arch
    ):
        repo_root, arch = repo_with_arch
        r = PathResolver(arch)
        resolved = r.resolve("@lib/equations", from_file=arch / "arch.neuro")
        assert resolved == (repo_root / "lib" / "equations.neuro").resolve()

    def test_lib_features_subdir_resolves(self, repo_with_arch):
        repo_root, arch = repo_with_arch
        r = PathResolver(arch)
        resolved = r.resolve(
            "@lib/features/hyperbolic_attention",
            from_file=arch / "arch.neuro",
        )
        assert resolved == (
            repo_root / "lib" / "features" / "hyperbolic_attention.neuro"
        ).resolve()

    def test_lib_and_brian_lib_resolve_to_same_file(self, repo_with_arch):
        """The whole point of ``@lib/`` is to be a cleaner spelling of
        ``@brian/lib/`` — both must land on the same file."""
        _, arch = repo_with_arch
        r = PathResolver(arch)
        via_lib = r.resolve(
            "@lib/equations", from_file=arch / "arch.neuro"
        )
        via_brian = r.resolve(
            "@brian/lib/equations", from_file=arch / "arch.neuro"
        )
        assert via_lib == via_brian


# ──────────────────────────────────────────────────────────────────────
# 4. Security: prefixes may not escape their scope's root
# ──────────────────────────────────────────────────────────────────────


class TestBrianPrefixSecurity:
    def test_brian_dotdot_escape_is_rejected(self, repo_with_arch):
        repo_root, arch = repo_with_arch
        # Drop a file ABOVE the repo root.
        (repo_root.parent / "secret.neuro").write_text(
            "nope", encoding="utf-8"
        )
        r = PathResolver(arch)
        with pytest.raises(ValueError, match="escapes"):
            r.resolve("@brian/../secret", from_file=arch / "arch.neuro")

    def test_lib_dotdot_escape_is_rejected(self, repo_with_arch):
        """``@lib/../...`` must not escape ``<repo>/lib/`` — even if
        the target file exists somewhere else in the repo."""
        repo_root, arch = repo_with_arch
        (repo_root / "outside_lib.neuro").write_text(
            "nope", encoding="utf-8"
        )
        r = PathResolver(arch)
        with pytest.raises(ValueError, match="escapes"):
            r.resolve("@lib/../outside_lib", from_file=arch / "arch.neuro")

    def test_brian_prefix_without_repo_root_is_explicit_error(
        self, tmp_path
    ):
        # No pyproject + no explicit repo_root → using @brian must
        # produce a clear error, not a confusing FileNotFoundError on
        # some half-constructed path.
        arch = tmp_path / "orphan_arch"
        arch.mkdir()
        (arch / "arch.neuro").write_text("", encoding="utf-8")
        r = PathResolver(arch)
        with pytest.raises(ValueError, match="repo root"):
            r.resolve("@brian/lib/equations", from_file=arch / "arch.neuro")

    def test_lib_prefix_without_repo_root_is_explicit_error(
        self, tmp_path
    ):
        arch = tmp_path / "orphan_arch"
        arch.mkdir()
        (arch / "arch.neuro").write_text("", encoding="utf-8")
        r = PathResolver(arch)
        with pytest.raises(ValueError, match="repo root"):
            r.resolve("@lib/equations", from_file=arch / "arch.neuro")
