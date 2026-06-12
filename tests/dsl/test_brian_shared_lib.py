# -*- coding: utf-8 -*-
"""Cross-architecture shared library resolution via the ``@brian/`` prefix.

The existing resolver supports two import scopes:

  ``@/...``    — anchored at the architecture root (e.g. ``@/lib/equations``
                 from ``architectures/master/arch.neuro`` resolves to
                 ``architectures/master/lib/equations.neuro``).
  ``./``, ``../`` — relative to the importing file.

This stage adds a third scope: ``@brian/...`` is anchored at the
**repository's shared** ``architectures/lib/`` directory so that
equations and features defined once can be reused across every
architecture (master, rcc_bowtie, future evol variants, …) without
copy-pasting.

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
            architectures/
                lib/
                    equations.neuro
                    features/
                        hyperbolic_attention.neuro
                        torus_rope.neuro
                master/
                    arch.neuro

    Returns ``(repo_root, arch_root)``.
    """
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='fake-brian'\n", encoding="utf-8"
    )

    # Shared lib lives under architectures/lib/
    lib = tmp_path / "architectures" / "lib"
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
        custom_lib_root = tmp_path / "custom_repo"
        (custom_lib_root / "architectures" / "lib").mkdir(parents=True)
        (custom_lib_root / "architectures" / "lib" / "x.neuro").write_text(
            "", encoding="utf-8"
        )

        r = PathResolver(arch, repo_root=custom_lib_root)
        assert r.repo_root == custom_lib_root.resolve()

    def test_no_pyproject_anywhere_leaves_repo_root_none(self, tmp_path):
        arch = tmp_path / "orphan_arch"
        arch.mkdir()
        (arch / "arch.neuro").write_text("", encoding="utf-8")
        r = PathResolver(arch)
        assert r.repo_root is None


# ──────────────────────────────────────────────────────────────────────
# 2. ``@brian/`` resolves under ``<repo>/lib/``
# ──────────────────────────────────────────────────────────────────────


class TestBrianPrefixResolution:
    def test_brian_equations_resolves_to_repo_lib_equations(self, repo_with_arch):
        repo_root, arch = repo_with_arch
        r = PathResolver(arch)
        resolved = r.resolve(
            "@brian/equations", from_file=arch / "arch.neuro"
        )
        assert resolved == (
            repo_root / "architectures" / "lib" / "equations.neuro"
        ).resolve()

    def test_brian_features_subdir_resolves(self, repo_with_arch):
        repo_root, arch = repo_with_arch
        r = PathResolver(arch)
        resolved = r.resolve(
            "@brian/features/hyperbolic_attention",
            from_file=arch / "arch.neuro",
        )
        assert resolved == (
            repo_root / "architectures" / "lib" / "features"
            / "hyperbolic_attention.neuro"
        ).resolve()

    def test_brian_explicit_neuro_suffix_works(self, repo_with_arch):
        repo_root, arch = repo_with_arch
        r = PathResolver(arch)
        resolved = r.resolve(
            "@brian/features/torus_rope.neuro",
            from_file=arch / "arch.neuro",
        )
        assert resolved == (
            repo_root / "architectures" / "lib" / "features"
            / "torus_rope.neuro"
        ).resolve()

    def test_brian_folder_falls_back_to_index_neuro(
        self, repo_with_arch
    ):
        repo_root, arch = repo_with_arch
        # Create a folder-module under @brian/
        pkg = repo_root / "architectures" / "lib" / "plasticity"
        pkg.mkdir()
        (pkg / "index.neuro").write_text("# pkg index", encoding="utf-8")

        r = PathResolver(arch)
        resolved = r.resolve(
            "@brian/plasticity", from_file=arch / "arch.neuro"
        )
        assert resolved == (pkg / "index.neuro").resolve()

    def test_brian_unknown_path_raises_file_not_found(self, repo_with_arch):
        _, arch = repo_with_arch
        r = PathResolver(arch)
        with pytest.raises(FileNotFoundError):
            r.resolve("@brian/does_not_exist", from_file=arch / "arch.neuro")


# ──────────────────────────────────────────────────────────────────────
# 3. Security: ``@brian/`` may not escape ``<repo>/architectures/lib/``
# ──────────────────────────────────────────────────────────────────────


class TestBrianPrefixSecurity:
    def test_brian_dotdot_escape_is_rejected(self, repo_with_arch):
        repo_root, arch = repo_with_arch
        # Even if the file exists, we refuse to resolve outside the
        # shared lib root (architectures/lib/).
        (repo_root / "architectures" / "secret.neuro").write_text(
            "nope", encoding="utf-8"
        )
        r = PathResolver(arch)
        with pytest.raises(ValueError, match="escapes"):
            r.resolve("@brian/../secret", from_file=arch / "arch.neuro")

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
            r.resolve("@brian/equations", from_file=arch / "arch.neuro")
