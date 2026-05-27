# -*- coding: utf-8 -*-
"""Stage 1 — Multi-file DSL path resolution and folder loading.

The compiler can now ingest an entire architecture *folder*. This stage
delivers:

  1. PathResolver — turns import specifiers into absolute file paths
        @/lib/foo        →  <arch_root>/lib/foo.neuro
        ./bar            →  <current_file_dir>/bar.neuro
        ../baz           →  <parent_of_current>/baz.neuro
     With folder-as-module fallback (mjs-style):
        @/modules/pfc    →  @/modules/pfc.neuro            (file)
                         →  @/modules/pfc/index.neuro      (if folder)

  2. FolderLoader.discover(arch_root) — walks a folder, returns a dict
     mapping absolute path → raw .neuro source for every file found.

Later stages add parsing of `module`/`import`/`export`, symbol tables,
and reference resolution. Stage 1 is foundation only; nothing here
parses DSL syntax beyond reading file bytes.
"""
import pytest
from pathlib import Path

from neuroslm.dsl.multifile import PathResolver, FolderLoader


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def arch_root(tmp_path):
    """Minimal architecture folder structure used by most tests."""
    (tmp_path / "arch.neuro").write_text("# top-level config", encoding="utf-8")

    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "dynamics.neuro").write_text("# shared dynamics", encoding="utf-8")
    (tmp_path / "lib" / "plasticity").mkdir()
    (tmp_path / "lib" / "plasticity" / "index.neuro").write_text(
        "# re-exports", encoding="utf-8"
    )
    (tmp_path / "lib" / "plasticity" / "hebbian.neuro").write_text(
        "# hebbian rule", encoding="utf-8"
    )

    (tmp_path / "modules").mkdir()
    (tmp_path / "modules" / "thalamus.neuro").write_text(
        "# single-file module", encoding="utf-8"
    )
    (tmp_path / "modules" / "pfc").mkdir()
    (tmp_path / "modules" / "pfc" / "index.neuro").write_text(
        "# pfc package", encoding="utf-8"
    )
    (tmp_path / "modules" / "pfc" / "layers.neuro").write_text(
        "# pfc internal", encoding="utf-8"
    )
    return tmp_path


# ── PathResolver — absolute (`@/...`) ───────────────────────────────────

class TestAbsolutePaths:
    def test_absolute_file(self, arch_root):
        r = PathResolver(arch_root)
        resolved = r.resolve("@/lib/dynamics", from_file=arch_root / "arch.neuro")
        assert resolved == arch_root / "lib" / "dynamics.neuro"

    def test_absolute_folder_module(self, arch_root):
        # `@/lib/plasticity` is a folder containing index.neuro → that wins
        r = PathResolver(arch_root)
        resolved = r.resolve("@/lib/plasticity", from_file=arch_root / "arch.neuro")
        assert resolved == arch_root / "lib" / "plasticity" / "index.neuro"

    def test_absolute_with_neuro_suffix_works(self, arch_root):
        r = PathResolver(arch_root)
        resolved = r.resolve("@/lib/dynamics.neuro",
                             from_file=arch_root / "arch.neuro")
        assert resolved == arch_root / "lib" / "dynamics.neuro"

    def test_absolute_nested_folder(self, arch_root):
        r = PathResolver(arch_root)
        resolved = r.resolve("@/modules/pfc",
                             from_file=arch_root / "arch.neuro")
        assert resolved == arch_root / "modules" / "pfc" / "index.neuro"


# ── PathResolver — relative (`./` and `../`) ────────────────────────────

class TestRelativePaths:
    def test_dot_slash_same_folder(self, arch_root):
        r = PathResolver(arch_root)
        # From pfc/index.neuro, `./layers` resolves to pfc/layers.neuro
        from_file = arch_root / "modules" / "pfc" / "index.neuro"
        resolved = r.resolve("./layers", from_file=from_file)
        assert resolved == arch_root / "modules" / "pfc" / "layers.neuro"

    def test_dotdot_parent(self, arch_root):
        r = PathResolver(arch_root)
        # From pfc/layers.neuro, `../thalamus` reaches modules/thalamus.neuro
        from_file = arch_root / "modules" / "pfc" / "layers.neuro"
        resolved = r.resolve("../thalamus", from_file=from_file)
        assert resolved == arch_root / "modules" / "thalamus.neuro"

    def test_relative_folder_module(self, arch_root):
        # From arch.neuro, `./modules/pfc` is a folder → index.neuro
        r = PathResolver(arch_root)
        resolved = r.resolve("./modules/pfc",
                             from_file=arch_root / "arch.neuro")
        assert resolved == arch_root / "modules" / "pfc" / "index.neuro"


# ── PathResolver — error reporting ──────────────────────────────────────

class TestPathErrors:
    def test_missing_path_raises(self, arch_root):
        r = PathResolver(arch_root)
        with pytest.raises(FileNotFoundError, match="does_not_exist"):
            r.resolve("@/does_not_exist", from_file=arch_root / "arch.neuro")

    def test_relative_without_from_file_raises(self, arch_root):
        r = PathResolver(arch_root)
        with pytest.raises(ValueError, match="from_file"):
            r.resolve("./something", from_file=None)

    def test_escape_above_root_rejected(self, arch_root):
        # `../../etc` from the arch root would escape — we don't allow that
        r = PathResolver(arch_root)
        from_file = arch_root / "arch.neuro"
        with pytest.raises(ValueError, match="escape"):
            r.resolve("../../somewhere", from_file=from_file)

    def test_unsupported_specifier(self, arch_root):
        # Must start with @/, ./, or ../
        r = PathResolver(arch_root)
        with pytest.raises(ValueError, match="specifier"):
            r.resolve("bare/path", from_file=arch_root / "arch.neuro")


# ── FolderLoader — discover all .neuro files in a folder ────────────────

class TestFolderLoader:
    def test_discovers_all_files(self, arch_root):
        loader = FolderLoader(arch_root)
        sources = loader.discover()
        # Every .neuro file from the fixture
        expected = {
            arch_root / "arch.neuro",
            arch_root / "lib" / "dynamics.neuro",
            arch_root / "lib" / "plasticity" / "index.neuro",
            arch_root / "lib" / "plasticity" / "hebbian.neuro",
            arch_root / "modules" / "thalamus.neuro",
            arch_root / "modules" / "pfc" / "index.neuro",
            arch_root / "modules" / "pfc" / "layers.neuro",
        }
        assert set(sources.keys()) == expected

    def test_returns_file_contents(self, arch_root):
        loader = FolderLoader(arch_root)
        sources = loader.discover()
        assert sources[arch_root / "lib" / "dynamics.neuro"] == "# shared dynamics"

    def test_empty_folder_raises(self, tmp_path):
        loader = FolderLoader(tmp_path)
        with pytest.raises(ValueError, match="no .neuro files"):
            loader.discover()

    def test_arch_neuro_present_check(self, arch_root):
        loader = FolderLoader(arch_root)
        assert loader.has_arch_root()

    def test_arch_neuro_missing(self, tmp_path):
        (tmp_path / "stray.neuro").write_text("# orphan", encoding="utf-8")
        loader = FolderLoader(tmp_path)
        assert not loader.has_arch_root()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
