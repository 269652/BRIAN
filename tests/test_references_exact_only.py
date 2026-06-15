"""Tests for the EXACT-ONLY reference contract.

Background
==========

The reference-aware ``brian clean`` family (logs / checkpoints / docs /
lfs) used to consider a file "referenced" if any of THREE matchers
fired:

  1. exact basename in an `exact` set
  2. stem (basename minus extension) in a `stems` set
  3. permissive glob match against tokens in a `globs` set, with
     three sub-rules (literal fnmatch, leading-`*` retry, and
     ≥8-char distinctive-segment fallback)

The H22 LFS-prune forensic (see ``docs/FINDINGS.md``) showed every
single one of 25 surviving LFS pointers was pinned by R1 only — but
those R1 hits were almost all coming from glob examples in docstrings
and test fixture filenames that *looked* like real references but
weren't.

Decision: collapse the three matchers to ONE — exact basename
equality. No globs, no stems, no distinctive-segment heuristics. If a
file isn't named verbatim in some `.md` / `.py` / `.json` / `.toml` /
etc. (excluding self-excluded `logs/` and `lfs_checkpoints/`), it
isn't protected.

These tests pin that contract.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from neuroslm.references import (
    ReferenceIndex,
    build_reference_index,
)


# ──────────────────────────────────────────────────────────────────────
# Direct API: ReferenceIndex.references(basename)
# ──────────────────────────────────────────────────────────────────────


class TestReferencesIsExactBasenameOnly:
    """``ReferenceIndex.references(basename)`` must return True iff the
    given basename appears verbatim in ``self.exact``. Stems, globs,
    and distinctive-segment fallbacks all yield False now."""

    def test_exact_basename_returns_true(self) -> None:
        idx = ReferenceIndex()
        idx.exact.add("dsl_arch_step10000.pt")
        assert idx.references("dsl_arch_step10000.pt") is True

    def test_basename_not_in_exact_returns_false(self) -> None:
        idx = ReferenceIndex()
        idx.exact.add("dsl_arch_step10000.pt")
        # Off by one digit — must NOT match.
        assert idx.references("dsl_arch_step10001.pt") is False

    def test_stem_match_does_not_protect(self) -> None:
        """``foo.pt`` referenced verbatim must NOT protect ``foo.json``."""
        idx = ReferenceIndex()
        idx.exact.add("foo.pt")
        # Old behaviour: stem `foo` would also live in ``idx.stems``
        # and `references("foo.json")` would return True.
        # New behaviour: extensions are part of the basename; no stem
        # propagation.
        assert idx.references("foo.json") is False

    def test_stems_field_is_never_consulted(self) -> None:
        """Even if someone manually stuffs a stem into ``idx.stems``,
        ``references()`` must ignore it."""
        idx = ReferenceIndex()
        idx.stems.add("foo")  # legacy field, no longer consulted
        assert idx.references("foo.pt") is False
        assert idx.references("foo.json") is False

    def test_glob_token_does_not_protect_matching_basename(self) -> None:
        """A glob token like ``dsl_arch_step*.pt`` must NOT protect any
        actual ``dsl_arch_step{N}.pt`` checkpoint."""
        idx = ReferenceIndex()
        # Even if a caller (incorrectly) stuffs a glob into either
        # ``exact`` or ``globs``, no matching by pattern is performed.
        idx.globs.add("dsl_arch_step*.pt")
        assert idx.references("dsl_arch_step1000.pt") is False
        assert idx.references("dsl_arch_step10000.pt") is False

    def test_distinctive_segment_does_not_protect(self) -> None:
        """A ≥8-char literal segment of a glob must NOT protect a
        basename that contains that segment as a substring."""
        idx = ReferenceIndex()
        idx.globs.add("*_keep_me_glob_step2kof2k.log")
        assert idx.references(
            "af758c381388_keep_me_glob_step2kof2k.log",
        ) is False

    def test_empty_index_protects_nothing(self) -> None:
        idx = ReferenceIndex()
        assert idx.references("any.pt") is False
        assert idx.references("README.md") is False


# ──────────────────────────────────────────────────────────────────────
# Scanner: build_reference_index populates only ``exact``
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def repo_with_mixed_tokens(tmp_path: Path) -> Path:
    """A tiny repo whose FINDINGS.md cites:
      * an exact basename            (`real_run_step5000.pt`)
      * a path-prefixed basename     (`lfs_checkpoints/path_prefixed.pt`)
      * a glob token                 (`dsl_arch_step*.pt`)
      * a distinctive-segment glob   (`*_keep_me_glob_step2kof2k.log`)
    """
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "FINDINGS.md").write_text(
        "# Findings\n\n"
        "## H42 — example\n\n"
        "Exact ref:           `real_run_step5000.pt`\n"
        "Path-prefixed:       `lfs_checkpoints/path_prefixed.pt`\n"
        "Glob ref (obsolete): `dsl_arch_step*.pt`\n"
        "Distinctive segment: `*_keep_me_glob_step2kof2k.log`\n",
        encoding="utf-8",
    )
    return tmp_path


class TestBuildReferenceIndexExactOnly:

    def test_exact_basename_is_indexed(
        self, repo_with_mixed_tokens: Path,
    ) -> None:
        idx = build_reference_index(repo_with_mixed_tokens)
        assert "real_run_step5000.pt" in idx.exact
        assert idx.references("real_run_step5000.pt") is True

    def test_path_prefixed_basename_is_indexed(
        self, repo_with_mixed_tokens: Path,
    ) -> None:
        """``lfs_checkpoints/path_prefixed.pt`` must register the
        basename ``path_prefixed.pt`` in ``exact`` so a citation with a
        path prefix still protects the file."""
        idx = build_reference_index(repo_with_mixed_tokens)
        assert "path_prefixed.pt" in idx.exact
        assert idx.references("path_prefixed.pt") is True

    def test_glob_tokens_are_dropped(
        self, repo_with_mixed_tokens: Path,
    ) -> None:
        """Tokens containing ``*`` / ``?`` must NOT be added to any
        index field — they are discarded outright."""
        idx = build_reference_index(repo_with_mixed_tokens)
        # No glob token should leak into ``exact``.
        for tok in idx.exact:
            assert "*" not in tok, (
                f"glob token {tok!r} leaked into exact set; expected "
                "build_reference_index to skip * / ? tokens entirely"
            )
            assert "?" not in tok, tok

    def test_glob_tokens_do_not_protect_matching_files(
        self, repo_with_mixed_tokens: Path,
    ) -> None:
        """End-to-end: the FINDINGS.md glob ``dsl_arch_step*.pt`` must
        NOT cause ``dsl_arch_step1000.pt`` to look referenced."""
        idx = build_reference_index(repo_with_mixed_tokens)
        assert idx.references("dsl_arch_step1000.pt") is False
        assert idx.references("dsl_arch_step10000.pt") is False

    def test_distinctive_segment_does_not_protect_in_real_scan(
        self, repo_with_mixed_tokens: Path,
    ) -> None:
        """End-to-end: the FINDINGS.md glob
        ``*_keep_me_glob_step2kof2k.log`` must NOT protect a real log
        file ``af758c381388_keep_me_glob_step2kof2k.log``."""
        idx = build_reference_index(repo_with_mixed_tokens)
        assert idx.references(
            "af758c381388_keep_me_glob_step2kof2k.log",
        ) is False

    def test_globs_and_stems_fields_remain_empty(
        self, repo_with_mixed_tokens: Path,
    ) -> None:
        """The dataclass fields ``globs`` and ``stems`` still exist for
        back-compat, but ``build_reference_index`` no longer populates
        them. This pins the invariant so a future refactor can't
        silently re-enable the old behaviour by re-filling them."""
        idx = build_reference_index(repo_with_mixed_tokens)
        assert idx.globs == set(), (
            f"build_reference_index populated globs={idx.globs!r}; "
            "must remain empty under exact-only contract"
        )
        assert idx.stems == set(), (
            f"build_reference_index populated stems={idx.stems!r}; "
            "must remain empty under exact-only contract"
        )

    def test_finding_markers_still_detected(
        self, repo_with_mixed_tokens: Path,
    ) -> None:
        """The finding-marker detection (used for docs-bucket
        protection) is orthogonal to the exact-only change and must
        keep working."""
        idx = build_reference_index(repo_with_mixed_tokens)
        findings = (repo_with_mixed_tokens / "docs" / "FINDINGS.md").resolve()
        assert findings in idx.finding_files


# ──────────────────────────────────────────────────────────────────────
# Regression: the .gitignore *.pt anti-pattern stays buried
# ──────────────────────────────────────────────────────────────────────


class TestGitignoreStylePatternsAreInert:
    """An earlier matcher treated any `.gitignore` entry like `*.pt` as
    a reference, which silently neutered the entire delete plan. The
    exact-only contract makes this regression unfalsifiable — glob
    tokens are never references, full stop. Pinned here so a future
    refactor can't bring it back."""

    def test_gitignore_pt_glob_does_not_protect_orphan(
        self, tmp_path: Path,
    ) -> None:
        (tmp_path / "lfs_checkpoints").mkdir()
        (tmp_path / "lfs_checkpoints" / "completely_orphan_step9000.pt") \
            .write_bytes(b"x" * 16)
        (tmp_path / ".gitignore").write_text(
            "*.pt\n*.log\n*.mem\n", encoding="utf-8",
        )
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "README.md").write_text(
            "# Docs\n", encoding="utf-8",
        )

        idx = build_reference_index(
            tmp_path,
            skip_dirs=("logs", "lfs_checkpoints", "checkpoints"),
        )
        assert not idx.references("completely_orphan_step9000.pt")

    def test_extension_only_glob_is_inert(self) -> None:
        """Asserts the symmetric direct-API case: even if a caller
        somehow gets ``*.pt`` into ``idx.globs``, ``references()``
        must return False for any ``.pt`` basename."""
        idx = ReferenceIndex()
        idx.globs.add("*.pt")
        assert idx.references("anything.pt") is False
        assert idx.references("nothing.log") is False


# ──────────────────────────────────────────────────────────────────────
# text_suffixes parameter — scope which file types count as references
# ──────────────────────────────────────────────────────────────────────


class TestBuildReferenceIndexSuffixScope:
    """``build_reference_index(..., text_suffixes={...})`` lets callers
    restrict which file types are scanned for reference tokens. The
    LFS pruner uses ``text_suffixes={".md"}`` so only scientific
    records (FINDINGS.md, technical_report.md, archived findings) can
    pin a checkpoint — random docstring examples in `.py` files, test
    fixture names, JSON ood-results blobs, and Claude permission
    allow-lists no longer accidentally protect LFS pointers.
    """

    @pytest.fixture
    def repo_with_refs_across_filetypes(self, tmp_path: Path) -> Path:
        """Same basename ``cited_in_md_only.pt`` is referenced in
        FINDINGS.md, while ``cited_in_py_only.pt`` is referenced in a
        `.py` file and ``cited_in_json_only.pt`` lives in a `.json`."""
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "FINDINGS.md").write_text(
            "# Findings\n\n"
            "## H42 — keep me\n\n"
            "**Status.** ✅ CONFIRMED — see `cited_in_md_only.pt`.\n",
            encoding="utf-8",
        )
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "example.py").write_text(
            'CKPT = "lfs_checkpoints/cited_in_py_only.pt"\n',
            encoding="utf-8",
        )
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "ood.json").write_text(
            '{"checkpoint_path": "lfs_checkpoints/cited_in_json_only.pt"}\n',
            encoding="utf-8",
        )
        return tmp_path

    def test_default_scans_all_text_filetypes(
        self, repo_with_refs_across_filetypes: Path,
    ) -> None:
        """Without ``text_suffixes`` override, all of py/json/md are
        scanned — back-compat with the existing ``brian clean`` flow
        for the logs / docs / checkpoints buckets."""
        idx = build_reference_index(repo_with_refs_across_filetypes)
        assert idx.references("cited_in_md_only.pt")
        assert idx.references("cited_in_py_only.pt")
        assert idx.references("cited_in_json_only.pt")

    def test_md_only_skips_py_and_json(
        self, repo_with_refs_across_filetypes: Path,
    ) -> None:
        """``text_suffixes={".md"}`` restricts scanning to markdown.
        References in `.py` / `.json` files are invisible."""
        idx = build_reference_index(
            repo_with_refs_across_filetypes,
            text_suffixes={".md"},
        )
        assert idx.references("cited_in_md_only.pt"), (
            "md-cited file must still be protected when scope is md-only"
        )
        assert not idx.references("cited_in_py_only.pt"), (
            "py-only reference must NOT protect a file when scope is md-only"
        )
        assert not idx.references("cited_in_json_only.pt"), (
            "json-only reference must NOT protect a file when scope is md-only"
        )

    def test_custom_suffixes_set(
        self, repo_with_refs_across_filetypes: Path,
    ) -> None:
        """A custom set ``{".md", ".json"}`` scans md+json, skips py."""
        idx = build_reference_index(
            repo_with_refs_across_filetypes,
            text_suffixes={".md", ".json"},
        )
        assert idx.references("cited_in_md_only.pt")
        assert idx.references("cited_in_json_only.pt")
        assert not idx.references("cited_in_py_only.pt")

    def test_empty_suffixes_scans_nothing(
        self, repo_with_refs_across_filetypes: Path,
    ) -> None:
        """``text_suffixes=set()`` is a degenerate but valid input —
        nothing gets scanned, nothing gets indexed. Pinned so a
        caller that builds the set dynamically and ends up with
        an empty one gets a defined behaviour."""
        idx = build_reference_index(
            repo_with_refs_across_filetypes,
            text_suffixes=set(),
        )
        assert idx.exact == set()
        assert idx.finding_files == set()
        assert not idx.references("cited_in_md_only.pt")

    def test_suffix_match_is_case_insensitive(
        self, tmp_path: Path,
    ) -> None:
        """``text_suffixes={".md"}`` matches ``.MD`` / ``.Md`` etc.
        Same case-insensitivity rule as the existing _TEXT_SUFFIXES
        gate uses ``p.suffix.lower()``."""
        (tmp_path / "FINDINGS.MD").write_text(
            "See `caps_suffix.pt`.\n", encoding="utf-8",
        )
        idx = build_reference_index(
            tmp_path, text_suffixes={".md"},
        )
        assert idx.references("caps_suffix.pt")


# ──────────────────────────────────────────────────────────────────────
# LFS pruner integration — MD-only scope at the boundary
# ──────────────────────────────────────────────────────────────────────


class TestLfsPrunerUsesMdOnlyScope:
    """End-to-end pin: ``neuroslm.tools.clean_lfs.run`` builds its
    reference index with ``text_suffixes={".md"}`` so that an LFS
    checkpoint referenced ONLY by a `.py` / `.json` / `.toml` is
    correctly marked prunable. Markdown references still protect.
    """

    def _make_lfs_pointer(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{'a' * 64}\n"
            "size 12345\n",
            encoding="utf-8",
        )

    def test_md_ref_protects_lfs_pointer(self, tmp_path: Path) -> None:
        from neuroslm.tools import clean_lfs as cl

        ck = tmp_path / "lfs_checkpoints" / "run_a"
        for step in (1000, 2000, 3000, 4000, 5000):
            self._make_lfs_pointer(ck / f"step{step:05d}.pt")
        # Reference one of the would-be-pruned steps in FINDINGS.md.
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "FINDINGS.md").write_text(
            "# Findings\n\nSee `step01000.pt` for H42 evidence.\n",
            encoding="utf-8",
        )

        rc = cl.run(
            root=tmp_path, force=False, keep_recent=2, use_git=False,
        )
        assert rc == 0
        # step01000 must survive because FINDINGS.md cites it.
        assert (ck / "step01000.pt").exists()

    def test_py_ref_does_not_protect_lfs_pointer(
        self, tmp_path: Path,
    ) -> None:
        """A `.py` file citing the checkpoint must NOT protect it
        under the MD-only scope. The oldest step gets pruned even
        though a `.py` docstring names it."""
        from neuroslm.tools import clean_lfs as cl

        ck = tmp_path / "lfs_checkpoints" / "run_a"
        for step in (1000, 2000, 3000, 4000, 5000):
            self._make_lfs_pointer(ck / f"step{step:05d}.pt")

        # The ONLY reference is in a `.py` docstring.
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "example.py").write_text(
            '"""\n'
            'Example usage:\n'
            '  CKPT=lfs_checkpoints/run_a/step01000.pt python eval.py\n'
            '"""\n',
            encoding="utf-8",
        )

        rc = cl.run(
            root=tmp_path, force=True, keep_recent=2, use_git=False,
        )
        assert rc == 0
        # step01000 must be GONE — `.py` references no longer protect.
        assert not (ck / "step01000.pt").exists(), (
            "py-only reference should NOT have protected this LFS pointer"
        )
        # Top-2 by step survive.
        assert (ck / "step04000.pt").exists()
        assert (ck / "step05000.pt").exists()

    def test_json_ref_does_not_protect_lfs_pointer(
        self, tmp_path: Path,
    ) -> None:
        """A `.json` ood-results blob naming the checkpoint must NOT
        protect it under the MD-only scope."""
        from neuroslm.tools import clean_lfs as cl

        ck = tmp_path / "lfs_checkpoints" / "run_a"
        for step in (1000, 2000, 3000, 4000, 5000):
            self._make_lfs_pointer(ck / f"step{step:05d}.pt")

        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "ood.json").write_text(
            '{"checkpoint_path": "lfs_checkpoints/run_a/step01000.pt",\n'
            ' "ppl": 42.0}\n',
            encoding="utf-8",
        )

        rc = cl.run(
            root=tmp_path, force=True, keep_recent=2, use_git=False,
        )
        assert rc == 0
        assert not (ck / "step01000.pt").exists(), (
            "json-only reference should NOT have protected this LFS pointer"
        )


# ──────────────────────────────────────────────────────────────────────
# Folder-name regex (_FOLDER_TOKEN_RE) — per-run log folders only
# ──────────────────────────────────────────────────────────────────────


class TestFolderTokenRegexIndexesRunFolders:
    """The 0001 logs-layout migration writes each training run into
    ``logs/<YYYYMMDD-HHMMSS>_<arch>_<params>_<sha>/``. These folder
    names carry NO file suffix, so the suffix-anchored ``_TOKEN_RE``
    can never see them. ``_FOLDER_TOKEN_RE`` is a second, narrowly
    anchored regex (``\\d{8}-\\d{6}_<word>``) that picks the bare folder
    name out of path-form citations and seeds it into ``idx.exact``,
    so ``brian clean logs`` can protect cited run folders without ever
    falling back to stem / glob matching.

    The pattern is *intentionally narrow* — anchored on the
    date-prefix shape — so arbitrary tokens (``foo_bar_baz``) don't
    accidentally pollute the index. Regression-pinned here so future
    refactors can't widen it into something that silently re-protects
    half the repo."""

    def test_path_form_citation_indexes_folder_basename(
        self, tmp_path: Path,
    ) -> None:
        """A doc citing the folder path-form ⇒ basename in ``idx.exact``."""
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "FINDINGS.md").write_text(
            "H99 evidence: `logs/20260602-110000_arch_42M_cited_def456/"
            "train.log`.\n",
            encoding="utf-8",
        )
        idx = build_reference_index(tmp_path)
        assert "20260602-110000_arch_42M_cited_def456" in idx.exact
        # And the basename of the contained file is also indexed via
        # the standard ``_TOKEN_RE`` path — but ``train.log`` is too
        # generic to be a useful folder discriminator. The folder
        # protection MUST come from the folder name itself.
        assert "train.log" in idx.exact

    def test_bare_folder_citation_is_indexed(
        self, tmp_path: Path,
    ) -> None:
        """A bare folder citation (no ``/train.log`` suffix) still
        registers — the regex anchor is the date prefix, not the
        trailing slash."""
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "FINDINGS.md").write_text(
            "See run `20260603-120000_arch_84M_bare_xyz789` for H100.\n",
            encoding="utf-8",
        )
        idx = build_reference_index(tmp_path)
        assert "20260603-120000_arch_84M_bare_xyz789" in idx.exact

    def test_non_date_prefixed_tokens_are_not_indexed(
        self, tmp_path: Path,
    ) -> None:
        """The regex is anchored on ``\\d{8}-\\d{6}_`` — arbitrary
        underscore-joined tokens must NOT match, so the folder
        protection set stays tight."""
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "FINDINGS.md").write_text(
            "Random tokens: foo_bar_baz qux_42M_step1k "
            "not_a_date_2026-arch.\n",
            encoding="utf-8",
        )
        idx = build_reference_index(tmp_path)
        assert "foo_bar_baz" not in idx.exact
        assert "qux_42M_step1k" not in idx.exact
        assert "not_a_date_2026-arch" not in idx.exact

    def test_uncited_folder_name_is_not_indexed(
        self, tmp_path: Path,
    ) -> None:
        """If no scanned text file mentions the folder name, the index
        must NOT contain it (this is what makes the
        protect-by-citation contract actually work)."""
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "FINDINGS.md").write_text(
            "No mention of any run folders here.\n", encoding="utf-8",
        )
        # And a real-but-uncited folder on disk:
        run_dir = tmp_path / "logs" / "20260604-130000_arch_42M_orphan_aaa111"
        run_dir.mkdir(parents=True)
        (run_dir / "train.log").write_text("step 1\n", encoding="utf-8")
        idx = build_reference_index(tmp_path)
        assert "20260604-130000_arch_42M_orphan_aaa111" not in idx.exact

