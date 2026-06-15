# -*- coding: utf-8 -*-
"""Multi-file `.neuro` DSL — path resolution and folder loading (Stage 1).

Stage 1 of the multi-file DSL. Two pieces:

  PathResolver  — turns import specifiers into absolute file paths
                  using a JavaScript/mjs-style scheme:
                     `@/foo`     absolute, from architecture root
                     `./foo`     relative to the importing file
                     `../foo`    one level up from the importing file
                  Folder-as-module fallback: if `foo` is a directory with
                  `index.neuro` inside, the specifier resolves to that file.

  FolderLoader  — walks an architecture root and returns every .neuro file
                  it contains as `{absolute_path: raw_source}`.

Later stages add a parser for the `module` / `import` / `export` syntax,
symbol tables, reference resolution, and shared `dynamics`/`function`
definitions. This file is foundational only.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .equations import DynamicsDecl


def _discover_repo_root(start: Path) -> Optional[Path]:
    """Walk up from ``start`` looking for a ``pyproject.toml`` marker.

    Returns the directory containing the first ``pyproject.toml`` found,
    or ``None`` if walking reaches the filesystem root without finding
    one. Pure-function helper — never raises.
    """
    cur = Path(start).resolve()
    if cur.is_file():
        cur = cur.parent
    # Bound the loop by parent-chain length so a pathological FS can't
    # spin forever.
    for candidate in (cur, *cur.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


class PathResolver:
    """Resolve `.neuro` import specifiers to absolute file paths.

    The resolver normalizes four specifier scopes:

      * ``@/<path>``       — anchored at the architecture root.
      * ``@brian/<path>``  — anchored at the repository root (so
        ``@brian/lib/equations`` resolves to ``<repo>/lib/equations.neuro``,
        ``@brian/architectures/master/arch`` resolves to that file).
      * ``@lib/<path>``    — anchored at ``<repo>/lib/`` (shorthand
        for the shared library; equivalent to ``@brian/lib/<path>``).
      * ``./<path>`` / ``../<path>`` — relative to the importing file.

    All forms fall back to ``index.neuro`` when a specifier names a
    directory.

    Args:
        arch_root: filesystem path to the architecture root. The ``@/``
            prefix is anchored here.
        repo_root: optional explicit repository root containing the
            shared ``/lib/`` directory used by ``@lib/...`` and as the
            anchor for ``@brian/...``. If not given, the resolver walks
            up from ``arch_root`` looking for ``pyproject.toml``.
    """

    def __init__(
        self,
        arch_root: Path,
        repo_root: Optional[Path] = None,
    ):
        self.arch_root = Path(arch_root).resolve()
        if repo_root is not None:
            self.repo_root: Optional[Path] = Path(repo_root).resolve()
        else:
            self.repo_root = _discover_repo_root(self.arch_root)

    # ── Public ──────────────────────────────────────────────────────

    def resolve(self, specifier: str, from_file: Optional[Path]) -> Path:
        """Resolve `specifier` (as used in an `import "<specifier>"`).

        Args:
            specifier: the import target, e.g. ``@/lib/dynamics``,
                ``@lib/features/hyperbolic_attention``,
                ``@brian/lib/equations``, ``./layers``.
            from_file: the file containing the import (required for
                relative paths; ignored for absolute prefixes).

        Returns:
            Absolute resolved path of the target `.neuro` file.

        Raises:
            ValueError: malformed specifier, relative-without-from_file,
                or path that escapes its scope's root.
            FileNotFoundError: resolved path doesn't exist (neither as a
                ``.neuro`` file nor a folder-module).
        """
        # ── @brian/<path> ── anchored at <repo_root> ────────────────
        if specifier.startswith("@brian/"):
            rest = specifier[len("@brian/"):]
            if self.repo_root is not None:
                scope_root = self.repo_root
                base = scope_root
                # rest already stripped; fall through to file resolution.
            else:
                # No repo_root → standalone workspace (typical for
                # unfolded DNA in pytest tmp dirs, vast.ai boxes, or
                # colab). Look in the workspace itself as the
                # "repo-equivalent" root.
                ws_candidate = (self.arch_root / rest).resolve()
                ws_resolved = self._try_resolve_file(ws_candidate)
                if ws_resolved is not None:
                    try:
                        ws_resolved.relative_to(self.arch_root)
                        return ws_resolved
                    except ValueError:
                        pass  # escapes workspace → error below
                raise ValueError(
                    f"specifier {specifier!r} uses the @brian/ prefix "
                    f"but no repo root was discovered (no pyproject.toml "
                    f"found by walking up from {self.arch_root}) and the "
                    f"workspace fallback at {self.arch_root / rest!s} "
                    f"does not exist either; pass repo_root=... "
                    f"explicitly to PathResolver"
                )
        # ── @lib/<path> ── anchored at <repo_root>/lib/ ─────────────
        elif specifier.startswith("@lib/"):
            rest = specifier[len("@lib/"):]
            if self.repo_root is not None:
                scope_root = self.repo_root / "lib"
                base = scope_root
            else:
                # Standalone-workspace fallback mirrors @brian/'s, but
                # rooted at <arch_root>/lib/ (where unfolded DNA
                # places its bundled lib).
                ws_candidate = (self.arch_root / "lib" / rest).resolve()
                ws_resolved = self._try_resolve_file(ws_candidate)
                if ws_resolved is not None:
                    try:
                        ws_resolved.relative_to(self.arch_root)
                        return ws_resolved
                    except ValueError:
                        pass
                raise ValueError(
                    f"specifier {specifier!r} uses the @lib/ prefix "
                    f"but no repo root was discovered (no pyproject.toml "
                    f"found by walking up from {self.arch_root}) and no "
                    f"workspace-local lib/{rest} was found either; pass "
                    f"repo_root=... explicitly to PathResolver"
                )
        elif specifier.startswith("@/"):
            scope_root = self.arch_root
            base = self.arch_root
            rest = specifier[2:]
        elif specifier.startswith("./") or specifier.startswith("../"):
            if from_file is None:
                raise ValueError(
                    f"relative specifier {specifier!r} needs from_file"
                )
            scope_root = self.arch_root
            base = Path(from_file).resolve().parent
            rest = specifier
        else:
            raise ValueError(
                f"unsupported specifier {specifier!r}: must start with "
                "@/ (architecture-root), @brian/ (repo root), @lib/ "
                "(shared lib), ./, or ../ (relative)"
            )

        candidate = (base / rest).resolve()

        # Reject specifiers that escape their scope's root. For @brian/
        # the scope is <repo_root>/architectures/lib/ (NOT the whole
        # repo); for the other scopes it's the architecture root.
        try:
            candidate.relative_to(scope_root)
        except ValueError:
            raise ValueError(
                f"specifier {specifier!r} escapes scope root {scope_root}"
            )

        # File resolution: try exact path, then with .neuro suffix, then
        # folder/index.neuro fallback.
        if candidate.is_file() and candidate.suffix == ".neuro":
            return candidate

        with_suffix = candidate.with_suffix(".neuro") if candidate.suffix == "" else candidate
        if with_suffix.is_file():
            return with_suffix

        index_in_folder = candidate / "index.neuro"
        if index_in_folder.is_file():
            return index_in_folder

        raise FileNotFoundError(
            f"could not resolve specifier {specifier!r}: tried "
            f"{candidate}, {with_suffix}, and {index_in_folder}"
        )

    @staticmethod
    def _try_resolve_file(candidate: Path) -> Optional[Path]:
        """Best-effort file resolution: returns the matching ``.neuro``
        path, or ``None`` if neither the exact path, the ``.neuro``
        suffix, nor the ``index.neuro`` fallback exists.

        Used by the ``@brian/`` workspace-local lookup, which needs a
        "does this exist?" check without raising — control then falls
        through to the canonical repo-anchored resolution.
        """
        if candidate.is_file() and candidate.suffix == ".neuro":
            return candidate
        with_suffix = (
            candidate.with_suffix(".neuro")
            if candidate.suffix == ""
            else candidate
        )
        if with_suffix.is_file():
            return with_suffix
        index_in_folder = candidate / "index.neuro"
        if index_in_folder.is_file():
            return index_in_folder
        return None


class FolderLoader:
    """Discover every `.neuro` file in an architecture folder.

    A thin wrapper that also exposes whether the folder has an `arch.neuro`
    at its root — that file is the "package config" of the architecture
    and is required by later stages.
    """

    def __init__(self, arch_root: Path):
        self.arch_root = Path(arch_root).resolve()

    def discover(self) -> Dict[Path, str]:
        """Return `{absolute_path: source}` for every .neuro file.

        Raises:
            ValueError: folder contains no .neuro files at all.
        """
        sources: Dict[Path, str] = {}
        for path in self.arch_root.rglob("*.neuro"):
            sources[path.resolve()] = path.read_text(encoding="utf-8")

        if not sources:
            raise ValueError(f"no .neuro files under {self.arch_root}")

        return sources

    def has_arch_root(self) -> bool:
        """True if `<arch_root>/arch.neuro` exists.

        Stage-1 callers use this as a sanity check before treating a
        directory as a full architecture.
        """
        return (self.arch_root / "arch.neuro").is_file()


# ════════════════════════════════════════════════════════════════════════
# Stage 2 — module parser: imports, exports, declaration extraction
# ════════════════════════════════════════════════════════════════════════
#
# A .neuro file is implicitly a module. Its public surface is whatever's
# prefixed with `export`; everything else is private. Imports are parsed
# into structured `ImportDecl`s but not resolved here (Stage 3's job).
#
# Implementation note: the parser walks the source twice — once to extract
# import statements line-by-line, then a brace-matching scan to slice
# declarations (population/synapse/dynamics/...) into their text spans.
# We keep the raw text for each declaration; downstream compiler stages
# already handle property parsing.


@dataclass
class ImportDecl:
    """A parsed `import` statement.

    Three forms:
      import { foo, bar }         from "..."   → names=["foo","bar"]
      import { foo as alias }     from "..."   → names=["foo"], aliases={"foo":"alias"}
      import "..."                              → names=[]  (side-effect)
    """
    specifier: str                     # the path string, e.g. "@/lib/dyn"
    names: List[str] = field(default_factory=list)
    aliases: Dict[str, str] = field(default_factory=dict)


@dataclass
class ModuleAST:
    """A parsed .neuro file, ready for reference resolution (Stage 3)."""
    path: Path
    imports: List[ImportDecl] = field(default_factory=list)
    exports: Dict[str, str] = field(default_factory=dict)   # name → decl text
    private: Dict[str, str] = field(default_factory=dict)   # name → decl text
    architecture: Optional[Dict] = None     # {name, properties} or None


# Top-level keywords whose blocks are extracted as declarations. Each block
# has the form `<keyword> <name> { ... }` (population, dynamics, function)
# or `<keyword> <src> -> <tgt> { ... }` (synapse, modulation).
_DECL_KEYWORDS_NAMED = (
    "population",
    "neurotransmitter",
    "dynamics",
    "function",
    "equation",
    "formal_spec",
    "sheaf",
    "complex",  # THSD simplicial complexes (Phase 6+)
    # §6.5 — genetic orchestrator declarations
    "gene",
    "protein",
    "metric",
    # §14 — toggleable mechanism block (see _extract_features in compiler.py)
    "feature",
    # §15 — LanguageCortex DSL surface (2026-06-15): declarative
    # expert-ensemble / teacher / CFD wiring. Each keyword maps to
    # the matching _extract_* in compiler.py.
    "expert",
    "distillation",
    "funnel",
    "warmup",
)
_DECL_KEYWORDS_ARROW = ("synapse", "modulation")


def parse_module(source: str, path: Path) -> ModuleAST:
    """Parse a .neuro file into its module-level structure.

    Recognised top-level constructs:
        architecture <name> { ... }   (only legal in arch.neuro)
        import { <names> } from "..."
        import "..."
        export <decl>
        <decl>                        (private)

    where `<decl>` is one of: population, synapse, neurotransmitter,
    modulation, dynamics, function, formal_spec, sheaf.

    Raises:
        ValueError: malformed import, dangling `export`, or duplicate
            exported name within the same file.
    """
    ast = ModuleAST(path=Path(path))

    # First strip out import statements (line-oriented), then handle the
    # rest of the source as block declarations.
    remaining_lines = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("import"):
            ast.imports.append(_parse_import(stripped))
        else:
            remaining_lines.append(line)
    body = "\n".join(remaining_lines)

    # Now walk the body looking for top-level blocks. We rely on the same
    # brace-matching technique used in compiler.py's `_split_top_level`,
    # but at the level of complete declarations.
    pos = 0
    while pos < len(body):
        ch = body[pos]
        if ch.isspace() or ch == "#":
            # Skip whitespace and comment lines
            if ch == "#":
                nl = body.find("\n", pos)
                pos = len(body) if nl == -1 else nl + 1
            else:
                pos += 1
            continue

        # Match `export` prefix
        is_export = False
        if body.startswith("export", pos) and (
            pos + 6 >= len(body) or not body[pos + 6].isalnum()
        ):
            is_export = True
            pos += 6
            while pos < len(body) and body[pos].isspace():
                pos += 1
            if pos >= len(body):
                raise ValueError(
                    f"{path}: dangling `export` with no declaration"
                )

        # Architecture block (top-level, only in arch.neuro)
        if body.startswith("architecture", pos):
            arch_name, arch_props, end = _slice_named_block(body, pos, "architecture")
            ast.architecture = {
                "name": arch_name,
                "properties": _parse_simple_props(arch_props),
            }
            pos = end
            continue

        # Named block: keyword <name> { ... }
        matched_named = False
        for kw in _DECL_KEYWORDS_NAMED:
            if body.startswith(kw, pos) and (
                pos + len(kw) < len(body) and body[pos + len(kw)].isspace()
            ):
                name, _, end = _slice_named_block(body, pos, kw)
                decl_text = body[pos:end]
                _record(ast, name, decl_text, is_export, path)
                pos = end
                matched_named = True
                break
        if matched_named:
            continue

        # Arrow block: keyword <src> -> <tgt> { ... }
        matched_arrow = False
        for kw in _DECL_KEYWORDS_ARROW:
            if body.startswith(kw, pos) and (
                pos + len(kw) < len(body) and body[pos + len(kw)].isspace()
            ):
                src, tgt, end = _slice_arrow_block(body, pos, kw)
                key = f"{src}__{tgt}"
                decl_text = body[pos:end]
                _record(ast, key, decl_text, is_export, path)
                pos = end
                matched_arrow = True
                break
        if matched_arrow:
            continue

        # Unrecognised token → skip to next line so a stray annotation
        # doesn't halt the whole parse.
        nl = body.find("\n", pos)
        pos = len(body) if nl == -1 else nl + 1

    return ast


# ── Helpers ────────────────────────────────────────────────────────────

# `import { foo, bar as baz } from "spec"`  or  `import "spec"`
_IMPORT_NAMED_RE = re.compile(
    r'^import\s*\{([^}]*)\}\s*from\s*"([^"]+)"\s*$'
)
_IMPORT_BARE_RE = re.compile(r'^import\s*"([^"]+)"\s*$')


def _parse_import(line: str) -> ImportDecl:
    m = _IMPORT_BARE_RE.match(line)
    if m:
        return ImportDecl(specifier=m.group(1))

    m = _IMPORT_NAMED_RE.match(line)
    if not m:
        raise ValueError(f"could not parse import: {line!r}")

    names_str, spec = m.group(1), m.group(2)
    names: List[str] = []
    aliases: Dict[str, str] = {}
    for part in names_str.split(","):
        part = part.strip()
        if not part:
            continue
        if " as " in part:
            orig, alias = (s.strip() for s in part.split(" as ", 1))
            names.append(orig)
            aliases[orig] = alias
        else:
            names.append(part)

    return ImportDecl(specifier=spec, names=names, aliases=aliases)


def _slice_named_block(body: str, start: int, keyword: str):
    """Slice `<keyword> <name> [(args)] { ... }`; return (name, body_inside, end_pos).

    The optional `(args)` between name and brace is for `function` declarations
    (e.g. `function decay(x, alpha) { ... }`); other keywords have nothing
    between name and `{`. The parens are silently skipped here — Stage 4's
    `_decl_kind_and_body` re-extracts them from the full decl text.
    """
    pos = start + len(keyword)
    while pos < len(body) and body[pos].isspace():
        pos += 1
    name_start = pos
    while pos < len(body) and (body[pos].isalnum() or body[pos] == "_"):
        pos += 1
    name = body[name_start:pos]
    while pos < len(body) and body[pos].isspace():
        pos += 1
    # Skip past `(args)` if present (function declarations)
    if pos < len(body) and body[pos] == "(":
        depth = 1
        pos += 1
        while pos < len(body) and depth > 0:
            if body[pos] == "(":
                depth += 1
            elif body[pos] == ")":
                depth -= 1
            pos += 1
        while pos < len(body) and body[pos].isspace():
            pos += 1
    if pos >= len(body) or body[pos] != "{":
        raise ValueError(f"expected `{{` after `{keyword} {name}`")
    inner, end = _slice_braced(body, pos)
    return name, inner, end


def _slice_arrow_block(body: str, start: int, keyword: str):
    """Slice `<keyword> <src> -> <tgt> { ... }`; return (src, tgt, end_pos)."""
    pos = start + len(keyword)
    while pos < len(body) and body[pos].isspace():
        pos += 1
    src_start = pos
    while pos < len(body) and (body[pos].isalnum() or body[pos] == "_"):
        pos += 1
    src = body[src_start:pos]
    while pos < len(body) and body[pos].isspace():
        pos += 1
    if not body.startswith("->", pos):
        raise ValueError(f"expected `->` in {keyword} declaration at pos {pos}")
    pos += 2
    while pos < len(body) and body[pos].isspace():
        pos += 1
    tgt_start = pos
    while pos < len(body) and (body[pos].isalnum() or body[pos] == "_"):
        pos += 1
    tgt = body[tgt_start:pos]
    while pos < len(body) and body[pos].isspace():
        pos += 1
    # Body block may be omitted for arrow declarations (e.g. `synapse a -> b`)
    if pos >= len(body) or body[pos] != "{":
        return src, tgt, pos
    _, end = _slice_braced(body, pos)
    return src, tgt, end


def _slice_braced(body: str, start: int):
    """Given `body[start] == '{'`, return (inside_text, position_after_close).

    Tracks string literals (so braces inside `"..."` are ignored) AND
    line comments starting with `#` (so a stray apostrophe in a comment
    like `# there's no bottleneck` does NOT enter string-tracking mode
    and swallow subsequent closing braces).
    """
    assert body[start] == "{"
    depth = 1
    i = start + 1
    in_str = None
    while i < len(body) and depth > 0:
        ch = body[i]
        if in_str:
            if ch == in_str:
                in_str = None
        elif ch == "#":
            # Skip to end of line — comments are not parsed.
            nl = body.find("\n", i)
            i = len(body) if nl == -1 else nl
            continue
        elif ch in ('"', "'"):
            in_str = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    if depth != 0:
        raise ValueError(f"unbalanced braces starting at pos {start}")
    return body[start + 1 : i - 1], i


def _parse_simple_props(body: str) -> Dict[str, str]:
    """Tiny key:value parser for the `architecture` block — strings as-is."""
    out: Dict[str, str] = {}
    for raw in body.split(","):
        raw = raw.strip()
        if not raw or ":" not in raw:
            continue
        k, v = raw.split(":", 1)
        out[k.strip()] = v.strip().strip('"\'')
    return out


def _record(ast: ModuleAST, name: str, decl_text: str,
            is_export: bool, path: Path) -> None:
    """Store the decl under exports or private, rejecting duplicates.

    Skip empty names (typically THSD blocks like 'complex' that don't
    parse as v2.0 declarations and are handled separately).
    """
    # Skip empty names (THSD blocks, etc.)
    if not name or not name.strip():
        return

    bucket = ast.exports if is_export else ast.private
    other = ast.exports if not is_export else ast.private
    if name in bucket or name in other:
        raise ValueError(f"{path}: duplicate declaration {name!r}")
    bucket[name] = decl_text


# ════════════════════════════════════════════════════════════════════════
# Stage 3 — Resolver: discover + parse + link cross-file imports
# ════════════════════════════════════════════════════════════════════════
#
# After Stage 1 (file discovery) and Stage 2 (per-file parsing), the
# Resolver glues everything together: it walks the architecture root,
# parses every file into a ModuleAST, resolves each import specifier to
# its target file, and validates that every imported name is actually
# exported by that target.
#
# The result is a `ResolvedProgram` — the input shape Stage 4 (lib-defined
# dynamics/function lookup) and Stage 5 (synapse/modulation equation
# codegen) consume to walk the architecture as a single coherent unit.


class ResolverError(Exception):
    """Raised when the multi-file program is structurally invalid:
    missing arch.neuro, unresolvable import path, or imported name
    not actually exported by its target file.
    """
    pass


@dataclass
class FunctionDecl:
    """A user-defined `function` from a lib file.

    Function calls in equation strings get inlined: a call `decay(x, 0.1)`
    in a population's `equation:` field is rewritten as `(1 - 0.1) * x`
    before SymPy parsing. (Inlining lives in Stage 5; Stage 4 only parses
    and collects.)
    """
    name: str
    args: List[str]
    equation: str


@dataclass
class ResolvedProgram:
    """A fully-linked multi-file architecture program.

    Attributes:
        arch_root: absolute path to the architecture root
        modules:   {absolute_file_path: ModuleAST}
        import_map: {file_path: {local_alias: (target_file, source_name)}}
                    — what each name refers to within each file's scope
        architecture: the `architecture { name, properties }` block from
                      arch.neuro, or {} if none
        user_dynamics: {(file, name): DynamicsDecl} — `dynamics` blocks
                       parsed from lib files (typically exported)
        user_functions: {(file, name): FunctionDecl} — same for `function`
                        blocks
    """
    arch_root: Path
    modules: Dict[Path, ModuleAST] = field(default_factory=dict)
    import_map: Dict[Path, Dict[str, tuple]] = field(default_factory=dict)
    architecture: Dict = field(default_factory=dict)
    user_dynamics: Dict[Tuple[Path, str], DynamicsDecl] = field(default_factory=dict)
    user_functions: Dict[Tuple[Path, str], FunctionDecl] = field(default_factory=dict)

    def lookup_dynamics(self, from_file: Path, name: str) -> Optional[DynamicsDecl]:
        """Look up a user-defined dynamics by the name visible in `from_file`.

        Search order:
          1. names this file imported (resolves alias → original name in
             target file)
          2. dynamics declared inside this file
          3. None (codegen will fall back to its built-in DYNAMICS_DECLS)
        """
        from_file = Path(from_file).resolve()

        # Imported dynamics?
        imports = self.import_map.get(from_file, {})
        if name in imports:
            target_file, src_name = imports[name]
            key = (target_file, src_name)
            if key in self.user_dynamics:
                return self.user_dynamics[key]

        # Locally defined dynamics?
        local_key = (from_file, name)
        if local_key in self.user_dynamics:
            return self.user_dynamics[local_key]

        return None

    def lookup_function(self, from_file: Path, name: str) -> Optional[FunctionDecl]:
        """Mirror of `lookup_dynamics` for `function` decls."""
        from_file = Path(from_file).resolve()

        imports = self.import_map.get(from_file, {})
        if name in imports:
            target_file, src_name = imports[name]
            key = (target_file, src_name)
            if key in self.user_functions:
                return self.user_functions[key]

        local_key = (from_file, name)
        if local_key in self.user_functions:
            return self.user_functions[local_key]

        return None

    def lookup(self, from_file: Path, name: str) -> str:
        """Return the raw declaration text for `name` in `from_file`'s scope.

        Search order:
          1. local exports + private declarations in `from_file`
          2. imported aliases — resolved to (target_file, source_name)
             and then looked up in the target's exports

        Raises ResolverError if the name resolves to nothing.
        """
        from_file = Path(from_file).resolve()
        ast = self.modules.get(from_file)
        if ast is None:
            raise ResolverError(f"unknown file {from_file}")

        if name in ast.exports:
            return ast.exports[name]
        if name in ast.private:
            return ast.private[name]

        imports = self.import_map.get(from_file, {})
        if name in imports:
            target_file, src_name = imports[name]
            target_ast = self.modules.get(target_file)
            if target_ast is None or src_name not in target_ast.exports:
                raise ResolverError(
                    f"{from_file}: import {name!r} → "
                    f"{target_file}::{src_name!r} not found"
                )
            return target_ast.exports[src_name]

        raise ResolverError(f"{from_file}: symbol {name!r} not found")


class Resolver:
    """Walk an architecture folder and produce a `ResolvedProgram`.

    Use:
        program = Resolver(arch_root).resolve()
        decl = program.lookup(file_path, "some_name")
    """

    def __init__(self, arch_root: Path):
        self.arch_root = Path(arch_root).resolve()
        self.loader = FolderLoader(self.arch_root)
        self.path_resolver = PathResolver(self.arch_root)

    def resolve(self) -> ResolvedProgram:
        if not self.loader.has_arch_root():
            raise ResolverError(
                f"missing arch.neuro at architecture root {self.arch_root}"
            )

        program = ResolvedProgram(arch_root=self.arch_root)

        # Pass 1: parse every file into a ModuleAST
        for path, src in self.loader.discover().items():
            try:
                program.modules[path] = parse_module(src, path=path)
            except ValueError as e:
                raise ResolverError(f"{path}: parse error: {e}") from e

        # Pull the `architecture { ... }` block out of arch.neuro
        arch_path = (self.arch_root / "arch.neuro").resolve()
        arch_ast = program.modules.get(arch_path)
        if arch_ast and arch_ast.architecture:
            program.architecture = arch_ast.architecture

        # Pass 1.5: lazily load shared-lib (`@brian/...`) imports. The
        # FolderLoader only walks the arch directory, so any file under
        # <repo>/lib/ that an arch module imports must be discovered and
        # parsed before Pass 2 can resolve those import targets. We iterate
        # to a fixed point because shared-lib files may themselves import
        # from other shared-lib files (or from arch-local `./` siblings,
        # though those are already loaded).
        self._lazy_load_shared_imports(program)

        # Pass 2: resolve every import to a target file + validate exports
        for file_path, ast in program.modules.items():
            file_imports: Dict[str, tuple] = {}
            for imp in ast.imports:
                try:
                    target_file = self.path_resolver.resolve(
                        imp.specifier, from_file=file_path
                    )
                except (FileNotFoundError, ValueError) as e:
                    raise ResolverError(
                        f"{file_path}: cannot resolve import {imp.specifier!r}: {e}"
                    ) from e
                target_file = target_file.resolve()

                target_ast = program.modules.get(target_file)
                if target_ast is None:
                    raise ResolverError(
                        f"{file_path}: import target {target_file} not loaded"
                    )

                for name in imp.names:
                    if name not in target_ast.exports:
                        raise ResolverError(
                            f"{file_path}: imported name {name!r} not "
                            f"exported by {target_file}"
                        )
                    local_alias = imp.aliases.get(name, name)
                    if local_alias in file_imports:
                        raise ResolverError(
                            f"{file_path}: duplicate import alias {local_alias!r}"
                        )
                    file_imports[local_alias] = (target_file, name)

            program.import_map[file_path] = file_imports

        # Pass 3: parse every `dynamics` / `function` declaration body into
        # structured form and register it on the program.
        for file_path, ast in program.modules.items():
            for name, decl_text in list(ast.exports.items()) + list(ast.private.items()):
                kind, header, body = _decl_kind_and_body(decl_text)
                if kind == "dynamics":
                    try:
                        decl = parse_dynamics_block(body)
                    except ValueError as e:
                        raise ResolverError(
                            f"{file_path}: malformed dynamics {name!r}: {e}"
                        ) from e
                    program.user_dynamics[(file_path, name)] = decl
                elif kind == "function":
                    try:
                        fn = parse_function_block(name, header, body)
                    except ValueError as e:
                        raise ResolverError(
                            f"{file_path}: malformed function {name!r}: {e}"
                        ) from e
                    program.user_functions[(file_path, name)] = fn

        return program

    # ── Internal: lazy-load `@brian/...` imports ────────────────────

    def _lazy_load_shared_imports(self, program: "ResolvedProgram") -> None:
        """Walk every loaded module's imports; for any specifier that
        points outside the arch root (currently only ``@brian/...``),
        resolve it, load + parse the target, register it in
        ``program.modules``, and recurse until fixed point.

        Pure side-effect: mutates ``program.modules`` in place.
        """
        pending: List[Path] = []
        for file_path, ast in program.modules.items():
            for imp in ast.imports:
                target = self._maybe_shared_target(imp.specifier, file_path)
                if target is not None and target not in program.modules:
                    pending.append(target)

        loaded: set[Path] = set()
        while pending:
            target = pending.pop()
            if target in loaded or target in program.modules:
                continue
            try:
                src = target.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                raise ResolverError(
                    f"cannot read shared-lib file {target}: {e}"
                ) from e
            try:
                program.modules[target] = parse_module(src, path=target)
            except ValueError as e:
                raise ResolverError(
                    f"{target}: parse error: {e}"
                ) from e
            loaded.add(target)
            # Recurse: this newly loaded file may import other shared
            # files (e.g. lib/features/foo.neuro imports
            # @brian/equations).
            for imp in program.modules[target].imports:
                child = self._maybe_shared_target(imp.specifier, target)
                if child is not None and child not in program.modules:
                    pending.append(child)

    def _maybe_shared_target(
        self, specifier: str, from_file: Path
    ) -> Optional[Path]:
        """Return the resolved absolute path if ``specifier`` is a
        shared-lib (``@brian/...`` or ``@lib/...``) import, else
        ``None``.

        Errors during resolution are re-raised as ``ResolverError`` so
        the caller gets a single, traceable failure mode.
        """
        if not (
            specifier.startswith("@brian/") or specifier.startswith("@lib/")
        ):
            return None
        try:
            return self.path_resolver.resolve(
                specifier, from_file=from_file
            ).resolve()
        except (FileNotFoundError, ValueError) as e:
            raise ResolverError(
                f"{from_file}: cannot resolve shared-lib import "
                f"{specifier!r}: {e}"
            ) from e

# Recognise the head of a stored declaration text and pull out its body.
# The optional ``export`` prefix is stripped because declarations may be
# stored verbatim (with or without the leading keyword) depending on
# whether the parser saw an export modifier.
_DECL_HEAD_RE = re.compile(
    r'^\s*(?:export\s+)?(?P<kind>population|synapse|neurotransmitter|'
    r'modulation|dynamics|function|equation|formal_spec|sheaf|'
    r'gene|protein|metric|feature)\s+'
    r'(?P<header>[^{]*)\{'
)


def _decl_kind_and_body(decl_text: str) -> Tuple[str, str, str]:
    """Return (kind, header, body) from a stored declaration string.

    `header` is whatever sits between the keyword and the opening brace
    (the name, plus `(args)` for functions, plus `src -> tgt` for
    synapse/modulation). `body` is the brace contents.
    """
    m = _DECL_HEAD_RE.match(decl_text)
    if not m:
        return ("", "", "")
    kind = m.group("kind")
    header = m.group("header").strip()
    # Body is the text between matching braces — reuse Stage-2 brace slicer.
    open_brace = decl_text.index("{", m.end() - 1)
    body, _ = _slice_braced(decl_text, open_brace)
    return kind, header, body


def parse_dynamics_block(body: str) -> DynamicsDecl:
    """Parse the body of a `dynamics name { ... }` block into a DynamicsDecl.

    Recognised fields (all optional, but exactly one of equation/ode):
        equation:  "y = ..."           algebraic form
        ode:       "dV/dt = ..."       differential form
        params:    { name: "init", ... }
        state:     { name: "init", ... }
        constants: { name: value, ... }
    """
    props = _split_top_level_kv(body)

    equation = props.get("equation")
    ode = props.get("ode")
    if equation is not None:
        equation = _strip_quotes(equation)
    if ode is not None:
        ode = _strip_quotes(ode)

    params = _parse_subdict(props.get("params", ""))
    state = _parse_subdict(props.get("state", ""))
    constants_raw = _parse_subdict(props.get("constants", ""))

    # Constants are numeric — convert from string
    constants: Dict[str, Any] = {}
    for k, v in constants_raw.items():
        try:
            constants[k] = float(v)
            if constants[k].is_integer() and "." not in v:
                constants[k] = int(constants[k])
        except (ValueError, AttributeError):
            constants[k] = v  # leave as-is (will fail loudly downstream)

    return DynamicsDecl(
        equation=equation,
        ode=ode,
        params=params,
        state=state,
        constants=constants,
    )


def parse_function_block(name: str, header: str, body: str) -> FunctionDecl:
    """Parse `function name(<args>) { equation: "..." }`.

    `header` may be either form, depending on caller context:
      - `name(args)`  — used when Stage-4's `_decl_kind_and_body` calls in
      - `(args)`      — used when the outer parser has already extracted
                        the name and just passes the arg list
    """
    header_stripped = header.strip()
    # Allow optional leading `name` in the header; ignore it (we use the
    # explicit `name` argument).
    header_stripped = re.sub(r'^\s*\w+\s*', '', header_stripped)
    m = re.match(r'^\((?P<args>[^)]*)\)\s*$', header_stripped)
    if not m:
        raise ValueError(
            f"function {name!r}: header must be `(args)` or `name(args)`, "
            f"got {header!r}"
        )

    args = [a.strip() for a in m.group("args").split(",") if a.strip()]

    props = _split_top_level_kv(body)
    equation = props.get("equation")
    if equation is None:
        raise ValueError(f"function {name!r}: missing `equation:` field")
    equation = _strip_quotes(equation)

    return FunctionDecl(name=name, args=args, equation=equation)


# ── Property-parsing helpers (shared with parse_dynamics_block) ────────

def _split_top_level_kv(body: str) -> Dict[str, str]:
    """Split `key: value, key: { ... }, ...` at depth 0 of braces/strings."""
    out: Dict[str, str] = {}
    buf: List[str] = []
    depth = 0
    in_str: Optional[str] = None

    def flush(parts: List[str]) -> None:
        piece = "".join(parts).strip()
        if not piece:
            return
        if ":" not in piece:
            return
        k, v = piece.split(":", 1)
        out[k.strip()] = v.strip()

    for ch in body:
        if in_str:
            buf.append(ch)
            if ch == in_str:
                in_str = None
            continue
        if ch in ('"', "'"):
            in_str = ch
            buf.append(ch)
        elif ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif (ch == "," or ch == "\n") and depth == 0:
            flush(buf)
            buf = []
        else:
            buf.append(ch)
    flush(buf)
    return out


def _parse_subdict(text: str) -> Dict[str, str]:
    """Parse `{ key: value, key: value }` into a flat string→string dict."""
    if not text:
        return {}
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    return {k: _strip_quotes(v) for k, v in _split_top_level_kv(text).items()}


def _strip_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] in ('"', "'") and text[-1] == text[0]:
        return text[1:-1]
    return text


# ════════════════════════════════════════════════════════════════════════
# Stage 6 — multi-file compiler entry point
# ════════════════════════════════════════════════════════════════════════
#
# `compile_folder(arch_root)` walks an architecture folder, runs the
# resolver to validate imports, then concatenates every declaration body
# into one synthetic source string and pipes it through the existing
# single-file `NeuroMLCompiler`. This keeps all the IR construction logic
# in one place and lets the same `CodeGenerator` consume the result.
#
# Trade-off: bare names are used throughout — no module prefixing — so an
# architecture must not have name collisions across its modules. For
# rcc_bowtie this holds (every region name is unique). A future stage can
# introduce automatic module-prefix qualification if collisions ever
# become a real problem.


def compile_folder(arch_root):
    """Compile a multi-file architecture folder into a single ProgramIR.

    The concatenation order is deliberate — it matters for codegen because
    synapse routing classifies edges as forward (current-step) vs back
    (last-step) based on population declaration order.

    Emission order:
      1. NT systems + globals from arch.neuro (declared first)
      2. Populations, in the order their modules appear in arch.neuro's
         `import` list — this is the user's canonical region order
      3. Synapses + modulations from arch.neuro (declared last so every
         endpoint they reference is already in scope)
      4. Formal specs + sheaves (any remaining global decls)

    Args:
        arch_root: path to the folder containing arch.neuro

    Returns:
        ProgramIR ready for CodeGenerator
    """
    from .compiler import NeuroMLCompiler

    resolver = Resolver(Path(arch_root))
    program = resolver.resolve()
    arch_path = (Path(arch_root).resolve() / "arch.neuro")
    arch_ast = program.modules.get(arch_path)
    if arch_ast is None:
        raise ValueError(f"missing arch.neuro at {arch_path}")

    parts: List[str] = []

    def _emit(decls: Dict[str, str], kinds: tuple) -> None:
        for _, decl_text in decls.items():
            kind, _, _ = _decl_kind_and_body(decl_text)
            if kind in kinds:
                parts.append(decl_text)

    # 0. Auto-include shared equation library (if it exists).
    # This makes equation definitions available to all modules.
    # We check the arch-local ``lib/equations.neuro`` (legacy layout)
    # first, then the repo-root ``<repo>/lib/equations.neuro``
    # (2026-06-15 layout). Both are best-effort — the resolver also
    # lazy-loads any explicitly imported equation files.
    lib_equations_path = Path(arch_root).resolve() / "lib" / "equations.neuro"
    if lib_equations_path.exists():
        try:
            with open(lib_equations_path, 'r') as f:
                parts.append(f.read())
        except Exception:
            pass
    else:
        repo_root = resolver.path_resolver.repo_root
        if repo_root is not None:
            repo_lib_eq = repo_root / "lib" / "equations.neuro"
            if repo_lib_eq.exists():
                try:
                    with open(repo_lib_eq, 'r') as f:
                        parts.append(f.read())
                except Exception:
                    pass

    # 0b. Shared library equations from <repo>/lib/.
    # Any `@brian/...` or `@lib/...` import the resolver pulled in
    # lives in program.modules and outside the arch root — emit every
    # equation those files export so they're visible to the single-file
    # compiler.
    arch_root_resolved = Path(arch_root).resolve()
    for file_path, ast in program.modules.items():
        try:
            file_path.relative_to(arch_root_resolved)
            in_arch = True
        except ValueError:
            in_arch = False
        if in_arch:
            continue
        # Transitively loaded (shared lib) — emit all equation decls.
        for _, decl_text in ast.exports.items():
            kind, _, _ = _decl_kind_and_body(decl_text)
            if kind == "equation":
                parts.append(decl_text)
        for _, decl_text in ast.private.items():
            kind, _, _ = _decl_kind_and_body(decl_text)
            if kind == "equation":
                parts.append(decl_text)

    # 1. Globals from arch.neuro — NT systems first
    _emit(arch_ast.private, ("neurotransmitter",))
    _emit(arch_ast.exports, ("neurotransmitter",))

    # 1b. Inline populations declared in arch.neuro itself (rare — most
    # archs use the modules/ import pattern — but tests + tiny standalone
    # archs may inline pops directly).
    _emit(arch_ast.private, ("population",))
    _emit(arch_ast.exports, ("population",))

    # 2. Populations + features in import order. Use the resolver's
    # PathResolver so `@brian/...` specifiers resolve correctly (a fresh
    # PathResolver would not have the repo_root cached).
    for imp in arch_ast.imports:
        try:
            target_file = resolver.path_resolver.resolve(
                imp.specifier, from_file=arch_path
            ).resolve()
        except (FileNotFoundError, ValueError):
            continue
        target_ast = program.modules.get(target_file)
        if target_ast is None:
            continue
        for imported_name in imp.names:
            if imported_name in target_ast.exports:
                parts.append(target_ast.exports[imported_name])

    # 2b. Inline feature blocks declared directly in arch.neuro.
    _emit(arch_ast.private, ("feature",))
    _emit(arch_ast.exports, ("feature",))

    # 3. Synapses + modulations from arch.neuro
    _emit(arch_ast.private, ("synapse", "modulation"))
    _emit(arch_ast.exports, ("synapse", "modulation"))

    # 4. Formal specs + sheaves
    _emit(arch_ast.private, ("formal_spec", "sheaf"))
    _emit(arch_ast.exports, ("formal_spec", "sheaf"))

    # 4b. §6.5 genetics: genes, proteins, metrics
    _emit(arch_ast.private, ("gene", "protein", "metric"))
    _emit(arch_ast.exports, ("gene", "protein", "metric"))

    # 5. Anything from non-arch files that wasn't already imported (private
    #    decls that don't appear in arch.neuro's import list). Rare in
    #    well-structured architectures but we don't want to silently drop.
    emitted: set = set()
    for imp in arch_ast.imports:
        try:
            tf = resolver.path_resolver.resolve(
                imp.specifier, from_file=arch_path
            ).resolve()
            emitted.add(tf)
        except (FileNotFoundError, ValueError):
            pass
    for file_path, ast in program.modules.items():
        if file_path == arch_path or file_path in emitted:
            continue
        for _, decl_text in ast.private.items():
            parts.append(decl_text)

    combined = "\n".join(parts)

    # THSD Phase 6+: Include full arch.neuro content for THSD parser context
    # The THSD parser needs the full file content to properly recognize THSD blocks
    # because it relies on surrounding declarations for validation
    architecture_metadata = None
    try:
        with open(arch_path, 'r', encoding='utf-8') as f:
            arch_content = f.read()
        # Add the full arch.neuro content so THSD parser can see THSD blocks
        # in their full context
        combined += "\n" + arch_content

        # Extract architecture metadata for ProgramIR
        if arch_ast and arch_ast.architecture:
            architecture_metadata = arch_ast.architecture
    except Exception:
        pass  # If reading fails, continue with partial combined DSL

    # Compile and attach architecture metadata.
    #
    # We route through ``compile_with_lib`` (not raw ``compile``) so
    # that ``module <name> = <Lib> { ... }`` instantiations declared
    # at the top level of arch.neuro get expanded with the shared
    # ``<repo>/lib/`` search path. Files without any module
    # instantiation pay zero overhead — ``compile_with_lib`` falls
    # through to ``compile`` immediately when no instances are found.
    #
    # Search order: arch-local ``lib/`` first (so an arch can shadow
    # a shared module with a local override), then the canonical
    # repo-root ``<repo>/lib/`` (2026-06-15 layout — lib moved out
    # of ``architectures/`` to live at the repo root alongside
    # ``modules/`` and ``blocks/``).
    lib_root_local = Path(arch_root).resolve() / "lib"
    repo_root = resolver.path_resolver.repo_root
    lib_root_repo = (repo_root / "lib") if repo_root is not None else None
    search_path = []
    if lib_root_local.exists():
        search_path.append(lib_root_local)
    if lib_root_repo is not None and lib_root_repo.exists() and lib_root_repo != lib_root_local:
        search_path.append(lib_root_repo)
    ir = NeuroMLCompiler.compile_with_lib(
        combined, lib_search_path=search_path or None
    )
    if architecture_metadata:
        ir.architecture = architecture_metadata

    return ir
