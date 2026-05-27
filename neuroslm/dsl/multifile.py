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
from typing import Dict, List, Optional


class PathResolver:
    """Resolve `.neuro` import specifiers to absolute file paths.

    The resolver normalizes mjs-style specifiers (`@/`, `./`, `../`) and
    falls back to `index.neuro` when a specifier names a directory.

    Args:
        arch_root: filesystem path to the architecture root. The `@/`
            prefix in specifiers is anchored here.
    """

    def __init__(self, arch_root: Path):
        self.arch_root = Path(arch_root).resolve()

    # ── Public ──────────────────────────────────────────────────────

    def resolve(self, specifier: str, from_file: Optional[Path]) -> Path:
        """Resolve `specifier` (as used in an `import "<specifier>"`).

        Args:
            specifier: the import target, e.g. "@/lib/dynamics", "./layers"
            from_file: the file containing the import (required for
                relative paths; ignored for absolute `@/...` paths).

        Returns:
            Absolute resolved path of the target `.neuro` file.

        Raises:
            ValueError: malformed specifier, relative-without-from_file,
                       or path that escapes the architecture root.
            FileNotFoundError: resolved path doesn't exist (neither as a
                               `.neuro` file nor a folder-module).
        """
        if specifier.startswith("@/"):
            base = self.arch_root
            rest = specifier[2:]
        elif specifier.startswith("./") or specifier.startswith("../"):
            if from_file is None:
                raise ValueError(
                    f"relative specifier {specifier!r} needs from_file"
                )
            base = Path(from_file).resolve().parent
            rest = specifier
        else:
            raise ValueError(
                f"unsupported specifier {specifier!r}: must start with "
                "@/ (absolute), ./, or ../ (relative)"
            )

        candidate = (base / rest).resolve()

        # Reject specifiers that escape the architecture root. This guards
        # against `import "../../etc/passwd"`-style mistakes (or worse).
        try:
            candidate.relative_to(self.arch_root)
        except ValueError:
            raise ValueError(
                f"specifier {specifier!r} escapes architecture root "
                f"{self.arch_root}"
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
    "formal_spec",
    "sheaf",
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
    """Slice `<keyword> <name> { ... }`; return (name, body_inside, end_pos)."""
    # Advance past keyword
    pos = start + len(keyword)
    while pos < len(body) and body[pos].isspace():
        pos += 1
    # Name
    name_start = pos
    while pos < len(body) and (body[pos].isalnum() or body[pos] == "_"):
        pos += 1
    name = body[name_start:pos]
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
    """Given `body[start] == '{'`, return (inside_text, position_after_close)."""
    assert body[start] == "{"
    depth = 1
    i = start + 1
    in_str = None
    while i < len(body) and depth > 0:
        ch = body[i]
        if in_str:
            if ch == in_str:
                in_str = None
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
    """Store the decl under exports or private, rejecting duplicates."""
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
class ResolvedProgram:
    """A fully-linked multi-file architecture program.

    Attributes:
        arch_root: absolute path to the architecture root
        modules:   {absolute_file_path: ModuleAST}
        import_map: {file_path: {local_alias: (target_file, source_name)}}
                    — what each name refers to within each file's scope
        architecture: the `architecture { name, properties }` block from
                      arch.neuro, or {} if none
    """
    arch_root: Path
    modules: Dict[Path, ModuleAST] = field(default_factory=dict)
    import_map: Dict[Path, Dict[str, tuple]] = field(default_factory=dict)
    architecture: Dict = field(default_factory=dict)

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

        return program
