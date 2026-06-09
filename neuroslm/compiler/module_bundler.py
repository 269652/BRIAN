# -*- coding: utf-8 -*-
"""Module bundler for DNA compilation: resolves and bundles all imports.

During DNA compilation, all imported modules must be bundled so that:
1. The DNA is self-contained (no dangling import references)
2. Module origins are tracked for evolution
3. Unfolding produces modularized code with imports preserved or inlined

This module provides:
- ModuleBundler: collects all imports and builds a module map
- BundledDSL: represents DSL with all modules resolved and metadata
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class Module:
    """A single .neuro module file with its source and origin."""
    path: Path
    source: str
    specifier: Optional[str] = None  # How it was imported (e.g., "@/lib/cortex")

    def to_dict(self) -> Dict:
        return {
            "path": str(self.path),
            "source": self.source,
            "specifier": self.specifier,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> Module:
        return cls(
            path=Path(d["path"]),
            source=d["source"],
            specifier=d.get("specifier"),
        )


@dataclass
class SourceMap:
    """Maps code regions back to their source modules.

    For evolved improvements, tracks which changes came from which
    modules/libraries. Enables modular evolution and attribution.
    """
    # Map from line range (start, end) to module specifier
    line_to_module: Dict[tuple, str] = field(default_factory=dict)
    # Reverse: module specifier to line ranges
    module_to_lines: Dict[str, List[tuple]] = field(default_factory=dict)
    # Named offsets for key sections (e.g., "main", "lib/cortex")
    section_offsets: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize to JSON-compatible format."""
        return {
            "line_to_module": {str(k): v for k, v in self.line_to_module.items()},
            "module_to_lines": {k: [str(l) for l in v] for k, v in self.module_to_lines.items()},
            "section_offsets": self.section_offsets,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> SourceMap:
        """Deserialize from dictionary."""
        # Parse line ranges back from string keys
        line_to_module = {}
        for k, v in d.get("line_to_module", {}).items():
            try:
                start, end = map(int, k.strip("()").split(","))
                line_to_module[(start, end)] = v
            except (ValueError, AttributeError):
                pass

        module_to_lines = {}
        for k, v in d.get("module_to_lines", {}).items():
            try:
                lines = [tuple(map(int, l.strip("()").split(","))) for l in v]
                module_to_lines[k] = lines
            except (ValueError, AttributeError):
                module_to_lines[k] = []

        return cls(
            line_to_module=line_to_module,
            module_to_lines=module_to_lines,
            section_offsets=d.get("section_offsets", {}),
        )


@dataclass
class BundledDSL:
    """DSL bundle with all modules collected and metadata.

    Represents a fully resolved DSL where all imports have been
    collected and tracked for evolution.
    """
    main_source: str
    modules: Dict[str, Module] = field(default_factory=dict)
    import_graph: Dict[str, List[str]] = field(default_factory=dict)
    source_map: Optional[SourceMap] = None

    def to_dict(self) -> Dict:
        """Serialize to dictionary for storage in DNA."""
        return {
            "main_source": self.main_source,
            "modules": {k: v.to_dict() for k, v in self.modules.items()},
            "import_graph": self.import_graph,
            "source_map": self.source_map.to_dict() if self.source_map else None,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> BundledDSL:
        """Deserialize from dictionary."""
        modules = {
            k: Module.from_dict(v) for k, v in d.get("modules", {}).items()
        }
        source_map_dict = d.get("source_map")
        source_map = (
            SourceMap.from_dict(source_map_dict) if source_map_dict else None
        )
        return cls(
            main_source=d["main_source"],
            modules=modules,
            import_graph=d.get("import_graph", {}),
            source_map=source_map,
        )

    def inline_imports(self) -> str:
        """Combine all modules into a single DSL string (inline style).

        Returns the main source with all imports replaced by their
        resolved content, and module boundaries marked.
        """
        result = []
        seen = set()

        def process_module(source: str, module_key: str) -> str:
            if module_key in seen:
                return source
            seen.add(module_key)

            # Remove import statements and replace with module content
            lines = source.split('\n')
            processed = []

            for line in lines:
                if re.match(r'^\s*import\s+', line):
                    # Extract import path
                    match = re.search(r'import\s+"([^"]+)"', line)
                    if match:
                        import_spec = match.group(1)
                        # Find the module that matches this import
                        for spec, mod in self.modules.items():
                            if import_spec in spec or spec == import_spec:
                                # Inline the module with boundary markers
                                processed.append(f'# ━━━ Module: {spec} ━━━')
                                processed.append(mod.source)
                                processed.append(f'# ━━━ End: {spec} ━━━')
                                break
                else:
                    processed.append(line)

            return '\n'.join(processed)

        result_str = process_module(self.main_source, "main")
        return result_str

    def preserve_imports(self) -> str:
        """Reconstruct DSL with import statements preserved.

        Returns DSL that references the modules, ready for reconstruction
        in a multi-file layout.
        """
        # Just return main source as-is; imports remain valid
        return self.main_source


class ModuleBundler:
    """Bundles all imports from a DSL file into a self-contained collection.

    Given a main .neuro file, recursively resolves all `import` statements
    and collects the imported modules. Tracks the import graph and module
    origins for evolution.
    """

    def __init__(self, arch_root: Path):
        self.arch_root = Path(arch_root).resolve()
        self.modules: Dict[str, Module] = {}
        self.import_graph: Dict[str, List[str]] = {}

    def resolve_import(self, specifier: str, from_file: Optional[Path]) -> Optional[Path]:
        """Resolve an import specifier to an absolute file path.

        Handles @/, ./, and ../ style paths.

        Returns:
            Absolute path to the file, or None if resolution fails.
        """
        try:
            if specifier.startswith("@/"):
                base = self.arch_root
                rest = specifier[2:]
            elif specifier.startswith("./") or specifier.startswith("../"):
                if from_file is None:
                    return None
                base = Path(from_file).resolve().parent
                rest = specifier
            else:
                # Bare specifier like "lib/cortex" — try as relative
                if from_file:
                    base = Path(from_file).resolve().parent
                    rest = rest if "/" in specifier else f"./{specifier}"
                else:
                    return None

            candidate = (base / rest).resolve()

            # Check bounds: must not escape arch_root
            try:
                candidate.relative_to(self.arch_root)
            except ValueError:
                return None

            # Try exact, with .neuro suffix, or index.neuro
            if candidate.is_file() and candidate.suffix == ".neuro":
                return candidate

            with_suffix = (
                candidate.with_suffix(".neuro")
                if candidate.suffix == ""
                else candidate
            )
            if with_suffix.is_file():
                return with_suffix

            index = candidate / "index.neuro"
            if index.is_file():
                return index

            return None

        except Exception:
            return None

    def bundle(self, main_file: Path) -> BundledDSL:
        """Bundle a main DSL file and all its imports.

        Args:
            main_file: Path to arch.neuro or main DSL file.

        Returns:
            BundledDSL with main source and all modules collected.
        """
        main_file = Path(main_file).resolve()

        if not main_file.exists():
            raise FileNotFoundError(f"Main file not found: {main_file}")

        main_source = main_file.read_text(encoding="utf-8")

        # Recursively collect imports
        self._collect_imports(main_source, main_file)

        # Generate source map
        source_map = self._generate_source_map(main_source)

        return BundledDSL(
            main_source=main_source,
            modules=dict(self.modules),
            import_graph=dict(self.import_graph),
            source_map=source_map,
        )

    def _collect_imports(
        self, source: str, from_file: Path, visited: Optional[Set[str]] = None
    ) -> None:
        """Recursively collect all imports from a source file.

        Supports both traditional DSL and ES6-style imports:
        - Traditional: import "@/lib/cortex"
        - ES6: import { x, y } from "@/lib/cortex"

        Args:
            source: DSL source code to parse.
            from_file: File being parsed (for relative path resolution).
            visited: Set of already-visited file paths (cycle detection).
        """
        if visited is None:
            visited = set()

        from_file_str = str(from_file.resolve())
        if from_file_str in visited:
            return
        visited.add(from_file_str)

        # Find all import statements (both traditional and ES6)
        # Traditional: import "@/path"
        # ES6: import { ... } from "@/path"
        traditional_pattern = r'import\s+"([^"]+)"'
        es6_pattern = r'from\s+"([^"]+)"'

        traditional_imports = re.findall(traditional_pattern, source)
        es6_imports = re.findall(es6_pattern, source)
        imports = traditional_imports + es6_imports

        # Remove duplicates while preserving order
        seen = set()
        unique_imports = []
        for imp in imports:
            if imp not in seen:
                seen.add(imp)
                unique_imports.append(imp)

        for spec in unique_imports:
            resolved = self.resolve_import(spec, from_file)
            if resolved is None:
                continue

            resolved_str = str(resolved)
            if resolved_str in visited:
                continue

            # Load the module
            try:
                module_source = resolved.read_text(encoding="utf-8")
            except Exception:
                continue

            # Store it
            self.modules[spec] = Module(
                path=resolved, source=module_source, specifier=spec
            )

            # Track import graph
            if from_file_str not in self.import_graph:
                self.import_graph[from_file_str] = []
            self.import_graph[from_file_str].append(spec)

            # Recurse
            self._collect_imports(module_source, resolved, visited)

    def _generate_source_map(self, main_source: str) -> SourceMap:
        """Generate a source map tracking which lines come from which modules.

        Args:
            main_source: The main DSL source code.

        Returns:
            SourceMap with line-to-module and module-to-lines mappings.
        """
        source_map = SourceMap()
        lines = main_source.split('\n')

        current_line = 0
        traditional_pattern = r'import\s+"([^"]+)"'
        es6_pattern = r'from\s+"([^"]+)"'

        for line_idx, line in enumerate(lines, start=1):
            # Check for traditional imports
            match = re.search(traditional_pattern, line)
            if match:
                spec = match.group(1)
                source_map.line_to_module[(line_idx, line_idx)] = spec
                if spec not in source_map.module_to_lines:
                    source_map.module_to_lines[spec] = []
                source_map.module_to_lines[spec].append((line_idx, line_idx))
                continue

            # Check for ES6 imports
            match = re.search(es6_pattern, line)
            if match:
                spec = match.group(1)
                source_map.line_to_module[(line_idx, line_idx)] = spec

                if spec not in source_map.module_to_lines:
                    source_map.module_to_lines[spec] = []
                source_map.module_to_lines[spec].append((line_idx, line_idx))

        # Track sections
        source_map.section_offsets["main"] = 0
        for spec in self.modules.keys():
            # Offset is approximate (could be enhanced with actual tracking)
            source_map.section_offsets[spec] = hash(spec) % 10000

        return source_map
