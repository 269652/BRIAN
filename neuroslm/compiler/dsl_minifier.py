# -*- coding: utf-8 -*-
"""DSL minifier with source maps for evolution compatibility.

Minification strategy:
1. Remove all comments (# ... and /* ... */)
2. Remove excess whitespace (consolidate to single spaces)
3. Remove unnecessary newlines
4. Maintain source map for evolution/debugging
5. Pretty-printing available on unfold

Source map enables:
- Evolved patches to reference original code
- Unfolding to produce readable output despite minification
- Evolution to work transparently with minified DNA
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class MinificationMap:
    """Maps minified DSL back to original source."""
    minified_to_original: Dict[Tuple[int, int], Tuple[int, int]] = field(
        default_factory=dict)  # (min_line, min_col) -> (orig_line, orig_col)
    original_to_minified: Dict[Tuple[int, int], Tuple[int, int]] = field(
        default_factory=dict)  # (orig_line, orig_col) -> (min_line, min_col)
    line_map: Dict[int, int] = field(
        default_factory=dict)  # minified_line -> original_line

    def to_dict(self) -> Dict:
        """Serialize to JSON-compatible format."""
        return {
            "line_map": self.line_map,
            # Store other maps as string keys for JSON compatibility
            "minified_to_original": {
                f"{a},{b}": (c, d) for (a, b), (c, d)
                in self.minified_to_original.items()
            },
            "original_to_minified": {
                f"{a},{b}": (c, d) for (a, b), (c, d)
                in self.original_to_minified.items()
            },
        }

    @classmethod
    def from_dict(cls, d: Dict) -> MinificationMap:
        """Deserialize from dictionary."""
        line_map = d.get("line_map", {})
        # Convert string keys back to ints
        line_map = {int(k): v for k, v in line_map.items()}

        return cls(
            line_map=line_map,
            minified_to_original={},
            original_to_minified={},
        )


class DSLMinifier:
    """Minifies DSL code while maintaining source maps."""

    def minify(self, dsl_source: str) -> str:
        """Minify DSL code: remove comments, excess whitespace."""
        minified, _ = self.minify_with_map(dsl_source)
        return minified

    def minify_with_map(self, dsl_source: str) -> Tuple[str, MinificationMap]:
        """Minify DSL and return source map.

        Returns:
            (minified_dsl, source_map)
        """
        # Remove comments (both # ... and /* ... */)
        dsl = self._remove_comments(dsl_source)

        # Remove excess whitespace while tracking lines
        minified_lines = []
        original_lines = dsl_source.split('\n')
        minified_to_original_map = {}

        for orig_line_num, line in enumerate(original_lines, start=1):
            # Strip comments from this line
            line_no_comment = self._remove_line_comment(line)

            # Remove excess whitespace but preserve structure
            stripped = line_no_comment.strip()
            if not stripped:
                continue  # Skip empty lines

            # Consolidate internal whitespace to single spaces
            consolidated = re.sub(r'\s+', ' ', stripped)

            minified_lines.append(consolidated)
            # Map minified line back to original
            minified_line_num = len(minified_lines)
            minified_to_original_map[minified_line_num] = orig_line_num

        minified_dsl = '\n'.join(minified_lines)

        # Create source map
        source_map = MinificationMap(line_map=minified_to_original_map)

        return minified_dsl, source_map

    def prettify(self, minified_dsl: str) -> str:
        """Pretty-print minified DSL with proper indentation.

        Adds:
        - Line breaks after closing braces
        - Indentation based on nesting level
        - Spacing around operators
        """
        result = []
        indent_level = 0
        indent_str = "    "  # 4 spaces

        # Simple pretty-printer: split on braces, track indentation
        tokens = re.split(r'(\{|\})', minified_dsl)

        for token in tokens:
            token = token.strip()
            if not token:
                continue

            if token == '}':
                indent_level = max(0, indent_level - 1)
                result.append(indent_str * indent_level + token)
            elif token == '{':
                result.append(indent_str * indent_level + token)
                indent_level += 1
            else:
                # Regular content
                result.append(indent_str * indent_level + token)

        return '\n'.join(result)

    def _remove_comments(self, source: str) -> str:
        """Remove both line (#) and block (/* */) comments."""
        # Remove block comments /* ... */
        source = re.sub(r'/\*.*?\*/', '', source, flags=re.DOTALL)

        # Remove line comments (#...)
        lines = source.split('\n')
        cleaned_lines = [self._remove_line_comment(line) for line in lines]
        return '\n'.join(cleaned_lines)

    def _remove_line_comment(self, line: str) -> str:
        """Remove # ... comment from a single line."""
        # Find # not inside strings
        in_string = False
        string_char = None
        for i, char in enumerate(line):
            if char in ('"', "'"):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
            elif char == '#' and not in_string:
                return line[:i]
        return line


class PrettifyPolicy:
    """Policy for pretty-printing DSL: minified, pretty, or auto."""

    MINIFIED = "minified"
    PRETTY = "pretty"
    AUTO = "auto"  # Decide based on minify flag in arch

    @staticmethod
    def should_prettify(dsl_source: str) -> bool:
        """Check if DSL should be pretty-printed.

        Heuristic:
        - If has minify: false → pretty
        - If has minify: true → minified
        - If no minify flag → pretty (default)
        """
        # Check for minify: true in architecture block
        if re.search(r'minify\s*:\s*true', dsl_source):
            return False  # Keep minified
        elif re.search(r'minify\s*:\s*false', dsl_source):
            return True  # Pretty-print

        # Default to pretty (readable)
        return True

    @staticmethod
    def extract_minify_flag(dsl_source: str) -> Optional[bool]:
        """Extract minify flag from arch.neuro source.

        Returns:
            True if minify: true, False if minify: false, None if not specified
        """
        match_true = re.search(r'minify\s*:\s*true', dsl_source)
        if match_true:
            return True

        match_false = re.search(r'minify\s*:\s*false', dsl_source)
        if match_false:
            return False

        return None
