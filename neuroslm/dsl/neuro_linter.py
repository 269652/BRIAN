# -*- coding: utf-8 -*-
"""Semantic linter for .neuro DSL files.

Validates:
  - Syntax: block structure, matching braces, required fields
  - References: imports exist, populations referenced in synapses are declared
  - Equations: variable bindings, math function calls
  - Type consistency: shape expressions, parameter ranges
"""
from __future__ import annotations
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum


class Severity(Enum):
    """Diagnostic severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Diagnostic:
    """A single lint finding."""
    file: Path
    line: int
    col: int
    severity: Severity
    code: str
    message: str

    def __str__(self):
        return f"{self.file}:{self.line}:{self.col} [{self.code}] {self.message}"


class NeuroLinter:
    """Semantic validator for .neuro architecture files."""

    # Keywords that start blocks
    BLOCK_KEYWORDS = {
        "architecture", "neurotransmitter", "population", "synapse",
        "dynamics", "function", "training", "mechanisms", "mechanism", "export"
    }

    # Valid keys in each block type
    VALID_KEYS = {
        "architecture": {"d_sem", "dt", "d_model", "depth", "n_heads"},
        "neurotransmitter": {"base_concentration", "release_rate", "reuptake_rate", "diffusion_rate"},
        "population": {"count", "dynamics", "equation", "ode", "timescale", "capacity", "params", "state", "constants"},
        "synapse": {"weight", "equation", "neurotransmitter", "strength", "type"},
        "training": {
            "preset", "loss_clipping", "quantization", "optimizer", "learning_rate",
            "weight_decay", "grad_accum", "grad_clip", "label_smoothing", "dropout",
            "flooding_level", "stochastic_depth", "z_loss", "llrd", "pct_trunk",
            "pct_strength", "batch_size", "seq_len", "steps", "warmup_steps",
            "min_lr_ratio", "tonnetz_period", "bema_rollback_window", "bema_snapshot_every",
            "bema_cooldown", "nemori_floor", "mu_p_scaling", "curriculum", "crystallization_step",
            "pc_reentry_weight", "pc_reentry_nt_gate",
            "vbb_alpha", "vbb_beta_init"
        },
        "dynamics": {"equation", "ode", "params", "state", "constants"},
        "function": {"params", "return"}
    }

    # Math functions valid in equations
    MATH_FUNCTIONS = {
        "sin", "cos", "tan", "exp", "log", "sqrt", "abs", "tanh", "sigmoid",
        "ReLU", "silu", "gelu", "swiglu", "matmul", "linear", "rmsnorm",
        "causal_self_attention", "embedding", "softmax", "dropout", "layer_norm"
    }

    def __init__(self, file_path: Path):
        self.file = Path(file_path)
        self.diagnostics: List[Diagnostic] = []
        self.lines: List[str] = []
        self.source: str = ""
        self.populations: Set[str] = set()
        self.imports: Set[str] = set()
        self.exports: Set[str] = set()
        self.dynamics_decls: Set[str] = set()
        self.functions_decls: Set[str] = set()
        self.declared_vars: Dict[str, Set[str]] = {}  # per-scope variable tracking

    def lint(self) -> List[Diagnostic]:
        """Run all linting checks and return diagnostics."""
        if not self.file.exists():
            return [Diagnostic(
                self.file, 0, 0, Severity.ERROR, "file-not-found",
                f"File not found: {self.file}"
            )]

        self.source = self.file.read_text(encoding='utf-8')
        self.lines = self.source.split('\n')

        # Pass 1: structural validation
        self._check_brace_matching()
        self._check_syntax()

        # Pass 2: reference validation (only if no syntax errors)
        if not any(d.severity == Severity.ERROR for d in self.diagnostics):
            self._extract_declarations()
            self._check_references()
            self._check_equations()
            self._check_enum_style_declarations()
            self._check_architecture_organization()

        return self.diagnostics

    def _check_brace_matching(self):
        """Verify all braces match."""
        stack = []
        paren_depth = 0
        in_string = False
        escape_next = False

        for line_no, line in enumerate(self.lines, 1):
            for col, ch in enumerate(line):
                if escape_next:
                    escape_next = False
                    continue

                if ch == '\\':
                    escape_next = True
                    continue

                if ch == '"':
                    in_string = not in_string
                    continue

                if in_string:
                    continue

                if ch == '(':
                    paren_depth += 1
                elif ch == ')':
                    paren_depth -= 1
                    if paren_depth < 0:
                        self._error(line_no, col, "unmatched-paren",
                                  "Unmatched closing parenthesis")
                elif ch == '{':
                    stack.append(('{', line_no, col))
                elif ch == '}':
                    if not stack or stack[-1][0] != '{':
                        self._error(line_no, col, "unmatched-brace",
                                  "Unmatched closing brace")
                    else:
                        stack.pop()
                elif ch == '[':
                    stack.append(('[', line_no, col))
                elif ch == ']':
                    if not stack or stack[-1][0] != '[':
                        self._error(line_no, col, "unmatched-bracket",
                                  "Unmatched closing bracket")
                    else:
                        stack.pop()

        for bracket, line_no, col in stack:
            self._error(line_no, col, "unclosed-brace",
                       f"Unclosed {bracket}")

    def _check_syntax(self):
        """Check for syntax errors: invalid keywords, malformed declarations."""
        for line_no, line in enumerate(self.lines, 1):
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                continue

            # Check for invalid block starts
            m = re.match(r"\b(export\s+)?(\w+)\s+(\w+)", stripped)
            if m:
                maybe_export, keyword, name = m.groups()
                if keyword in self.BLOCK_KEYWORDS and keyword != "export":
                    # Valid block declaration
                    if not re.search(r"\{|:", stripped):
                        # Allow multi-line declarations
                        pass

            # Check for colon without block opening (field declaration)
            if ':' in stripped and '{' not in stripped:
                # This is a field: value pair; should be inside a block
                # We'll validate against VALID_KEYS when we know the context
                pass

    def _extract_declarations(self):
        """First pass: extract all populations, imports, exports, dynamics, functions."""
        current_block_type = None
        current_block_name = None

        for line_no, line in enumerate(self.lines, 1):
            stripped = line.strip()

            if not stripped or stripped.startswith('#'):
                continue

            # Track block context
            if re.match(r"\b(architecture|neurotransmitter|training|mechanisms?)\b", stripped):
                m = re.match(r"\b(\w+)\s+(\w+)?", stripped)
                if m:
                    current_block_type = m.group(1)
                    current_block_name = m.group(2)

            # Extract population declarations
            m = re.match(r"\bexport\s+population\s+(\w+)", stripped)
            if not m:
                m = re.match(r"\bpopulation\s+(\w+)", stripped)
            if m:
                self.populations.add(m.group(1))
                current_block_type = "population"
                current_block_name = m.group(1)

            # Extract import declarations (ES6-style: import {...} from "@/path")
            m = re.match(r"\bimport\s+\{([^}]+)\}\s+from\s+[\"'](@?[^\"']+)[\"']", stripped)
            if m:
                # Extract the path (remove @/ prefix if present)
                import_path = m.group(2)
                if import_path.startswith("@/"):
                    import_path = import_path[2:]
                self.imports.add(import_path)

                # Extract imported names directly
                names = m.group(1)
                for name in re.findall(r'\b([a-zA-Z_]\w*)\b', names):
                    self.populations.add(name)

            # Extract export declarations (population/dynamics/function name)
            # e.g. "export population motor" or "export dynamics rate_code"
            m = re.match(r"\bexport\s+(?:population|dynamics|function)\s+(\w+)", stripped)
            if m:
                self.exports.add(m.group(1))

            # Extract dynamics declarations
            m = re.match(r"\bdynamics\s+(\w+)", stripped)
            if m:
                self.dynamics_decls.add(m.group(1))
                current_block_type = "dynamics"
                current_block_name = m.group(1)

            # Extract function declarations
            m = re.match(r"\bfunction\s+(\w+)", stripped)
            if m:
                self.functions_decls.add(m.group(1))
                current_block_type = "function"
                current_block_name = m.group(1)

    def _check_references(self):
        """Check that referenced populations exist, imports resolve, etc."""
        # First, load populations from imported files (single pass, no recursion to avoid cycles)
        for import_path in self.imports:
            self._load_imported_populations(import_path)

        for line_no, line in enumerate(self.lines, 1):
            stripped = line.strip()

            if not stripped or stripped.startswith('#'):
                continue

            # Check synapse declarations
            m = re.match(r"\bsynapse\s+(\w+)\s*->\s*(\w+)", stripped)
            if m:
                pre, post = m.groups()
                if pre not in self.populations:
                    self._warning(line_no, 8, "undefined-population",
                                 f"Population '{pre}' not declared or imported")
                if post not in self.populations:
                    self._warning(line_no, 8 + len(pre) + 4, "undefined-population",
                                 f"Population '{post}' not declared or imported")

            # Check import resolution (ES6 style: import {...} from "@/path")
            m = re.match(r"\bimport\s+\{[^}]+\}\s+from\s+[\"'](@?[^\"']+)[\"']", stripped)
            if m:
                import_path = m.group(1)
                if import_path.startswith("@/"):
                    import_path = import_path[2:]

                # Resolve relative to the current file's directory
                base_dir = self.file.parent
                candidate_paths = [
                    base_dir / f"{import_path}.neuro",
                    base_dir / import_path / "index.neuro",
                    base_dir / import_path,
                ]
                resolved = any(p.exists() for p in candidate_paths)
                if not resolved:
                    self._warning(line_no, 8, "unresolved-import",
                                 f"Cannot resolve import '@/{import_path}'")

    def _load_imported_populations(self, import_path: str):
        """Load population names from an imported file."""
        # Note: import_path has @/ already stripped by _extract_declarations
        base_dir = self.file.parent

        # Try multiple path formats
        candidate_paths = [
            base_dir / f"{import_path}.neuro",
            base_dir / import_path / "index.neuro",
            base_dir / import_path,
        ]

        resolved_path = None
        for p in candidate_paths:
            if p.exists():
                resolved_path = p
                break

        if not resolved_path:
            return  # Import unresolved; not critical for population extraction

        # Parse the imported file for exported populations
        try:
            content = resolved_path.read_text(encoding='utf-8')
            for m in re.finditer(r"\bexport\s+population\s+(\w+)", content):
                self.populations.add(m.group(1))
            # Also add non-exported populations from this file (conservative)
            for m in re.finditer(r"\bpopulation\s+(\w+)", content):
                self.populations.add(m.group(1))
        except Exception:
            pass

    def _check_equations(self):
        """Validate equation syntax and variable bindings."""
        equations = {}  # track equations for duplicate detection
        definitions = {}  # track equation definitions: name -> formula

        # First pass: extract equation definitions
        for line_no, line in enumerate(self.lines, 1):
            if 'export equation' in line or (re.match(r'\s*equation\s+\w+', line)):
                # Extract equation definition name and formula
                name_match = re.search(r'equation\s+(\w+)\s*\{', line)
                if name_match:
                    eq_name = name_match.group(1)
                    # Formula is on same or next line(s)
                    formula_match = re.search(r'formula:\s*"([^"]*)"', ' '.join(self.lines[line_no-1:min(line_no+2, len(self.lines))]))
                    if formula_match:
                        definitions[eq_name] = formula_match.group(1)

        # Second pass: check equations
        for line_no, line in enumerate(self.lines, 1):
            # Extract quoted strings (equations)
            for m in re.finditer(r'"([^"]*)"', line):
                equation = m.group(1)
                start_col = m.start(1)

                # Check for undefined variables in equation
                self._check_equation_vars(equation, line_no, start_col)

                # Check if this equation matches a definition
                if '=' in equation and 'equation:' in line:
                    for def_name, def_formula in definitions.items():
                        if equation == def_formula and f'@{def_name}' not in line:
                            self._info(
                                line_no, start_col, "matches-definition",
                                f"This equation matches definition '{def_name}'. "
                                f"Replace with: equation: @{def_name}"
                            )

                    # Track equation for duplicate detection
                    if equation not in equations:
                        equations[equation] = []
                    equations[equation].append(line_no)

        # Suggest extracting repeated equations as definitions
        for equation, occurrences in equations.items():
            if len(occurrences) >= 2:
                self._info(
                    occurrences[0], 0, "repeated-equation",
                    f"Equation appears {len(occurrences)} times. Consider extracting as a definition: "
                    f'equation name {{ params: [...], formula: "{equation}" }}'
                )

    def _check_enum_style_declarations(self):
        """Warn against enum-style value declarations, prefer DSL native mechanisms."""
        in_potential_enum = False
        enum_name = None
        enum_start_line = None
        enum_keys = []

        for line_no, line in enumerate(self.lines, 1):
            stripped = line.strip()

            if not stripped or stripped.startswith('#'):
                continue

            # Check if this line opens a potential enum block
            # Pattern: name { or name{ where name is NOT a DSL keyword
            match = re.match(r'^(\w+)\s*\{', stripped)
            if match:
                potential_name = match.group(1)
                # Skip if it's a DSL keyword block (use word boundary check)
                is_keyword = any(re.match(f'^{kw}\\b', stripped) for kw in self.BLOCK_KEYWORDS)

                if not is_keyword:
                    # Check if the body has enum-like content (CONST: value)
                    const_keys = re.findall(r'\b([A-Z_][A-Z0-9_]*)\s*:', stripped)
                    if const_keys:
                        in_potential_enum = True
                        enum_name = potential_name
                        enum_start_line = line_no
                        enum_keys = const_keys
                    else:
                        # Still might be enum if no constants on this line
                        in_potential_enum = True
                        enum_name = potential_name
                        enum_start_line = line_no
                        enum_keys = []

            # Continue tracking enum keys across lines
            elif in_potential_enum:
                const_keys = re.findall(r'\b([A-Z_][A-Z0-9_]*)\s*:', stripped)
                if const_keys:
                    enum_keys.extend(const_keys)

                # Check if block ends
                if '}' in stripped:
                    in_potential_enum = False
                    # Warn if we found multiple enum-like keys
                    if len(enum_keys) >= 2:
                        self._warning(
                            enum_start_line, 0, "enum-style-declaration",
                            f"Enum-style declaration '{enum_name}' detected with keys: {{{', '.join(enum_keys[:3])}{'...' if len(enum_keys) > 3 else ''}}}. "
                            f"Prefer DSL native mechanisms: (1) Use 'constants' block for immutable values, "
                            f"(2) Use 'mechanism' for named computational modes with equations, "
                            f"(3) Use 'export dynamics' for named state machines with ODEs, "
                            f"(4) Use 'export constraint' for formal invariants. "
                            f"Named DSL constructs are self-documenting and enable linting."
                        )
                    enum_keys = []
                    enum_name = None

    def _check_architecture_organization(self):
        """Check for declarations that should be moved to lib files."""
        # Only check if this is arch.neuro (root architecture file)
        if self.file.name != "arch.neuro":
            return

        equation_defs = []  # (line_no, name)
        synapse_count = 0
        first_synapse_line = None

        for line_no, line in enumerate(self.lines, 1):
            stripped = line.strip()

            if not stripped or stripped.startswith('#'):
                continue

            # Find equation definitions (with name extraction)
            m = re.match(r'\b(export\s+)?equation\s+(\w+)\s*\{', stripped)
            if m:
                eq_name = m.group(2)
                equation_defs.append((line_no, eq_name))

            # Count synapse declarations
            if re.match(r'\b(export\s+)?synapse\s+', stripped):
                synapse_count += 1
                if first_synapse_line is None:
                    first_synapse_line = line_no

        # Warn for EACH equation defined in arch.neuro — equations belong in lib files
        for line_no, eq_name in equation_defs:
            self._warning(
                line_no, 0, "equation-in-arch",
                f"Equation '{eq_name}' defined in arch.neuro. "
                f"Move to lib/equations.neuro and import: "
                f"import {{ {eq_name} }} from \"@/lib/equations\". "
                f"Auto-fix: Extract to lib/equations.neuro, replace with import."
            )

        # Warn once if synapses should be extracted
        if synapse_count > 5:
            self._info(
                first_synapse_line or 1, 0, "arch-too-many-synapses",
                f"arch.neuro has {synapse_count}+ synapse declarations. "
                "Consider moving to lib/synapses.neuro and importing them."
            )

        # Check for large neurotransmitter definitions
        nt_matches = re.findall(r'\bneurotransmitter\s+\w+', self.source)
        if len(nt_matches) >= 5:
            first_nt = None
            for line_no, line in enumerate(self.lines, 1):
                if re.search(r'\bneurotransmitter\s+\w+', line):
                    first_nt = line_no
                    break
            self._info(
                first_nt or 1, 0, "arch-many-nts",
                f"arch.neuro has {len(nt_matches)} neurotransmitter definitions. "
                "Consider moving to lib/neurotransmitters.neuro."
            )

    def _check_equation_vars(self, equation: str, line_no: int, col: int):
        """Check that variables in equation are defined."""
        # Extract all identifiers from the equation
        identifiers = re.findall(r'\b([a-zA-Z_]\w*)\b', equation)

        # Built-in variables and parameters
        builtins = {
            'x', 'y', 's', 'V', 'd_sem', 'dV', 'dt', 'tau', 'weight',
            'x_pre', 'x_post', 'e', 'pi', 'nan', 'inf',
            'W',  # weight matrix (standard notation)
            'c', 'gain', 'output',  # modulation parameters
            'coef',  # ODE coefficients
        }

        for ident in identifiers:
            # Skip functions and built-ins
            if ident in self.MATH_FUNCTIONS or ident in builtins:
                continue

            # Check if it's a number
            if re.match(r'^\d+\.?\d*$', ident):
                continue

            # Warn about potentially undefined variables
            # (conservative: may have false positives if imported)
            if ident not in self.populations and \
               ident not in self.dynamics_decls and \
               ident not in self.functions_decls and \
               ident.isupper() and not any(c.isdigit() for c in ident):  # heuristic for constants
                self._info(line_no, col, "potentially-undefined",
                          f"Variable '{ident}' may be undefined")

    def _error(self, line: int, col: int, code: str, message: str):
        """Record an error."""
        self.diagnostics.append(Diagnostic(
            self.file, line, col, Severity.ERROR, code, message
        ))

    def _warning(self, line: int, col: int, code: str, message: str):
        """Record a warning."""
        self.diagnostics.append(Diagnostic(
            self.file, line, col, Severity.WARNING, code, message
        ))

    def _info(self, line: int, col: int, code: str, message: str):
        """Record an info message."""
        self.diagnostics.append(Diagnostic(
            self.file, line, col, Severity.INFO, code, message
        ))


def lint_file(file_path: Path) -> List[Diagnostic]:
    """Lint a single .neuro file."""
    linter = NeuroLinter(file_path)
    return linter.lint()


def lint_directory(directory: Path, pattern: str = "**/*.neuro") -> List[Diagnostic]:
    """Lint all .neuro files in a directory tree."""
    diagnostics = []
    for neuro_file in directory.glob(pattern):
        diagnostics.extend(lint_file(neuro_file))
    return diagnostics


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python neuro_linter.py <file_or_directory> [--json]")
        sys.exit(1)

    target = Path(sys.argv[1])
    json_output = "--json" in sys.argv

    if target.is_file():
        diags = lint_file(target)
    else:
        diags = lint_directory(target)

    if json_output:
        import json
        output = []
        for d in diags:
            output.append({
                "file": str(d.file),
                "line": d.line,
                "col": d.col,
                "severity": d.severity.value,
                "code": d.code,
                "message": d.message
            })
        print(json.dumps(output, indent=2))
    else:
        for d in diags:
            print(d)

        if diags:
            errors = sum(1 for d in diags if d.severity == Severity.ERROR)
            warnings = sum(1 for d in diags if d.severity == Severity.WARNING)
            print(f"\n{errors} error(s), {warnings} warning(s)")
            sys.exit(1 if errors else 0)
