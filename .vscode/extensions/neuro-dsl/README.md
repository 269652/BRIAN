# NeuroSLM DSL Extension

Syntax highlighting, semantic linting, and intellisense for `.neuro` architecture files.

## Features

### 1. Syntax Highlighting
- **Keywords**: `architecture`, `neurotransmitter`, `population`, `synapse`, `import`, `export`, `dynamics`, `function`, `training`, `mechanism`
- **Type colors**: Parameters, math functions, strings, numbers
- **Block structure**: Automatic bracket matching and indentation
- **Equations**: Math operators and functions highlighted within equation strings

### 2. Go-to-Definition for References
- **Ctrl+Click** (or **F12**, **Go to Definition**) on `@reference` names to jump to their definition
- Supports `@equation`, `@population`, `@dynamics`, `@function` references
- Works across imported files via `import { ... } from "@/path"`
- Navigates to the exact definition line

### 3. Semantic Linter
The linter validates `.neuro` files for semantic correctness:

#### Structural Checks
- ✓ Matching braces, brackets, parentheses
- ✓ Syntax validation
- ✓ Required field presence

#### Reference Checks
- ✓ Population references in synapses
- ✓ Import resolution (ES6-style: `import { name } from "@/path"`)
- ✓ Cross-file population discovery
- ✓ Exported vs. private declaration tracking

#### Semantic Checks
- ✓ Equation variable binding (detects potentially undefined variables)
- ✓ Math function recognition (sin, cos, exp, ReLU, etc.)
- ✓ Built-in variable tracking (x, y, s, V, etc.)
- ✓ Enum-style declaration detection (warns against manual enum-like blocks, suggests DSL native mechanisms)

## Installation

1. Copy the extension directory to your VSCode extensions folder:
   ```bash
   cp -r .vscode/extensions/neuro-dsl ~/.vscode/extensions/
   ```

2. Reload VSCode

3. Open any `.neuro` file to see syntax highlighting

## Usage

### Command Line Linting
Run the linter on `.neuro` files from the terminal:

```bash
# Lint a single file
python neuroslm/dsl/neuro_linter.py architectures/rcc_bowtie/arch.neuro

# Lint a directory
python neuroslm/dsl/neuro_linter.py architectures/

# Output as JSON
python neuroslm/dsl/neuro_linter.py architectures/ --json
```

### VSCode Integration
- Open any `.neuro` file
- Syntax highlighting is automatic
- Linting errors/warnings appear in the editor's Problems panel

## Diagnostics

### Error Types

| Code | Severity | Meaning |
|------|----------|---------|
| `unmatched-paren` | Error | Unmatched `(` or `)` |
| `unmatched-brace` | Error | Unmatched `{` or `}` |
| `unmatched-bracket` | Error | Unmatched `[` or `]` |
| `unclosed-brace` | Error | Opening brace with no closing match |
| `undefined-population` | Warning | Population referenced but not declared |
| `unresolved-import` | Warning | Import path cannot be resolved |
| `enum-style-declaration` | Warning | Enum-style constant block detected; use DSL native mechanisms instead |
| `potentially-undefined` | Info | Variable may be undefined in equation |

## Example: Valid `.neuro` File

```neuro
architecture rcc_bowtie {
    d_sem: 256,
    dt: 0.01
}

import { sensory } from "@/modules/sensory"
import { motor } from "@/modules/motor"

export population processing {
    count: 512,
    dynamics: "rate_code",
    equation: "y = ReLU(x)"
}

synapse sensory -> processing {
    weight: 0.7,
    neurotransmitter: "glutamate"
}

synapse processing -> motor {
    weight: 0.5,
    equation: "y = weight * (x_pre @ W)"
}
```

## Limitations

1. **Single-pass analysis**: Does not perform deep cross-file semantic checking beyond import paths
2. **Runtime variables**: Cannot validate variables bound at runtime (e.g., matrix dimensions in equations)
3. **No recursive import checking**: Does not follow transitive imports for population discovery
4. **Conservative warnings**: Some warnings may be false positives for complex multi-file architectures

## Architecture

```
.vscode/extensions/neuro-dsl/
├── package.json                  # Extension manifest
├── extension.js                  # Extension entry point
├── language-configuration.json   # Bracket/indentation rules
└── syntaxes/
    └── neuro.tmLanguage.json     # TextMate grammar (syntax highlighting)

neuroslm/dsl/
├── neuro_linter.py              # Python linter implementation
└── tests/
    └── test_neuro_linter.py     # Comprehensive linter tests
```

## Testing

Run the linter test suite:

```bash
python -m pytest tests/dsl/test_neuro_linter.py -v
```

Tests cover:
- Brace/bracket matching
- Declaration extraction (population, import, export, dynamics, function)
- Reference validation (synapse, import resolution)
- Equation variable checking
- Comment handling
- Integration with realistic `.neuro` content

## Development

To extend the linter:

1. Add new validation logic to `NeuroLinter` class in `neuroslm/dsl/neuro_linter.py`
2. Add corresponding tests to `tests/dsl/test_neuro_linter.py`
3. Update the grammar in `.vscode/extensions/neuro-dsl/syntaxes/neuro.tmLanguage.json` for new syntax

### Key Classes

- **`NeuroLinter`**: Main linter class
  - `lint()`: Run all checks
  - `_check_brace_matching()`: Structural validation
  - `_extract_declarations()`: Parse the file for declarations
  - `_check_references()`: Validate cross-references
  - `_check_equations()`: Validate equations

- **`Diagnostic`**: Lint finding with location and severity

## Future Enhancements

- [ ] Real-time linting as you type (Language Server Protocol)
- [x] Go-to-definition for references (`@equation`, `@population`, `@dynamics`, `@function`)
- [ ] Hover tooltips with declaration info
- [ ] Code formatting (prettier-style)
- [ ] Recursive import checking for deep validation
- [ ] Type inference for shape expressions
- [ ] Autocomplete for `@reference` names
