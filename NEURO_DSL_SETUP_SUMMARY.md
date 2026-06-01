# NeuroSLM .neuro DSL — Complete IDE Integration Summary

This document summarizes the complete VSCode IDE integration and compiler validation system for `.neuro` architecture files.

## What's Been Implemented

### 1. VSCode Extension (Full-Featured)
**Location:** `.vscode/extensions/neuro-dsl/`

**Features:**
- ✅ **Syntax Highlighting**: Color-coded keywords, strings, numbers, comments
- ✅ **Real-Time Linting**: Validates on save and while typing (1s debounce)
- ✅ **Autocomplete**: Keywords, functions, variables, and population names
- ✅ **Diagnostics Panel**: Shows errors, warnings, and info messages
- ✅ **Cross-File Resolution**: Resolves ES6 imports to discover populations

**What the extension does:**
1. Detects `.neuro` files automatically
2. Runs linter on save (full validation)
3. Runs linter on type changes (debounced 1s to avoid lag)
4. Shows all issues in the Problems panel (Ctrl+Shift+M)
5. Provides autocomplete suggestions (Ctrl+Space)
6. Integrates with Python venv for linting

### 2. Automatic Installation
**Scripts:** `scripts/install.ps1` (Windows) and `scripts/install.sh` (Unix)

**What the install script does:**
1. Creates/reuses Python venv
2. Installs Python dependencies via pip
3. **Automatically detects VSCode installation** and installs extension to:
   - Windows: `C:\Users\<user>\AppData\Roaming\Code\User\extensions`
   - Linux/macOS: `~/.config/Code/User/extensions` or `~/.vscode/extensions`
4. Creates `.vscode/settings.json` with `.neuro` language configuration
5. Verifies the CLI is working
6. Displays success message with next steps

**To use:**
```powershell
# Windows
.\scripts\install.ps1

# Linux/macOS
bash scripts/install.sh
```

### 3. Compiler Integration
**File:** `neuroslm/dsl/compiler.py`

**Changes to `NeuroMLCompiler.compile_file()`:**
1. Runs linter **before** compilation
2. **Exits immediately** if structural errors found
3. Provides detailed error messages with line/column numbers
4. Continues to warnings only (reference issues may be in imported files)

**Error flow:**
```
compile_file("arch.neuro")
  ↓
NeuroLinter.lint()  ← runs first
  ↓
Errors found? → raise NeuroMLError with details
  ↓
No errors → proceed with compilation
```

### 4. Semantic Linter
**File:** `neuroslm/dsl/neuro_linter.py`

**Validation categories:**

| Category | Severity | Checks |
|----------|----------|--------|
| Structural | ERROR | Brace/bracket/paren matching, balanced blocks |
| References | WARNING | Population declarations, import resolution |
| Semantic | INFO | Equation variable binding, math function recognition |

**Diagnostic codes:**
- `unmatched-paren`, `unmatched-brace`, `unmatched-bracket` (errors)
- `unclosed-brace` (error)
- `undefined-population`, `unresolved-import` (warnings)
- `potentially-undefined` (info)

### 5. Test Coverage
**File:** `tests/dsl/test_neuro_linter.py`

**Test suite:** 18 tests, all passing
- 3 TestBraceMatching: structural validation
- 4 TestDeclarationExtraction: population/import/export parsing
- 2 TestReferenceValidation: cross-reference checking
- 2 TestEquationValidation: variable and function recognition
- 1 TestComments: comment handling
- 2 TestImportResolution: path resolution
- 1 TestIntegration: realistic architecture
- 3 TestCompilerIntegration: compiler validation

**Run tests:**
```bash
pytest tests/dsl/test_neuro_linter.py -v
```

## Usage Guide

### For End Users

#### Installation
```bash
# Run once to set up everything
.\scripts\install.ps1                    # Windows
bash scripts/install.sh                  # Linux/macOS

# Restart VSCode
# Open any .neuro file — linting is automatic!
```

#### Using the IDE

**Real-time validation while editing:**
- Open any `.neuro` file
- Errors appear immediately in the Problems panel
- Hover over squiggly lines for error details

**Autocomplete:**
- Press `Ctrl+Space` while editing
- Get suggestions for keywords, functions, variables, populations

**Command line linting:**
```bash
# Single file
python neuroslm/dsl/neuro_linter.py architectures/rcc_bowtie/arch.neuro

# Directory
python neuroslm/dsl/neuro_linter.py architectures/

# JSON output for automation
python neuroslm/dsl/neuro_linter.py architectures/ --json
```

### For Developers

#### Compiler validation
```python
from neuroslm.dsl.compiler import NeuroMLCompiler, NeuroMLError

try:
    ir = NeuroMLCompiler.compile_file("arch.neuro")
    # File is syntactically valid, proceed with compilation
except NeuroMLError as e:
    # Linting found errors
    print(f"Validation failed:\n{e}")
    # Error message includes line/col and specific issues
```

#### Running linter programmatically
```python
from neuroslm.dsl.neuro_linter import lint_file, Severity
from pathlib import Path

diagnostics = lint_file(Path("arch.neuro"))

errors = [d for d in diagnostics if d.severity == Severity.ERROR]
warnings = [d for d in diagnostics if d.severity == Severity.WARNING]

for d in errors:
    print(f"{d.file}:{d.line}:{d.col} [{d.code}] {d.message}")
```

#### CI/CD Integration
```bash
#!/bin/bash
# ci_lint.sh

# Lint all architectures
python neuroslm/dsl/neuro_linter.py architectures/ --json > results.json

# Fail if any errors
ERROR_COUNT=$(jq '[.[] | select(.severity == "error")] | length' results.json)
[ "$ERROR_COUNT" -eq 0 ] || exit 1
```

## Architecture

```
.vscode/extensions/neuro-dsl/
├── extension.js                    ← VSCode extension (linting + autocomplete)
├── package.json                    ← Manifest (activation events, commands)
├── language-configuration.json     ← Bracket/indent rules
├── syntaxes/neuro.tmLanguage.json ← Color syntax highlighting
├── README.md                       ← Extension documentation
└── <no node_modules>              (users install from npm)

scripts/
├── install.ps1                     ← Windows setup (detects VSCode)
└── install.sh                      ← Unix setup (detects VSCode)

neuroslm/dsl/
├── compiler.py                     ← Updated to run linter first
├── neuro_linter.py                 ← Semantic validator (3 severity levels)
└── <other DSL files>

tests/dsl/
├── test_neuro_linter.py            ← 18 passing tests
└── <other DSL tests>

docs/
└── dsl_editor_setup.md             ← User guide for IDE setup
```

## Key Features

### Early Error Detection
The compiler **validates syntax before compilation**, preventing mysterious errors during the compilation stage:

```
Before: "Unexpected token at line 42" ← confusing
After:  "Unclosed { at line 30" ← clear, actionable
```

### IDE Integration
Errors appear in-editor as you type, with full diagnostic info:
- Line and column numbers
- Severity (error/warning/info)
- Specific error code (e.g., `unclosed-brace`)
- Helpful error message

### Autocomplete
While editing, you get smart suggestions for:
- DSL keywords: `population`, `synapse`, `import`, `export`
- Math functions: `sin`, `cos`, `exp`, `log`, `sqrt`, `ReLU`, `tanh`, `sigmoid`, etc.
- Built-in variables: `x`, `y`, `s`, `V`, `d_sem`, `dt`
- Population names from current file

### Cross-File Validation
The linter understands ES6-style imports:
```neuro
import { sensory, motor } from "@/modules/sensory"
```

And automatically discovers exported populations from imported files, reducing false positives.

### CI/CD Ready
JSON output mode for easy integration:
```bash
python neuroslm/dsl/neuro_linter.py architectures/ --json | jq '.'
```

## Troubleshooting

### "Extension not installed"
- Run `.\scripts\install.ps1` or `bash scripts/install.sh`
- Restart VSCode
- Check that `.vscode/extensions/neuro-dsl/` exists

### "Python not found" when linting
- Ensure venv is activated: `.venv\Scripts\Activate.ps1` or `source .venv/bin/activate`
- Or run install script again

### Linting is slow
- Normal: ~100ms per file
- Linting is debounced to 1s while typing
- For 100+ file architectures, consider disabling auto-linting

### Extension not showing colors
- Verify file has `.neuro` extension
- Reload VSCode: `Ctrl+Shift+P` → "Developer: Reload Window"

## Next Steps (Optional Enhancements)

Potential future improvements (not implemented yet):
- [ ] Language Server Protocol (LSP) for faster real-time linting
- [ ] Recursive import validation (currently single-pass)
- [ ] Shape expression type checking
- [ ] Go-to-definition for synapses and imports
- [ ] Code formatting (prettier-style)
- [ ] Snippet templates for common blocks

## Files Changed

### New Files
- `.vscode/extensions/neuro-dsl/extension.js` (270 lines, real-time linting + autocomplete)
- `neuroslm/dsl/neuro_linter.py` (350 lines, semantic validator)
- `tests/dsl/test_neuro_linter.py` (300 lines, 18 tests)
- `docs/dsl_editor_setup.md` (comprehensive setup guide)

### Modified Files
- `neuroslm/dsl/compiler.py` (added linter validation before compilation)
- `scripts/install.ps1` (added VSCode extension installation)
- `scripts/install.sh` (added VSCode extension installation)

## Commits

```
10251c4 feat: .neuro DSL syntax highlighting + linter
da05157 feat: VSCode IDE integration + compiler linter validation
78fe425 docs: comprehensive VSCode IDE setup guide
```

## Summary

You now have:
1. **Automatic IDE setup** via install scripts
2. **Real-time error detection** while editing
3. **Smart autocomplete** for DSL constructs
4. **Compiler validation** that fails fast on errors
5. **CI/CD ready** JSON output for automation
6. **Comprehensive tests** (18 passing tests)
7. **Full documentation** for users and developers

The system catches errors early, provides clear diagnostics, and integrates seamlessly with the existing compiler.
