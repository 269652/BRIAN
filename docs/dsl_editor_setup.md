# .neuro DSL Editor Setup Guide

This guide explains how to set up the complete .neuro DSL development environment with syntax highlighting, real-time linting, autocomplete, and compiler validation.

## Quick Start

### 1. Run the Install Script

The easiest way to set everything up:

**Windows (PowerShell):**
```powershell
.\scripts\install.ps1
```

**Linux/macOS (bash):**
```bash
bash scripts/install.sh
```

This script will:
- Install Python dependencies
- Copy the VSCode extension to your extensions directory
- Create workspace settings for `.neuro` files
- Verify the installation

### 2. Restart VSCode

Close and reopen VSCode to activate the extension.

### 3. Verify Setup

1. Open any `.neuro` file (e.g., `architectures/rcc_bowtie/arch.neuro`)
2. You should see:
   - ✅ Colored syntax highlighting (keywords in blue, strings in green)
   - ✅ Real-time linting errors/warnings in the Problems panel
   - ✅ Autocomplete suggestions when you type (Ctrl+Space)
   - ✅ Error messages for undefined populations, unmatched braces, etc.

## Features

### Real-Time Linting

The extension validates your `.neuro` files automatically:
- **On save**: Full linting
- **On type**: Validation with 1-second debounce (to avoid lag)
- **Diagnostics panel**: View all errors/warnings in `View → Problems`

**Diagnostic levels:**
- 🔴 **Error** (red): Structural issues (unmatched braces, syntax errors)
- 🟡 **Warning** (yellow): Reference issues (undefined populations, unresolved imports)
- ℹ️ **Info** (blue): Potential issues (possibly undefined variables)

### Autocomplete

Press **Ctrl+Space** (or **Cmd+Space** on Mac) to get suggestions for:
- Keywords: `population`, `synapse`, `import`, `export`, `architecture`
- Built-in functions: `sin`, `cos`, `exp`, `log`, `sqrt`, `ReLU`, `tanh`, `sigmoid`
- Built-in variables: `x`, `y`, `s`, `V`, `d_sem`, `dt`
- Population names from your current file

### Compiler Validation

When you compile a `.neuro` file, the compiler automatically:
1. Runs the linter first
2. **Exits immediately** if there are structural errors (unmatched braces, syntax errors)
3. Provides detailed error messages with line/column numbers

Example:
```
Linting failed: 1 error(s)
  arch.neuro:42:5 Unclosed {
```

### Command-Line Linting

From the command line:

```bash
# Lint a single file
python neuroslm/dsl/neuro_linter.py architectures/rcc_bowtie/arch.neuro

# Lint all .neuro files in a directory
python neuroslm/dsl/neuro_linter.py architectures/

# Output results as JSON (useful for CI/automation)
python neuroslm/dsl/neuro_linter.py architectures/ --json > lint_results.json
```

## Understanding Linter Output

The linter reports three types of diagnostics:

| Symbol | Meaning | Action |
|--------|---------|--------|
| 🔴 | Error | Must fix before deployment |
| 🟡 | Warning | Review; usually safe to ignore for imported modules |
| ℹ️ | Info | Informational; may indicate a design issue |

### Common Warnings

**"Population 'X' not declared or imported"**
- Cause: A synapse references a population not declared in the file or its imports
- Fix: Add the import or check the spelling
- Example:
  ```neuro
  import { sensory } from "@/modules/sensory"
  
  synapse sensory -> motor {  # ❌ motor not imported
    weight: 0.5
  }
  ```
- Solution:
  ```neuro
  import { sensory } from "@/modules/sensory"
  import { motor } from "@/modules/motor"
  
  synapse sensory -> motor {  # ✅ motor imported
    weight: 0.5
  }
  ```

**"Cannot resolve import '@/X'"**
- Cause: Import path doesn't exist
- Fix: Check that the file exists at the path
- Example:
  ```neuro
  import { foo } from "@/modules/nonexistent"  # ❌ file doesn't exist
  ```
- Solution:
  ```neuro
  import { foo } from "@/modules/sensory"  # ✅ file exists
  ```

**"Variable 'X' may be undefined"**
- Cause: A variable in an equation may not be bound
- Fix: Check the equation context; usually safe to ignore if it's a runtime parameter
- Example:
  ```neuro
  equation: "y = W * x"  # ℹ️ W may be undefined (it's OK — bound at runtime)
  ```

## Integration with CI/Deployment

To add linting to your CI pipeline:

```bash
#!/bin/bash
# ci_lint.sh

set -e

# Lint all architectures
python neuroslm/dsl/neuro_linter.py architectures/ --json > .lint_results.json

# Fail if any errors
ERROR_COUNT=$(jq '[.[] | select(.severity == "error")] | length' .lint_results.json)
if [ "$ERROR_COUNT" -gt 0 ]; then
  echo "Linting failed: $ERROR_COUNT error(s)"
  jq '.[] | select(.severity == "error")' .lint_results.json
  exit 1
fi

echo "Linting passed!"
```

## Writing Valid `.neuro` Files

### Best Practices

1. **Always import populations before using them in synapses**
   ```neuro
   import { pop_a, pop_b } from "@/modules/sensory"
   
   synapse pop_a -> pop_b { weight: 0.5 }
   ```

2. **Export public populations**
   ```neuro
   export population my_region {
       count: 256,
       equation: "y = ReLU(x)"
   }
   ```

3. **Use descriptive variable names in equations**
   ```neuro
   equation: "y = alpha * x + beta"  # ✅ Clear
   equation: "y = a * x + b"         # ⚠️ Less clear
   ```

4. **Organize imports by module**
   ```neuro
   # Sensory input
   import { visual, auditory } from "@/modules/sensory"
   
   # Motor output
   import { motor } from "@/modules/motor"
   
   # Processing
   import { gws, pfc } from "@/modules/cortex"
   ```

## Troubleshooting

### Syntax highlighting not working

1. Ensure the file has `.neuro` extension
2. Reload VSCode: `Ctrl+Shift+P` → "Developer: Reload Window"
3. Check extension is enabled: `Ctrl+Shift+X` → search "NeuroSLM"

### Linter reports false positives

1. Check that imports use the correct format: `import { name } from "@/path"`
2. Verify the imported file path exists (case-sensitive on Unix)
3. Note: Single-pass linter doesn't follow transitive imports; if module A imports B which imports C, C's exports won't be visible in A

### Linter is slow

- The linter currently does a single pass with import resolution
- For very large architectures (100+ files), consider running it in CI only, not on save

## Compiler Validation Details

The compiler now integrates the linter to catch errors early:

### How It Works

1. You call `compile_file("arch.neuro")`
2. Linter validates structure (braces, brackets, parens)
3. If errors found → raises `NeuroMLError` immediately
4. If no errors → proceeds with compilation

### Example

**With linting (catches errors early):**
```python
from neuroslm.dsl.compiler import NeuroMLCompiler

try:
    ir = NeuroMLCompiler.compile_file("arch.neuro")
except NeuroMLError as e:
    print(f"Validation failed: {e}")
    # Error message includes line/col and specific issues
```

**Error output:**
```
NeuroMLError: Linting failed: 1 error(s)
  arch.neuro:42:5 Unclosed {
```

## Advanced Usage

### Manual Linting from Command Line

```bash
# Lint a single file
python -m neuroslm.dsl.neuro_linter arch.neuro

# Lint a directory
python -m neuroslm.dsl.neuro_linter architectures/

# JSON output (for CI/automation)
python -m neuroslm.dsl.neuro_linter architectures/ --json > results.json
```

### CI/CD Integration

```bash
#!/bin/bash
# ci_lint.sh

set -e

# Lint all architectures
python -m neuroslm.dsl.neuro_linter architectures/ --json > .lint_results.json

# Fail if errors (warnings are OK)
ERROR_COUNT=$(jq '[.[] | select(.severity == "error")] | length' .lint_results.json)
if [ "$ERROR_COUNT" -gt 0 ]; then
  echo "Linting failed: $ERROR_COUNT error(s)"
  exit 1
fi

echo "✓ Linting passed"
```

### Disabling Auto-Linting

If you want to turn off real-time linting:

1. Edit `.vscode/settings.json` (at repo root)
2. Comment out the `neuro-dsl` configuration
3. You can still run linting manually via command line

## Troubleshooting

### "Python not found" when opening .neuro files

The extension looks for Python in this order:
1. `.venv/Scripts/python.exe` (Windows venv)
2. `.venv/bin/python` (Linux/macOS venv)
3. System `python3` or `python`

**Fix:** Ensure you've run the install script and activated the venv

### Linting is slow

- Linting is debounced to 1 second while typing
- Full lint on save takes ~100ms for typical files
- For very large architectures (100+ files), consider disabling auto-linting

### Extension not showing color/linting

1. Verify file has `.neuro` extension
2. Check that extension is installed: `Ctrl+Shift+X` → search "NeuroSLM"
3. Reload VSCode: `Ctrl+Shift+P` → "Developer: Reload Window"

## See Also

- [Linter Documentation](.vscode/extensions/neuro-dsl/README.md)
- [DSL Reference](dsl.md)
- [Architecture Specification](architecture.md)
