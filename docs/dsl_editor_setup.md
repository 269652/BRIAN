# .neuro DSL Editor Setup Guide

This guide explains how to set up syntax highlighting and linting for `.neuro` architecture files in VSCode.

## Quick Start

### 1. Enable the VSCode Extension

The extension is already in `.vscode/extensions/neuro-dsl/`. VSCode should auto-discover it. To verify:

1. Open VSCode settings: `Ctrl+,` (or `Cmd+,` on Mac)
2. Search for "neuro"
3. You should see the extension listed as "NeuroSLM DSL"

### 2. Verify Syntax Highlighting

1. Open any `.neuro` file (e.g., `architectures/rcc_bowtie/arch.neuro`)
2. You should see:
   - Keywords in blue (`architecture`, `population`, `synapse`, etc.)
   - Strings in green (equations in quotes)
   - Numbers in orange
   - Comments in gray

### 3. Run the Linter

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

## See Also

- [Linter Documentation](.vscode/extensions/neuro-dsl/README.md)
- [DSL Reference](dsl.md)
- [Architecture Specification](architecture.md)
