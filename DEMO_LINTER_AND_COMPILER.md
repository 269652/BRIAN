# .neuro DSL Linter & Compiler Integration — Live Demo

This document shows the complete system in action.

## Demo 1: Linter Catches Reference Errors

**File:** `demo.neuro`
```neuro
architecture demo {
    d_sem: 256,
    dt: 0.01
}

population sensory {
    count: 64,
    dynamics: "rate_code"
}

# ERROR: 'motor' population is not declared
synapse sensory -> motor {
    weight: 0.5
}
```

**Running the linter:**
```bash
python neuroslm/dsl/neuro_linter.py demo.neuro
```

**Output:**
```
demo.neuro:12:20 [undefined-population] Population 'motor' not declared or imported

0 error(s), 1 warning(s)
```

**What happened:**
- The linter identified that the synapse references an undefined population
- It reports the file, line, column, code, and message
- The warning doesn't prevent execution (it's a reference issue that might be in an imported file)

---

## Demo 2: Compiler Prevents Compilation with Structural Errors

**File:** `bad.neuro`
```neuro
architecture test {
    d_sem: 256
}

population sensory {
    count: 64,
    dynamics: "rate_code"

# Missing closing brace - structural error!
```

**Calling the compiler:**
```python
from neuroslm.dsl.compiler import NeuroMLCompiler, NeuroMLError

try:
    result = NeuroMLCompiler.compile_file("bad.neuro")
except NeuroMLError as e:
    print(f"Compilation blocked:\n{e}")
```

**Output:**
```
Compilation blocked:
Linting failed: 1 error(s)
  bad.neuro:5:19 Unclosed {
```

**What happened:**
1. `compile_file()` calls `NeuroLinter` first (before any compilation)
2. The linter detects an unclosed brace (structural error)
3. The compiler raises `NeuroMLError` immediately
4. Compilation never happens — error is caught early

---

## Demo 3: Valid File Passes Both Linter and Compiler

**File:** `valid.neuro`
```neuro
architecture demo {
    d_sem: 256,
    dt: 0.01
}

population sensory {
    count: 64,
    dynamics: "rate_code"
}

population motor {
    count: 32,
    equation: "y = ReLU(x)"
}

synapse sensory -> motor {
    weight: 0.7,
    neurotransmitter: "glutamate"
}
```

**Running linter:**
```bash
python neuroslm/dsl/neuro_linter.py valid.neuro
```

**Output:**
```
0 error(s), 0 warning(s)
```

**Calling compiler:**
```python
from neuroslm.dsl.compiler import NeuroMLCompiler

result = NeuroMLCompiler.compile_file("valid.neuro")
# Compilation succeeds — linting found no errors
print("Compilation succeeded!")
```

**Output:**
```
Compilation succeeded!
```

**What happened:**
1. Linter validates all populations are declared
2. No structural errors
3. Compiler proceeds with compilation
4. File is successfully compiled

---

## Demo 4: VSCode IDE Real-Time Validation

When you open `demo.neuro` in VSCode (after running `install.ps1`):

**In the editor:**
```
architecture demo {
    d_sem: 256,
    dt: 0.01
}

population sensory {
    count: 64,
    dynamics: "rate_code"
}

synapse sensory -> motor {  ← yellow squiggly line here
    weight: 0.5
}
```

**In the Problems panel:**
```
demo.neuro (1 warning)
  Line 12, Column 20: [undefined-population] Population 'motor' not declared or imported
```

**Autocomplete (Ctrl+Space after "->"):**
```
motor
sensory
```

---

## Demo 5: JSON Output for CI/CD

**Running linter with JSON output:**
```bash
python neuroslm/dsl/neuro_linter.py demo.neuro --json
```

**Output:**
```json
[
  {
    "file": "demo.neuro",
    "line": 12,
    "col": 20,
    "severity": "warning",
    "code": "undefined-population",
    "message": "Population 'motor' not declared or imported"
  }
]
```

**CI script usage:**
```bash
#!/bin/bash
# Lint and fail on errors
python neuroslm/dsl/neuro_linter.py architectures/ --json > results.json
ERROR_COUNT=$(jq '[.[] | select(.severity == "error")] | length' results.json)
exit $ERROR_COUNT
```

---

## Three Severity Levels Explained

### 🔴 ERROR (Red)
**When:** Structural issues that break the file
- Unmatched braces `{` `}` `[` `]` `(`  `)`
- Syntax errors

**Example:**
```
bad.neuro:5:19 [unclosed-brace] Unclosed {
```

**What to do:** **Fix immediately** — prevents compilation

### 🟡 WARNING (Yellow)
**When:** Reference issues that may be correct (e.g., in imported files)
- Undefined population in synapse
- Unresolved import path

**Example:**
```
arch.neuro:42:8 [undefined-population] Population 'X' not declared or imported
```

**What to do:** Check if it should be imported or declared

### ℹ️ INFO (Blue)
**When:** Potentially undefined variables
- Variable in equation that might not be bound

**Example:**
```
arch.neuro:50:15 [potentially-undefined] Variable 'W' may be undefined
```

**What to do:** Usually safe to ignore (bound at runtime)

---

## Diagnostic Codes Reference

| Code | Severity | Meaning |
|------|----------|---------|
| `unmatched-paren` | Error | `(` without matching `)` |
| `unmatched-brace` | Error | `{` without matching `}` |
| `unmatched-bracket` | Error | `[` without matching `]` |
| `unclosed-brace` | Error | Opening brace with no match |
| `undefined-population` | Warning | Population in synapse not found |
| `unresolved-import` | Warning | Import path doesn't exist |
| `potentially-undefined` | Info | Variable may not be bound |

---

## Complete Workflow

### Step 1: Edit `.neuro` File
- Open in VSCode
- Errors show in real-time
- Autocomplete helps you type

### Step 2: Run Linter (Optional)
```bash
python neuroslm/dsl/neuro_linter.py arch.neuro
```

### Step 3: Compile
```python
ir = NeuroMLCompiler.compile_file("arch.neuro")
# If linter found errors → raises NeuroMLError
# If no errors → compilation proceeds
```

### Step 4: Use IR
```python
# Now safe to use ir.populations, ir.synapses, etc.
for pop in ir.populations:
    print(f"{pop.name}: {pop.count} neurons")
```

---

## Key Takeaways

1. **Linter runs first** — before any compilation
2. **Errors block compilation** — structural issues caught early
3. **Warnings don't block** — but should be reviewed
4. **IDE integration** — real-time validation while editing
5. **JSON output** — easy CI/CD integration
6. **Clear diagnostics** — file:line:col with specific error codes

This system ensures `.neuro` files are syntactically valid before they reach the compiler, saving time and preventing confusing errors downstream.
