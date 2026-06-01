# VSCode NeuroSLM DSL Extension — Troubleshooting Guide

If you're not seeing syntax highlighting or linting diagnostics, follow these steps.

## Issue: No Syntax Highlighting or Linting

### Step 1: Hard Reload VSCode

1. **Close VSCode completely**
2. **Open VSCode again**
3. **Open any `.neuro` file**

If still not working, proceed to Step 2.

### Step 2: Check Extension Installation

1. Press `Ctrl+Shift+X` (Extensions)
2. Search for "NeuroSLM" or "neuro-dsl"
3. You should see "NeuroSLM DSL" listed
4. Check that it's **enabled** (if it shows "Disable", it's enabled)

If not listed, proceed to Step 3.

### Step 3: Manually Install Extension

```powershell
# PowerShell
$appdata = [System.Environment]::GetFolderPath("ApplicationData")
$ext_dest = "$appdata\Code\User\extensions\neuro-dsl"

# Remove old version if it exists
if (Test-Path $ext_dest) {
    Remove-Item $ext_dest -Recurse -Force
}

# Copy extension
Copy-Item -Path ".vscode\extensions\neuro-dsl" -Destination $ext_dest -Recurse -Force
Write-Host "✓ Extension installed to $ext_dest"
```

Then close and reopen VSCode.

### Step 4: Check Output Channel

1. Open VSCode
2. View → Output (or `Ctrl+Shift+U`)
3. In the dropdown menu, select **"NeuroSLM DSL"**
4. Open a `.neuro` file
5. Look for messages like:
   - `NeuroSLM DSL extension activating...`
   - `Linting on open: ...`
   - `Found Python at: ...`

**What to look for:**

✓ **Good output:**
```
NeuroSLM DSL extension activating...
✓ NeuroSLM DSL extension activated successfully
Linting on open: /path/to/arch.neuro
Using Python: C:\Users\...\SLM\.venv\Scripts\python.exe
Found 0 diagnostic(s)
```

✗ **Bad output:**
```
Linter error: spawn ENOENT
Error: Python executable not found
```

### Step 5: Check Language Association

1. Open a `.neuro` file
2. Look at the **bottom right** of VSCode
3. You should see **"neuro"** as the language

If you see "Plain Text" instead:
1. Click on "Plain Text"
2. Select "NeuroSLM DSL" or "neuro"

### Step 6: Verify Python and Linter Work

```powershell
# Activate venv
.venv\Scripts\Activate.ps1

# Test linter directly
python neuroslm/dsl/neuro_linter.py architectures/rcc_bowtie/arch.neuro

# Should output diagnostics or "0 error(s), X warning(s)"
```

If linter doesn't work, the extension won't either.

---

## Issue: Linting Works but No Diagnostics Shown

### Check Problems Panel

1. View → Problems (or `Ctrl+Shift+M`)
2. Make sure **NeuroSLM DSL** is listed in the dropdown
3. Open a `.neuro` file with an error (e.g., undefined population)

### Check Settings

Verify `.vscode/settings.json` exists and contains:

```json
{
  "files.associations": {
    "*.neuro": "neuro"
  },
  "[neuro]": {
    "editor.defaultFormatter": "null",
    "editor.formatOnSave": false,
    "editor.wordBasedSuggestions": "off"
  }
}
```

---

## Issue: Autocomplete Not Working

### Try Triggering Manually

1. Open a `.neuro` file
2. Start typing (e.g., "pop")
3. Press `Ctrl+Space` to trigger autocomplete
4. You should see suggestions like "population"

If nothing appears:
- Check Output channel (Step 4 above)
- Verify the file is recognized as `neuro` language
- Make sure you're inside the project folder (not just a loose file)

---

## Issue: Extension Throws Error

### Check the Output Channel

View → Output → Select "NeuroSLM DSL"

**Common errors and fixes:**

| Error | Cause | Fix |
|-------|-------|-----|
| `spawn ENOENT` | Python not found | Run `install.ps1` again; ensure `.venv` exists |
| `No module named neuroslm` | Dependencies not installed | Run `install.ps1` again |
| `Python version < 3.10` | Wrong Python | Ensure venv is in `.venv/` |
| `JSON parse error` | Linter output malformed | Check linter works: `python neuroslm/dsl/neuro_linter.py arch.neuro` |

---

## Nuclear Option: Complete Reset

If nothing works, start from scratch:

```powershell
# 1. Remove installed extension
$appdata = [System.Environment]::GetFolderPath("ApplicationData")
$ext_dir = "$appdata\Code\User\extensions\neuro-dsl"
if (Test-Path $ext_dir) {
    Remove-Item $ext_dir -Recurse -Force
    Write-Host "✓ Removed VSCode extension"
}

# 2. Delete venv (optional but recommended)
if (Test-Path ".venv") {
    Remove-Item ".venv" -Recurse -Force
    Write-Host "✓ Removed venv"
}

# 3. Run install script again
.\scripts\install.ps1

# 4. Restart VSCode
# (Close completely and reopen)
```

---

## Verify Installation Checklist

After setup, verify these points:

- [ ] `.venv\Scripts\python.exe` exists
- [ ] `.vscode\extensions\neuro-dsl\extension.js` exists (locally)
- [ ] `%APPDATA%\Code\User\extensions\neuro-dsl\extension.js` exists (installed)
- [ ] `.vscode\settings.json` has `"*.neuro": "neuro"` association
- [ ] Opening `.neuro` file shows **"neuro"** in bottom-right language selector
- [ ] Output channel shows "extension activated successfully"
- [ ] `python neuroslm/dsl/neuro_linter.py arch.neuro` works from command line
- [ ] Problems panel shows at least one diagnostic when opening `arch.neuro`

---

## Getting Help

If you're still stuck, provide this information:

1. **Output channel output** (View → Output → "NeuroSLM DSL")
2. **What you see in bottom-right** (language selector)
3. **Whether linter works from command line:**
   ```powershell
   .venv\Scripts\Activate.ps1
   python neuroslm/dsl/neuro_linter.py architectures/rcc_bowtie/arch.neuro
   ```
4. **Python version:**
   ```powershell
   python --version
   ```

---

## Quick Test

**Create a test file `test.neuro`:**
```neuro
population sensory {
    count: 256
}

synapse sensory -> undefined_pop {
    weight: 0.5
}
```

**Expected results:**
- Line 6 should have a yellow squiggly line under `undefined_pop`
- Problems panel should show: `[undefined-population] Population 'undefined_pop' not declared or imported`
- When you type "syn" on a new line, autocomplete should suggest "synapse"

If these work, your setup is correct!
