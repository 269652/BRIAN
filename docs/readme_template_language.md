# README Template Language Documentation

**Version:** 2.0  
**Command:** `brian update-readme`

## Overview

The README template system supports:
1. **Metric interpolation** from TOML
2. **Claim definitions** with evidence
3. **Log citations** (inline excerpts)
4. **Claim references** (accessing claim fields)

## Syntax Reference

### 1. Metrics: `${METRIC_NAME}`

Simple variable substitution from `docs/readme_metrics.toml`.

**Rules:**
- Only `${UPPERCASE_NAMES}` are treated as placeholders
- Lowercase `$var` or `$1.50` in prose are left untouched

**Example:**

`docs/readme_metrics.toml`:
```toml
[layer_a]
LAYER_A_TEST_COUNT = 42

[layer_b]
LAYER_B_BEST_GAP_RATIO = 6.55
```

`README.template.md`:
```markdown
Tests passing: ${LAYER_A_TEST_COUNT}
Gap ratio: ${LAYER_B_BEST_GAP_RATIO}
```

**Output:**
```markdown
Tests passing: 42
Gap ratio: 6.55
```

---

### 2. Claims: `$claim{ ... }`

Define scientific claims with metadata at the top of the template.

**Syntax:**
```
$claim{
    id: "unique_id",
    hypothesis: "H22",
    checkpoint: "hf://path/to/checkpoint.pt",
    train_ppl: 23.6,
    ood_ppl: 155.0,
    gap_ratio: 6.55,
    back: [
        ["logs/path/to/file.log", start_line, end_line]
    ],
    falsify: [],
    custom_field: "any value"
}
```

**Fields:**
- `id` (required): Unique identifier for referencing
- `hypothesis`: Hypothesis ID (e.g., "H22")
- `checkpoint`: Model checkpoint path
- `train_ppl`: Training perplexity
- `ood_ppl`: Out-of-distribution perplexity
- `gap_ratio`: Generalization gap ratio
- `back`: List of supporting evidence (log citations)
- `falsify`: List of contradicting evidence
- **Custom fields**: Any additional key-value pairs

**Example:**
```markdown
$claim{
    id: "H22_best",
    hypothesis: "H22",
    checkpoint: "hf://model/step10000.pt",
    train_ppl: 23.6,
    ood_ppl: 155.0,
    gap_ratio: 6.55,
    date: "2026-06-15",
    back: [
        ["logs/20260615/arch/175931_0_10000.log", 802, 804]
    ]
}
```

**Behavior:**
- Claim definitions are **removed** from the final output
- Claims must be defined **before** they're referenced
- Best practice: Define all claims at the top of the template

---

### 3. Claim References: `${claim.ID.field}`

Access claim fields anywhere in the template.

**Syntax:**
```
${claim.ID.field}
${claim.ID.field[index]}  # Array access
```

**Examples:**
```markdown
OOD PPL: ${claim.H22_best.ood_ppl}
Checkpoint: ${claim.H22_best.checkpoint}
Date: ${claim.H22_best.date}
Gap ratio: ${claim.H22_best.gap_ratio}
```

**Output:**
```markdown
OOD PPL: 155.0
Checkpoint: hf://model/step10000.pt
Date: 2026-06-15
Gap ratio: 6.55
```

**Array access:**
```markdown
First evidence: ${claim.H22_best.back[0]}
```

This expands to a full log citation (see below).

---

### 4. Log Citations: `$cite(path, start, end)`

Inline log file excerpts as code blocks.

**Syntax:**
```
$cite(relative/path/to/file.log, start_line, end_line)
```

**Parameters:**
- `path`: Relative to repo root
- `start_line`: 1-indexed, inclusive
- `end_line`: 1-indexed, inclusive

**Example:**
```markdown
Training completed successfully:

$cite(logs/20260615/arch/175931_0_10000.log, 802, 804)
```

**Output:**
````markdown
Training completed successfully:

```
[train_dsl] final OOD: wikitext ppl=155.0  train_ppl=23.6  gap_ratio=6.55
[train_dsl] wrote final OOD result → logs/vast/benchmarks/ood/ood_final_20260615-175931.json
[train_dsl] done.
```
````

**Line range rules:**
- `$cite(file.log, 3, 3)` — single line (line 3)
- `$cite(file.log, 1, 5)` — lines 1-5 inclusive
- Line numbers are 1-indexed (like text editors)

---

## Complete Example

`docs/README.template.md`:
```markdown
$claim{
    id: "H22_smollm2",
    hypothesis: "H22",
    checkpoint: "hf://moritzroessler/BRIAN/checkpoints/20260615-175931/step10000.pt",
    train_ppl: 23.6,
    ood_ppl: 155.0,
    gap_ratio: 6.55,
    arch: "neuroslm-full-dna-arch",
    back: [
        ["logs/20260615/neuroslm-full-dna-arch/175931_0_10000.log", 802, 804]
    ]
}

$claim{
    id: "baseline",
    hypothesis: "flat_transformer",
    ood_ppl: 180.3,
    gap_ratio: 3.99
}

# BRIAN

> ${LAYER_A_TEST_COUNT} tests passing

## Latest Results

Best run achieved **${claim.H22_smollm2.gap_ratio}** gap ratio vs baseline ${claim.baseline.gap_ratio}.

### Evidence

$cite(logs/20260615/neuroslm-full-dna-arch/175931_0_10000.log, 802, 804)

Improvement: ${LAYER_B_IMPROVEMENT_PCT}% better than flat transformer.
```

**Output:**
```markdown
# BRIAN

> 42 tests passing

## Latest Results

Best run achieved **6.55** gap ratio vs baseline 3.99.

### Evidence

```
[train_dsl] final OOD: wikitext ppl=155.0  train_ppl=23.6  gap_ratio=6.55
[train_dsl] wrote final OOD result → logs/vast/benchmarks/ood/ood_final_20260615-175931.json
[train_dsl] done.
```

Improvement: 64% better than flat transformer.
```

---

## Usage

### Generate README
```bash
brian update-readme
```

### Check if stale (pre-commit)
```bash
brian update-readme --check
```

**Exit codes:**
- `0` — Success (or README is up-to-date in `--check` mode)
- `1` — Error (missing metrics, missing claims, or stale README)

---

## Error Handling

### Missing metric
```
README template has 1 unresolved metric(s): MISSING_METRIC
Add them to docs/readme_metrics.toml
```

**Fix:** Add the key to `docs/readme_metrics.toml`

### Missing claim
```
Claim 'NONEXISTENT' referenced but not defined
```

**Fix:** Add `$claim{ id: "NONEXISTENT", ... }` at top of template

### Missing log file
```
Log file not found: logs/missing.log
```

**Fix:** 
- Verify log path is correct (relative to repo root)
- Check that migration has been applied (old logs may be renamed)
- Use `brian migrate --list` to see migration status

### Stale README
```
README.md is stale — run `brian update-readme` and stage the result.
```

**Fix:** Run `brian update-readme` and commit the updated README.md

---

## Best Practices

### 1. Define claims at the top
```markdown
$claim{ id: "claim1", ... }
$claim{ id: "claim2", ... }
$claim{ id: "claim3", ... }

# Document starts here...
```

### 2. Use descriptive claim IDs
```
Good:  H22_smollm2_best, baseline_flat_transformer
Bad:   claim1, test, foo
```

### 3. Keep log citations short
```
$cite(file.log, 800, 803)  ✅ 3-4 lines, focused
$cite(file.log, 1, 1000)   ❌ Too long, context lost
```

### 4. One claim per run/experiment
```markdown
$claim{
    id: "H22_run1",
    date: "2026-06-15",
    ...
}

$claim{
    id: "H22_run2", 
    date: "2026-06-16",
    ...
}
```

### 5. Document falsifying evidence
```markdown
$claim{
    id: "hypothesis",
    back: [["good_result.log", 10, 12]],
    falsify: [["bad_result.log", 50, 55]]  # Honest reporting!
}
```

---

## Migration from v1

**v1 (simple metrics):**
```markdown
Tests: ${LAYER_A_TEST_COUNT}
```

**v2 (same, still works):**
```markdown
Tests: ${LAYER_A_TEST_COUNT}
```

**v2 (new features):**
```markdown
$claim{ id: "best", ood_ppl: 155.0 }
Best OOD: ${claim.best.ood_ppl}
$cite(logs/file.log, 1, 3)
```

**No breaking changes** — old templates still work!

---

## Architecture

```
Template (README.template.md)
    ↓
Parser extracts $claim{...} → Claims registry
    ↓
$cite(...) → Read log files
    ↓
${claim.ID.field} → Lookup in registry
    ↓
${METRIC} → Lookup in metrics TOML
    ↓
Rendered README.md
```

**Files:**
- `neuroslm/readme_renderer_v2.py` — Template engine
- `docs/README.template.md` — Source template
- `docs/readme_metrics.toml` — Metrics data
- `README.md` — Generated output

**Tests:**
- `tests/test_readme_renderer_v2.py` — Full test suite

---

## Future Extensions

Potential additions:
- `$table{ ... }` — Generate comparison tables
- `$graph{ ... }` — Inline plots from metrics
- `$hypothesis{ ... }` — Link to hypothesis/ folder
- `${claim.ID.lean_proof}` — Link to Lean verification

Open an issue to request features!
