# Log Restructure Plan — 2026-06-16

## Summary

Restructure training log storage from:
- `logs/<YYYYMMDD>/<arch>/<HHMMSS>_<sha>/train.log`

To:
- `logs/<YYYYMMDD>/<arch>/<HHMMSS>_<start>_<end>.log`

Where:
- `HHMMSS` = UTC boot time (e.g., `175931` for 17:59:31 UTC)
- `start` = starting step (e.g., `0` or `10000`)
- `end` = ending step (e.g., `10000`, `20000`)

**Example:**
```
logs/20260615/master.dna/175931_0_10000.log
logs/20260615/master.dna/184943_10000_20000.log
```

**Rationale:**
- `train.log` is too generic — every run has identical filename
- Step range in filename makes it immediately clear what each log contains
- Boot time still provides unique per-run identity
- Arch folder name should match actual architecture name (not hash-based)

## Specific Run to Preserve

**20260615-175931** first completed 10k run:
- Path: `logs/20260615/arch_unk_dna-arch/185105_41084160/train.log`
- OOD PPL: 155.0
- Train PPL: 23.6
- Gap Ratio: 6.55
- Checkpoint: `hf://moritzroessler/BRIAN/checkpoints/20260615-175931_c19bf629_neuroslm-full-dna-arch/step10000.pt`

## Components to Update

### 1. Document the Run
- [ ] Add entry to `docs/FINDINGS.md` referencing the specific log
- [ ] Use new path format: `logs/20260615/neuroslm-full-dna-arch/175931_0_10000.log`

### 2. Migration (0002_log_name_refactor)
- [ ] Create `neuroslm/migrations/0002_log_name_refactor.py`
- [ ] Parse existing `train.log` files to extract step range from content
- [ ] Rename to `<HHMMSS>_<start>_<end>.log`
- [ ] Update folder names from hash-based to actual arch names
- [ ] Handle edge cases (unparseable logs go to `_unsorted_legacy/`)

### 3. Log Pusher
- [ ] Update `scripts/log_pusher.sh::_compose_logfile()`
- [ ] Detect current step from `/workspace/train.log` content
- [ ] Form path: `logs/$DAY/$ARCH_NAME/$BOOT_TIME_${START_STEP}_${CURRENT_STEP}.log`
- [ ] On each push, overwrite same file (step range updates in place)

### 4. Clean Command
- [ ] Update `neuroslm/tools/clean.py::_enumerate_logs()`
- [ ] Match pattern: `logs/<8-digit>/<arch>/<6-digit>_<digits>_<digits>.log`
- [ ] Reference protection via folder name + log basename

### 5. Reference Checker
- [ ] Update `neuroslm/references.py`
- [ ] Ensure new log format patterns are captured in `_TOKEN_RE`
- [ ] Update folder token extraction

### 6. Tests
- [ ] Update `tests/test_migration_0001.py` (may need new test file for 0002)
- [ ] Update `tests/training/test_log_pusher_naming.py`
- [ ] Update `tests/test_clean.py`
- [ ] Update `tests/test_clean_lfs.py`
- [ ] Update `tests/test_references_exact_only.py`

### 7. Documentation
- [ ] Update `docs/REPO_SETUP.md`
- [ ] Update `CLAUDE.md` (log naming contract section)
- [ ] Update `.gitignore` patterns
- [ ] Update `brian.toml` comments if needed

## Implementation Order

1. Document the preserved run in FINDINGS.md
2. Create migration 0002
3. Update log_pusher.sh
4. Update clean.py
5. Update references.py
6. Update all tests
7. Update all documentation
8. Run `brian migrate --list` to verify
9. Run `brian migrate 0002 --force` to apply
10. Run `brian clean logs` dry-run to verify

## Edge Cases

1. **Logs still in progress** — log_pusher updates step range on each push
2. **Resumed runs** — new log file with new boot time, appropriate start step
3. **Failed runs** — step range shows actual completion (e.g., `175931_0_7800.log`)
4. **Unparseable logs** — migration moves to `_unsorted_legacy/`

## Testing Strategy

1. Unit tests for each updated component
2. Integration test: full deploy → log → clean → reference check cycle
3. Migration dry-run on copy of logs/
4. Verify brian.toml extra_keep still works

## Rollback Plan

If issues arise:
1. Migration is idempotent (re-running is safe)
2. Original files preserved until `brian clean --force`
3. Can revert migration by deleting `.brian/migrations.json` entry
4. Git history preserves old structure

## Success Criteria

- [ ] All tests pass
- [ ] `brian migrate --list` shows 0002 as APPLIED with 0 ops
- [ ] `brian clean logs` dry-run shows correct protection
- [ ] New deploys create correctly-named logs
- [ ] Documentation updated and consistent
- [ ] No breaking changes to existing workflows
