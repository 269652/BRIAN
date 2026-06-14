# Repo Setup — fresh clone to working dev environment

This document is the **single-source procedure** to take an empty
machine to a fully-working NeuroSLM checkout: code, LFS-tracked
checkpoints, SSH-authenticated push, pre-commit testing, and a clean
ledger of applied migrations. Follow it in order.

> **You should NOT need any of the troubleshooting sections at the
> bottom**; they exist for the cases where something went wrong.

---

## 1. Clone the repo over SSH

Use **SSH** (not HTTPS + PAT). SSH keys never expire, never end up
embedded in `.git/config`, and never accidentally appear in screenshots.

```bash
# (one-time per machine) generate an ed25519 key if you don't have one
ssh-keygen -t ed25519 -C "<your-email>"
# print the public key, paste it into github.com -> Settings -> SSH keys
cat ~/.ssh/id_ed25519.pub

# verify auth before cloning
ssh -T git@github.com
# expected: "Hi <github-username>! You've successfully authenticated"

# clone
git clone git@github.com:269652/BRIAN.git SLM
cd SLM
```

**Windows / PowerShell:** the key path is `$env:USERPROFILE\.ssh\id_ed25519.pub`.
Use `Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub | Set-Clipboard` to copy.

---

## 2. Install Git LFS and pull the binary objects

Checkpoints (`*.pt`, `*.mem`, `*.mem.json`, `*.dna.json`) are stored in
LFS — the repo holds a pointer; the actual blob is on the LFS server.
After cloning you have pointers only.

```bash
git lfs install                # one-time per user account
git lfs pull                   # downloads every blob for current HEAD
git lfs ls-files --long | head # sanity check: shows <oid> * <path>
```

The `*` flag in `ls-files --long` means the blob is present locally;
`-` means it's a pointer only. If you see lots of `-` flags after
`git lfs pull`, your LFS bandwidth quota may be exhausted (see
Troubleshooting §T3).

---

## 3. Create the Python environment

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate
# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# install the package in editable mode
pip install -e .
pip install pytest pytest-timeout    # required by the test suite
```

Verify:

```bash
python -c "import neuroslm ; print(neuroslm.__file__)"
brian --help                        # or: python -m neuroslm.cli --help
```

---

## 4. Install the pre-commit test hook

The repo uses a **plain shell hook** at `.git/hooks/pre-commit` (no
`pre-commit` framework) that runs `pytest tests/dsl/ -m "not slow"`
before every commit and rejects the commit if anything fails.

The hook isn't tracked in the repo (it lives in `.git/`), so a fresh
clone has no hook. Add it:

```bash
# scripted install (recommended)
python scripts/install_git_hooks.py     # if present in your branch

# manual install (if no script):
cat > .git/hooks/pre-commit <<'EOF'
#!/bin/sh
if [ -f .venv/bin/python ]; then
    PYTHON=.venv/bin/python
elif [ -f .venv/Scripts/python.exe ]; then
    PYTHON=.venv/Scripts/python.exe
else
    PYTHON=python
fi
echo "Running tests before commit..."
$PYTHON -m pytest tests/dsl/ -q -m "not slow"
if [ $? -ne 0 ]; then
    echo ""
    echo "Tests failed. Commit rejected."
    exit 1
fi
echo "All tests passed. Proceeding with commit."
exit 0
EOF
chmod +x .git/hooks/pre-commit
```

Verify the hook fires:

```bash
git commit --allow-empty -m "test hook"
# you should see "Running tests before commit..." then 683+ tests run
```

---

## 5. Run pending migrations

The migration framework (`brian migrate`) tracks **versioned, idempotent**
repo transformations in `.brian/migrations.json`. After a fresh clone
some migrations may be pending — for example, if logs were committed
in the old `logs/vast/*.log` layout but never reorganized.

```bash
# show what's applied / pending / drifting
brian migrate --list

# expected (fresh clone, all clean):
#   [OK]    APPLIED       0001_logs_to_run_folders    no ops  ...

# if anything is PENDING with N > 0 ops:
brian migrate --all              # dry-run every pending migration
brian migrate --all --force      # actually apply them
```

Status legend:

| Glyph     | Kind           | Meaning |
|-----------|----------------|---------|
| `[OK]`    | `APPLIED`      | In ledger AND `plan()` returns 0 ops — nothing to do. |
| `[PEND]`  | `PENDING`      | Not in ledger AND `plan()` returns ops — `--force` to apply. |
| `[DRIFT]` | `DRIFT`        | In ledger BUT `plan()` returns ops — repo changed under us. Investigate. |
| `[NOOP]`  | `NOOP_PENDING` | Not in ledger AND `plan()` returns 0 ops — `--force` to record as applied. |

**`DRIFT` is the one to worry about**: it means somebody added a file
(e.g. a new `logs/vast/*.log`) that the already-applied migration would
have touched. Re-applying is safe (the migration is idempotent), but
investigate the source first.

---

## 6. Smoke test — repo is healthy

These three commands together prove the environment is wired up:

```bash
# (a) reference-aware LFS pruner — must finish in <5s on a normal repo
brian clean lfs

# (b) full test suite, including the meta-tools we just set up
.venv/Scripts/python.exe -m pytest tests/test_clean.py tests/test_clean_lfs.py tests/test_migrate_framework.py tests/test_migration_0001.py -q
# expected: 85 passed

# (c) DSL training smoke test (10 steps, synthetic data, CPU)
python -m neuroslm.train_dsl --arch architectures/rcc_bowtie --scale 30m_p4 --steps 10 --mode synthetic --device cpu
```

If all three return cleanly, you're done.

---

## Troubleshooting

### T1 — `git push` hangs forever

Two known causes:

**(a) A broken loose ref.** Look for empty files under `.git/refs/remotes/origin/`:

```bash
find .git/refs/remotes/origin -size 0    # Linux / macOS / Git Bash
# Windows PowerShell:
Get-ChildItem .git\refs\remotes\origin -File -Recurse | Where-Object { $_.Length -eq 0 }
```

Delete any empty file you find, then `git fetch origin --prune` to repair.

**(b) A stale credential helper popping up on every push.** This
manifests as the push hanging silently on Windows. Fix by switching
to SSH (Section 1) — SSH bypasses the credential helper entirely.

### T2 — A Personal Access Token (PAT) appears in the remote URL

If `git remote -v` shows `https://ghp_XXXX...@github.com/...`, the
PAT is **committed in plaintext** to your local `.git/config`. Rotate
the PAT immediately on github.com (Settings → Developer settings →
Tokens) and swap to SSH:

```bash
git remote set-url origin git@github.com:269652/BRIAN.git
git remote -v        # confirm "git@github.com:..." not "https://..."
```

### T3 — `git lfs pull` shows "this repository exceeded its LFS budget"

GitHub's free LFS quota is 1 GB storage + 1 GB/month bandwidth. Once
exhausted the only fixes are (a) upgrade the LFS pack on github.com,
(b) wait for the monthly reset, or (c) move large binaries off LFS.
While waiting you can still work on text files; LFS files will just
be pointer placeholders.

### T4 — `brian migrate --list` shows `DRIFT` for a migration

The migration was applied previously, but the current repo state
contains files that `plan()` would have touched. This is usually
benign (new files added since last migration). To re-apply:

```bash
brian migrate <id> --force            # applies the new ops, updates ledger
# OR, if you want to fully redo:
brian migrate <id> --rerun --force    # escape hatch: re-apply even if applied
```

### T5 — Pre-commit hook doesn't run

Check that the hook is executable (`ls -la .git/hooks/pre-commit`).
On Windows under MINGW64 / Git Bash this is automatic; under raw
PowerShell the executable bit might not be set. Re-run Section 4 to
overwrite the hook.

### T6 — `pytest` complains about missing `pytest-timeout`

```bash
pip install pytest pytest-timeout
```

This isn't pulled in automatically by `pip install -e .` because it's
a dev-only dependency.

---

## Reference: what the brian CLI offers

| Command | Purpose |
|---------|---------|
| `brian clean logs` | Dry-run delete unreferenced training logs. |
| `brian clean checkpoints` | Dry-run delete unreferenced `*.pt` files. |
| `brian clean docs` | Dry-run delete unreferenced markdown drafts. |
| `brian clean lfs` | Per-run LFS pruner (log-gated `_best` protection). |
| `brian clean all --force` | Apply all four cleaners. |
| `brian migrate --list` | Status of every migration. |
| `brian migrate <id>` | Dry-run a single migration. |
| `brian migrate <id> --force` | Apply it; record in `.brian/migrations.json`. |
| `brian migrate --all --force` | Apply every PENDING migration in order. |
| `brian lint <arch>` | Lint a `.neuro` architecture file. |
| `brian test` | Run the project test suite. |
| `brian push` | Push current branch (legacy; prefer plain `git push` now). |

All `clean` and `migrate` commands are **dry-run by default**; nothing
mutates the working tree without `--force`. The migration ledger lives
at `.brian/migrations.json` and is committed alongside the migrations
themselves, so the team shares one source of truth about what has been
applied where.
