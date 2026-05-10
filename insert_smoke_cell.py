"""Insert a temporary smoke-test cell before the ablation cell (8049a7fd)."""
import io, json

NB = "colab_run.ipynb"

SMOKE_SOURCE = """\
# 4b) SMOKE TEST — 50 steps, save+push every 10 (run this first to verify pipeline)
# ─── Temporary: delete or skip once you've confirmed it works ─────────────────────
import os, glob, subprocess

# Refresh credentials
_tok = ""
try:
    from google.colab import userdata as _ud
    _tok = (_ud.get("GITHUB") or "").strip()
except Exception:
    _tok = os.environ.get("GITHUB", "").strip()
if _tok:
    os.environ["GITHUB"] = _tok
    os.environ["GITHUB_TOKEN"] = _tok
    with open(os.path.expanduser("~/.git-credentials"), "w") as _cf:
        _cf.write("https://" + _tok + "@github.com\\n")
    subprocess.run(["git", "config", "--global", "credential.helper", "store"], check=False)
    print("Credentials refreshed (" + str(len(_tok)) + " chars)")
else:
    print("No GITHUB token — checkpoint push will be skipped")

os.environ["PYTHONUNBUFFERED"] = "1"

_SMOKE_DIR = "/content/checkpoints_smoke"
_smoke_cmd = (
    "cd /content/neuroslm && python -u -m neuroslm.train"
    " --preset " + PRESET +
    " --steps 50"
    " --batch_size " + str(BATCH_SIZE) +
    " --grad_accum " + str(GRAD_ACCUM) +
    " --ckpt_dir " + _SMOKE_DIR +
    " --device " + DEVICE +
    " --mode " + MODE +
    " --chat_ratio " + str(CHAT_RATIO) +
    " --save_every 10"
    " --log_every 10"
    " --seed 0"
    + (" --overwrite_ckpt" if OVERWRITE_CKPT else "")
)
print("=" * 60)
print("  SMOKE TEST: 50 steps | save+push every 10 steps")
print("=" * 60)
print(_smoke_cmd)
get_ipython().system(_smoke_cmd)

# Show what was saved and pushed
_ckpts = sorted(glob.glob(_SMOKE_DIR + "/*.pt"))
print("\\n" + "=" * 60)
if _ckpts:
    print("  SMOKE TEST ✓  " + str(len(_ckpts)) + " checkpoint(s) saved+pushed:")
    for _c in _ckpts:
        print("    " + os.path.basename(_c))
    print("  Pipeline OK — proceed to ablation (cell 5).")
else:
    print("  SMOKE TEST ✗  no checkpoints saved — fix errors above.")
print("=" * 60)
"""

with io.open(NB, encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Find the index of the ablation cell
ablation_idx = next(
    (i for i, c in enumerate(cells) if c.get("id") == "8049a7fd"), None
)
if ablation_idx is None:
    print("ERROR: ablation cell 8049a7fd not found")
    exit(1)

# Check if smoke cell already exists
if any(c.get("id") == "a5b7c9d1" for c in cells):
    print("Smoke cell already present — skipping")
    exit(0)

smoke_cell = {
    "id": "a5b7c9d1",
    "cell_type": "code",
    "source": SMOKE_SOURCE.splitlines(keepends=True),
    "metadata": {},
    "outputs": [],
    "execution_count": None,
}

cells.insert(ablation_idx, smoke_cell)
nb["cells"] = cells

with io.open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Inserted smoke-test cell at index {ablation_idx} (before 8049a7fd)")
