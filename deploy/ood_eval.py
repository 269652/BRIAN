"""Direct OOD eval deploy — bypasses bash for Windows speed.

Invoked indirectly by `brian eval ood [<ckpt> | --latest]`. Can also be
run standalone:

Usage:
    python deploy/ood_eval.py                                # last DSL ckpt
    CKPT=lfs_checkpoints/dsl_arch_step9000.pt python deploy/ood_eval.py
    ARGV: optional positional checkpoint path (overrides CKPT env var).

Reads checkpoint path from (priority):
    1. argv[1]
    2. CKPT env var
    3. highest dsl_arch_*_step*.pt under lfs_checkpoints/
"""
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

env_path = Path(".env")
for line in env_path.read_text().splitlines():
    if line.startswith(("VAST_AI=", "GITHUB_PAT=", "VAST_API_KEY=", "GITHUB=")):
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

VAST_API_KEY = os.environ.get("VAST_API_KEY") or os.environ.get("VAST_AI")
GITHUB = os.environ.get("GITHUB") or os.environ.get("GITHUB_PAT")
assert VAST_API_KEY and GITHUB, "missing VAST_API_KEY or GITHUB in env"


def _pick_default_ckpt() -> str:
    """Highest-step dsl_arch_*_step*.pt under lfs_checkpoints/."""
    candidates = []
    for p in glob.glob("lfs_checkpoints/dsl_arch_*step*.pt"):
        m = re.search(r"_step(\d+)\.pt$", p)
        if m:
            candidates.append((int(m.group(1)), p.replace("\\", "/")))
    candidates.sort()
    return candidates[-1][1] if candidates else "lfs_checkpoints/dsl_arch_step10000.pt"


CKPT = (sys.argv[1] if len(sys.argv) > 1 else
        os.environ.get("CKPT") or _pick_default_ckpt())
BRANCH = os.environ.get("BRANCH", "arch/rcc-p4-loss-clip")
_run_tag = re.search(r"_step(\d+)\.pt$", CKPT)
_step = _run_tag.group(1) if _run_tag else "?"
ROLE_TAG = os.environ.get("ROLE_TAG", f"dsl-step{_step}")
REPO_SLUG = "269652/BRIAN"
VAST_IMAGE = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"
OOD_DIR = "logs/vast/benchmarks/ood"
OUTPUT_FILE = f"{OOD_DIR}/ood_results_{ROLE_TAG}.json"
MAX_OOD_WINDOWS = 200
BATCH_SIZE = 4

ONSTART = f"""set -e
export DEBIAN_FRONTEND=noninteractive
export GITHUB='{GITHUB}' HF_TOKEN='' VAST_API_KEY='{VAST_API_KEY}'
date -u +"ood-eval boot @ %Y-%m-%dT%H:%M:%SZ"

(command -v git >/dev/null 2>&1 && command -v git-lfs >/dev/null 2>&1) \\
    || (apt-get update -y && apt-get install -y git git-lfs)
git lfs install --skip-smudge

mkdir -p /workspace && cd /workspace
echo "── cloning {BRANCH} ──"
GIT_LFS_SKIP_SMUDGE=1 git clone --branch '{BRANCH}' --single-branch \\
    "https://x-access-token:${{GITHUB}}@github.com/{REPO_SLUG}.git" brian
cd brian
echo "── pulling LFS object {CKPT} ──"
git lfs pull --include='{CKPT}'

echo "── bootstrap (pip deps) ──"
SKIP_LFS_RESUME=1 bash scripts/vast_bootstrap.sh

echo "── running OOD eval ──"
mkdir -p "{OOD_DIR}"
python -u brian_ood_test.py \\
  --checkpoint "{CKPT}" \\
  --max_ood_windows {MAX_OOD_WINDOWS} \\
  --batch_size {BATCH_SIZE} \\
  --output "{OUTPUT_FILE}" 2>&1 | tee "{OUTPUT_FILE}.log"

echo "── pushing results ──"
git config user.email "ood-eval@vast.local"
git config user.name "ood-eval-bot"
git add "{OUTPUT_FILE}" "{OUTPUT_FILE}.log" 2>/dev/null || true
git commit -m "ood eval ({ROLE_TAG}) on {BRANCH}" 2>/dev/null || echo "nothing to commit"
PUSH_URL="https://x-access-token:${{GITHUB}}@github.com/{REPO_SLUG}.git"
for i in 1 2 3 4 5; do
    if git -c credential.helper= push "${{PUSH_URL}}" {BRANCH} 2>&1 | grep -q "{BRANCH} -> {BRANCH}"; then
        echo "✓ pushed"; break
    fi
    git -c credential.helper= fetch "${{PUSH_URL}}" {BRANCH}
    git rebase FETCH_HEAD || true
    sleep 5
done

echo "── self-destroy ──"
pip install -q vastai 2>&1 | tail -3 || true
vastai set api-key "$VAST_API_KEY" 2>&1 || true
SELF_ID="${{INSTANCE_ID:-}}"
if [ -z "$SELF_ID" ]; then
    SELF_ID=$(vastai show instances --raw 2>/dev/null | python3 -c "
import sys, json
for i in json.load(sys.stdin):
    if (i.get('label') or '').startswith('neuroslm-ood-'):
        print(i.get('id','')); break")
fi
[ -n "$SELF_ID" ] && yes y | vastai destroy instance "$SELF_ID" 2>&1
echo "done"
"""

# Call vastai as a subprocess (each call gets a fresh argparse instance).
VASTAI_EXE = Path(".venv-2/Scripts/vastai.exe")
if not VASTAI_EXE.is_file():
    VASTAI_EXE = "vastai"

def vastai(*args, capture=False):
    cmd = [str(VASTAI_EXE)] + list(args)
    if capture:
        r = subprocess.run(cmd, capture_output=True, text=True,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        return r.stdout, r.returncode
    return subprocess.call(cmd)

# Set the api key
print("setting api key...")
vastai("set", "api-key", VAST_API_KEY)

# Search offers — restrict to known-good A100 SXM4 only.
# PCIE A100s + cheap MIG slices had "loading" hangs (no container after
# 30 min) on offer 38965155, which had to be destroyed manually.
print("searching offers (A100_SXM4 only, reliability>0.995, min 40GB)...")
offers_text, rc = vastai(
    "search", "offers",
    "gpu_name=A100_SXM4 num_gpus=1 rentable=true verified=true "
    "reliability>0.995 disk_space>=60 inet_down>=200",
    "-o", "dph+", "--raw", capture=True)
# Strip any banner before the JSON
start = offers_text.find("[")
offers = json.loads(offers_text[start:]) if start >= 0 else []
# Post-filter by GPU RAM (in MB). 5% margin so 40 GB cards reporting
# ~40537 MB qualify; rejects 16-20 GB MIG slices that look like A100s.
MIN_GPU_MB = int(40 * 1024 * 0.95)
before = len(offers)
offers = [o for o in offers if (o.get("gpu_ram") or 0) >= MIN_GPU_MB]
if before > len(offers):
    print(f"  filtered {before - len(offers)}/{before} offers by gpu_ram>={MIN_GPU_MB} MB")
if not offers:
    print("no offers matching SXM4 + reliability + 40GB filter!"); sys.exit(1)
offer_id = offers[0]["id"]
print(f"picked offer {offer_id} (${offers[0]['dph_total']}/hr, "
      f"{offers[0]['gpu_name']}, {offers[0].get('gpu_ram','?')} MB)")

# Create the instance
print("creating instance...")
env_arg = f"-e GITHUB={GITHUB} -e HF_TOKEN= -e VAST_API_KEY={VAST_API_KEY}"
vastai("create", "instance", str(offer_id),
       "--image", VAST_IMAGE,
       "--disk", "60",
       "--label", f"neuroslm-ood-{ROLE_TAG}",
       "--env", env_arg,
       "--onstart-cmd", ONSTART)
print("done")
