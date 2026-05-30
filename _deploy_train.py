"""Direct deploy of the 10k DSL training run with OOD mid-eval enabled."""
import json
import os
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
assert VAST_API_KEY and GITHUB

BRANCH = "arch/rcc-p4-loss-clip"
REPO_SLUG = "269652/BRIAN"
VAST_IMAGE = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"
OOD_EVERY = int(os.environ.get("OOD_EVERY", "3000"))
STEPS = int(os.environ.get("STEPS", "10000"))
# Per-run label suffix so parallel runs don't collide on labels +
# their per-run-id checkpoint filenames stay separable in git.
LABEL_SUFFIX = os.environ.get("LABEL_SUFFIX", "")
LABEL = "neuroslm-full" + (f"-{LABEL_SUFFIX}" if LABEL_SUFFIX else "")

ONSTART = f"""set -e
export DEBIAN_FRONTEND=noninteractive
export GITHUB='{GITHUB}' HF_TOKEN='' VAST_API_KEY='{VAST_API_KEY}'
date -u +"train boot @ %Y-%m-%dT%H:%M:%SZ"

(command -v git >/dev/null 2>&1 && command -v git-lfs >/dev/null 2>&1) \\
    || (apt-get update -y && apt-get install -y git git-lfs)
git lfs install --skip-smudge

mkdir -p /workspace && cd /workspace
echo "── cloning {BRANCH} ──"
GIT_LFS_SKIP_SMUDGE=1 git clone --branch '{BRANCH}' --single-branch \\
    "https://x-access-token:${{GITHUB}}@github.com/{REPO_SLUG}.git" brian
cd brian

echo "── bootstrap (pip deps, SKIP_LFS_RESUME=1) ──"
SKIP_LFS_RESUME=1 bash scripts/vast_bootstrap.sh

echo "── starting log-pusher (background) ──"
INSTANCE_ID="$(hostname)" PUSH_INTERVAL=300 \\
    BRANCH='{BRANCH}' REPO_SLUG='{REPO_SLUG}' \\
    nohup bash scripts/log_pusher.sh > /workspace/log_pusher.log 2>&1 &
LOG_PUSHER_PID=$!

echo "── starting DSL training (10k steps, mid-OOD every {OOD_EVERY}) ──"
ARCH=rcc_bowtie STEPS={STEPS} OOD_EVERY={OOD_EVERY} FRESH=1 \\
    bash scripts/vast_train_dsl_loop.sh 2>&1 | tee /workspace/train.log

echo "── stopping log-pusher ──"
kill $LOG_PUSHER_PID 2>/dev/null || true
sleep 2

echo "── final log push ──"
SOURCE_LOG=/workspace/train.log INSTANCE_ID="$(hostname)" PUSH_INTERVAL=1 \\
    BRANCH='{BRANCH}' REPO_SLUG='{REPO_SLUG}' \\
    timeout 30 bash scripts/log_pusher.sh 2>&1 | head -10 || echo "log push timeout"

echo "── pushing checkpoints + OOD mid-eval JSONs ──"
cd /workspace/brian
git config user.email "vast-train@brian.local"
git config user.name "vast-train"
for ckpt in lfs_checkpoints/dsl_arch_*.pt; do
    [ -e "$ckpt" ] || continue
    git add "$ckpt" 2>/dev/null || true
done
git add logs/vast/benchmarks/ood/ood_mid_*.json 2>/dev/null || true
if ! git diff --cached --quiet 2>/dev/null; then
    git commit -m "chkpt+mid-ood: training run @ $(date -u +%Y-%m-%dT%H:%M:%SZ)" >/dev/null 2>&1 || true
    PUSH_URL="https://x-access-token:${{GITHUB}}@github.com/{REPO_SLUG}.git"
    timeout 600 git push "$PUSH_URL" "HEAD:{BRANCH}" 2>&1 | sed "s#${{GITHUB}}#***#g" || true
fi

echo "── self-destroy ──"
pip install -q vastai 2>&1 | tail -3 || true
vastai set api-key "$VAST_API_KEY" 2>&1 || true
SELF_ID="${{INSTANCE_ID:-}}"
if [ -z "$SELF_ID" ]; then
    SELF_ID=$(vastai show instances --raw 2>/dev/null | python3 -c "
import sys, json
for i in json.load(sys.stdin):
    if (i.get('label') or '').startswith('neuroslm-full'):
        print(i.get('id','')); break")
fi
[ -n "$SELF_ID" ] && yes y | vastai destroy instance "$SELF_ID" 2>&1
echo "done"
"""

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

print("setting api key...")
vastai("set", "api-key", VAST_API_KEY)

print("searching A100 offers...")
offers_text, _ = vastai(
    "search", "offers",
    "gpu_name in [A100_SXM4,A100_PCIE,A100_SXM,A100X] num_gpus=1 "
    "rentable=true verified=true reliability>0.99 disk_space>=60",
    "-o", "dph+", "--raw", capture=True)
start = offers_text.find("[")
offers = json.loads(offers_text[start:]) if start >= 0 else []
if not offers:
    sys.exit("no offers")
o = offers[0]
print(f"picked offer {o['id']} ({o['gpu_name']}, ${o['dph_total']}/hr)")

print("creating instance...")
env_arg = f"-e GITHUB={GITHUB} -e HF_TOKEN= -e VAST_API_KEY={VAST_API_KEY}"
vastai("create", "instance", str(o["id"]),
       "--image", VAST_IMAGE,
       "--disk", "60",
       "--label", LABEL,
       "--env", env_arg,
       "--onstart-cmd", ONSTART)
print(f"done (label={LABEL})")
