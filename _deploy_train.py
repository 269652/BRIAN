"""Direct deploy of DSL or DNA training runs with OOD mid-eval enabled.

For DSL training, reads hardware + scale variants from arch.neuro:
  - `hardware { gpu_name, num_gpus, min_reliability, min_inet_mbps,
                dist_strategy, precision }`
  - `scales { <variant>: { d_model, depth, ..., hardware? } }`

For DNA training, unfolds the DNA and reads the same config from its
embedded arch.neuro block.

Environment variables:
  DNA=<path>              path to .dna file (e.g., dna/evol/arch.dna)
  ARCH=<name>             architecture folder name (if DNA not set)
  SCALE=<name>            scale variant from arch.neuro (e.g., 300m, 1b, 7b)
  STEPS=10000             training steps (default: 40000)
  OOD_EVERY=500           OOD eval frequency (default: 500)
"""
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

BRANCH = os.environ.get("BRANCH", "arch/rcc-p4-loss-clip")
REPO_SLUG = "269652/BRIAN"
VAST_IMAGE = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"
OOD_EVERY = int(os.environ.get("OOD_EVERY", "500"))
STEPS = int(os.environ.get("STEPS", "40000"))
DNA = os.environ.get("DNA", "")
ARCH = os.environ.get("ARCH", "rcc_bowtie")
SCALE = os.environ.get("SCALE", "")
LABEL_SUFFIX = os.environ.get("LABEL_SUFFIX", "")

# ── Detect mode: DNA vs DSL ──
USE_DNA = bool(DNA)
if USE_DNA:
    mode_label = f"-dna-{Path(DNA).stem}"
else:
    mode_label = ""

LABEL = "neuroslm-full" + (f"-{LABEL_SUFFIX}" if LABEL_SUFFIX else "") \
    + mode_label + (f"-{SCALE}" if SCALE else "")

# ── Read hardware + scale from arch.neuro ──
sys.path.insert(0, str(Path(__file__).parent))
from neuroslm.dsl.training_config import load_training_config_from_arch

if USE_DNA:
    # DNA mode: read training config from embedded arch.neuro in the DNA
    from neuroslm.compiler.ribosome import RibosomeCompiler
    print(f"DNA mode: {DNA}")
    try:
        compiler = RibosomeCompiler()
        dsl_code = compiler.dna_translator.translate_from_file(str(DNA))
        import re as _re
        m = _re.search(r"\barchitecture\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{",
                      dsl_code)
        if not m:
            raise ValueError(f"Cannot extract `architecture <name>` from {DNA}")
        arch_name = m.group(1)
        ARCH_ROOT = Path("architectures") / arch_name
        if not (ARCH_ROOT / "arch.neuro").is_file():
            raise FileNotFoundError(
                f"DNA references {arch_name} but "
                f"{ARCH_ROOT}/arch.neuro not found")
        tc = load_training_config_from_arch(ARCH_ROOT)
        dna_arch_name = arch_name
    except Exception as e:
        print(f"✗ Failed to load DNA {DNA}: {e}", file=sys.stderr)
        sys.exit(1)
else:
    # DSL mode: read from architecture folder
    ARCH_ROOT = Path("architectures") / ARCH
    tc = load_training_config_from_arch(ARCH_ROOT)
    dna_arch_name = None

scale_name = SCALE or tc.scales.default
if scale_name and scale_name in tc.scales.variants:
    scale = tc.scales.variants[scale_name]
    hw = scale.hardware or tc.hardware
    print(f"scale: {scale_name} (~{scale.approx_params})  "
          f"d_model={scale.d_model} depth={scale.depth} "
          f"batch={scale.batch_size} ctx={scale.seq_len} grad_accum={scale.grad_accum}")
else:
    hw = tc.hardware
    scale = None
    print(f"no scale variant — using default training {{}} block")
print(f"hardware: gpu={hw.gpu_name} num_gpus={hw.num_gpus} "
      f"dist={hw.dist_strategy} prec={hw.precision} "
      f"reliability>{hw.min_reliability} inet>={hw.min_inet_mbps}")

# Compose the offer-search query from the hardware envelope
offer_query = (
    f"gpu_name={hw.gpu_name} num_gpus={hw.num_gpus} "
    f"rentable=true verified=true "
    f"reliability>{hw.min_reliability} disk_space>=60 "
    f"inet_down>={hw.min_inet_mbps}"
)
    # GPU RAM filter is applied as a POST-filter after the search since
    # vast.ai's CLI quoting for gpu_ram>=N is unreliable across versions.

# ── Per-scale env vars for the training script ──
scale_env = ""
if scale is not None:
    scale_env = (
        f"export SCALE={scale_name} "
        f"D_MODEL={scale.d_model} "
        f"DEPTH={scale.depth} "
        f"N_HEADS={scale.n_heads} "
        f"MAX_CTX={scale.max_ctx} "
        f"BATCH_SIZE={scale.batch_size} "
        f"SEQ_LEN={scale.seq_len} "
        f"GRAD_ACCUM={scale.grad_accum} "
    )
disk_gib = 60 if hw.num_gpus <= 2 else 120 if hw.num_gpus <= 4 else 200
launch_cmd = ("torchrun --nproc_per_node=" + str(hw.num_gpus)
               if hw.num_gpus > 1 and hw.dist_strategy != "single"
               else "python")

# ── Build training command (DNA or DSL mode) ──
if USE_DNA:
    training_cmd = f"DNA={DNA} STEPS={STEPS} OOD_EVERY={OOD_EVERY} FRESH=1 \\\n    bash scripts/vast_train_dna_loop.sh 2>&1 | tee /workspace/train.log"
    arch_name_for_log = dna_arch_name or Path(DNA).stem
else:
    training_cmd = f"ARCH={ARCH} STEPS={STEPS} OOD_EVERY={OOD_EVERY} FRESH=1 \\\n    bash scripts/vast_train_dsl_loop.sh 2>&1 | tee /workspace/train.log"
    arch_name_for_log = ARCH

ONSTART = f"""set -e
export DEBIAN_FRONTEND=noninteractive
export GITHUB='{GITHUB}' HF_TOKEN='' VAST_API_KEY='{VAST_API_KEY}'
{scale_env}
export DIST_STRATEGY={hw.dist_strategy}
export NUM_GPUS={hw.num_gpus}
export PRECISION={hw.precision}
# Expandable-segments allocator dramatically reduces fragmentation on
# long runs that mix bursty large tensors (CE backward, diff-attention
# softmax) with small tensors (genetics overlays, optimizer state) —
# fragmentation was OOMing 100M runs at step 100-220 even with chunked
# CE because the allocator couldn't find contiguous regions for the
# next backward's gradient buffer.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
    ARCH_NAME='{arch_name_for_log}' LABEL='{LABEL}' TOTAL_STEPS='{STEPS}' \\
    nohup bash scripts/log_pusher.sh > /workspace/log_pusher.log 2>&1 &
LOG_PUSHER_PID=$!

echo "── starting {'DNA' if USE_DNA else 'DSL'} training (scale={scale_name}, dist={hw.dist_strategy}, {STEPS} steps, mid-OOD every {OOD_EVERY}) ──"
{training_cmd}

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

print(f"searching offers: {offer_query}")
offers_text, _ = vastai(
    "search", "offers", offer_query,
    "-o", "dph+", "--raw", capture=True)
start = offers_text.find("[")
offers = json.loads(offers_text[start:]) if start >= 0 else []
# Post-filter by GPU RAM (in MB). 5% margin so 40 GB cards reporting
# ~40537 MB still qualify.
if hw.min_gpu_mem_gib > 0:
    min_mb = int(hw.min_gpu_mem_gib * 1024 * 0.95)
    before = len(offers)
    offers = [o for o in offers if (o.get("gpu_ram") or 0) >= min_mb]
    if before > len(offers):
        print(f"filtered {before - len(offers)}/{before} offers by gpu_ram>={min_mb}MB")
if not offers:
    sys.exit("no offers")
o = offers[0]
print(f"picked offer {o['id']} ({o['gpu_name']} x{o.get('num_gpus','?')}, ${o['dph_total']}/hr)")

print("creating instance...")
env_arg = f"-e GITHUB={GITHUB} -e HF_TOKEN= -e VAST_API_KEY={VAST_API_KEY}"
vastai("create", "instance", str(o["id"]),
       "--image", VAST_IMAGE,
       "--disk", str(disk_gib),
       "--label", LABEL,
       "--env", env_arg,
       "--onstart-cmd", ONSTART)
print(f"done (label={LABEL})")
