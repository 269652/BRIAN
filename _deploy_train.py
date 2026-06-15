"""Direct deploy of DSL or DNA training runs with OOD mid-eval enabled.

For DSL training, reads hardware + scale variants from arch.neuro:
  - `hardware { gpu_name, num_gpus, min_reliability, min_inet_mbps,
                dist_strategy, precision }`
  - `scales { <variant>: { d_model, depth, ..., hardware? } }`

For DNA training, unfolds the DNA and reads the same config from its
embedded arch.neuro block.

Source of truth (precedence, highest wins):
  1. Explicit env vars: ``DNA=...`` or ``ARCH=...``
  2. ``brian.toml`` (``[current].dna`` if set, else ``[current].arch``)
  3. Built-in fallback ``ARCH=rcc_bowtie``

This means a one-line edit to ``brian.toml`` retargets every deploy —
without touching this file. The env vars stay supported for CI / one-
off overrides.

Environment variables:
  DNA=<path>              path to .dna file (e.g., dna/evol/arch.dna)
                          — overrides brian.toml
  ARCH=<name>             architecture folder name
                          — overrides brian.toml
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

# ── Checkpoint / log cadence ──
# Defaults restored after H24 (instance 41031063) lost all checkpoints
# when the box self-destroyed before the (then end-of-training-only)
# push. The legacy ``train.py`` pushed after every save; the DSL
# trainer rewrite silently dropped that. We now default to a
# 500-step save+push cadence so a vast.ai outage can never eat more
# than ~500 steps of progress. ``cli._deploy_dsl`` /
# ``cli._deploy_dna`` populate these from ``brian.toml [defaults]``.
# Set ``PUSH_EVERY=0`` to disable Git LFS pushes entirely (local
# saves only).
LOG_EVERY = int(os.environ.get("LOG_EVERY", "20"))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "500"))
PUSH_EVERY = int(os.environ.get("PUSH_EVERY", "500"))

# ── Resolve current arch/DNA from brian.toml (env vars still win) ──
sys.path.insert(0, str(Path(__file__).parent))
from neuroslm.project_config import load_project_config

_proj = load_project_config()
_env_arch = os.environ.get("ARCH", "")
# BRIAN_SOURCE_DNA is set by ``cli._deploy_dna`` after it pre-compiles
# the DNA via ``prepare_run_workspace``. It carries the ORIGINAL .dna
# path for labelling / logging only — the actual arch tree the box
# trains from lives at ARCH=<workspace> (a .neuro/arch/temp path).
# Legacy ``DNA=...`` env var: tolerated for back-compat but treated as
# pure metadata, NEVER re-compiled here (that's cli.cmd_deploy's job
# per the canonical-pipeline contract).
_source_dna = os.environ.get("BRIAN_SOURCE_DNA", "") or os.environ.get("DNA", "")

if _env_arch:
    ARCH = _env_arch
elif _proj.is_dna_mode:
    # brian.toml says DNA mode but ``cli._deploy_dna`` didn't run
    # (operator invoked _deploy_train.py directly). This is no longer
    # a supported entry point — the canonical pipeline requires the
    # cli to pre-compile the workspace. Bail with a clear message.
    print(
        f"✗ brian.toml selects DNA mode (dna={_proj.dna}) but ARCH "
        f"env var is not set.\n"
        f"  Run via the CLI: brian deploy --dna {_proj.dna}\n"
        f"  (the CLI pre-compiles the DNA into .neuro/arch/temp/ and "
        f"sets ARCH for this script).",
        file=sys.stderr,
    )
    sys.exit(1)
else:
    # arch field is "architectures/<name>"; the legacy ARCH env var
    # expects just the leaf name (the scripts join "architectures/$ARCH").
    ARCH = _proj.arch.split("/")[-1] if "/" in _proj.arch else _proj.arch
    print(f"[brian.toml] DSL mode: ARCH={ARCH}")

SCALE = os.environ.get("SCALE", "")
LABEL_SUFFIX = os.environ.get("LABEL_SUFFIX", "")

# ── Detect mode purely from BRIAN_SOURCE_DNA (labelling only) ──
USE_DNA = bool(_source_dna)
if USE_DNA:
    mode_label = f"-dna-{Path(_source_dna).stem}"
else:
    mode_label = ""

LABEL = "neuroslm-full" + (f"-{LABEL_SUFFIX}" if LABEL_SUFFIX else "") \
    + mode_label + (f"-{SCALE}" if SCALE else "")

# ── Read hardware + scale from arch.neuro (one path for DNA + DSL) ──
# After the canonical-pipeline refactor (2026-06-12), ``ARCH`` always
# points at a directory containing ``arch.neuro`` — either an existing
# ``architectures/<name>/`` (DSL mode) or the prepared
# ``.neuro/arch/temp/`` workspace that cli._deploy_dna unfolded from a
# DNA snapshot. Either way, ``load_training_config_from_arch`` is the
# only call needed; this script no longer imports the DNA compiler and
# therefore no longer pulls torch as a transitive dependency.
from neuroslm.dsl.training_config import load_training_config_from_arch

# ARCH may be either a bare name ("current" / "master") or a workspace
# path (e.g. "C:/.../.neuro/arch/temp"). If it's a path that exists, use
# it directly; otherwise treat it as a name under architectures/.
_arch_as_path = Path(ARCH)
if _arch_as_path.is_dir() and (_arch_as_path / "arch.neuro").is_file():
    ARCH_ROOT = _arch_as_path
    # For logging only — strip drive prefix on Windows to keep the
    # banner short.
    arch_display = _arch_as_path.name
else:
    ARCH_ROOT = Path("architectures") / ARCH
    arch_display = ARCH

if not (ARCH_ROOT / "arch.neuro").is_file():
    print(f"✗ {ARCH_ROOT}/arch.neuro not found", file=sys.stderr)
    sys.exit(1)
tc = load_training_config_from_arch(ARCH_ROOT)
dna_arch_name = Path(_source_dna).stem if USE_DNA else None
if USE_DNA:
    print(f"DNA mode (canonical pipeline): source={_source_dna} "
          f"workspace={ARCH_ROOT}")

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

# ── Build training command ──
# After the canonical-pipeline refactor, both DNA-mode and DSL-mode
# runs use the same on-box wrapper: ``vast_train_dna_loop.sh`` if a
# source DNA is present (so on-box evolution can rewrite the snapshot
# back to BRIAN_SOURCE_DNA), otherwise the DSL loop. Both wrappers
# receive ``ARCH=<arch-or-workspace>`` and read the prepared tree —
# neither one re-compiles a DNA on the box.
if USE_DNA:
    training_cmd = (
        f"ARCH={ARCH} BRIAN_SOURCE_DNA={_source_dna} "
        f"STEPS={STEPS} OOD_EVERY={OOD_EVERY} "
        f"LOG_EVERY={LOG_EVERY} SAVE_EVERY={SAVE_EVERY} "
        f"PUSH_EVERY={PUSH_EVERY} FRESH=1 \\\n"
        f"    bash scripts/vast_train_dna_loop.sh 2>&1 | tee /workspace/train.log"
    )
    arch_name_for_log = dna_arch_name or arch_display
else:
    training_cmd = (
        f"ARCH={ARCH} STEPS={STEPS} OOD_EVERY={OOD_EVERY} "
        f"LOG_EVERY={LOG_EVERY} SAVE_EVERY={SAVE_EVERY} "
        f"PUSH_EVERY={PUSH_EVERY} FRESH=1 \\\n"
        f"    bash scripts/vast_train_dsl_loop.sh 2>&1 | tee /workspace/train.log"
    )
    arch_name_for_log = ARCH

ONSTART = f"""set -e
export DEBIAN_FRONTEND=noninteractive
export GITHUB='{GITHUB}' HF_TOKEN='' VAST_API_KEY='{VAST_API_KEY}'
{scale_env}
export DIST_STRATEGY={hw.dist_strategy}
export NUM_GPUS={hw.num_gpus}
export PRECISION={hw.precision}
# DSL_ARCH_LABEL feeds neuroslm.train_dsl module-level _ARCH_LABEL which
# becomes the third component of the per-run checkpoint subdir name
# (lfs_checkpoints/<RUN_ID>_<GIT>_<ARCH_LABEL>/step<N>.pt). Without
# this export the box would default to "run" and lose per-deploy
# traceability. Pinned by tests/test_checkpoint_path_layout.py.
export DSL_ARCH_LABEL='{LABEL}'
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
# Recursive scan: H24+ uses per-run subdirs
# ``lfs_checkpoints/<RUN_ID>_<GIT>_<ARCH>/step<N>.pt`` so a flat
# ``lfs_checkpoints/dsl_arch_*.pt`` glob misses everything.
# Regression-pinned by tests/test_checkpoint_path_layout.py
# ::TestDeployPushGlob::test_push_glob_pattern_is_recursive.
find lfs_checkpoints -type f -name '*.pt' 2>/dev/null | while read -r ckpt; do
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

# Use the on-PATH ``vastai`` binary. The canonical venv (./.venv per
# CLAUDE.md §13) installs it into ``.venv/Scripts/`` — when this
# script is launched via ``cli._deploy_dna`` that path is on PATH
# automatically. The legacy sibling-venv lookup was removed in the
# 2026-06-12 canonical-pipeline refactor.
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
