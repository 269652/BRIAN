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
    if line.startswith(("VAST_AI=", "GITHUB_PAT=", "VAST_API_KEY=",
                         "GITHUB=", "HF_TOKEN=", "HF_REPO_ID=")):
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

# ── Checkpoint push backend (2026-06-15) ──
# Run 41063959 hung at step 500 because the synchronous Git LFS push
# of a 569 MB object inside the training loop raced the background
# ``log_pusher.sh`` (both try to ``git push origin master`` to the
# same repo at the same time). Switching the default to HuggingFace
# Hub avoids the race entirely — HF uses a single sync HTTPS PUT.
#
# These three propagate via the ONSTART export block to the on-box
# trainer; ``vast_train_dsl_loop.sh`` / ``vast_train_dna_loop.sh``
# forward them as ``--push_backend`` to ``python -m neuroslm.train_dsl``.
# ``cli._deploy_dsl`` / ``cli._deploy_dna`` populate them from
# ``brian.toml [defaults] push_backend`` / ``hf_repo_id``.
HF_TOKEN = os.environ.get("HF_TOKEN", "")
CHECKPOINT_PUSH_BACKEND = os.environ.get(
    "CHECKPOINT_PUSH_BACKEND", "hf")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "moritzroessler/BRIAN")

# ── Pre-flight: warn loudly if the chosen backend lacks credentials ──
# The on-box pusher fails open (prints + skips, never crashes the run),
# so a missing token would otherwise only surface in the training log
# 500+ steps in. Catch it here so the user can fix the .env file
# BEFORE renting a $1.50/hr GPU that produces non-pushable artefacts.
if CHECKPOINT_PUSH_BACKEND == "hf" and not HF_TOKEN:
    print(
        "⚠ HF_TOKEN is empty but push_backend='hf'. The on-box "
        "trainer will SAVE checkpoints locally but SKIP the HF Hub "
        "upload (and the box may self-destruct before you can rsync "
        "them). Fix one of:\n"
        "    1. Add HF_TOKEN=hf_... to .env "
        "(see .env.example for the auth chain), or\n"
        "    2. Set push_backend = 'lfs' in brian.toml [defaults] "
        "to fall back to Git LFS, or\n"
        "    3. Set push_backend = 'none' to disable remote push.\n"
        "    Continuing in 3s — Ctrl-C to abort.",
        file=sys.stderr,
        flush=True,
    )
    import time as _t
    _t.sleep(3)

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
        f"PUSH_EVERY={PUSH_EVERY} "
        f"CHECKPOINT_PUSH_BACKEND={CHECKPOINT_PUSH_BACKEND} "
        f"HF_REPO_ID={HF_REPO_ID} FRESH=1 \\\n"
        f"    bash scripts/vast_train_dna_loop.sh 2>&1 | tee /workspace/train.log"
    )
    arch_name_for_log = dna_arch_name or arch_display
else:
    training_cmd = (
        f"ARCH={ARCH} STEPS={STEPS} OOD_EVERY={OOD_EVERY} "
        f"LOG_EVERY={LOG_EVERY} SAVE_EVERY={SAVE_EVERY} "
        f"PUSH_EVERY={PUSH_EVERY} "
        f"CHECKPOINT_PUSH_BACKEND={CHECKPOINT_PUSH_BACKEND} "
        f"HF_REPO_ID={HF_REPO_ID} FRESH=1 \\\n"
        f"    bash scripts/vast_train_dsl_loop.sh 2>&1 | tee /workspace/train.log"
    )
    arch_name_for_log = ARCH

ONSTART = f"""set -eo pipefail
export DEBIAN_FRONTEND=noninteractive
export GITHUB='{GITHUB}' HF_TOKEN='{HF_TOKEN}' VAST_API_KEY='{VAST_API_KEY}'
export CHECKPOINT_PUSH_BACKEND='{CHECKPOINT_PUSH_BACKEND}'
export HF_REPO_ID='{HF_REPO_ID}'
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
# LOG_EVERY makes the pusher step-driven: it polls every 30s and pushes
# only when the trainer crosses a LOG_EVERY-step boundary, so log
# pushes line up exactly with the rows printed in the live log
# (no more "push fires mid-step at a random wall-clock second").
# PUSH_INTERVAL is kept as a safety fallback when LOG_EVERY=0.
INSTANCE_ID="$(hostname)" LOG_EVERY={LOG_EVERY} PUSH_INTERVAL=300 \\
    BRANCH='{BRANCH}' REPO_SLUG='{REPO_SLUG}' \\
    ARCH_NAME='{arch_name_for_log}' LABEL='{LABEL}' TOTAL_STEPS='{STEPS}' \\
    nohup bash scripts/log_pusher.sh > /workspace/log_pusher.log 2>&1 &
LOG_PUSHER_PID=$!
# Echo the pid + initial liveness so `brian logs <id>` makes it clear
# the background pusher actually launched (without this, a crash on
# startup is silent — bug 41084160 was undiagnosable for hours). The
# pusher itself drops marker files at /workspace/log_pusher_alive,
# log_pusher_last_push_OK, log_pusher_last_push_FAIL that the user
# can list via `vastai execute <id> 'ls -la /workspace/log_pusher_*'`
# to detect health/failure mode without SSH (which vast.ai often
# firewalls from the operator's network).
echo "    log_pusher pid=$LOG_PUSHER_PID"
sleep 2  # give it a moment to write its first marker
if kill -0 "$LOG_PUSHER_PID" 2>/dev/null; then
    echo "    log_pusher: still alive after 2s ✓"
else
    echo "    log_pusher: ✗ DIED on startup — log_pusher.log tail:"
    tail -20 /workspace/log_pusher.log 2>/dev/null \\
        | sed 's/^/      /' || echo "      (no log_pusher.log yet)"
fi

echo "── starting {'DNA' if USE_DNA else 'DSL'} training (scale={scale_name}, dist={hw.dist_strategy}, {STEPS} steps, mid-OOD every {OOD_EVERY}) ──"
# Disable -e around the training pipe so we can capture the LEFT-side
# exit code via ${{PIPESTATUS[0]}} — `tee` ALWAYS exits 0 and would
# otherwise mask the training crash (the H24 / 41031063 failure mode).
# pipefail (set above) makes this defensive, but PIPESTATUS is the
# unambiguous source of truth.
set +e
{training_cmd}
TRAIN_RC=${{PIPESTATUS[0]}}
set -e
echo "── training exited with code $TRAIN_RC ──"

echo "── stopping log-pusher ──"
kill $LOG_PUSHER_PID 2>/dev/null || true
sleep 2

# ── Final log push — GATES the self-destroy ──
# Hard contract (operator rule, 2026-06-15): the on-box script MUST
# write the final log to origin BEFORE any ``vastai destroy``. If the
# push fails we leave the instance alive — paying $0.73/h for an extra
# hour is cheap insurance against losing the crash trace. Locked by
# tests/test_deploy_failure_safety.py::TestSelfDestroyIsGatedOnLogPush.
#
# ONESHOT=1: log_pusher.sh runs a single commit+push and exits with
# the REAL status. We previously piped through ``| head -30`` which
# always closed the pipe → SIGPIPE=141 → false-positive trip of the
# gate even on successful pushes (instance 41048619 trace 2026-06-15).
# Locked by ::TestDeployUsesOneshotForFinalLog.
echo "── final log push (GATES self-destroy) ──"
LOG_PUSH_RC=0
ONESHOT=1 SOURCE_LOG=/workspace/train.log INSTANCE_ID="$(hostname)" \\
    BRANCH='{BRANCH}' REPO_SLUG='{REPO_SLUG}' \\
    ARCH_NAME='{arch_name_for_log}' LABEL='{LABEL}' TOTAL_STEPS='{STEPS}' \\
    timeout 180 bash scripts/log_pusher.sh || LOG_PUSH_RC=$?
if [ "$LOG_PUSH_RC" -ne 0 ]; then
    echo "✗ FINAL LOG PUSH FAILED (rc=$LOG_PUSH_RC) — keeping instance alive"
    echo "  for forensics. Use 'brian ps' to find the id, then"
    echo "  'brian destroy <id>' to clean up. Training rc was $TRAIN_RC."
    exit 1
fi
echo "✓ final log pushed"

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
    git commit -m "chkpt+mid-ood: training run @ $(date -u +%Y-%m-%dT%H:%M:%SZ) (rc=$TRAIN_RC)" >/dev/null 2>&1 || true
    PUSH_URL="https://x-access-token:${{GITHUB}}@github.com/{REPO_SLUG}.git"
    timeout 600 git push "$PUSH_URL" "HEAD:{BRANCH}" 2>&1 | sed "s#${{GITHUB}}#***#g" || true
fi

# ── Gated self-destroy ──
# TRAIN_RC == 0 → destroy immediately (don't pay for an idle GPU).
# TRAIN_RC != 0 → KEEP_ALIVE_ON_FAIL minutes of forensic window
#                 (default 60), THEN destroy. Operator can SSH in
#                 during that window to pull state or inspect.
# KEEP_ALIVE_ON_FAIL == 0 → disable auto-destroy on failure entirely
#                            (operator must `brian destroy <id>`).
KEEP_ALIVE_ON_FAIL="${{KEEP_ALIVE_ON_FAIL:-60}}"
if [ "$TRAIN_RC" -eq 0 ]; then
    echo "── training succeeded → self-destroy ──"
elif [ "$KEEP_ALIVE_ON_FAIL" -eq 0 ]; then
    echo "── training FAILED (rc=$TRAIN_RC), KEEP_ALIVE_ON_FAIL=0 → no auto-destroy ──"
    echo "  Instance will remain until 'brian destroy <id>'."
    exit 1
else
    echo "── training FAILED (rc=$TRAIN_RC) → keep-alive ${{KEEP_ALIVE_ON_FAIL}} min then self-destroy ──"
    sleep $((KEEP_ALIVE_ON_FAIL * 60))
    echo "── keep-alive window elapsed → self-destroy ──"
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


# ─────────────────────────────────────────────────────────────────────
# Boot watchdog (locked by tests/test_deploy_failure_safety.py)
# ─────────────────────────────────────────────────────────────────────
# When ``vastai create instance`` succeeds the CLI prints a Python-repr
# dict with the new contract id. The host may then silently fail to
# bring up the container (exactly what happened to 41045637 on
# 2026-06-15: vast accepted the create, we got an id, no container
# was ever scheduled). The watchdog polls the REST API until the
# instance reports ``actual_status == "running"`` or a clear failure.

import re as _re

_CONTRACT_RE = _re.compile(r"['\"]new_contract['\"]\s*:\s*(\d+)")


def _parse_new_contract_id(out):
    """Extract ``new_contract`` int from ``vastai create instance`` output.

    Returns None when no contract id is present (typical for
    ``success=False`` host-unavailable responses)."""
    if not out:
        return None
    m = _CONTRACT_RE.search(out)
    return int(m.group(1)) if m else None


def _default_status_fn(instance_id):
    """Look up vast.ai instance status via REST API.

    Returns one of: ``"running"``, ``"loading"``, ``"exited"``,
    ``"stopped"``, ``"scheduling"``, ``"gone"``, or whatever vast
    reports verbatim. ``"gone"`` is our synthetic value for the
    ``{"instances": null}`` response (contract destroyed / never
    existed)."""
    import urllib.request, urllib.error
    import json as _json
    url = f"https://console.vast.ai/api/v0/instances/{instance_id}/"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {VAST_API_KEY}",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        body = r.read().decode("utf-8", errors="replace")
    data = _json.loads(body)
    inst = data.get("instances")
    if inst is None:
        # Contract garbage-collected — host never came up or destroyed.
        return "gone"
    if isinstance(inst, list):  # defensive: not the documented shape
        inst = inst[0] if inst else None
        if inst is None:
            return "gone"
    return inst.get("actual_status") or "?"


def _wait_for_instance_ready(instance_id, timeout=600, poll_interval=15,
                              status_fn=None):
    """Poll until the instance is ``running`` or a terminal state.

    Returns 0 on success, 1 on timeout, 2 on terminal-state-before-ready.
    Transient ``status_fn`` exceptions are tolerated for up to a few
    consecutive failures so a network blip doesn't kill the watch.
    """
    import time
    if status_fn is None:
        status_fn = _default_status_fn

    TERMINAL = {"exited", "stopped", "destroyed", "gone"}
    deadline = time.time() + timeout
    last_status = "?"
    transient_streak = 0
    MAX_TRANSIENT = 3
    while time.time() < deadline:
        try:
            status = status_fn(instance_id)
            transient_streak = 0
        except Exception as exc:
            transient_streak += 1
            print(f"  [watchdog] transient API error ({transient_streak}/"
                  f"{MAX_TRANSIENT}): {exc!r}", flush=True)
            if transient_streak >= MAX_TRANSIENT:
                print(f"✗ instance {instance_id} watchdog: "
                      f"{MAX_TRANSIENT} consecutive API failures, "
                      f"giving up.", flush=True)
                return 1
            time.sleep(poll_interval)
            continue

        last_status = status
        if status == "running":
            print(f"✓ instance {instance_id} ready (status=running)",
                  flush=True)
            return 0
        if status in TERMINAL:
            print(f"✗ instance {instance_id} reached terminal state "
                  f"'{status}' before ever becoming running — host "
                  f"failed to boot the container.", flush=True)
            return 2

        print(f"  [watchdog] instance {instance_id} status={status}, "
              f"waiting {poll_interval}s...", flush=True)
        time.sleep(poll_interval)

    print(f"✗ instance {instance_id} did not reach 'running' within "
          f"{timeout}s (last status={last_status}).", flush=True)
    return 1


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
create_out, create_rc = vastai(
    "create", "instance", str(o["id"]),
    "--image", VAST_IMAGE,
    "--disk", str(disk_gib),
    "--label", LABEL,
    "--env", env_arg,
    "--onstart-cmd", ONSTART,
    capture=True,
)
print(create_out, end="" if create_out.endswith("\n") else "\n")
if create_rc != 0:
    sys.exit(f"✗ vastai create failed (rc={create_rc}); aborting deploy.")

# ── Boot watchdog ──
# vast accepting the create call doesn't mean the container will boot.
# Instance 41045637 (2026-06-15) is the canonical example: contract id
# returned, container never created, log-pusher never started, no
# checkpoints ever produced. The watchdog catches this BEFORE the
# operator walks away thinking training is in progress.
new_id = _parse_new_contract_id(create_out)
if new_id is None:
    sys.exit(
        f"✗ vastai create returned no contract id (output above) — "
        f"the host probably failed to schedule. Try `brian deploy` again."
    )
print(f"new_contract={new_id} (label={LABEL}) — watching for boot...")

# Operator override knobs. ``BOOT_TIMEOUT_SEC`` is generous (10 min)
# because a cold pytorch image pull on a slow host can take 5+ min.
boot_timeout = int(os.environ.get("BOOT_TIMEOUT_SEC", "600"))
boot_poll = int(os.environ.get("BOOT_POLL_SEC", "15"))

watch_rc = _wait_for_instance_ready(
    instance_id=new_id,
    timeout=boot_timeout,
    poll_interval=boot_poll,
)
if watch_rc != 0:
    # Best-effort: tell vast to destroy the zombie contract so we
    # don't keep paying. We tolerate failure here — if the contract
    # is already gone the destroy call will no-op.
    print(f"✗ instance {new_id} failed to boot — attempting to "
          f"destroy zombie contract to stop billing...", flush=True)
    try:
        vastai("destroy", "instance", str(new_id), capture=True)
    except Exception as exc:  # pragma: no cover — best-effort cleanup
        print(f"  (destroy attempt failed: {exc!r})", flush=True)
    sys.exit(
        f"✗ instance {new_id} did not boot within {boot_timeout}s "
        f"(watchdog rc={watch_rc}). Offer {o['id']} (machine "
        f"{o.get('machine_id', '?')}) is unhealthy — re-run "
        f"`brian deploy` to try another."
    )
print(f"done (label={LABEL}, id={new_id})")
