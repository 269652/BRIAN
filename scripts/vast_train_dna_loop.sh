#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# vast_train_dna_loop.sh — crash-resilient training loop for DNA-based runs.
#
# Canonical pipeline (2026-06-12 refactor):
#   1. Compile DNA → DSL → HypergraphIR via prepare_run_workspace on the
#      vast.ai box (same helper cli.cmd_train + cli._deploy_dna use locally).
#      The workspace lives at $REPO_DIR/.neuro/arch/temp/ and is the SINGLE
#      source of truth — there is no separate on-box RibosomeCompiler call.
#   2. Train via `python -m neuroslm.train_dsl --arch .neuro/arch/temp ...`.
#      The training script reads runtime hyperparameters (steps, batch,
#      seq_len) from the prepared arch.neuro's `training { ... }` block.
#
# Source of truth for DNA (highest priority first):
#   1. BRIAN_SOURCE_DNA env var (set by cli._deploy_dna)
#   2. DNA env var (legacy / CI overrides)
#   3. brian.toml [current].dna   (workspace config)
#   4. hardcoded fallback: "dna/evol/arch.dna"
#
# Tunable env vars (with defaults):
#   BRIAN_SOURCE_DNA=<path>             path to evolved DNA file (preferred)
#   DNA=<path>                          legacy alias for BRIAN_SOURCE_DNA
#   STEPS=10000                         target step count
#   BATCH=4 SEQ_LEN=256 D_SEM=256
#   LOG_EVERY=20 SAVE_EVERY=1000
#   CKPT_DIR=$REPO_DIR/lfs_checkpoints
#   MAX_RESTARTS=1000
#
# Resume semantics: the harness's checkpoint format is independent of
# Brain's. On restart we load the most recent dsl_arch_step*.pt from
# CKPT_DIR if present (same as DSL training).
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
cd "$REPO_DIR"

# Resolve DNA: BRIAN_SOURCE_DNA wins, then legacy DNA, then brian.toml,
# then hardcoded fallback. cli._deploy_dna sets BRIAN_SOURCE_DNA after
# its local pre-compile; the legacy `DNA=` and brian.toml fallbacks are
# kept so the script still works when invoked directly (CI smoke runs,
# manual operator commands, etc).
DNA="${BRIAN_SOURCE_DNA:-${DNA:-}}"
if [ -z "$DNA" ]; then
    DNA="$(python3 - <<'PY' 2>/dev/null || echo dna/evol/arch.dna
from neuroslm.project_config import load_project_config
cfg = load_project_config()
print(cfg.dna if cfg.dna else "dna/evol/arch.dna")
PY
)"
fi
DNA="${DNA:-dna/evol/arch.dna}"
PRESET="${PRESET:-rcc_bowtie_30m_p4}"   # sizes the DSL LM trunk to match P4

if [ ! -f "$DNA" ]; then
    echo "✗ missing $DNA — is the DNA file there?" >&2
    exit 1
fi

# ── Canonical pipeline step 1: DNA → DSL → HypergraphIR ──────────────
# prepare_run_workspace unpacks the DNA into .neuro/arch/temp/ and lifts
# the HypergraphIR. This is the SAME helper cli.cmd_train + cli._deploy_dna
# use locally — single source of truth for the DNA→DSL compile, no
# bespoke RibosomeCompiler invocation here.
WORKSPACE_DIR="$REPO_DIR/.neuro/arch/temp"
echo "── compiling DNA → DSL → HypergraphIR workspace ──"
python3 - <<PY
from neuroslm.compiler.run_workspace import prepare_run_workspace
ws = prepare_run_workspace(dna="$DNA")
print(f"  workspace: {ws.arch_root}")
print(f"  source:    {ws.source_kind}={ws.source_path}")
print(f"  IR:        {len(ws.hypergraph_ir.nodes)} nodes, "
      f"{len(ws.hypergraph_ir.hyperedges)} edges")
PY
if [ $? -ne 0 ]; then
    echo "✗ prepare_run_workspace failed — DNA compile aborted." >&2
    exit 1
fi

# Read training hyperparameter defaults from the prepared workspace's
# arch.neuro. Same helper the local CLI uses; no DNA-specific code path.
_arch_default() {
    # Usage: _arch_default <attr> <fallback>
    python3 - "$WORKSPACE_DIR" "$1" "$2" <<'PY' 2>/dev/null || echo "$2"
import sys
from pathlib import Path
arch_root, attr, fallback = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    from neuroslm.dsl.training_config import load_training_config_from_arch
    cfg = load_training_config_from_arch(Path(arch_root))
    print(getattr(cfg, attr, fallback))
except Exception:
    print(fallback)
PY
}
STEPS="${STEPS:-$(_arch_default steps 10000)}"
BATCH="${BATCH:-$(_arch_default batch_size 4)}"
SEQ_LEN="${SEQ_LEN:-$(_arch_default seq_len 1024)}"
D_SEM="${D_SEM:-384}"                   # P4 d_hidden (overridden by PRESET)
DATA="${DATA:-real}"
MODE="${MODE:-mix}"
CHAT_RATIO="${CHAT_RATIO:-0.6}"
LOG_EVERY="${LOG_EVERY:-20}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
# PUSH_EVERY > 0 → push each periodically-saved checkpoint to Git LFS
# after writing it locally. Closes the H24 loss-hole (instance
# 41031063 lost its 3 k-step run because we only pushed at
# end-of-training). Default 0 preserves back-compat; the canonical
# entry point (cli.cmd_deploy) sets PUSH_EVERY=500 from brian.toml
# [defaults].push_every.
PUSH_EVERY="${PUSH_EVERY:-0}"
OOD_EVERY="${OOD_EVERY:-0}"
CKPT_DIR="${CKPT_DIR:-$REPO_DIR/lfs_checkpoints}"
MAX_RESTARTS="${MAX_RESTARTS:-1000}"

mkdir -p "$CKPT_DIR"

# Extract architecture name from the prepared workspace's arch.neuro
# (cheap regex; no DNA recompile).
ARCH_NAME=$(python3 -c "
import re
from pathlib import Path
try:
    src = Path('$WORKSPACE_DIR/arch.neuro').read_text(encoding='utf-8')
    m = re.search(r'\barchitecture\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{', src)
    print(m.group(1) if m else Path('$DNA').stem)
except Exception:
    print(Path('$DNA').stem)
" 2>/dev/null || echo "dna-model")

echo "════════════════════════════════════════════════════════════════"
echo "  DNA-driven training loop (canonical pipeline)"
echo "  dna=$DNA  ->  workspace=$WORKSPACE_DIR  (arch=$ARCH_NAME)"
echo "  steps=$STEPS batch=$BATCH seq_len=$SEQ_LEN d_sem=$D_SEM"
echo "  ckpt_dir=$CKPT_DIR"
echo "════════════════════════════════════════════════════════════════"

restart=0
while [ "$restart" -lt "$MAX_RESTARTS" ]; do
    echo ""
    echo "▶ launch attempt $((restart+1)) @ $(date -u +%H:%M:%SZ)"

    python -u -m neuroslm.train_dsl \
        --arch "$WORKSPACE_DIR" \
        --model dsl_lm \
        --preset "$PRESET" \
        --data "$DATA" \
        --mode "$MODE" \
        --chat_ratio "$CHAT_RATIO" \
        --steps "$STEPS" \
        --batch "$BATCH" \
        --seq_len "$SEQ_LEN" \
        --d_sem "$D_SEM" \
        --device cuda \
        --log_every "$LOG_EVERY" \
        --save_every "$SAVE_EVERY" \
        --push_every "$PUSH_EVERY" \
        --ood_every "$OOD_EVERY" \
        --ckpt_dir "$CKPT_DIR" \
        --resume
    rc=$?

    if [ "$rc" -eq 0 ]; then
        echo "✓ training reached target ($STEPS steps). Done."
        break
    fi

    echo "⚠ training exited with code $rc — restarting in 15s..."
    restart=$((restart+1))
    sleep 15
done

if [ "$restart" -ge "$MAX_RESTARTS" ]; then
    echo "✗ hit MAX_RESTARTS=$MAX_RESTARTS; giving up." >&2
    exit 1
fi
