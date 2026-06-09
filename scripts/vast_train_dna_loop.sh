#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# vast_train_dna_loop.sh — crash-resilient training loop for DNA-based runs.
#
# Mirrors vast_train_dsl_loop.sh but trains from an evolved DNA file instead
# of a DSL architecture folder. The model is built from the DNA via the
# Ribosome compiler + BRIAN harness, with loss clipping and other pipeline
# behavior read from the DNA's embedded arch.neuro block.
#
# Source of truth for DNA (highest priority first):
#   1. DNA env var (legacy / CI overrides)
#   2. brian.toml [current].dna   (workspace config)
#   3. hardcoded fallback: "dna/evol/arch.dna"
#
# Tunable env vars (with defaults):
#   DNA=<path>                          path to evolved DNA file
#                                        (auto-resolved from brian.toml if unset)
#   STEPS=10000                         target step count
#   BATCH=4 SEQ_LEN=256 D_SEM=256
#   LOG_EVERY=20 SAVE_EVERY=1000
#   CKPT_DIR=$REPO_DIR/lfs_checkpoints
#   MAX_RESTARTS=1000
#
# Evolutionary inner loop (DNA-mode only, all optional):
#   EVOLVE=1                            enable HeatmapHook + propose →
#                                        gate → save_checkpoint cycle
#                                        in-process during training
#                                        (set to 0 to disable; default 1)
#   HEATMAP_EVERY=50                    grad-norm rollup cadence into
#                                        the hypergraph-IR heatmap
#   MUTATE_EVERY=500                    cadence of propose / gate /
#                                        persist cycles
#   SAVE_HEATMAP_EVERY=500              cadence for writing the live
#                                        heatmap to CKPT_DIR/evolution/
#                                        live_heatmap.json
#   HOT_THRESHOLD=0.7                   normalised-heat threshold for
#                                        HOT (triggers a mutation)
#   COLD_THRESHOLD=0.1                  normalised-heat threshold for
#                                        COLD (triggers a prune proposal)
#
# Resume semantics: the harness's checkpoint format is independent of
# Brain's. On restart we load the most recent dsl_arch_step*.pt from
# CKPT_DIR if present (same as DSL training).
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
cd "$REPO_DIR"

# Resolve DNA from brian.toml if not set in the environment.
if [ -z "${DNA:-}" ]; then
    DNA="$(python3 - <<'PY' 2>/dev/null || echo dna/evol/arch.dna
from neuroslm.project_config import load_project_config
cfg = load_project_config()
print(cfg.dna if cfg.dna else "dna/evol/arch.dna")
PY
)"
fi
DNA="${DNA:-dna/evol/arch.dna}"
PRESET="${PRESET:-rcc_bowtie_30m_p4}"   # sizes the DSL LM trunk to match P4

# For DNA-based training, read runtime hyperparameters from the DNA's
# embedded arch.neuro block. The DNA compiler unfolds it on-the-fly.
_dna_default() {
    # Usage: _dna_default <attr> <fallback>
    python3 - "$DNA" "$1" "$2" <<'PY' 2>/dev/null || echo "$2"
import sys
dna_path, attr, fallback = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    from neuroslm.compiler.ribosome import RibosomeCompiler
    import re
    compiler = RibosomeCompiler()
    dsl_code = compiler.dna_translator.translate_from_file(dna_path)
    m = re.search(r"\barchitecture\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{", dsl_code)
    if not m:
        print(fallback)
        sys.exit(0)
    arch_name = m.group(1)
    from pathlib import Path
    arch_root = Path("architectures") / arch_name
    from neuroslm.dsl.training_config import load_training_config_from_arch
    cfg = load_training_config_from_arch(arch_root)
    print(getattr(cfg, attr, fallback))
except Exception:
    print(fallback)
PY
}
STEPS="${STEPS:-$(_dna_default steps 10000)}"
BATCH="${BATCH:-$(_dna_default batch_size 4)}"
SEQ_LEN="${SEQ_LEN:-$(_dna_default seq_len 1024)}"
D_SEM="${D_SEM:-384}"                   # P4 d_hidden (overridden by PRESET)
DATA="${DATA:-real}"
MODE="${MODE:-mix}"
CHAT_RATIO="${CHAT_RATIO:-0.6}"
LOG_EVERY="${LOG_EVERY:-20}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
OOD_EVERY="${OOD_EVERY:-0}"
CKPT_DIR="${CKPT_DIR:-$REPO_DIR/lfs_checkpoints}"
MAX_RESTARTS="${MAX_RESTARTS:-1000}"

# Evolutionary inner-loop knobs (defaults documented in the header).
# Set EVOLVE=0 to disable epigenetic evolution for this run.
EVOLVE="${EVOLVE:-1}"
HEATMAP_EVERY="${HEATMAP_EVERY:-50}"
MUTATE_EVERY="${MUTATE_EVERY:-500}"
SAVE_HEATMAP_EVERY="${SAVE_HEATMAP_EVERY:-500}"
HOT_THRESHOLD="${HOT_THRESHOLD:-0.7}"
COLD_THRESHOLD="${COLD_THRESHOLD:-0.1}"

# Build the evolve-flag block as a single argument list so it expands
# correctly under set -u.
EVOLVE_ARGS=()
if [ "$EVOLVE" = "1" ]; then
    EVOLVE_ARGS=(
        --evolve
        --heatmap_every       "$HEATMAP_EVERY"
        --mutate_every        "$MUTATE_EVERY"
        --save_heatmap_every  "$SAVE_HEATMAP_EVERY"
        --hot_threshold       "$HOT_THRESHOLD"
        --cold_threshold      "$COLD_THRESHOLD"
    )
fi

mkdir -p "$CKPT_DIR"

if [ ! -f "$DNA" ]; then
    echo "✗ missing $DNA — is the DNA file there?" >&2
    exit 1
fi

# Extract architecture name from DNA for logging
ARCH_NAME=$(python3 -c "
import sys, re
from pathlib import Path
from neuroslm.compiler.ribosome import RibosomeCompiler
try:
    compiler = RibosomeCompiler()
    dsl_code = compiler.dna_translator.translate_from_file('$DNA')
    m = re.search(r'\barchitecture\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{', dsl_code)
    print(m.group(1) if m else Path('$DNA').stem)
except:
    print(Path('$DNA').stem)
" 2>/dev/null || echo "dna-model")

echo "════════════════════════════════════════════════════════════════"
echo "  DNA-driven training loop"
echo "  dna=$DNA (arch=$ARCH_NAME)"
echo "  steps=$STEPS batch=$BATCH seq_len=$SEQ_LEN d_sem=$D_SEM"
echo "  ckpt_dir=$CKPT_DIR"
if [ "$EVOLVE" = "1" ]; then
    echo "  evolution: ENABLED  heatmap_every=$HEATMAP_EVERY"
    echo "             mutate_every=$MUTATE_EVERY save_heatmap_every=$SAVE_HEATMAP_EVERY"
    echo "             hot>$HOT_THRESHOLD cold<$COLD_THRESHOLD"
else
    echo "  evolution: disabled  (set EVOLVE=1 to enable)"
fi
echo "════════════════════════════════════════════════════════════════"

restart=0
while [ "$restart" -lt "$MAX_RESTARTS" ]; do
    echo ""
    echo "▶ launch attempt $((restart+1)) @ $(date -u +%H:%M:%SZ)"

    python -u -m neuroslm.train_dsl \
        --dna "$DNA" \
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
        --ood_every "$OOD_EVERY" \
        --ckpt_dir "$CKPT_DIR" \
        --resume \
        "${EVOLVE_ARGS[@]}"
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
