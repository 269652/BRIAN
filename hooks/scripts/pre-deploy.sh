#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# hooks/scripts/pre-deploy.sh — Linux / macOS impl of the pre-deploy hook.
#
# Symmetric with hooks/scripts/pre-deploy.ps1. The hook runner picks
# this one when platform.system() != "Windows".
#
# Invoked by neuroslm.hooks.run_hook("pre-deploy", REPO_ROOT). The
# runner sets:
#   BRIAN_HOOK_NAME=pre-deploy
#   BRIAN_REPO_ROOT=<absolute repo path>
# Stdout / stderr stream live to the user's terminal (no PIPE).
# ──────────────────────────────────────────────────────────────────

set -euo pipefail

cd "${BRIAN_REPO_ROOT:-.}"

echo "[pre-deploy] cwd=$(pwd)"
echo "[pre-deploy] step 1/2: brian dna compile (architectures/master → dna/master/arch.dna)"
# Explicit paths (not the brian.toml defaults) because the user's
# convention is:
#   architectures/master       = source-of-truth template (edit here)
#   dna/<branch>/arch.dna      = canonical compiled DNA the deploy trains on.
#                                Path mirrors the active git branch from
#                                brian.toml [defaults].branch so a fork on a
#                                feature branch never clobbers master's DNA.
#   architectures/current/     = live working copy (auto-overwritten below)
# Using `python -m neuroslm.cli` instead of the `brian` entry point
# so the hook works even when the package is checked out but not
# pip-installed.
python -m neuroslm.cli dna compile architectures/master --output dna/master/arch.dna

echo "[pre-deploy] step 2/2: brian dna unfold (dna/master/arch.dna → architectures/current/arch.neuro)"
# Unfolding back into architectures/current keeps the on-disk working
# copy bit-identical to the bytes the deploy will actually train on.
# Catches "I edited current/ but forgot to compile" foot-guns AND
# "I edited master/ but forgot to propagate to current/" both.
python -m neuroslm.cli dna unfold dna/master/arch.dna --output architectures/current/arch.neuro

echo "[pre-deploy] ok — master → DNA → current are all in sync"
