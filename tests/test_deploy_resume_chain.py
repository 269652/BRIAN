# -*- coding: utf-8 -*-
"""Cross-component test: ``RESUME_FROM`` propagates end-to-end through
the deploy chain.

The chain has five hops:

  brian deploy --resume X / --latest
       │   sets extra['RESUME_FROM']
       ▼
  _deploy_dsl/_deploy_dna           (neuroslm.cli)
       │   merges into env passed to subprocess
       ▼
  _deploy_train.py                  (RESUME_FROM module const + ONSTART)
       │   writes ONSTART script with `export RESUME_FROM='...'`
       ▼
  vast_train_dsl_loop.sh            (RESUME_ARGS bash array)
       │   builds `--resume_from "$RESUME_FROM"` when non-empty
       ▼
  neuroslm.train_dsl --resume_from PATH_OR_URI

A break in any hop silently disables the resume feature. Each hop
gets one focused test here.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────
# Hop 1: brian deploy → extra['RESUME_FROM']
#   (already covered by test_cli_hf_chat.TestCmdDeployResume)
# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────
# Hop 2: VastConnector._build_env() sets RESUME_FROM in the subprocess env.
# The old _deploy_train.py hop was removed in the connector refactor
# (2026-06-18). Equivalent contracts now live in test_connectors.py::
#   test_G_build_env_propagates_all_fields (RESUME_FROM present)
#   test_I_vast_launch_calls_vast_train_sh  (subprocess call shape)
# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────
# Hop 4: bash wrappers build RESUME_ARGS array correctly
# ─────────────────────────────────────────────────────────────────────


class TestBashLoopsForwardResumeFrom:
    """Both vast_train_dsl_loop.sh and vast_train_dna_loop.sh must
    consult ``$RESUME_FROM`` and switch from ``--resume`` to
    ``--resume_from "$RESUME_FROM"`` when non-empty."""

    @pytest.fixture(params=["vast_train_dsl_loop.sh",
                            "vast_train_dna_loop.sh"])
    def loop_src(self, request) -> str:
        path = REPO_ROOT / "scripts" / request.param
        return path.read_text(encoding="utf-8")

    def test_reads_env_with_default_empty(self, loop_src):
        assert 'RESUME_FROM="${RESUME_FROM:-}"' in loop_src

    def test_default_args_is_legacy_resume(self, loop_src):
        assert 'RESUME_ARGS=("--resume")' in loop_src

    def test_non_empty_switches_to_resume_from(self, loop_src):
        # The crucial branch — when RESUME_FROM is set, use the
        # new flag, not the legacy globber.
        assert 'if [ -n "$RESUME_FROM" ]' in loop_src
        assert 'RESUME_ARGS=("--resume_from" "$RESUME_FROM")' in loop_src

    def test_array_expansion_appears_in_python_invocation(self, loop_src):
        """The ``${RESUME_ARGS[@]}`` expansion must be the actual arg
        passed to python -m neuroslm.train_dsl, not just declared."""
        assert '"${RESUME_ARGS[@]}"' in loop_src
        # And it must appear inside a `python … -m neuroslm.train_dsl`
        # call — find the train_dsl invocation (skipping comment lines
        # that start with #) and check the array expansion is in the
        # multi-line call's continuation.
        non_comment_lines = [
            ln for ln in loop_src.split("\n")
            if not ln.lstrip().startswith("#")
        ]
        non_comment = "\n".join(non_comment_lines)
        m = re.search(
            r"python\s+(?:-\w+\s+)*-m\s+neuroslm\.train_dsl\b",
            non_comment,
        )
        assert m, "scripts must invoke `python -m neuroslm.train_dsl`"
        # Get the next ~2000 chars after the match start
        block_start = m.start()
        block = non_comment[block_start: block_start + 2000]
        assert '"${RESUME_ARGS[@]}"' in block, \
            f"RESUME_ARGS not in train_dsl invocation. Block:\n{block[:500]}"


# ─────────────────────────────────────────────────────────────────────
# Hop 5: train_dsl.py reads --resume_from / RESUME_FROM env
#   (already covered by test_train_dsl_resume_from.py)
# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────
# End-to-end sanity: env var name is consistent everywhere
# ─────────────────────────────────────────────────────────────────────


class TestEnvVarNameConsistency:
    """RESUME_FROM (not RESUME, RESUMEFROM, RESUME_PATH, etc.) is the
    contract. Drift between CLI / deploy / bash / trainer would silently
    break the chain."""

    def test_env_var_is_RESUME_FROM_everywhere(self):
        # _deploy_train.py was removed in the connector refactor (2026-06-18).
        # Its role is now played by VastConnector._build_env() in vast.py.
        files = [
            REPO_ROOT / "neuroslm" / "cli.py",
            REPO_ROOT / "neuroslm" / "connectors" / "vast.py",
            REPO_ROOT / "scripts" / "vast_train_dsl_loop.sh",
            REPO_ROOT / "scripts" / "vast_train_dna_loop.sh",
            REPO_ROOT / "neuroslm" / "train_dsl.py",
        ]
        for f in files:
            text = f.read_text(encoding="utf-8")
            assert "RESUME_FROM" in text, \
                f"{f.name} missing RESUME_FROM — chain broken at this hop"
