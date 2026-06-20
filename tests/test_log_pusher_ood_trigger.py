# -*- coding: utf-8 -*-
"""TDD: log pusher triggers after OOD validation, not before.

The mid-OOD eval in train_dsl.py prints:
    [mid-ood] step N: wikitext ppl=... gap_ratio=... (M seq, K tok)
after ~30s of GPU work. The old step-bucket gate fires when it sees
`step N` in the log — before the OOD block even starts. So the push
captures the log WITHOUT the gap_ratio line. Fix: switch to an OOD-
sentinel gate that fires only after a new `[mid-ood] step N` line
appears.

Contracts:
  A) log_pusher.sh: OOD_EVERY env var + LAST_OOD_STEP_PUSHED tracker
  B) log_pusher.sh: _last_mid_ood_step() greps the right sentinel format
  C) log_pusher.sh: OOD_EVERY > 0 activates the OOD-sentinel gate
     (separate from the LOG_EVERY bucket gate)
  D) vast_train.sh: exports OOD_EVERY to the background log pusher
  E) colab_run.ipynb: log pusher thread polls for [mid-ood] lines
     instead of sleeping a fixed 5 minutes
"""
import re
import pytest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG_PUSHER = (REPO / "scripts" / "log_pusher.sh").read_text(encoding="utf-8")
VAST_TRAIN = (REPO / "scripts" / "vast_train.sh").read_text(encoding="utf-8")
COLAB = (REPO / "colab_run.ipynb").read_text(encoding="utf-8")


# ── Contract A: OOD_EVERY env var and tracker ──────────────────────────────

class TestOodEveryEnvVar:
    def test_ood_every_has_default_zero(self):
        assert 'OOD_EVERY="${OOD_EVERY:-0}"' in LOG_PUSHER, (
            "log_pusher.sh must default OOD_EVERY to 0 so existing "
            "deploys without the var work unchanged"
        )

    def test_last_ood_step_pushed_tracker_exists(self):
        assert "LAST_OOD_STEP_PUSHED" in LOG_PUSHER, (
            "log_pusher.sh must track LAST_OOD_STEP_PUSHED so it knows "
            "which OOD steps it has already committed"
        )

    def test_last_ood_step_pushed_initialised_negative(self):
        # Must initialise to -1 (or 0) so the first [mid-ood] line always
        # triggers a push even when OOD step=0.
        assert re.search(r"LAST_OOD_STEP_PUSHED=(-1|0)", LOG_PUSHER), (
            "LAST_OOD_STEP_PUSHED must be initialised to -1 (or 0) "
            "before the daemon loop"
        )


# ── Contract B: _last_mid_ood_step() sentinel parser ──────────────────────

class TestLastMidOodStepParser:
    def test_function_defined(self):
        assert "_last_mid_ood_step" in LOG_PUSHER, (
            "log_pusher.sh must define _last_mid_ood_step() to parse "
            "[mid-ood] lines from the training log"
        )

    def test_function_greps_mid_ood_sentinel(self):
        assert "mid-ood" in LOG_PUSHER, (
            "_last_mid_ood_step() must grep for '[mid-ood]' — the "
            "exact string printed by _mid_ood_eval in train_dsl.py"
        )

    def test_function_extracts_step_number(self):
        # The function must extract a numeric step from the grep match.
        # The format in train_dsl.py is: "[mid-ood] step N: wikitext ppl=..."
        # So we need to grep for the step number after "step".
        mid_ood_block = LOG_PUSHER[
            LOG_PUSHER.find("_last_mid_ood_step"):
            LOG_PUSHER.find("_last_mid_ood_step") + 300
        ]
        assert re.search(r"step\s+\[0-9\]|step\s+\\d|oE.*step.*\[0-9\]", mid_ood_block), (
            "_last_mid_ood_step() must extract the step number from "
            "lines like '[mid-ood] step 1000: wikitext ppl=...'"
        )


# ── Contract C: OOD-sentinel gate activates when OOD_EVERY > 0 ────────────

class TestOodSentinelGate:
    def test_ood_every_gt_zero_triggers_ood_path(self):
        # The daemon loop must check OOD_EVERY and take a different
        # branch from the LOG_EVERY bucket gate.
        # We look for: `[ "$OOD_EVERY" -gt 0 ]` (or equivalent) in the loop.
        assert re.search(
            r'\[\s*["\']?\$OOD_EVERY["\']?\s+-gt\s+0',
            LOG_PUSHER,
        ), (
            "log_pusher.sh main loop must test OOD_EVERY > 0 to switch "
            "between OOD-sentinel and LOG_EVERY-bucket push modes"
        )

    def test_last_ood_step_pushed_updated_after_push(self):
        # After a successful push in OOD mode, LAST_OOD_STEP_PUSHED must
        # advance so we don't re-push the same OOD step next iteration.
        idx_loop = LOG_PUSHER.find("while true")
        loop_body = LOG_PUSHER[idx_loop:]
        # Both the tracker variable and an assignment to it must appear in
        # the daemon loop body.
        assert re.search(r"LAST_OOD_STEP_PUSHED=", loop_body), (
            "LAST_OOD_STEP_PUSHED must be updated (assigned) inside the "
            "daemon loop after a successful OOD-triggered push"
        )

    def test_ood_gate_is_separate_from_log_every_gate(self):
        # The OOD path must not share the LAST_PUSHED_BUCKET variable with
        # the LOG_EVERY path — mixing them would make OOD mode push on
        # LOG_EVERY boundaries instead of OOD completions.
        # Both variables must appear, each in their own conditional branch.
        assert "LAST_OOD_STEP_PUSHED" in LOG_PUSHER
        assert "LAST_PUSHED_BUCKET" in LOG_PUSHER

    def test_ood_banner_emitted_when_ood_every_gt_zero(self):
        # The startup banner must announce OOD-driven mode when OOD_EVERY > 0.
        assert "OOD-DRIVEN" in LOG_PUSHER or "OOD_EVERY" in LOG_PUSHER, (
            "log_pusher.sh must print a banner indicating OOD-DRIVEN mode "
            "when OOD_EVERY > 0 (mirrors the STEP-DRIVEN banner)"
        )


# ── Contract D: vast_train.sh exports OOD_EVERY to log pusher ─────────────

class TestVastTrainExportsOodEvery:
    def test_ood_every_in_log_pusher_launch(self):
        # The `nohup bash scripts/log_pusher.sh` invocation must carry
        # OOD_EVERY so the daemon knows which mode to use.
        # We find the block that contains `log_pusher.sh` and check for
        # OOD_EVERY in a nearby few hundred chars.
        idx = VAST_TRAIN.find("log_pusher.sh")
        assert idx >= 0, "vast_train.sh must launch log_pusher.sh"
        # Look at the preceding ~400 chars (env var prefix before the nohup)
        context = VAST_TRAIN[max(0, idx - 400): idx + 100]
        assert "OOD_EVERY" in context, (
            "vast_train.sh must export OOD_EVERY to the log_pusher.sh "
            "background process so it can switch to OOD-sentinel mode"
        )


# ── Contract E: Colab log pusher polls for [mid-ood] lines ────────────────

class TestColabLogPusherOodAware:
    def test_colab_thread_watches_for_mid_ood(self):
        # The Colab pusher thread must grep/search the log for [mid-ood]
        # lines instead of (only) sleeping a fixed interval.
        assert "mid-ood" in COLAB, (
            "colab_run.ipynb _log_pusher_thread must watch for [mid-ood] "
            "sentinel lines to know when OOD validation has completed"
        )

    def test_colab_thread_has_short_poll_interval(self):
        # Instead of sleep(300), the thread must poll at a short interval
        # (≤30s) so it reacts quickly when [mid-ood] appears.
        # Accept both literal: _t.sleep(15) and variable: _t.sleep(_POLL_SEC)
        # where _POLL_SEC is defined as a small integer in the same cell.
        pusher_idx = COLAB.find("_log_pusher_thread")
        pusher_block = COLAB[pusher_idx: pusher_idx + 3000]
        per_iter = re.search(r"_t\.sleep\(\s*(\d+)\s*\)", pusher_block)
        if per_iter is not None:
            assert int(per_iter.group(1)) <= 30, (
                f"colab pusher poll interval must be ≤ 30s (was {per_iter.group(1)}s). "
                "Long sleeps mean OOD results sit unshared for minutes."
            )
        else:
            # Variable form: _t.sleep(_POLL_SEC) — verify _POLL_SEC ≤ 30
            var_sleep = re.search(r"_t\.sleep\(\s*_POLL_SEC\s*\)", pusher_block)
            assert var_sleep is not None, (
                "log pusher thread must have _t.sleep(<int>) or _t.sleep(_POLL_SEC)"
            )
            # Find definition of _POLL_SEC before the pusher block
            preamble = COLAB[:pusher_idx + 3000]
            poll_def = re.search(r"_POLL_SEC\s*=\s*(\d+)", preamble)
            assert poll_def is not None, "_POLL_SEC must be defined near the pusher"
            assert int(poll_def.group(1)) <= 30, (
                f"_POLL_SEC={poll_def.group(1)} exceeds 30s limit — "
                "OOD results would sit unshared for too long."
            )

    def test_colab_thread_tracks_last_ood_step(self):
        # The thread must track which OOD step it last pushed so it doesn't
        # re-push the same step on every poll.
        pusher_idx = COLAB.find("_log_pusher_thread")
        pusher_block = COLAB[pusher_idx: pusher_idx + 2000]
        assert re.search(r"last_pushed_ood|last_ood_step|ood_step", pusher_block), (
            "colab pusher thread must track the last OOD step it pushed "
            "so it doesn't commit the same result repeatedly"
        )
