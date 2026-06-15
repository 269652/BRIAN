# -*- coding: utf-8 -*-
"""Regression tests for the deploy "failure-safety" contract.

Captures the silent-failure mode that destroyed instance 41045637
(2026-06-15): vast.ai accepted the create call, returned a contract
id, but the container never booted. Our deploy code printed
``done`` and exited, never noticing.

Combined with the H24 mode (instance 41031063, 2026-06-15): training
crashed mid-run, the ``tee /workspace/train.log`` pipe swallowed the
crash exit code, ``set -e`` therefore didn't trip, and the on-box
``── self-destroy ──`` block fired anyway — taking 3k steps of
checkpoints + the crash log to /dev/null.

Three intertwined contracts pinned here
────────────────────────────────────────
A. **Boot watchdog**: after ``vastai create``, the deploy script
   parses the new contract id from output and polls vast.ai until
   the instance reaches ``actual_status == "running"`` (or a clear
   timeout / terminal-state error). No silent walk-away.

B. **Pipefail on the training pipe**: ``set -eo pipefail`` is enabled
   in ONSTART AND the training command captures
   ``TRAIN_RC=${PIPESTATUS[0]}`` so a crash inside the script
   propagates past ``tee``.

C. **Gated self-destroy** (the user's hard rule): an error MUST
   write the final log to origin BEFORE any ``vastai destroy``. If
   the log push fails, the instance stays alive for forensics. On
   training failure, the script sleeps ``KEEP_ALIVE_ON_FAIL``
   minutes (default 60) before destroying, so an operator can SSH in
   and recover state. ``KEEP_ALIVE_ON_FAIL=0`` disables auto-destroy
   on failure entirely.

All three are locked by grepping the ``_deploy_train.py`` source —
matching the convention of ``test_checkpoint_push_cadence.py``.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_SCRIPT = REPO_ROOT / "_deploy_train.py"


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def deploy_src() -> str:
    """Raw text of ``_deploy_train.py`` — grepped by most tests below."""
    return DEPLOY_SCRIPT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def deploy_module():
    """Import ``_deploy_train`` under a controlled env so we can
    actually call the helper functions (boot watchdog, parser) WITHOUT
    triggering the vast.ai network calls at module load.

    Strategy: stub VAST_API_KEY + GITHUB env vars, then short-circuit
    the module by reading the source and exec'ing only the helper
    definitions. This avoids the module-level vastai("set", ...) call.
    """
    src = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    # Cut at the first vastai( call so the helper defs are loaded but
    # the network-touching block is not. The functions we want
    # (_parse_new_contract_id, _wait_for_instance_ready) must be defined
    # ABOVE that cut point.
    cut = src.find('vastai("set"')
    if cut < 0:
        cut = src.find("vastai('set'")
    assert cut > 0, "could not find the vastai('set', 'api-key') guard line"
    pruned = src[:cut] + "\nVASTAI_EXE = 'vastai'  # stub\n"

    # Stub env vars so the assert at top passes.
    os.environ.setdefault("VAST_API_KEY", "test-key")
    os.environ.setdefault("GITHUB", "test-pat")
    os.environ.setdefault("ARCH", "current")

    spec = importlib.util.spec_from_loader("_deploy_train_helpers", loader=None)
    mod = importlib.util.module_from_spec(spec)
    # Provide __file__ so any Path(__file__).parent references resolve.
    mod.__file__ = str(DEPLOY_SCRIPT)
    try:
        exec(compile(pruned, str(DEPLOY_SCRIPT), "exec"), mod.__dict__)
    except SystemExit:
        # Some early validation might sys.exit when ARCH doesn't resolve;
        # that's fine — the helpers we want are already defined.
        pass
    return mod


# ─────────────────────────────────────────────────────────────────────
# A. Boot watchdog
# ─────────────────────────────────────────────────────────────────────


class TestParseNewContractId:
    """``vastai create instance`` prints
    ``Started. {'success': True, 'new_contract': 41045637, ...}`` on
    success. Parse the integer id robustly (the dict is Python-repr,
    not strict JSON)."""

    def test_parses_real_started_line(self, deploy_module):
        out = (
            "creating instance...\n"
            "Started. {'success': True, 'new_contract': 41045637, "
            "'price': 0.7342592592592593}\n"
        )
        assert deploy_module._parse_new_contract_id(out) == 41045637

    def test_returns_none_on_no_match(self, deploy_module):
        assert deploy_module._parse_new_contract_id("some other output") is None

    def test_returns_none_on_create_failure(self, deploy_module):
        # vast prints success=False when the host is unavailable.
        out = "Started. {'success': False, 'msg': 'host unavailable'}\n"
        # No new_contract key at all → must return None.
        assert deploy_module._parse_new_contract_id(out) is None

    def test_tolerates_double_quotes(self, deploy_module):
        # Some vast versions emit valid JSON; parser should still work.
        out = 'Started. {"success": true, "new_contract": 41045637}\n'
        assert deploy_module._parse_new_contract_id(out) == 41045637


class TestWaitForInstanceReady:
    """The watchdog polls a ``status_fn`` callable until the instance
    reaches ``running``, times out, or hits a terminal state."""

    def test_returns_zero_when_running(self, deploy_module):
        states = ["loading", "running"]
        def status_fn(_id):
            return states.pop(0) if states else "running"
        rc = deploy_module._wait_for_instance_ready(
            instance_id=41045637, timeout=5.0, poll_interval=0.01,
            status_fn=status_fn,
        )
        assert rc == 0

    def test_returns_nonzero_on_timeout(self, deploy_module):
        def status_fn(_id):
            return "loading"
        rc = deploy_module._wait_for_instance_ready(
            instance_id=41045637, timeout=0.05, poll_interval=0.01,
            status_fn=status_fn,
        )
        assert rc != 0, "loading forever must time out"

    def test_returns_nonzero_on_terminal_state_before_running(self, deploy_module):
        """If the host fails to bring up the container the API reports
        ``exited`` (or ``destroyed``) without ever passing through
        ``running``. The watchdog must surface this as a failure
        rather than block until timeout."""
        def status_fn(_id):
            return "exited"
        rc = deploy_module._wait_for_instance_ready(
            instance_id=41045637, timeout=5.0, poll_interval=0.01,
            status_fn=status_fn,
        )
        assert rc != 0

    def test_returns_nonzero_when_instance_disappears(self, deploy_module):
        """The exact 41045637 mode: the REST API returns
        ``{"instances": null}`` once the contract is gone. Encode that
        as the special status ``"gone"`` from status_fn."""
        def status_fn(_id):
            return "gone"
        rc = deploy_module._wait_for_instance_ready(
            instance_id=41045637, timeout=5.0, poll_interval=0.01,
            status_fn=status_fn,
        )
        assert rc != 0

    def test_tolerates_transient_api_error(self, deploy_module):
        """A single API blip (network hiccup) shouldn't kill the watch.
        After retry, the watchdog should see ``running`` and succeed."""
        calls = ["ERROR", "loading", "running"]
        def status_fn(_id):
            v = calls.pop(0)
            if v == "ERROR":
                raise RuntimeError("transient network blip")
            return v
        rc = deploy_module._wait_for_instance_ready(
            instance_id=41045637, timeout=5.0, poll_interval=0.01,
            status_fn=status_fn,
        )
        assert rc == 0

    def test_signature_has_safe_defaults(self, deploy_module):
        """Operators call the watchdog with only the instance id —
        defaults must be sensible (timeout >= 5 min, poll_interval
        small enough to feel responsive)."""
        import inspect
        sig = inspect.signature(deploy_module._wait_for_instance_ready)
        # timeout default >= 300 s (5 min — vast image-pull can take 5+)
        assert sig.parameters["timeout"].default >= 300
        # poll_interval default >= 5 s (don't spam the API) and <= 30 s
        pi = sig.parameters["poll_interval"].default
        assert 5 <= pi <= 30


class TestCreateCallCapturesOutput:
    """``vastai create instance`` must be called with ``capture=True``
    in the deploy script so we can parse the contract id."""

    def test_create_call_uses_capture_true(self, deploy_src):
        # Multi-line call — search for the "create" literal first,
        # then look forward up to 800 chars for capture=True. This
        # tolerates the kwargs being on different lines.
        m = re.search(
            r'vastai\(\s*["\']create["\'][\s\S]{0,800}?capture\s*=\s*True',
            deploy_src,
        )
        assert m, (
            "vastai('create', ...) must use capture=True so the deploy "
            "script can parse new_contract and call _wait_for_instance_ready"
        )

    def test_wait_for_ready_called_after_create(self, deploy_src):
        # Anchor on the call site (uses kwargs) NOT the def (uses
        # positional params followed by `=default`). The def has
        # `instance_id, timeout=600` — the call has `instance_id=new_id`.
        m_create = re.search(r'vastai\(\s*["\']create["\']', deploy_src)
        m_wait = re.search(
            r"_wait_for_instance_ready\(\s*instance_id\s*=", deploy_src
        )
        assert m_create, "no `vastai('create', ...)` call found"
        assert m_wait, "_wait_for_instance_ready must be CALLED (not just defined)"
        assert m_wait.start() > m_create.start(), (
            "the watchdog call must come AFTER the create call"
        )

    def test_deploy_exits_nonzero_on_watchdog_failure(self, deploy_src):
        """If the watchdog times out / sees a terminal state, the
        deploy script must propagate that as a non-zero exit so the
        local caller (``brian deploy``) doesn't print success."""
        # Anchor on the CALL site (kwargs form) not the def.
        m = re.search(
            r"_wait_for_instance_ready\(\s*instance_id\s*=", deploy_src
        )
        assert m, "_wait_for_instance_ready must be called with instance_id="
        # Within 1000 chars after the call there must be a sys.exit
        # (the watchdog-failure branch).
        snippet = deploy_src[m.start():m.start() + 1000]
        assert "sys.exit" in snippet, (
            "expected sys.exit() within 1000 chars after the "
            "_wait_for_instance_ready call (non-zero on watchdog failure)"
        )


# ─────────────────────────────────────────────────────────────────────
# B. Pipefail on the training pipe
# ─────────────────────────────────────────────────────────────────────


class TestPipefailContract:
    """``set -e`` alone doesn't trip on a failed pipe — ``set -o pipefail``
    is required. And we must capture the LEFT side of the pipe
    (training) explicitly via ``${PIPESTATUS[0]}``."""

    def test_pipefail_enabled_in_onstart(self, deploy_src):
        assert (
            "set -eo pipefail" in deploy_src
            or "set -e -o pipefail" in deploy_src
            or "set -o pipefail" in deploy_src
        ), (
            "ONSTART must enable pipefail so a training crash before "
            "the `| tee /workspace/train.log` pipe isn't swallowed"
        )

    def test_train_rc_captured_from_pipestatus(self, deploy_src):
        # Must use ${PIPESTATUS[0]} to grab the leftmost (training) exit
        # code. Bare $? would already work with pipefail but PIPESTATUS
        # is unambiguous and survives if someone disables pipefail later.
        assert "PIPESTATUS[0]" in deploy_src
        assert "TRAIN_RC" in deploy_src


# ─────────────────────────────────────────────────────────────────────
# C. Gated self-destroy (the user's hard rule)
# ─────────────────────────────────────────────────────────────────────


class TestSelfDestroyIsGatedOnLogPush:
    """An error MUST write the error log BEFORE any vastai destroy.
    If the log push fails, the instance stays alive — operator can
    SSH in for forensics."""

    def test_final_log_push_before_destroy(self, deploy_src):
        # Anchor on the ONSTART box-drawing markers (── ... ──) —
        # those only appear in the on-box shell text, never in a
        # Python comment (the file's docstring uses the word
        # "self-destroyed" which would otherwise match a plain
        # `find("self-destroy")`).
        idx_log = deploy_src.find("── final log push")
        idx_dst = deploy_src.find("── self-destroy ──")
        assert idx_log >= 0, "missing `── final log push ──` block"
        assert idx_dst >= 0, "missing `── self-destroy ──` block"
        assert idx_log < idx_dst, (
            "self-destroy block MUST come AFTER final log push — the "
            "hard rule the operator stated 2026-06-15"
        )

    def test_log_push_failure_exits_before_destroy(self, deploy_src):
        """The shell pattern must be: capture log_pusher exit into
        LOG_PUSH_RC, then ``if [ $LOG_PUSH_RC -ne 0 ]; then ... exit``
        BEFORE reaching the destroy block."""
        idx_rc = deploy_src.find("LOG_PUSH_RC")
        idx_dst = deploy_src.find("── self-destroy ──")
        assert idx_rc >= 0, "expected LOG_PUSH_RC variable in ONSTART"
        assert idx_dst > idx_rc, "destroy must come after LOG_PUSH_RC"
        # Between LOG_PUSH_RC and destroy, there must be an `exit`
        # somewhere — the gate.
        between = deploy_src[idx_rc:idx_dst]
        assert re.search(r"LOG_PUSH_RC[^\n]*\n[\s\S]{0,400}?\bexit\b", between), (
            "expected an `exit` statement gated on LOG_PUSH_RC != 0 "
            "between the log push and the self-destroy block"
        )


class TestKeepAliveOnFailure:
    """On training failure, the box must wait KEEP_ALIVE_ON_FAIL
    minutes before destroying itself — giving an operator a window
    to SSH in and pull state. ``KEEP_ALIVE_ON_FAIL=0`` disables
    auto-destroy entirely (operator must clean up manually)."""

    def test_keep_alive_on_fail_env_referenced(self, deploy_src):
        assert "KEEP_ALIVE_ON_FAIL" in deploy_src, (
            "ONSTART must consult KEEP_ALIVE_ON_FAIL (default 60 min)"
        )

    def test_keep_alive_default_60_min(self, deploy_src):
        # The default substitution: ${KEEP_ALIVE_ON_FAIL:-60}
        assert (
            "${KEEP_ALIVE_ON_FAIL:-60}" in deploy_src
            or "KEEP_ALIVE_ON_FAIL:-60}" in deploy_src
        ), "default keep-alive window on failure must be 60 min"

    def test_keep_alive_zero_disables_auto_destroy(self, deploy_src):
        """KEEP_ALIVE_ON_FAIL=0 → exit without destroy. Look for an
        ``[ ... -eq 0 ]`` or ``== 0`` guard followed by ``exit``."""
        assert re.search(
            r'KEEP_ALIVE_ON_FAIL["\s]*-eq\s+0|KEEP_ALIVE_ON_FAIL.*==\s*"?0"?',
            deploy_src,
        ), (
            "expected a guard branch for KEEP_ALIVE_ON_FAIL == 0 that "
            "exits without calling vastai destroy"
        )

    def test_failure_path_sleeps_before_destroy(self, deploy_src):
        """The failure branch must include a sleep keyed on
        KEEP_ALIVE_ON_FAIL minutes before reaching the destroy
        command."""
        # The sleep should be `sleep $((KEEP_ALIVE_ON_FAIL * 60))`
        assert re.search(
            r"sleep\s+\$\(\(\s*KEEP_ALIVE_ON_FAIL\s*\*\s*60\s*\)\)",
            deploy_src,
        ), (
            "expected `sleep $((KEEP_ALIVE_ON_FAIL * 60))` so the "
            "failure window is honored before destroy"
        )

    def test_success_path_destroys_immediately(self, deploy_src):
        """On TRAIN_RC == 0, the box should destroy without waiting
        (paying $0.73/h for a successful idle box is bad)."""
        # Look for `if [ "$TRAIN_RC" -eq 0 ]` or similar branch.
        assert re.search(
            r'TRAIN_RC["\s]*-eq\s+0|TRAIN_RC.*==\s*"?0"?',
            deploy_src,
        ), "expected a TRAIN_RC == 0 branch that triggers immediate destroy"
