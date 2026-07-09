# -*- coding: utf-8 -*-
"""`vast_train_dsl_loop.sh` forwards the H52/H53 explore flags to train_dsl.py.

Content-level pin (no live vast.ai runner in CI): the script must read the
EXPLORE_*/USE_MODULATIONS env vars VastConnector's onstart sets, default them
to train_dsl.py's own off-by-default semantics, and forward them as CLI args
to the actual `python -u -m neuroslm.train_dsl` invocation — otherwise a real
training deploy silently never exercises the real-trunk probe, exactly the
gap the user found.
"""
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "vast_train_dsl_loop.sh"


def _text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_script_has_valid_bash_syntax():
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_reads_explore_env_vars_with_off_by_default():
    src = _text()
    assert 'EXPLORE_EVERY="${EXPLORE_EVERY:-0}"' in src
    assert 'EXPLORE_POP="${EXPLORE_POP:-24}"' in src
    assert 'EXPLORE_GENS="${EXPLORE_GENS:-10}"' in src
    assert 'EXPLORE_LEN="${EXPLORE_LEN:-8}"' in src
    assert 'EXPLORE_SITES="${EXPLORE_SITES:-2}"' in src
    assert 'USE_MODULATIONS="${USE_MODULATIONS:-0}"' in src


def test_forwards_explore_flags_to_train_dsl_invocation():
    src = _text()
    assert '"--explore_every" "$EXPLORE_EVERY"' in src
    assert '"--explore_pop" "$EXPLORE_POP"' in src
    assert '"--explore_gens" "$EXPLORE_GENS"' in src
    assert '"--explore_len" "$EXPLORE_LEN"' in src
    assert '"--explore_sites" "$EXPLORE_SITES"' in src
    assert '"${EXPLORE_ARGS[@]}"' in src


def test_use_modulations_is_conditional_not_always_on():
    # --use_modulations is a boolean flag; must only be appended when
    # USE_MODULATIONS=1, never unconditionally (that would force-install
    # banked winners on every deploy, including ones that never asked for it).
    src = _text()
    assert 'if [ "$USE_MODULATIONS" = "1" ]; then' in src
    assert 'EXPLORE_ARGS+=("--use_modulations")' in src
