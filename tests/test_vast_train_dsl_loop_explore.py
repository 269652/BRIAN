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


def _bash() -> str:
    """Resolve a real POSIX bash, not whatever "bash" happens to mean on
    PATH. On Windows, bare "bash" can resolve to the WSL launcher shim
    (System32\\bash.exe) instead of git-bash — WSL's bash reports a
    /mnt/c/... cwd and mangles Windows-style paths passed as arguments
    (confirmed live: a Windows arch.neuro path came through as
    "C:UsersmorrosslDocuments..." with all separators stripped). Reuses
    the same resolution VastConnector._find_bash() uses for real
    deploys, so tests exercise the same shell the deploy path does."""
    from neuroslm.connectors.vast import VastConnector
    return VastConnector._find_bash()


def test_script_has_valid_bash_syntax():
    r = subprocess.run([_bash(), "-n", str(SCRIPT)], capture_output=True, text=True)
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


# ──────────────────────────────────────────────────────────────────────
# 2026-08-11 bug: BATCH/SEQ_LEN silently discarded a declared scales{}
# variant's own values.
#
# _arch_default reads TrainingConfig's TOP-LEVEL seq_len/batch_size
# fields (defaults 256/4) via getattr(cfg, attr, fallback) -- DIFFERENT
# fields from scales.<SCALE>.seq_len/batch_size, which is where every
# arch with a scales{} block (the documented, intended way to declare
# per-scale dims) actually puts its real values. The resolved (wrong)
# value then got exported as SEQ_LEN/BATCH, which train_dsl.py's own
# scale-override logic treats as an explicit override that wins over
# the correct scales.<SCALE>.seq_len -- silently training every scaled
# arch at whatever the top-level default happened to be, regardless of
# what its scales{} block declared.
# ──────────────────────────────────────────────────────────────────────

import tempfile
import textwrap


def _source_scale_default_snippet():
    """Extract _arch_root=...  through the end of _scale_default()'s
    definition from the real script (stable anchors, not line numbers),
    so the behavioral tests below exercise the ACTUAL function, not a
    reimplementation of it."""
    src = _text()
    start = src.index('_arch_root="architectures/${ARCH}"')
    marker = "_scale_default() {"
    fn_start = src.index(marker, start)
    # Find the matching closing brace: the function body's first line
    # at column 0 that is exactly "}".
    end = src.index("\n}\n", fn_start) + len("\n}\n")
    return src[start:end]


def _run_scale_default(arch_root, attr, fallback, scale=None, cwd=None):
    import os
    import sys
    import subprocess as sp
    snippet = _source_scale_default_snippet()
    # The real script calls bare `python3` (correct on the vast.ai box);
    # this dev environment's venv python may not be `python3` on PATH,
    # so substitute the CURRENT interpreter (which has neuroslm
    # importable) for exactly this test — the shell logic under test
    # (scale-vs-top-level resolution) is unaffected by which python
    # runs the embedded snippet.
    py = sys.executable.replace("\\", "/")
    snippet = snippet.replace("python3 -", f'"{py}" -')
    script = snippet + f'\n_scale_default {attr} {fallback}\n'
    env = dict(os.environ)
    env.pop("SCALE", None)
    if scale is not None:
        env["SCALE"] = scale
    # The extracted snippet's first line is `_arch_root="architectures/
    # ${ARCH}"` — ARCH must be set (the leaf name) for it to resolve to
    # the temp arch dir, exactly as the real script expects.
    env["ARCH"] = arch_root.removeprefix("architectures/")
    r = sp.run([_bash(), "-c", script], cwd=str(cwd or SCRIPT.parent.parent),
              capture_output=True, text=True, env=env)
    return r.stdout.strip(), r


def test_scale_default_function_exists():
    src = _text()
    assert "_scale_default() {" in src, (
        "the scale-aware resolver must exist — _arch_default alone reads "
        "the wrong (top-level) config field for seq_len/batch_size")


def test_batch_and_seq_len_use_the_scale_aware_resolver():
    src = _text()
    assert 'BATCH="${BATCH:-$(_scale_default batch_size 4)}"' in src
    assert 'SEQ_LEN="${SEQ_LEN:-$(_scale_default seq_len 1024)}"' in src
    # must NOT still be calling the scale-blind resolver for these two
    assert '_arch_default batch_size' not in src
    assert '_arch_default seq_len' not in src


def test_scale_variant_value_wins_when_scale_is_set(tmp_path):
    """The actual bug: a scales{} variant's seq_len/batch_size must be
    used when SCALE selects it — this is what silently broke every
    scaled deploy this session."""
    arch_dir = tmp_path / "architectures" / "probe-arch"
    arch_dir.mkdir(parents=True)
    (arch_dir / "arch.neuro").write_text(textwrap.dedent("""
        architecture probe: { d_sem: 64, dt: 0.01 }
        training {
            preset: "rcc_bowtie_30m_p4"
            scales: {
                default: "test_scale"
                test_scale: {
                    d_model: 64
                    depth: 2
                    n_heads: 2
                    max_ctx: 999
                    batch_size: 77
                    seq_len: 999
                    grad_accum: 1
                }
            }
        }
    """), encoding="utf-8")
    rel = "architectures/probe-arch"
    out, r = _run_scale_default(rel, "seq_len", "1024", scale="test_scale",
                                cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert out == "999", (
        f"expected the scale variant's seq_len=999, got {out!r} "
        f"(stderr: {r.stderr})")

    out, r = _run_scale_default(rel, "batch_size", "4", scale="test_scale",
                                cwd=tmp_path)
    assert out == "77", (
        f"expected the scale variant's batch_size=77, got {out!r} "
        f"(stderr: {r.stderr})")


def test_falls_back_to_top_level_default_when_scale_unset(tmp_path):
    """Backward compatibility: an arch with no active SCALE (or no
    scales{} block at all) must keep the pre-fix behaviour, not crash."""
    arch_dir = tmp_path / "architectures" / "no-scale-arch"
    arch_dir.mkdir(parents=True)
    (arch_dir / "arch.neuro").write_text(textwrap.dedent("""
        architecture probe: { d_sem: 64, dt: 0.01 }
        training {
            preset: "rcc_bowtie_30m_p4"
            seq_len: 512
        }
    """), encoding="utf-8")
    rel = "architectures/no-scale-arch"
    out, r = _run_scale_default(rel, "seq_len", "1024", scale=None,
                                cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert out == "512", (
        f"with no SCALE set, must fall back to the top-level declared "
        f"value, got {out!r}")
