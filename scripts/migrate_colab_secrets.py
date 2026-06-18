# -*- coding: utf-8 -*-
"""One-shot migration: rewrite the duplicated Colab secret try/except
chains in colab_run.ipynb to use neuroslm.utils.secrets.get_secret.

Pattern matched (5 occurrences as of 2026-06-18):

    _tok = os.environ.get("GITHUB", "").strip()
    if not _tok:
        try:
            from google.colab import userdata as _ud
            _tok = (_ud.get("GITHUB") or "").strip()
        except Exception: pass

Replaced with:

    from neuroslm.utils.secrets import get_secret
    _tok = get_secret("GITHUB", aliases=("GITHUB_TOKEN", "GH_TOKEN")) or ""

Variable name (``_tok`` vs ``_token``) and key name (``GITHUB`` vs
``HF_TOKEN``) are preserved by capturing them in the regex.

Idempotent: re-running has no effect because the replacement no longer
contains ``from google.colab import userdata``.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

NB = Path(r"c:\Users\morrossl\Documents\Private\SLM\colab_run.ipynb")


# The literal try/except chain we want to collapse.
# Captures:
#   1. variable name (\w+, e.g. _tok / _token / _hf_tok)
#   2. secret key  ("GITHUB" / "HF_TOKEN" / etc.)
PATTERN = re.compile(
    r'(\w+) = os\.environ\.get\("([^"]+)", ""\)\.strip\(\)\n'
    r'if not \1:\n'
    r'    try:\n'
    r'        from google\.colab import userdata as _ud\n'
    r'        \1 = \(_ud\.get\("\2"\) or ""\)\.strip\(\)\n'
    r'    except Exception: pass'
)


# Inverted-order variant used in cell 3 (clone step):
#
#     _token = ""
#     try:
#         from google.colab import userdata as _ud
#         _token = (_ud.get("GITHUB") or "").strip()
#     except Exception:
#         _token = os.environ.get("GITHUB", "").strip()
PATTERN_INVERTED = re.compile(
    r'(\w+) = ""\n'
    r'try:\n'
    r'    from google\.colab import userdata as _ud\n'
    r'    \1 = \(_ud\.get\("([^"]+)"\) or ""\)\.strip\(\)\n'
    r'except Exception:\n'
    r'    \1 = os\.environ\.get\("\2", ""\)\.strip\(\)'
)


# Cell 7 variant — try block contains credential-write side effects.
# We hoist those side effects out so they ALWAYS run when the token
# resolves (regardless of which backend supplied it).
#
#     _tok = os.environ.get("GITHUB", "").strip()
#     if not _tok:
#         try:
#             from google.colab import userdata as _ud
#             _tok = (_ud.get("GITHUB") or "").strip()
#             if _tok:
#                 os.environ["GITHUB"] = _tok
#                 with open(os.path.expanduser("~/.git-credentials"), "w") as _f:
#                     _f.write(f"https://{_tok}@github.com\n")
#                 subprocess.run(["git", "config", "--global", "credential.helper", "store"], check=False)
#         except Exception: pass
PATTERN_WITH_SIDE_EFFECTS = re.compile(
    r'(\w+) = os\.environ\.get\("([^"]+)", ""\)\.strip\(\)\n'
    r'if not \1:\n'
    r'    try:\n'
    r'        from google\.colab import userdata as _ud\n'
    r'        \1 = \(_ud\.get\("\2"\) or ""\)\.strip\(\)\n'
    r'        if \1:\n'
    r'            os\.environ\["\2"\] = \1\n'
    r'            with open\(os\.path\.expanduser\("~/.git-credentials"\), "w"\) as _f:\n'
    r'                _f\.write\(f"https://\{\1\}@github\.com\\n"\)\n'
    r'            subprocess\.run\(\["git", "config", "--global", "credential\.helper", "store"\], check=False\)\n'
    r'    except Exception: pass'
)


# Per-secret alias map: which alternate env-var names to try.
ALIASES = {
    "GITHUB":   ('"GITHUB_TOKEN", "GH_TOKEN"',),
    "HF_TOKEN": ('"HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN"',),
}


def _replacement(match: re.Match) -> str:
    var, key = match.group(1), match.group(2)
    alias_tup = ALIASES.get(key, ())
    alias_arg = f", aliases=({alias_tup[0]},)" if alias_tup else ""
    return (
        f'from neuroslm.utils.secrets import get_secret\n'
        f'{var} = get_secret("{key}"{alias_arg}) or ""'
    )


def _replacement_with_side_effects(match: re.Match) -> str:
    """Same as _replacement, but re-emits the credential side-effects
    OUTSIDE the resolver — they now fire whenever the token resolves,
    not just on the Colab-userdata branch."""
    var, key = match.group(1), match.group(2)
    alias_tup = ALIASES.get(key, ())
    alias_arg = f", aliases=({alias_tup[0]},)" if alias_tup else ""
    return (
        f'from neuroslm.utils.secrets import get_secret\n'
        f'{var} = get_secret("{key}"{alias_arg}) or ""\n'
        f'if {var}:\n'
        f'    with open(os.path.expanduser("~/.git-credentials"), "w") as _f:\n'
        f'        _f.write(f"https://{{{var}}}@github.com\\n")\n'
        f'    subprocess.run(["git", "config", "--global", "credential.helper", "store"], check=False)'
    )


def main() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    total_subs = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        # Apply most-specific patterns FIRST so the generic one doesn't
        # eat the cell-7 prefix and leave the side-effect block orphaned.
        src, n1 = PATTERN_WITH_SIDE_EFFECTS.subn(
            _replacement_with_side_effects, src,
        )
        src, n2 = PATTERN_INVERTED.subn(_replacement, src)
        src, n3 = PATTERN.subn(_replacement, src)
        n = n1 + n2 + n3
        if n == 0:
            continue
        total_subs += n
        # Notebook source is stored as a list of lines (each ending in \n
        # except possibly the last). Round-trip via splitlines(keepends=True).
        cell["source"] = src.splitlines(keepends=True)
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Rewrote {total_subs} try/except chains in {NB.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
