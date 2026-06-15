"""Contracts for the LM-trunk + multicortex wiring on the canonical
``architectures/master/`` graph.

Pinned by the 2026-06-15 NFG cross-alignment audit, which found that:

1. **Fix A — the three cortex_* specialists were dangling sinks.**
   ``thalamus -> cortex_{code,general,reasoning}`` provided the only
   edges they participated in. Their actual training-gradient output
   lives in the ``funnel teacher_ensemble { target: pfc }`` IR row
   (declarative MoE distillation), but the NFG renders only synapses
   + modulations, so they showed up as terminal nodes. This file
   pins the forward synapse ``cortex_* -> pfc`` so the visible graph
   matches the actual MoE→LM gradient.

2. **Fix B — ``lib/LanguageCortex.neuro`` defaulted to a population
   that doesn't exist.**  Its ``target: lm_trunk`` default pointed at
   ``lm_trunk``, which is not in any population roster in the repo;
   the master arch correctly overrode with ``target: pfc``, but any
   future call site that forgot to override would compile a funnel
   into a non-existent population. The new default is ``pfc`` —
   the population that plays the LM-trunk role in the bowtie.

3. **Fix C — the ``brain`` module had zero NFG-visible edges.**  The
   ``module brain = LanguageCortex { ... }`` produces a ``funnel``
   IR row that projects expert ensemble outputs into ``pfc``. That's
   a *distillation gradient*, not a forward synapse, so the renderer
   never drew it. Fix C teaches the NFG IR to emit per-expert
   ``funnel`` edges (``cortex_* -> target``, ``kind="funnel"``) so the
   brain module's projection is visible on the graph.

Without these three contracts, the NFG diagram silently misrepresents
the actual gradient topology of the architecture.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
MASTER_ARCH = REPO_ROOT / "architectures" / "master" / "arch.neuro"
LANG_CORTEX = REPO_ROOT / "lib" / "LanguageCortex.neuro"


def _parse_synapses(arch_text: str) -> list[tuple[str, str]]:
    """Return [(src, tgt)] for every ``synapse SRC -> TGT { ... }`` row."""
    return re.findall(r"^\s*synapse\s+(\w+)\s*->\s*(\w+)\s*\{", arch_text, flags=re.M)


def _out_degree(arch_text: str) -> dict[str, int]:
    out = defaultdict(int)
    for src, _ in _parse_synapses(arch_text):
        out[src] += 1
    return out


# ──────────────────────────────────────────────────────────────────────
# Fix A — cortex_* specialists must have an outgoing synapse
# ──────────────────────────────────────────────────────────────────────

class TestFixACortexSpecialistsHaveOutgoingEdge:
    """The three MoE expert populations (``cortex_code``,
    ``cortex_general``, ``cortex_reasoning``) must each have at least
    one outgoing synapse. The canonical target is ``pfc`` because the
    declarative ``funnel teacher_ensemble`` already projects there;
    the synapse mirrors the funnel so the gradient topology is
    NFG-visible."""

    @pytest.fixture(scope="class")
    def arch_text(self) -> str:
        return MASTER_ARCH.read_text(encoding="utf-8")

    @pytest.mark.parametrize("cortex", ["cortex_code", "cortex_general", "cortex_reasoning"])
    def test_cortex_has_outgoing_synapse(self, arch_text: str, cortex: str):
        out = _out_degree(arch_text)
        assert out[cortex] > 0, (
            f"{cortex!r} is a dangling sink — no outgoing synapse "
            f"in architectures/master/arch.neuro. The MoE→LM gradient "
            f"flows through the declarative `funnel teacher_ensemble "
            f"{{ target: pfc }}` row, but the NFG renders only "
            f"synapses + modulations, so without an explicit "
            f"`synapse {cortex} -> pfc` row the population looks "
            f"orphan on the rendered graph."
        )

    @pytest.mark.parametrize("cortex", ["cortex_code", "cortex_general", "cortex_reasoning"])
    def test_cortex_projects_to_pfc(self, arch_text: str, cortex: str):
        """The canonical target is ``pfc`` (matches
        ``funnel teacher_ensemble { target: pfc }``)."""
        syns = _parse_synapses(arch_text)
        targets = {tgt for src, tgt in syns if src == cortex}
        assert "pfc" in targets, (
            f"{cortex} must have `synapse {cortex} -> pfc` (mirrors "
            f"the funnel target in the declarative `module brain = "
            f"LanguageCortex {{ target: pfc }}` block). Found "
            f"outgoing targets: {sorted(targets)}"
        )


# ──────────────────────────────────────────────────────────────────────
# Fix B — LanguageCortex lib default target must be an existing pop
# ──────────────────────────────────────────────────────────────────────

class TestFixBLanguageCortexDefaultTarget:
    """The ``lib/LanguageCortex.neuro`` default ``target:`` must point
    at a population that actually exists in the master arch, so that
    any future call site which forgets to override ``target:``
    compiles into a real funnel instead of a dangling reference."""

    @pytest.fixture(scope="class")
    def lib_text(self) -> str:
        return LANG_CORTEX.read_text(encoding="utf-8")

    def test_default_target_is_pfc(self, lib_text: str):
        """The lib's params default must be ``target: pfc``."""
        # Look in the params block for `target: <name>`. We tolerate
        # whitespace + comma but not arbitrary garbage.
        m = re.search(
            r"params\s*:\s*\{[^}]*?\btarget\s*:\s*(\w+)",
            lib_text,
            flags=re.S,
        )
        assert m is not None, (
            "lib/LanguageCortex.neuro params block has no `target:` "
            "default — every funnel needs a target population"
        )
        assert m.group(1) == "pfc", (
            f"lib/LanguageCortex.neuro default target is "
            f"{m.group(1)!r}; expected 'pfc' so the default points "
            f"at a population that actually exists in the master "
            f"arch (`lm_trunk` was the old default but does not "
            f"appear in any populations roster — see the 2026-06-15 "
            f"NFG cross-alignment audit)."
        )

    def test_default_target_population_exists_in_master(self, lib_text: str):
        """Whatever the default target is, it must be a member of
        the master arch's ``populations:`` list. Catches drift if
        somebody changes the default later."""
        m = re.search(
            r"params\s*:\s*\{[^}]*?\btarget\s*:\s*(\w+)",
            lib_text,
            flags=re.S,
        )
        assert m is not None
        default_target = m.group(1)

        arch_text = MASTER_ARCH.read_text(encoding="utf-8")
        trunk = re.search(r"populations:\s*\[(.*?)\]", arch_text, flags=re.S)
        assert trunk, "master arch.neuro missing `populations: [...]` block"
        pops = {p.strip() for p in trunk.group(1).split(",")}

        assert default_target in pops, (
            f"lib/LanguageCortex.neuro default target "
            f"{default_target!r} is not in the master arch's "
            f"populations roster {sorted(pops)} — a default that "
            f"points at a non-existent population is a footgun "
            f"(any call site that omits `target:` compiles into a "
            f"dangling funnel)."
        )

    def test_no_stale_lm_trunk_reference_remains(self, lib_text: str):
        """The string ``lm_trunk`` must not appear in the lib at all
        after the fix — neither in the params default nor in the
        body. Stale references confuse readers and make grep noisy."""
        # Allow it inside a comment that documents the migration:
        # only literal `lm_trunk` outside of `#` comments is forbidden.
        non_comment_lines = [
            ln.split("#", 1)[0] for ln in lib_text.splitlines()
        ]
        non_comment_body = "\n".join(non_comment_lines)
        assert "lm_trunk" not in non_comment_body, (
            "lib/LanguageCortex.neuro still mentions `lm_trunk` "
            "outside of comments — the population does not exist "
            "in the master arch and the new default is `pfc`. "
            "Comment-only mentions (e.g. migration notes) are OK."
        )
