"""LM-Expert mixture-of-experts: every expert returns trunk-vocab logits.

Architectural change vs. the legacy ``MultiCortexEnsemble`` path
================================================================

Legacy chain (broken, see scripts/diagnose_cortex_init.py):

    ids ──▶ GPT2Backbone ──▶ hidden(B,T,768)
                              │
                              ▼   random Xavier
                          Linear(768→512)
                              │
                              ▼
                          LayerNorm   ◀── "cortex_pre_head_norm" band-aid
                              │
                              ▼   tied to RANDOM trunk embedding
                          Linear(512→V_trunk)
                              │
                              ▼
                          softmax → CE ≈ ln(V) (uniform baseline)

The frozen 700M of pretrained GPT-2 weights produced real information
that two stacked random matrices then converted back to noise.

New chain (this module):

    ids ──▶ GPT2LMHeadModel ──▶ logits(B, T, V_expert)   [pretrained head]
                                  │
                                  ▼   VocabBridge.apply
                              logits(B, T, V_trunk)
                                  │
                                  ▼
                              softmax → CE ≈ 3-5 nats (real LM)

Two paths
=========

* **Fast path** (same tokenizer as trunk): ``model(ids).logits`` directly.
  Same speed as the legacy backbone forward, but with the pretrained
  head wired in. Initial CE on natural English drops from ~10.85 to
  ~3-5 nats overnight.

* **Bridge path** (different tokenizer, e.g. Qwen): per-sample retoken
  + char-offset alignment + sparse vocab bridge. The bridge maps each
  trunk vocab id to the closest matching expert vocab id by
  string-equality of the decoded surface form; trunk ids with no
  expert equivalent get masked to a very-negative logit (≈ "expert
  abstains"). Adds ~10% step time at seq=2048; only invoked for
  experts whose tokenizer differs from the trunk's.

References
==========
* Timkey & van Schijndel 2021, "All Bark and No Bite: Rogue Dimensions
  in Transformer Language Models Obscure Representational Quality"
  — explains the rogue-dim anisotropy that made the legacy path
  catastrophic. With the new path the expert's own LayerNorm sees the
  rogue dim during pretraining; we don't have to fight it.
* Standard MoE topology: every expert outputs in the SAME logit space,
  router weights mix per-token. See Switch Transformer / Mixtral.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "LMExpert",
    "LMExpertEnsemble",
    "VocabBridge",
    "_LOGIT_CHUNK_T",
    "_align_by_char_offsets",
    "_align_by_char_offsets_exact",
    "_load_lm_cached",
    "_load_tokenizer_cached",
    "_split_lm",
    "build_lm_expert_ensemble",
    "register_expert_alias",
    "resolve_expert_alias",
]


# ──────────────────────────────────────────────────────────────────────
# Expert model alias registry — DSL-level plug-and-play (Item 5)
# ──────────────────────────────────────────────────────────────────────
#
# The DSL author writes a short alias (``smollm2_360m``), a full HF
# model id (``HuggingFaceTB/SmolLM2-360M``), or an ``hf://`` URL. The
# resolver normalises any of these to a canonical HF model id, which
# is what every loading path inside ``LMExpert`` actually consumes.
#
# Why a registry instead of "just pass the HF id":
# 1. **Typo safety on bare names.** A bare alias without ``/`` (e.g.
#    ``smollm_360m``) is checked against the registry; if missing, we
#    raise a ValueError listing the known aliases. The user finds the
#    typo before the HF hub returns a 404.
# 2. **Stable short names in arch.neuro.** ``smollm2_360m`` is easier
#    to read in a 1200-line arch file than ``HuggingFaceTB/SmolLM2-360M``
#    and gives us a single place to update if a vendor renames a repo.
# 3. **One canonical form per model.** The cache (``_LM_CACHE``,
#    ``_TOKENIZER_CACHE``, ``_VOCAB_BRIDGE_CACHE``) is keyed on the
#    canonical id, so loading via an alias and via its HF id share one
#    cache entry.
#
# Registration:
#   * Built-in entries below cover the families we ship support for.
#   * ``register_expert_alias("new_alias", "owner/repo")`` extends the
#     registry at runtime (used by tests; production arch.neuro should
#     prefer the canonical HF id for novel models).
_EXPERT_ALIAS_REGISTRY: "dict[str, str]" = {
    # Legacy GPT-2 family — bare names are HF-canonical (no owner prefix
    # needed; the HF hub accepts them directly).
    "gpt2":         "gpt2",
    "gpt2-medium":  "gpt2-medium",
    "gpt2-large":   "gpt2-large",
    "gpt2-xl":      "gpt2-xl",
    "distilgpt2":   "distilgpt2",
    # SmolLM2 — HuggingFaceTB's compact, 4T-token-trained family.
    # Card: https://huggingface.co/HuggingFaceTB/SmolLM2-360M
    "smollm2_135m": "HuggingFaceTB/SmolLM2-135M",
    "smollm2_360m": "HuggingFaceTB/SmolLM2-360M",
    "smollm2_1_7b": "HuggingFaceTB/SmolLM2-1.7B",
    # Qwen2.5 — strong small-LM reasoning baseline.
    "qwen2_5_0_5b": "Qwen/Qwen2.5-0.5B",
    "qwen2_5_1_5b": "Qwen/Qwen2.5-1.5B",
    # Microsoft code SLM — the current `code` expert in arch.neuro.
    "codegpt_py":   "microsoft/CodeGPT-small-py",
}


def resolve_expert_alias(model_id_or_alias: str) -> str:
    """Resolve a DSL-side expert id to a canonical HuggingFace model id.

    Resolution rules (first match wins):

    1. ``hf://owner/repo``  →  ``owner/repo`` (URL scheme strip).
    2. ``owner/repo``       →  identity (already canonical; we trust
                               any string with a ``/`` because checking
                               existence would require a network call).
    3. Bare name in the registry → registered canonical id.
    4. Bare name NOT in the registry → ``ValueError`` listing the
       known aliases (typo safety).

    The function is **pure** — no HF I/O, no caching. Safe to call
    in tests and DSL parsing.
    """
    if not isinstance(model_id_or_alias, str) or not model_id_or_alias:
        raise ValueError(
            f"resolve_expert_alias: expected a non-empty string, "
            f"got {model_id_or_alias!r}"
        )
    s = model_id_or_alias.strip()

    # Rule 1: URL scheme strip.
    if s.startswith("hf://"):
        return s[len("hf://"):]

    # Rule 2: owner/repo form is always canonical (trust + typo escape).
    if "/" in s:
        return s

    # Rule 3: bare name in registry.
    if s in _EXPERT_ALIAS_REGISTRY:
        return _EXPERT_ALIAS_REGISTRY[s]

    # Rule 4: typo. List known aliases sorted for a stable error message.
    known = sorted(_EXPERT_ALIAS_REGISTRY.keys())
    raise ValueError(
        f"resolve_expert_alias: unknown expert alias {s!r}. "
        f"Either use a HuggingFace ``owner/repo`` id, the ``hf://`` URL "
        f"form, or register an alias first. Known aliases: {known}"
    )


def register_expert_alias(alias: str, canonical_hf_id: str) -> None:
    """Register a new alias → canonical HF id mapping.

    ``alias``         must be a bare name (no ``/``); enforced.
    ``canonical_hf_id`` must contain ``/`` (i.e. ``owner/repo``),
                       except for the legacy gpt2 family which is
                       pre-registered with bare names.

    Idempotent: re-registering with the same target is a no-op;
    re-registering with a different target overwrites.
    """
    if not isinstance(alias, str) or not alias or "/" in alias:
        raise ValueError(
            f"register_expert_alias: alias must be a bare name "
            f"(no ``/``), got {alias!r}"
        )
    if not isinstance(canonical_hf_id, str) or not canonical_hf_id:
        raise ValueError(
            f"register_expert_alias: canonical_hf_id must be a "
            f"non-empty string, got {canonical_hf_id!r}"
        )
    # Refuse to register a bare-name target — that's an alias chain
    # waiting to break. The only exception is the legacy gpt2 family
    # which the hub accepts as canonical (and is pre-seeded above).
    if "/" not in canonical_hf_id:
        legacy_ok = {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
                     "distilgpt2"}
        if canonical_hf_id not in legacy_ok:
            raise ValueError(
                f"register_expert_alias: canonical_hf_id {canonical_hf_id!r} "
                f"must be of the form ``owner/repo``. Bare names are only "
                f"allowed for the pre-seeded legacy gpt2 family."
            )
    _EXPERT_ALIAS_REGISTRY[alias] = canonical_hf_id


# ──────────────────────────────────────────────────────────────────────
# Process-wide pretrained-weight cache
# ──────────────────────────────────────────────────────────────────────
#
# ``AutoModelForCausalLM.from_pretrained`` reads ~500MB-1GB of safetensors
# off disk and constructs an ``nn.Module`` graph — 5-8 seconds per call
# on a warm OS cache, longer cold. Across our test suite the same model
# id (typically "gpt2") is requested 6+ times in the same Python process,
# costing ~30s of pure I/O before any actual test logic runs. Production
# training also instantiates the same expert in multiple harness paths
# (smoking-gun CE probe, forward-shape check, ensemble construction).
#
# Frozen experts (``freeze=True``, the default and the production case)
# are stateless during forward — sharing the underlying ``nn.Module``
# between ``LMExpert`` instances cannot leak gradients or mutate weights.
# So we cache aggressively for the frozen path and bypass for the rare
# unfrozen case (training a teacher).
#
# Keyed on ``model_id`` only. ``device`` placement happens later via
# ``.to(device)`` and is per-instance — moving a shared module to a
# different device would corrupt other holders, so we never call
# ``.to()`` on the cached object itself.
_LM_CACHE: "dict[str, object]" = {}
_TOKENIZER_CACHE: "dict[str, object]" = {}
# Memoised VocabBridge instances. Keyed on the public identity of the
# tokenizer pair so that test code creating two fresh tokenizer objects
# for the same model id still hits the cache.
_VOCAB_BRIDGE_CACHE: "dict[tuple, object]" = {}


def _load_lm_cached(model_id: str):
    """Return the cached ``AutoModelForCausalLM`` for ``model_id``,
    loading it from HF on first call. Subsequent callers receive the
    SAME object — safe only for frozen / eval-mode use.

    Loader dispatch (torch < 2.6 / CVE-2025-32434 aware)
    ---------------------------------------------------
    1. **Prefer safetensors**: pass ``use_safetensors=True`` so HF
       bypasses ``torch.load`` entirely when the repo ships a
       ``model.safetensors`` file. This is the only loader path that
       works on torch < 2.6 without raising the CVE-2025-32434
       check.
    2. **Fall back for legacy .bin repos** (e.g. ``gpt2``): retry
       with ``use_safetensors=False`` + ``weights_only=False`` so the
       legacy checkpoint loads on torch < 2.6. Safe in our use case —
       every ``model_id`` we load comes from a hard-coded ``arch.neuro``
       config (no user-controlled paths), and ``weights_only=False`` is
       the only way to load pre-safetensors checkpoints on torch < 2.6
       without forcing a system-package upgrade.

    Regression-pinned by
    ``tests/training/test_lm_expert_safetensors_loader.py``.
    """
    cached = _LM_CACHE.get(model_id)
    if cached is not None:
        return cached
    from transformers import AutoModelForCausalLM

    # Load in bfloat16: frozen expert output is already bf16, so the
    # fusion loop's `.to(bf16)` becomes a no-op.  Without this, each
    # expert produces a fp32 (B, T, V) tensor (6.1 GB at B=16 T=2048
    # V=50257) and `.to(bf16)` allocates a second 3.1 GB copy while the
    # fp32 is still alive — 9.2 GB peak just for the conversion → OOM.
    _bf16 = torch.bfloat16

    # Path 1: safetensors-only (works on every torch version)
    try:
        lm = AutoModelForCausalLM.from_pretrained(
            model_id, use_safetensors=True, dtype=_bf16,
        )
    except Exception as exc:
        msg = str(exc)
        # Distinguish the CVE-2025-32434 path from a real failure.
        # The exact error string is documented; match on the stable
        # substring ``torch.load`` + ``v2.6`` which together are
        # unique to this version-restriction error.
        is_cve = (
            "torch.load" in msg
            and ("v2.6" in msg or "weights_only" in msg)
        )
        # ``no file named model.safetensors`` is what HF says when
        # the repo has no safetensors at all — also a fall-back case.
        is_no_safetensors = (
            "safetensors" in msg.lower()
            and ("not found" in msg.lower() or "no file" in msg.lower())
        )
        if not (is_cve or is_no_safetensors):
            raise
        # Path 2: legacy .bin with weights_only=False. The CVE check
        # is bypassed in this mode for repos that pre-date safetensors.
        import warnings
        warnings.warn(
            f"_load_lm_cached({model_id!r}): safetensors load failed "
            f"({type(exc).__name__}); retrying with use_safetensors=False "
            f"and weights_only=False. This is required for legacy .bin "
            f"checkpoints (e.g. gpt2) on torch < 2.6.",
            RuntimeWarning,
        )
        lm = AutoModelForCausalLM.from_pretrained(
            model_id,
            use_safetensors=False,
            weights_only=False,
            dtype=_bf16,
        )

    lm.eval()
    _LM_CACHE[model_id] = lm
    return lm


def _load_tokenizer_cached(model_id: str):
    """Return the cached ``AutoTokenizer`` for ``model_id``.

    Tokenizers are stateless after construction, so sharing is
    unconditionally safe. Returns ``None`` if the repo has no
    tokenizer; callers should fall back to the trunk tokenizer."""
    if model_id in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_id]
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
    except Exception:
        tok = None
    _TOKENIZER_CACHE[model_id] = tok
    return tok


# Default fallback magnitude when no max-mapped reference is available
# (degenerate all-unmapped position). The bridge ``apply`` method
# computes a PER-POSITION abstain value relative to the mapped max:
# ``abstain = max(mapped_logits) - ln(V_trunk)``. This gives each
# unmapped slot ~uniform-baseline probability mass ``1/V_trunk``,
# yielding CE ≈ ln(V_trunk) on unmapped targets (the principled
# "expert genuinely doesn't know" baseline).
#
# Why per-position-relative not a global constant:
# Pretrained LM logits have arbitrary additive baselines (gpt2 sits
# around -65; a freshly-init head sits around 0). A global negative
# constant (the old ``-1e4``) blows up ``cortex_loss_ema`` because
# the harness's ``_cortex_fusion_aux_step`` recomputes
#   ``ce_cx = F.cross_entropy(cortex_logits.float(), targets)``
# and unmapped-target CE ≈ |abstain| ≈ 10000 nats. A global modest
# constant (e.g. -12) can be ABOVE the expert's mapped logits and
# dominate the softmax → CE blows up the other way (observed
# 17-nat ensemble CE on plain English with two gpt2-family experts).
# The relative formulation is invariant to the expert's logit baseline.
# Regression-pinned by ``tests/training/test_lm_expert_abstain_safety.py``.
_ABSTAIN_LOGIT: float = -10.0  # legacy constant kept for back-compat
                               # callers that don't use VocabBridge.apply


# ──────────────────────────────────────────────────────────────────────
# Vocab bridge: trunk vocab id → expert vocab id (or -1)
# ──────────────────────────────────────────────────────────────────────


class VocabBridge:
    """Maps trunk vocab ids to expert vocab ids by string-equality.

    Built once per ``(trunk_tokenizer, expert_tokenizer)`` pair. For
    same-tokenizer pairs, ``is_identity == True`` and ``apply`` is a
    no-op pass-through. For different tokenizers, ``coverage`` reports
    the fraction of trunk vocab that has a string-equivalent in the
    expert vocab; tokens with no equivalent map to ``-1`` and their
    bridged logit is set to ``_ABSTAIN_LOGIT``.

    The bridge stores its main table as a ``LongTensor[V_trunk]`` so
    application is a single ``gather`` on the last dim.
    """

    __slots__ = (
        "trunk_to_expert",
        "is_identity",
        "coverage",
        "vocab_size_trunk",
        "vocab_size_expert",
    )

    def __init__(
        self,
        trunk_to_expert: torch.Tensor,
        is_identity: bool,
        coverage: float,
        vocab_size_trunk: int,
        vocab_size_expert: int,
    ) -> None:
        self.trunk_to_expert = trunk_to_expert
        self.is_identity = is_identity
        self.coverage = coverage
        self.vocab_size_trunk = vocab_size_trunk
        self.vocab_size_expert = vocab_size_expert

    @classmethod
    def build(
        cls,
        trunk_tokenizer,
        expert_tokenizer,
    ) -> "VocabBridge":
        """Construct a bridge table from two HF tokenizers.

        Same-tokenizer fast path: builds an identity tensor (``coverage=1.0``).

        Different-tokenizer path: walks every trunk vocab id, decodes
        it to its surface string, re-encodes with the expert tokenizer,
        and records the *first* expert id when the encoded sequence
        has length 1. Multi-token expert encodings are skipped (mapped
        to ``-1``) because there's no canonical single-token expert
        equivalent for "this trunk token".

        Memoised on ``(trunk.name_or_path, expert.name_or_path,
        v_trunk, v_expert)``. The cross-tok build walks the entire
        trunk vocab and is O(V) Python loops — paying it once per
        ``(trunk, expert)`` pair instead of once per ``LMExpert`` is
        the difference between a ~3s init and a ~30ms init.
        """
        # Build a cache key from the public identity of the tokenizers.
        # ``name_or_path`` is the HF model id (e.g. "gpt2"); vocab_size
        # disambiguates fine-tunes that reuse the same base id with an
        # extended vocab.
        v_trunk = int(trunk_tokenizer.vocab_size)
        v_expert = int(expert_tokenizer.vocab_size)
        cache_key = (
            str(getattr(trunk_tokenizer, "name_or_path", id(trunk_tokenizer))),
            str(getattr(expert_tokenizer, "name_or_path", id(expert_tokenizer))),
            v_trunk,
            v_expert,
        )
        cached = _VOCAB_BRIDGE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        # Same-tokenizer detection: identical class + identical vocab size
        # AND a sample of token strings round-trips identically.
        if (
            v_trunk == v_expert
            and type(trunk_tokenizer) is type(expert_tokenizer)
            and getattr(trunk_tokenizer, "name_or_path", None)
                == getattr(expert_tokenizer, "name_or_path", None)
        ):
            bridge = cls(
                trunk_to_expert=torch.arange(v_trunk, dtype=torch.long),
                is_identity=True,
                coverage=1.0,
                vocab_size_trunk=v_trunk,
                vocab_size_expert=v_expert,
            )
            _VOCAB_BRIDGE_CACHE[cache_key] = bridge
            return bridge

        # Cross-tokenizer: build by surface-string equality
        bridge_tensor = torch.full((v_trunk,), -1, dtype=torch.long)
        n_mapped = 0
        for tid in range(v_trunk):
            try:
                s = trunk_tokenizer.decode([tid])
            except Exception:
                continue
            if not s:
                continue
            try:
                eids = expert_tokenizer.encode(s, add_special_tokens=False)
            except Exception:
                continue
            if len(eids) == 1 and 0 <= eids[0] < v_expert:
                bridge_tensor[tid] = eids[0]
                n_mapped += 1

        coverage = n_mapped / max(1, v_trunk)
        bridge = cls(
            trunk_to_expert=bridge_tensor,
            is_identity=False,
            coverage=float(coverage),
            vocab_size_trunk=v_trunk,
            vocab_size_expert=v_expert,
        )
        _VOCAB_BRIDGE_CACHE[cache_key] = bridge
        return bridge

    def apply(self, expert_logits: torch.Tensor) -> torch.Tensor:
        """Project ``(..., V_expert)`` logits into ``(..., V_trunk)`` space.

        For same-tok bridges this is a no-op (returns the input). For
        cross-tok bridges, gathers expert logits by the trunk→expert
        table and fills unmapped trunk slots with a PER-POSITION abstain
        value such that each unmapped slot carries roughly uniform
        baseline mass (``1/V_trunk``) in the softmax.

        Why per-position-relative
        -------------------------
        Pretrained LM logits have an arbitrary additive baseline
        (gpt2's sit around -65; a freshly-init head sits around 0).
        A *global* abstain constant has two failure modes:

        * Too small (e.g. ``-1e4``): unmapped-target cross-entropy
          explodes to ~10 000 nats per such position, poisoning
          ``cortex_loss_ema`` and forcing ``α_eff → 0`` (deploy
          40923107).
        * Too large (e.g. ``-12``): abstain logits may sit ABOVE the
          expert's mapped logits (gpt2 baseline ~-65) and dominate
          the softmax → CE blows up the other way (~17 nats observed).

        The per-position formulation ``abstain = max(mapped) -
        ln(V_trunk)`` is invariant to the expert's logit baseline and
        always yields ``p(unmapped slot) ≈ 1/V_trunk`` — exactly the
        uniform fallback semantic of "expert abstains".

        Regression-pinned by
        ``tests/training/test_lm_expert_abstain_safety.py``.
        """
        if self.is_identity:
            return expert_logits

        # Move the bridge to the same device as the logits on first call
        idx = self.trunk_to_expert.to(expert_logits.device)
        # Replace -1 with 0 to avoid out-of-range gather; we mask after
        idx_safe = idx.clamp(min=0)
        # Broadcast gather: for every (..., trunk_id), pick
        # expert_logits[..., idx[trunk_id]]
        gathered = expert_logits.index_select(-1, idx_safe)
        # Mask the unmapped slots
        mask = (idx == -1).to(gathered.device)
        if not mask.any():
            return gathered

        # Per-position abstain: ``max(mapped) - ln(V_trunk)``.
        # We need the max OVER the mapped slots only — push unmapped
        # values to -inf before taking amax so they don't contaminate
        # the reference. Use float for the masked clone to keep dtypes
        # consistent (no implicit promotion in `where`).
        neg_inf = torch.full_like(gathered, float("-inf"))
        mapped_only = torch.where(mask, neg_inf, gathered)
        max_mapped = mapped_only.amax(dim=-1, keepdim=True)  # (..., 1)
        # Degenerate case: all trunk slots unmapped in this row →
        # amax is -inf. Use 0.0 as the reference (the abstain will
        # then be -ln(V_trunk), i.e. plain uniform).
        degenerate = ~torch.isfinite(max_mapped)
        max_mapped = torch.where(
            degenerate, torch.zeros_like(max_mapped), max_mapped,
        )
        ln_v = float(torch.log(torch.tensor(
            float(self.vocab_size_trunk),
            device=gathered.device, dtype=gathered.dtype,
        )))
        abstain = max_mapped - ln_v
        # Fill the unmapped trunk slots with the per-position abstain
        # value. Mapped slots keep their expert-derived logits.
        return torch.where(mask, abstain.expand_as(gathered), gathered)


# ──────────────────────────────────────────────────────────────────────
# Char-offset alignment (the heart of the bridge path)
# ──────────────────────────────────────────────────────────────────────


def _align_by_char_offsets(
    trunk_offsets: Sequence[Tuple[int, int]],
    expert_offsets: Sequence[Tuple[int, int]],
) -> List[int]:
    """For every trunk position t, return the expert position e_t whose
    end-character offset is the smallest one ≥ trunk[t].end.

    The mapping is monotone non-decreasing (no time-travel) and covers
    every trunk position. If the expert tokenisation ends earlier than
    the trunk's (rare; happens with EOS / special tokens), the last
    expert position is returned for the trailing trunk positions.

    .. warning::
       **Legacy helper.** The bridge no longer uses this in
       ``_forward_bridge`` because the "smallest e with e_end >= t_end"
       rule yields wrong-horizon predictions when boundaries don't
       match (the expert at position e has already SEEN the trunk's
       target and is predicting content PAST trunk's horizon). Use
       :func:`_align_by_char_offsets_exact` instead, which returns
       ``-1`` at mismatched positions so the caller can abstain. Kept
       in the public surface for back-compat with tests + external
       diagnostics.
    """
    out: List[int] = []
    e_idx = 0
    n_e = len(expert_offsets)
    for t_start, t_end in trunk_offsets:
        # Advance e_idx until expert_offsets[e_idx].end >= t_end
        while (
            e_idx < n_e - 1
            and expert_offsets[e_idx][1] < t_end
        ):
            e_idx += 1
        out.append(e_idx)
    return out


def _align_by_char_offsets_exact(
    trunk_offsets: Sequence[Tuple[int, int]],
    expert_offsets: Sequence[Tuple[int, int]],
) -> List[int]:
    """For every trunk position t, return the expert position e_t whose
    end-character offset EQUALS trunk[t].end EXACTLY, or ``-1`` if no
    such expert position exists.

    Why exact-only matters (vs. the legacy "smallest e with e_end >= t_end")
    ----------------------------------------------------------------------
    In next-token prediction, trunk position t carries a hidden state
    that has consumed trunk_tokens[0..t] (decoded text up to
    ``trunk_offsets[t][1]``) and predicts trunk_token[t+1] (which
    starts at ``trunk_offsets[t][1]``).

    Expert position e carries a hidden state that has consumed
    expert_tokens[0..e] (decoded text up to ``expert_offsets[e][1]``)
    and predicts the next expert token (which starts at
    ``expert_offsets[e][1]``).

    For the expert's prediction to be a valid distillation target for
    trunk's prediction, both must be predicting content that STARTS at
    the same character offset:

        ``expert_offsets[e][1] == trunk_offsets[t][1]`` (exact match).

    When boundaries don't match, the legacy "smallest e with e_end >=
    t_end" alignment picks an expert position whose end-offset is
    STRICTLY GREATER than trunk's. Two failures compound:

    * **One-step leakage**: the expert at e has SEEN trunk's target
      as part of its input prefix.
    * **Wrong-horizon prediction**: the expert at e predicts content
      starting PAST trunk's prediction horizon. Using its logits as
      trunk's target trains trunk toward "what comes after trunk's
      target", not toward trunk's target itself.

    Empirical impact
    ----------------
    On a natural-English paragraph with gpt2 trunk + SmolLM2 expert::

        gpt2 own next-token CE     = 3.016 nats   (baseline)
        legacy smallest-ge bridge  = 3.068 nats   (+0.05 vs gpt2)
        exact-end bridge           = 2.798 nats   (-0.22 vs gpt2 !)

    ~95% of natural-English trunk positions align exactly with SmolLM2;
    the remaining ~5% abstain (uniform) instead of contributing
    wrong-horizon noise. See ``scripts/diagnose_bridge_ce.py`` for the
    full experiment and the H22 forensic in ``docs/FINDINGS.md``.

    Regression-pinned by
    ``tests/training/test_lm_expert_bridge_exact_alignment.py``.
    """
    out: List[int] = []
    e_idx = 0
    n_e = len(expert_offsets)
    for _t_start, t_end in trunk_offsets:
        # Advance e_idx until expert_offsets[e_idx].end >= t_end
        while e_idx < n_e and expert_offsets[e_idx][1] < t_end:
            e_idx += 1
        if e_idx < n_e and expert_offsets[e_idx][1] == t_end:
            out.append(e_idx)
        else:
            out.append(-1)
    return out


# ──────────────────────────────────────────────────────────────────────
# Memory-efficient forward helpers
# ──────────────────────────────────────────────────────────────────────

_LOGIT_CHUNK_T: int = 256
# Maximum positions per chunk when applying lm_head in _forward_same_tok.
# Peak GPU tensor at the head step: (B, _LOGIT_CHUNK_T, V) = 384 MB at
# B=16, V=50257 bf16 — vs 3 GB for the full T=2048 sequence.


def _split_lm(lm):
    """Return (backbone, lm_head) if the model exposes them, else (None, None).

    GPT-2:          (lm.transformer, lm.lm_head)
    Llama/SmolLM2:  (lm.model,       lm.lm_head)
    Unknown layout: (None, None) — caller falls back to lm(ids) directly.
    """
    lm_head = getattr(lm, 'lm_head', None)
    if lm_head is None:
        return None, None
    backbone = getattr(lm, 'model', None) or getattr(lm, 'transformer', None)
    if backbone is None:
        return None, None
    return backbone, lm_head


# ──────────────────────────────────────────────────────────────────────
# LMExpert
# ──────────────────────────────────────────────────────────────────────


class LMExpert(nn.Module):
    """One HuggingFace causal-LM expert that returns trunk-vocab logits.

    Construction lazily imports ``transformers`` so test environments
    without it can still import this module.

    Parameters
    ----------
    model_id : str
        HuggingFace model id (e.g. ``"gpt2-medium"``,
        ``"Qwen/Qwen2.5-0.5B"``).
    domain : str
        Routing key that will be used by the
        :class:`~neuroslm.cortex.ThalamicRouter`.
    trunk_tokenizer : PreTrainedTokenizerBase
        The tokenizer the *trunk* uses. Required because the expert
        must return logits in the trunk's vocab space; the
        :class:`VocabBridge` is built from this and the expert's own
        tokenizer.
    freeze : bool, default True
        When True, every expert parameter has ``requires_grad=False``.
        Distillation gradients still flow into the trunk; the expert
        is the teacher.
    """

    def __init__(
        self,
        model_id: str,
        domain: str,
        trunk_tokenizer,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LMExpert requires the `transformers` package. "
                "Install it with: pip install transformers"
            ) from exc

        self.model_id = resolve_expert_alias(str(model_id))
        self.domain = str(domain)
        # Frozen experts share weights process-wide (see _LM_CACHE
        # comment). Unfrozen experts bypass the cache so training
        # gradients don't leak across LMExpert instances.
        if freeze:
            self.lm = _load_lm_cached(self.model_id)
        else:
            self.lm = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.lm.eval()  # frozen experts run in eval mode
        if freeze:
            for p in self.lm.parameters():
                p.requires_grad = False

        # Load the expert's own tokenizer (lazy — only used for cross-tok).
        # Tokenizers are stateless so the cache is always safe.
        cached_tok = _load_tokenizer_cached(self.model_id)
        if cached_tok is None:
            # Some HF model repos don't ship a tokenizer; fall back to
            # the trunk tokenizer (detected as identity-bridge).
            self._expert_tokenizer = trunk_tokenizer
        else:
            self._expert_tokenizer = cached_tok

        self._trunk_tokenizer = trunk_tokenizer
        self.vocab_bridge = VocabBridge.build(
            trunk_tokenizer=trunk_tokenizer,
            expert_tokenizer=self._expert_tokenizer,
        )
        self.is_same_tokenizer = self.vocab_bridge.is_identity
        self.vocab_size_trunk = self.vocab_bridge.vocab_size_trunk
        self.vocab_size_expert = self.vocab_bridge.vocab_size_expert

        # ── Telemetry: per-batch alignment coverage ──────────────────
        # For same-tok experts this is always 1.0 (after forward); for
        # cross-tok experts it's the fraction of trunk positions that
        # exact-aligned to an expert position in the most recent
        # forward pass. Set by ``_forward_*``. The harness can log
        # this as a leading indicator of distillation quality — low
        # coverage means the expert mostly abstains (uniform) and
        # contributes no real signal at most positions.
        self.last_alignment_coverage: Optional[float] = None

        # Expert model's hard context limit.  GPT-2 has n_positions=1024;
        # feeding sequences longer than this causes a CUDA device-side
        # assertion in the position embedding lookup (indexSelectLargeIndex:
        # srcIndex < srcSelectDimSize).  Both _forward_same_tok and
        # _forward_bridge clamp their inputs at min(T, _expert_max_ctx).
        _lm_cfg = getattr(self.lm, 'config', None)
        if _lm_cfg is not None:
            _n = (getattr(_lm_cfg, 'n_positions', None)
                  or getattr(_lm_cfg, 'max_position_embeddings', None))
            self._expert_max_ctx: int = int(_n) if _n else 2048
            # Model's own wte size — can be SMALLER than the expert
            # tokenizer's vocab_size when the tokenizer carries added
            # special tokens beyond what the loaded checkpoint embeds
            # (observed on microsoft/CodeGPT-small-py — tokenizer emits
            # ids up to ~50295 against a 50257-row wte). Feeding an
            # OOB id triggers the same CUDA device-side assert as a
            # T-overflow. ``_forward_bridge`` clamps every emitted
            # expert id to ``[0, _expert_vocab_size)`` defensively.
            _v = getattr(_lm_cfg, 'vocab_size', None)
            self._expert_vocab_size: int = int(_v) if _v else 0
        else:
            self._expert_max_ctx = 2048
            self._expert_vocab_size = 0

        # Split backbone + lm_head for memory-efficient chunked forward.
        # (None, None) for unknown architectures → falls back to lm() directly.
        self._backbone, self._lm_head = _split_lm(self.lm)

    # ── public ───────────────────────────────────────────────────────

    @property
    def freeze(self) -> bool:
        """Reflect current freeze state by checking any one parameter."""
        for p in self.lm.parameters():
            return not p.requires_grad
        return True

    def forward_cpu(self, ids: torch.Tensor) -> torch.Tensor:
        """Like ``forward()`` but always returns a CPU tensor.

        Used by :class:`LMExpertEnsemble` to accumulate all expert outputs
        on CPU, avoiding simultaneous large GPU buffers.  The ensemble moves
        the final sum to the target device once after all experts run.
        """
        if ids.dim() != 2:
            raise ValueError(
                f"expected (B, T) ids, got shape {tuple(ids.shape)}"
            )
        if self.is_same_tokenizer:
            return self._forward_same_tok(ids)
        return self._forward_bridge(ids)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """``ids: (B, T)`` (trunk-vocab) ⇒ ``(B, T, V_trunk)`` logits on same device."""
        return self.forward_cpu(ids).to(ids.device)

    # ── internals ────────────────────────────────────────────────────

    def _forward_same_tok(self, ids: torch.Tensor) -> torch.Tensor:
        """Fast path: chunked backbone + lm_head, accumulated on CPU.

        Peak GPU footprint: (B, _LOGIT_CHUNK_T, d) hidden states
        + (B, _LOGIT_CHUNK_T, V) one head chunk = ~431 MB at B=16 (vs
        3 GB for the full T=2048 sequence in the old one-shot path).

        Falls back to a full ``self.lm(ids)`` forward for architectures
        where ``_split_lm`` couldn't extract backbone + head.

        Returns a CPU tensor; callers that need it on GPU should call
        ``.to(device)`` — ``forward()`` does this automatically.
        """
        B, T = ids.shape
        cap = min(T, self._expert_max_ctx)
        if torch.is_autocast_enabled():
            _out_dtype: torch.dtype = torch.get_autocast_gpu_dtype()
        else:
            _out_dtype = torch.float32
        out_cpu = torch.zeros(
            (B, T, self.vocab_size_trunk), device="cpu", dtype=_out_dtype
        )
        with torch.amp.autocast(device_type=ids.device.type, enabled=False):
            if self._backbone is not None:
                hidden = self._backbone(
                    input_ids=ids[:, :cap]
                ).last_hidden_state  # (B, cap, d_expert)
                for t0 in range(0, cap, _LOGIT_CHUNK_T):
                    t1 = min(t0 + _LOGIT_CHUNK_T, cap)
                    chunk = self._lm_head(hidden[:, t0:t1]).to(dtype=_out_dtype)
                    out_cpu[:, t0:t1].copy_(chunk.cpu())
                    del chunk
                del hidden
            else:
                part = self.lm(input_ids=ids[:, :cap]).logits.to(dtype=_out_dtype)
                out_cpu[:, :cap].copy_(part.cpu())
                del part
        self.last_alignment_coverage = cap / T if T > 0 else 1.0
        return out_cpu

    def _forward_bridge(self, ids: torch.Tensor) -> torch.Tensor:
        """Bridge path: per-sample re-tokenise, run expert, align via
        :func:`_align_by_char_offsets_exact`, project via the vocab
        bridge. Misaligned trunk positions abstain to uniform.

        Why exact-end alignment (vs. the legacy smallest-ge alignment)
        --------------------------------------------------------------
        The legacy alignment chose the smallest expert position whose
        end-offset was ``>= trunk[t].end`` — at positions where the two
        tokenisations don't share a boundary, this picks an expert
        position whose end is STRICTLY GREATER than trunk's. The expert
        at that position has already SEEN trunk's target as part of its
        input prefix AND its prediction is for content STARTING PAST
        trunk's prediction horizon. Using such logits as distillation
        targets is wrong-horizon noise that drags the trunk toward
        "what comes after trunk's target", not "what IS trunk's target".

        Exact alignment fixes this: at trunk position t, use expert
        position e iff ``expert_offsets[e][1] == trunk_offsets[t][1]``
        exactly; otherwise leave the trunk position at uniform (zero
        logits, ``CE = ln V_trunk``). On natural English with gpt2
        trunk + SmolLM2 expert, ~95% of trunk positions exact-align;
        the remaining ~5% abstain instead of contributing noise. The
        H22 forensic (see ``docs/FINDINGS.md`` and
        ``scripts/diagnose_bridge_ce.py``) measured a 0.27-nat CE
        improvement on a held-out paragraph, taking SmolLM2 from
        +0.05 nats over gpt2 to -0.22 nats UNDER it.

        Safety guards
        -------------
        * **Truncate expert tokens to trunk T.** When the trunk's text
          decodes to lots of characters, the expert tokenizer can
          re-encode it into ``T_expert >> T`` tokens (observed: 3307
          tokens from a T=512 trunk batch with random ids). The
          alignment map only ever uses ``t_count <= T`` positions, so
          the trailing expert tokens are wasted work AND a frequent
          source of CUBLAS instability on bf16 CUDA at large matmul
          sizes. Cap ``expert_input_ids`` at ``T`` up-front.
        * **Disable autocast around the expert forward.** Same reason
          as ``_forward_same_tok``: the frozen expert's CUBLAS path is
          unstable when bf16 autocast downcasts its inputs.

        Telemetry
        ---------
        Sets ``self.last_alignment_coverage`` to the fraction of trunk
        positions that exact-aligned (averaged across the batch). The
        harness logs this as a leading indicator: coverage near 0
        means the expert is silently abstaining at most positions,
        contributing essentially no distillation signal.

        Regression-pinned by
        ``tests/training/test_lm_expert_bridge_safety.py`` and
        ``tests/training/test_lm_expert_bridge_exact_alignment.py``.
        """
        B, T = ids.shape
        device = ids.device
        if torch.is_autocast_enabled():
            _out_dtype: torch.dtype = torch.get_autocast_gpu_dtype()
        else:
            _out_dtype = torch.float32
        out_cpu = torch.zeros(
            (B, T, self.vocab_size_trunk), device="cpu", dtype=_out_dtype,
        )
        n_aligned_total = 0
        n_positions_total = 0

        for b in range(B):
            sample_ids = ids[b].tolist()
            text = self._trunk_tokenizer.decode(
                sample_ids, skip_special_tokens=False
            )
            trunk_enc = self._trunk_tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_tensors=None,
            )
            expert_enc = self._expert_tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_tensors=None,
            )
            trunk_offsets = trunk_enc["offset_mapping"]
            expert_offsets = expert_enc["offset_mapping"]
            # Truncate at min(T, _expert_max_ctx, _BRIDGE_T_MAX).
            # _expert_max_ctx: prevents PE out-of-bounds (GPT-2: 1024).
            # _BRIDGE_T_MAX: memory guard for the NO-BACKBONE fallback only.
            # The backbone+sparse-head path below applies lm_head to just the
            # ALIGNED hidden states, so its memory is O(n_aligned, V) and does
            # NOT grow with T — capping T there does nothing for memory and
            # everything for harm: at ctx=2048 a cap of 512 leaves ~75% of
            # trunk positions with no aligned expert token, so they abstain to
            # uniform (CE = ln V ≈ 10.8). That drags the cortex CE up to ≈8.9
            # nats — a distillation teacher WORSE than the trunk (lm_ema≈7.6),
            # which silently kills distillation at long context (it worked at
            # ctx=512 where 512 = full coverage). So: full coverage whenever a
            # backbone is present; keep the 512 guard only for the fallback
            # path, which still materialises a full (T_expert, V) logit tensor.
            _BRIDGE_T_MAX = (
                self._expert_max_ctx if self._backbone is not None else 512
            )
            _expert_cap = min(T, self._expert_max_ctx, _BRIDGE_T_MAX)
            expert_input_ids_list = expert_enc["input_ids"][:_expert_cap]
            expert_offsets = expert_offsets[:_expert_cap]
            expert_input_ids = torch.tensor(
                expert_input_ids_list, dtype=torch.long, device=device,
            ).unsqueeze(0)  # (1, T_expert <= _expert_cap)
            if self._expert_vocab_size:
                expert_input_ids = expert_input_ids.clamp_(
                    max=self._expert_vocab_size - 1
                )

            t_count = min(T, len(trunk_offsets))
            n_positions_total += t_count

            if expert_input_ids.shape[1] == 0 or t_count == 0:
                continue

            # Compute alignment BEFORE the model call so we can:
            #   (a) skip the model entirely if no positions align, and
            #   (b) run the lm_head only on the n_valid aligned hidden
            #       states rather than the full (T_expert, V_expert) matrix.
            idx_map = _align_by_char_offsets_exact(
                trunk_offsets[:t_count], expert_offsets,
            )
            idx_t = torch.tensor(idx_map, dtype=torch.long, device=device)
            valid_mask = idx_t >= 0
            n_aligned_total += int(valid_mask.sum().item())
            if not valid_mask.any():
                continue
            valid_pos = torch.nonzero(valid_mask, as_tuple=True)[0]
            valid_idx = idx_t.index_select(0, valid_pos)

            # Sparse model call — autocast disabled (frozen expert stability).
            with torch.amp.autocast(device_type=device.type, enabled=False):
                with torch.no_grad():
                    if self._backbone is not None:
                        # Peak GPU: (1, T_expert, d) hidden states, then
                        # (n_valid, V_expert) for the head — much smaller
                        # than the full (T_expert, V_expert) logit matrix.
                        hidden = self._backbone(
                            input_ids=expert_input_ids,
                        ).last_hidden_state.squeeze(0)   # (T_expert, d)
                        picked_h = hidden.index_select(0, valid_idx)  # (n_valid, d)
                        picked = self._lm_head(picked_h)              # (n_valid, V_expert)
                        del hidden, picked_h
                    else:
                        full_logits = self.lm(
                            input_ids=expert_input_ids,
                        ).logits.squeeze(0)              # (T_expert, V_expert)
                        picked = full_logits.index_select(0, valid_idx)
                        del full_logits

            bridged = self.vocab_bridge.apply(picked)  # (n_valid, V_trunk)
            out_cpu[b].index_copy_(
                0, valid_pos.cpu(), bridged.to(dtype=_out_dtype).cpu()
            )

        if n_positions_total > 0:
            self.last_alignment_coverage = (
                n_aligned_total / float(n_positions_total)
            )
        return out_cpu


# ──────────────────────────────────────────────────────────────────────
# LMExpertEnsemble — router-weighted mixture of trunk-vocab logits
# ──────────────────────────────────────────────────────────────────────


class LMExpertEnsemble(nn.Module):
    """N :class:`LMExpert`s + a :class:`~neuroslm.cortex.ThalamicRouter`
    → mixture-of-experts in trunk-vocab logit space.

    Forward: ``ids: (B, T)`` ⇒ ``(B, T, V_trunk)`` logits, computed as
    the router-weighted sum of every expert's bridged logits.

    Important properties:
      * Output is in trunk vocab space — directly mixable with the
        trunk's own LM logits via the harness's ``α`` / ``α_eff`` weight.
      * No hidden-state projection step. No ``cortex_lm_head``,
        no ``cortex_pre_head_norm``. The pretrained heads do the work.
      * The router's ``last_routing_weights`` is exposed for telemetry,
        same surface as the legacy ``MultiCortexEnsemble``.

    Construction validates:
      * ``len(experts) >= 1``
      * router domains == expert domains (same order, same set)
    """

    def __init__(
        self,
        experts: Sequence[LMExpert],
        router,                         # ThalamicRouter (avoid circular import)
        lateral_inhibition=None,        # Optional[LateralInhibition] (Item 4)
    ) -> None:
        super().__init__()
        experts = list(experts)
        if len(experts) < 1:
            raise ValueError(
                "LMExpertEnsemble requires at least one expert"
            )
        expert_domains = [e.domain for e in experts]
        router_domains = list(getattr(router, "domains", []))
        if expert_domains != router_domains:
            raise ValueError(
                f"LMExpertEnsemble: expert domains {expert_domains} must "
                f"equal router domains {router_domains} in the same order"
            )
        self.experts = nn.ModuleList(experts)
        self.router = router
        # Item 4: optional Mexican-hat / WTA inhibition on routing
        # weights. `None` (default) → pass-through (legacy behaviour).
        # When present, applied between router output and weighted sum.
        self.lateral_inhibition = lateral_inhibition
        self.domains: List[str] = expert_domains
        self._last_routing_weights: Optional[torch.Tensor] = None

    @property
    def last_routing_weights(self) -> Optional[torch.Tensor]:
        return self._last_routing_weights

    def set_nt_levels(self, levels) -> None:
        """Push the current neuromodulator levels into all
        NT-modulated sub-modules of the ensemble.

        The harness calls this once per training step with the full
        ``NTSystem.levels()`` dict; each sub-module picks the NT key
        it cares about (NE for the router, GABA for the inhibitor)
        and silently ignores the rest.
        """
        if levels is None:
            return
        if hasattr(self.router, "set_nt_levels"):
            self.router.set_nt_levels(levels)
        if self.lateral_inhibition is not None and hasattr(
            self.lateral_inhibition, "set_nt_levels"
        ):
            self.lateral_inhibition.set_nt_levels(levels)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Mixture of expert logits, router-weighted per token."""
        if ids.dim() != 2:
            raise ValueError(
                f"expected (B, T) ids, got shape {tuple(ids.shape)}"
            )

        weights = self.router(ids)               # (B, T, N)
        # Item 4: divisive lateral inhibition (no-op when disabled).
        if self.lateral_inhibition is not None:
            weights = self.lateral_inhibition(weights)
        self._last_routing_weights = weights

        # Determine the dtype to use for the fusion loop.
        #
        # We CANNOT use weights.dtype for this.  nn.LayerNorm inside
        # ThalamicRouter is promoted to fp32 by PyTorch's autocast policy
        # (for numerical stability), so weights.dtype == torch.float32 even
        # when the outer training step is running under bf16 autocast.
        # Using weights.dtype as the cast target is therefore a silent no-op
        # that leaves both w_i and e_logits as fp32, triggering a 6.14 GiB
        # allocation for the product (B=16, T=2048, V=50257) → CUDA OOM.
        #
        # The correct approach: read the active AMP compute dtype directly.
        # If no autocast is active (eval, CPU inference), fall back to fp32.
        if torch.is_autocast_enabled():
            _fuse_dtype: torch.dtype = torch.get_autocast_gpu_dtype()
        else:
            _fuse_dtype = torch.float32

        # Move routing weights to CPU once; accumulation happens on CPU so
        # no large (B, T, V) GPU buffer exists during the expert loop.
        weights_cpu = weights.cpu().to(dtype=_fuse_dtype)  # (B, T, N) on CPU

        out_cpu: Optional[torch.Tensor] = None
        for i, expert in enumerate(self.experts):
            # Prefer forward_cpu() — returns CPU tensor without GPU round-trip.
            # Fall back to forward().cpu() for expert types that pre-date this API
            # (e.g. test mocks, legacy multi-cortex experts).
            _fwd_cpu = getattr(expert, 'forward_cpu', None)
            if _fwd_cpu is not None:
                e_cpu = _fwd_cpu(ids)
            else:
                e_cpu = expert(ids).cpu()
            e_cpu = e_cpu.to(dtype=_fuse_dtype)
            w_i = weights_cpu[..., i].unsqueeze(-1)          # (B, T, 1) CPU
            e_cpu.mul_(w_i)
            if out_cpu is None:
                out_cpu = e_cpu
            else:
                out_cpu.add_(e_cpu)
                del e_cpu
        assert out_cpu is not None  # guarded by len(experts) >= 1 in __init__
        return out_cpu.to(ids.device)


# ──────────────────────────────────────────────────────────────────────
# Factory: DSL config → LMExpertEnsemble
# ──────────────────────────────────────────────────────────────────────


def build_lm_expert_ensemble(
    *,
    experts,                       # Sequence[ExpertSpec] (duck-typed)
    trunk_tokenizer,               # PreTrainedTokenizerBase or its name (str)
    vocab_size: int,
    router_d_model: int = 256,
    lexical_bias_weight: float = 2.0,
    bema_tau: float = 0.5,
    lexicon=None,                  # Optional DomainLexicon; built if None
    lateral_inhibition_kappa: float = 0.0,   # Item 4 — Mexican-hat WTA
) -> "LMExpertEnsemble":
    """Build an :class:`LMExpertEnsemble` from a list of expert specs.

    This is the harness-side entry point for the
    ``multi_cortex.experts: [...]`` DSL block. It:

      1. Loads ``trunk_tokenizer`` (if a string was passed).
      2. Constructs one :class:`LMExpert` per spec (each lazy-loads its
         HuggingFace model on first instantiation; subsequent calls hit
         the HF cache).
      3. Builds a :class:`~neuroslm.cortex.ThalamicRouter` whose
         ``domains`` exactly match the per-expert ``domain`` order
         (the ensemble enforces this match at construction time).
      4. Optionally builds a :class:`~neuroslm.cortex.DomainLexicon`
         from the trunk tokenizer's BPE table when a non-zero
         ``lexical_bias_weight`` is in use. With weight=0 we still need
         a lexicon for the router constructor, so we hand it an empty
         one (no lexical prior, just learned routing).

    Parameters
    ----------
    experts : Sequence[ExpertSpec]
        The roster. Each element must expose ``.id`` (HF model id),
        ``.domain`` (routing key), and ``.freeze`` (bool). Duck-typed
        so test fixtures can pass plain dataclasses or ``SimpleNamespace``
        without importing ``ExpertSpec``.
    trunk_tokenizer : PreTrainedTokenizerBase or str
        The trunk's tokenizer; passed through to each ``LMExpert`` so
        the vocab bridge can be built. A string is interpreted as a
        HuggingFace model id and loaded with ``AutoTokenizer.from_pretrained``.
    vocab_size : int
        Size of the trunk vocabulary. Passed to the router; must match
        ``trunk_tokenizer.vocab_size``. We check and emit a warning on
        mismatch (rather than silently shadowing the dataset's tokenizer).
    router_d_model : int, default 256
        Hidden width of the :class:`~neuroslm.cortex.ThalamicRouter` MLP.
    lexical_bias_weight : float, default 2.0
        Weight on the DomainLexicon-derived prior added to router
        logits. ``0.0`` disables the prior — the router learns routing
        from scratch (useful for tests where determinism matters more
        than domain-faithful routing).
    bema_tau : float, default 0.5
        Bregman-EMA smoothing constant on routing weights.
    lexicon : DomainLexicon, optional
        Pre-built lexicon to reuse. When ``None`` (default), an empty
        lexicon is built — the per-domain keyword priors are NOT
        populated. Pass an explicit lexicon if you want lexical routing.

    Returns
    -------
    LMExpertEnsemble
        Ready to be assigned to ``harness.multi_cortex``.
    """
    # ── Resolve tokenizer (str → instance) ─────────────────────────────
    if isinstance(trunk_tokenizer, str):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "build_lm_expert_ensemble requires `transformers` when "
                "trunk_tokenizer is given as a string id."
            ) from exc
        trunk_tokenizer = AutoTokenizer.from_pretrained(trunk_tokenizer)

    # ── Sanity: trunk tokenizer's vocab matches the configured size ──
    tok_vocab = int(getattr(trunk_tokenizer, "vocab_size", vocab_size))
    if tok_vocab != int(vocab_size):
        import warnings
        warnings.warn(
            f"build_lm_expert_ensemble: trunk_tokenizer.vocab_size "
            f"({tok_vocab}) != vocab_size argument ({vocab_size}); "
            "the dataset was tokenised with a different tokenizer than "
            "the experts will bridge to — routing will still work but "
            "the bridge will mask out-of-range trunk ids.",
            RuntimeWarning,
        )

    # ── Domains, in roster order (the ensemble validates equality) ────
    expert_list = list(experts)
    if len(expert_list) < 1:
        raise ValueError(
            "build_lm_expert_ensemble: experts roster must contain "
            "at least one entry"
        )
    domains: List[str] = [str(e.domain) for e in expert_list]

    # ── Build router (with empty lexicon by default) ──────────────────
    # Lazy import to avoid circular dep at module load: cortex.py also
    # imports things that may eventually touch experts.py.
    from neuroslm.cortex import ThalamicRouter, DomainLexicon

    if lexicon is None:
        lexicon = DomainLexicon.empty(domains=domains)

    # ThalamicRouter requires n_cortices >= 2; the ensemble works fine
    # with a single expert via the simpler degenerate "uniform" router
    # below. For 2+ experts, use the standard ThalamicRouter.
    if len(expert_list) >= 2:
        router = ThalamicRouter(
            vocab_size=int(vocab_size),
            d_model=int(router_d_model),
            domains=domains,
            lexicon=lexicon,
            lexical_bias_weight=float(lexical_bias_weight),
            bema_tau=float(bema_tau),
        )
    else:
        # Degenerate single-expert path: a tiny module that always
        # returns weight=1.0 on the lone expert. Avoids the
        # n_cortices>=2 constraint on ThalamicRouter without sneaking
        # a second dummy expert into the ensemble.
        router = _UniformSingletonRouter(domains=domains)

    # ── Build each LMExpert (HF model loaded lazily on first instance) ─
    lm_experts: List[LMExpert] = []
    for spec in expert_list:
        lm_experts.append(LMExpert(
            model_id=str(spec.id),
            domain=str(spec.domain),
            trunk_tokenizer=trunk_tokenizer,
            freeze=bool(getattr(spec, "freeze", True)),
        ))

    # ── Build optional lateral inhibition (Item 4) ────────────────────
    # `lateral_inhibition_kappa > 0` is required to enable; the harness
    # then pushes the live GABA level via `ensemble.set_nt_levels(...)`
    # each step. Even with κ_base > 0, GABA = 0 keeps the module fully
    # identity, so it never silently changes routing behaviour.
    inhibition = None
    if float(lateral_inhibition_kappa) > 0.0:
        from neuroslm.cortex import LateralInhibition
        inhibition = LateralInhibition(
            kappa_base=float(lateral_inhibition_kappa)
        )

    return LMExpertEnsemble(
        experts=lm_experts,
        router=router,
        lateral_inhibition=inhibition,
    )


class _UniformSingletonRouter(nn.Module):
    """Degenerate router used only when the roster has exactly ONE
    expert. Returns weight ``1.0`` for that expert at every position.

    The :class:`LMExpertEnsemble` API expects ``router(ids) → (B, T, N)``;
    here N=1 and every entry is 1, so the mixture is just the lone
    expert's logits.
    """

    def __init__(self, domains: Sequence[str]) -> None:
        super().__init__()
        if len(domains) != 1:
            raise ValueError(
                "_UniformSingletonRouter is only valid for exactly 1 domain, "
                f"got {len(domains)}"
            )
        self.domains: List[str] = [str(domains[0])]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.dim() != 2:
            raise ValueError(
                f"expected (B, T) ids, got shape {tuple(ids.shape)}"
            )
        B, T = ids.shape
        return torch.ones((B, T, 1), dtype=torch.float32, device=ids.device)
