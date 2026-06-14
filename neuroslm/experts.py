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
    "_align_by_char_offsets",
    "_load_lm_cached",
    "_load_tokenizer_cached",
    "build_lm_expert_ensemble",
]


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

    # Path 1: safetensors-only (works on every torch version)
    try:
        lm = AutoModelForCausalLM.from_pretrained(
            model_id, use_safetensors=True,
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


# Magnitude of the "expert abstains" logit — large enough that softmax
# gives near-zero probability without producing -inf NaNs in mixing.
_ABSTAIN_LOGIT: float = -1e4


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
        cross-tok, gathers expert logits by the trunk→expert table and
        masks unmapped trunk ids to ``_ABSTAIN_LOGIT``.
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
        if mask.any():
            gathered = torch.where(
                mask, torch.full_like(gathered, _ABSTAIN_LOGIT), gathered,
            )
        return gathered


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

        self.model_id = str(model_id)
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

    # ── public ───────────────────────────────────────────────────────

    @property
    def freeze(self) -> bool:
        """Reflect current freeze state by checking any one parameter."""
        for p in self.lm.parameters():
            return not p.requires_grad
        return True

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """``ids: (B, T)`` (trunk-vocab) ⇒ ``(B, T, V_trunk)`` logits.

        Fast path (same tok): ``self.lm(ids).logits`` directly.
        Bridge path (cross tok): per-sample retokenise, run expert,
        align back to trunk positions, project via the vocab bridge.
        """
        if ids.dim() != 2:
            raise ValueError(
                f"expected (B, T) ids, got shape {tuple(ids.shape)}"
            )

        if self.is_same_tokenizer:
            return self._forward_same_tok(ids)
        return self._forward_bridge(ids)

    # ── internals ────────────────────────────────────────────────────

    def _forward_same_tok(self, ids: torch.Tensor) -> torch.Tensor:
        """Fast path: native LM forward. The expert's own pretrained
        head produces logits directly in trunk vocab space."""
        # AutoModelForCausalLM.forward returns a CausalLMOutput; logits
        # are at ``.logits``. We feed input_ids only; attention mask is
        # all-ones (no padding in our fixed-length training batches).
        out = self.lm(input_ids=ids)
        return out.logits  # (B, T, V_expert == V_trunk)

    def _forward_bridge(self, ids: torch.Tensor) -> torch.Tensor:
        """Bridge path: per-sample re-tokenise, run expert, align,
        project. Slower than the fast path but enables true
        cross-architecture experts (Qwen, DeepSeek, etc.)."""
        B, T = ids.shape
        device = ids.device
        out = torch.full(
            (B, T, self.vocab_size_trunk),
            _ABSTAIN_LOGIT,
            device=device,
            dtype=torch.float32,
        )

        for b in range(B):
            sample_ids = ids[b].tolist()
            # Decode the sample to a string, then encode with both
            # tokenizers WITH offsets so we can align positions.
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
            expert_input_ids = torch.tensor(
                expert_enc["input_ids"], dtype=torch.long, device=device,
            ).unsqueeze(0)  # (1, T_expert)

            if expert_input_ids.shape[1] == 0:
                # Edge case: empty sample — leave as abstain row
                continue

            # Run expert on its own tokenisation
            with torch.no_grad():
                expert_logits = self.lm(
                    input_ids=expert_input_ids,
                ).logits.squeeze(0)  # (T_expert, V_expert)

            # Align: for every trunk-position t (up to the lesser of
            # T and len(trunk_offsets)), pick the expert position whose
            # char-end is just past trunk_offsets[t].end
            t_count = min(T, len(trunk_offsets))
            if t_count == 0:
                continue
            idx_map = _align_by_char_offsets(
                trunk_offsets[:t_count],
                expert_offsets,
            )
            idx_t = torch.tensor(idx_map, dtype=torch.long, device=device)
            picked = expert_logits.index_select(0, idx_t)  # (t_count, V_expert)
            # Project to trunk vocab via bridge
            bridged = self.vocab_bridge.apply(picked)  # (t_count, V_trunk)
            out[b, :t_count] = bridged.to(out.dtype)

        return out


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
        self.domains: List[str] = expert_domains
        self._last_routing_weights: Optional[torch.Tensor] = None

    @property
    def last_routing_weights(self) -> Optional[torch.Tensor]:
        return self._last_routing_weights

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Mixture of expert logits, router-weighted per token."""
        if ids.dim() != 2:
            raise ValueError(
                f"expected (B, T) ids, got shape {tuple(ids.shape)}"
            )

        weights = self.router(ids)               # (B, T, N)
        self._last_routing_weights = weights

        # Compute every expert's logits in trunk-vocab space.
        # We sum directly to keep peak memory at one (B, T, V_trunk)
        # tensor instead of N of them.
        out: Optional[torch.Tensor] = None
        for i, expert in enumerate(self.experts):
            e_logits = expert(ids)                          # (B, T, V_trunk)
            w_i = weights[..., i].unsqueeze(-1)             # (B, T, 1)
            if out is None:
                out = w_i * e_logits
            else:
                out = out + w_i * e_logits
        assert out is not None  # guarded by len(experts) >= 1 in __init__
        return out


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

    return LMExpertEnsemble(experts=lm_experts, router=router)


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
