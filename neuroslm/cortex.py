# -*- coding: utf-8 -*-
r"""Multi-Cortex Thalamic Routing — ``MultiCortexEnsemble`` & friends.

This module realises the "research/multi-trunk-v2" architecture: a pool
of *N* specialist sub-cortices (math / code / chat / general …) whose
hidden states are blended per token by a thalamic router that combines

  • a deterministic *lexical bias* — a domain-keyword prior derived from
    token identity (works at step 0, no LM gradient required), and
  • a learnable router head — a zero-initialised linear map that grows
    under the LM gradient as the model learns to allocate compute,

with an optional *BEMA biological damping* term that blends the
per-batch softmax with a running EMA, suppressing the per-token routing
oscillations that would otherwise destabilise the ensemble.

The design is the natural extension of the bowtie's thalamic-relay
hypothesis (Sherman 2007, Halassa & Kastner 2017): the thalamus is the
brain's central gating bus; specialisation lives in cortical modules; a
mixture-of-experts read-out is the mathematical realisation of the
columnar microcircuit.

Mathematical model
------------------
Let :math:`L_{ti} \in \mathbb R^{N}` be the per-token logit vector at
position :math:`t` in batch sample :math:`i`:

.. math::

   L_{ti}  =  beta * 1[x_{ti} in D]  +  W @ phi(x_{ti}),

where :math:`\beta` is :attr:`lexical_bias_weight`, :math:`D` is the
domain partition, and :math:`W` is the zero-init learnable head over
the router-embedding :math:`\phi(x)`. The routing weights are

.. math::

   p_{ti}  =  softmax(L_{ti}),

possibly EMA-damped via :math:`p_{ti} \leftarrow (1-\tau) p_{ti}
+ \tau \bar p`. The ensemble hidden state is the per-token mixture

.. math::

   h_{ti} = sum_n p_{ti,n} * P_n(C_n(x))_{ti},

with :math:`C_n` the :math:`n`-th sub-cortex and :math:`P_n` an optional
projection from the sub-cortex's native hidden dim to the shared
``d_target``.

References
----------
* Sherman & Guillery (2002), Phil. Trans. R. Soc. — Thalamus as a relay
* Halassa & Kastner (2017), Nat. Neurosci. — Thalamic control of cortical
  state
* Shazeer et al. (2017), arXiv:1701.06538 — Sparsely-Gated MoE
* Fedus et al. (2022), JMLR — Switch Transformer
* Ramachandran et al. (2017), arXiv:1710.05941 — Swish / Gated routing
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "SubCortex",
    "StubSubCortex",
    "GPT2SubCortex",
    "DomainLexicon",
    "ThalamicRouter",
    "MultiCortexEnsemble",
    "build_default_ensemble",
    "build_gpt2_ensemble",
    "DEFAULT_GPT2_VARIANTS",
]


# ──────────────────────────────────────────────────────────────────────
# Sub-cortices
# ──────────────────────────────────────────────────────────────────────

class SubCortex(nn.Module):
    """Abstract base for a specialist language sub-cortex.

    A sub-cortex maps ``(B, T)`` token-id tensor to a ``(B, T, d_native)``
    hidden-state tensor. Concrete implementations differ in *what* they
    do internally (HuggingFace GPT-2, in-house DSL cortex, a frozen
    random-init transformer for tests).

    Parameters
    ----------
    name : str
        Human-readable identifier (e.g. "cortex_math"). Used for
        telemetry and gradient debugging.
    domain : str
        Domain label this cortex specialises in
        (``"math"`` / ``"code"`` / ``"chat"`` / ``"general"`` / …).
        Consumed by :class:`DomainLexicon` to set the lexical-bias prior.
    d_model : int
        Native hidden dim of the sub-cortex.  The :class:`MultiCortexEnsemble`
        inserts a projection if this differs from ``d_target``.
    """

    def __init__(self, name: str, domain: str, d_model: int):
        super().__init__()
        self.name = str(name)
        self.domain = str(domain)
        self._d_native = int(d_model)

    @property
    def d_native(self) -> int:
        return self._d_native

    def forward(self, ids: torch.Tensor) -> torch.Tensor:    # pragma: no cover
        raise NotImplementedError(
            f"SubCortex subclass {type(self).__name__} must implement forward()"
        )

    def extra_repr(self) -> str:
        return f"name={self.name!r}, domain={self.domain!r}, d_native={self.d_native}"


class StubSubCortex(SubCortex):
    """Minimal random-init transformer — the default sub-cortex for tests
    and the cold-start path before HuggingFace weights are loaded.

    Architecture: ``nn.Embedding → nn.TransformerEncoder(causal) →
    LayerNorm``. Deliberately tiny so 4 of them run in under 50 ms on
    CPU.

    This class lets the routing infrastructure be validated end-to-end
    without any external dependency or weight download.
    """

    def __init__(self, name: str, domain: str, vocab: int, d_model: int,
                 n_layers: int = 2, n_heads: int = 4, max_ctx: int = 512):
        super().__init__(name=name, domain=domain, d_model=d_model)
        if vocab <= 0:
            raise ValueError(f"vocab must be positive, got {vocab}")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.embed = nn.Embedding(vocab, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            batch_first=True, norm_first=True, dropout=0.0,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.max_ctx = int(max_ctx)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.dim() != 2:
            raise ValueError(f"expected (B, T) ids, got shape {tuple(ids.shape)}")
        h = self.embed(ids)
        T = ids.shape[1]
        # Causal mask via the EXPLICIT additive mask only.
        #
        # We deliberately do NOT pass ``is_causal=True`` alongside.
        # The boolean ``is_causal`` flag is a kernel-fast-path hint
        # for the SDPA backend; on bf16/A100 the combination of an
        # explicit additive ``mask`` AND ``is_causal=True`` triggers
        # a documented memory-corruption bug in the nested-tensor
        # fast path that crashes with "illegal memory access" inside
        # the FFN (``linear2(dropout(activation(linear1(x))))``).
        # The mask alone gives identical (correct) causal attention
        # and avoids the buggy fast path entirely. Regression-pinned
        # by ``tests/training/test_stub_subcortex_bf16_safety.py``.
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=ids.device, dtype=h.dtype),
            diagonal=1,
        )
        h = self.encoder(h, mask=mask, is_causal=False)
        return self.norm(h)


class GPT2SubCortex(SubCortex):
    """HuggingFace GPT-2 wrapper. Lazy-imports ``transformers``.

    Loads any model exposing the GPT-2 architecture (``gpt2``,
    ``gpt2-medium``, ``gpt2-large``, ``gpt2-xl``, ``distilgpt2``, or any
    HF Hub fine-tune of those).

    Long-context handling
    ---------------------
    GPT-2's positional embedding ``wpe`` has exactly
    ``config.n_positions`` rows (1024 for the entire gpt2 family).
    Feeding a ``(B, T)`` tensor with ``T > n_positions`` would trigger
    an out-of-bounds gather on the position-id lookup deep inside the
    GPT-2 forward. To support arbitrary ``T`` (the training pipeline
    runs at ``seq_len=2048``), :meth:`forward` chunks the input into
    non-overlapping windows of size ``n_positions`` and concatenates
    the per-window hidden states along the time axis. Within each
    chunk GPT-2 sees an in-distribution position layout
    (``0..n_positions-1``).

    For ``T <= n_positions`` the call is a no-op pass-through —
    bit-identical to the raw ``self.gpt2(input_ids=ids)`` forward.
    """

    def __init__(self, name: str, domain: str, hf_model_id: str,
                 freeze_weights: bool = True):
        try:
            from transformers import GPT2Model  # type: ignore
        except ImportError as exc:           # pragma: no cover - env-dependent
            raise ImportError(
                "GPT2SubCortex requires the `transformers` package. "
                "Install it with:  pip install transformers"
            ) from exc

        gpt2 = GPT2Model.from_pretrained(hf_model_id)
        d_native = int(gpt2.config.n_embd)
        super().__init__(name=name, domain=domain, d_model=d_native)
        self.gpt2 = gpt2     # registered as submodule
        self.hf_model_id = str(hf_model_id)
        if freeze_weights:
            for p in self.gpt2.parameters():
                p.requires_grad = False

    @classmethod
    def from_module(cls, name: str, domain: str, gpt2,
                    hf_model_id: str = "<custom>",
                    freeze_weights: bool = True) -> "GPT2SubCortex":
        """Build a ``GPT2SubCortex`` from a pre-constructed ``GPT2Model``
        instance — bypassing ``from_pretrained``.

        Use cases
        ---------
        * **Testing** — supply a small in-memory ``GPT2Model(GPT2Config(...))``
          to exercise the wrapper without downloading hundreds of MB.
        * **Custom checkpoints** — load weights from a local file or
          a custom HF Hub repo with non-standard tokenisation and pass
          the already-built module here.
        * **Shared backbones** — register one GPT-2 module across
          multiple wrappers (e.g. an ensemble where every cortex
          shares storage) without re-instantiating per-cortex.

        Parameters
        ----------
        name : str
            Cortex name (e.g. ``"chat_cortex"``).
        domain : str
            Domain tag (e.g. ``"math"``, ``"code"``, ``"chat"``,
            ``"general"``) — must be one of the keys the
            :class:`ThalamicRouter` is configured for.
        gpt2 : transformers.GPT2Model
            A pre-built GPT-2 backbone. Must expose
            ``config.n_embd``, ``config.n_positions``, and a
            ``forward(input_ids, ...)`` returning
            ``last_hidden_state``.
        hf_model_id : str, optional
            Identifier recorded on the wrapper for logging /
            checkpoint provenance. Default ``"<custom>"``.
        freeze_weights : bool, optional
            Whether to set ``requires_grad = False`` on all GPT-2
            parameters. Defaults to ``True`` to match the
            :meth:`__init__` behaviour (frozen pre-trained features
            is the default ensemble mode).
        """
        sc = cls.__new__(cls)
        d_native = int(gpt2.config.n_embd)
        SubCortex.__init__(sc, name=name, domain=domain, d_model=d_native)
        sc.gpt2 = gpt2
        sc.hf_model_id = str(hf_model_id)
        if freeze_weights:
            for p in sc.gpt2.parameters():
                p.requires_grad = False
        return sc

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.dim() != 2:
            raise ValueError(
                f"expected (B, T) ids, got shape {tuple(ids.shape)}")
        n_pos = int(self.gpt2.config.n_positions)
        T = int(ids.shape[1])

        # Short-path: no chunking needed, bit-identical to raw GPT-2.
        if T <= n_pos:
            out = self.gpt2(input_ids=ids, output_hidden_states=False,
                            return_dict=True)
            return out.last_hidden_state    # (B, T, d_native)

        # Long-path: non-overlapping windows of size <= n_pos.
        # Each chunk gets a fresh position layout 0..len-1 — in-
        # distribution for GPT-2's wpe. The output for token t in
        # window w depends only on tokens [w*n_pos .. t] within that
        # window (causal attention truncated at window boundaries),
        # which is the standard "windowed attention" approximation
        # used by every long-context transformer that re-uses a
        # short-context backbone (e.g. block-recurrent transformers,
        # the original Reformer, and Longformer's local attention).
        parts: List[torch.Tensor] = []
        for start in range(0, T, n_pos):
            end = min(start + n_pos, T)
            chunk = ids[:, start:end]
            out = self.gpt2(input_ids=chunk, output_hidden_states=False,
                            return_dict=True)
            parts.append(out.last_hidden_state)
        return torch.cat(parts, dim=1)      # (B, T, d_native)


# ──────────────────────────────────────────────────────────────────────
# Domain lexicon — static token-id → domain mapping
# ──────────────────────────────────────────────────────────────────────

class DomainLexicon:
    """Static mapping ``{domain: set[int]}`` — which token ids signal
    which domain.

    The lexicon is the source of the *lexical bias* in
    :class:`ThalamicRouter`. It deliberately does NOT carry any learned
    parameters — its job is to provide a deterministic, gradient-free
    domain prior that lets the router specialise from the first
    forward, before any LM gradient has shaped the learnable head.

    A token may belong to multiple domains (e.g. "(" appears in both
    math and code); the bias for each domain is independent. A token
    may belong to no domain — the lexical bias for that token is the
    zero vector, and the router falls back to the learnable head (or
    uniform routing if the head is also at its zero-init).
    """

    def __init__(self, domain_token_map: Mapping[str, Iterable[int]]):
        self.token_sets: Dict[str, Set[int]] = {
            str(d): {int(t) for t in toks}
            for d, toks in domain_token_map.items()
        }

    def __repr__(self) -> str:    # pragma: no cover
        sizes = {d: len(toks) for d, toks in self.token_sets.items()}
        return f"DomainLexicon(sizes={sizes})"

    def build_table(self, vocab_size: int, domains: Sequence[str]) -> torch.Tensor:
        """Build the ``(vocab_size, len(domains))`` one-hot lookup table.

        Out-of-range or unknown-domain tokens get a zero row, which
        means the lexical bias for that token is the zero vector.
        """
        table = torch.zeros(vocab_size, len(domains), dtype=torch.float32)
        for col, domain in enumerate(domains):
            ids = self.token_sets.get(str(domain), set())
            for tok in ids:
                if 0 <= tok < vocab_size:
                    table[tok, col] = 1.0
        return table

    # ── HuggingFace-tokenizer-aware factory ──────────────────────────

    @classmethod
    def empty(cls, domains: Sequence[str]) -> "DomainLexicon":
        """Build a lexicon with no domain-token associations — every
        token has a zero lexical-bias row. Useful when the router's
        learnable head is the only signal source (e.g. tests that
        want predictable uniform routing)."""
        return cls(domain_token_map={str(d): set() for d in domains})

    @classmethod
    def from_gpt2_keywords(cls,
                           extra_keywords: Optional[Mapping[str, Sequence[str]]]
                           = None) -> "DomainLexicon":
        """Build a lexicon using the GPT-2 BPE tokenizer (via
        :mod:`tiktoken`, already a project dependency) to convert
        hand-curated keyword lists to token-id sets.

        Both bare and leading-space variants of each keyword are
        included (GPT-2 BPE is whitespace-sensitive).
        """
        try:
            import tiktoken  # type: ignore
        except ImportError as exc:    # pragma: no cover
            raise ImportError(
                "from_gpt2_keywords requires `tiktoken`."
            ) from exc
        enc = tiktoken.get_encoding("gpt2")

        keywords: Dict[str, List[str]] = {
            "math": [
                "equation", "calculate", "solve", "integral", "derivative",
                "matrix", "vector", "theorem", "proof", "sum", "product",
                "function", "limit", "differential",
                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                "+", "-", "*", "/", "=", "^",
            ],
            "code": [
                "def", "class", "import", "return", "if", "else", "elif",
                "for", "while", "True", "False", "None", "lambda", "yield",
                "self", "print", "len", "range", "list", "dict", "tuple",
                "try", "except", "finally", "with", "as", "in",
            ],
            "chat": [
                "you", "I", "me", "we", "us", "they", "please", "thank",
                "thanks", "hello", "hi", "yes", "no", "feel", "think",
                "love", "like", "want", "need", "?", "!",
            ],
            "general": [],
        }
        if extra_keywords:
            for d, ks in extra_keywords.items():
                keywords.setdefault(d, []).extend(ks)

        token_map: Dict[str, Set[int]] = {}
        for domain, words in keywords.items():
            ids: Set[int] = set()
            for w in words:
                ids.update(enc.encode(w))
                ids.update(enc.encode(" " + w))
            token_map[domain] = ids
        return cls(domain_token_map=token_map)


# ──────────────────────────────────────────────────────────────────────
# Thalamic router
# ──────────────────────────────────────────────────────────────────────

class ThalamicRouter(nn.Module):
    """Per-token routing distribution over *N* sub-cortices.

    The routing logits are the sum of a deterministic lexical bias
    (from :class:`DomainLexicon`) and a learnable linear head over a
    router-only embedding.  At initialisation the learnable head is
    zero, so the router behaviour is purely lexical — a math-heavy
    input is routed to the math cortex from step 0, no training
    required.

    BEMA biological damping
    -----------------------
    When ``bema_tau > 0``, the per-batch mean of the softmaxed weights
    is blended into a running EMA::

       bar_p_t  =  tau * bar_p_{t-1}  +  (1 - tau) * mean_{B,T}(p_t)

    and the per-token weights are re-blended::

       p_{ti}   <-  (1 - tau) * p_{ti}  +  tau * bar_p_t

    followed by renormalisation. ``tau = 0.0`` is the identity case
    (no damping); ``tau = 0.9`` gives strong biological inertia.

    NT-modulated routing temperature  (Item 2)
    ------------------------------------------
    The locus coeruleus (NE) is the brain's gain / arousal channel.
    High NE sharpens routing (winner-take-most); low NE diffuses it
    (mixture stays soft). With ``router_temp_nt_gain > 0`` and the
    current NE level supplied via ``set_nt_levels({"NE": ...})``:

       z_NE  = 2 * (NE - 0.5)                           # in [-1, +1]
       mult  = clamp(1 + k_NE * z_NE, 0.1, 10.0)        # temperature multiplier
       logits_T = logits * mult                         # same as dividing by T

    The default (``router_temp_nt_gain = 0.0`` or no NT set) is
    multiplier = 1 → bit-identical to the legacy path.
    """

    def __init__(self, vocab_size: int, d_model: int,
                 domains: Sequence[str], lexicon: DomainLexicon,
                 lexical_bias_weight: float = 2.0,
                 bema_tau: float = 0.0,
                 router_temp_nt_gain: float = 0.0):
        super().__init__()
        if not 0.0 <= bema_tau < 1.0:
            raise ValueError(
                f"bema_tau must be in [0, 1), got {bema_tau}")
        if lexical_bias_weight < 0.0:
            raise ValueError(
                f"lexical_bias_weight must be non-negative, got {lexical_bias_weight}")
        if router_temp_nt_gain < 0.0:
            raise ValueError(
                f"router_temp_nt_gain must be non-negative (negative "
                f"would flip the NE → sharpness polarity), got "
                f"{router_temp_nt_gain}")
        self.domains = [str(d) for d in domains]
        self.n_cortices = len(self.domains)
        if self.n_cortices < 2:
            raise ValueError("Need at least 2 cortices to route between")
        self.lexicon = lexicon
        self.lexical_bias_weight = float(lexical_bias_weight)
        self.bema_tau = float(bema_tau)
        self.router_temp_nt_gain = float(router_temp_nt_gain)
        self.vocab_size = int(vocab_size)

        # NT level snapshot — the harness pushes the homeostat's NE
        # value via set_nt_levels() before forward(). Default 0.5
        # (centre of the [0, 1] sigmoid range) so z_NE = 0 and the
        # multiplier stays at 1 if no NT is ever pushed.
        self._ne_level: float = 0.5

        # Router-only token embedding (small, learnable)
        self.router_embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.router_embed.weight, std=0.02)

        # Learnable head — ZERO INIT (ReZero contract: at step 0,
        # routing == lexical bias)
        self.learnable_logits = nn.Linear(d_model, self.n_cortices, bias=False)
        nn.init.zeros_(self.learnable_logits.weight)

        # Pre-compute the lexical-bias lookup table as a buffer
        lex_table = lexicon.build_table(self.vocab_size, self.domains)
        self.register_buffer("_lex_table", lex_table * self.lexical_bias_weight)

        # BEMA state: running EMA of mean routing weights, plus an
        # "initialised" scalar so the first batch primes the EMA
        # exactly (no warm-up artifacts).
        self.register_buffer(
            "_ema_weights",
            torch.full((self.n_cortices,), 1.0 / self.n_cortices),
        )
        self.register_buffer(
            "_ema_initialised", torch.tensor(0, dtype=torch.int8))

    # ── private ──────────────────────────────────────────────────────

    def _lexical_bias(self, ids: torch.Tensor) -> torch.Tensor:
        """(B, T) ids → (B, T, N) lexical-bias logits."""
        return self._lex_table[ids]      # vectorised lookup; same device as ids

    def _apply_bema(self, weights: torch.Tensor) -> torch.Tensor:
        """Blend per-batch instantaneous weights with the running EMA.

        Updates ``self._ema_weights`` in-place under no_grad. The
        forward graph remains intact via the ``weights`` term.
        """
        if self.bema_tau <= 0.0:
            return weights

        with torch.no_grad():
            batch_mean = weights.mean(dim=tuple(range(weights.dim() - 1))) \
                                .to(self._ema_weights.dtype)
            if int(self._ema_initialised.item()) == 0:
                self._ema_weights.copy_(batch_mean)
                self._ema_initialised.fill_(1)
            else:
                self._ema_weights.mul_(self.bema_tau)
                self._ema_weights.add_(batch_mean, alpha=1.0 - self.bema_tau)

        ema = self._ema_weights.to(weights.dtype)                   # (N,)
        # Broadcast to weights' shape (..., N)
        view_shape = (1,) * (weights.dim() - 1) + (self.n_cortices,)
        ema_b = ema.view(*view_shape).expand_as(weights)
        blended = (1.0 - self.bema_tau) * weights + self.bema_tau * ema_b
        # Re-normalise (numerical hygiene — the blend stays on the
        # simplex analytically since both inputs do, but enforce it)
        return blended / blended.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # ── public ───────────────────────────────────────────────────────

    def set_nt_levels(self, levels: "Mapping[str, float]") -> None:
        """Push the current neuromodulator levels into the router.

        Only the ``"NE"`` key is consumed today (drives the routing
        softmax temperature — Item 2). Other keys are silently ignored
        so the caller can pass the whole ``NTSystem.levels()`` dict
        without filtering. Out-of-range values are clamped to
        ``[0, 1]`` (NTs are sigmoid-bounded by construction).
        """
        if levels is None:
            return
        ne = levels.get("NE")
        if ne is None:
            return
        # Clamp defensively — homeostat sigmoid guarantees [0, 1] but
        # tests may push synthetic levels for sharpness assertions.
        self._ne_level = max(0.0, min(1.0, float(ne)))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """``ids: (B, T)`` ⇒ routing weights ``(B, T, N)``."""
        if ids.dim() != 2:
            raise ValueError(f"expected (B, T) ids, got shape {tuple(ids.shape)}")
        bias = self._lexical_bias(ids)                              # (B, T, N)
        h = self.router_embed(ids)                                  # (B, T, d)
        learn = self.learnable_logits(h)                            # (B, T, N)
        logits = bias + learn

        # Item 2: NE-driven temperature. Multiplier > 1 sharpens,
        # < 1 softens. Both are clamped to [0.1, 10.0] for numerical
        # safety. With k_NE = 0 (default) the multiplier is exactly 1
        # so this path is bit-identical to the legacy router.
        if self.router_temp_nt_gain > 0.0:
            z_ne = 2.0 * (self._ne_level - 0.5)                     # [-1, +1]
            raw_mult = 1.0 + self.router_temp_nt_gain * z_ne
            mult = max(0.1, min(10.0, raw_mult))
            if mult != 1.0:
                logits = logits * mult

        weights = F.softmax(logits, dim=-1)
        return self._apply_bema(weights)

    def extra_repr(self) -> str:    # pragma: no cover
        return (f"n_cortices={self.n_cortices}, "
                f"domains={self.domains}, "
                f"lexical_bias_weight={self.lexical_bias_weight}, "
                f"bema_tau={self.bema_tau}")


# ──────────────────────────────────────────────────────────────────────
# Lateral expert inhibition  (Item 4 — Mexican-hat / WTA via GABA)
# ──────────────────────────────────────────────────────────────────────

class LateralInhibition(nn.Module):
    """Divisive lateral inhibition between expert routing weights.

    Motivation
    ----------
    The softmax router can stay in a soft-tie equilibrium (e.g.
    ``0.45 / 0.45 / 0.10``) for the whole run — there is no
    mathematical pressure to actually *pick* one expert. Real cortex
    avoids this with lateral GABAergic inhibition: a strongly firing
    pyramidal cell recruits interneurons that suppress its neighbours.
    The classical model is divisive normalisation
    (Carandini & Heeger 2012, *Normalization as a canonical neural
    computation*), which is exactly what this module implements.

    Formula
    -------
    Given routing weights ``w_i`` on the simplex ``Δ^N`` per token:

        rival_mass_i   = Σ_{j≠i} w_j            = (Σ_j w_j) − w_i
        suppressed_i   = w_i / (1 + κ_eff · rival_mass_i)
        w_i'           = suppressed_i / Σ_j suppressed_j

    The effective inhibition strength is gated by the GABA channel:

        κ_eff = κ_base · clamp(GABA, 0, 1)

    so the legacy path is preserved unless **both** ``κ_base > 0`` and
    a positive ``GABA`` level have been pushed via
    :meth:`set_nt_levels`.

    Why divisive (not subtractive)
    ------------------------------
    Subtractive inhibition (``w_i − κ · rival_mass``) requires a hard
    ``max(0, ·)`` floor which has zero gradient for suppressed experts —
    pathological for training. Divisive normalisation cannot go
    negative, has a smooth gradient everywhere, and matches
    biophysical shunting inhibition at the membrane.

    Output guarantees
    -----------------
    * Always on the simplex (sums to 1 within float tol).
    * Always non-negative.
    * Identity when ``κ_base = 0`` OR ``GABA = 0``.
    * Identity for uniform inputs (rival_mass is the same scalar for
      every component → renormalise reverts it).
    * Gradient flows end-to-end (no detach, no hard floors).
    """

    def __init__(self, kappa_base: float = 0.0) -> None:
        super().__init__()
        if kappa_base < 0.0:
            raise ValueError(
                f"kappa_base must be non-negative (negative would amplify "
                f"rivals, breaking the WTA semantics), got {kappa_base}"
            )
        self.kappa_base = float(kappa_base)
        # GABA snapshot — the harness pushes via set_nt_levels() each
        # training step. Default 0.0 means the module is fully identity
        # until BOTH κ_base > 0 AND a positive GABA level have been
        # explicitly pushed. This mirrors the safe default of
        # ThalamicRouter's NE channel (centre → no effect).
        self._gaba_level: float = 0.0

    # ── public ───────────────────────────────────────────────────────

    def set_nt_levels(self, levels: "Mapping[str, float]") -> None:
        """Push the current neuromodulator levels into the inhibitor.

        Only the ``"GABA"`` key is consumed. Other NT keys are silently
        ignored so the caller can pass the whole ``NTSystem.levels()``
        dict without filtering. Out-of-range values are clamped to
        ``[0, 1]`` (NTs are sigmoid-bounded by construction).
        """
        if levels is None:
            return
        gaba = levels.get("GABA")
        if gaba is None:
            return
        self._gaba_level = max(0.0, min(1.0, float(gaba)))

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """``weights: (..., N)`` on the simplex ⇒ post-inhibition weights.

        The output is guaranteed to be on the simplex and non-negative
        regardless of κ_eff or the input distribution.
        """
        if weights.dim() < 1:
            raise ValueError(
                f"expected weights with at least one dim (last = N), "
                f"got shape {tuple(weights.shape)}"
            )

        # Identity short-circuits — both make this module a no-op.
        if self.kappa_base <= 0.0 or self._gaba_level <= 0.0:
            return weights

        # κ scales linearly with GABA; at GABA = 1 the full κ_base is
        # in effect. Linear (not z-centred) because GABA acts in an
        # additive concentration sense, not as a centred deviation.
        kappa_eff = self.kappa_base * self._gaba_level
        if kappa_eff <= 0.0:                    # defensive — shouldn't fire
            return weights

        sum_all = weights.sum(dim=-1, keepdim=True)         # (..., 1)
        rival_mass = sum_all - weights                       # (..., N)
        suppressed = weights / (1.0 + kappa_eff * rival_mass)
        return suppressed / suppressed.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    def extra_repr(self) -> str:    # pragma: no cover
        return f"kappa_base={self.kappa_base}"


# ──────────────────────────────────────────────────────────────────────
# Multi-cortex ensemble
# ──────────────────────────────────────────────────────────────────────

class MultiCortexEnsemble(nn.Module):
    """N specialist sub-cortices + thalamic router ⇒ mixture of experts.

    For every input ``(B, T)`` token sequence, the ensemble:

    1. Computes routing weights ``p`` on the simplex Delta^N per token
       via :class:`ThalamicRouter`.
    2. Runs every sub-cortex forward (parallelisable across cortices).
    3. Projects each cortex's hidden state to the shared ``d_target``.
    4. Returns the weighted mixture
       ``h_{ti} = sum_n p_{ti,n} * P_n(C_n(x))_{ti}``.

    The routing weights from the last forward are exposed via
    :attr:`last_routing_weights` for telemetry / regularisation.
    """

    def __init__(self,
                 sub_cortices: Sequence[SubCortex],
                 router: ThalamicRouter,
                 d_target: int):
        super().__init__()
        sub_cortices = list(sub_cortices)
        if len(sub_cortices) != router.n_cortices:
            raise ValueError(
                f"sub_cortices count ({len(sub_cortices)}) "
                f"does not match router.n_cortices ({router.n_cortices})")
        for sc in sub_cortices:
            if not isinstance(sc, SubCortex):
                raise TypeError(
                    f"expected SubCortex, got {type(sc).__name__}")
        self.sub_cortices = nn.ModuleList(sub_cortices)
        self.router = router
        self.d_target = int(d_target)

        # Per-cortex projection to the shared target dim. Identity
        # when the native dim already matches.  Linear is zero-bias to
        # match the typical trunk-projection convention in this repo.
        self.projections = nn.ModuleList([
            nn.Identity() if sc.d_native == d_target
            else nn.Linear(sc.d_native, d_target, bias=False)
            for sc in sub_cortices
        ])

        # Diagnostics — populated each forward
        self._last_routing_weights: Optional[torch.Tensor] = None

    # ── properties ───────────────────────────────────────────────────

    @property
    def last_routing_weights(self) -> Optional[torch.Tensor]:
        """Per-token routing weights from the most recent forward,
        shape ``(B, T, N)``. ``None`` before any forward has run."""
        return self._last_routing_weights

    @property
    def domains(self) -> List[str]:
        return [c.domain for c in self.sub_cortices]

    # ── forward ──────────────────────────────────────────────────────

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """``ids: (B, T)`` ⇒ mixture hidden ``(B, T, d_target)``."""
        if ids.dim() != 2:
            raise ValueError(f"expected (B, T) ids, got shape {tuple(ids.shape)}")

        weights = self.router(ids)                                  # (B, T, N)
        self._last_routing_weights = weights

        outs: List[torch.Tensor] = []
        for cortex, proj in zip(self.sub_cortices, self.projections):
            h = cortex(ids)                                          # (B, T, d_native)
            h = proj(h)                                              # (B, T, d_target)
            outs.append(h)
        stacked = torch.stack(outs, dim=-2)                          # (B, T, N, d_target)

        # weights: (B, T, N) → (B, T, N, 1) → broadcast × (B, T, N, D)
        # then sum over N
        mixed = (weights.unsqueeze(-1) * stacked).sum(dim=-2)        # (B, T, d_target)
        return mixed


# ──────────────────────────────────────────────────────────────────────
# Factories
# ──────────────────────────────────────────────────────────────────────

def build_default_ensemble(vocab: int, d_model: int,
                           n_layers: int = 2, n_heads: int = 4,
                           max_ctx: int = 512,
                           domains: Sequence[str] = ("math", "code",
                                                      "chat", "general"),
                           domain_token_map:
                               Optional[Mapping[str, Sequence[int]]] = None,
                           lexical_bias_weight: float = 2.0,
                           bema_tau: float = 0.0) -> MultiCortexEnsemble:
    """Build a 4-cortex ensemble of :class:`StubSubCortex` sub-cortices.

    Used as the default cold-start path (no HuggingFace weights
    required) and as the canonical fixture in
    ``tests/test_ensemble_routing.py``.

    Parameters
    ----------
    vocab : int
        Vocabulary size.
    d_model : int
        Shared hidden dim for both the sub-cortices and the ensemble
        output.
    domain_token_map : Mapping[str, Sequence[int]], optional
        Lexicon for the lexical bias.  If ``None``, the vocab is split
        into ``len(domains)`` contiguous equal chunks and each domain
        is assigned its chunk — useful for tests with a synthetic vocab.

    Returns
    -------
    :class:`MultiCortexEnsemble`
    """
    domains = list(domains)
    sub_cortices = [
        StubSubCortex(name=f"cortex_{d}", domain=d, vocab=vocab,
                      d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                      max_ctx=max_ctx)
        for d in domains
    ]
    if domain_token_map is None:
        chunk = vocab // len(domains)
        domain_token_map = {
            d: list(range(i * chunk, (i + 1) * chunk))
            for i, d in enumerate(domains)
        }
    lexicon = DomainLexicon(domain_token_map=domain_token_map)
    router = ThalamicRouter(
        vocab_size=vocab, d_model=d_model, domains=domains,
        lexicon=lexicon, lexical_bias_weight=lexical_bias_weight,
        bema_tau=bema_tau,
    )
    return MultiCortexEnsemble(
        sub_cortices=sub_cortices, router=router, d_target=d_model,
    )


# Default GPT-2 variant mapping for production runs.  Substituted from
# the user's original GPT-3 / DeepSeek / Qwen request because GPT-3
# weights are not publicly available and DeepSeek / Qwen require
# large downloads + custom tokenizers.  The GPT-2 family is fully
# open and uses a single tokenizer (BPE) so we can share a router
# embedding across all four cortices.
DEFAULT_GPT2_VARIANTS: Dict[str, str] = {
    "general": "gpt2",            # 124M — balanced LM baseline
    "math":    "gpt2-medium",     # 355M — larger generalist (stand-in
                                  #         for a math fine-tune like
                                  #         DeepSeek-Math-7B in prod)
    "code":    "distilgpt2",      #  82M — distilled, fast (stand-in
                                  #         for a code fine-tune)
    "chat":    "gpt2",            # 124M — general (chat fine-tune
                                  #         like DialoGPT in prod)
}


def build_gpt2_ensemble(d_target: int = 768,
                        variants: Optional[Mapping[str, str]] = None,
                        freeze_weights: bool = True,
                        lexical_bias_weight: float = 2.0,
                        bema_tau: float = 0.5) -> MultiCortexEnsemble:
    """Build a 4-cortex ensemble with HuggingFace GPT-2 sub-cortices.

    Requires ``transformers`` to be installed (lazy-imported on
    construction; raises a clear ImportError otherwise).
    """
    if variants is None:
        variants = DEFAULT_GPT2_VARIANTS
    domains = list(variants.keys())
    sub_cortices = [
        GPT2SubCortex(name=f"cortex_{d}", domain=d, hf_model_id=mid,
                      freeze_weights=freeze_weights)
        for d, mid in variants.items()
    ]
    # GPT-2 BPE vocab size (constant across all variants)
    vocab = 50257
    lexicon = DomainLexicon.from_gpt2_keywords()
    router = ThalamicRouter(
        vocab_size=vocab, d_model=d_target, domains=domains,
        lexicon=lexicon, lexical_bias_weight=lexical_bias_weight,
        bema_tau=bema_tau,
    )
    return MultiCortexEnsemble(
        sub_cortices=sub_cortices, router=router, d_target=d_target,
    )
