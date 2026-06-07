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
        # Causal mask — additive (-inf above diagonal, 0 on/below)
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=ids.device, dtype=h.dtype),
            diagonal=1,
        )
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.norm(h)


class GPT2SubCortex(SubCortex):
    """HuggingFace GPT-2 wrapper. Lazy-imports ``transformers``.

    Loads any model exposing the GPT-2 architecture (``gpt2``,
    ``gpt2-medium``, ``gpt2-large``, ``gpt2-xl``, ``distilgpt2``, or any
    HF Hub fine-tune of those).
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

    def forward(self, ids: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        out = self.gpt2(input_ids=ids, output_hidden_states=False,
                        return_dict=True)
        return out.last_hidden_state    # (B, T, d_native)


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
    """

    def __init__(self, vocab_size: int, d_model: int,
                 domains: Sequence[str], lexicon: DomainLexicon,
                 lexical_bias_weight: float = 2.0,
                 bema_tau: float = 0.0):
        super().__init__()
        if not 0.0 <= bema_tau < 1.0:
            raise ValueError(
                f"bema_tau must be in [0, 1), got {bema_tau}")
        if lexical_bias_weight < 0.0:
            raise ValueError(
                f"lexical_bias_weight must be non-negative, got {lexical_bias_weight}")
        self.domains = [str(d) for d in domains]
        self.n_cortices = len(self.domains)
        if self.n_cortices < 2:
            raise ValueError("Need at least 2 cortices to route between")
        self.lexicon = lexicon
        self.lexical_bias_weight = float(lexical_bias_weight)
        self.bema_tau = float(bema_tau)
        self.vocab_size = int(vocab_size)

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

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """``ids: (B, T)`` ⇒ routing weights ``(B, T, N)``."""
        if ids.dim() != 2:
            raise ValueError(f"expected (B, T) ids, got shape {tuple(ids.shape)}")
        bias = self._lexical_bias(ids)                              # (B, T, N)
        h = self.router_embed(ids)                                  # (B, T, d)
        learn = self.learnable_logits(h)                            # (B, T, N)
        logits = bias + learn
        weights = F.softmax(logits, dim=-1)
        return self._apply_bema(weights)

    def extra_repr(self) -> str:    # pragma: no cover
        return (f"n_cortices={self.n_cortices}, "
                f"domains={self.domains}, "
                f"lexical_bias_weight={self.lexical_bias_weight}, "
                f"bema_tau={self.bema_tau}")


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
