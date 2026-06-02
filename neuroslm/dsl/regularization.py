# -*- coding: utf-8 -*-
"""`regularization { ... }` block — five OOD interventions, math-first.

Each intervention is parsed into a structured dataclass that the BRIAN
harness consumes in PR2. The math for each lives in
`architectures/rcc_bowtie/lib/regularizers.neuro` as canonical equations;
the dataclasses here are the *parameterization* of those equations.

Interventions (rationale in docs/technical_report.md §3):

  A. dar              — Distributional Adversarial Reweighting (anti-shortcut)
  B. pcc              — Predictive Contrastive Coding (replaces PCT)
  C. isotropy         — Whitening loss on token embeddings (anti-collapse)
  D. cmd              — Cross-Module Disagreement (JSD between heads)
  E. adaptive_mixture — Entropy-targeted mixture controller

All default disabled → zero behavioral change vs. legacy arch.neuro.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

from .training_config import (
    _split_top_level_kv,
    _strip_braces,
    _strip_quotes,
    _parse_bool,
)


# ── Intervention A: Distributional Adversarial Reweighting ────────────

@dataclass
class DARConfig:
    """Source-discriminator + gradient-reversal reweighter.

    Math (see lib/regularizers.neuro :: dar_loss):
        d(h)     = σ(W_d · GRL_α(h))            # source classifier
        L_disc   = BCE(d(h), domain_label)
        w_i      = exp(lam · L_ce_i · 1[minority(i)])
        L_total  = E[w_i · L_ce_i] + L_disc
    """
    enabled: bool = False
    lam: float = 1.0          # reweighting temperature (lambda)
    hidden: int = 64          # discriminator MLP hidden dim
    grl_alpha: float = 0.1    # GRL gradient flip scale


# ── Intervention B: Predictive Contrastive Coding ────────────────────

@dataclass
class PCCConfig:
    """InfoNCE over (h_t, h_{t+k}) within / across documents.

    Replaces the reconstructive PCT loss with a contrastive objective
    that pushes deep layers toward invariant features.

    Math (see lib/regularizers.neuro :: pcc_loss):
        z_t      = proj(h_t^(n))                  # at selected layers n
        z_pos    = proj(h_{t+k}^(n))              # same-document future
        z_neg_j  ~ buffer of cross-document samples
        L_pcc    = -log( exp(<z_t, z_pos>/τ) /
                         Σ_{j=neg} exp(<z_t, z_neg_j>/τ) )
    """
    enabled: bool = False
    k: int = 4                       # prediction horizon (future offset)
    n_negatives: int = 64            # cross-document negatives per anchor
    tau: float = 0.1                 # InfoNCE temperature
    layers: List[int] = field(default_factory=list)  # [] = all layers


# ── Intervention C: Isotropy whitening ───────────────────────────────

@dataclass
class IsotropyConfig:
    """Online whitening loss that pushes the Gram matrix toward I.

    Math (see lib/regularizers.neuro :: isotropy_loss):
        H_buf  ∈ R^{N x d}              # rolling buffer of token embeddings
        G      = H_bufᵀ H_buf / N
        L_iso  = ||G - I||_F² / d²

    `distance`:
      "frobenius" — squared Frobenius norm (default, smooth, cheap)
      "log_det"   — -log|det(G)| (sharper isotropy; numerically delicate)
    """
    enabled: bool = False
    weight: float = 0.01
    buffer: int = 4096
    distance: str = "frobenius"


# ── Intervention D: Cross-Module Disagreement ────────────────────────

@dataclass
class CMDConfig:
    """Penalize predictive divergence between two read-out heads.

    Math (see lib/regularizers.neuro :: cmd_loss):
        p_a = softmax(W_a · h_a)       # head A logits (e.g. lm)
        p_b = softmax(W_b · h_b)       # head B logits (e.g. narrative)
        m   = 0.5 · (p_a + p_b)
        JSD = 0.5 · KL(p_a || m) + 0.5 · KL(p_b || m)
        L_cmd = weight · JSD

    `divergence ∈ {jsd, kl_sym, l1}`. Default `jsd` — bounded in [0, ln 2].
    """
    enabled: bool = False
    weight: float = 0.05
    divergence: str = "jsd"
    heads: List[str] = field(default_factory=lambda: ["lm", "narrative"])


# ── Intervention E: Adaptive mixture controller ──────────────────────

@dataclass
class AdaptiveMixtureConfig:
    """Closed-loop controller that anneals chat_ratio to preserve entropy.

    Math (see lib/regularizers.neuro :: adaptive_mixture_update):
        H_t = -E_{x ∈ prose_probe} [ Σ_v p_θ(v|x) log p_θ(v|x) ]
        chat_ratio_{t+1} = clip( chat_ratio_t · (H_t / H_target)^γ,
                                 [min_ratio, max_ratio] )

    When H_t < H_target (entropy collapsed on prose), chat_ratio shrinks.
    """
    enabled: bool = False
    target_entropy: float = 4.5
    probe_interval: int = 100     # measure entropy every N training steps
    gamma: float = 2.0            # control gain exponent
    min_ratio: float = 0.10
    max_ratio: float = 0.80


# ── Top-level container ──────────────────────────────────────────────

@dataclass
class RegularizationConfig:
    """Container for the five OOD interventions."""
    dar: DARConfig = field(default_factory=DARConfig)
    pcc: PCCConfig = field(default_factory=PCCConfig)
    isotropy: IsotropyConfig = field(default_factory=IsotropyConfig)
    cmd: CMDConfig = field(default_factory=CMDConfig)
    adaptive_mixture: AdaptiveMixtureConfig = field(
        default_factory=AdaptiveMixtureConfig)

    def any_enabled(self) -> bool:
        return any([
            self.dar.enabled,
            self.pcc.enabled,
            self.isotropy.enabled,
            self.cmd.enabled,
            self.adaptive_mixture.enabled,
        ])


# ── Validation vocabularies ──────────────────────────────────────────
# These mirror lib/vocabulary.neuro so the DSL and Python agree on the
# enum surface. Kept in sync via tests/dsl/test_vocabulary_parity.py
# (PR4 — for now both files just need to match by inspection).

_VALID_CMD_DIVERGENCES = {"jsd", "kl_sym", "l1"}
_VALID_ISOTROPY_DISTANCES = {"frobenius", "log_det"}


# ── Parser ───────────────────────────────────────────────────────────

def parse_regularization_block(body: str) -> RegularizationConfig:
    """Parse the body of `regularization { ... }` (braces stripped).

    Empty body → all interventions disabled. Unknown sub-blocks are
    silently ignored for forward-compat with future interventions.
    """
    cfg = RegularizationConfig()
    if not body or not body.strip():
        return cfg

    props = _split_top_level_kv(body)

    if "dar" in props:
        cfg.dar = _parse_dar(props["dar"])
    if "pcc" in props:
        cfg.pcc = _parse_pcc(props["pcc"])
    if "isotropy" in props:
        cfg.isotropy = _parse_isotropy(props["isotropy"])
    if "cmd" in props:
        cfg.cmd = _parse_cmd(props["cmd"])
    if "adaptive_mixture" in props:
        cfg.adaptive_mixture = _parse_adaptive_mixture(props["adaptive_mixture"])

    return cfg


# ── Sub-parsers ──────────────────────────────────────────────────────

def _parse_dar(raw: str) -> DARConfig:
    p = _split_top_level_kv(_strip_braces(raw))
    out = DARConfig()
    if "enabled" in p:    out.enabled = _parse_bool(p["enabled"])
    if "lambda" in p:     out.lam = float(p["lambda"])
    if "lam" in p:        out.lam = float(p["lam"])
    if "hidden" in p:     out.hidden = int(p["hidden"])
    if "grl_alpha" in p:  out.grl_alpha = float(p["grl_alpha"])
    return out


def _parse_pcc(raw: str) -> PCCConfig:
    p = _split_top_level_kv(_strip_braces(raw))
    out = PCCConfig()
    if "enabled" in p:     out.enabled = _parse_bool(p["enabled"])
    if "k" in p:           out.k = int(p["k"])
    if "n_negatives" in p: out.n_negatives = int(p["n_negatives"])
    if "tau" in p:         out.tau = float(p["tau"])
    if "layers" in p:      out.layers = _parse_int_list(p["layers"])
    return out


def _parse_isotropy(raw: str) -> IsotropyConfig:
    p = _split_top_level_kv(_strip_braces(raw))
    out = IsotropyConfig()
    if "enabled" in p:  out.enabled = _parse_bool(p["enabled"])
    if "weight" in p:   out.weight = float(p["weight"])
    if "buffer" in p:   out.buffer = int(p["buffer"])
    if "distance" in p:
        dist = _strip_quotes(p["distance"])
        if dist not in _VALID_ISOTROPY_DISTANCES:
            raise ValueError(
                f"isotropy.distance={dist!r}; expected one of "
                f"{sorted(_VALID_ISOTROPY_DISTANCES)}"
            )
        out.distance = dist
    return out


def _parse_cmd(raw: str) -> CMDConfig:
    p = _split_top_level_kv(_strip_braces(raw))
    out = CMDConfig()
    if "enabled" in p:  out.enabled = _parse_bool(p["enabled"])
    if "weight" in p:   out.weight = float(p["weight"])
    if "divergence" in p:
        div = _strip_quotes(p["divergence"])
        if div not in _VALID_CMD_DIVERGENCES:
            raise ValueError(
                f"cmd.divergence={div!r}; expected one of "
                f"{sorted(_VALID_CMD_DIVERGENCES)}"
            )
        out.divergence = div
    if "heads" in p:    out.heads = _parse_string_list(p["heads"])
    return out


def _parse_adaptive_mixture(raw: str) -> AdaptiveMixtureConfig:
    p = _split_top_level_kv(_strip_braces(raw))
    out = AdaptiveMixtureConfig()
    if "enabled" in p:        out.enabled = _parse_bool(p["enabled"])
    if "target_entropy" in p: out.target_entropy = float(p["target_entropy"])
    if "probe_interval" in p: out.probe_interval = int(p["probe_interval"])
    if "gamma" in p:          out.gamma = float(p["gamma"])
    if "min_ratio" in p:      out.min_ratio = float(p["min_ratio"])
    if "max_ratio" in p:      out.max_ratio = float(p["max_ratio"])
    if out.min_ratio > out.max_ratio:
        raise ValueError(
            f"adaptive_mixture.min_ratio={out.min_ratio} > "
            f"max_ratio={out.max_ratio}"
        )
    return out


# ── List parsers (DSL: [a, b, c]) ────────────────────────────────────

def _parse_int_list(raw: str) -> List[int]:
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if not raw.strip():
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_string_list(raw: str) -> List[str]:
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if not raw.strip():
        return []
    return [_strip_quotes(x.strip()) for x in raw.split(",") if x.strip()]
