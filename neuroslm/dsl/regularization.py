# -*- coding: utf-8 -*-
"""`regularization { ... }` block — five OOD interventions, math-first.

Each intervention is parsed into a structured dataclass that the BRIAN
harness consumes in PR2. The math for each lives in
`architectures/master/lib/regularizers.neuro` as canonical equations;
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
    lam: float = 1.0          # reweighting temperature (lambda, sample weighting)
    hidden: int = 64          # discriminator MLP hidden dim
    grl_alpha: float = 0.1    # GRL gradient flip scale
    weight: float = 0.1       # global scale applied to the discriminator BCE
                              # before it enters the total loss. 0.1 is the
                              # standard adversarial-training default (Ganin
                              # et al. 2015). 1.0 = legacy (drowns LM signal).


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
    weight: float = 0.1              # global scale on the InfoNCE term before
                                     # it enters the total loss. Standard CPC
                                     # value (Oord et al. 2018, He et al.
                                     # 2020). Raw InfoNCE saturates near
                                     # log(n_negatives+1) ≈ 4 when the model
                                     # hasn't learned predictive features, so
                                     # weight=1.0 dominates the LM gradient
                                     # and freezes convergence.


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

        direction = "balance" (default, CORRECTED — entropy-parity):
            chat_ratio_{t+1} = clip( chat_ratio_t · (H_target / H_t)^γ,
                                      [min_ratio, max_ratio] )
          → high prose entropy (H_t > H_target) ⇒ SHRINK chat (the model
            needs more prose exposure to bring H down to target).

        direction = "amplify" (legacy bug — kept for ablation only):
            chat_ratio_{t+1} = clip( chat_ratio_t · (H_t / H_target)^γ,
                                      [min_ratio, max_ratio] )
          → high prose entropy ⇒ GROW chat (the failure mode that
            drove the 2026-06-03 gap_ratio regression: chat_ratio
            ran from 0.60 to 0.80 max in <100 steps while wikitext
            ppl barely improved).

    Default max_ratio is now 0.50 (was 0.80) so prose always receives
    at least half of the training tokens regardless of controller drift.
    """
    enabled: bool = False
    target_entropy: float = 4.5
    probe_interval: int = 100     # measure entropy every N training steps
    gamma: float = 2.0            # control gain exponent
    min_ratio: float = 0.10
    max_ratio: float = 0.50       # CHANGED 2026-06-03 from 0.80 (see docstring)
    direction: str = "balance"    # NEW 2026-06-03 ("balance" | "amplify")

    # ── Neuromechanical stabilisation (added 2026-06-03 second pass) ──
    # First "balance" run still failed because the controller observes
    # the entropy of *training-data logits*, not a held-out prose probe.
    # At init H_t ≈ log(V) ≈ 10.5 regardless of input — so the first
    # update slams chat_ratio from 0.60 to the min_ratio floor in 20
    # steps, before the LM has any features, then training stalls.
    #
    # Fix mirrors the three properties of biological gain controllers
    # (retinal adaptation, thalamic relay neurons):
    #   • controller_warmup_steps — startup grace; no updates fire until
    #     the LM has had time to form baseline features (≥ ~2k steps).
    #     Mirrors the "settling" time of retinal photoreceptors.
    #   • max_step_delta — slew-rate limit on |Δratio| per update.
    #     Mirrors the membrane time-constant: gain cannot change faster
    #     than the integrator allows, even if the input demands it.
    #   • entropy_ema_alpha — low-pass filter on the entropy observation.
    #     A single bad mini-batch cannot drive the controller; the
    #     signal must be sustained for ≈ 1/α probes to take full effect.
    #
    # Together these turn a single-update bang-bang controller into a
    # damped first-order system with a well-defined time constant.
    #
    # NOTE on defaults: the dataclass defaults are intentionally
    # *no-ops* (warmup=0, max_step_delta=1.0, ema_alpha=1.0) so the
    # legacy/test behaviour is unchanged unless a caller opts in.
    # The production arch.neuro config sets the protective values
    # (warmup=2000, max_step_delta=0.03, ema_alpha=0.1).
    controller_warmup_steps: int = 0       # 0 = updates fire immediately
    max_step_delta: float = 1.0            # 1.0 = no slew limit
    entropy_ema_alpha: float = 1.0         # 1.0 = no EMA smoothing


# ── Intervention F: Frequency-balanced cross-entropy ─────────────────

@dataclass
class FreqBalanceConfig:
    """Token-level inverse-frequency reweighting toward an OOD target.

    Math (see lib/regularizers.neuro :: freq_balance_*):
        freq_train[v]  = pre-computed unigram freq on training sample
        freq_target[v] = pre-computed unigram freq on OOD probe sample
        ratio[v]       = freq_target[v] / (freq_train[v] + ε)
        w[v]           = clip( ratio[v]^β, w_min, w_max )
        w[v]           = w[v] / mean(w[v])                  # mean-normalised
        L_freq         = mean( w[targets] * CE_per_token )

    Parameters mirror Mikolov 2013 / Cui et al. 2019 (class-balanced
    loss): β = 0.5 is the square-root smoothing default; β = 1.0 is
    exact inverse frequency; β = 0 is identity. The mean-normalisation
    keeps the loss scale unchanged on average so the LR schedule
    remains valid; only the per-token *direction* of the gradient is
    biased toward the OOD distribution.
    """
    enabled: bool = False
    beta: float = 0.5             # sqrt smoothing (Mikolov 2013)
    w_min: float = 0.2            # clip floor (prevent zero-weight tokens)
    w_max: float = 5.0            # clip ceiling (prevent loss explosion)


# ── Intervention G: Cross-Distribution Gradient Alignment (CDGA) ─────

@dataclass
class CDGAConfig:
    """Gradient-surgery against a frozen OOD anchor batch.

    Math (see lib/cdga.neuro for full derivation + proofs):
        g_train  = ∇L(x_train; θ)
        g_anchor = ∇L(x_anchor; θ)   # on held-out prose, no weight update
        c        = max(0, -<g_train, g_anchor> / <g_anchor, g_anchor>)
        g_aligned = g_train - α · c · g_anchor

    `c` is non-zero only when the training step would actively HURT
    the anchor (dot product negative). The surgery projects out that
    conflicting component — a single-task analogue of PCGrad (Yu et
    al. 2020, NeurIPS) where the "second task" is the same LM loss
    on the OOD distribution.

    `alpha_max`: strength cap. 1.0 = full projection; 0 = telemetry only.
    `warmup_steps`: linear ramp of α from 0 to alpha_max. CDGA only
        makes sense after the LM has formed baseline features; before
        warmup completes both gradients are noise and surgery is
        a coin-flip.
    `refresh_every`: re-sample the anchor batch every N optimizer
        steps. Higher = cheaper. Default 4 → +25% step time when
        anchor batch is 1/4 size of training batch.
    """
    enabled: bool = False
    alpha_max: float = 1.0
    warmup_steps: int = 2000      # mirrors outer warmup
    refresh_every: int = 4


# ── Intervention H: Pontryagin / Hopfion-lite topological charge ─────

@dataclass
class PontryaginTopoChargeConfig:
    """Per-head topological-charge diagnostic + optional soft penalty.

    Math (see neuroslm/mechanisms/topo_charge.py):
        n_h(t)         = F.normalize(W_proj @ attn_out_h(t))    on S^2
        Q_h            = sum_l sum_t  Omega(n_t, n_{t+1}, n_{t+2}) / (4*pi)
        eps_ortho      = sum_l mean(1 - n_{l+1} . n_l)
        L_topo         = alpha * mean((Q_h - Q_target)^2)
                         + gamma * eps_ortho

    Default disabled. When enabled with alpha=gamma=0 the mechanism
    is DIAGNOSTIC-ONLY: Q_h and eps_ortho are computed and logged
    every step, but zero is added to the loss (no gradient pressure).
    Setting alpha > 0 or gamma > 0 turns the soft penalty on.

    Per CLAUDE.md sec 14 the diagnostic-only branch is STRUCTURAL
    (the penalty multiplies by alpha and gamma; both zero -> exact
    zero contribution), NOT an early-return. The mechanism is not
    decorative.

    Reference: docs/formal_framework.md (cellular sheaf cohomology);
    Berg & Lueschner 1981 (Nucl. Phys. B 190); van Oosterom & Strang
    1983 (IEEE Trans. Biomed. Eng. 30).
    """
    enabled: bool = False
    alpha: float = 0.0            # weight on (Q_h - Q_target)^2 term
    gamma: float = 0.0            # weight on inter-layer eps_ortho term
    Q_target: float = 0.0         # target winding (default: irrotational)
    weight_init_std: float = 0.02 # std of Linear(head_dim, 3) init


# ── Intervention I: Liouville Symplectic Residual ────────────────────

@dataclass
class LiouvilleSymplecticConfig:
    """Noether-residual regularizer from the leapfrog of a learned Hamiltonian.

    Splits d_model into canonical coordinates (q, p) and advances them
    via one Stoermer-Verlet step of H(q,p) = KE(p) + V(q) + W(q).
    The Noether residual L_Noether = (H_final - H_initial)^2 is added
    to the LM loss with weight noether_strength. At noether_strength=0
    the block runs as a diagnostic (H_diff is logged) with zero loss
    contribution.

    Fields:
        enabled          -- wire the mechanism at all (False = zero overhead).
        noether_strength -- weight on L_Noether (>= 0).
        dtau_init        -- initial leapfrog stride (> 0).
        potential_kind   -- "quadratic" (closed-form, stable) or "swiglu".
        w_rank           -- rank of the LowRankPairwise W potential.
    """
    enabled: bool = False
    noether_strength: float = 0.0
    dtau_init: float = 0.1
    potential_kind: str = "quadratic"
    w_rank: int = 4


# ── Intervention J: KJPLA-v2 Phase Lattice Attention ────────────────

@dataclass
class KJPLAPhaseLatticeConfig:
    """Kuramoto-Josephson Phase Lattice Attention config.

    Adds per-(head, layer, token) phase φ with:
      - Intra-layer Kuramoto sync (eta)
      - Inter-layer Josephson order parameter R_ℓ (K_h)
      - Phase-gated attention logits (beta_h)

    At josephson_strength=0 the mechanism runs diagnostically (R_ℓ is
    logged) with zero loss contribution.  At josephson_strength>0 the
    term L_J = -(1/L) Σ K̄_h · R_ℓ is added to the loss.

    entropy_strength > 0 adds a soft phase-entropy floor loss to prevent
    phase collapse (all heads synchronize to the same phase).

    Fields:
        enabled            -- wire the mechanism (False = zero overhead).
        josephson_strength -- weight on the Josephson coupling loss (>= 0).
        entropy_strength   -- weight on the phase-entropy floor loss (>= 0).
        eps_H              -- entropy floor in nats (> 0).
    """
    enabled: bool = False
    josephson_strength: float = 0.0
    entropy_strength: float = 0.0
    eps_H: float = 0.5


# ── Top-level container ──────────────────────────────────────────────

@dataclass
class RegularizationConfig:
    """Container for the five OOD interventions.

    `warmup_steps`: number of training steps over which the global
    intervention strength ramps linearly from 0 → 1. This is the
    architectural fix for training instability: at step 0, hidden
    states are random noise, so InfoNCE (PCC) explodes, Isotropy
    pushes toward identity prematurely, and DAR's gradient reversal
    disrupts representation formation. The warmup lets the LM
    establish baseline features before regularizers engage.

    Recommended: 2000-5000 steps (≈ 5-15% of total training).
    Set to 0 to disable warmup (full strength from step 0 — the
    legacy behaviour that caused gnorm=10-200 instability).
    """
    dar: DARConfig = field(default_factory=DARConfig)
    pcc: PCCConfig = field(default_factory=PCCConfig)
    isotropy: IsotropyConfig = field(default_factory=IsotropyConfig)
    cmd: CMDConfig = field(default_factory=CMDConfig)
    adaptive_mixture: AdaptiveMixtureConfig = field(
        default_factory=AdaptiveMixtureConfig)
    freq_balance: FreqBalanceConfig = field(
        default_factory=FreqBalanceConfig)
    cdga: CDGAConfig = field(default_factory=CDGAConfig)
    pontryagin_topo_charge: PontryaginTopoChargeConfig = field(
        default_factory=PontryaginTopoChargeConfig)
    liouville_symplectic: LiouvilleSymplecticConfig = field(
        default_factory=LiouvilleSymplecticConfig)
    kjpla_phase_lattice: KJPLAPhaseLatticeConfig = field(
        default_factory=KJPLAPhaseLatticeConfig)
    warmup_steps: int = 2000
    activation_step: int = 0
    """First global step at which any aux loss may be non-zero.

    Capacity-First protocol (Hypothesis H-A2):
    Aux losses must NOT fire until the LM trunk has reached its
    pretrain capacity floor (train PPL ≈ stable).  Setting
    activation_step > 0 hard-gates all aux losses to zero for the
    first N steps — the warmup ramp then starts at activation_step
    rather than at step 0.

    Recommended: activation_step = 4 × warmup_steps (i.e. the full
    warmup epoch fires well after pretrain PPL plateaus).
    Set to 0 (default) for legacy behaviour: interventions fire from
    step 0 with only the warmup ramp for protection.
    """
    isotropy_activation_step: int = -1
    """Per-intervention override for isotropy (whitening/rank collapse guard).

    -1 (default): use the global activation_step value.
    >= 0: isotropy fires independently at this step, even before the
    global activation_step gate opens for DAR/PCC/CMD.

    Rationale: representation rank collapse (erank → 1) can happen
    during the Capacity-First window when all aux losses are gated off.
    The whitening loss is low-weight and safe to fire early; other
    interventions (DAR, PCC) require more LM capacity before they help.

    Recommended for SmolLM: isotropy_activation_step: 1000
    (fires after initial stabilisation, ~1500 steps before DAR/PCC).
    """

    def any_enabled(self) -> bool:
        return any([
            self.dar.enabled,
            self.pcc.enabled,
            self.isotropy.enabled,
            self.cmd.enabled,
            self.adaptive_mixture.enabled,
            self.freq_balance.enabled,
            self.cdga.enabled,
            self.pontryagin_topo_charge.enabled,
            self.liouville_symplectic.enabled,
            self.kjpla_phase_lattice.enabled,
        ])


# ── Validation vocabularies ──────────────────────────────────────────
# These mirror lib/vocabulary.neuro so the DSL and Python agree on the
# enum surface. Kept in sync via tests/dsl/test_vocabulary_parity.py
# (PR4 — for now both files just need to match by inspection).

_VALID_CMD_DIVERGENCES = {"jsd", "kl_sym", "l1"}
_VALID_ISOTROPY_DISTANCES = {"frobenius", "log_det"}
_VALID_MIXTURE_DIRECTIONS = {"balance", "amplify"}


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

    if "warmup_steps" in props:
        cfg.warmup_steps = int(props["warmup_steps"])
    if "activation_step" in props:
        cfg.activation_step = int(props["activation_step"])
    if "isotropy_activation_step" in props:
        cfg.isotropy_activation_step = int(props["isotropy_activation_step"])
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
    if "freq_balance" in props:
        cfg.freq_balance = _parse_freq_balance(props["freq_balance"])
    if "cdga" in props:
        cfg.cdga = _parse_cdga(props["cdga"])
    if "pontryagin_topo_charge" in props:
        cfg.pontryagin_topo_charge = _parse_pontryagin_topo_charge(
            props["pontryagin_topo_charge"])
    if "liouville_symplectic" in props:
        cfg.liouville_symplectic = _parse_liouville_symplectic(
            props["liouville_symplectic"])
    if "kjpla_phase_lattice" in props:
        cfg.kjpla_phase_lattice = _parse_kjpla_phase_lattice(
            props["kjpla_phase_lattice"])

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
    if "weight" in p:     out.weight = float(p["weight"])
    return out


def _parse_pcc(raw: str) -> PCCConfig:
    p = _split_top_level_kv(_strip_braces(raw))
    out = PCCConfig()
    if "enabled" in p:     out.enabled = _parse_bool(p["enabled"])
    if "k" in p:           out.k = int(p["k"])
    if "n_negatives" in p: out.n_negatives = int(p["n_negatives"])
    if "tau" in p:         out.tau = float(p["tau"])
    if "layers" in p:      out.layers = _parse_int_list(p["layers"])
    if "weight" in p:      out.weight = float(p["weight"])
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
    if "direction" in p:
        d = _strip_quotes(p["direction"])
        if d not in _VALID_MIXTURE_DIRECTIONS:
            raise ValueError(
                f"adaptive_mixture.direction={d!r}; expected one of "
                f"{sorted(_VALID_MIXTURE_DIRECTIONS)}"
            )
        out.direction = d
    if "controller_warmup_steps" in p:
        out.controller_warmup_steps = int(p["controller_warmup_steps"])
    if "max_step_delta" in p:
        out.max_step_delta = float(p["max_step_delta"])
    if "entropy_ema_alpha" in p:
        out.entropy_ema_alpha = float(p["entropy_ema_alpha"])
    if out.min_ratio > out.max_ratio:
        raise ValueError(
            f"adaptive_mixture.min_ratio={out.min_ratio} > "
            f"max_ratio={out.max_ratio}"
        )
    return out


def _parse_freq_balance(raw: str) -> FreqBalanceConfig:
    p = _split_top_level_kv(_strip_braces(raw))
    out = FreqBalanceConfig()
    if "enabled" in p: out.enabled = _parse_bool(p["enabled"])
    if "beta" in p:    out.beta = float(p["beta"])
    if "w_min" in p:   out.w_min = float(p["w_min"])
    if "w_max" in p:   out.w_max = float(p["w_max"])
    if out.w_min > out.w_max:
        raise ValueError(
            f"freq_balance.w_min={out.w_min} > w_max={out.w_max}")
    return out


def _parse_cdga(raw: str) -> CDGAConfig:
    p = _split_top_level_kv(_strip_braces(raw))
    out = CDGAConfig()
    if "enabled" in p:       out.enabled = _parse_bool(p["enabled"])
    if "alpha_max" in p:     out.alpha_max = float(p["alpha_max"])
    if "warmup_steps" in p:  out.warmup_steps = int(p["warmup_steps"])
    if "refresh_every" in p: out.refresh_every = int(p["refresh_every"])
    if out.alpha_max < 0.0:
        raise ValueError(f"cdga.alpha_max={out.alpha_max} must be >= 0")
    if out.refresh_every < 1:
        raise ValueError(
            f"cdga.refresh_every={out.refresh_every} must be >= 1")
    return out


def _parse_pontryagin_topo_charge(raw: str) -> PontryaginTopoChargeConfig:
    p = _split_top_level_kv(_strip_braces(raw))
    out = PontryaginTopoChargeConfig()
    if "enabled" in p:           out.enabled = _parse_bool(p["enabled"])
    if "alpha" in p:             out.alpha = float(p["alpha"])
    if "gamma" in p:             out.gamma = float(p["gamma"])
    if "Q_target" in p:          out.Q_target = float(p["Q_target"])
    if "weight_init_std" in p:
        out.weight_init_std = float(p["weight_init_std"])
    if out.alpha < 0.0:
        raise ValueError(
            f"pontryagin_topo_charge.alpha={out.alpha} must be >= 0")
    if out.gamma < 0.0:
        raise ValueError(
            f"pontryagin_topo_charge.gamma={out.gamma} must be >= 0")
    if out.weight_init_std <= 0.0:
        raise ValueError(
            f"pontryagin_topo_charge.weight_init_std="
            f"{out.weight_init_std} must be > 0")
    return out


def _parse_kjpla_phase_lattice(raw: str) -> KJPLAPhaseLatticeConfig:
    p = _split_top_level_kv(_strip_braces(raw))
    out = KJPLAPhaseLatticeConfig()
    if "enabled" in p:              out.enabled = _parse_bool(p["enabled"])
    if "josephson_strength" in p:   out.josephson_strength = float(p["josephson_strength"])
    if "entropy_strength" in p:     out.entropy_strength = float(p["entropy_strength"])
    if "eps_H" in p:                out.eps_H = float(p["eps_H"])
    if out.josephson_strength < 0.0:
        raise ValueError(
            f"kjpla_phase_lattice.josephson_strength="
            f"{out.josephson_strength} must be >= 0")
    if out.entropy_strength < 0.0:
        raise ValueError(
            f"kjpla_phase_lattice.entropy_strength="
            f"{out.entropy_strength} must be >= 0")
    if out.eps_H <= 0.0:
        raise ValueError(
            f"kjpla_phase_lattice.eps_H={out.eps_H} must be > 0")
    return out


def _parse_liouville_symplectic(raw: str) -> LiouvilleSymplecticConfig:
    p = _split_top_level_kv(_strip_braces(raw))
    out = LiouvilleSymplecticConfig()
    if "enabled" in p:           out.enabled = _parse_bool(p["enabled"])
    if "noether_strength" in p:  out.noether_strength = float(p["noether_strength"])
    if "dtau_init" in p:         out.dtau_init = float(p["dtau_init"])
    if "potential_kind" in p:    out.potential_kind = _strip_quotes(p["potential_kind"])
    if "w_rank" in p:            out.w_rank = int(p["w_rank"])
    if out.noether_strength < 0.0:
        raise ValueError(
            f"liouville_symplectic.noether_strength={out.noether_strength} "
            f"must be >= 0")
    if out.dtau_init <= 0.0:
        raise ValueError(
            f"liouville_symplectic.dtau_init={out.dtau_init} must be > 0")
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
