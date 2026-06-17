# -*- coding: utf-8 -*-
"""TRUNK-OPT: Measurement & Provability Layer for the LM trunk.

Implements all Phase 1, 4 probes for the four-phase LM optimisation
plan (TRUNK-OPT).  All classes are pure-observability: nothing here
changes the forward pass or the loss function.

Classes
-------
GradientBudgetTracker
    Measures what fraction of total gradient energy is attributable
    to the LM loss alone.  Hypothesis H-A1: ≥ 0.70 in pretrain.

LayerGradientProbe
    Per-block gradient L2 norms + uniformity ratio.
    Hypothesis H-B1: uniformity_ratio ≤ 2.0 at convergence.

BitsPerParamMeter
    Information per trainable parameter: (ln V − CE) / |θ|.
    Hypothesis H-C1: monotone increase during training.

PACBayesBound
    McAllester-style PAC-Bayes generalisation upper bound.
    Hypothesis H-C2: bound ≤ train_CE + 1.0 nat at convergence.

SharpnessProbe
    Empirical sharpness: E[L(θ + ε)] − L(θ) over random ε ~ N(0, ρI).
    Lower sharpness → better generalisation (Keskar et al. 2017).

EffectiveRankProbe
    Roy & Vetterli (2007) effective rank of hidden-state matrix:
        erank(H) = exp(H_ent(σ/||σ||₁)) ∈ [1, min(B,T,d)]
    Collapse if rank approaches 1.

TrunkOptMonitor
    Thin aggregator that BRIANHarness calls every step via
    ``harness.attach_trunk_opt_monitor(monitor)``.

References
----------
McAllester (1999, 2003) — PAC-Bayes bounds.
Roy & Vetterli (2007) — effective rank.
Keskar et al. (2017) — large-batch sharpness.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Iterator, Optional

import torch
import torch.nn as nn

__all__ = [
    "GradientBudgetTracker",
    "LayerGradientProbe",
    "BitsPerParamMeter",
    "PACBayesBound",
    "SharpnessProbe",
    "EffectiveRankProbe",
    "SpectralPowerLawProbe",
    "TrunkOptMonitor",
]

# ─────────────────────────────────────────────────────────────────────
# Phase 1A  ─  GradientBudgetTracker
# ─────────────────────────────────────────────────────────────────────

class GradientBudgetTracker:
    """Fraction of gradient energy going to the LM loss.

    Usage pattern (must be called *inside* the backward pass):

        tracker = GradientBudgetTracker()

        # 1. Forward + backward of the LM-only loss:
        loss_lm.backward(retain_graph=True)
        lm_norm = tracker.lm_grad_norm(model)

        # 2. Zero grads; full (LM + aux) backward:
        opt.zero_grad()
        total_loss.backward()
        total_norm = tracker.total_grad_norm(model)

        budget = tracker.budget(lm_norm, total_norm)

    Alternatively, if the harness stashes lm-only grads as a separate
    snapshot (see ``harness._stash_lm_grad_norm``), call
    ``budget(lm_norm, total_norm)`` directly.

    Hypothesis H-A1
    ---------------
    During pretrain (aux losses disabled) budget = 1.0 by construction.
    Once aux losses are enabled, budget should remain ≥ 0.70 to
    ensure the LM signal dominates.  If budget < 0.70, the
    aux-loss weights are too high and must be reduced.
    """

    @staticmethod
    def _grad_norm(model: nn.Module) -> float:
        """L2 norm of all .grad tensors currently attached to model."""
        total_sq: float = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_sq += p.grad.detach().float().norm(2).item() ** 2
        return math.sqrt(total_sq)

    def lm_grad_norm(self, model: nn.Module) -> float:
        """Compute the gradient L2 norm for the LM-only backward pass.

        Call this *after* ``loss_lm.backward(retain_graph=True)`` and
        *before* zeroing gradients or adding aux losses.
        """
        return self._grad_norm(model)

    def total_grad_norm(self, model: nn.Module) -> float:
        """Compute the gradient L2 norm after the full backward pass."""
        return self._grad_norm(model)

    @staticmethod
    def budget(lm_grad_norm: float, total_grad_norm: float) -> float:
        """Return LM gradient fraction ∈ [0, 1].

        budget = ||∇L_LM|| / ||∇L_total||

        Edge case: if both norms are zero (frozen model / early step
        with all zero grads) return 1.0 — the LM loss "owns" 100% of
        nothing, which is the correct interpretation for logging.
        """
        if total_grad_norm < 1e-12:
            return 1.0
        return float(min(1.0, lm_grad_norm / total_grad_norm))


# ─────────────────────────────────────────────────────────────────────
# Phase 1B  ─  LayerGradientProbe
# ─────────────────────────────────────────────────────────────────────

class LayerGradientProbe:
    """Per-block gradient L2 norms + uniformity ratio.

    Hypothesis H-B1
    ---------------
    At convergence, the max-to-mean gradient ratio across transformer
    blocks (uniformity_ratio) should be ≤ 2.0.  A ratio > 4.0
    signals vanishing/exploding gradients in specific layers.

    ``compute`` iterates over an iterable of ``nn.Module`` children
    (typically ``language_model.blocks`` or
    ``language_model.children()``).  Returns a dict keyed by integer
    block index.
    """

    @staticmethod
    def compute(
        layers: Iterable[nn.Module],
    ) -> Dict[int, float]:
        """Return {block_index: grad_L2_norm} for each child module."""
        norms: Dict[int, float] = {}
        for idx, layer in enumerate(layers):
            sq: float = 0.0
            for p in layer.parameters():
                if p.grad is not None:
                    sq += p.grad.detach().float().norm(2).item() ** 2
            norms[idx] = math.sqrt(sq)
        return norms

    @staticmethod
    def uniformity_ratio(norms: Dict[int, float]) -> float:
        """max(norms) / mean(norms), or 1.0 for single-layer models.

        Returns ≥ 1.0 and is finite.
        """
        if not norms:
            return 1.0
        vals = list(norms.values())
        mx   = max(vals)
        mean = sum(vals) / len(vals)
        if mean < 1e-12:
            return 1.0
        return float(mx / mean)


# ─────────────────────────────────────────────────────────────────────
# Phase 1C  ─  BitsPerParamMeter
# ─────────────────────────────────────────────────────────────────────

class BitsPerParamMeter:
    r"""Information per trainable parameter.

    .. math::

        \text{bits\_per\_param} = \frac{\max(0,\;\ln V - \text{CE})}{|\theta|}

    where V is the vocabulary size, CE is the current cross-entropy
    loss (in nats), and |θ| is the number of trainable parameters.

    Semantics
    ---------
    *  At init (CE ≈ ln V), bits_per_param ≈ 0.
    *  As training proceeds CE falls → bits_per_param rises.
    *  Hypothesis H-C1: strictly monotone increase ↔ model is learning.

    Clamped to 0 when CE > ln(V) (e.g. if flooding pushes CE above
    the random baseline — theoretically impossible but numerically
    can occur on the first step).
    """

    def __init__(self, vocab_size: int, n_trainable: int) -> None:
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be ≥ 1, got {vocab_size}")
        if n_trainable < 1:
            raise ValueError(f"n_trainable must be ≥ 1, got {n_trainable}")
        self._log_v = math.log(vocab_size)
        self._n = n_trainable

    def compute(self, ce: float) -> float:
        """Bits per param for cross-entropy `ce` (nats)."""
        return max(0.0, (self._log_v - ce) / self._n)


# ─────────────────────────────────────────────────────────────────────
# Phase 1D & 4  ─  PACBayesBound
# ─────────────────────────────────────────────────────────────────────

class PACBayesBound:
    r"""McAllester PAC-Bayes generalisation upper bound.

    The bound states that with probability ≥ 1 − δ over the training
    set:

    .. math::

        \mathbb{E}[L_{\text{OOD}}] \;\leq\;
        \hat{L}_{\text{train}}
        + \sqrt{\frac{D_{KL}(Q\|P) + \ln(2\sqrt{n}/\delta)}{2n}}

    where
    *  Q is the *posterior* (current model θ),
    *  P is the *prior* (θ₀ = init),
    *  D_KL(Q‖P) = ||θ − θ₀||² / (2σ²)  (isotropic Gaussian prior),
    *  n = n_train (number of training tokens seen),
    *  δ = confidence level (e.g. 0.05 for 95%).

    Parameters
    ----------
    n_train : int
        Number of training tokens / samples.
    delta : float
        Failure probability.  Bound holds w.p. ≥ 1 − δ.
    prior_sigma : float
        Standard deviation of the isotropic Gaussian prior on each
        weight.  Larger σ → more diffuse prior → looser bound.
        Use σ ≈ init_std (Xavier or He init scale) for a tight bound.

    Hypothesis H-C2
    ---------------
    At convergence: pac_bayes_bound ≤ train_CE + 1.0 nat.
    This confirms the model has genuinely generalised, not merely
    memorised the training distribution.

    Hypothesis H-D1 (Phase 4 experimental)
    ----------------------------------------
    Stronger weight_decay → smaller ||θ − θ₀||² → smaller KL →
    tighter bound.  Measurable by comparing bound values between runs
    with different WD schedules on the same held-out set.
    """

    def __init__(
        self,
        n_train: int,
        delta: float = 0.05,
        prior_sigma: float = 1.0,
    ) -> None:
        if n_train < 1:
            raise ValueError(f"n_train must be ≥ 1, got {n_train}")
        if not (0.0 < delta < 1.0):
            raise ValueError(f"delta must be in (0,1), got {delta}")
        if prior_sigma <= 0.0:
            raise ValueError(f"prior_sigma must be > 0, got {prior_sigma}")
        self._n = n_train
        self._delta = delta
        self._sigma2 = prior_sigma ** 2

    def compute(self, train_ce: float, kl_div: float) -> float:
        r"""Compute the bound.

        Parameters
        ----------
        train_ce : float
            Empirical training cross-entropy (nats).
        kl_div : float
            KL divergence D_KL(Q‖P) = ||θ − θ₀||² / (2σ²).
            Use ``kl_from_model`` or ``kl_from_params`` to compute.

        Returns
        -------
        float
            Upper bound on OOD loss.  Always ≥ train_ce and ≥ 0.
        """
        n = self._n
        complexity = (kl_div + math.log(2.0 * math.sqrt(n) / self._delta)) / (
            2.0 * n
        )
        bound = train_ce + math.sqrt(max(0.0, complexity))
        return max(0.0, bound)

    def kl_from_params(
        self,
        theta: torch.Tensor,
        theta0: torch.Tensor,
    ) -> float:
        """KL from flat parameter vectors θ and θ₀.

        D_KL(N(θ, σ²I) ‖ N(θ₀, σ²I)) = ||θ − θ₀||² / (2σ²)
        """
        diff_sq = float(((theta.float() - theta0.float()) ** 2).sum().item())
        return diff_sq / (2.0 * self._sigma2)

    def kl_from_model(
        self,
        model: nn.Module,
        prior_state: Dict[str, torch.Tensor],
    ) -> float:
        """KL from model parameters vs a prior state-dict snapshot.

        Parameters
        ----------
        model : nn.Module
            The current model.
        prior_state : dict
            State dict captured at initialisation (θ₀).

        Returns
        -------
        float
            KL divergence.
        """
        total_sq: float = 0.0
        for name, param in model.named_parameters():
            p0 = prior_state.get(name)
            if p0 is None:
                continue
            diff = param.float() - p0.float().to(param.device)
            total_sq += float((diff ** 2).sum().item())
        return total_sq / (2.0 * self._sigma2)


# ─────────────────────────────────────────────────────────────────────
# Phase 1E  ─  SharpnessProbe
# ─────────────────────────────────────────────────────────────────────

class SharpnessProbe:
    r"""Empirical sharpness via random additive perturbations.

    .. math::

        \text{sharpness} = \frac{1}{K} \sum_{k=1}^{K}
            \bigl[L(\theta + \varepsilon_k) - L(\theta)\bigr]

    where :math:`\varepsilon_k \sim \mathcal{N}(0, \rho^2 I)`.

    A flatter minimum (lower sharpness) generalises better.  SAM /
    gradient clipping both reduce sharpness; this probe makes it
    measurable.

    Parameters
    ----------
    rho : float
        Perturbation magnitude (std of Gaussian ε).
    n_samples : int
        Number of Monte-Carlo samples (default 4).
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        rho: float = 0.05,
        n_samples: int = 4,
        seed: int = 0,
    ) -> None:
        self._rho = rho
        self._n = n_samples
        self._seed = seed

    @torch.no_grad()
    def measure(
        self,
        harness,
        ids: torch.Tensor,
        targets: torch.Tensor,
        base_loss: float,
    ) -> float:
        """Measure sharpness = mean loss increase under Gaussian noise.

        Parameters
        ----------
        harness : BRIANHarness
            The harness (provides ``compute_loss``).
        ids, targets : Tensor
            Input / target batch.
        base_loss : float
            L(θ) already computed (avoids a redundant forward).

        Returns
        -------
        float
            Mean sharpness (nats).  Non-negative up to float noise.
        """
        rng = torch.Generator(device=ids.device)
        rng.manual_seed(self._seed)

        # Collect parameter tensors
        params = [p for p in harness.parameters() if p.requires_grad]

        total_gap: float = 0.0
        for _ in range(self._n):
            # Draw perturbations and save them so we can restore exactly.
            perturbations = [
                torch.empty_like(p).normal_(mean=0.0, std=self._rho,
                                            generator=rng)
                for p in params
            ]
            # Add ε
            with torch.no_grad():
                for p, eps in zip(params, perturbations):
                    p.data.add_(eps)
            # Evaluate perturbed loss
            with torch.no_grad():
                perturbed = float(
                    harness.compute_loss(ids, targets).detach().item()
                )
            total_gap += perturbed - base_loss
            # Restore parameters by subtracting the SAME ε
            with torch.no_grad():
                for p, eps in zip(params, perturbations):
                    p.data.sub_(eps)

        return total_gap / max(1, self._n)


# ─────────────────────────────────────────────────────────────────────
# Phase 1F  ─  EffectiveRankProbe
# ─────────────────────────────────────────────────────────────────────

class EffectiveRankProbe:
    r"""Roy & Vetterli (2007) effective rank of a matrix.

    .. math::

        \operatorname{erank}(H) = \exp\!\Bigl(
            H\!\left(\frac{\sigma}{\|\sigma\|_1}\right)
        \Bigr)

    where H(·) is the Shannon entropy and σ are the singular values
    of H.  The result lies in [1, min(rows, cols)]:

    *  rank-1 matrix → erank ≈ 1
    *  full-rank matrix → erank ≈ min(rows, cols)

    Collapse detection
    ------------------
    Hidden states that collapse to a low-dimensional subspace (rank
    approaching 1) indicate representation collapse, often caused by
    over-strong isotropy regularisation or a too-small learning rate.
    Track erank of the last hidden layer per step.
    """

    @staticmethod
    def compute(H: torch.Tensor) -> float:
        """Effective rank of matrix H ∈ R^{m×n}.

        Parameters
        ----------
        H : Tensor, shape (m, n)
            Any 2-D real-valued matrix (hidden states, embedding matrix…).

        Returns
        -------
        float
            Effective rank ∈ [1, min(m, n)].
        """
        H_f = H.float()
        # Use truncated SVD for speed on large matrices.
        try:
            sv = torch.linalg.svdvals(H_f)
        except Exception:
            sv = torch.svd(H_f, compute_uv=False).S

        sv = sv[sv > 1e-10]
        if sv.numel() == 0:
            return 1.0

        sv_sum = sv.sum()
        if sv_sum < 1e-10:
            return 1.0

        # Normalised singular values = probability distribution
        p = sv / sv_sum
        # Shannon entropy of p (nats)
        log_p = torch.log(p.clamp(min=1e-10))
        entropy = float(-(p * log_p).sum().item())
        return float(math.exp(entropy))


# ─────────────────────────────────────────────────────────────────────
# Phase 1G  ─  SpectralPowerLawProbe   (NOVEL geometric invariant)
# ─────────────────────────────────────────────────────────────────────

class SpectralPowerLawProbe:
    r"""Power-law spectrum geometry of hidden-state activations.

    A *novel intrinsic invariant* of the representation manifold,
    computed from one SVD per step.  Probes whether the trunk has
    converged onto a **biological 1/f cortical signature** — the
    scale-free statistics found across mammalian cortex (He 2014,
    Voytek & Knight 2015) and predicted to be the optimal
    representational regime for hierarchical multi-scale tasks
    (Bahri et al. 2020, Bordelon et al. 2020).

    Mathematics
    -----------
    Given hidden-state matrix :math:`H \in \mathbb{R}^{N \times d}`,
    let :math:`\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_K > 0`
    be the top-:math:`K` significant singular values
    (:math:`K \le k_{\max}`).  Three quantities are returned:

    1. **Power-law exponent** :math:`\alpha`

       .. math::

           \log \sigma_i \;\approx\; \log C - \alpha \cdot \log(i),
           \qquad i = 1, \dots, K.

       Estimated by ordinary least squares on
       :math:`(\log i,\, \log \sigma_i)`.

    2. **Goodness-of-fit** :math:`R^2`

       .. math::

           R^2 \;=\; 1 \;-\;
           \frac{\sum_i (\log\sigma_i - \widehat{\log\sigma_i})^2}
                {\sum_i (\log\sigma_i - \overline{\log\sigma})^2}
           \;\in\; [0, 1].

       :math:`R^2 \to 1` iff the spectrum is genuinely scale-free.

    3. **Participation ratio** :math:`D_{\mathrm{PR}}`
       (Wegner 1980; Edwards–Thouless 1972):

       .. math::

           D_{\mathrm{PR}} \;=\;
           \frac{\bigl(\sum_i \sigma_i^2\bigr)^{2}}{\sum_i \sigma_i^4}
           \;\in\; [1, K].

       Counts the *effective* number of variance-carrying directions
       using the :math:`L^2/L^4` ratio of squared singular values
       (eigenvalues of the Gram matrix).  Complements
       :class:`EffectiveRankProbe` (Shannon entropy of
       :math:`\sigma_i / \sum \sigma_j`) — :math:`D_{\mathrm{PR}}` is
       more sensitive to dominant modes; ``erank`` weights the tail
       more heavily.

    Provable invariance (intrinsic geometric quantities)
    ----------------------------------------------------
    Let :math:`Q \in O(d)` be any orthogonal matrix and
    :math:`c > 0` any positive scalar.  Then for
    :math:`H' = c \cdot H \cdot Q`:

      * :math:`\sigma_i(H') = c \cdot \sigma_i(H)`  (singular values scale)
      * Hence :math:`\log \sigma_i(H') = \log c + \log \sigma_i(H)`
        — a *vertical shift* in the log-log plot, leaving the
        **slope** :math:`\alpha` and **goodness** :math:`R^2`
        invariant.
      * And :math:`\sigma_i^2(H') = c^2 \sigma_i^2(H)` makes
        :math:`D_{\mathrm{PR}}(H') =
        (c^2 \sum \sigma_i^2)^2 / (c^4 \sum \sigma_i^4)
        = D_{\mathrm{PR}}(H)`.

    All three metrics are therefore *intrinsic* — they describe the
    geometry of the representation manifold itself, independent of
    any rotation, basis choice, or overall scaling.

    Interpretation legend
    ---------------------
    ====================  ======================================
    Regime                Geometric meaning
    --------------------  --------------------------------------
    α ≈ 1.0, R² > 0.9     **1/f cortical signature**  (target!)
    α ≈ 2.0, R² > 0.9     Brownian / random-walk-like
    α ≳ 3.0, R² > 0.9     Bottleneck / compressive collapse
    α ≲ 0.5               Flat / white-noise / uniform spectrum
    R² < 0.5              Bumpy spectrum — not scale-free
    D_PR ≪ K              Concentrated in few dominant modes
    D_PR ≈ K              Uniformly distributed variance
    ====================  ======================================

    References
    ----------
    He BJ (2014). *Scale-free brain activity: past, present,
        and future.*  Trends Cogn Sci 18(9):480–487.
    Voytek B, Knight RT (2015). *Dynamic network communication as
        a unifying neural basis for cognition.*
        Biol Psychiatry 77(12):1089–1097.
    Bahri Y et al. (2020). *Statistical mechanics of deep
        learning.*  Annu Rev Cond Matt Phys 11:501–528.
    Bordelon B, Canatar A, Pehlevan C (2020). *Spectrum dependent
        learning curves in kernel regression and wide neural
        networks.*  ICML 2020.
    Wegner F (1980). *Inverse participation ratio in 2+ε
        dimensions.*  Z Phys B Cond Matt 36(3):209–214.
    Edwards JT, Thouless DJ (1972). *Numerical studies of
        localization in disordered systems.*  J Phys C 5(8):807.
    """

    @staticmethod
    def compute(H: torch.Tensor, k_max: int = 64) -> Dict[str, float]:
        """Compute (α, R², D_PR) from hidden-state matrix H.

        Parameters
        ----------
        H : Tensor, shape (N, d)
            Hidden-state matrix.  ``N`` is the flattened batch×time
            dimension, ``d`` is the feature dimension.
        k_max : int, default 64
            Maximum number of leading singular values to include in
            the power-law fit.  Caps SVD cost at O(k_max · d) and
            avoids polluting the tail with floating-point noise.

        Returns
        -------
        dict with three keys:
            ``"alpha"`` : float — power-law exponent (≥ 0 in nominal use)
            ``"r2"``    : float ∈ [0, 1] — coefficient of determination
            ``"d_pr"``  : float ∈ [1, K] — participation ratio

        Degenerate inputs
        -----------------
        * Empty / all-zero ``H``      → (0.0, 0.0, 1.0)
        * Fewer than 3 nonzero SVs    → (0.0, 0.0, max(1.0, nnz))
        * Zero log-x variance         → (0.0, 0.0, D_PR)
        """
        H_f = H.float()
        try:
            sv = torch.linalg.svdvals(H_f)
        except Exception:
            sv = torch.svd(H_f, compute_uv=False).S

        if sv.numel() == 0:
            return {"alpha": 0.0, "r2": 0.0, "d_pr": 1.0}

        # Relative-threshold filter: keep σ_i above floating-point noise
        sv_max = sv.max()
        if float(sv_max.item()) < 1e-30:
            return {"alpha": 0.0, "r2": 0.0, "d_pr": 1.0}
        sv = sv[sv > 1e-8 * sv_max]
        nnz = int(sv.numel())

        # Participation ratio is well-defined for any nnz ≥ 1
        s2 = sv.pow(2)
        s4 = sv.pow(4)
        sum_s2 = float(s2.sum().item())
        sum_s4 = float(s4.sum().item())
        d_pr = (sum_s2 * sum_s2) / max(sum_s4, 1e-30)

        # Power-law fit requires ≥ 3 distinct points
        if nnz < 3:
            return {"alpha": 0.0, "r2": 0.0, "d_pr": d_pr}

        K = min(nnz, int(k_max))
        sv_k = sv[:K]

        # log-log regression: log σ_i = log C - α · log(i+1)
        rank = torch.arange(1, K + 1, dtype=torch.float32, device=sv.device)
        x = torch.log(rank)
        y = torch.log(sv_k.float().clamp(min=1e-30))

        x_mean = x.mean()
        y_mean = y.mean()
        x_c = x - x_mean
        y_c = y - y_mean
        var_x = float((x_c * x_c).sum().item())
        if var_x < 1e-20:
            return {"alpha": 0.0, "r2": 0.0, "d_pr": d_pr}

        cov_xy = float((x_c * y_c).sum().item())
        slope = cov_xy / var_x
        alpha = float(-slope)  # σ decreases → slope < 0 → α > 0

        # R² = 1 - SS_res / SS_tot
        y_pred = slope * x_c  # centred prediction
        ss_res = float(((y_c - y_pred) ** 2).sum().item())
        ss_tot = float((y_c * y_c).sum().item())
        if ss_tot < 1e-30:
            # All log σ_i equal → flat spectrum → power-law is degenerate
            return {"alpha": 0.0, "r2": 0.0, "d_pr": d_pr}
        r2 = 1.0 - (ss_res / ss_tot)
        r2 = max(0.0, min(1.0, r2))  # clamp to [0, 1]

        return {"alpha": alpha, "r2": r2, "d_pr": d_pr}


# ─────────────────────────────────────────────────────────────────────
# TrunkOptMonitor  ─  thin aggregator plugged into BRIANHarness
# ─────────────────────────────────────────────────────────────────────

class TrunkOptMonitor:
    """Aggregates all TRUNK-OPT probes; wired into BRIANHarness.

    BRIANHarness calls two hooks per step:

    1. ``on_compute_loss(harness, logits, h_last)`` — after forward,
       before backward.  Records effective_rank, bits_per_param.

    2. ``on_train_step_post_backward(harness)`` — after backward + clip,
       before optimizer.step().  Records grad_budget, layer norms.

    Attach via::

        from neuroslm.emergent.trunk_opt import TrunkOptMonitor
        harness.attach_trunk_opt_monitor(TrunkOptMonitor())

    Metrics written to ``harness._metrics`` use the prefix
    ``trunk_opt_*``.
    """

    def __init__(
        self,
        n_train: int = 1_000_000,
        pac_delta: float = 0.05,
        prior_sigma: float = 0.02,
    ) -> None:
        self._grad_tracker = GradientBudgetTracker()
        self._layer_probe   = LayerGradientProbe()
        self._rank_probe     = EffectiveRankProbe()
        self._power_law      = SpectralPowerLawProbe()
        self._pac_bound: Optional[PACBayesBound] = None
        self._prior_state: Optional[Dict[str, torch.Tensor]] = None
        self._n_train = n_train
        self._pac_delta = pac_delta
        self._prior_sigma = prior_sigma
        self._bpp_meter: Optional[BitsPerParamMeter] = None
        # Stash of the LM-only gradient norm (set inside compute_loss
        # before aux losses are added, if the harness supports it).
        self._lm_grad_norm_snapshot: float = 0.0

    # ── Initialisation hooks ─────────────────────────────────────────

    def init_pac_snapshot(self, model: nn.Module) -> None:
        """Snapshot the model weights as the PAC-Bayes prior (θ₀).

        Call once after ``harness.build_optimizer()`` and before the
        first ``train_step``.
        """
        self._prior_state = {
            k: v.detach().cpu().clone()
            for k, v in model.named_parameters()
        }
        self._pac_bound = PACBayesBound(
            n_train=self._n_train,
            delta=self._pac_delta,
            prior_sigma=self._prior_sigma,
        )

    def init_bpp_meter(self, vocab_size: int, n_trainable: int) -> None:
        """Create the BitsPerParamMeter once vocab + param count are known."""
        self._bpp_meter = BitsPerParamMeter(vocab_size, n_trainable)

    # ── Per-step hooks ───────────────────────────────────────────────

    def on_compute_loss(
        self,
        harness,
        logits: torch.Tensor,
        h_last: Optional[torch.Tensor],
    ) -> None:
        """Called inside compute_loss after forward, before backward."""
        # Effective rank of the last hidden state
        if h_last is not None:
            B, T, D = h_last.shape
            H2 = h_last.detach().reshape(B * T, D)
            rank = self._rank_probe.compute(H2)
            harness._metrics["trunk_opt_effective_rank"] = rank

            # Spectral power-law geometry — novel intrinsic invariant
            # (α, R², D_PR).  Re-uses the same hidden-state SVD path.
            pl = self._power_law.compute(H2)
            harness._metrics["trunk_opt_power_alpha"] = pl["alpha"]
            harness._metrics["trunk_opt_power_r2"]    = pl["r2"]
            harness._metrics["trunk_opt_dpr"]         = pl["d_pr"]

        # Bits per param
        if self._bpp_meter is not None:
            ce = harness._metrics.get("lm_loss",
                                      float(logits.new_tensor(0.0).item()))
            bpp = self._bpp_meter.compute(ce)
            harness._metrics["trunk_opt_bits_per_param"] = bpp

    def on_train_step_post_backward(
        self,
        harness,
        lm_grad_norm: float,
    ) -> None:
        """Called after backward, before clip+step.

        Parameters
        ----------
        harness : BRIANHarness
        lm_grad_norm : float
            The ||∇L_LM|| captured just after the LM-only backward.
            The harness stores this as ``_lm_grad_norm_snapshot``.
        """
        if harness.language_model is None:
            return

        # Total gradient norm across all model parameters post-backward.
        total_norm = self._grad_tracker.total_grad_norm(harness.language_model)

        # Gradient budget: ideally ||∇L_LM|| / ||∇L_total||.
        # A true LM-only backward would require a second forward/backward
        # pass (expensive). Instead we use a loss-space proxy:
        #   budget_proxy = L_LM / L_total  ∈ (0, 1]
        # When aux losses are zero (activation_step gate), this equals 1.0
        # by construction. After aux activate it tracks the LM fraction of
        # the joint objective, which is directionally correct.
        lm_v   = getattr(harness, "_last_lm_loss_value", 0.0)
        tot_v  = getattr(harness, "_last_total_loss_value", lm_v)
        if lm_grad_norm > 0.0:
            # True gradient measurement available (future: optional 2nd bwd)
            budget = self._grad_tracker.budget(lm_grad_norm, total_norm)
        else:
            # Fall back to loss-space proxy (denominator ≥ numerator always)
            budget = lm_v / max(tot_v, lm_v, 1e-8)
        harness._metrics["trunk_opt_grad_budget"] = budget
        harness._metrics["trunk_opt_total_grad_norm"] = total_norm

        # Per-layer gradient norms + uniformity ratio
        if hasattr(harness.language_model, "blocks"):
            layer_norms = self._layer_probe.compute(
                harness.language_model.blocks
            )
            ratio = self._layer_probe.uniformity_ratio(layer_norms)
            harness._metrics["trunk_opt_layer_uniformity"] = ratio

        # PAC-Bayes bound (only if initialised)
        if self._pac_bound is not None and self._prior_state is not None:
            train_ce = harness._metrics.get("lm_loss", 0.0)
            kl = self._pac_bound.kl_from_model(
                harness.language_model, self._prior_state
            )
            bound = self._pac_bound.compute(train_ce, kl)
            harness._metrics["trunk_opt_pac_bayes_kl"]    = kl
            harness._metrics["trunk_opt_pac_bayes_bound"] = bound

    # ── Backfill for test compatibility ──────────────────────────────

    def on_compute_loss_simple(
        self,
        harness,
        ce: float,
        h_last: Optional[torch.Tensor],
    ) -> None:
        """Same as on_compute_loss but accepts plain-float CE.

        Used in unit tests where the full logit tensor is not available.
        """
        if h_last is not None:
            B, T, D = h_last.shape
            H2 = h_last.detach().reshape(B * T, D)
            rank = self._rank_probe.compute(H2)
            harness._metrics["trunk_opt_effective_rank"] = rank

            # Spectral power-law geometry — mirrors on_compute_loss
            pl = self._power_law.compute(H2)
            harness._metrics["trunk_opt_power_alpha"] = pl["alpha"]
            harness._metrics["trunk_opt_power_r2"]    = pl["r2"]
            harness._metrics["trunk_opt_dpr"]         = pl["d_pr"]

        if self._bpp_meter is not None:
            bpp = self._bpp_meter.compute(ce)
            harness._metrics["trunk_opt_bits_per_param"] = bpp
