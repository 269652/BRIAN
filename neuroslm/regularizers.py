# -*- coding: utf-8 -*-
"""Five OOD-generalization interventions, math-aligned with
`architectures/rcc_bowtie/lib/regularizers.neuro`.

Each module is a self-contained `nn.Module`. The `RegularizationController`
composes them and is wired into `BRIANHarness.compute_loss`. All five are
no-ops when `enabled=false` so the controller is safe to leave wired in
production builds.

Symbol map (matches the canonical equations in regularizers.neuro):
    h            — hidden states, shape (B, T, d) or (B*T, d)
    lm_logits    — final logits from the LM head, shape (B, T, V)
    per_sample_ce— per-sequence cross-entropy losses, shape (B,)
    domain_labels— per-sequence source id, shape (B,) ∈ {0=text, 1=chat}
"""
from __future__ import annotations
import math
import warnings
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroslm.dsl.regularization import (
    DARConfig, PCCConfig, IsotropyConfig, CMDConfig, AdaptiveMixtureConfig,
    RegularizationConfig,
)


# ══════════════════════════════════════════════════════════════════════
# Gradient Reversal Layer (Ganin & Lempitsky 2015)
# ══════════════════════════════════════════════════════════════════════

class GradientReversal(torch.autograd.Function):
    """Identity in the forward, flipped+scaled gradient in the backward.

        forward:   y = x
        backward:  dx = -alpha · dy

    Used by DARReweighter so that the discriminator drives the
    representation toward domain-invariance while still being trained.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


# ══════════════════════════════════════════════════════════════════════
# Intervention A — Distributional Adversarial Reweighting
# ══════════════════════════════════════════════════════════════════════

class DARReweighter(nn.Module):
    """GRL + domain discriminator + per-sample CE reweighting.

    Math (lib/regularizers.neuro :: dar_discriminator, dar_loss):
        d_logits = GRL_α(pool(h)) · W_d + b_d
        L_disc   = BCE(d_logits, domain_label)
        minority_mask_i = 1 if domain_i is minority in this batch else 0
        w_i      = exp(λ · L_ce_i · minority_mask_i)
        weighted_ce = mean( w_i · L_ce_i ) / mean(w_i)    # self-normalised
        total_aux  = L_disc       (weighted_ce replaces the unweighted CE
                                    on the caller's main-loss path)
    """
    def __init__(self, cfg: DARConfig, d_model: int):
        super().__init__()
        self.cfg = cfg
        self.discriminator = nn.Sequential(
            nn.Linear(d_model, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, 1),
        )
        # Zero-init the final layer so untrained DAR contributes no aux
        # at step 0 (the GRL still flows zero gradient until the
        # discriminator starts predicting).
        nn.init.zeros_(self.discriminator[-1].weight)
        nn.init.zeros_(self.discriminator[-1].bias)
        self._warned_no_labels = False

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        """(B, T, d) → (B, d) via mean pooling."""
        if h.dim() == 3:
            return h.mean(dim=1)
        return h

    def _minority_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """For each sample, 1.0 if its class is in the minority this batch."""
        with torch.no_grad():
            counts = torch.bincount(labels, minlength=2).float()
            minority_class = int(counts.argmin().item())
            return (labels == minority_class).float()

    def forward(self, h: torch.Tensor, per_sample_ce: torch.Tensor,
                domain_labels: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = per_sample_ce.device
        zero = torch.zeros((), device=device)
        if not self.cfg.enabled:
            return {"weighted_ce": per_sample_ce.mean(),
                    "disc_loss": zero, "total_aux": zero}
        if domain_labels is None:
            if not self._warned_no_labels:
                warnings.warn(
                    "DARReweighter enabled but harness.domain_id_fn is None; "
                    "DAR is a no-op until labels are wired through the "
                    "data loader. See docs/technical_report.md §3 A.",
                    RuntimeWarning, stacklevel=2)
                self._warned_no_labels = True
            return {"weighted_ce": per_sample_ce.mean(),
                    "disc_loss": zero, "total_aux": zero}

        # Discriminator path (GRL → MLP → BCE)
        pooled = self._pool(h)
        d_logits = self.discriminator(
            GradientReversal.apply(pooled, self.cfg.grl_alpha)
        ).squeeze(-1)
        labels_f = domain_labels.float().to(d_logits.device)
        disc_loss = F.binary_cross_entropy_with_logits(d_logits, labels_f)

        # Reweighting path. Detached CE inside the weight (we want the
        # weights to *select* samples, not back-prop through the weighting
        # operation — that would create a runaway gain loop).
        minority = self._minority_mask(domain_labels.long())
        with torch.no_grad():
            log_w = self.cfg.lam * per_sample_ce.detach() * minority
            # Numerical guard: clamp before exp
            log_w = log_w.clamp(min=-20.0, max=20.0)
            w = log_w.exp()
            w = w / w.mean().clamp(min=1e-8)   # self-normalise → mean(w)=1
        weighted_ce = (w * per_sample_ce).mean()

        return {"weighted_ce": weighted_ce,
                "disc_loss": disc_loss,
                "total_aux": disc_loss}


# ══════════════════════════════════════════════════════════════════════
# Intervention B — Predictive Contrastive Coding (replaces PCT)
# ══════════════════════════════════════════════════════════════════════

class PCCLoss(nn.Module):
    """InfoNCE between (h_t, h_{t+k}) with cross-document negatives.

    Math (lib/regularizers.neuro :: pcc_loss):
        z_t       = proj(h_t)
        z_pos     = proj(h_{t+k})
        z_neg_j   ~ buffer of past z's from other documents
        L_pcc     = -log( exp(<z_t, z_pos>/τ)
                          / Σ_{neg} exp(<z_t, z_neg>/τ) )

    The negatives buffer is a circular tensor of size `n_negatives × d`,
    updated each forward with *detached* z's. This implements the
    "cross-document" property cheaply: yesterday's batch supplies
    today's negatives.
    """
    def __init__(self, cfg: PCCConfig, d_model: int):
        super().__init__()
        self.cfg = cfg
        self.k = max(1, cfg.k)
        self.tau = max(1e-6, cfg.tau)
        self.n_negatives = max(1, cfg.n_negatives)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        # Init at identity so the first forward isn't pathologically far
        # from a vanilla cosine of h.
        nn.init.eye_(self.proj.weight)
        # Circular buffer of past anchors (detached, normalised)
        self.register_buffer(
            "neg_buf", torch.zeros(self.n_negatives, d_model))
        self.register_buffer(
            "neg_buf_n_written", torch.zeros((), dtype=torch.long))

    def _push_negatives(self, z: torch.Tensor) -> None:
        """Write `z` rows into the circular buffer (detached)."""
        with torch.no_grad():
            flat = z.detach().reshape(-1, z.shape[-1])
            n = flat.shape[0]
            if n == 0:
                return
            start = int(self.neg_buf_n_written.item() % self.n_negatives)
            for i in range(n):
                self.neg_buf[(start + i) % self.n_negatives] = flat[i]
            self.neg_buf_n_written += n

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if not self.cfg.enabled:
            return h.new_zeros(())
        if h.dim() != 3 or h.shape[1] <= self.k:
            return h.new_zeros(())

        B, T, d = h.shape
        # Project + L2-normalise so the inner product equals cosine.
        z = F.normalize(self.proj(h), dim=-1)              # (B, T, d)
        anchor = z[:, :-self.k, :]                          # (B, T-k, d)
        positive = z[:, self.k:, :]                         # (B, T-k, d)
        anchor_flat = anchor.reshape(-1, d)                # (N, d)
        positive_flat = positive.reshape(-1, d)

        # Positives logits: dot product / τ
        pos_logit = (anchor_flat * positive_flat).sum(-1) / self.tau  # (N,)

        # Negatives logits: anchor · buffer^T / τ
        # Snapshot the buffer BEFORE updating it (the update writes via
        # detach() but `self.neg_buf` is still a buffer tensor that
        # autograd would later see as mutated → "in-place modification"
        # error). Clone to a graph-detached tensor that's safe to read.
        neg_view = self.neg_buf.detach().clone()
        neg_logits = (anchor_flat @ neg_view.t()) / self.tau          # (N, M)

        # Stable log-softmax: log Z = logsumexp(concat(pos, neg))
        # Concatenate along the last dim
        all_logits = torch.cat([pos_logit.unsqueeze(-1), neg_logits], dim=-1)
        log_z = torch.logsumexp(all_logits, dim=-1)
        loss = -(pos_logit - log_z).mean()

        # Update buffer with the current anchor's z (cross-doc negatives
        # for *future* batches). Detached so it doesn't grow the graph.
        self._push_negatives(anchor_flat)
        return loss


# ══════════════════════════════════════════════════════════════════════
# Intervention C — Isotropy whitening
# ══════════════════════════════════════════════════════════════════════

class IsotropyLoss(nn.Module):
    """Online whitening loss that pushes Gram(H) toward I.

    Math (lib/regularizers.neuro :: isotropy_loss_frobenius):
        G        = H_bufᵀ H_buf / N
        L_iso    = ||G - I||_F^2 / d²        # default ("frobenius")
                 = -log|det(G)|              # alternative ("log_det")

    The buffer is a rolling window of the last `cfg.buffer` token
    embeddings, detached. Loss is computed on the **current batch** mixed
    with the buffered history, so gradient flows back into `h`.
    """
    def __init__(self, cfg: IsotropyConfig, d_model: int):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.register_buffer(
            "buf", torch.zeros(max(1, cfg.buffer), d_model))
        self.register_buffer(
            "buf_n_written", torch.zeros((), dtype=torch.long))

    def get_buffer_view(self) -> torch.Tensor:
        """Return the populated portion of the buffer (for tests)."""
        n = int(min(self.buf_n_written.item(), self.buf.shape[0]))
        return self.buf[:n] if n > 0 else self.buf[:1]

    def _push(self, flat: torch.Tensor) -> None:
        with torch.no_grad():
            n = flat.shape[0]
            B = self.buf.shape[0]
            if n == 0:
                return
            start = int(self.buf_n_written.item() % B)
            # Wrap-around write
            if n >= B:
                self.buf.copy_(flat[-B:])
                self.buf_n_written += n
                return
            end = start + n
            if end <= B:
                self.buf[start:end] = flat
            else:
                first = B - start
                self.buf[start:] = flat[:first]
                self.buf[:n - first] = flat[first:]
            self.buf_n_written += n

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if not self.cfg.enabled:
            return h.new_zeros(())
        flat = h.reshape(-1, h.shape[-1])
        # Push the detached version into the buffer for future steps.
        self._push(flat.detach())
        # Compute Gram on the *graph-attached* current batch only.
        # Including the buffer here would freeze the loss (zeros init
        # produces a singular Gram → garbage gradient).
        N = flat.shape[0]
        G = (flat.t() @ flat) / max(1, N)
        d = self.d_model
        I = torch.eye(d, device=G.device, dtype=G.dtype)
        if self.cfg.distance == "log_det":
            # Use slogdet for numerical safety; regularise with εI to
            # keep the determinant strictly positive on rank-deficient
            # batches (e.g. N < d).
            eps = 1e-4
            sign, log_abs = torch.slogdet(G + eps * I)
            # If the matrix flipped sign (rank-deficient), fall back to
            # Frobenius — we never want a negative loss.
            if sign.item() <= 0:
                loss = ((G - I) ** 2).sum() / (d * d)
            else:
                loss = -log_abs
        else:  # "frobenius" (default)
            loss = ((G - I) ** 2).sum() / (d * d)
        return self.cfg.weight * loss


# ══════════════════════════════════════════════════════════════════════
# Intervention D — Cross-Module Disagreement
# ══════════════════════════════════════════════════════════════════════

class CMDLoss(nn.Module):
    """Penalize divergence between LM read-out and a second 'narrative' head.

    Math (lib/regularizers.neuro :: cmd_loss_jsd):
        p_lm   = softmax(lm_logits)
        p_narr = softmax(head(h))
        m      = (p_lm + p_narr) / 2
        JSD    = 0.5·KL(p_lm‖m) + 0.5·KL(p_narr‖m)
        L_cmd  = weight · JSD

    The narrative head is a separate linear projection from `h` to vocab.
    By construction it sees the same trunk representation but produces an
    *independent* logit — any disagreement is therefore a property the
    LM head specifically learns. Penalising it pushes both heads toward
    the shared (and hopefully OOD-robust) signal.
    """
    def __init__(self, cfg: CMDConfig, d_model: int, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    def _kl(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """KL(p || q) with both as already-normalised probabilities."""
        eps = 1e-12
        return (p * (p.add(eps).log() - q.add(eps).log())).sum(-1)

    def forward(self, h: torch.Tensor, lm_logits: torch.Tensor) -> torch.Tensor:
        if not self.cfg.enabled:
            return h.new_zeros(())
        narr_logits = self.head(h)
        # Match shape; allow the caller to pass (B,T,V) or (N,V)
        if narr_logits.shape != lm_logits.shape:
            narr_logits = narr_logits.reshape(lm_logits.shape)

        if self.cfg.divergence == "l1":
            p_lm = F.softmax(lm_logits, dim=-1)
            p_narr = F.softmax(narr_logits, dim=-1)
            loss = (p_lm - p_narr).abs().mean()
        elif self.cfg.divergence == "kl_sym":
            p_lm = F.softmax(lm_logits, dim=-1)
            p_narr = F.softmax(narr_logits, dim=-1)
            loss = 0.5 * (self._kl(p_lm, p_narr).mean()
                           + self._kl(p_narr, p_lm).mean())
        else:  # "jsd" (default)
            p_lm = F.softmax(lm_logits, dim=-1)
            p_narr = F.softmax(narr_logits, dim=-1)
            m = 0.5 * (p_lm + p_narr)
            loss = 0.5 * (self._kl(p_lm, m).mean()
                           + self._kl(p_narr, m).mean())
        return self.cfg.weight * loss


# ══════════════════════════════════════════════════════════════════════
# Intervention E — Adaptive mixture controller
# ══════════════════════════════════════════════════════════════════════

class AdaptiveMixtureController(nn.Module):
    """Closed-loop controller for the dataloader's chat_ratio.

    Math (lib/regularizers.neuro :: adaptive_mixture_update):
        H_t      = -mean( Σ_v p(v) log p(v) )    over a sampled set of tokens
        chat_t+1 = clip( chat_t · (H_t / H_target)^γ, [r_min, r_max] )

    The harness calls `observe_logits` every step. The first
    `probe_interval` observations are accumulated; on every Nth step the
    smoothed entropy fires `_update_ratio`. The dataloader reads the
    current value via the harness's `RatioRef` injected at construction.
    """
    def __init__(self, cfg: AdaptiveMixtureConfig, initial_ratio: float):
        super().__init__()
        self.cfg = cfg
        self._ratio = float(initial_ratio)
        self._step = 0
        self._ent_sum = 0.0
        self._ent_n = 0

    def ratio(self) -> float:
        return self._ratio

    def observe_logits(self, logits: torch.Tensor) -> None:
        """Accumulate entropy; fire the controller every probe_interval."""
        if not self.cfg.enabled:
            return
        with torch.no_grad():
            # Sample to avoid OOM at (B*T*V) scale.
            flat = logits.reshape(-1, logits.shape[-1])
            n_sample = min(256, flat.shape[0])
            idx = torch.randint(0, flat.shape[0], (n_sample,),
                                 device=flat.device)
            sampled = flat.index_select(0, idx)
            log_p = F.log_softmax(sampled, dim=-1)
            ent = -(log_p.exp() * log_p).sum(-1).mean()
            self._ent_sum += float(ent.item())
            self._ent_n += 1
        self._step += 1
        if self._step % max(1, self.cfg.probe_interval) == 0:
            mean_H = self._ent_sum / max(1, self._ent_n)
            self._update_ratio(mean_H)
            self._ent_sum, self._ent_n = 0.0, 0

    def _update_ratio(self, H_t: float) -> None:
        gain = (H_t / max(1e-6, self.cfg.target_entropy)) ** self.cfg.gamma
        new_ratio = self._ratio * gain
        new_ratio = max(self.cfg.min_ratio,
                        min(self.cfg.max_ratio, new_ratio))
        self._ratio = float(new_ratio)


# ══════════════════════════════════════════════════════════════════════
# Top-level controller — composes the five interventions
# ══════════════════════════════════════════════════════════════════════

class RegularizationController(nn.Module):
    """Owns the five interventions and assembles their aux losses.

    Wired into `BRIANHarness` via `harness.reg_controller`. The harness
    calls `collect_aux(...)` inside `compute_loss` and adds the returned
    `total` to the final scalar loss before backward.

    **Warmup**: To prevent early-training instability (random hidden
    states make InfoNCE explode, Isotropy push toward identity
    prematurely, DAR's gradient reversal disrupt representations),
    a linear warmup multiplier ramps the aux loss contribution from
    0 → 1 over `cfg.warmup_steps`. The interventions still RUN every
    step (so internal state like PCC negatives buffer and AdaptiveMixture
    entropy probe accumulate correctly), but their contribution to the
    training loss is scaled.
    """
    def __init__(self, cfg: RegularizationConfig,
                 d_model: int, vocab_size: int,
                 initial_chat_ratio: float = 0.6):
        super().__init__()
        self.cfg = cfg
        self.dar = DARReweighter(cfg.dar, d_model)
        self.pcc = PCCLoss(cfg.pcc, d_model)
        self.isotropy = IsotropyLoss(cfg.isotropy, d_model)
        self.cmd = CMDLoss(cfg.cmd, d_model, vocab_size)
        self.adaptive_mixture = AdaptiveMixtureController(
            cfg.adaptive_mixture, initial_chat_ratio)
        # Step counter for warmup. Incremented every collect_aux call
        # (which is exactly once per optimizer step).
        self.register_buffer(
            "_reg_step", torch.zeros(1, dtype=torch.long), persistent=True)

    def warmup_multiplier(self) -> float:
        """Linear ramp 0 → 1 over cfg.warmup_steps. Returns 1.0 after."""
        w = int(self.cfg.warmup_steps)
        if w <= 0:
            return 1.0
        s = int(self._reg_step.item())
        return min(1.0, s / float(w))

    def collect_aux(self,
                    h: torch.Tensor,
                    lm_logits: torch.Tensor,
                    per_sample_ce: torch.Tensor,
                    domain_labels: Optional[torch.Tensor]
                    ) -> Dict[str, torch.Tensor]:
        """Compute every enabled aux loss; return per-key dict + sum.

        Returned keys: dar, pcc, isotropy, cmd, total, weighted_ce,
        warmup_mult.
        Disabled interventions contribute exact zeros.
        Warmup multiplier scales the aggregated `total` but per-key
        values reported reflect the SCALED contribution (so logs show
        what actually entered the gradient).
        """
        zero = lm_logits.new_zeros(())
        # DAR runs on h and CE; returns the *replacement* CE for the caller.
        dar_out = self.dar(h, per_sample_ce, domain_labels)
        pcc_loss = self.pcc(h) if self.cfg.pcc.enabled else zero
        iso_loss = self.isotropy(h) if self.cfg.isotropy.enabled else zero
        cmd_loss = self.cmd(h, lm_logits) if self.cfg.cmd.enabled else zero

        # Mixture controller observes the logits (no contribution to loss)
        self.adaptive_mixture.observe_logits(lm_logits)

        # Linear warmup: scale the contribution but keep internal state
        # updates (e.g. PCC buffer, AdaptiveMixture entropy probe) running
        # at full rate so they're warm when the multiplier hits 1.0.
        # Per-intervention `weight` knobs (cfg.dar.weight, cfg.pcc.weight)
        # gate the raw loss BEFORE warmup: this is the architectural fix
        # for "PCC InfoNCE saturates at log(N) and drowns the LM signal"
        # — defaults are 0.1 (CPC literature standard, Oord et al. 2018).
        # Isotropy and CMD already apply their `weight` internally.
        mult = self.warmup_multiplier()
        scaled_dar = dar_out["total_aux"] * (mult * self.cfg.dar.weight)
        scaled_pcc = pcc_loss * (mult * self.cfg.pcc.weight)
        scaled_iso = iso_loss * mult
        scaled_cmd = cmd_loss * mult

        total = scaled_pcc + scaled_iso + scaled_cmd + scaled_dar

        # Advance the step counter (one tick per optimizer call).
        self._reg_step += 1

        return {
            "dar": scaled_dar,
            "weighted_ce": dar_out["weighted_ce"],
            "pcc": scaled_pcc,
            "isotropy": scaled_iso,
            "cmd": scaled_cmd,
            "total": total,
            "warmup_mult": lm_logits.new_tensor(mult),
        }
