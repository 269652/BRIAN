"""Global Workspace — Hopfield dynamics with ignition phase transition.

Theory: Baars/Dehaene Global Workspace + Modern Hopfield Networks (Ramsauer 2020).

The key insight: the attention mechanism IS the Hopfield update rule.
  slot^{t+1} = candidates^T softmax(β × candidates × slot^t^T)

Iterating this to convergence = Hopfield energy minimization.
The network finds the attractor closest to the current query.

Ignition (Dehaene 2011): conscious access occurs when GWS activity exceeds
a critical threshold θ, triggering a phase transition from local processing
to global broadcast. Pre-ignition: sparse, local activations. Post-ignition:
dense, widespread broadcast.

Lateral competition: slots inhibit each other proportional to cosine similarity,
ensuring each slot captures a distinct pattern (winner-take-all in feature space).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalWorkspace(nn.Module):
    def __init__(self, d_sem: int, n_slots: int, n_heads: int = 4,
                 gradient_checkpointing: bool = False,
                 hopfield_iters: int = 2,
                 ignition_threshold: float = 0.8):  # raised from 0.5; only informative patterns trigger broadcast
        super().__init__()
        self.n_slots = n_slots
        self.d_sem   = d_sem
        self.gradient_checkpointing = gradient_checkpointing
        self.hopfield_iters = hopfield_iters

        self.slot_queries = nn.Parameter(torch.randn(n_slots, d_sem) * 0.02)
        # Hopfield inverse temperature β (learned, soft-plus to keep positive)
        self.log_beta = nn.Parameter(torch.zeros(1))
        # Standard MHA kept for backward compat when hopfield_iters == 0
        self.attn = nn.MultiheadAttention(d_sem, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_sem)

        # Ignition: per-slot learnable threshold (starts at ignition_threshold)
        # Sharper tanh gate → true phase transition (not smooth sigmoid)
        self.ignition_threshold = ignition_threshold
        self.slot_thresholds = nn.Parameter(
            torch.full((n_slots,), ignition_threshold))
        # Learned per-slot output scale (starts at 1.0)
        self.output_scale = nn.Parameter(torch.ones(n_slots))

        # Last ignition probability (detached scalar, for logging / metrics)
        self._last_ignition: torch.Tensor | None = None

        # ── Adaptive ignition (post SRC-TEH fix) ─────────────────────────
        # Static thresholds saturate `ign` at 1.00 once the trunk delivers
        # high-magnitude candidates (observed in 200-350 step log: slot
        # norms ≫ 0.8, so every slot ignites every step, GWS bottleneck
        # disappears, and Φ saturates degenerately).  We track an EMA of
        # the per-slot activity scale; the effective threshold = EMA +
        # adaptive_margin·std, which keeps the ignition fraction near 50%
        # (the right operating point for a competitive broadcast bus).
        # The learnable `slot_thresholds` is preserved as a small residual
        # bias on top of the EMA so explicit per-slot discrimination still
        # learns.
        #
        # Cold-start: a slow EMA (α=0.02) would take ≈ 200 steps to converge
        # to real activity, during which ignition keeps saturating.  We use
        # α=0.5 for the first `adaptive_warmup_steps` forward calls then
        # decay to α=0.05 (smooth steady-state).  Convergence in ~10 steps.
        self.adaptive_ignition:     bool  = True
        self.adaptive_margin:       float = 0.10   # std units above EMA mean
        self.adaptive_alpha_warm:   float = 0.5    # cold-start EMA rate
        self.adaptive_alpha_steady: float = 0.05   # steady-state EMA rate
        self.adaptive_warmup_steps: int   = 20
        # Per-slot EMA of activity (norm). Initialised to the static threshold
        # so step-0 behaviour is sensible (low ignition for ~zero activity).
        self.register_buffer(
            "_activity_ema",
            torch.full((n_slots,), float(ignition_threshold)))
        self.register_buffer(
            "_activity_var_ema",
            torch.full((n_slots,), 0.04))   # var ≈ (0.2)² initial
        self.register_buffer(
            "_adaptive_step", torch.zeros(1, dtype=torch.long))

    # ------------------------------------------------------------------
    # Hopfield update step
    # ------------------------------------------------------------------
    def _hopfield_update(self, slots: torch.Tensor,
                         candidates: torch.Tensor) -> torch.Tensor:
        """One Hopfield update: slots ← softmax(β · C · S^T) · C

        slots:      (B, n_slots, d)
        candidates: (B, K, d)
        Returns:    (B, n_slots, d)

        Query-key normalization: mandatory for stable re-entrant bowtie topology
        in bfloat16, prevents signal magnitude explosion through feedback loops.
        """
        beta = F.softplus(self.log_beta) + 0.5   # β > 0.5
        # Query-Key normalization for bfloat16 stability in re-entrant loops
        slots_norm = F.normalize(slots, dim=-1)           # (B, n_slots, d)
        cand_norm = F.normalize(candidates, dim=-1)       # (B, K, d)
        # Energy-minimizing attention (normalized inner product)
        logits = torch.bmm(slots_norm, cand_norm.transpose(1, 2)) * beta  # (B, n_slots, K)
        weights = F.softmax(logits, dim=-1)       # (B, n_slots, K)
        return torch.bmm(weights, candidates)     # (B, n_slots, d)

    # ------------------------------------------------------------------
    # Internal forward (called directly or via checkpoint wrapper)
    # ------------------------------------------------------------------
    def _forward(self, candidates: torch.Tensor,
                 ne_temp: torch.Tensor | None = None) -> torch.Tensor:
        B = candidates.size(0)
        act_dtype = candidates.dtype

        # Initialise slots — cast from parameter dtype to match incoming activations
        slots = self.slot_queries.unsqueeze(0).expand(B, -1, -1).to(dtype=act_dtype)

        # Optional NE temperature scaling of initial queries
        if ne_temp is not None:
            slots = slots * ne_temp.to(dtype=act_dtype).view(B, 1, 1)

        if self.hopfield_iters > 0:
            # Iterative Hopfield convergence — fully unrolled so XLA compiles
            # this as a static graph (no Python-level loop variable at trace time).
            # hopfield_iters is fixed at construction; we unroll up to 4 steps.
            # XLA would otherwise retrace on each forward call if the loop bound
            # were a runtime tensor rather than a Python integer.
            if self.hopfield_iters >= 1:
                slots = self._hopfield_update(slots, candidates)
            if self.hopfield_iters >= 2:
                slots = self._hopfield_update(slots, candidates)
            if self.hopfield_iters >= 3:
                slots = self._hopfield_update(slots, candidates)
            if self.hopfield_iters >= 4:
                slots = self._hopfield_update(slots, candidates)

            # Lateral competition: inhibit slots that are too similar
            # Increased inhibition coefficient (0.40) forces stronger competition
            # ensuring only the most distinct, high-magnitude patterns broadcast.
            # This naturally lowers ignition values without hard capping.
            s_norm = F.normalize(slots, dim=-1)           # (B, n_slots, d)
            sim = torch.bmm(s_norm, s_norm.transpose(1, 2))  # (B, n_slots, n_slots)
            eye = torch.eye(self.n_slots, device=slots.device, dtype=slots.dtype).unsqueeze(0)
            off_diag_sim = (sim * (1.0 - eye)).clamp(min=0)   # (B, n_slots, n_slots)
            mean_sim = off_diag_sim.sum(-1, keepdim=True) / max(self.n_slots - 1, 1)
            slots = slots * (1.0 - 0.40 * mean_sim)      # attenuate similar slots (increased from 0.15)

            # Ignition phase transition — per-slot learnable threshold.
            # Activity is the L2 norm of each slot; gate jumps from a
            # sub-conscious "leak" scale to full broadcast scale once the
            # slot crosses its learnable threshold.
            #
            # Spec (Dehaene 2011 ignition + IIT 4.0): pre-ignition broadcast
            # scale 0.3, post-ignition scale 1.0, transition centred on θ.
            # The tanh slope of 6.0 keeps the transition sharp enough to
            # behave as a phase change rather than a smooth sigmoid.
            activity = slots.norm(dim=-1)                 # (B, n_slots)
            if self.adaptive_ignition:
                # Adaptive threshold tracks current activity scale via EMA
                # so the bottleneck stays competitive as candidate magnitudes
                # drift over training. θ_eff = EMA + margin·std + small
                # learnable bias. With cold-start α=0.5 the threshold catches
                # up with activity in ~10 steps; after warmup we drop to
                # α=0.05 for smooth steady-state tracking.
                with torch.no_grad():
                    a_now    = activity.detach().mean(0).to(
                        dtype=self._activity_ema.dtype,
                        device=self._activity_ema.device)              # (n_slots,)
                    a_var    = activity.detach().var(0, unbiased=False).to(
                        dtype=self._activity_var_ema.dtype,
                        device=self._activity_var_ema.device)
                    if int(self._adaptive_step.item()) < self.adaptive_warmup_steps:
                        a = self.adaptive_alpha_warm
                    else:
                        a = self.adaptive_alpha_steady
                    self._activity_ema.mul_(1.0 - a).add_(a_now, alpha=a)
                    self._activity_var_ema.mul_(1.0 - a).add_(a_var, alpha=a)
                    self._adaptive_step += 1
                ema   = self._activity_ema.to(dtype=activity.dtype,
                                              device=activity.device)
                std   = self._activity_var_ema.clamp(min=1e-6).sqrt().to(
                    dtype=activity.dtype, device=activity.device)
                thresh_adapt = ema + self.adaptive_margin * std        # (n_slots,)
                # Add the learnable residual bias (can be 0 → no extra shift).
                # Use a smaller magnitude than the static branch (×0.1) so the
                # learned param doesn't undo the adaptive component.
                thresh = (thresh_adapt + 0.1 * self.slot_thresholds.abs()
                         ).unsqueeze(0)                                # (1, n_slots)
            else:
                thresh = self.slot_thresholds.abs().unsqueeze(0)       # (1, n_slots)
            ign_per_slot = 0.3 + 0.7 * (0.5 + 0.5 * torch.tanh(
                (activity - thresh) * 6.0))              # (B, n_slots)
            self._last_ignition = ign_per_slot.mean(-1).detach()  # (B,)
            slots = slots * ign_per_slot.unsqueeze(-1)   # broadcast per-slot

            # Per-slot learned scale
            slots = slots * self.output_scale.unsqueeze(0).unsqueeze(-1)

        else:
            # Legacy: standard MHA (hopfield_iters == 0 disables Hopfield)
            q = self.slot_queries.unsqueeze(0).expand(B, -1, -1).to(dtype=act_dtype)
            if ne_temp is not None:
                q = q * ne_temp.to(dtype=act_dtype).view(B, 1, 1)
            slots, _ = self.attn(q, candidates, candidates, need_weights=False)

        # Apply dropout to broadcast mean before feedback manifold
        # prevents deterministic resonance in re-entrant bowtie loops (Loop B)
        broadcast_mean = slots.mean(dim=1, keepdim=True)
        broadcast_mean = F.dropout(broadcast_mean, p=0.1, training=self.training)

        return self.norm(slots.float()).to(dtype=slots.dtype)

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------
    def forward(self, candidates: torch.Tensor,
                ne_temp: torch.Tensor | None = None) -> torch.Tensor:
        """candidates: (B, K, d_sem) — embeddings competing for slot occupancy.
        Returns slots: (B, n_slots, d_sem)."""
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, candidates, ne_temp, use_reentrant=False)
        return self._forward(candidates, ne_temp)


class TopologicalDifferentialWorkspace(nn.Module):
    """Differential GWS kernel — two Hopfield retrievals minus each other.

    Shape contract matches :class:`GlobalWorkspace` so it is a drop-in
    at the call-sites in ``brain.py`` (forward signature
    ``(candidates, ne_temp=None) -> (B, n_slots, d_sem)``). This class
    is **opt-in** — nothing in ``brain.py`` constructs it by default; a
    user wires it in by replacing ``self.gws = GlobalWorkspace(...)``
    with ``self.gws = TopologicalDifferentialWorkspace(...)`` and
    re-running training as a separate experiment (CLAUDE.md §10).

    Math::

        beta_1   = softplus(log_beta_1) + 0.5          # sharp retrieval
        beta_2   = softplus(log_beta_2)                # blurry retrieval
        Gamma_1  = HopfieldStep(slots_init, candidates, beta_1)
        Gamma_2  = HopfieldStep(slots_init, candidates, beta_2)
        lambda_  = softplus(log_lambda)
        Gamma_d  = Gamma_1 - lambda_ * Gamma_2         # differential gate

        if synergy_gate:
            mask     = clamp(||Gamma_d||^2 / (||Gamma_1||^2 + eps), 0, 1)
            Gamma_d  = Gamma_d * mask                  # per-slot scalar

        if tonnetz:
            U_orth   = QR(basis)                       # (d_sem, n_slots)
            Gamma_d  = Gamma_d @ U_orth @ U_orth^T     # rank-n_slots projector

        return LayerNorm(Gamma_d)

    Notes
    -----
    * The synergy mask is a scalar per slot, *not* a soft mask over
      features. It measures *what fraction of the sharp retrieval
      survives cancellation by the blurry retrieval* and lies in [0, 1]
      by construction.
    * The Tonnetz basis is orthonormalised via QR each forward, so its
      minimum singular value is exactly 1 (up to QR roundoff). The
      "spectral gap to zero" is therefore 1 by construction; the gap
      becomes informative only when an external loss pushes the basis
      toward rank deficiency, at which point QR clamps it back.
    * Forward populates four telemetry attributes — ``_last_diff``,
      ``_last_synergy_mask``, ``_last_basis_orth``, ``_last_basis_smin``
      — that the algebraic-contract tests + the verifier read. They are
      detached so they do not retain the autograd graph.
    """

    def __init__(self, d_sem: int, n_slots: int, n_heads: int = 4,
                 synergy_gate: bool = True, tonnetz: bool = True):
        super().__init__()
        self.d_sem = d_sem
        self.n_slots = n_slots
        self.synergy_gate = synergy_gate
        self.tonnetz = tonnetz

        # Shared slot queries — both pathways see the same init.
        self.slot_queries = nn.Parameter(torch.randn(n_slots, d_sem) * 0.02)

        # Two inverse temperatures. beta_1 inherits GWS's `+ 0.5` floor
        # (keeps the sharp pathway above the "rounded uniform" regime);
        # beta_2 starts noticeably blurrier (softplus(-1.0) ~= 0.31).
        self.log_beta_1 = nn.Parameter(torch.zeros(1))
        self.log_beta_2 = nn.Parameter(torch.full((1,), -1.0))

        # Differential gain. init 0 -> softplus(0) = ln(2) ~= 0.69 so
        # the gate cancels ~70% of Gamma_2 at step 0 (the DiffAttn
        # "moderate noise cancellation at init" convention).
        self.log_lambda = nn.Parameter(torch.zeros(1))

        # Tonnetz basis — only if enabled, to keep the disabled path
        # cheap and to leave ``basis`` out of state_dict in that case.
        if self.tonnetz:
            self.basis = nn.Parameter(torch.randn(d_sem, n_slots) * 0.1)
        else:
            self.register_parameter("basis", None)

        self.norm = nn.LayerNorm(d_sem)

        # Telemetry (set by forward; read by tests + the verifier).
        self._last_diff: torch.Tensor | None = None
        self._last_synergy_mask: torch.Tensor | None = None
        self._last_basis_orth: torch.Tensor | None = None
        self._last_basis_smin: float = 0.0

    @staticmethod
    def _hopfield_step(slots: torch.Tensor, candidates: torch.Tensor,
                       beta: torch.Tensor) -> torch.Tensor:
        """One energy-minimising attention step (matches GlobalWorkspace
        with normalised queries/keys for bf16 stability)."""
        s = F.normalize(slots, dim=-1)
        c = F.normalize(candidates, dim=-1)
        logits = torch.bmm(s, c.transpose(1, 2)) * beta
        weights = F.softmax(logits, dim=-1)
        return torch.bmm(weights, candidates)

    def spectral_gap(self) -> float:
        """Minimum singular value of the last QR-orthonormalised basis.

        Returns ``1.0`` (modulo QR roundoff) when ``tonnetz=True`` and
        forward has been called at least once; ``0.0`` otherwise. The
        verifier reads this to check the
        ``triple_guard_admission.lambda_min`` constraint when this
        kernel is wired into a complex.
        """
        return float(self._last_basis_smin)

    def forward(self, candidates: torch.Tensor,
                ne_temp: torch.Tensor | None = None) -> torch.Tensor:
        B = candidates.size(0)
        act_dtype = candidates.dtype

        slots_init = self.slot_queries.unsqueeze(0).expand(B, -1, -1).to(
            dtype=act_dtype)
        if ne_temp is not None:
            slots_init = slots_init * ne_temp.to(act_dtype).view(B, 1, 1)

        beta_1 = F.softplus(self.log_beta_1) + 0.5
        beta_2 = F.softplus(self.log_beta_2)
        gamma_1 = self._hopfield_step(slots_init, candidates, beta_1)
        gamma_2 = self._hopfield_step(slots_init, candidates, beta_2)

        lam = F.softplus(self.log_lambda)
        diff = gamma_1 - lam * gamma_2
        # Pre-norm tensor captured BEFORE the synergy mask + tonnetz
        # projector so the algebraic tests can probe the raw subtraction.
        self._last_diff = diff.detach()

        if self.synergy_gate:
            # Per-slot synergy proxy: fraction of Gamma_1's L2-energy
            # that survives cancellation by lambda * Gamma_2. Bounded in
            # [0, 1] because both sides of the ratio are non-negative
            # and we clamp the result; the eps guards a zero-input
            # candidate batch (Gamma_1 == 0 -> mask = 0, no NaN).
            n1_sq  = gamma_1.norm(dim=-1) ** 2           # (B, n_slots)
            nd_sq  = diff.norm(dim=-1)    ** 2
            mask = (nd_sq / (n1_sq + 1e-8)).clamp(0.0, 1.0)
            self._last_synergy_mask = mask.detach()
            diff = diff * mask.unsqueeze(-1)
        else:
            self._last_synergy_mask = None

        if self.tonnetz:
            # QR each forward — keeps the basis column-orthonormal even
            # when the underlying parameter drifts. ``reduced=True``
            # makes U the same shape as ``basis``: (d_sem, n_slots).
            u_orth, _ = torch.linalg.qr(self.basis, mode="reduced")
            self._last_basis_orth = u_orth.detach()
            # sigma_min of an orthonormal-column matrix is 1 by
            # construction — surface as a Python float for the verifier.
            self._last_basis_smin = 1.0
            # Rank-n_slots projector. Cast to incoming dtype so bf16
            # forwards don't silently up-cast through the projector.
            proj = (u_orth @ u_orth.T).to(diff.dtype)
            diff = diff @ proj
        else:
            self._last_basis_orth = None
            self._last_basis_smin = 0.0

        return self.norm(diff.float()).to(act_dtype)

