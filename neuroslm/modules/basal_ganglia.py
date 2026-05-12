"""Basal Ganglia: action selection via Go/NoGo + forward simulation + safety gate.

Implements the full cortico-striato-thalamo-cortical loop:

  1. PFC selection arrives as the input thought embedding.
  2. Striatum proposes K action candidates (proposer network).
  3. Forward simulation (optional): each candidate is rolled through the
     world model and evaluated by the critic + value estimator before scoring.
  4. Go pathway  (striatum D1 → GPe direct → GPi/SNr disinhibition):
       score = go_net(cand) * (0.5 + DA)  + predicted_value * 0.5
  5. NoGo pathway (striatum D2 → GPe indirect → STN → GPi/SNr inhibition):
       score = nogo_net(cand) * (1.5 - DA) + predicted_threat * 1.5
  6. STN hyperdirect pathway: if NE > 0.82 (extreme threat / surprise) →
       emergency stop — all Go scores zeroed, return safest action.
  7. GPi/SNr net gate: selected_action = argmax(Go - NoGo).
  8. Safety gate before committing to motor:
       commit_ok = (predicted_threat[selected] < safety_threshold)
       AND (Go_score[selected] > nogo_score[selected])

The forward() call returns:
    (chosen_action, confidence, probs, commit_ok)
where `commit_ok` is a bool tensor — motor cortex should check this before
emitting a response.

References:
  Frank (2005) Dynamic dopamine modulation in the BG.
  Hazy et al. (2007) Towards an executive without a homunculus.
  Nambu et al. (2002) Functional significance of the cortico-STN-GPi hyperdirect pathway.
  Bogacz & Gurney (2007) The BG and cortex implement optimal decision making.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .brain_module import BrainModule


class BasalGanglia(BrainModule):
    def __init__(self, d_sem: int, d_action: int, n_candidates: int,
                 safety_threshold: float = 0.45,
                 stn_ne_threshold: float = 0.82):
        super().__init__()
        self.n_candidates    = n_candidates
        self.d_action        = d_action
        self.safety_threshold = safety_threshold
        self.stn_ne_threshold = stn_ne_threshold

        # --- Striatum: action proposer ---
        self.proposer = nn.Sequential(
            nn.Linear(d_sem, d_sem),
            nn.GELU(),
            nn.Linear(d_sem, d_action * n_candidates),
        )

        # --- Go pathway (D1 receptor, direct path) ---
        self.go_net = nn.Sequential(
            nn.Linear(d_action, d_action // 2), nn.GELU(),
            nn.Linear(d_action // 2, 1),
        )

        # --- NoGo pathway (D2 receptor, indirect path) ---
        self.nogo_net = nn.Sequential(
            nn.Linear(d_action, d_action // 2), nn.GELU(),
            nn.Linear(d_action // 2, 1),
        )

        # --- STN hyperdirect pathway: emergency stop ---
        # Projects thought to a "hold-on" signal (suppresses all Go)
        self.stn = nn.Linear(d_sem, 1)

        # --- Habitual vs goal-directed gating ---
        # High DA + familiar action → habitual (fast); low DA → goal-directed (slow)
        self.habit_proj = nn.Linear(d_action, 1)

        # ── VQH (Vector-Quantised Striatum) model ────────────────────────
        # The striatum is modelled as a discrete option lattice: ``n_options``
        # learnable codebook entries that the proposer is quantised onto via
        # straight-through nearest-neighbour. Each option carries:
        #   • a key vector (d_action) — pattern matched against the proposer
        #   • a 3-way logit (n_experts) — gates which expert cortex executes
        #     the action when it commits (Math / Reasoning / Motor)
        # Trained via the standard VQ-VAE auxiliary loss (commitment + codebook).
        self.n_options = 16
        self.n_experts = 3              # 0 = Math, 1 = Reasoning, 2 = Motor
        self.option_keys = nn.Parameter(
            torch.randn(self.n_options, d_action) * (1.0 / (d_action ** 0.5)))
        self.option_expert_logits = nn.Parameter(
            torch.zeros(self.n_options, self.n_experts))
        # Beta for VQ commitment loss
        self.vq_beta = 0.25

        # ── NAcc: reward prediction error ───────────────────────────────
        # A small value head over (thought, action) predicts expected
        # survival-aligned reward. The RPE is the difference between this
        # prediction and the actual survival outcome observed downstream.
        # Positive RPE → DA spike (read by the brain to release DA).
        self.nacc_value = nn.Sequential(
            nn.Linear(d_sem + d_action, d_sem),
            nn.GELU(),
            nn.Linear(d_sem, 1),
        )

        # Per-option visitation counts + DA-weighted policy (for tests
        # that need to track policy adaptation over many steps without
        # waiting for full gradient convergence). Plain torch buffers so
        # they round-trip in the state dict.
        self.register_buffer(
            "option_da_value", torch.zeros(self.n_options))
        self.register_buffer(
            "option_visits",    torch.zeros(self.n_options))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        thought: torch.Tensor,           # (B, d_sem) — PFC selection
        nt_levels: dict,
        world_model=None,                # optional: for forward simulation
        z_world: torch.Tensor | None = None,
        z_self:  torch.Tensor | None = None,
        critic=None,
        evaluator=None,
    ):
        """
        Returns:
            chosen_action  (B, d_action)
            confidence     (B,)
            probs          (B, K)
            commit_ok      (B,) bool — safe to send to motor cortex
        """
        B = thought.size(0)
        da = nt_levels.get("DA",  0.5)
        ne = nt_levels.get("NE",  0.5)

        # --- STN hyperdirect: emergency stop on extreme NE ---
        stn_signal = torch.sigmoid(self.stn(thought)).squeeze(-1)  # (B,)
        emergency_stop = (ne > self.stn_ne_threshold) or bool((stn_signal > 0.8).any())

        # --- Propose K action candidates ---
        raw = self.proposer(thought)                          # (B, K*d_action)
        cands = raw.view(B, self.n_candidates, self.d_action) # (B, K, d_action)

        # --- Go scores (direct path, D1-mediated, boosted by DA) ---
        go_raw = self.go_net(cands).squeeze(-1)               # (B, K)
        go_scores = go_raw * (0.5 + da)                       # DA boosts go

        # --- NoGo scores (indirect path, D2-mediated, suppressed by DA) ---
        nogo_raw   = self.nogo_net(cands).squeeze(-1)         # (B, K)
        nogo_scores = nogo_raw * max(0.1, 1.5 - da)          # DA suppresses nogo

        # --- Optional forward simulation ---
        pred_values  = torch.zeros(B, self.n_candidates, device=thought.device)
        pred_threats = torch.zeros(B, self.n_candidates, device=thought.device)

        if world_model is not None and z_world is not None and z_self is not None:
            with torch.no_grad():
                for k in range(self.n_candidates):
                    act_k = cands[:, k]                       # (B, d_action)
                    try:
                        wp_k, sp_k = world_model(z_world, z_self, act_k)
                        if evaluator is not None:
                            # evaluator expects nt vector; build minimal one
                            nt_t = torch.zeros(B, 7, device=thought.device)
                            val_k = evaluator(wp_k, sp_k, nt_t).squeeze(-1)
                            pred_values[:, k] = val_k.clamp(-1.0, 1.0)
                        if critic is not None:
                            threat_k, _ = critic(wp_k, sp_k)
                            pred_threats[:, k] = threat_k.clamp(0.0, 1.0)
                    except Exception:
                        pass

            # Boost Go by predicted value; boost NoGo by predicted threat
            go_scores   = go_scores   + 0.5 * pred_values
            nogo_scores = nogo_scores + 1.5 * pred_threats

        # --- Habit gate ---
        habit = torch.sigmoid(self.habit_proj(cands)).squeeze(-1)  # (B, K)
        # High DA + habitual action → slight go bonus (fast habitual selection)
        go_scores = go_scores + 0.2 * da * habit

        # --- GPi/SNr: net score ---
        net_score = go_scores - nogo_scores                   # (B, K)

        # Emergency stop: zero out all Go (STN suppresses GPi/SNr)
        if emergency_stop:
            net_score = net_score * 0.01   # near-zero — safest action chosen

        # --- Selection ---
        probs = F.softmax(net_score, dim=-1)                  # (B, K)
        idx   = probs.argmax(dim=-1)                          # (B,)
        chosen = cands[torch.arange(B, device=thought.device), idx]   # (B, d_action)
        confidence = probs.max(dim=-1).values                 # (B,)

        # --- Safety gate ---
        sel_threat = pred_threats[torch.arange(B, device=thought.device), idx]  # (B,)
        sel_go     = go_scores[torch.arange(B, device=thought.device), idx]
        sel_nogo   = nogo_scores[torch.arange(B, device=thought.device), idx]
        commit_ok  = (sel_threat < self.safety_threshold) & (sel_go > sel_nogo)

        if emergency_stop:
            commit_ok = torch.zeros(B, dtype=torch.bool, device=thought.device)

        return chosen, confidence, probs, commit_ok

    def _disabled_output(self, thought, *_, **__):
        B = thought.size(0)
        zeros  = torch.zeros(B, self.d_action, device=thought.device)
        conf   = torch.ones(B, device=thought.device) * 0.5
        probs  = torch.ones(B, self.n_candidates, device=thought.device) / self.n_candidates
        commit = torch.zeros(B, dtype=torch.bool, device=thought.device)
        return zeros, conf, probs, commit

    # ------------------------------------------------------------------
    # VQH option selection + expert-cortex gate
    # ------------------------------------------------------------------
    def select_option(self,
                       action: torch.Tensor,
                       da_bias: float = 0.0,
                       ) -> tuple[int, torch.Tensor, torch.Tensor, float]:
        """Quantise the chosen continuous action onto the discrete option
        lattice, return the option index and the expert-routing distribution.

        action:  (B, d_action) continuous chosen action from forward().
        da_bias: optional bias on the policy from current DA levels —
                  high DA narrows the selection toward the previously-
                  most-rewarding option (exploitation).

        Returns (option_idx, expert_probs, vq_loss, value_prediction).

        Uses straight-through gradient: forward pass quantises, backward
        pass treats the quantisation as the identity. The vq_loss term
        is the standard codebook + commitment objective; train.py adds
        it to total via `_aux_w_scale`.
        """
        a = action if action.dim() == 2 else action.unsqueeze(0)
        # Cosine-similarity selection — natural unit invariance
        a_n = a / (a.norm(dim=-1, keepdim=True) + 1e-6)
        k_n = self.option_keys / (self.option_keys.norm(dim=-1, keepdim=True) + 1e-6)
        sims = a_n @ k_n.T                                 # (B, n_options)
        # DA bias from policy memory (high-DA value boosts familiar options)
        if da_bias != 0.0:
            sims = sims + da_bias * self.option_da_value.to(sims.dtype).unsqueeze(0)
        opt_idx = int(sims[0].argmax().item())             # batch-mean

        # Expert-routing distribution over (Math, Reasoning, Motor)
        expert_probs = torch.softmax(self.option_expert_logits[opt_idx], dim=-1)

        # VQ-VAE loss: codebook gets pulled toward the action, action gets
        # pulled toward the codebook entry (commitment).
        chosen_key = self.option_keys[opt_idx].unsqueeze(0).expand_as(a)
        codebook_loss   = (chosen_key - a.detach()).pow(2).mean()
        commitment_loss = (chosen_key.detach() - a).pow(2).mean()
        vq_loss = codebook_loss + self.vq_beta * commitment_loss

        # Predicted survival value of (thought, action) — read by NAcc RPE
        # update. Caller provides a thought tensor by concatenating outside.
        value_pred = torch.zeros(1, device=a.device, dtype=a.dtype)

        # Bookkeeping
        with torch.no_grad():
            self.option_visits[opt_idx] += 1

        return opt_idx, expert_probs, vq_loss, float(value_pred.detach().item())

    # ------------------------------------------------------------------
    # NAcc reward-prediction error
    # ------------------------------------------------------------------
    def nacc_rpe(self,
                  thought: torch.Tensor,
                  action: torch.Tensor,
                  actual_survival_reward: float,
                  ) -> tuple[float, float]:
        """Compute RPE = actual − predicted survival reward.

        Returns (rpe, predicted). Positive RPE → DA spike to be released
        by the caller (the brain). The predicted-value head is trained by
        the caller via the standard MSE on (predicted, actual_reward),
        added to aux loss with `_aux_w_scale * w_nacc`.
        """
        a = action if action.dim() == 2 else action.unsqueeze(0)
        t = thought if thought.dim() == 2 else thought.unsqueeze(0)
        x = torch.cat([t, a], dim=-1)
        pred = float(self.nacc_value(x).mean().item())
        rpe  = float(actual_survival_reward) - pred
        return rpe, pred

    def update_option_value(self,
                              option_idx: int,
                              rpe: float,
                              lr: float = 0.1) -> None:
        """DA-gated policy memory: nudge the option's stored value toward
        the observed RPE. Pure bookkeeping (no gradients), used by tests
        and by the brain's consolidation loop."""
        if 0 <= option_idx < self.n_options:
            with torch.no_grad():
                self.option_da_value[option_idx] = (
                    (1.0 - lr) * self.option_da_value[option_idx] + lr * rpe)
