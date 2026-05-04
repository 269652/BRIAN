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
