"""Consciousness metrics: measurable indicators of integrated information,
neural oscillations, coherence, and phenomenal binding.

Inspired by:
  - Integrated Information Theory (IIT): Φ as a measure of irreducibility
  - Global Workspace Theory: broadcast/ignition as consciousness correlate
  - Neural oscillations: gamma (binding), theta (memory), alpha (idling)
  - Coherence: phase synchronization across modules
  - Metacognition: confidence calibration as self-awareness proxy

These metrics are computed each tick and logged. They don't directly
affect computation but provide interpretability and can drive training
rewards (e.g., higher Φ → bonus reward in mesolimbic circuit).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class ConsciousnessMetrics(nn.Module):
    """Tracks and computes consciousness-related metrics each tick."""

    def __init__(self, d_sem: int, n_modules: int = 8, history_len: int = 64):
        super().__init__()
        self.d_sem = d_sem
        self.n_modules = n_modules
        self.history_len = history_len

        # Oscillation tracking buffers (circular)
        self.register_buffer("_tick", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_gamma_history", torch.zeros(history_len))   # 30-100 Hz binding
        self.register_buffer("_theta_history", torch.zeros(history_len))   # 4-8 Hz memory
        self.register_buffer("_alpha_history", torch.zeros(history_len))   # 8-12 Hz idle
        self.register_buffer("_phi_history", torch.zeros(history_len))     # integration
        self.register_buffer("_coherence_history", torch.zeros(history_len))

        # Module activity matrix for integration computation
        self.register_buffer("_module_activities", torch.zeros(n_modules, d_sem))

        # Ignition detector: detects when GWS broadcast causes widespread
        # activation (neural correlate of conscious access)
        self.ignition_threshold = 0.6

    @torch.no_grad()
    def update(self, module_outputs: dict[str, torch.Tensor],
               gws_slots: torch.Tensor,
               floating_thought: torch.Tensor,
               novelty: torch.Tensor,
               routing: torch.Tensor) -> dict:
        """Compute all metrics for this tick.

        Args:
            module_outputs: {name: (B, D)} outputs from brain modules
            gws_slots: (B, N, D) GWS broadcast content
            floating_thought: (B, D) current thought state
            novelty: (B,) novelty signal
            routing: (B, N_routes) thalamic routing distribution

        Returns: dict of scalar metrics
        """
        idx = int(self._tick.item()) % self.history_len
        self._tick += 1

        B = floating_thought.size(0)
        device = floating_thought.device

        # 1) Gamma oscillation proxy: binding coherence across GWS slots
        # High gamma = tight binding between represented items
        if gws_slots.size(1) > 1:
            slot_norms = F.normalize(gws_slots, dim=-1)
            sim_matrix = torch.bmm(slot_norms, slot_norms.transpose(1, 2))
            # Off-diagonal mean = binding strength
            mask = ~torch.eye(sim_matrix.size(1), device=device, dtype=torch.bool)
            gamma = sim_matrix[:, mask.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)].view(B, -1).mean()
        else:
            gamma = torch.tensor(0.0, device=device)
        gamma = gamma.clamp(0.0, 1.0)
        self._gamma_history[idx] = float(gamma)

        # 2) Theta oscillation proxy: memory retrieval activity
        # Theta correlates with hippocampal-cortical communication
        theta = novelty.mean().clamp(0.0, 1.0)
        self._theta_history[idx] = float(theta)

        # 3) Alpha oscillation proxy: idling / suppression
        # High alpha when routing is uniform (nothing salient selected)
        routing_entropy = -(routing * (routing + 1e-8).log()).sum(-1).mean()
        max_entropy = math.log(routing.size(-1))
        alpha = (routing_entropy / max_entropy).clamp(0.0, 1.0)
        self._alpha_history[idx] = float(alpha)

        # 4) Φ (Phi) proxy: integrated information
        # Approximate as mutual information between module pairs
        # (full IIT is intractable; we use a differentiable proxy)
        phi = self._compute_phi_proxy(module_outputs, device)
        self._phi_history[idx] = phi

        # 5) Coherence: phase alignment of module outputs with GWS
        coherence = self._compute_coherence(module_outputs, gws_slots)
        self._coherence_history[idx] = coherence

        # 6) Ignition detection: did GWS broadcast cause widespread activation?
        active_modules = sum(
            1 for v in module_outputs.values()
            if v.norm(dim=-1).mean() > self.ignition_threshold
        )
        ignition = active_modules / max(len(module_outputs), 1)

        # 7) Metacognitive accuracy proxy: thought stability
        # Stable thought = higher confidence = more metacognitive access
        thought_magnitude = floating_thought.norm(dim=-1).mean()
        metacognition = torch.sigmoid(thought_magnitude - 1.0)

        # 8) Phenomenal binding: how unified is the current experience?
        # Cross-correlation between thought and qualia-modulated slots
        binding = float(gamma) * float(coherence)

        metrics = {
            "gamma": float(gamma),           # binding oscillation
            "theta": float(theta),           # memory oscillation
            "alpha": float(alpha),           # idle oscillation
            "phi": phi,                      # integrated information
            "coherence": coherence,          # phase coherence
            "ignition": ignition,            # conscious access event
            "metacognition": float(metacognition),
            "binding": binding,              # phenomenal unity
            "tick": int(self._tick.item()),
        }
        return metrics

    def _compute_phi_proxy(self, module_outputs: dict[str, torch.Tensor],
                           device) -> float:
        """Approximate Φ as average pairwise cosine similarity minus
        what you'd get from independent modules (partition baseline).
        High Φ = modules are more correlated together than apart."""
        vecs = [v.mean(0) if v.dim() > 1 else v for v in module_outputs.values()]
        if len(vecs) < 2:
            return 0.0
        vecs = [F.normalize(v.flatten().unsqueeze(0), dim=-1) for v in vecs[:8]]
        try:
            stacked = torch.cat(vecs, dim=0)  # (N, D_flat)
            # Trim to same size
            min_d = min(v.size(-1) for v in vecs)
            stacked = stacked[:, :min_d]
            sim = F.cosine_similarity(stacked.unsqueeze(0), stacked.unsqueeze(1), dim=-1)
            n = sim.size(0)
            mask = ~torch.eye(n, device=device, dtype=torch.bool)
            phi = sim[mask].mean().item()
        except Exception:
            phi = 0.0
        return max(0.0, phi)

    def _compute_coherence(self, module_outputs: dict[str, torch.Tensor],
                           gws_slots: torch.Tensor) -> float:
        """How aligned are module outputs with the GWS broadcast?"""
        gws_mean = gws_slots.mean(dim=(0, 1))  # (D,)
        gws_norm = F.normalize(gws_mean.unsqueeze(0), dim=-1)
        similarities = []
        for v in module_outputs.values():
            v_flat = v.mean(0).flatten()[:gws_mean.size(0)]
            if v_flat.size(0) < gws_mean.size(0):
                continue
            v_norm = F.normalize(v_flat.unsqueeze(0), dim=-1)
            sim = F.cosine_similarity(v_norm, gws_norm).item()
            similarities.append(sim)
        return sum(similarities) / max(len(similarities), 1)

    def oscillation_spectrum(self) -> dict:
        """Get recent oscillation power spectrum."""
        return {
            "gamma_mean": float(self._gamma_history.mean()),
            "gamma_std": float(self._gamma_history.std()),
            "theta_mean": float(self._theta_history.mean()),
            "alpha_mean": float(self._alpha_history.mean()),
            "phi_mean": float(self._phi_history.mean()),
            "coherence_mean": float(self._coherence_history.mean()),
        }


# Need F for normalize
import torch.nn.functional as F
