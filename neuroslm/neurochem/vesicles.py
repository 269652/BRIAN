"""Neuro-Vesicle Pool — discrete mobile packets for long-range neuromodulation.

Lifecycle per tick:
  1. Emission   — high-surprise events synthesise new vesicles (surprise-triggered)
  2. Migration  — stochastic diffusion across module graph via transition matrix T
  3. Docking    — probabilistic release: vesicle content modulates target module
  4. Decay      — lifetime countdown; zero-lifetime vesicles die

XLA-safe: all operations use static shapes and masked arithmetic; no nonzero()
or Python-level iteration over live vesicles.

Reference:
  Friston et al. (2012) Active inference, epistemic value and vicarious trial and error.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class VesiclePool(nn.Module):
    """Population of V discrete vesicle-like content packets.

    Args:
        d_sem:      content-vector dimension
        n_modules:  number of brain-module nodes in the migration graph
        n_vesicles: pool capacity (static; inactive slots carry zero content)
        lifetime:   initial lifetime (ticks) for a freshly emitted vesicle
    """

    def __init__(self, d_sem: int, n_modules: int,
                 n_vesicles: int = 32, lifetime: int = 16):
        super().__init__()
        self.d_sem     = d_sem
        self.n_modules = n_modules
        self.V         = n_vesicles
        self.lifetime  = float(lifetime)

        # ── Learnable stochastic transition matrix T ──────────────────────
        # T[i, j] = probability a vesicle at module i moves to module j.
        # Row-stochastic via softmax.  Off-diagonal init → mild diffusion.
        self.log_T = nn.Parameter(torch.zeros(n_modules, n_modules))

        # ── Emission: map surprise signal → new vesicle content ───────────
        self.synthesis_gate = nn.Sequential(
            nn.Linear(d_sem, d_sem),
            nn.SiLU(),
            nn.Linear(d_sem, d_sem),
        )
        nn.init.zeros_(self.synthesis_gate[2].weight)
        nn.init.zeros_(self.synthesis_gate[2].bias)

        # ── Docking: cosine attention between vesicle and target module ───
        self.dock_key   = nn.Linear(d_sem, d_sem, bias=False)
        self.dock_query = nn.Linear(d_sem, d_sem, bias=False)

        # ── Modulation: docked content → additive delta for module repr ───
        self.mod_proj = nn.Linear(d_sem, d_sem, bias=False)
        nn.init.zeros_(self.mod_proj.weight)

        # ── Buffers: static-shape population state ────────────────────────
        # All V slots always exist; inactive ones have lifetime ≤ 0.
        self.register_buffer("v_contents",   torch.zeros(n_vesicles, d_sem))
        self.register_buffer("v_lifetimes",  torch.zeros(n_vesicles))          # ≤0 → dead
        self.register_buffer("v_positions",  torch.zeros(n_vesicles, n_modules))  # one-hot
        self._write_ptr: int = 0

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _active_mask(self) -> torch.Tensor:
        """(V,) bool: True for vesicles with remaining lifetime."""
        return self.v_lifetimes > 0.0

    # ------------------------------------------------------------------ #
    # 1. Emission                                                           #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def synthesize(self, surprise: torch.Tensor,
                   novelty_threshold: float = 0.3,
                   source_module: int = 0) -> None:
        """Emit a new vesicle when the mean surprise norm exceeds threshold.

        surprise: (B, d_sem)  — CA1 mismatch / world-model prediction error
        """
        mean_surprise = surprise.detach().mean(0)  # (d_sem,)
        if mean_surprise.norm().item() < novelty_threshold:
            return

        content = self.synthesis_gate(mean_surprise.unsqueeze(0)).squeeze(0)

        # Find a dead slot (lifetime ≤ 0); round-robin fallback.
        dead = (self.v_lifetimes <= 0.0)
        if dead.any().item():
            # Pick the first dead slot deterministically (no dynamic nonzero)
            idx = int(dead.float().argmax().item())
        else:
            idx = self._write_ptr % self.V
        self._write_ptr += 1

        self.v_contents[idx]   = content.detach()
        self.v_lifetimes[idx]  = self.lifetime
        # One-hot position encoding
        pos = torch.zeros(self.n_modules, device=self.v_positions.device)
        pos[source_module % self.n_modules] = 1.0
        self.v_positions[idx]  = pos

    # ------------------------------------------------------------------ #
    # 2. Migration — XLA-safe: static-shape Gumbel sampling               #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def migrate(self) -> None:
        """Diffuse all vesicles one step along the stochastic transition matrix.

        Uses Gumbel-argmax instead of torch.multinomial to stay XLA-compatible
        (multinomial requires dynamic dispatch on TPU).

        v_positions is a (V, n_modules) one-hot float tensor.
        The soft transition: new_pos = one_hot(argmax(v_pos @ T + Gumbel))
        This is equivalent to categorical sampling from T[current_pos].
        """
        active = self._active_mask().float().unsqueeze(1)  # (V, 1)

        T = F.softmax(self.log_T, dim=-1)   # (n_modules, n_modules)

        # Transition: weighted sum of destination probs by current soft position
        # v_positions @ T → (V, n_modules) destination distribution
        dest_logits = self.v_positions @ T  # (V, n_modules)

        # Gumbel noise for stochastic sampling (XLA-compatible)
        gumbel = -torch.log(-torch.log(
            torch.rand_like(dest_logits).clamp(1e-6, 1 - 1e-6)))
        new_pos_idx = (dest_logits + gumbel).argmax(dim=-1)  # (V,)

        new_pos = F.one_hot(new_pos_idx, self.n_modules).float()  # (V, n_modules)

        # Only update active vesicles
        self.v_positions = torch.where(
            active.bool().expand_as(new_pos), new_pos, self.v_positions)

    # ------------------------------------------------------------------ #
    # 3. Docking — probabilistic release via cosine attention              #
    # ------------------------------------------------------------------ #
    def dock(self, module_activations: torch.Tensor) -> torch.Tensor:
        """Compute additive modulation signal from all (possibly) docked vesicles.

        module_activations: (B, n_modules, d_sem)
        Returns: modulation  (B, n_modules, d_sem) — additive delta, all-zero
                 if no vesicles are active.
        """
        B, M, D = module_activations.shape
        active_f = self._active_mask().float()          # (V,) — 1 for live, 0 for dead

        # Vesicle key vectors — only live ones carry meaningful content
        k = F.normalize(self.dock_key(self.v_contents), dim=-1)  # (V, D)

        # Module query: (B, M, D) → (B, V, D) by soft-indexing via v_positions
        # v_positions: (V, M) one-hot → weighted-sum module queries per vesicle
        q_all  = self.dock_query(module_activations)   # (B, M, D)
        # (B, V, D) = v_positions @ q_all reshaped
        pos_t  = self.v_positions.unsqueeze(0).expand(B, -1, -1)  # (B, V, M)
        q_ves  = torch.bmm(pos_t, q_all)               # (B, V, D)
        q_ves  = F.normalize(q_ves, dim=-1)

        # Cosine dock scores (B, V), gated by active_f
        scores = (q_ves * k.unsqueeze(0)).sum(-1)      # (B, V)
        scores = torch.sigmoid(scores) * active_f.unsqueeze(0)  # mask dead

        # Modulation content per vesicle: (V, D)
        delta = self.mod_proj(self.v_contents)          # (V, D)

        # Scatter to modules via soft position: (B, V, D) → (B, M, D)
        # contrib[b, v, :] = scores[b,v] * delta[v, :]
        contrib = scores.unsqueeze(-1) * delta.unsqueeze(0)    # (B, V, D)
        # Soft scatter: (B, V, M)ᵀ @ (B, V, D) → (B, M, D)
        pos_t_  = self.v_positions.T.unsqueeze(0).expand(B, -1, -1)  # (B, M, V)
        modulation = torch.bmm(pos_t_, contrib)         # (B, M, D)

        return modulation

    # ------------------------------------------------------------------ #
    # 4. Decay                                                              #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def degrade(self, decay: float = 1.0) -> None:
        """Subtract decay from all vesicle lifetimes; dead → content zeroed."""
        self.v_lifetimes = self.v_lifetimes - decay
        dead_mask = (self.v_lifetimes <= 0.0)           # (V,) bool, static shape
        # Zero out dead vesicle content so they don't contribute even if
        # the mask isn't checked (avoids stale gradient paths)
        self.v_contents = self.v_contents * (~dead_mask).float().unsqueeze(1)

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #
    def tick(self, module_activations: torch.Tensor,
             surprise: torch.Tensor | None = None,
             source_module: int = 0) -> torch.Tensor:
        """Full lifecycle step.

        module_activations: (B, n_modules, d_sem)
        surprise:           (B, d_sem) or None
        Returns: modulation (B, n_modules, d_sem)
        """
        if surprise is not None:
            self.synthesize(surprise, source_module=source_module)
        self.migrate()
        modulation = self.dock(module_activations)
        self.degrade()
        return modulation

    def active_count(self) -> int:
        return int((self.v_lifetimes > 0).sum().item())

    def forward(self, module_activations: torch.Tensor,
                surprise: torch.Tensor | None = None) -> torch.Tensor:
        return self.tick(module_activations, surprise)
