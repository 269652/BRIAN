"""Neuro-Vesicle Pool — discrete content packets for neuromodulation.

Inspired by synaptic vesicles: discrete packets of neurotransmitter content
that migrate between modules, dock to release their payload, and modulate
downstream activations. This provides a biologically-grounded mechanism for
long-range, asynchronous neuromodulation that complements the NT scalar signals.

Architecture:
  VesiclePool maintains V vesicles, each with:
    - content vector (d_sem): what semantic payload the vesicle carries
    - lifetime τ: how long the vesicle persists before degradation
    - position: which module it is currently at (0..n_modules-1)
    - active flag: whether the vesicle is alive

  Each tick:
    1. Synthesis: high-surprise events generate new vesicles
    2. Migration: vesicles diffuse across modules via a learned transition matrix
    3. Docking: vesicles dock to modules via cosine attention → modulation signal
    4. Degradation: vesicle lifetimes decay; depleted vesicles die

Modulation output (B, n_modules, d_sem) is added to module activations
as a slow, long-range neuromodulatory signal — distinct from moment-to-moment
attention which is fast and local.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class VesiclePool(nn.Module):
    """Pool of V synaptic-vesicle-like content packets.

    Args:
        d_sem:      semantic dimension (content vector size)
        n_modules:  number of brain modules (migration graph nodes)
        n_vesicles: maximum live vesicles at any time
        lifetime:   initial lifetime counter (ticks until degradation)
    """

    def __init__(self, d_sem: int, n_modules: int,
                 n_vesicles: int = 32, lifetime: int = 16):
        super().__init__()
        self.d_sem     = d_sem
        self.n_modules = n_modules
        self.V         = n_vesicles
        self.lifetime  = lifetime

        # Learned inter-module migration transition matrix (row-stochastic)
        # T[i,j] = probability vesicle at module i migrates to module j per tick
        self.log_T = nn.Parameter(torch.zeros(n_modules, n_modules))

        # Synthesis gate: maps surprise signal → new vesicle content
        self.synthesis_gate = nn.Sequential(
            nn.Linear(d_sem, d_sem),
            nn.SiLU(),
            nn.Linear(d_sem, d_sem),
        )
        nn.init.zeros_(self.synthesis_gate[2].weight)
        nn.init.zeros_(self.synthesis_gate[2].bias)

        # Docking attention: vesicle content × module activation → dock score
        self.dock_key   = nn.Linear(d_sem, d_sem, bias=False)
        self.dock_query = nn.Linear(d_sem, d_sem, bias=False)

        # Modulation projection: docked content → modulation delta
        self.mod_proj = nn.Linear(d_sem, d_sem, bias=False)
        nn.init.zeros_(self.mod_proj.weight)

        # --- Buffers (non-trainable state) ---
        self.register_buffer("v_contents",  torch.zeros(n_vesicles, d_sem))
        self.register_buffer("v_lifetimes", torch.zeros(n_vesicles))
        self.register_buffer("v_positions", torch.zeros(n_vesicles, dtype=torch.long))
        self.register_buffer("v_active",    torch.zeros(n_vesicles, dtype=torch.bool))
        self._write_ptr: int = 0

    # ------------------------------------------------------------------
    # Synthesis: create new vesicle from a surprise event
    # ------------------------------------------------------------------
    @torch.no_grad()
    def synthesize(self, surprise: torch.Tensor,
                   novelty_threshold: float = 0.5,
                   source_module: int = 0) -> None:
        """Synthesise new vesicles where surprise exceeds threshold.

        surprise: (B, d_sem) — surprise / novelty signal from CA1 or world model
        """
        mean_surprise = surprise.mean(0)   # (d_sem,) — pool over batch

        if mean_surprise.norm().item() < novelty_threshold:
            return

        content = self.synthesis_gate(mean_surprise.unsqueeze(0)).squeeze(0)

        # Write to next available slot (ring buffer, overwrites dead first)
        # Find a dead slot; fall back to oldest
        dead = (~self.v_active).nonzero(as_tuple=True)[0]
        if dead.numel() > 0:
            idx = int(dead[0].item())
        else:
            idx = self._write_ptr % self.V

        self.v_contents[idx]  = content.detach()
        self.v_lifetimes[idx] = float(self.lifetime)
        self.v_positions[idx] = source_module % self.n_modules
        self.v_active[idx]    = True
        self._write_ptr      += 1

    # ------------------------------------------------------------------
    # Migration: vectorized stochastic diffusion (SIMD-friendly)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def migrate(self) -> None:
        """Migrate all active vesicles in a single batched multinomial sample.

        Replaces the per-vesicle Python loop with a vectorized operation:
          1. Gather transition probabilities for all active vesicles at once.
          2. Call torch.multinomial on the resulting (V_live, n_modules) matrix —
             one sample per row, executed as a single GPU/TPU kernel.
        This maps naturally to TPU SIMD lanes and avoids Python-level dispatch.
        """
        active_idx = self.v_active.nonzero(as_tuple=True)[0]
        if active_idx.numel() == 0:
            return
        T      = F.softmax(self.log_T, dim=-1)           # (n_modules, n_modules)
        pos    = self.v_positions[active_idx]             # (V_live,)
        probs  = T[pos]                                   # (V_live, n_modules)
        new_pos = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (V_live,)
        self.v_positions[active_idx] = new_pos

    # ------------------------------------------------------------------
    # Docking: vectorized FiLM-style modulation (SIMD-friendly)
    # ------------------------------------------------------------------
    def dock(self, module_activations: torch.Tensor) -> torch.Tensor:
        """Compute modulation signal from all active vesicles in one pass.

        Replaces the per-vesicle Python loop with:
          1. Batch-project all active vesicle contents into key space.
          2. Gather module queries at each vesicle's target position.
          3. Compute all cosine dock-scores in a single (B, V_live) op.
          4. Accumulate per-module modulation via scatter_add_.

        module_activations: (B, n_modules, d_sem)
        Returns: modulation (B, n_modules, d_sem) — additive delta
        """
        B      = module_activations.size(0)
        device = module_activations.device
        modulation = torch.zeros_like(module_activations)

        active_idx = self.v_active.nonzero(as_tuple=True)[0]
        V_live = active_idx.numel()
        if V_live == 0:
            return modulation

        v_pos      = self.v_positions[active_idx]           # (V_live,)
        v_contents = self.v_contents[active_idx]            # (V_live, d_sem)

        # Projected keys (V_live, d_sem)
        k = F.normalize(self.dock_key(v_contents), dim=-1)

        # Module query tensor: (B, n_modules, d_sem)
        q_all = self.dock_query(module_activations)         # (B, n_modules, d_sem)

        # Gather queries for each vesicle's target module: (B, V_live, d_sem)
        pos_exp = v_pos.unsqueeze(0).expand(B, -1)          # (B, V_live)
        q = q_all.gather(
            1,
            pos_exp.unsqueeze(-1).expand(B, V_live, q_all.size(-1))
        )  # (B, V_live, d_sem)
        q = F.normalize(q, dim=-1)

        # Cosine dock scores: (B, V_live)
        scores = torch.sigmoid(
            (q * k.unsqueeze(0)).sum(-1)                    # (B, V_live)
        )

        # Modulation deltas: (V_live, d_sem)
        delta = self.mod_proj(v_contents)                   # (V_live, d_sem)

        # Weighted contributions: (B, V_live, d_sem)
        contrib = scores.unsqueeze(-1) * delta.unsqueeze(0)

        # Scatter-add into module positions — fully vectorized, no Python loop
        pos_sc = pos_exp.unsqueeze(-1).expand_as(contrib)   # (B, V_live, d_sem)
        modulation.scatter_add_(1, pos_sc, contrib)

        return modulation

    # ------------------------------------------------------------------
    # Degradation: reduce lifetime, deactivate depleted vesicles
    # ------------------------------------------------------------------
    @torch.no_grad()
    def degrade(self, decay: float = 1.0) -> None:
        """Reduce all active vesicle lifetimes by decay; kill if ≤ 0."""
        active_idx = self.v_active.nonzero(as_tuple=True)[0]
        if active_idx.numel() == 0:
            return
        self.v_lifetimes[active_idx] -= decay
        dead = active_idx[self.v_lifetimes[active_idx] <= 0]
        if dead.numel() > 0:
            self.v_active[dead] = False

    # ------------------------------------------------------------------
    # Tick: synthesize → migrate → dock → degrade
    # ------------------------------------------------------------------
    def tick(self, module_activations: torch.Tensor,
             surprise: torch.Tensor | None = None,
             source_module: int = 0) -> torch.Tensor:
        """One simulation tick.

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
        return int(self.v_active.sum().item())

    def forward(self, module_activations: torch.Tensor,
                surprise: torch.Tensor | None = None) -> torch.Tensor:
        return self.tick(module_activations, surprise)
