"""BDNF / NGF: neurotrophic factors that grow or prune projections.

Each projection in `ProjectionGraph` gets a scalar `trophic_level` ∈ [0, 1].
Update rule (per tick):
    Δ trophic_level = BDNF · co_activation - NGF_decay - disuse
where:
  - co_activation  = cosine similarity of recent src/dst activity scalars
                     (positive Hebbian: 'fire together, wire together')
  - BDNF           = global level, raised by positive reward / RPE
  - NGF_decay      = small constant (slow forgetting)
  - disuse         = penalty when neither end fires

Effects:
  - The trophic level multiplicatively scales the projection's signal-carrying
    linear map weight (potentiation / depression).
  - When trophic_level drops below `prune_threshold`, the projection is
    DISABLED (zero contribution).
  - When it would saturate above 1.0, the system can SPAWN a new projection
    along an inferred high-coactivation edge (sprouting).

This is an inference-time / training-time process (no SGD); changes persist
in buffers and are saved with the checkpoint.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .projections import ProjectionGraph, Projection


class TrophicSystem(nn.Module):
    def __init__(self, graph: ProjectionGraph,
                 prune_threshold: float = 0.05,
                 sprout_threshold: float = 0.95,
                 bdnf_baseline: float = 0.005,
                 ngf_decay: float = 0.002,
                 max_projections: int = 64,
                 phi_boost: float = 2.0):
        super().__init__()
        self.graph = graph
        self.prune_threshold = prune_threshold
        self.sprout_threshold = sprout_threshold
        self.bdnf_baseline = bdnf_baseline
        self.ngf_decay = ngf_decay
        self.max_projections = max_projections
        # phi_boost: multiplier on BDNF when Φ is high (couples structural
        # growth to high integrated-information states — locks high-Φ pathways)
        self.phi_boost = phi_boost
        n = len(graph.projections)
        # Start every existing projection at mid trophic level
        self.register_buffer("trophic", torch.full((n,), 0.5))
        self.register_buffer("active",  torch.ones(n))
        # Co-activation EMA per projection (over recent ticks)
        self.register_buffer("ema_coact", torch.zeros(n))
        self._steps = 0

    @torch.no_grad()
    def _try_sprout(self, activities: dict[str, torch.Tensor]) -> int:
        """Spawn new projections between high-activity, weakly-connected modules.

        Implements structural plasticity (Hebb): when two modules consistently
        co-activate but have no/weak existing trophic-supported edge, sprout
        a new projection.  Capped at self.max_projections.

        Returns the number of new projections created this call.
        """
        if len(self.graph.projections) >= self.max_projections:
            return 0

        # High-activity modules (mean activity in top quartile)
        names = list(activities.keys())
        if len(names) < 2:
            return 0
        acts = {n: float(activities[n].mean()) for n in names}
        sorted_names = sorted(names, key=lambda n: -acts[n])
        top_n = max(2, len(sorted_names) // 4)
        hot = sorted_names[:top_n]

        # Build set of existing edges
        existing = {(p.src, p.dst) for p in self.graph.projections}

        # Find best candidate: co-activation × novelty (no existing edge)
        spawned = 0
        for i, src in enumerate(hot):
            if spawned > 0:
                break  # at most one sprout per update tick (slow growth)
            for dst in hot[i + 1:]:
                if (src, dst) in existing or (dst, src) in existing:
                    continue
                if src not in self.graph.regions or dst not in self.graph.regions:
                    continue
                # Hebbian co-activation (both fire above 0.5)
                if acts[src] > 0.5 and acts[dst] > 0.5:
                    new_proj = Projection(src, dst, "Glu", release_scale=0.4)
                    self.graph.add_projection(new_proj)
                    # Extend trophic buffers by one slot
                    self.trophic   = torch.cat([self.trophic,   torch.full((1,), 0.5, device=self.trophic.device)])
                    self.active    = torch.cat([self.active,    torch.ones(1, device=self.active.device)])
                    self.ema_coact = torch.cat([self.ema_coact, torch.zeros(1, device=self.ema_coact.device)])
                    spawned += 1
                    break
        return spawned

    @torch.no_grad()
    def update_sdnr_gated(self, activities: dict[str, torch.Tensor],
                         signal_contrib: torch.Tensor | None = None,
                         noise_contrib: torch.Tensor | None = None,
                         sdnr_threshold: float = 0.5,
                         maturity: float | None = None,
                         prune_mat_threshold: float = 0.3):
        """SDNR-gated structural pruning: prune projections with low signal-to-noise.

        If a re-entrant projection contributes more to global variance (noise)
        than to predictive accuracy (signal), aggressively prune its trophic level.

        Args:
            activities: {region: (B,)} tensor of module activities
            signal_contrib: (n_projections,) — contribution to predictive accuracy
            noise_contrib: (n_projections,) — contribution to global variance
            sdnr_threshold: SDNR below this triggers aggressive pruning
            maturity / prune_mat_threshold: see `update()` — softens the gate
                so a random-init graph cannot prune itself to n_active=0.
        """
        if signal_contrib is None or noise_contrib is None:
            return  # skip if signal/noise not available

        # Compute SDNR per projection (signal / noise + eps)
        sdnr = signal_contrib / (noise_contrib + 1e-6)
        low_quality = sdnr < sdnr_threshold  # (n_projections,) bool

        # Aggressively prune low-quality re-entrant projections
        prune_rate = 0.05  # drop trophic by 5% per update
        self.trophic[low_quality] = self.trophic[low_quality] * (1.0 - prune_rate)
        self.trophic.clamp_(0.0, 1.0)

        # Deactivate if pruned below threshold — gated on MAT.
        if (maturity is None) or (float(maturity) >= prune_mat_threshold):
            below_prune = self.trophic < self.prune_threshold
            self.active[below_prune] = 0.0

    @torch.no_grad()
    def update(self, activities: dict[str, torch.Tensor], bdnf: float, ngf: float,
               phi: float = 0.0, fiedler: float = 1.0,
               maturity: float | None = None,
               prune_mat_threshold: float = 0.3):
        """activities: {region: (B,) ∈ [0,1]}.
        bdnf, ngf:  scalar floats from Brain (driven by reward / novelty).
        phi:        Integrated information proxy ∈ [0, 1].
                    High Φ boosts BDNF, locking high-integration pathways.
        fiedler:    Spectral gap λ₁ of the module interaction graph ∈ [0, 2].
                    Near 0 → graph nearly disconnected → homeostatic BDNF
                    boost to strengthen weak cross-module connections and
                    prevent a collapse in global Φ (Cheeger's inequality).
                    Large → well-integrated → normal pruning dynamics.
        maturity:   MAT scalar ∈ [0, 1]. When provided, pruning (active[i]=0)
                    is suppressed while MAT < `prune_mat_threshold` — fixes
                    the "n_active: 0" graph-collapse on bring-up where the
                    random-init projections look low-quality and all get
                    pruned before they can learn.
        prune_mat_threshold: MAT level at and above which pruning re-engages.
        """
        # Pruning gate: disable structural deactivation below MAT threshold.
        # Trophic level itself still drifts so plasticity accumulates state;
        # we just refuse to set active[i] = 0 until the network has stabilised.
        _allow_prune = (maturity is None) or (float(maturity) >= prune_mat_threshold)
        # Φ-gated BDNF: high integration states release more trophic factor.
        # Fiedler-gated homeostasis: when spectral gap is small (graph close
        # to disconnected), release extra BDNF to rewire the fault line.
        fiedler_boost = max(0.0, 1.0 - fiedler / 0.3) * 2.0   # large when λ₁ < 0.3
        bdnf_phi = bdnf * (1.0 + self.phi_boost * max(0.0, phi) + fiedler_boost)
        # Scale neurotrophin signals so they don't overwhelm the dynamics.
        bdnf = max(0.0, min(0.05, bdnf_phi * 0.05))
        ngf  = max(0.0, min(0.01, ngf  * 0.01))
        self._steps += 1
        for i, p in enumerate(self.graph.projections):
            a = activities.get(p.src)
            b = activities.get(p.dst)
            if a is None or b is None:
                co = 0.0
            else:
                co = float((a * b).mean().clamp(0.0, 1.0))
            self.ema_coact[i] = 0.95 * self.ema_coact[i] + 0.05 * co
            growth = (bdnf + self.bdnf_baseline) * (0.1 + self.ema_coact[i])
            decay  = ngf + self.ngf_decay + 0.001 * (1.0 - self.ema_coact[i])
            new = (self.trophic[i] + growth - decay).clamp(0.0, 1.0)
            self.trophic[i] = new
            if _allow_prune and new < self.prune_threshold:
                self.active[i] = 0.0
            elif new > self.prune_threshold * 2.0 and self.active[i] == 0.0:
                self.active[i] = 1.0

        # Homeostatic sprouting: when many edges saturate, attempt to grow new
        # connections between high-activity, currently-disconnected modules.
        # Triggered when at least 25% of projections are above sprout_threshold.
        n_total = self.trophic.numel()
        n_saturated = int((self.trophic > self.sprout_threshold).sum().item())
        if n_total > 0 and n_saturated / n_total >= 0.25:
            self._try_sprout(activities)

    def gain(self, idx: int) -> float:
        """Multiplicative gain to apply to projection idx's signal map."""
        return float(self.active[idx] * (0.2 + 1.6 * self.trophic[idx]))

    def stats(self) -> dict:
        return {
            "n_projections": int(self.active.numel()),
            "n_active":      int(self.active.sum().item()),
            "n_pruned":      int((self.active == 0).sum().item()),
            "trophic_mean":  float(self.trophic.mean().item()),
            "trophic_max":   float(self.trophic.max().item()),
            "trophic_min":   float(self.trophic.min().item()),
            "saturated":     int((self.trophic > self.sprout_threshold).sum().item()),
        }
