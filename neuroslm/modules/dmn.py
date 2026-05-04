"""Default Mode Network: routing orchestrator + associative query generator.

The DMN serves two roles in this architecture:

1. **Slow scheduler / query generator** (~every 4 sensory ticks):
   - Integrates GWS slots and floating thought into an associative recall query
   - Drives hippocampal enrichment each cycle
   - Produces a stop signal: high = model should speak, low = keep thinking

2. **Topology controller** (via `topology` attribute):
   - 'baseline': only language cortex is active (vanilla transformer path)
   - 'full':     routes neural flow through all brain modules in order

Topology 'full' wiring (controlled by brain.py using this module's routing config):
   language → sensory → association → thalamus → GWS ← [world, self, thought]
   GWS → DMN (query) → hippocampus (enrichment) → GWS (enriched)
   GWS (enriched) → PFC (selection, NT-scored) → basal_ganglia (forward sim)
   basal_ganglia → forward_model → critic → evaluator → motor (if commit_ok)
   motor_bias → language (second pass) → output tokens

Mind-wandering: when no input is forced and the stop signal is low, the DMN
drives internal associations — the floating thought drifts through the memory
graph, modulated by qualia/NT state.

References:
  Buckner et al. (2008) The brain's default network.
  Raichle (2015) The brain's default mode network.
  Andrews-Hanna (2012) The brain's default network and its role in cognition.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .brain_module import BrainModule


# Routing order for the 'full' topology (used by brain.py as documentation / contract)
FULL_ROUTING_ORDER = [
    "language",          # 1. token embedding + transformer blocks
    "sensory",           # 2. semantic → sensory representation
    "association",       # 3. multi-modal binding
    "thalamus",          # 4. content-aware MoE routing gate
    "world_model",       # 5. recurrent world state prediction
    "self_model",        # 6. self-state encoding
    "critic",            # 7. fast threat detection (pre-GWS)
    "gws",               # 8. global workspace broadcast
    "dmn_query",         # 9. DMN: generates recall query (this module)
    "hippocampus",       # 10. multi-dimensional memory enrichment
    "pfc",               # 11. NT-driven executive selection
    "thought_transformer",# 12. reasoning amplification
    "basal_ganglia",     # 13. forward simulation + action selection
    "forward_model",     # 14. predict action outcome
    "evaluator",         # 15. value estimation
    "motor",             # 16. discrete action + language bias (if commit_ok)
    "language_output",   # 17. motor-conditioned logits
]

BASELINE_ROUTING_ORDER = ["language"]   # only language cortex


class DefaultModeNetwork(BrainModule):
    def __init__(self, d_sem: int, n_slots: int, n_layers: int,
                 topology: str = "full"):
        super().__init__()
        self.d_sem     = d_sem
        self.n_slots   = n_slots
        self.topology  = topology   # 'baseline' or 'full'
        self._tick     = 0          # sensory tick counter

        # Input: flattened GWS slots + floating thought
        in_dim = d_sem * n_slots + d_sem
        layers = []
        cur = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(cur, d_sem * 2), nn.GELU()]
            cur = d_sem * 2
        self.mlp = nn.Sequential(*layers)

        # Query head: drives hippocampal recall
        self.query_head = nn.Linear(cur, d_sem)

        # Stop head: when to switch from thinking to speaking
        # Positive → keep thinking; negative → emit output
        self.stop_head  = nn.Linear(cur, 1)

        # NT-modulated query gate: ACh + 5HT suppress, DA boosts
        self.nt_gate    = nn.Linear(7, 1)   # 7 NTs → scalar gate

        # Mind-wandering drift: adds stochastic exploration to query
        # when mind is not focused (low NE, low ACh = diffuse mode)
        self.drift_proj = nn.Linear(d_sem, d_sem)

    # ------------------------------------------------------------------
    # Routing helpers (used by brain.py to decide what to run)
    # ------------------------------------------------------------------
    @property
    def routing_order(self) -> list[str]:
        return (FULL_ROUTING_ORDER if self.topology == "full"
                else BASELINE_ROUTING_ORDER)

    def is_full(self) -> bool:
        return self.topology == "full"

    def use_baseline(self):
        self.topology = "baseline"

    def use_full(self):
        self.topology = "full"

    # ------------------------------------------------------------------
    # Forward: generate recall query + stop signal
    # ------------------------------------------------------------------
    def forward(self, gws_slots: torch.Tensor,
                floating_thought: torch.Tensor,
                nt_levels: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        gws_slots:       (B, S, d_sem)
        floating_thought:(B, d_sem)
        nt_levels:       dict of NT scalars (optional)

        Returns:
            query      (B, d_sem) — query for hippocampal recall
            stop_logit (B,)       — >0 means "keep thinking", <0 means "emit"
        """
        if nt_levels is None:
            nt_levels = {}
        B = gws_slots.size(0)
        self._tick += 1

        x = torch.cat([gws_slots.reshape(B, -1), floating_thought], dim=-1)
        h = self.mlp(x)

        query     = self.query_head(h)    # (B, d_sem)
        stop_logit = self.stop_head(h).squeeze(-1)   # (B,)

        # NT gate: ACh / 5HT suppress associative wandering; NE + DA boost it
        nt_vec = torch.tensor([
            nt_levels.get("DA",   0.5),
            nt_levels.get("NE",   0.5),
            nt_levels.get("5HT",  0.5),
            nt_levels.get("ACh",  0.5),
            nt_levels.get("eCB",  0.5),
            nt_levels.get("Glu",  0.5),
            nt_levels.get("GABA", 0.5),
        ], device=gws_slots.device, dtype=gws_slots.dtype).unsqueeze(0).expand(B, -1)

        nt_gate = torch.sigmoid(self.nt_gate(nt_vec))   # (B, 1)
        query   = query * nt_gate                        # NT-modulated strength

        # Mind-wandering: low NE + low ACh → add random drift (diffuse mode)
        ne  = nt_levels.get("NE",  0.5)
        ach = nt_levels.get("ACh", 0.5)
        if ne < 0.35 and ach < 0.35:
            drift = torch.randn_like(query) * 0.15
            query = query + self.drift_proj(drift)

        return query, stop_logit

    def _disabled_output(self, gws_slots, floating_thought, *_, **__):
        B = gws_slots.size(0)
        return floating_thought, torch.zeros(B, device=gws_slots.device)
