"""Mixture of Depths + CALM early exit — dynamic per-token compute allocation.

MoD (Raposo et al. 2024): easy tokens skip layers via a learned router.
CALM (Schuster et al. 2022): confident tokens exit the entire stack early,
  carrying their frozen hidden state through remaining layers.

Combined effect:
  - MoD:  per-layer token routing (spatial compute allocation)
  - CALM: cross-layer early exit (depth compute allocation)
  - Net:  hard tokens get full depth + full width; trivial tokens get neither

CALM threshold decays with layer depth:
  θ_l = θ_base × exp(−decay × l / (L−1))
Easy layers (l ≈ 0) have high θ (almost never exit); deep layers have lower θ
so tokens that remain uncertain deep in the stack still have a chance to exit.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CALM: Confident Adaptive Language Modeling early-exit head
# ---------------------------------------------------------------------------

class CALMHead(nn.Module):
    """Per-token confidence estimator for CALM early exit.

    Predicts how "done" each token position is after this layer.
    Zero-initialised so no exits happen at the start of training;
    the network learns when exiting is safe.

    Threshold schedule: θ_l = θ_base × exp(−decay × l/(L−1))
    High threshold at shallow layers (almost never exit early) → decays
    so tokens still undecided at deep layers can still exit.
    """

    def __init__(self, dim: int, base_threshold: float = 0.9, decay: float = 2.0):
        super().__init__()
        hidden = max(16, dim // 16)
        self.head = nn.Sequential(
            nn.Linear(dim, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, 1, bias=True),
        )
        self.base_threshold = base_threshold
        self.decay = decay
        nn.init.zeros_(self.head[0].weight)
        nn.init.zeros_(self.head[2].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → confidence: (B, T) ∈ [0, 1]"""
        return torch.sigmoid(self.head(x).squeeze(-1))

    def threshold(self, layer_idx: int, n_layers: int) -> float:
        if n_layers <= 1:
            return self.base_threshold
        return self.base_threshold * math.exp(
            -self.decay * layer_idx / max(n_layers - 1, 1))


class MoDRouter(nn.Module):
    """Per-layer router: 2-layer MLP scores token hardness for top-k selection.

    Hard tokens (novel concepts, reasoning steps) receive full layer depth.
    Easy tokens (function words, predictable continuations) skip via residual.
    The MLP router captures non-linear token hardness better than a linear probe.
    """

    def __init__(self, dim: int, capacity_ratio: float = 0.5, n_nt: int = 0):
        super().__init__()
        self.capacity_ratio = capacity_ratio
        # 2-layer MLP: captures token hardness (local context, surprisal proxy)
        hidden = max(32, dim // 8)
        self.router = nn.Sequential(
            nn.Linear(dim, hidden, bias=True),
            nn.SiLU(),
            nn.Linear(hidden, 1, bias=True),
        )
        # Zero-init so routing starts uniform (all tokens score 0)
        nn.init.zeros_(self.router[0].weight)
        nn.init.zeros_(self.router[2].weight)
        nn.init.zeros_(self.router[2].bias)

        if n_nt > 0:
            self.nt_capacity = nn.Linear(n_nt, 1, bias=False)
            nn.init.zeros_(self.nt_capacity.weight)
        else:
            self.nt_capacity = None

    def forward(self, x: torch.Tensor,
                nt: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (selected_x, indices, router_logits).

        selected_x: (B, C, D) — top-C hard tokens
        indices:    (B, C)    — sorted positions (causal order preserved)
        router_logits: (B, T, 1) — raw scores for aux loss
        """
        B, T, D = x.shape
        router_logits = self.router(x)  # (B, T, 1)

        cap = self.capacity_ratio
        if self.nt_capacity is not None and nt is not None:
            nt_mod = torch.sigmoid(self.nt_capacity(nt))   # (B, 1)
            cap = cap * (0.5 + nt_mod.squeeze(-1).mean().item())
            cap = float(min(1.0, max(0.1, cap)))

        C = max(1, int(T * cap))
        scores = router_logits.squeeze(-1)                  # (B, T)
        _, topk_idx = scores.topk(C, dim=-1, sorted=False)
        topk_idx, _ = topk_idx.sort(dim=-1)                 # preserve causal order
        selected_x = x.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, D))
        return selected_x, topk_idx, router_logits


class MoDBlock(nn.Module):
    """Transformer block with Mixture-of-Depths routing + CALM early-exit head.

    Only top-C tokens (by router score) pass through the attention+MLP.
    Remaining tokens get identity (residual passthrough).
    CALM head provides per-token confidence scores for cross-layer early exit
    (managed by LanguageCortex.forward, not by this block directly).
    """

    def __init__(self, dim: int, n_heads: int, max_ctx: int,
                 n_kv_heads: int | None = None,
                 n_nt: int = 0, capacity_ratio: float = 0.5,
                 use_diff_attn: bool = True):
        super().__init__()
        from .common import SwiGLU, RMSNorm
        self.router = MoDRouter(dim, capacity_ratio, n_nt)
        self.calm_head = CALMHead(dim)          # CALM early-exit confidence

        if use_diff_attn:
            from .differential_attention import DifferentialAttention
            self.attn = DifferentialAttention(dim, n_heads, max_ctx, n_kv_heads, n_nt)
        else:
            from .common import CausalSelfAttention
            self.attn = CausalSelfAttention(dim, n_heads, max_ctx, n_kv_heads, n_nt=n_nt)

        self.n1 = RMSNorm(dim)
        self.n2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim)
        self._base_capacity_ratio = capacity_ratio
        # Maturity-aware compute gating: set externally by Brain/Language before
        # each forward pass. 1.0 = full decoder capacity (legacy). When < 0.2
        # the block becomes a pure passthrough to save FLOPs while the network
        # is still language-bootstrapping. Plain Python float → XLA-static.
        self._maturity_gate: float = 1.0

    def forward(self, x: torch.Tensor,
                nt: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape

        # ── Maturity-aware compute skip ────────────────────────────────────
        # If MAT < 0.2 the expert layers add ~random noise; cheaper and safer
        # to passthrough until the LM head has stabilised. This is a static
        # Python branch (maturity is set outside the trace), so XLA is happy.
        if self._maturity_gate < 0.2:
            return x

        # Maturity also softly modulates the router capacity so deeper layers
        # come online gradually (0.2..1.0 maps to 0.5..1.0× of base capacity).
        # The router itself uses the value via the attribute below.
        m = self._maturity_gate
        eff_cap_scale = 0.5 + 0.5 * min(1.0, max(0.0, (m - 0.2) / 0.8))
        # Temporarily override capacity_ratio for this call only
        _orig_cap = self.router.capacity_ratio
        self.router.capacity_ratio = _orig_cap * eff_cap_scale

        # Route: select which tokens get processed
        selected_x, indices, router_logits = self.router(x, nt)
        # Restore capacity_ratio for any external readers / aux loss
        self.router.capacity_ratio = _orig_cap
        C = selected_x.size(1)

        if C == T:
            # All tokens selected — standard path
            x = x + self.attn(self.n1(x), nt=nt)
            x = x + self.mlp(self.n2(x))
            return x

        # Process only selected tokens
        h = self.attn(self.n1(selected_x), nt=nt)
        h = h + self.mlp(self.n2(selected_x + h))

        # Scatter back: selected tokens get residual + output; others get identity
        out = x.clone()
        out.scatter_(1, indices.unsqueeze(-1).expand(-1, -1, D), selected_x + h)

        # Store router logits for auxiliary load-balancing loss
        self._last_router_logits = router_logits

        return out

    @property
    def router_aux_loss(self) -> torch.Tensor:
        """Load-balancing loss: keeps mean routing fraction ≈ capacity_ratio.
        Also penalises variance to prevent batch-level routing collapse."""
        logits = getattr(self, '_last_router_logits', None)
        if logits is None:
            return torch.tensor(0.0)
        probs = torch.sigmoid(logits.squeeze(-1))       # (B, T)
        mean_prob = probs.mean(dim=-1)                  # (B,)
        target = self.router.capacity_ratio if hasattr(self.router, 'capacity_ratio') \
                 else 0.5
        mean_loss = ((mean_prob - target) ** 2).mean()
        # Entropy bonus: reward routing diversity within each batch item
        entropy = -(probs * (probs + 1e-8).log() +
                    (1 - probs) * (1 - probs + 1e-8).log()).mean()
        return mean_loss - 0.01 * entropy
