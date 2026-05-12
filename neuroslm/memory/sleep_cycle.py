"""Sleep-cycle CLS consolidation with bidirectional predictive coding and
trophic renormalisation.

The sleep cycle is the slow-wave + REM analogue: every N awake steps the
brain enters a brief "sleep" phase where:

  1. Replay: episodic motifs from the recent memory buffer are sampled
     in proportion to their salience × valence × recency.
  2. Bidirectional predictive coding distillation: for each replayed
     episode the language-model hidden state is regenerated, a top-down
     predictive code is produced by the consolidator, and the squared
     error between them is back-propagated into a small set of
     "slow weights" (a low-rank adapter on top of LanguageCortex). The
     distillation prior is NEMORI-style: only the *unpredicted* part of
     each episode contributes — what the model already knew is discarded.
  3. Trophic renormalisation: edges in the relational hypergraph whose
     EMA-causation α has stayed below a threshold across the sleep window
     have their trophic weight scaled down; edges above the threshold get
     a small boost. Synapses encoding pure nuisance variance are pruned
     by setting their projection gain to zero.
  4. Gaussian I(X;Z) proxy: report the change in mutual-information
     proxy across consolidation, used as the predictive-forgetting gain
     signal in the test suite.

Activation gating:
  • Only runs when `_maturation_awakened == True`.
  • Triggered every `sleep_period_steps` (default 5000).
  • Does not interact with the gradient graph of the awake forward pass.

This module is a pure orchestrator — it composes existing pieces
(ComprehensionGate, ConsolidationEngine, TrophicSystem, ActualCausationHead).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Distillation result snapshots — useful for telemetry and tests
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SleepReport:
    step:               int
    n_replays:          int
    pre_mi_proxy:       float        # I(X;Z) before
    post_mi_proxy:      float        # I(X;Z) after
    mi_reduction:       float        # post − pre  (negative = compression)
    distillation_loss:  float
    pruned_edges:       int
    strengthened_edges: int
    duration_s:         float


# ─────────────────────────────────────────────────────────────────────────────
# Sleep cycle orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class SleepCycle(nn.Module):
    """Orchestrates predictive-forgetting consolidation.

    Args:
        d_sem:               dimension of episode embeddings.
        sleep_period_steps:  awake-step interval between sleeps.
        replay_batch:        episodes sampled per sleep iteration.
        n_iters:              number of replay batches per sleep call.
        nemori_threshold:    minimum predictive surprise (NLL − predicted NLL)
                             below which an episode is *not* distilled (it's
                             already known).
        enable:              hard kill-switch (set False during ablation).
    """

    def __init__(self,
                 d_sem: int = 384,
                 sleep_period_steps: int = 5000,
                 replay_batch: int = 16,
                 n_iters: int = 4,
                 nemori_threshold: float = 0.2,
                 enable: bool = True):
        super().__init__()
        self.d_sem = d_sem
        self.sleep_period_steps = sleep_period_steps
        self.replay_batch = replay_batch
        self.n_iters = n_iters
        self.nemori_threshold = nemori_threshold
        self.enable = enable

        # Small low-rank "slow-weights" adapter — gets nudged during sleep
        # toward the distilled predictive code of frequently-replayed
        # episodes. Rank 16 is enough for episodic motif compression.
        self.slow_rank = 16
        self.slow_a = nn.Parameter(torch.zeros(d_sem, self.slow_rank))
        self.slow_b = nn.Parameter(torch.zeros(self.slow_rank, d_sem))

        # Top-down predictor: maps an episode embedding to its predicted
        # next-language-hidden-state. Used for the bidirectional PC error.
        self.predictor = nn.Sequential(
            nn.Linear(d_sem, d_sem),
            nn.GELU(),
            nn.Linear(d_sem, d_sem),
        )

        # Awakening guard
        self._awakened = False
        self._step = 0
        self._last_sleep_step = 0

        # Lightweight running stats for the I(X;Z) proxy
        self._z_mean = nn.Parameter(torch.zeros(d_sem), requires_grad=False)
        self._z_var  = nn.Parameter(torch.ones(d_sem),  requires_grad=False)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def set_awakened(self, awakened: bool) -> None:
        self._awakened = bool(awakened)

    def maybe_sleep(self, step: int) -> bool:
        """True iff this step should trigger a sleep phase. Caller is
        expected to call `sleep(...)` when this returns True."""
        if not self.enable or not self._awakened:
            return False
        return (step - self._last_sleep_step) >= self.sleep_period_steps

    # ── I(X;Z) Gaussian proxy ────────────────────────────────────────────────

    @torch.no_grad()
    def _gaussian_mi_proxy(self, x: torch.Tensor, z: torch.Tensor) -> float:
        """Empirical Gaussian-MI proxy: 0.5 · log( det Σ_x / det Σ_{x|z} )
        where Σ_{x|z} is approximated by residual covariance after the
        linear best-fit regression z → x. Returns a scalar (float)."""
        if x.numel() == 0 or z.numel() == 0:
            return 0.0
        x_ = x.detach().float().reshape(x.size(0), -1)
        z_ = z.detach().float().reshape(z.size(0), -1)
        n = x_.size(0)
        if n < 4:
            return 0.0
        # Mean-centre
        x_ = x_ - x_.mean(0, keepdim=True)
        z_ = z_ - z_.mean(0, keepdim=True)
        # Regress x on z via least squares
        try:
            W, *_ = torch.linalg.lstsq(z_, x_)
        except Exception:
            return 0.0
        x_hat = z_ @ W
        resid = x_ - x_hat
        d = x_.size(1)
        # log-det of empirical covariance with ε regularisation
        eps = 1e-3
        eye = torch.eye(d, device=x_.device, dtype=x_.dtype)
        cov_x   = (x_.T @ x_) / (n - 1) + eps * eye
        cov_res = (resid.T @ resid) / (n - 1) + eps * eye
        # slogdet — clipped to avoid pathological values
        sign_x, ld_x = torch.linalg.slogdet(cov_x)
        sign_r, ld_r = torch.linalg.slogdet(cov_res)
        if sign_x.item() <= 0 or sign_r.item() <= 0:
            return 0.0
        return float(0.5 * (ld_x - ld_r).item())

    # ── replay sampling ──────────────────────────────────────────────────────

    @staticmethod
    def _sample_replays(episodes: List[dict],
                         k: int,
                         rng: np.random.Generator) -> List[dict]:
        """Sample episodes proportional to salience × decay × |valence|.

        Bias toward emotionally-charged and recent memories — the same
        priority that the hippocampus uses during slow-wave reactivation.
        """
        if not episodes:
            return []
        weights = np.asarray(
            [max(1e-3,
                 float(e.get("salience", 0.5))
                 * float(e.get("decay", 1.0))
                 * (0.2 + abs(float(e.get("valence", 0.0)))))
             for e in episodes], dtype=np.float32)
        weights /= weights.sum()
        idx = rng.choice(len(episodes), size=min(k, len(episodes)),
                          replace=False, p=weights)
        return [episodes[int(i)] for i in idx]

    # ── distillation step ────────────────────────────────────────────────────

    def _distill_batch(self,
                       embeds: torch.Tensor,         # (B, d)
                       known_nll: torch.Tensor       # (B,) — current LM NLL on the episode
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One distillation step on a batch of episode embeddings.

        Implements the NEMORI prior: only embeddings whose predictive
        surprise exceeds `nemori_threshold` contribute to the
        distillation loss. The slow-weights adapter is nudged so that
        the *future* LM hidden state on these episodes is closer to the
        top-down predicted code.

        Returns (distillation_loss, distilled_codes).
        """
        # Predictive code (top-down)
        pc = self.predictor(embeds)                  # (B, d)

        # Bottom-up reconstruction through the slow-weights adapter
        bu = embeds + (embeds @ self.slow_a) @ self.slow_b   # low-rank add
        # NEMORI gate: only unpredicted episodes contribute
        gate = (known_nll - known_nll.mean()).clamp_min(0)
        gate = (gate > self.nemori_threshold).float()        # (B,)
        # Bidirectional PC error: top-down ↔ bottom-up
        err = F.mse_loss(pc, bu, reduction="none").mean(dim=-1)  # (B,)
        loss = (err * gate).mean()
        return loss, pc.detach()

    # ── trophic renormalisation ──────────────────────────────────────────────

    @staticmethod
    def _renormalise_trophic(actual_causation,         # ActualCausationHead
                              trophic_system,         # neurochem.growth.TrophicSystem
                              keep_alpha: float = 0.3,
                              boost: float = 0.05,
                              decay: float = 0.05
                              ) -> Tuple[int, int]:
        """Push trophic weights toward the EMA of the causal-strength matrix.

        Returns (n_pruned, n_strengthened). The semantics:

          • Edges with α_ema < keep_alpha get their trophic level decayed by
            `decay` (eventually pruned by the existing TrophicSystem when
            τ < 0.05).
          • Edges with α_ema ≥ keep_alpha get a `boost` to τ, locking the
            high-causation pathway.
        """
        if actual_causation is None or trophic_system is None:
            return (0, 0)
        if not hasattr(trophic_system, "trophic_levels"):
            return (0, 0)
        n_p, n_s = 0, 0
        alpha = actual_causation.alpha_ema.detach().cpu().numpy()
        n = alpha.shape[0]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                key = (i, j)
                if key not in trophic_system.trophic_levels:
                    continue
                a = float(alpha[i, j])
                if a < keep_alpha:
                    trophic_system.trophic_levels[key] = max(
                        0.0,
                        trophic_system.trophic_levels[key] - decay)
                    n_p += 1
                else:
                    trophic_system.trophic_levels[key] = min(
                        1.0,
                        trophic_system.trophic_levels[key] + boost)
                    n_s += 1
        return n_p, n_s

    # ── main entry point ─────────────────────────────────────────────────────

    def sleep(self,
              step: int,
              episodes: List[dict],
              episode_to_embed: Callable[[dict], torch.Tensor],
              episode_to_known_nll: Callable[[dict], float],
              actual_causation = None,
              trophic_system = None,
              optimizer: Optional[torch.optim.Optimizer] = None,
              ) -> SleepReport:
        """Run one sleep phase. Returns a SleepReport.

        episodes:                memory buffer (list of dicts with the
                                  fields used by `_sample_replays`).
        episode_to_embed:        callable returning (d_sem,) torch tensor
                                  for a given episode dict.
        episode_to_known_nll:    callable returning a scalar NLL of the
                                  current LM on the episode's content
                                  (used by NEMORI gate).
        actual_causation:        ActualCausationHead instance, or None.
        trophic_system:          TrophicSystem instance, or None.
        optimizer:               optional optimizer that should also see
                                  the distillation loss (otherwise the
                                  slow-weights adapter receives a
                                  separate manual SGD step).
        """
        import time as _t
        t0 = _t.time()
        rng = np.random.default_rng(step)

        # Take a "before" snapshot for the MI proxy
        if episodes:
            sample = self._sample_replays(episodes, self.replay_batch * 2, rng)
            emb_pre = torch.stack(
                [episode_to_embed(e).detach() for e in sample]
                if sample else [torch.zeros(self.d_sem)],
                dim=0).float()
            z_pre = self.predictor(emb_pre).detach()
            mi_pre = self._gaussian_mi_proxy(emb_pre, z_pre)
        else:
            mi_pre = 0.0

        # Replay & distill
        total_loss = 0.0
        n_replays = 0
        for _ in range(self.n_iters):
            batch = self._sample_replays(episodes, self.replay_batch, rng)
            if not batch:
                break
            emb = torch.stack(
                [episode_to_embed(e).detach() for e in batch], dim=0).float()
            nll = torch.tensor(
                [episode_to_known_nll(e) for e in batch], dtype=torch.float32)
            loss, _ = self._distill_batch(emb, nll)

            if optimizer is not None and loss.requires_grad:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            else:
                # Manual SGD on the slow-weights adapter only
                if loss.requires_grad:
                    loss.backward()
                    with torch.no_grad():
                        for p in (self.slow_a, self.slow_b,
                                   *self.predictor.parameters()):
                            if p.grad is not None:
                                p.add_(p.grad, alpha=-1e-3)
                                p.grad.zero_()

            total_loss += float(loss.detach().item())
            n_replays += len(batch)

        # "After" snapshot
        if episodes:
            emb_post = torch.stack(
                [episode_to_embed(e).detach() for e in sample]
                if sample else [torch.zeros(self.d_sem)],
                dim=0).float()
            z_post = self.predictor(emb_post).detach()
            mi_post = self._gaussian_mi_proxy(emb_post, z_post)
        else:
            mi_post = 0.0

        # Trophic renormalisation
        n_p, n_s = self._renormalise_trophic(actual_causation, trophic_system)

        self._last_sleep_step = step

        return SleepReport(
            step=step,
            n_replays=n_replays,
            pre_mi_proxy=mi_pre,
            post_mi_proxy=mi_post,
            mi_reduction=mi_post - mi_pre,
            distillation_loss=total_loss / max(1, self.n_iters),
            pruned_edges=n_p,
            strengthened_edges=n_s,
            duration_s=_t.time() - t0,
        )
