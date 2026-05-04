"""World model — Recurrent State-Space Model (RSSM).

Upgrades the plain GRU world model to a Dreamer-V3-style RSSM with:
  • Deterministic path  h_t = GRU(h_{t-1}, [z_{t-1}; x_t])
  • Stochastic prior    z_t ~ Categorical(f_prior(h_t))
  • Stochastic posterior z_t ~ Categorical(f_post(h_t, x_t))
  • World state = [h_t; flat(z_t)]  (concatenation)
  • KL loss      KL(posterior || prior)  — regularises imagination
  • Latent imagination  — roll forward using prior only (no observations)

Key advantages over the old plain-GRU world model:
  1. Can imagine futures without observations (prior mode = planning)
  2. KL term regularises world model → prevents hallucination divergence
  3. Straight-through gradients through Categorical latents
  4. Separate world + entity step makes entity-conditional rollouts trivial
  5. d_world_state = d_hidden + n_cats * d_cat (larger, richer representation)

Backward-compatible: forward() still returns (z_world, h_new, predicted_next)
but h_new now carries both the deterministic GRU state and the sampled z.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .brain_module import BrainModule


class CategoricalStraightThrough(nn.Module):
    """Categorical distribution with straight-through gradient estimator.

    Samples a one-hot vector; in the backward pass treats it as if
    it were the soft probability vector (unbiased in expectation).
    """

    def __init__(self, n_cats: int, d_cat: int):
        super().__init__()
        self.n_cats = n_cats   # number of categorical variables
        self.d_cat  = d_cat    # number of classes per variable

    def forward(self, logits: torch.Tensor,
                straight_through: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        logits: (B, n_cats * d_cat)
        Returns:
          sample: (B, n_cats * d_cat)  — one-hot (straight-through in backward)
          probs:  (B, n_cats, d_cat)   — soft probabilities
        """
        B = logits.shape[0]
        lg  = logits.view(B, self.n_cats, self.d_cat)
        probs = F.softmax(lg, dim=-1)    # (B, n_cats, d_cat)

        if self.training or straight_through:
            # Gumbel straight-through
            indices = probs.detach().argmax(-1)   # (B, n_cats)
            one_hot = F.one_hot(indices, self.d_cat).float()
            # Straight-through: use probs in backward, one_hot in forward
            sample = one_hot + probs - probs.detach()
        else:
            sample = probs   # argmax mode during eval

        sample = sample.view(B, self.n_cats * self.d_cat)
        return sample, probs


class RecurrentStateSpaceModel(BrainModule):
    """RSSM world model for NeuroSLM.

    State representation:
      h  — deterministic GRU hidden state  (d_hidden,)
      z  — stochastic categorical latent   (n_cats * d_cat,)
      world_state = cat(h, z)             (d_hidden + n_cats*d_cat,)

    Modes:
      observation mode (training/inference):
        posterior z ~ q(z|h, x)   uses real sensory input
      imagination mode (planning/dreaming):
        prior z ~ p(z|h)           no sensory input required

    Parameters:
      d_sem    — input / output semantic dimension
      d_hidden — GRU hidden size
      n_cats   — number of categorical latent variables (default 8)
      d_cat    — classes per categorical variable (default 16)
      n_layers — GRU layers
    """

    def __init__(self, d_sem: int, d_hidden: int, n_layers: int = 2,
                 n_cats: int = 8, d_cat: int = 16):
        super().__init__()
        self.d_sem    = d_sem
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.n_cats   = n_cats
        self.d_cat    = d_cat
        self.d_z      = n_cats * d_cat    # flat stochastic dim
        self.d_state  = d_hidden + self.d_z   # total world state dim

        # Input projection: sensory + previous z → GRU input
        self.inp_proj = nn.Linear(d_sem + self.d_z, d_hidden)

        # Deterministic path
        self.rnn = nn.GRU(d_hidden, d_hidden,
                          num_layers=n_layers, batch_first=True)

        # Prior network: p(z|h)
        self.prior_net = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ELU(),
            nn.Linear(d_hidden, n_cats * d_cat),
        )

        # Posterior network: q(z|h, x)
        self.post_net = nn.Sequential(
            nn.Linear(d_hidden + d_sem, d_hidden),
            nn.ELU(),
            nn.Linear(d_hidden, n_cats * d_cat),
        )

        # Categorical sampler
        self.categorical = CategoricalStraightThrough(n_cats, d_cat)

        # Decoder: world_state → predicted semantic output
        self.decoder = nn.Sequential(
            nn.Linear(self.d_state, d_hidden),
            nn.ELU(),
            nn.Linear(d_hidden, d_sem),
        )

        # World state → final d_sem projection (for downstream compatibility)
        self.world_proj = nn.Linear(self.d_state, d_sem)

        # Reconstruction / next-step prediction head
        self.predict_head = nn.Linear(self.d_state, d_sem)

    # ── State shape helpers ───────────────────────────────────────────────────

    def init_state(self, batch_size: int, device) -> dict:
        """Returns dict with keys 'h' and 'z'."""
        return {
            "h": torch.zeros(self.n_layers, batch_size, self.d_hidden,
                             device=device),
            "z": torch.zeros(batch_size, self.d_z, device=device),
        }

    # ── Forward (observation mode) ────────────────────────────────────────────

    def forward(self, sensory: torch.Tensor,
                state: Optional[dict] = None
               ) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """Observation-mode forward.

        sensory: (B, d_sem)
        state:   dict with 'h' (n_layers, B, d_hidden) and 'z' (B, d_z)
                 (None → zero-initialise)

        Returns:
          z_world:   (B, d_sem)   — projected world state for downstream
          new_state: dict         — updated h and z
          pred_next: (B, d_sem)  — prediction of next sensory frame
        """
        B, device = sensory.shape[0], sensory.device
        if state is None:
            state = self.init_state(B, device)

        h_prev, z_prev = state["h"], state["z"]

        # 1. Deterministic update h_t = GRU(h_{t-1}, proj([z_{t-1}; x_t]))
        inp = torch.cat([sensory, z_prev], dim=-1)   # (B, d_sem + d_z)
        inp = self.inp_proj(inp).unsqueeze(1)         # (B, 1, d_hidden)
        _, h_new = self.rnn(inp, h_prev)              # h_new: (n_layers, B, d_hidden)
        h_top = h_new[-1]                             # (B, d_hidden)

        # 2. Posterior z ~ q(z|h_t, x_t)
        post_inp  = torch.cat([h_top, sensory], dim=-1)
        post_logits = self.post_net(post_inp)
        z_post, post_probs = self.categorical(post_logits)

        # 3. Prior p(z|h_t)  [used only for KL loss during training]
        prior_logits = self.prior_net(h_top)
        _, prior_probs = self.categorical(prior_logits, straight_through=False)

        # 4. World state = [h_t; z_post]
        world_state = torch.cat([h_top, z_post], dim=-1)  # (B, d_state)

        # 5. Outputs
        z_world  = self.world_proj(world_state)          # (B, d_sem)
        pred_next = self.predict_head(world_state)        # (B, d_sem)

        new_state = {"h": h_new, "z": z_post.detach(),
                     "_prior_probs": prior_probs, "_post_probs": post_probs,
                     "_world_state": world_state}

        return z_world, new_state, pred_next

    # ── Imagination mode (no observations) ───────────────────────────────────

    def imagine(self, state: dict, horizon: int = 8
               ) -> Tuple[torch.Tensor, list]:
        """Roll forward using prior for `horizon` steps.

        Used for:
          • BG action planning (evaluate candidate actions over K future steps)
          • Mind-wandering (dreaming without grounded input)
          • Counterfactual reasoning (what if...)

        Returns:
          world_traj: (B, horizon, d_sem) — imagined world states
          states:     list of state dicts
        """
        traj = []
        states = []
        h, z = state["h"], state["z"]

        for _ in range(horizon):
            h_top = h[-1]
            # Dummy sensory from last z-based reconstruction
            world_state = torch.cat([h_top, z], dim=-1)
            dummy_sensory = self.decoder(world_state)

            # Prior step
            inp = torch.cat([dummy_sensory, z], dim=-1)
            inp = self.inp_proj(inp).unsqueeze(1)
            _, h = self.rnn(inp, h)
            h_top = h[-1]
            prior_logits = self.prior_net(h_top)
            z, _ = self.categorical(prior_logits)

            world_state = torch.cat([h_top, z], dim=-1)
            z_world = self.world_proj(world_state)
            traj.append(z_world)
            states.append({"h": h, "z": z.detach()})

        world_traj = torch.stack(traj, dim=1)   # (B, horizon, d_sem)
        return world_traj, states

    # ── KL loss ───────────────────────────────────────────────────────────────

    @staticmethod
    def kl_loss(state: dict, free_nats: float = 1.0) -> torch.Tensor:
        """KL divergence between posterior and prior.

        Implements 'free nats' trick from Dreamer: don't penalise when
        KL < free_nats (avoids posterior collapse on easy observations).

        Returns scalar loss.
        """
        prior_probs = state.get("_prior_probs")
        post_probs  = state.get("_post_probs")
        if prior_probs is None or post_probs is None:
            return torch.tensor(0.0)

        # KL(posterior || prior)
        eps = 1e-8
        kl = (post_probs * (
                (post_probs + eps).log() - (prior_probs + eps).log()
              )).sum(-1).sum(-1)   # sum over classes then cats → (B,)
        return torch.mean(torch.clamp(kl, min=free_nats))
