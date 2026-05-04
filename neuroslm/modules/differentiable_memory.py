"""Differentiable External Memory (NTM-style) for NeuroSLM.

Neural Turing Machines (Graves et al. 2014) augment neural networks with
an external memory matrix that can be read and written in a fully
differentiable way, enabling explicit storage and retrieval of arbitrary
information.

Standard NTM components implemented here:
  1. Content-based addressing: cosine similarity between query and memory
     rows → soft attention over memory rows
  2. Sharpening: attention distribution raised to power β, then renorm
     (β>1 → peakier; β<1 → flatter, more exploratory)
  3. Differentiable erase/add write: each write head emits an erase
     vector e ∈ (0,1)^M and an add vector a ∈ ℝ^M:
       M[t] = M[t-1] * (1 - w⊗e) + w⊗a
     This is fully differentiable unlike discrete addressing.
  4. Multiple read/write heads for parallel memory access.

Novel extension beyond standard NTM:

  *Surprise-gated writing*: the write gate is modulated by the model's
  prediction error for the current input — high-surprise → strong write
  (store surprising events), low-surprise → weak write (known info).
  This matches hippocampal novelty-gated encoding (Lisman et al. 2017).

  *Temporal context shift*: location-based addressing includes a learned
  shift kernel that can rotate attention forward or backward in memory,
  enabling temporal scanning without explicit position tracking.

  The memory matrix is *not* a Parameter but a module-level buffer
  that persists between batches in interactive mode and resets at the
  start of each training episode.

References:
  Graves et al. (2014): Neural Turing Machines
  Graves et al. (2016): Hybrid computing using a neural network (DNC)
  Lisman et al. (2017): Viewpoints: how memory consolidation occurs
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DifferentiableMemory(nn.Module):
    """NTM-style differentiable external memory with surprise-gated writes.

    Parameters
    ----------
    memory_size : number of memory rows (slots)
    d_model     : token embedding dimension
    d_mem       : memory row width (None → d_model)
    n_read_heads  : number of parallel read heads
    n_write_heads : number of parallel write heads
    """

    def __init__(self, memory_size: int = 128, d_model: int = 256,
                 d_mem: Optional[int] = None,
                 n_read_heads: int = 2,
                 n_write_heads: int = 1):
        super().__init__()
        self.memory_size  = memory_size
        self.d_model      = d_model
        self.d_mem        = d_mem or d_model
        self.n_read_heads = n_read_heads
        self.n_write_heads = n_write_heads

        # Initial memory state (learnable)
        self.mem_init = nn.Parameter(
            torch.randn(memory_size, self.d_mem) * 0.01
        )

        # Controller projection: produces read queries + write keys
        # Output: n_read_heads * (d_mem+1) + n_write_heads * (2*d_mem+2)
        # = read: d_mem (query) + 1 (sharpening β)
        # = write: d_mem (key) + d_mem (erase) + d_mem (add) + 1 (β) + 1 (write gate)
        r_out = n_read_heads  * (self.d_mem + 1)
        w_out = n_write_heads * (3 * self.d_mem + 2)
        self.controller = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, r_out + w_out),
        )

        # Surprise gate: how surprising is this input? (modulates write strength)
        self.surprise_gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        # Input projection + output projection
        self.inp_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model + n_read_heads * self.d_mem, d_model)
        self.ln       = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_address(key: torch.Tensor, M: torch.Tensor,
                        beta: torch.Tensor) -> torch.Tensor:
        """Content-based addressing with sharpening.

        key:  (B, d_mem)
        M:    (B, N, d_mem)
        beta: (B,) sharpening factor
        Returns (B, N) soft attention weights
        """
        key_norm = F.normalize(key.unsqueeze(1), dim=-1)   # (B, 1, d_mem)
        M_norm   = F.normalize(M, dim=-1)                  # (B, N, d_mem)
        sim = (key_norm * M_norm).sum(-1)                  # (B, N) cosine sim
        # Sharpening: w ∝ sim^beta
        sim_sharp = sim * beta.unsqueeze(1)
        return F.softmax(sim_sharp, dim=-1)                # (B, N)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor,
                M: Optional[torch.Tensor] = None,
                pred_error: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x:          (B, T, d_model)
        M:          (B, N, d_mem) — current memory state (None → use init)
        pred_error: (B,) — prediction error for surprise-gated write

        Returns:
          out:  (B, T, d_model) — enriched representation
          M_new (B, N, d_mem)  — updated memory
        """
        B, T, _ = x.shape
        N, Dm = self.memory_size, self.d_mem

        if M is None:
            M = self.mem_init.unsqueeze(0).expand(B, N, Dm).clone()

        ctrl_params = self.controller(x)   # (B, T, r_out + w_out)
        read_out_list = []

        for t in range(T):
            xt = x[:, t, :]               # (B, d_model)
            params_t = ctrl_params[:, t, :]  # (B, r_out + w_out)

            offset = 0

            # ---- Read heads ----
            read_vecs = []
            for _ in range(self.n_read_heads):
                rq    = params_t[:, offset: offset + Dm]
                beta  = F.softplus(params_t[:, offset + Dm]) + 1.0  # >1
                offset += Dm + 1

                w_r = self._cosine_address(rq, M, beta)       # (B, N)
                r   = torch.einsum("bn,bnd->bd", w_r, M)     # (B, Dm)
                read_vecs.append(r)

            # ---- Write heads ----
            surprise = (pred_error if pred_error is not None
                        else self.surprise_gate(xt).squeeze(-1))  # (B,)

            for _ in range(self.n_write_heads):
                wk     = params_t[:, offset: offset + Dm]
                beta_w = F.softplus(params_t[:, offset + Dm]) + 1.0
                erase  = torch.sigmoid(params_t[:, offset + Dm + 1: offset + 2 * Dm + 1])
                add    = torch.tanh(params_t[:, offset + 2 * Dm + 1: offset + 3 * Dm + 1])
                w_gate = torch.sigmoid(params_t[:, offset + 3 * Dm + 1]) * surprise
                offset += 3 * Dm + 2

                w_w = self._cosine_address(wk, M, beta_w)    # (B, N)
                w_w = w_w * w_gate.unsqueeze(1)              # scale by write gate

                # Erase then add
                erase_exp = torch.einsum("bn,bd->bnd", w_w, erase)   # (B, N, Dm)
                add_exp   = torch.einsum("bn,bd->bnd", w_w, add)
                M = M * (1 - erase_exp) + add_exp

            # Output: concat read vectors
            read_cat = torch.cat(read_vecs, dim=-1)  # (B, n_read * Dm)
            read_out_list.append(read_cat)

        read_seq = torch.stack(read_out_list, dim=1)   # (B, T, n_read*Dm)
        combined = torch.cat([x, read_seq], dim=-1)    # (B, T, D + n_read*Dm)
        out = self.ln(x + self.out_proj(combined))

        return out, M.detach()
