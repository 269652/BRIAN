"""Common building blocks: RMSNorm, SwiGLU MLP, RoPE attention block."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .neuro_attention import NeuromodulatedScale, HebbianTrace


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or int(dim * 8 / 3)
        # Round to multiple of 8 for efficiency
        hidden = (hidden + 7) // 8 * 8
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0,
                     device=None, dtype=torch.float32):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, : x.size(-2), :]
    sin = sin[None, None, : x.size(-2), :]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    out = torch.stack([rx1, rx2], dim=-1).flatten(-2)
    return out


class CausalSelfAttention(nn.Module):
    """Multi-head attention with GQA + neuromodulated temperature + Hebbian trace.

    Novel mechanisms (no existing model has these):
    1. NT-modulated attention temperature: DA sharpens, NE broadens, ACh precision
    2. Hebbian fast-weight trace: accumulated co-activation biases attention
    """
    def __init__(self, dim: int, n_heads: int, max_ctx: int,
                 n_kv_heads: int | None = None,
                 n_nt: int = 0, hebbian_rank: int = 0,
                 dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads  # default: MHA
        assert n_heads % self.n_kv_heads == 0
        self.n_groups = n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads
        self.dropout = float(dropout)
        self.resid_drop = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()

        # Separate Q and KV projections for GQA
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * self.n_kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        cos, sin = build_rope_cache(max_ctx, self.head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Novel: Neuromodulated attention temperature (if NT system active)
        self.nt_scale = NeuromodulatedScale(n_nt, n_heads) if n_nt > 0 else None
        # Novel: Hebbian attention trace (fast-weight relational memory)
        self.hebbian = HebbianTrace(self.head_dim, rank=hebbian_rank) if hebbian_rank > 0 else None

    def forward(self, x: torch.Tensor,
                nt: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        # Q: (B, T, n_heads, head_dim) → (B, n_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # KV: (B, T, 2*n_kv_heads, head_dim)
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (B, n_kv_heads, T, head_dim)

        # Query-Key normalisation: unit-sphere projection before RoPE.
        # Prevents distributional shift from causing attention entropy collapse
        # ("stale" keys/queries that all look similar after training diverges).
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # RoPE cache: if T exceeds the buffered cache (e.g. GIF OOD
        # probe evaluates at T=358 against a model with max_ctx=128),
        # rebuild on-the-fly. RoPE is parameter-free so this is a pure
        # recomputation of trig tables — no learned state lost.
        if self.cos.size(0) < T:
            cos, sin = build_rope_cache(T, self.head_dim,
                                        device=x.device, dtype=x.dtype)
        else:
            cos, sin = self.cos, self.sin
        q = apply_rope(q, cos.to(q.dtype), sin.to(q.dtype))
        k = apply_rope(k, cos.to(k.dtype), sin.to(k.dtype))

        # Expand KV heads to match Q heads for GQA
        if self.n_groups > 1:
            k = k[:, :, None, :, :].expand(-1, -1, self.n_groups, -1, -1).reshape(B, self.n_heads, T, self.head_dim)
            v = v[:, :, None, :, :].expand(-1, -1, self.n_groups, -1, -1).reshape(B, self.n_heads, T, self.head_dim)

        # ---- Novel: NT-modulated attention temperature ----
        # DA sharpens attention (higher scale → lower temperature → exploit)
        # NE broadens attention (lower scale → higher temperature → explore)
        if self.nt_scale is not None and nt is not None:
            scale = self.nt_scale(nt)  # (B, H, 1, 1)
            q = q * scale  # modulates attention sharpness per-head

        # ---- Novel: Hebbian trace bias ----
        # Accumulated co-activation creates persistent relational memory
        if self.hebbian is not None:
            hebb_bias = self.hebbian(q, k)  # (B, H, T, T)
            # Manual attention with Hebbian bias (can't use SDPA with custom bias + causal)
            attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            attn = attn + hebb_bias
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(causal_mask, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            if self.dropout > 0 and self.training:
                attn = F.dropout(attn, p=self.dropout)
            y = attn @ v
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out(y))


class MemoryCrossAttention(nn.Module):
    """Cross-attention block: queries from sequence x, K/V from external memory.

    Used by the SRC-TEH redesign to fold retrieved consolidated memory entries
    (and the bowtie's pooled output) into the last two trunk layers without
    breaking causal masking on the self-attention path.  Zero-init on the
    output projection so the block starts as a pure pass-through (identity);
    existing checkpoints without `memory_kv` arguments behave unchanged.
    """

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.q_norm   = RMSNorm(dim)
        self.kv_norm  = RMSNorm(dim)
        self.q_proj   = nn.Linear(dim, dim, bias=False)
        self.kv_proj  = nn.Linear(dim, 2 * dim, bias=False)
        self.out      = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.out.weight)        # identity at init

    def forward(self, x: torch.Tensor, memory_kv: torch.Tensor) -> torch.Tensor:
        """x: (B, T, dim).  memory_kv: (B, M, dim) or (M, dim)."""
        B, T, C = x.shape
        if memory_kv.dim() == 2:
            memory_kv = memory_kv.unsqueeze(0).expand(B, -1, -1)
        # Ensure dtype + device match the streaming hidden state.
        memory_kv = memory_kv.to(dtype=x.dtype, device=x.device)
        M = memory_kv.size(1)

        q  = self.q_proj(self.q_norm(x.float()).to(x.dtype))
        kv = self.kv_proj(self.kv_norm(memory_kv.float()).to(memory_kv.dtype))
        k, v = kv.chunk(2, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)

        # No causal mask — memory entries are unordered.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)


class TransformerBlock(nn.Module):
    """Self-attention + SwiGLU block with optional memory cross-attention.

    When `enable_memory_xattn=True` an additional zero-init cross-attention
    head is appended after the MLP.  At forward time, pass `memory_kv` to
    inject extra K/V rows (consolidated memory, bowtie pooled output, etc.).
    """

    def __init__(self, dim: int, n_heads: int, max_ctx: int,
                 n_kv_heads: int | None = None,
                 n_nt: int = 0, hebbian_rank: int = 0,
                 enable_memory_xattn: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.n1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, max_ctx, n_kv_heads,
                                        n_nt=n_nt, hebbian_rank=hebbian_rank,
                                        dropout=dropout)
        self.n2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim)
        self.mlp_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enable_memory_xattn = bool(enable_memory_xattn)
        if self.enable_memory_xattn:
            self.mem_xattn = MemoryCrossAttention(dim, n_heads)

    def forward(self, x: torch.Tensor,
                nt: torch.Tensor | None = None,
                memory_kv: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.n1(x), nt=nt)
        x = x + self.mlp_drop(self.mlp(self.n2(x)))
        if self.enable_memory_xattn and memory_kv is not None:
            x = x + self.mem_xattn(x, memory_kv)
        return x
