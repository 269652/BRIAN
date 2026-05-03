"""Central configuration for NeuroSLM.

All dimensions, layer counts, and training hyperparameters live here so
scaling up is a one-file change.
"""
from dataclasses import dataclass, field


@dataclass
class BrainConfig:
    # ---- Shared semantic embedding space (the "GWS bus" dim) ----
    d_sem: int = 256          # shared semantic embedding dimension
    d_hidden: int = 384       # internal hidden dim for most modules
    vocab_size: int = 50257   # GPT-2 BPE vocab (via tiktoken)

    # ---- Sensory / language cortex ----
    lang_layers: int = 4
    lang_heads: int = 6
    lang_kv_heads: int | None = None  # GQA: if set, KV heads < Q heads (saves ~30% attn params)
    lang_ctx: int = 512       # max context tokens

    # ---- World / self / forward models (SSM-style, but we use GRU for simplicity) ----
    world_layers: int = 2
    self_layers: int = 1
    forward_layers: int = 2

    # ---- Global workspace ----
    gws_slots: int = 8        # number of broadcast slots
    gws_heads: int = 4

    # ---- DMN / PFC ----
    dmn_layers: int = 2
    pfc_layers: int = 2
    pfc_heads: int = 4

    # ---- Hippocampus ----
    hippo_capacity: int = 4096    # max stored episodes
    hippo_topk: int = 4           # recalls per query
    hippo_sparse_k: int = 32      # DG sparse code active units (out of d_sem)
    novelty_threshold: float = 0.6

    # ---- Basal ganglia ----
    bg_action_dim: int = 256
    bg_n_candidates: int = 4

    # ---- Neuromodulators ----
    n_neuromods: int = 4          # DA, NE, 5HT, ACh

    # ---- Loop control ----
    dmn_period: int = 4           # sensory ticks per DMN tick
    max_thinking_steps: int = 6   # max planning iterations before forced output

    # ---- Floating thought ----
    thought_alpha: float = 0.3    # base ACh-modulated update rate

    # ---- Training ----
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_clip: float = 1.0

    # ---- Loss weights for multi-objective training ----
    w_lm: float = 1.0          # language modeling
    w_world: float = 0.3       # world model next-state prediction
    w_self: float = 0.1        # self model consistency
    w_forward: float = 0.2     # forward model
    w_value: float = 0.1       # evaluator (placeholder; needs RL signal)
    w_motor: float = 0.05      # SPEAK gate auxiliary loss
    w_pred_coding: float = 0.1  # predictive coding (inter-layer surprise) — novel
    speak_conf_threshold: float = 0.25  # min next-token confidence to want to SPEAK

    # ---- Intelligence-density features ----
    gradient_checkpointing: bool = False
    hebbian_rank: int = 0         # Hebbian trace rank (0=off, 8=default for novel attention)
    use_moe: bool = False
    moe_experts: int = 8
    moe_top_k: int = 2
    use_adaptive_compute: bool = False
    max_ponder_steps: int = 8

    # ---- Ablation ----
    baseline: bool = False  # True = vanilla transformer only, no bio modules


# ----- Preset sizes -----
def tiny() -> BrainConfig:
    """~5M params. Sanity test."""
    c = BrainConfig()
    c.d_sem = 128
    c.d_hidden = 192
    c.lang_layers = 2
    c.lang_heads = 4
    c.lang_ctx = 256
    c.dmn_layers = 1
    c.pfc_layers = 1
    return c


def small() -> BrainConfig:
    """~15M params. CPU-trainable in hours."""
    return BrainConfig()


def medium() -> BrainConfig:
    """~80M params. GPU recommended."""
    c = BrainConfig()
    c.d_sem = 512
    c.d_hidden = 768
    c.lang_layers = 8
    c.lang_heads = 8
    c.lang_ctx = 1024
    c.dmn_layers = 4
    c.pfc_layers = 4
    return c


def large() -> BrainConfig:
    """~100M params, allocated for maximum intelligence density.

    Design choices (per the refactor):
      * Narrower d_sem (256) to maximize compute-per-param.
      * Deeper recurrent transformer (lang_layers=8) where most params
        live; AdaptiveComputeBlock + ponder gives effective depth >24.
      * Wider PFC for planning (pfc_layers=3) but conservative DMN.
      * Long context (1024) so causal patterns over discourse are visible.
      * Larger gws_slots (12) = more concurrent broadcast streams.

    Sized to fit T4 16GB at batch_size=2 with grad checkpointing on the
    language cortex. Budget rationale:
        embed/unembed:  50257 * 256 * 2  ≈ 26M
        8 trans blocks: 8 * (4 * 256² + ff(4×))  ≈ 25M
        MoE (8 experts, top-2): ≈ 30M (only ~25% active per token)
        modules + memory heads: ≈ 20M
        TOTAL                                    ≈ 100M
    """
    c = BrainConfig()
    c.d_sem = 256
    c.d_hidden = 384
    c.lang_layers = 8
    c.lang_heads = 8
    c.lang_ctx = 1024
    c.dmn_layers = 3
    c.pfc_layers = 3
    c.pfc_heads = 4
    c.gws_slots = 12
    c.gws_heads = 4
    c.world_layers = 2
    c.forward_layers = 2
    c.hippo_capacity = 8192
    c.hippo_topk = 6
    c.max_thinking_steps = 12      # recurrent ponder depth
    c.warmup_steps = 500
    c.lr = 2.5e-4
    return c


def xl() -> BrainConfig:
    """~350M params — fits on A100 (40GB) with batch_size=4.

    Scales up from `large` (100M on T4 15GB) to use 40GB A100.
    The Brain has many bio modules beyond the language cortex
    (PFC, DMN, cerebellum, cortical sheet, neural geometry, etc.)
    so d_hidden must stay moderate.

    Budget (approximate):
        embed/unembed:  50257 * 512 * 2          ≈ 51M
        16 trans blocks: 16 * (4*512² + ff)       ≈ 100M
        16 geometry adapters                       ≈ 50M
        bio modules (pfc, dmn, hippo, cerebellum,
          cortical sheet, entorhinal, claustrum,
          neural geometry, neurochemistry, etc.)   ≈ 150M
        TOTAL                                      ≈ 350M
        FP32 weights: ~1.4GB  +  activations/grads: ~15GB
        Fits A100 40GB at batch=4, ctx=2048
    """
    c = BrainConfig()
    c.d_sem = 512
    c.d_hidden = 512
    c.lang_layers = 16
    c.lang_heads = 8
    c.lang_kv_heads = 2           # GQA: 4:1 ratio (Qwen-style), saves ~30% attn params
    c.lang_ctx = 2048
    c.dmn_layers = 3
    c.pfc_layers = 3
    c.pfc_heads = 8
    c.gws_slots = 12
    c.gws_heads = 8
    c.world_layers = 2
    c.self_layers = 2
    c.forward_layers = 2
    c.hippo_capacity = 8192
    c.hippo_topk = 8
    c.hippo_sparse_k = 128
    c.max_thinking_steps = 16
    c.warmup_steps = 1000
    c.lr = 2e-4
    c.weight_decay = 0.1
    c.gradient_checkpointing = True
    c.hebbian_rank = 8            # Novel: Hebbian attention trace for in-context learning
    c.use_moe = False
    c.moe_experts = 8
    c.moe_top_k = 2
    c.use_adaptive_compute = False
    c.max_ponder_steps = 8
    return c


def xxl() -> BrainConfig:
    """~10B params — requires multi-GPU (4×A100 or 8×A100).

    Target: outperform Qwen2.5-7B and compete with 14B-class models.
    """
    c = BrainConfig()
    c.d_sem = 2048
    c.d_hidden = 4096
    c.lang_layers = 32
    c.lang_heads = 32
    c.lang_ctx = 4096
    c.dmn_layers = 6
    c.pfc_layers = 6
    c.pfc_heads = 16
    c.gws_slots = 24
    c.gws_heads = 16
    c.world_layers = 4
    c.self_layers = 3
    c.forward_layers = 4
    c.hippo_capacity = 32768
    c.hippo_topk = 12
    c.hippo_sparse_k = 256
    c.max_thinking_steps = 24
    c.warmup_steps = 2000
    c.lr = 1e-4
    c.weight_decay = 0.1
    c.gradient_checkpointing = True
    c.use_moe = True
    c.moe_experts = 16
    c.moe_top_k = 2
    c.use_adaptive_compute = True
    c.max_ponder_steps = 12
    return c


PRESETS = {"tiny": tiny, "small": small, "medium": medium, "large": large, "xl": xl, "xxl": xxl}
