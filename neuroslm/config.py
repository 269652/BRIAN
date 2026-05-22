# -*- coding: utf-8 -*-
"""Central configuration for NeuroSLM.

All dimensions, layer counts, training hyperparameters, and per-module
enable flags live here — scaling up or toggling brain areas is a one-file change.

Per-module flags (all default True):
    enable_<area>: bool   — set False to bypass that brain area at runtime

Neural topology:
    neural_topology: str  — 'baseline' (language only) or 'full' (all modules)
"""
from dataclasses import dataclass


@dataclass
class BrainConfig:
    # ---- Novel modules ----
    enable_active_dendrite: bool = False
    enable_dynamic_routing_moe: bool = False
    enable_htm: bool = False
    enable_relational_attention: bool = False
    enable_fast_weight: bool = False
    enable_differentiable_memory: bool = False
    enable_phase_modulated_attention: bool = False
    enable_neurogenesis: bool = False
    enable_predictive_coding_loss: bool = False
    enable_causal_inference: bool = False
    # ---- Shared semantic embedding space (the "GWS bus" dim) ----
    d_sem: int = 256
    d_hidden: int = 384
    vocab_size: int = 50257   # GPT-2 BPE vocab (via tiktoken)

    # ---- Sensory / language cortex ----
    lang_layers: int = 4
    lang_heads: int = 6
    lang_kv_heads: int | None = None
    lang_ctx: int = 512

    # ---- World / self / forward models ----
    world_layers: int = 2
    self_layers: int = 1
    forward_layers: int = 2

    # ---- Global workspace ----
    gws_slots: int = 8
    gws_heads: int = 4

    # ---- DMN / PFC ----
    dmn_layers: int = 2
    pfc_layers: int = 2
    pfc_heads: int = 4

    # ---- Hippocampus ----
    hippo_capacity: int = 4096
    hippo_topk: int = 4
    hippo_sparse_k: int = 32
    novelty_threshold: float = 0.6

    # ---- Basal ganglia ----
    bg_action_dim: int = 256
    bg_n_candidates: int = 4

    # ---- Neuromodulators ----
    n_neuromods: int = 4   # DA, NE, 5HT, ACh

    # ---- Loop control ----
    dmn_period: int = 4
    max_thinking_steps: int = 6

    # ---- Floating thought ----
    thought_alpha: float = 0.3

    # ---- Training ----
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_clip: float = 1.0

    # ---- Regularization / generalization (OOD) ----
    # Dropout applied at the token embedding, inside the standard transformer
    # blocks (attention + residual), and just before the LM head. 0.0 keeps
    # the legacy no-dropout behavior; 0.1 is the GPT-2-scale default that
    # markedly narrows the train→OOD perplexity gap for a 100M model on a
    # limited data mix.
    dropout: float = 0.1
    # Decoupled weight decay: when True, AdamW/Adafactor decay only the 2-D
    # weight matrices (Linear/embedding) and exempt 1-D params (RMSNorm gains,
    # biases, NT levels). Applying decay to norms/biases is a known
    # generalization regression.
    decoupled_wd: bool = True
    # Cosine LR-decay horizon. The schedule fully anneals the LR to
    # `lr * min_lr_ratio` by this step. 0 → use the run's total --steps. For a
    # run you intend to STOP and evaluate at 10k, set this to 10000 (or pass
    # --lr_decay_steps 10000): otherwise, with the default --steps 100000, the
    # LR is still ~98% of peak at step 10k and the model never sees the
    # annealing phase that produces the biggest perplexity drop.
    lr_decay_steps: int = 0
    # Floor of the cosine schedule as a fraction of peak LR (e.g. 0.1 → LR
    # bottoms out at 10% of peak rather than 0).
    min_lr_ratio: float = 0.1
    # Label smoothing for the LM cross-entropy. 0.0 = off (default; label
    # smoothing can inflate raw perplexity). Available for calibration runs.
    label_smoothing: float = 0.0

    # ---- Post-awakening convergence / stabilization ----
    # Fixes the second-half divergence where a gnorm spike triggered a
    # self-amplifying collapse (lm_loss↑ → maturity↓ → aux-gate/pruning shift
    # → bigger perturbation → lm_loss↑). See docs/architecture.md §5.1.
    #
    # (A) Auxiliary-loss ramp length, in steps, measured FROM the awakening
    #     ramp start — a fixed horizon-independent schedule. Replaces the old
    #     `steps_ramped / (total_steps - step)` ramp whose denominator shrank
    #     to ~0 near the end, slamming all aux losses to full weight in the
    #     final ~10% of training (the proximate cause of the collapse).
    aux_ramp_steps: int = 2000
    # (B) Maturity ratchet: once maturity's high-water mark crosses
    #     `maturity_awaken_floor`, the smoothed MAT is monotonic
    #     non-decreasing, so a transient loss spike can no longer unwind the
    #     control state (aux gates + pruning thresholds) and amplify itself.
    maturity_ratchet: bool = True
    maturity_awaken_floor: float = 0.3
    # (C) Freeze structural pruning once the model has matured. Mid/late-run
    #     pruning removes projection capacity right when the model is doing
    #     its best work (observed: 6/16 projections pruned at peak maturity).
    #     Once the maturity high-water mark reaches `prune_freeze_mat`, pruning
    #     is latched off for the rest of the run (growth/BDNF still allowed).
    freeze_pruning_after_maturation: bool = True
    prune_freeze_mat: float = 0.6
    # (D) Gradient-spike rejection: skip the optimizer step when the pre-clip
    #     grad norm exceeds `grad_spike_factor × EMA(gnorm)`. A single bad
    #     batch (clipping caps magnitude, not direction) can no longer kick
    #     the model into divergence. 0 disables. Active only after
    #     `grad_spike_warmup` steps (so the EMA is established).
    grad_spike_factor: float = 3.0
    grad_spike_warmup: int = 100

    # ---- Loss weights ----
    w_lm: float = 1.0
    w_world: float = 0.3
    w_self: float = 0.1
    w_forward: float = 0.2
    w_value: float = 0.1
    w_motor: float = 0.05
    w_pred_coding: float = 0.1
    speak_conf_threshold: float = 0.25

    # ---- Intelligence-density features ----
    gradient_checkpointing: bool = False
    hebbian_rank: int = 0
    mod_capacity: float = 1.0
    use_moe: bool = False
    moe_experts: int = 8
    moe_top_k: int = 2
    use_adaptive_compute: bool = False
    max_ponder_steps: int = 8

    # ---- Memory ----
    consolidate_every: int = 500      # consolidate episodic→semantic every N steps

    # ---- Ablation ----
    baseline: bool = False          # True = vanilla transformer only
    baseline_lang_layers: int = 0   # 0 = use lang_layers; >0 overrides for param-parity

    # ================================================================
    # Neural topology: 'baseline' (language only) or 'full' (all modules)
    # ================================================================
    neural_topology: str = "full"

    # ================================================================
    # Per-module enable flags
    # Set any to False to bypass that brain area without removing it.
    # Brain areas that are disabled return neutral passthrough outputs.
    # ================================================================
    enable_hippocampus:       bool = True
    enable_pfc:               bool = True
    enable_basal_ganglia:     bool = True
    enable_dmn:               bool = True
    enable_thalamus:          bool = True
    enable_cerebellum:        bool = True
    enable_cortical_sheet:    bool = True
    enable_entorhinal:        bool = True
    enable_claustrum:         bool = True
    enable_gws:               bool = True
    enable_world_model:       bool = True
    enable_self_model:        bool = True
    enable_critic:            bool = True
    enable_neural_geometry:   bool = True
    enable_qualia:            bool = True
    enable_thought_transformer: bool = True
    enable_oscillations:      bool = True
    enable_narrative:         bool = True
    enable_mesolimbic:        bool = True

    # ---- Novel cognitive modules ----
    enable_tom:               bool = False  # Theory of Mind
    enable_rssm:              bool = False  # Recurrent State Space Model (world model)
    enable_active_inference:  bool = False  # Free Energy / predictive coding
    enable_hypergraph:        bool = True   # multidimensional hypergraph memory
    enable_entity_store:      bool = True   # entity recognition + per-entity profiles
    enable_vesicles:          bool = True   # Neuro-vesicle homeostatic GABA regulation (enabled for stability)
    n_vesicles:               int  = 32     # max live vesicles
    vesicle_lifetime:         int  = 16     # ticks until vesicle degradation

    # ---- Emotional / subcortical modules ----
    enable_amygdala:          bool = True   # emotional tagging + fear conditioning
    enable_acc:               bool = True   # anterior cingulate: conflict monitoring
    enable_insula:            bool = True   # interoception + gut feelings
    enable_lateral_habenula:  bool = True   # anti-reward, aversion learning
    amygdala_d_emotion:       int  = 32     # amygdala emotional rep dimension

    # ---- Novel ML objectives ----
    enable_cpc:               bool = False  # contrastive predictive coding loss
    cpc_steps:                int  = 5      # CPC prediction horizon
    cpc_negatives:            int  = 32     # CPC negative samples
    w_cpc:                    float = 0.05  # CPC loss weight

    # ---- RSSM dimensions ----
    rssm_n_cats: int = 8    # number of categorical latent variables
    rssm_d_cat:  int = 16   # classes per categorical variable

    # ---- Theory of Mind dimensions ----
    tom_d_style:    int = 64   # entity style embedding size
    tom_n_heads:    int = 4
    tom_n_layers:   int = 2

    # ---- Active inference ----
    active_inf_layers: int = 3    # predictive hierarchy depth

    # ---- Entity store ----
    entity_d_style: int = 64     # entity style fingerprint dimension

    # ---- Loss weights (new) ----
    w_kl_world:    float = 0.1   # RSSM KL divergence
    w_free_energy: float = 0.05  # active inference free energy
    w_social:      float = 0.1   # social prediction error

    # ---- Φ (integrated information) objective ----
    # Maximises the IIT-style Gaussian-MI lower bound across the
    # bowtie module bipartition. Loss term is `-w_phi * phi`, so the
    # gradient pushes module outputs toward configurations where no
    # bipartition disconnects them cheaply (high integration).
    enable_phi_objective: bool = True
    w_phi:                float = 0.02
    # When phi exceeds this threshold (logged in `_last_phi`) the trophic
    # system increases BDNF on active edges to "lock in" the integrated
    # configuration (Dehaene structural selection).
    phi_lock_threshold:   float = 0.5

    # ================================================================
    # SRC-TEH (Shared Reading Cortex + Token-Level Expert Heads)
    # See docs/RFC.md.  Default OFF so legacy presets / tests keep
    # working unchanged; flipped on by the xl preset.
    # ================================================================
    enable_src_teh:        bool  = False
    # Mid-trunk bowtie tap (0-based layer after which the trunk emits an
    # AttentionPool pooled summary to feed the bowtie).  None → middle.
    mid_trunk_tap_layer:   int   = 0       # 0 = auto (= lang_layers // 2)
    # Memory-augmented trunk (RETRO-style).  Last N blocks accept extra
    # K/V rows: pooled bowtie output + top-N consolidated memory entries.
    enable_memory_xattn:   bool  = False
    n_memory_xattn_layers: int   = 2
    n_memory_entries:      int   = 64       # top-N consolidated entries
    # Expert-Choice routing across {Lang, Math, Reason}.
    n_token_experts:       int   = 3
    expert_capacity_factor: float = 1.5
    expert_n_blocks:       int   = 3
    expert_n_heads:        int   = 8
    w_expert_aux:          float = 0.01
    # Latent Program Bus (replaces vesicle gating for routing).
    enable_latent_bus:     bool  = False
    bus_dim:               int   = 16
    bus_ema_alpha:         float = 0.5
    # Lazy Bowtie + EMA fallback.
    bowtie_period:         int   = 4          # run heavy bowtie every K steps
    bowtie_ema_alpha:      float = 0.4        # EMA decay for off-step fallback
    # Softened trophic gate: pruning disabled while MAT < trophic_prune_mat.
    # Bumped from 0.3 → 0.6 so structural pruning stays suppressed for the
    # post-awakening window where projections are still resolving — closes
    # the observed "n_active: 2/16" collapse pattern.
    trophic_prune_mat:     float = 0.6


# ----- Preset sizes -----
def tiny() -> BrainConfig:
    """~27M params (full bio stack). Sanity test."""
    c = BrainConfig()
    c.d_sem = 128
    c.d_hidden = 192
    c.lang_layers = 2
    c.lang_heads = 4
    c.lang_ctx = 256
    c.dmn_layers = 1
    c.pfc_layers = 1
    # Baseline param parity: 37 vanilla layers ≈ 26.9M ≈ full 26.8M.
    c.baseline_lang_layers = 37
    return c


def small() -> BrainConfig:
    """~93M params (full bio stack). CPU-trainable in hours."""
    c = BrainConfig()
    # Baseline param parity: 40 vanilla layers ≈ 94.0M ≈ full 93.2M.
    c.baseline_lang_layers = 40
    return c


def medium() -> BrainConfig:
    """~389M params (full bio stack). GPU recommended."""
    c = BrainConfig()
    c.d_sem = 512
    c.d_hidden = 768
    c.lang_layers = 8
    c.lang_heads = 8
    c.lang_ctx = 1024
    c.dmn_layers = 4
    c.pfc_layers = 4
    # Baseline param parity: 47 vanilla layers ≈ 388.8M ≈ full 389.2M.
    c.baseline_lang_layers = 47
    return c


def large() -> BrainConfig:
    """~100M params. T4 16GB at batch_size=2 with grad checkpointing."""
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
    c.max_thinking_steps = 12
    c.warmup_steps = 500
    c.lr = 2.5e-4
    # OOD generalization: decoupled weight decay bumped 0.01 → 0.05 (applies
    # only to 2-D matrices now) and GPT-scale dropout. Both narrow the
    # train→WikiText gap without hurting in-distribution loss at 100M params.
    c.weight_decay = 0.05
    c.dropout = 0.1
    # Baseline param parity: 47 vanilla layers ≈ 106.9M ≈ the full model's
    # 107.5M, so the --baseline ablation is a fair same-size comparison
    # (otherwise the 8-layer baseline is only ~35M).
    c.baseline_lang_layers = 47
    return c


def xl() -> BrainConfig:
    """~240M params — A100 (40GB). SRC-TEH topology (RFC).

    Two-tier design:
      • Shared Reading Cortex: 10 layers @ d_hidden=576 (≈60% of params)
      • 3 token-level expert heads (Language/Math/Reason): 3 blocks @ d_hidden each
      • Mid-trunk bowtie tap @ layer 5, RETRO-style memory injection on last 2 layers
      • Latent Program Bus (16-dim) for across-step reasoning state
      • Lazy bowtie (every 4 steps with EMA fallback)
    """
    c = BrainConfig()
    # Tier-1 trunk width — modest bump from 512 → 576 makes room for the
    # shared reading cortex to do the LM/comprehension heavy lifting.
    c.d_sem = 384
    c.d_hidden = 576

    # 10 deep trunk layers (down from 12) — the freed param budget goes
    # to 3 token-level experts (Lang/Math/Reason) at d_hidden=576.
    c.lang_layers = 10
    c.lang_heads = 8
    c.lang_kv_heads = None
    c.lang_ctx = 2048

    # Moderate control / workspace sizes
    c.dmn_layers = 3
    c.pfc_layers = 3
    c.pfc_heads = 8
    c.gws_slots = 8
    c.gws_heads = 8

    # Keep light world/self models
    c.world_layers = 2
    c.self_layers = 1
    c.forward_layers = 2

    # Smaller hippocampus footprint
    c.hippo_capacity = 4096
    c.hippo_topk = 6
    c.hippo_sparse_k = 64

    c.max_thinking_steps = 12
    c.warmup_steps = 800
    c.lr = 2e-4
    c.weight_decay = 0.1
    c.gradient_checkpointing = True
    c.hebbian_rank = 4
    c.mod_capacity = 0.8

    # Baseline param parity: 60 vanilla layers ≈ 279.9M ≈ full 280.3M
    # (full SRC-TEH model at d_hidden=576).
    c.baseline_lang_layers = 60

    # Novel modules: keep defaults conservative to limit params
    c.enable_rssm = False
    c.rssm_n_cats = 8
    c.rssm_d_cat  = 16
    c.enable_active_inference = False
    c.active_inf_layers = 2
    c.enable_tom = False
    c.tom_d_style = 64
    c.tom_n_heads = 4

    # ---- SRC-TEH topology (default ON for xl) ----
    c.enable_src_teh         = True
    c.enable_memory_xattn    = True
    c.n_memory_xattn_layers  = 2
    c.n_memory_entries       = 64
    c.mid_trunk_tap_layer    = 5      # tap at half-depth (layer 5 of 10)
    c.n_token_experts        = 3
    c.expert_capacity_factor = 1.5
    c.expert_n_blocks        = 3
    c.expert_n_heads         = 8
    c.w_expert_aux           = 0.01
    c.enable_latent_bus      = True
    c.bus_dim                = 16
    c.bus_ema_alpha          = 0.5
    c.bowtie_period          = 4
    c.bowtie_ema_alpha       = 0.4
    c.trophic_prune_mat      = 0.6

    return c


def xxl() -> BrainConfig:
    """~10B params — multi-GPU (4×A100 or 8×A100)."""
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
    # Novel modules
    c.enable_rssm = True
    c.rssm_n_cats = 32
    c.rssm_d_cat  = 32
    c.enable_active_inference = True
    c.active_inf_layers = 4
    c.enable_tom = True
    c.tom_d_style = 256
    c.tom_n_heads = 16
    c.tom_n_layers = 4
    return c


PRESETS = {
    "tiny": tiny, "small": small, "medium": medium,
    "large": large, "xl": xl, "xxl": xxl,
}
