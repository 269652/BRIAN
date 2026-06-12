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

    # ---- Predictive Coding Trunk (PCT) — top-down generative trunk ----
    # Architectural mechanism change: each adjacent layer pair gets a
    # top-down predictor g_n that must predict h_n from h_{n+1}. The
    # prediction error e_n = h_n - g_n(h_{n+1}) is summed across all layer
    # pairs into a free-energy loss term added to the LM total. This
    # changes what the trunk's gradients optimize: features become
    # generative inverses of the layer above, not whatever-minimizes-
    # next-token-loss. See neuroslm/modules/predictive_coding_trunk.py
    # for the full theory note. Off by default (legacy compatible).
    use_predictive_coding_trunk: bool = False
    # "loss_only"  — forward unchanged, only the FE loss is added (cheap)
    # "feedback"   — previous-layer error projects into next block's input
    pct_mode: str = "loss_only"
    # Weight of the free-energy term in the total loss. 0.1 is a reasonable
    # starting point — strong enough to shape representations, small enough
    # not to swamp the LM cross-entropy in early training.
    pct_lambda_fe: float = 0.1
    # Hidden-dim multiplier for the top-down predictor MLP (small = cheap).
    pct_hidden_mult: float = 0.5
    # Forward error-feedback scale when pct_mode == "feedback". Zero-init
    # projection means start is bit-identical to standard trunk; this scales
    # the learned correction once the projection has trained up.
    pct_feedback_alpha: float = 0.05
    # Add an extra predictor for the embedding from the first block's
    # output. Slight extra cost, gives embeddings direct generative pressure.
    pct_include_embedding_predictor: bool = True

    # ---- Synthesis-v1: Predictive-Dropout + Smooth-Gated-Bus ----
    # Predictive-Dropout: channel-mask whose per-channel keep probability
    # comes from PCT's per-channel prediction error. Drops well-predicted
    # (low-info) channels and keeps surprising ones. Requires PCT on.
    # See neuroslm/modules/predictive_dropout.py.
    use_predictive_dropout: bool = False
    pdrop_base_keep: float = 0.5
    pdrop_beta: float = 4.0
    pdrop_per_token: bool = False

    # Smooth-Gated-Bus: temporally-smooth replacement for the ReZero
    # zero-init scalars (lambda_motor/mem/thought). gate(t) =
    # max·sigmoid((t−center)/width) per gate. Drives module-injection
    # contribution to be non-zero from step 0 while remaining
    # smoothly-growing through training. See
    # neuroslm/modules/smooth_gated_bus.py.
    use_smooth_gated_bus: bool = False
    sgb_default_center: float = 2000.0
    sgb_default_width: float = 500.0

    # ---- Ablation-knob feature flags (default False, opt-in) ──────────
    # These four flags exist purely to give clean A/B comparisons; each one
    # toggles a single mechanism without touching any other moving part.
    # Per CLAUDE.md §10, flipping any of them creates a new experimental
    # condition that should be run as its own short eval, not silently
    # merged into the default config.
    #
    # use_tdw — swap the Hopfield/ignition GlobalWorkspace for the
    #   TopologicalDifferentialWorkspace defined in modules/workspace.py.
    #   Same forward signature, so brain.py:1366/2340 call sites are
    #   untouched. The TDW kernel subtracts a blurry Hopfield retrieval
    #   from a sharp one and (optionally) projects the result onto an
    #   orthonormalised Tonnetz basis; this replaces ignition, it does
    #   not add anything on top.
    use_tdw: bool = False
    # use_diff_attn — DifferentialAttention is already in the cortex via
    #   DiffTransformerBlock at the every-3rd-layer slot of the
    #   interleaved [Std, Diff, MoD] pattern. When this flag is True the
    #   cortex is rebuilt with EVERY non-baseline block as
    #   DiffTransformerBlock (uniform DiffAttn cortex). Default False
    #   keeps today's interleaved pattern bit-identical.
    use_diff_attn: bool = False
    # use_tonnetz_prior — instantiate the TonnetzPrior adjacency-loss
    #   regulariser (modules/tonnetz_prior.py). Penalises the per-batch
    #   token-cooccurrence Laplacian when its algebraic-connectivity
    #   eigenvalue λ_1 falls below `tonnetz_gap_threshold`. Adds a single
    #   scalar loss term, zero new parameters at the trunk.
    use_tonnetz_prior: bool = False
    tonnetz_gap_threshold: float = 0.3
    w_tonnetz: float = 0.01
    # use_expert_ensemble — wired no-op reservation for the in-flight
    #   neuroslm/experts.py module (unstaged at time of writing). The
    #   flag is checked at Brain construction but no expert ensemble is
    #   built yet; once experts.py is committed its plumbing slots in
    #   behind this same flag without a schema change.
    use_expert_ensemble: bool = False

    # ---- Free-energy temporal ramp (synthesis-v1 stability fix) ----
    # When True, the PCT free-energy loss is multiplied by a smooth
    # sigmoid gate(step). At step 0 PCT contributes ~0; at step >>
    # fe_gate_center it contributes the full pct_lambda_fe weight. Same
    # SmoothTemporalGate primitive as SGB, just applied to the loss weight.
    fe_gate_enable: bool = False
    fe_gate_center: float = 1000.0
    fe_gate_width: float = 300.0

    # ---- Surprise-Gated Branching EMA (synth-v2 late-divergence fix) ----
    # Meta-optimizer maintaining a slow stable EMA of weights and a
    # snapshot at the historical lowest EMA-smoothed PPL. EMA mixing
    # rate alpha_eff = (1/avg_ppl) * exp(-gamma * |d(ppl)/dt|), so when
    # PPL is rising fast the stable shadow freezes. On catastrophic
    # spike (current_ppl > bema_collapse_ratio * best_ema_ppl) the
    # trunk is reverted to params_best. Fixes the synth-v1 pathology
    # where training hit ppl 59 at step 3800, spiked to 257/426 from
    # step 4800, and never recovered. See §7.5 in docs/architecture.md
    # and neuroslm/intelligence/branching_ema.py.
    use_branching_ema: bool = False
    bema_history_len: int = 10
    bema_gamma: float = 5.0   # see branching_ema.py for the regime table
    bema_alpha_cap: float = 0.01
    bema_update_every: int = 1
    bema_best_ema_alpha: float = 0.1
    bema_collapse_ratio: float = 3.0

    # ---- RCC Bowtie: Read, Copy, Commit (closed-loop write-back cut) ----
    # Diagnosis (after synth-v1/v2 + BEMA failures): the trunk is being
    # vandalized by closed-loop write-backs from cognitive modules. Each
    # of motor_lang_bias, memory_kv injection, from_sem(thought),
    # latent-bus bias, and scatter-add expert residual is a path along
    # which an immature module's random-init noise can push the language
    # manifold into a worse basin -- BEMA-style late-divergence + spike
    # patterns we kept seeing.
    #
    # Phase-1 RCC: disable all five write-back paths. Bio modules still
    # run (for Phi, aux losses, world/self/forward predictions) but their
    # outputs do NOT enter the trunk's forward pass. Equivalent to a pure
    # LM trunk + full bio stack running in observation-only mode.
    #
    # Phases 2-4 will re-introduce a bounded, low-rank, gated commit path
    # from a sidecar sandbox state z_cog into the final 1-2 trunk blocks.
    # See docs/architecture.md (TBD section) and tests/test_rcc_bowtie.py.
    use_rcc_bowtie: bool = False

    # RCC Bowtie Phase 2: also gate the NEUROTRANSMITTER MODULATION path
    # from the trophic-managed cognitive subsystem into the trunk. When
    # the trophic system grows/prunes projections mid-training, the NT
    # field's distribution shifts — and that field is plumbed into every
    # TransformerBlock's attention temperature (via Linear(n_neuromods,
    # n_heads) in neuro_attention.py). Setting nt=None at the trunk's
    # input makes attention temperature INPUT-INDEPENDENT, so the trophic
    # system can keep mutating its connectome for cognitive use without
    # perturbing the language manifold. Closes the second closed-loop
    # write path that P1 (forward-only cut) left open.
    rcc_freeze_nt_modulation: bool = False

    # RCC Bowtie Phase 3: PARAMETER CLOSURE ISOLATION.
    # P1 cut forward write-back. P2 cut NT modulation. P3 cuts the third
    # leak: bio-side ops (consolidator at every consolidate_every steps,
    # trophic grow/prune events, sleep cycle, causal.prune) mutate
    # nn.Parameter.data in-place under torch.no_grad(). Adam's momentum
    # state for those params goes stale → next step applies wrong-state
    # momentum to mutated weights → PPL spike at the same step every run.
    #
    # When True, train.py builds AdamW ONLY over trunk params (language.*
    # + sgb.* + lambda_motor/mem/thought). Bio params are not tracked by
    # Adam; bio-side machinery is free to mutate them without causing
    # optimizer-state inconsistency.
    rcc_isolate_optimizer: bool = False

    # Per-sample loss clipping (data-robust training).
    # Three independent RCC runs (P1/P2/P3) spiked from ppl 125 to 493 at
    # EXACTLY step 1500 with the SAME seed. With seed=0 the data iterator
    # is deterministic — same hard batch lands at same step every time.
    # When True, in brain._chunked_ce path each sequence's loss is clipped
    # at `loss_clip_factor x median(batch)` BEFORE averaging, so a single
    # pathological sequence can't yank the batch gradient. See
    # docs/STEP1500_INVESTIGATION.md.
    # Off by default (legacy bit-identical when False).
    loss_clip_robust: bool = False
    loss_clip_factor: float = 3.0

    # ---- Recursive Reasoning Cortex (Universal-Transformer-style) ----
    # When True, ReasoningCortex.forward_tokens loops its expert_blocks
    # `recursive_iters` times with weight-sharing — depth-multiplying the
    # reasoning expert at constant parameter count. The deepened output flows
    # through the existing bowtie / thought / from_sem path, ReZero-gated by
    # λ_thought (§5.3), so it cannot destabilize the LM trunk. Targets
    # reasoning benchmarks (HellaSwag/ARC) where iterative refinement helps
    # more than width. Forward FLOPs scale linearly with `recursive_iters`.
    # See §5.4.
    recursive_reasoning: bool = True
    recursive_iters: int = 4

    # ---- ReZero-style gated module → LM forward injections ----
    # Replace the maturity-phase gates on the FORWARD paths from bio modules
    # into the LM trunk (`_motor_phase` on motor_lang_bias, `_mem_phase` on
    # the memory_kv injection, conditioning of `lang_thought` via from_sem)
    # with zero-init learnable scalars (one per injection). The model behaves
    # identically to the pure isolated-trunk LM at t=0 (every λ=0 → no module
    # contribution → no discontinuity at awakening), and each λ then grows
    # under LM gradient ONLY as far as that injection actually helps next-
    # token prediction. Eliminates the forward-side awakening wobble that
    # `detach_trunk_from_aux` alone can't address. Reference: ReZero
    # (Bachlechner 2020) / LayerScale (Touvron 2021). See §5.3.
    use_rezero_injection_gates: bool = True

    # ---- Trunk gradient isolation (architectural convergence fix) ----
    # Feed the bio/cognitive modules a STOP-GRADIENT copy of the trunk's
    # semantic output `sem`. The LM trunk (language cortex) is then shaped
    # ONLY by the LM loss (+ its own deep-supervision pred_coding) — the same
    # clean regime as the stable infancy phase — while every auxiliary loss
    # (world / self / motor / Φ / cpc / htm / ...) trains its own module
    # reading a fixed representation, instead of pushing a large, random-init
    # gradient back into the representation the LM depends on. That backward
    # path is the root cause of the post-awakening divergence (gnorm jumped
    # from ~1.5 LM-only to ~14 once aux engaged; engaging it harder/earlier
    # diverged faster). Forward couplings (motor bias, memory-xattn, thought
    # conditioning) are unchanged — only the aux-loss gradient into the trunk
    # is cut. See docs/architecture.md §5.2.
    detach_trunk_from_aux: bool = True

    # ---- Post-awakening convergence / stabilization ----
    # Fixes the second-half divergence where a gnorm spike triggered a
    # self-amplifying collapse (lm_loss↑ → maturity↓ → aux-gate/pruning shift
    # → bigger perturbation → lm_loss↑). See docs/architecture.md §5.1.
    #
    # (A) Maturity-GATED auxiliary-loss weight. The master aux scale ramps
    #     from ~0 → 1 as the maturity index climbs from `aux_gate_mat_lo` to
    #     `aux_gate_mat_hi`, i.e. aux objectives only gain strength once the LM
    #     is genuinely good (MAT 0.5≈PPL220, 0.65≈PPL50). This replaces a
    #     step-schedule ramp: a step ramp either blew up late (denominator →0)
    #     or, if made fixed-length, slammed aux to full by ~step 2.5k and
    #     overwhelmed the still-immature LM (observed early divergence). Tying
    #     aux strength to maturity is self-correcting — if the LM regresses,
    #     MAT falls and aux automatically backs off.
    aux_gate_mat_lo: float = 0.50
    aux_gate_mat_hi: float = 0.65
    # (B) Maturity dynamics. The smoothed MAT now uses an ASYMMETRIC EMA: it
    #     rises at `maturity_ema_alpha` but falls at the slower
    #     `maturity_fall_alpha`, so a transient loss spike barely dents it
    #     (no whipsaw) while a SUSTAINED regression still lowers it — keeping
    #     the maturity-fall "recovery valve" that lets aux disengage so the
    #     model can refocus on LM. A hard ratchet (monotonic non-decreasing)
    #     is available but OFF by default: it removed that recovery path and
    #     made a collapse unrecoverable.
    maturity_ema_alpha: float = 0.05
    maturity_fall_alpha: float = 0.01
    maturity_ratchet: bool = False
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


def pct_30m() -> BrainConfig:
    """~30M params, PCT enabled. Ablation preset for arch/predictive-coding-trunk.

    Goal: cheapest valid run to falsify the PCT hypothesis (≥2× lower OOD
    gap_ratio at matched train PPL) before scaling to 107M. Matches
    `large()`'s regularization regime so the only architectural difference
    on a parity comparison is the PCT trunk update vs standard residual.
    """
    c = BrainConfig()
    c.d_sem = 192
    c.d_hidden = 384
    c.lang_layers = 4
    c.lang_heads = 6
    c.lang_ctx = 1024
    c.dmn_layers = 2
    c.pfc_layers = 2
    c.pfc_heads = 4
    c.gws_slots = 8
    c.gws_heads = 4
    c.world_layers = 1
    c.forward_layers = 1
    c.hippo_capacity = 2048
    c.hippo_topk = 4
    c.max_thinking_steps = 6
    c.warmup_steps = 300
    c.lr = 3e-4
    # Same OOD regularization regime as `large()` so the comparison is fair.
    c.weight_decay = 0.05
    c.dropout = 0.1
    # PCT enabled — the point of this preset.
    c.use_predictive_coding_trunk = True
    c.pct_mode = "loss_only"
    c.pct_lambda_fe = 0.1
    c.pct_hidden_mult = 0.5
    c.pct_include_embedding_predictor = True
    # Baseline param parity: ~32M @ 12 vanilla layers (no novel modules)
    c.baseline_lang_layers = 12
    return c


def synth_30m() -> BrainConfig:
    """~30M (actual ~68M w/ bio stack) — synthesis-v1: PCT + Smooth-Gated-Bus
    + Predictive-Dropout, bottom-up PC heads disabled (top-down only),
    stronger free-energy weighting. The cumulative test of whether
    topology-only changes can push gap_ratio down from ~5x at our scale.

    Combines:
      • PCT (top-down free-energy on trunk) with lambda_fe BUMPED 0.1 -> 2.0
        — the v1 result showed lambda_fe=0.1 was too weak; FE was acting as
        a soft regularizer not a dominant objective.
      • PredictiveDropout — channel-mask driven by PCT per-channel error.
        Drops well-predicted (low-info) channels (Information-Bottleneck
        regularization mechanically coupled to the PCT mechanism).
      • SmoothGatedBus — replaces ReZero zero-init step scalars with
        temporally-smooth sigmoid ramps so module injections are non-zero
        from step 0 (no "trunk-only" early phase).
      • Bottom-up PredictiveCodingHeads OFF (set automatically inside
        LanguageCortex when PCT is on) so top-down doesn't fight bottom-up.

    Falsifiable hypothesis: gap_ratio drops from ~5x (PCT-only) to 2-3x.
    """
    c = BrainConfig()
    c.d_sem = 192
    c.d_hidden = 384
    c.lang_layers = 4
    c.lang_heads = 6
    c.lang_ctx = 1024
    c.dmn_layers = 2
    c.pfc_layers = 2
    c.pfc_heads = 4
    c.gws_slots = 8
    c.gws_heads = 4
    c.world_layers = 1
    c.forward_layers = 1
    c.hippo_capacity = 2048
    c.hippo_topk = 4
    c.max_thinking_steps = 6
    c.warmup_steps = 300
    c.lr = 3e-4
    c.weight_decay = 0.05
    c.dropout = 0.1   # baseline dropout
    # PCT — stronger weighting, throttled by temporal ramp (below)
    c.use_predictive_coding_trunk = True
    c.pct_mode = "loss_only"
    c.pct_lambda_fe = 2.0    # full strength once the FE-gate has opened
    c.pct_hidden_mult = 0.5
    c.pct_include_embedding_predictor = True
    # FE temporal-ramp — fixes the chronic high-gnorm regime of synth-v1 try-1
    # where pct_lambda_fe=2 from step 0 dominated the gradient with
    # structureless noise (predictors weren't yet meaningful). Ramp opens
    # around step 1000 so the trunk has time to learn basic next-token
    # before PCT shapes its representations.
    c.fe_gate_enable = True
    c.fe_gate_center = 1000.0
    c.fe_gate_width = 300.0
    # Predictive-Dropout DISABLED in v1-attempt-2: most speculative of the
    # three additions, and the destabilization profile (loss stuck high
    # post step 500, gnorm 5+ chronically) is consistent with PCT lambda=2
    # alone driving the gradient noise. We'll add PD back as synth-v2 if
    # PCT-with-ramp + SGB lands a clean gap_ratio improvement.
    c.use_predictive_dropout = False
    # Smooth-Gated-Bus (unchanged from try-1)
    c.use_smooth_gated_bus = True
    c.use_rezero_injection_gates = True   # required: SGB is the gate vehicle
    c.sgb_default_center = 2000.0
    c.sgb_default_width = 500.0
    # Match large() OOD regimen
    c.baseline_lang_layers = 12
    return c


def synth_30m_bema() -> BrainConfig:
    """synth_30m + Surprise-Gated Branching EMA.

    Direct fix for the synth-v1 late-divergence pathology observed in
    the 2026-05-25 10k log:
        step 3800: ppl  59  (BEST seen)
        step 4800: ppl 257  (spike)
        step 5600: ppl 426  (catastrophic spike)
        step 5800-10000: stuck at ppl 200-270, never recovers.

    Adds the Branching-EMA meta-optimizer on top of synth_30m's stack
    (PCT + SGB + FE-ramp + top-down-only). Hypothesis: BEMA collapse-on-
    spike + stable-shadow EMA will keep the trunk anchored to the best
    basin seen during training, letting the final best.pt actually
    reflect the lowest-PPL trajectory rather than an early save point.
    """
    c = synth_30m()
    c.use_branching_ema = True
    c.bema_history_len = 10
    c.bema_gamma = 5.0
    c.bema_alpha_cap = 0.01
    c.bema_update_every = 1
    c.bema_best_ema_alpha = 0.1
    c.bema_collapse_ratio = 3.0
    return c


def rcc_bowtie_30m_p1() -> BrainConfig:
    """RCC Bowtie Phase 1 — all cognitive-to-trunk write-back paths CUT.

    Diagnosis: synth-v1 hit ppl 59 at step 3800 then diverged to ppl 200+.
    BEMA caught the spikes but hammer-looped (881 collapses, no progress).
    Root cause: the trunk is being vandalized by closed-loop write-backs
    from random-init bio modules.

    Phase 1 disables ALL five write-back paths:
      1. motor_lang_bias  (motor cortex -> h_lang final block)
      2. memory_kv        (bowtie EMA -> language memory cross-attn)
      3. from_sem(thought)+ bus_bias (latent program bus -> trunk prefix)
      4. scatter_add expert residual (token experts -> h_enriched)
      5. (No 5th — covered by the above set.)

    Bio modules still run — Phi proxy, world/self/forward predictions,
    aux losses all train normally. They just don't write back into the
    language manifold. Phase 2+ will re-introduce a low-rank, gated,
    norm-capped commit path through a CommitGate.

    Hypothesis: stable training run, no awakening collapse, no late
    divergence. Final train_ppl should at least match the pure-baseline
    LM (47-layer vanilla transformer at same param count).

    Built on synth_30m_bema for two reasons:
      a) PCT (top-down free-energy) targets the trunk's gradient
         mechanics directly and is consistent with RCC (operates within
         the trunk, not via cognitive write-back).
      b) BEMA stays as a safety net but should fire ~0 collapses now
         that the closed loop is cut (the synth-v1 spike pattern was
         driven by aux-noise injection that no longer happens).
    """
    c = synth_30m_bema()
    c.use_branching_ema = False  # Disable BEMA, RCC is the correct fix
    c.use_rcc_bowtie = True
    return c


def rcc_bowtie_30m_p2() -> BrainConfig:
    """RCC Bowtie Phase 2: cut the neurotransmitter-modulation leak.

    P1 closed forward write-back (motor_bias, mem_kv, from_sem thought,
    GWS broadcast) from cognitive modules into h_lang. The P1 10k run
    confirmed this was a real bug — gnorm 30+ spikes that would have
    catastrophically diverged synth-v1 were absorbed cleanly, and best.pt
    advanced past step 4000. BUT around step 4700 the model regressed
    from ppl 73 → 245+ coincident with a trophic growth event
    (16→17 projections), confirming a SECOND closed-loop leak: the
    trophic-system mutates the projection graph mid-training, the NT
    field shifts, and the trunk's attention temperature (driven by NT
    via neuro_attention.py:Linear(n_neuromods, n_heads)) is yanked.

    P2 fix: pass nt=None at the trunk's input. The trophic system can
    still grow/prune projections and modulate NT levels for cognitive
    use, but the trunk's attention is no longer a function of those.
    Trunk's input environment is now fully invariant w.r.t. cognitive
    dynamics — the language manifold is fully isolated.

    Hypothesis: best.pt continues to advance smoothly past step 4000,
    no late-training regression, sub-100 ppl basin maintained through
    step 10000.
    """
    c = rcc_bowtie_30m_p1()
    c.rcc_freeze_nt_modulation = True
    return c


def rcc_bowtie_30m_p3() -> BrainConfig:
    """RCC Bowtie Phase 3: PARAMETER CLOSURE ISOLATION.

    P1 cut forward write-back. P2 cut NT modulation. P3 fixes the third
    deterministic leak observed in both P1 and P2 runs: at exactly
    step 1500 (the third firing of consolidate_every=500), PPL spikes
    from ~125 to ~493. Root cause: the consolidator + causal.prune +
    trophic grow/prune mutate nn.Parameter.data in-place; Adam's
    momentum/variance state for those params is computed against
    pre-mutation weights and applied post-mutation, producing
    catastrophic gradient steps.

    P3 fix: AdamW only trains TRUNK params (language.*, sgb.*,
    lambda_motor/mem/thought). Bio params are untracked by Adam, so
    bio-side machinery can mutate them freely without causing
    optimizer-state inconsistency. The trunk and the cognitive sidecar
    are now two separate optimization closures.

    Hypothesis: deterministic step-1500/2000/2500/... spikes vanish.
    """
    c = rcc_bowtie_30m_p2()
    c.rcc_isolate_optimizer = True
    return c


def rcc_bowtie_30m_p4() -> BrainConfig:
    """RCC P3 + per-sample loss clipping (data-robust training).

    P1/P2/P3 all spiked from ppl 125 to 493 at EXACTLY step 1500 — same
    deterministic data ordering puts the same pathological batch there
    every run. Architectural fixes don't address it because the bug is
    in the data, not the model.

    P4 adds adaptive per-sequence loss clipping (Huber-style at the
    sequence level): clip each sequence's loss at 3 * batch median
    before averaging. A single outlier sequence can no longer dominate
    the batch gradient. See docs/STEP1500_INVESTIGATION.md Option 2.
    Used in production at Phi / GPT-3 / Cerebras.
    """
    c = rcc_bowtie_30m_p3()
    c.loss_clip_robust = True
    c.loss_clip_factor = 3.0
    return c


PRESETS = {
    "tiny": tiny, "small": small, "medium": medium,
    "large": large, "xl": xl, "xxl": xxl,
    "pct_30m": pct_30m,
    "synth_30m": synth_30m,
    "synth_30m_bema": synth_30m_bema,
    "rcc_bowtie_30m_p1": rcc_bowtie_30m_p1,
    "rcc_bowtie_30m_p2": rcc_bowtie_30m_p2,
    "rcc_bowtie_30m_p3": rcc_bowtie_30m_p3,
    "rcc_bowtie_30m_p4": rcc_bowtie_30m_p4,
}
