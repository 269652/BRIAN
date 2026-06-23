# Mechanics Library

A documentation-first, machine-readable catalog of reusable neural mechanisms for
language models, expressed in the `.neuro` DSL as `mechanic NAME { ... }` blocks and
parsed by [`neuroslm/dsl/mechanic_parser.py`](neuroslm/dsl/mechanic_parser.py).

Each entry declares **what** it computes (`equation`), **how** it is implemented
(`impl`), **what** is configurable (`params`), and **when** to use it (`when_to_use`),
with paper references and empirical evidence. This catalog is the vocabulary the
auto-evolution / mutation engine draws from when grafting mechanisms into an
architecture. It is validated by
[`tests/dsl/test_mechanics_library.py`](tests/dsl/test_mechanics_library.py) — every
mechanic must parse, carry the load-bearing fields, and be listed here.

**74 mechanics** across three folders:

| Folder | Role | Count |
| --- | --- | --- |
| [`mechanics/`](mechanics) | Computational primitives | 54 |
| [`dynamics/`](dynamics) | Training & optimization | 11 |
| [`structures/`](structures) | Wiring patterns | 9 |

## Mechanics — `mechanics/`

Computational primitives — attention, position, feed-forward, normalization, sparse routing, and alternative sequence mixers.

### attention

| Mechanic | Summary | Spec |
| --- | --- | --- |
| `attention_sink` | Always retains the first n_sink tokens in the KV cache alongside a rolling window, stabilising softmax and enabling infinite-length strea... | [attention_sink.neuro](mechanics/attention_sink.neuro) |
| `differential_attention` | Noise-cancelling attention: each head pair computes two softmax maps and subtracts them | [differential_attention.neuro](mechanics/differential_attention.neuro) |
| `episodic_memory` | External episodic key-value cache: write hidden states, retrieve k nearest neighbours, blend into residual via zero-init α gate (ReZero) | [episodic_memory.neuro](mechanics/episodic_memory.neuro) |
| `flash_attention` | Exact attention computed in SRAM tiles with online softmax — never materializes the N×N score matrix | [flash_attention.neuro](mechanics/flash_attention.neuro) |
| `gqa` | Interpolates between MHA and MQA: g key/value heads shared across n_heads queries, shrinking the KV cache g/n_heads× | [gqa.neuro](mechanics/gqa.neuro) |
| `grid_positions` | Multi-scale sinusoidal grid-cell positional encoding at K φ-ratio scales; provable length-OOD extrapolation via co-prime period tiling | [grid_positions.neuro](mechanics/grid_positions.neuro) |
| `kjpla_phase_lattice` | Per-head phase φ evolving via Kuramoto sync and Josephson inter-layer coupling; phase gates attention logits | [kjpla_phase_lattice.neuro](mechanics/kjpla_phase_lattice.neuro) |
| `logit_soft_cap` | Tanh soft-cap that smoothly bounds attention scores and final logits to ±cap — taming spikes without hard clipping | [logit_soft_cap.neuro](mechanics/logit_soft_cap.neuro) |
| `mla` | Caches a small joint latent for K and V (low-rank), with a separate decoupled RoPE branch, slashing KV-cache size while keeping MHA-grade... | [mla.neuro](mechanics/mla.neuro) |
| `mqa` | All query heads attend to one shared key/value head, collapsing the KV cache to a single head for fast decoding | [mqa.neuro](mechanics/mqa.neuro) |
| `nfo` | Residual block: Kuramoto oscillators on the sequence + Swift-Hohenberg amplitude control + coherence-gated write-back (ReZero) | [nfo.neuro](mechanics/nfo.neuro) |
| `qk_norm` | Normalize Q and K (per-head) before Q·Kᵀ so attention logits cannot blow up during training | [qk_norm.neuro](mechanics/qk_norm.neuro) |
| `semantic_turbulence` | Three physics-inspired mechanisms: RG cascade (Kolmogorov multi-scale), GPE superfluid field (coherence), NT criticality (Beggs-Plenz bra... | [semantic_turbulence.neuro](mechanics/semantic_turbulence.neuro) |
| `sliding_window_attention` | Restricts each query to a fixed-size local window of recent keys, giving linear-in-sequence cost and a bounded KV cache | [sliding_window_attention.neuro](mechanics/sliding_window_attention.neuro) |
| `tonnetz_attention` | Causal attention with a toroidal (Tonnetz) mask: tokens attend to harmonically-related positions | [tonnetz_attention.neuro](mechanics/tonnetz_attention.neuro) |

### feedforward

| Mechanic | Summary | Spec |
| --- | --- | --- |
| `gated_mlp` | Generic gated-linear-unit FFN with a configurable gate activation (SwiGLU/GeGLU/ReGLU as special cases) | [gated_mlp.neuro](mechanics/gated_mlp.neuro) |
| `geglu` | GELU-gated FFN: one projection is GELU-activated and gates another before down-projection | [geglu.neuro](mechanics/geglu.neuro) |
| `swiglu` | Swish-gated FFN: one projection gates another via SiLU before the down-projection | [swiglu.neuro](mechanics/swiglu.neuro) |

### normalization

| Mechanic | Summary | Spec |
| --- | --- | --- |
| `deepnorm` | Up-scales the residual branch by α and down-scales sublayer init by β so post-norm transformers train at extreme depth | [deepnorm.neuro](mechanics/deepnorm.neuro) |
| `layernorm` | Normalizes each token's features to zero mean and unit variance — then applies a learned gain and bias | [layernorm.neuro](mechanics/layernorm.neuro) |
| `rmsnorm` | Re-scales activations by their root-mean-square only — LayerNorm without mean subtraction or bias | [rmsnorm.neuro](mechanics/rmsnorm.neuro) |

### physics

| Mechanic | Summary | Spec |
| --- | --- | --- |
| `liouville_symplectic` | Stoermer-Verlet leapfrog step on a learned Hamiltonian; Noether loss penalises energy non-conservation | [liouville_symplectic.neuro](mechanics/liouville_symplectic.neuro) |
| `pontryagin_topo_charge` | Skyrmion topological-charge diagnostic on attention heads; optional soft penalty toward integer Q | [pontryagin_topo_charge.neuro](mechanics/pontryagin_topo_charge.neuro) |

### position

| Mechanic | Summary | Spec |
| --- | --- | --- |
| `alibi` | Adds a head-specific linear penalty proportional to query-key distance directly to attention logits | [alibi.neuro](mechanics/alibi.neuro) |
| `nope` | No explicit position signal; the causal mask alone lets the network infer token order | [nope.neuro](mechanics/nope.neuro) |
| `rope` | Encodes absolute position by rotating Q/K pairs; dot products become relative-position dependent | [rope.neuro](mechanics/rope.neuro) |

### regularizer

| Mechanic | Summary | Spec |
| --- | --- | --- |
| `adaptive_mixture` | Closed-loop PI controller that adjusts the chat/prose training-data ratio to keep OOD entropy at target | [adaptive_mixture.neuro](mechanics/adaptive_mixture.neuro) |
| `cdga` | Gradient surgery: subtract the component of the training gradient that conflicts with an OOD anchor gradient | [cdga.neuro](mechanics/cdga.neuro) |
| `cmd` | Penalise Jensen-Shannon divergence between two output heads; forces complementary predictions | [cmd.neuro](mechanics/cmd.neuro) |
| `dar` | Re-weight training samples adversarially to equalise source distributions; prevents shortcut learning | [dar.neuro](mechanics/dar.neuro) |
| `freq_balance` | Reweight per-token CE loss by inverse token frequency ratio between training and OOD distributions | [freq_balance.neuro](mechanics/freq_balance.neuro) |
| `isotropy` | Push the token-embedding Gram matrix toward the identity; prevents rank collapse | [isotropy.neuro](mechanics/isotropy.neuro) |
| `pcc` | InfoNCE contrastive loss between a token's representation and its future self; builds predictive features | [pcc.neuro](mechanics/pcc.neuro) |

### routing

| Mechanic | Summary | Spec |
| --- | --- | --- |
| `expert_choice_routing` | Inverted MoE routing: each expert picks its top-capacity tokens, giving exact load balance with no auxiliary loss | [expert_choice_routing.neuro](mechanics/expert_choice_routing.neuro) |
| `mixture_of_depths` | Route only the top-C tokens through attention+FFN; unrouted tokens get identity residual | [mixture_of_depths.neuro](mechanics/mixture_of_depths.neuro) |
| `multi_cortex` | Frozen pretrained LM experts route logits through a per-token context gate; trunk learns a delta correction; KL distillation transfers ex... | [multi_cortex.neuro](mechanics/multi_cortex.neuro) |
| `shared_expert` | DeepSeekMoE shared-expert isolation: a few always-on experts capture common knowledge so routed experts specialise | [shared_expert.neuro](mechanics/shared_expert.neuro) |
| `sparse_moe` | Sparsely-gated MoE: a router sends each token to its top-k experts, with a load-balancing auxiliary loss | [sparse_moe.neuro](mechanics/sparse_moe.neuro) |

### sequence_mixer

| Mechanic | Summary | Spec |
| --- | --- | --- |
| `gated_linear_attention` | Linear attention plus a learned per-channel forget gate, trained in a chunked parallel form | [gated_linear_attention.neuro](mechanics/gated_linear_attention.neuro) |
| `hyena` | Subquadratic attention substitute: interleaved implicit long convolutions and elementwise gating | [hyena.neuro](mechanics/hyena.neuro) |
| `linear_attention` | Attention without softmax: a kernel feature map φ makes it an O(N) recurrent sum of states | [linear_attention.neuro](mechanics/linear_attention.neuro) |
| `mamba_ssm` | Selective SSM: input-dependent A,B,C run as an O(N) hardware-aware parallel scan | [mamba_ssm.neuro](mechanics/mamba_ssm.neuro) |
| `retnet` | Retention: fixed exponential decay γ gives one op with parallel, recurrent, and chunkwise forms | [retnet.neuro](mechanics/retnet.neuro) |

### training_dynamics

| Mechanic | Summary | Spec |
| --- | --- | --- |
| `allostasis` | Synthetic HPA axis: integrates multi-modal stress over ~40 steps to damp NE runaway, trophic growth, and LR during sustained crisis | [allostasis.neuro](mechanics/allostasis.neuro) |
| `bures_manifold_alignment` | Sliced Wasserstein₂ between trunk and expert representation distributions; gradient flows into trunk to prevent erank collapse | [bures_manifold_alignment.neuro](mechanics/bures_manifold_alignment.neuro) |
| `cosine_lm_head` | Replace linear LM head with cosine-similarity logits: z_i = τ·cos(h, W_i); eliminates magnitude as a domain-confidence DoF | [cosine_lm_head.neuro](mechanics/cosine_lm_head.neuro) |
| `divisive_grad_norm` | Replace hard gradient clip with smooth divisive normalisation: g' = g·c/√(c²+‖g‖²) | [divisive_grad_norm.neuro](mechanics/divisive_grad_norm.neuro) |
| `fisher_rao_retrieval` | Fisher-Rao information metric on the stalk of each brain-region simplex; precision-weighted inner product replaces Euclidean similarity | [fisher_rao_retrieval.neuro](mechanics/fisher_rao_retrieval.neuro) |
| `gif` | Geometric Information Funnel: 7 interlocking sub-mechanisms that close the train-PPL / OOD-PPL gap | [gif.neuro](mechanics/gif.neuro) |
| `loss_variance_damping` | BCM-rule metaplastic damping: reduces lr_eff when loss variance exceeds a healthy reference σ_ref | [loss_variance_damping.neuro](mechanics/loss_variance_damping.neuro) |
| `mspcc` | Generalises single-waist VBB into a per-layer cascade: MDRV-VBB free energy at every adjacent layer pair, geometrically weighted toward t... | [mspcc.neuro](mechanics/mspcc.neuro) |
| `predictive_coding_head` | PC reentry: motor hidden state predicts sensory hidden state; prediction error adds to loss; NT-gated gradient flows through both populat... | [predictive_coding_head.neuro](mechanics/predictive_coding_head.neuro) |
| `riemannian_motor_projection` | Hyperbolic tanh-projection of h_motor: h_proj = ρ·tanh(‖h‖/ρ)·h/‖h‖ with learnable curvature R=1/ρ²; caps magnitude geometrically | [riemannian_motor_projection.neuro](mechanics/riemannian_motor_projection.neuro) |
| `vbb` | Friston-style variational free energy at the bowtie bottleneck: F = KL(q‖p) − E_q[log p(x\|z)] | [vbb.neuro](mechanics/vbb.neuro) |

## Dynamics — `dynamics/`

Training & optimization — optimizers, learning-rate schedules, losses, and regularizers.

### optimizer

| Mechanic | Summary | Spec |
| --- | --- | --- |
| `adamw` | Adam with weight decay decoupled from the gradient-based update (regularisation — not L2 in the loss) | [adamw.neuro](dynamics/adamw.neuro) |
| `lion` | EvoLved sIgn mOmeNtum: update direction is the sign of an interpolated momentum — one state buffer and uniform step magnitude | [lion.neuro](dynamics/lion.neuro) |
| `muon` | MomentUm Orthogonalised by Newton-schulz: orthogonalise each 2D weight's update before applying it | [muon.neuro](dynamics/muon.neuro) |

### regularizer

| Mechanic | Summary | Spec |
| --- | --- | --- |
| `dropout` | Randomly zero a fraction p of activations during training (inverted-dropout scaling) to prevent co-adaptation; identity at eval | [dropout.neuro](dynamics/dropout.neuro) |
| `ema_weights` | Maintain a shadow copy of the weights as an exponential moving average (Polyak averaging) and evaluate with it for lower-variance often-b... | [ema_weights.neuro](dynamics/ema_weights.neuro) |
| `gradient_clipping` | Rescale the whole gradient so its global norm never exceeds max_norm — caps exploding-gradient steps without changing direction | [gradient_clipping.neuro](dynamics/gradient_clipping.neuro) |
| `label_smoothing` | Replace the one-hot target with a soft target (1−smoothing on the truth; the rest spread over other classes) to curb overconfidence | [label_smoothing.neuro](dynamics/label_smoothing.neuro) |
| `weight_decay` | Decoupled L2 weight shrinkage applied to the parameter at each step — typically excluding norm gains / biases / embeddings | [weight_decay.neuro](dynamics/weight_decay.neuro) |
| `z_loss` | Penalise the squared log-partition (logsumexp) of softmax logits to keep them from drifting — stabilises router and output softmaxes | [z_loss.neuro](dynamics/z_loss.neuro) |

### schedule

| Mechanic | Summary | Spec |
| --- | --- | --- |
| `cosine_schedule` | Linear warmup to peak lr then half-cosine decay down to min_lr_ratio·peak by total_steps | [cosine_schedule.neuro](dynamics/cosine_schedule.neuro) |
| `wsd_schedule` | Warmup → long constant (stable) plateau → short final decay; the stable phase can be extended and checkpoints branched off mid-run | [wsd_schedule.neuro](dynamics/wsd_schedule.neuro) |

## Structures — `structures/`

Wiring patterns — how blocks compose: residual stream, pre/post/sandwich norm, parallel & MoE/MoD blocks, weight tying, init.

### normalization

| Mechanic | Summary | Spec |
| --- | --- | --- |
| `sandwich_norm` | Wrap each sublayer in two norms — pre-norm on the input AND post-norm on the branch output before the residual add | [sandwich_norm.neuro](structures/sandwich_norm.neuro) |

### structure

| Mechanic | Summary | Spec |
| --- | --- | --- |
| `depth_scaled_init` | Scale residual-branch output projections by 1/sqrt(2N) at init so the residual stream variance stays bounded across N layers | [depth_scaled_init.neuro](structures/depth_scaled_init.neuro) |
| `mod_block` | Mixture-of-Depths: a router selects a capacity-limited subset of tokens to process; the rest skip the block via residual | [mod_block.neuro](structures/mod_block.neuro) |
| `moe_block` | Transformer block whose dense FFN sublayer is replaced by a routed Mixture-of-Experts layer (+ optional shared expert) | [moe_block.neuro](structures/moe_block.neuro) |
| `parallel_block` | Run attention and FFN on the SAME normalized input in parallel and sum both branches into the residual stream | [parallel_block.neuro](structures/parallel_block.neuro) |
| `postnorm_block` | Post-LN transformer block (Vaswani 2017): normalize the residual sum after each sublayer | [postnorm_block.neuro](structures/postnorm_block.neuro) |
| `prenorm_block` | Pre-LN transformer block: normalize inside each residual branch so the residual stream stays unnormalized | [prenorm_block.neuro](structures/prenorm_block.neuro) |
| `residual_stream` | Treat the residual connection as a shared linear bus that every sublayer reads from and writes to additively | [residual_stream.neuro](structures/residual_stream.neuro) |
| `weight_tying` | Share the input embedding matrix with the output (unembedding) projection so they are one set of weights | [weight_tying.neuro](structures/weight_tying.neuro) |

