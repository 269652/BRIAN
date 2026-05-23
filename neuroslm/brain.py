"""NeuroSLM Brain — modular, DNA-free neuroscience-inspired language model.

Each brain area is self-contained (inherits BrainModule) and can be toggled:
    brain.hippo.disable()           # bypass hippocampus
    brain.pfc.enable()              # restore PFC
    brain.cfg.neural_topology = 'baseline'  # language-only path

Neural topology modes:
    'baseline' — vanilla transformer only (ablation / fast inference)
    'full'     — routes through all enabled modules (default)

Training insights → memory:
    After every forward pass with targets, high-surprise × high-comprehension
    observations are written into the RelationalMemoryGraph as semantic insights.

Memory consolidation:
    Every cfg.consolidate_every steps, episodic memories are clustered into
    abstract semantic / schema nodes and causal rules are extracted.

Memory checkpoints (.mem files):
    brain.save_memory_checkpoint(path) / brain.load_memory_checkpoint(path)
    Stored in lfs_checkpoints/ and tracked via Git LFS.
"""
from __future__ import annotations
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BrainConfig
from .modules.language import LanguageCortex
from .modules.sensory import (TextSensoryCortex, SensoryFrameEncoder,
                              TopicClassifier, TOPIC_MATH, TOPIC_REASONING,
                              TOPIC_LANGUAGE, TOPIC_DEFAULT)
from .modules.math import MathCortex
from .modules.reasoning import ReasoningCortex
from .modules.association import AssociationCortex
from .modules.world_model import WorldModel
from .modules.self_model import SelfModel
from .modules.workspace import GlobalWorkspace
from .modules.hippocampus import Hippocampus
from .modules.dmn import DefaultModeNetwork
from .modules.pfc import PrefrontalCortex
from .modules.basal_ganglia import BasalGanglia
from .modules.forward_model import ForwardModel
from .modules.evaluator import Evaluator
from .modules.motor import MotorCortex, ACTION_NAMES, ACTION_INDEX
from .modules.thalamus import Thalamus
from .modules.critic import SubconsciousCritic
from .modules.qualia import QualiaState
from .modules.thought_transformer import ThoughtTransformer
from .modules.consciousness import ConsciousnessMetrics, estimate_fiedler
from .modules.cortical_column import CorticalSheet
from .modules.entorhinal import EntorhinalCortex
from .modules.claustrum import Claustrum
from .modules.cerebellum import Cerebellum
from .modules.neural_geometry import NeuralGeometryEngine
from .modules.boundary import BoundaryDetector

from .neurochem import (
    TransmitterSystem, NT_NAMES,
    ReceptorBank, NTShapeRegistry, Receptor, GPCRBank,
    Projection, ProjectionGraph,
    VTA, NucleusAccumbens, LocusCoeruleus, RapheNuclei, BasalForebrain,
    SubstantiaNigra, PeriaqueductalGray, HypothalamicCRH,
    Homeostasis, ReuptakeSystem, ReceptorAdaptation,
    GatedProjectionGraph, MesolimbicCircuit, PlasticityGate,
)
from .neurochem.growth import TrophicSystem
from .learning import LearningLayer
from .learned_opt import LearnedBackprop


class Brain(nn.Module):
    def __init__(self, cfg: BrainConfig):
        super().__init__()
        self.cfg = cfg

        # ---- Baseline: vanilla transformer only ----
        if getattr(cfg, 'baseline', False):
            _bl_layers = getattr(cfg, 'baseline_lang_layers', 0) or cfg.lang_layers
            self.language = LanguageCortex(
                cfg.vocab_size, cfg.d_hidden, cfg.d_sem,
                _bl_layers, cfg.lang_heads, cfg.lang_ctx,
                n_kv_heads=cfg.lang_kv_heads,
                gradient_checkpointing=cfg.gradient_checkpointing,
                dropout=getattr(cfg, 'dropout', 0.0),
                baseline=True)
            self._baseline = True
            return
        self._baseline = False

        # ---- Language cortex (SRC-TEH Tier-1 Shared Reading Cortex) ----
        from .neurochem.transmitters import N_NT
        _src_teh = bool(getattr(cfg, 'enable_src_teh', False))
        _mid_tap = getattr(cfg, 'mid_trunk_tap_layer', 0) or (cfg.lang_layers // 2)
        self.language = LanguageCortex(
            cfg.vocab_size, cfg.d_hidden, cfg.d_sem,
            cfg.lang_layers, cfg.lang_heads, cfg.lang_ctx,
            n_kv_heads=cfg.lang_kv_heads,
            n_nt=N_NT,
            hebbian_rank=getattr(cfg, 'hebbian_rank', 8),
            gradient_checkpointing=cfg.gradient_checkpointing,
            mod_capacity=getattr(cfg, 'mod_capacity', 1.0),
            baseline=False,
            enable_memory_xattn=bool(getattr(cfg, 'enable_memory_xattn', False)),
            n_memory_xattn_layers=int(getattr(cfg, 'n_memory_xattn_layers', 2)),
            mid_trunk_tap_layer=int(_mid_tap),
            use_attention_pool=_src_teh,
            dropout=getattr(cfg, 'dropout', 0.0),
            use_predictive_coding_trunk=bool(getattr(cfg, 'use_predictive_coding_trunk', False)),
            pct_mode=str(getattr(cfg, 'pct_mode', 'loss_only')),
            pct_feedback_alpha=float(getattr(cfg, 'pct_feedback_alpha', 0.05)),
            pct_hidden_mult=float(getattr(cfg, 'pct_hidden_mult', 0.5)),
            pct_lambda_fe=float(getattr(cfg, 'pct_lambda_fe', 0.1)),
            pct_include_embedding_predictor=bool(getattr(cfg, 'pct_include_embedding_predictor', True)),
            )

        # ── ReZero-style zero-init gates on module → LM forward injections ──
        # Each scalar λ starts at 0, so at init the LM behaves identically to
        # the pure isolated-trunk model (no module contribution, no awakening
        # discontinuity). LM gradient then grows each λ only as far as that
        # specific injection actually improves next-token prediction. See §5.3.
        import torch as _t
        self.lambda_motor   = nn.Parameter(_t.zeros(1))  # gates motor_lang_bias
        self.lambda_mem     = nn.Parameter(_t.zeros(1))  # gates memory_kv injection
        self.lambda_thought = nn.Parameter(_t.zeros(1))  # gates from_sem thought conditioning

        # ---- Sensory + association ----
        self.sensory     = TextSensoryCortex(cfg.d_sem)
        self.association = AssociationCortex(cfg.d_sem)

        # ---- Global workspace ----
        self.gws = GlobalWorkspace(cfg.d_sem, cfg.gws_slots, cfg.gws_heads)

        # ---- Thalamus ----
        self.thalamus = Thalamus(cfg.d_sem, cfg.d_hidden)

        # ---- World + self models ----
        self.world  = WorldModel(cfg.d_sem, cfg.d_hidden, cfg.world_layers)
        self.self_m = SelfModel(cfg.d_sem, cfg.bg_action_dim, cfg.n_neuromods,
                                cfg.d_hidden, cfg.self_layers)

        # ---- Core bio modules (all inherit BrainModule) ----
        self.hippo  = Hippocampus(cfg.d_sem, cfg.hippo_capacity,
                                  cfg.hippo_topk, cfg.hippo_sparse_k)
        self.dmn    = DefaultModeNetwork(cfg.d_sem, cfg.gws_slots,
                                         cfg.dmn_layers,
                                         topology=cfg.neural_topology)
        self.pfc    = PrefrontalCortex(cfg.d_sem, cfg.pfc_layers, cfg.pfc_heads)
        self.bg     = BasalGanglia(cfg.d_sem, cfg.bg_action_dim, cfg.bg_n_candidates)
        self.critic = SubconsciousCritic(cfg.d_sem)

        # ---- Forward / evaluator / motor ----
        self.forward_m = ForwardModel(cfg.d_sem, cfg.bg_action_dim, cfg.forward_layers)
        self.evaluator = Evaluator(cfg.d_sem, len(NT_NAMES))
        self.motor     = MotorCortex(cfg.bg_action_dim, cfg.d_sem, cfg.d_hidden)

        # ---- Neuroanatomical structures ----
        self.cortical_sheet = CorticalSheet(cfg.d_sem, n_columns=4, n_minicolumns=8)
        self.entorhinal     = EntorhinalCortex(cfg.d_sem, n_modules=4,
                                               cells_per_module=32, n_places=64)
        self.claustrum      = Claustrum(cfg.d_sem, n_modalities=8)
        self.cerebellum     = Cerebellum(cfg.d_sem, expansion=4)
        self.neural_geometry = NeuralGeometryEngine(cfg.d_sem, n_fractal_levels=3)

        # ---- Cognitive modules ----
        self.qualia           = QualiaState(cfg.d_sem, len(NT_NAMES))
        self.thought_transformer = ThoughtTransformer(
            d_sem=cfg.d_sem, n_thought_tokens=4, n_layers=2, n_heads=4)
        self.consciousness    = ConsciousnessMetrics(d_sem=cfg.d_sem)

        # ---- Subcortical / emotional brain areas ----
        from .modules.amygdala          import Amygdala
        from .modules.anterior_cingulate import AnteriorCingulateCortex
        from .modules.insula             import Insula
        from .neurochem.lateral_habenula import LateralHabenula

        self.amygdala = (
            Amygdala(cfg.d_sem,
                     d_emotion=getattr(cfg, 'amygdala_d_emotion', 32),
                     n_nt=len(NT_NAMES))
            if getattr(cfg, 'enable_amygdala', True) else None
        )
        self.acc = (
            AnteriorCingulateCortex(cfg.d_sem, n_nt=len(NT_NAMES))
            if getattr(cfg, 'enable_acc', True) else None
        )
        self.insula = (
            Insula(cfg.d_sem, n_nt=len(NT_NAMES))
            if getattr(cfg, 'enable_insula', True) else None
        )
        self.lhb = (
            LateralHabenula(n_nt=len(NT_NAMES))
            if getattr(cfg, 'enable_lateral_habenula', True) else None
        )

        # ---- Contrastive Predictive Coding (novel training objective) ----
        from .intelligence.contrastive_predictive_coding import ContrastivePredictiveCoding
        self.cpc = (
            ContrastivePredictiveCoding(
                d_model=cfg.d_sem,
                max_steps=getattr(cfg, 'cpc_steps', 5),
                n_negatives=getattr(cfg, 'cpc_negatives', 32),
            ) if getattr(cfg, 'enable_cpc', False) else None
        )

        # ---- Memory systems ----
        from .memory.episodic import EpisodicMemory
        from .memory.consolidated import ConsolidatedMemory
        from .memory.narrative import NarrativeBuffer, NarrativeSystem
        from .memory.mesolimbic import MesolimbicTagger
        from .memory.hippocampal import HippocampalEnrichment
        from .memory.relational_graph import RelationalMemoryGraph
        from .memory.causal import CausalRuleStore
        from .memory.comprehension_gate import ComprehensionGate
        from .memory.consolidation import MemoryConsolidator

        self.episodic         = EpisodicMemory(maxlen=2048)
        self.consolidated     = ConsolidatedMemory()
        self.narrative_self   = NarrativeBuffer(maxlen=2048)
        self.narrative_world  = NarrativeBuffer(maxlen=2048)
        self.narrative_system = NarrativeSystem(cfg.d_sem)
        self.mesolimbic_tagger = MesolimbicTagger()
        self.hippocampal      = HippocampalEnrichment(self.consolidated)
        self.relational_memory = RelationalMemoryGraph(max_nodes=8192)
        self.causal           = CausalRuleStore(merge_threshold=0.86, min_support=2)
        self.comprehension_gate = ComprehensionGate(
            threshold=0.05, target_write_rate=0.10)
        self.consolidator     = MemoryConsolidator(
            self.relational_memory, self.causal)

        self._last_memory_id = None
        self._global_step    = 0
        # Auxiliary-loss scale (0..1) — set per-step by train.py to gate every
        # non-LM loss together during infancy. 1.0 = full aux weights.
        self._aux_w_scale    = 1.0
        # Infancy flag — when True, structural/homeostatic side-effects
        # (trophic update, vesicle dynamics) are suppressed so they don't
        # ride on random-init gradients.
        # NOTE: hard infancy gates are being phased out in favour of the
        # continuous Maturity Index (`maturity`); this flag is now derived
        # from `maturity < 0.3` inside update_maturity. Default is False
        # so test/inference paths start "mature" (full module fade-in),
        # while train.py drives it back to True via update_maturity()
        # whenever the LM loss is still near random init.
        self._infancy        = False
        # ── Maturity Index (MAT virtual protein) ──────────────────────────
        # Continuous scalar in [0, 1] computed as 1 - L_lm / L_random and
        # smoothed with an EMA. Drives expert fade-in, MoD compute gating,
        # ε-routing strength, and GABA tolerance. Updated by train.py via
        # `brain.update_maturity(lm_loss)`. Defaults to 1.0 so the expert
        # cortices use full-weight residuals outside training (tests etc.).
        self.register_buffer("maturity", torch.ones(()))
        self._maturity_ema_alpha = 0.05   # EMA smoothing (slow integration)
        # Cached previous-step module outputs (for ActualCausationHead's
        # (t, t+1) transition). Populated post-awakening in forward_lm.
        self._prev_module_outputs: torch.Tensor | None = None
        # Strong-α causal edges observed last step (for κ_cause vesicle emission)
        self._last_causal_edges: list = []

        # ── BRIAN narrative + causal stack (post-awakening only) ────────
        # ActualCausationHead, PersonalityVector, and SleepCycle are
        # constructed eagerly so their parameters participate in the model
        # state-dict and checkpoint round-trip cleanly. All three are
        # **inert during infancy** — their forward effects only kick in
        # once `_maturation_awakened` is True (set by train.py).
        from .modules.actual_causation import ActualCausationHead
        from .modules.survival_causal import SurvivalCausalHead
        from .modules.sensory import SensoryVAE
        from .neurochem.personality import PersonalityVector
        from .memory.sleep_cycle import SleepCycle
        self.actual_causation = ActualCausationHead(
            n_modules=8, d_sem=cfg.d_sem)
        self.personality = PersonalityVector(enable=True)
        self.sleep_cycle = SleepCycle(
            d_sem=cfg.d_sem,
            sleep_period_steps=5000,
            enable=True)
        # ── Cognitive Closure stack — embodied survival loop ─────────────
        # SensoryVAE compresses (vision, affordance, homeostatic) grid
        # frames into the d_sem manifold. SurvivalCausalHead estimates
        # how much each action contributes to next-step survival vars.
        # Both are part of the model state-dict; their gradient effects
        # are gated through `_aux_w_scale` like every other aux loss.
        self.sensory_vae = SensoryVAE(d_sem=cfg.d_sem)
        self.survival_causal = SurvivalCausalHead(
            d_action=cfg.bg_action_dim if hasattr(cfg, 'bg_action_dim') else 256)
        # Persistent survival-state vector (energy, hydration, integrity).
        # Initialised to 1.0; ticked via Homeostasis.step on each embodied
        # forward pass. Tests can mutate it directly via set_survival().
        self.register_buffer(
            "survival_state",
            torch.ones(3, dtype=torch.float32))

        # ---- Intelligence flow ----
        from .intelligence.reflection import SpontaneousReflection
        from .intelligence.metrics import IntelligenceMetrics
        from .intelligence.orchestrator import NeuralOrchestrator
        from .intelligence.oscillations import NeuralOscillationTracker

        self.reflection = SpontaneousReflection(cfg.d_sem)
        self.metrics    = IntelligenceMetrics()
        from .intelligence.orchestrator import (
            STAGE_SENSORY, STAGE_THALAMUS, STAGE_STATE_MODELS, STAGE_SUBCORTICAL,
            STAGE_QUALIA, STAGE_GWS, STAGE_MEMORY, STAGE_COGNITIVE_CTL,
            STAGE_EXECUTIVE, STAGE_CONSCIOUSNESS, STAGE_MOTOR,
        )
        self._ORCH_STAGES = dict(
            sensory=STAGE_SENSORY, thalamus=STAGE_THALAMUS,
            state_models=STAGE_STATE_MODELS, subcortical=STAGE_SUBCORTICAL,
            qualia=STAGE_QUALIA, gws=STAGE_GWS, memory=STAGE_MEMORY,
            cognitive_ctl=STAGE_COGNITIVE_CTL, executive=STAGE_EXECUTIVE,
            consciousness=STAGE_CONSCIOUSNESS, motor=STAGE_MOTOR,
        )
        # All modules that can participate in orchestrated routing
        _orch_module_names = [
            'sensory', 'association', 'thalamus',
            'world', 'self_m',
            'amygdala', 'insula',
            'qualia',
            'gws', 'neural_geometry',
            'hippo', 'entorhinal', 'cerebellum',
            'pfc', 'acc',
            'bg', 'forward_m', 'evaluator',
            'dmn', 'thought_transformer', 'claustrum',
            'motor',
        ]
        self.orchestrator = NeuralOrchestrator(
            cfg.d_sem,
            _orch_module_names,
            n_heads=4, baseline=False)
        self.oscillation_tracker = NeuralOscillationTracker(
            cfg.d_sem, n_regions=8, window_size=64)
        self.oscillation_tracker.register_regions([
            'language', 'pfc', 'dmn', 'hippo', 'world',
            'cerebellum', 'gws', 'motor',
        ])

        # Spectral boundary screening (Fiedler MIP detector)
        # Estimates the algebraic connectivity λ₁ of the inter-module graph
        # to track Integrated Information irreducibility (IIT 4.0).  Low λ₁
        # signals near-disconnection → trophic homeostasis triggers extra BDNF.
        self.boundary_detector = BoundaryDetector(
            n_modules=max(32, len(_orch_module_names)),
            ema_decay=0.95, recompute_every=10)

        # ---- Novel cognitive modules: HyperGraph, EntityStore, ToM, Active Inference ----
        from .memory.hypergraph import MemoryHyperGraph
        from .memory.entity_store import EntityStore
        from .modules.theory_of_mind import TheoryOfMindModule
        from .intelligence.active_inference import FreeEnergyProcessor
        from .neurochem.vesicles import VesiclePool

        self.hypergraph = (
            MemoryHyperGraph(d_emb=cfg.d_sem,
                             max_nodes=getattr(cfg, 'hippo_capacity', 4096) * 4)
            if getattr(cfg, 'enable_hypergraph', True) else None
        )
        self.entity_store = (
            EntityStore(d_emb=cfg.d_sem,
                        d_style=getattr(cfg, 'entity_d_style', 64))
            if getattr(cfg, 'enable_entity_store', True) else None
        )
        self.tom = (
            TheoryOfMindModule(
                d_sem=cfg.d_sem,
                d_style=getattr(cfg, 'tom_d_style', 64),
                d_belief=6,
                n_heads=getattr(cfg, 'tom_n_heads', 4),
            ) if getattr(cfg, 'enable_tom', False) else None
        )
        self.active_inference = (
            FreeEnergyProcessor(
                d_sem=cfg.d_sem,
                n_layers=getattr(cfg, 'active_inf_layers', 3),
            ) if getattr(cfg, 'enable_active_inference', False) else None
        )
        # VesiclePool: slow long-range neuromodulation via content packets
        _n_brain_modules = 8  # approximate count for migration graph
        self.vesicle_pool = (
            VesiclePool(
                d_sem=cfg.d_sem,
                n_modules=_n_brain_modules,
                n_vesicles=getattr(cfg, 'n_vesicles', 32),
                lifetime=getattr(cfg, 'vesicle_lifetime', 16),
            ) if getattr(cfg, 'enable_vesicles', False) else None
        )

        # RSSM: replace WorldModel with RSSM if enabled
        self._use_rssm = getattr(cfg, 'enable_rssm', False)
        if self._use_rssm:
            from .modules.world_model import RecurrentStateSpaceModel
            self.world = RecurrentStateSpaceModel(
                d_sem=cfg.d_sem,
                d_hidden=cfg.d_hidden,
                n_layers=cfg.world_layers,
                n_cats=getattr(cfg, 'rssm_n_cats', 8),
                d_cat=getattr(cfg, 'rssm_d_cat', 16),
            )

        self._active_entity_id: str | None = None
        self.entities: dict = {}

        # ---- Novel ML / neuroscience modules ----
        from .modules.active_dendrite        import ActiveDendriteLayer
        from .modules.dynamic_routing_moe    import DynamicRoutingMoE
        from .modules.htm_layer              import HTMLayer
        from .modules.relational_attention   import RelationalAttentionBlock
        from .modules.fast_weight            import FastWeightLayer
        from .modules.differentiable_memory  import DifferentiableMemory
        from .modules.phase_modulated_attention import PhaseModulatedAttention
        from .modules.neurogenesis           import NeurogenesisLayer
        from .modules.predictive_coding_loss import PredictiveCodingLoss
        from .modules.causal_inference       import CausalInferenceModule

        self.active_dendrite = (
            ActiveDendriteLayer(cfg.d_sem, d_context=cfg.d_sem, n_branches=8, k_winners=2)
            if getattr(cfg, 'enable_active_dendrite', False) else None
        )
        self.dynamic_routing_moe = (
            DynamicRoutingMoE(cfg.d_sem,
                              n_experts=getattr(cfg, 'moe_experts', 8),
                              top_k=getattr(cfg, 'moe_top_k', 2))
            if getattr(cfg, 'enable_dynamic_routing_moe', False) else None
        )
        self.htm = (
            HTMLayer(cfg.d_sem, n_scales=3,
                     sparsity_k=max(1, cfg.d_sem // 4))
            if getattr(cfg, 'enable_htm', False) else None
        )
        self.relational_attn = (
            RelationalAttentionBlock(cfg.d_sem, n_heads=cfg.gws_heads)
            if getattr(cfg, 'enable_relational_attention', False) else None
        )
        self.fast_weight = (
            FastWeightLayer(cfg.d_sem, decay=0.95, base_eta=0.1, n_heads=4)
            if getattr(cfg, 'enable_fast_weight', False) else None
        )
        self.diff_memory = (
            DifferentiableMemory(memory_size=128, d_model=cfg.d_sem)
            if getattr(cfg, 'enable_differentiable_memory', False) else None
        )
        self.phase_attn = (
            PhaseModulatedAttention(cfg.d_sem, n_heads=cfg.lang_heads)
            if getattr(cfg, 'enable_phase_modulated_attention', False) else None
        )
        self.neurogenesis = (
            NeurogenesisLayer(cfg.d_sem,
                              max_neurons=min(cfg.d_sem * 4, 2048))
            if getattr(cfg, 'enable_neurogenesis', False) else None
        )
        self.pred_coding = (
            PredictiveCodingLoss(cfg.d_sem, n_scales=3)
            if getattr(cfg, 'enable_predictive_coding_loss', False) else None
        )
        self.causal_module = (
            CausalInferenceModule(cfg.d_sem, n_vars=8, d_causal=32)
            if getattr(cfg, 'enable_causal_inference', False) else None
        )

        # ---- Neurochemistry ----
        self.transmitters         = TransmitterSystem()
        self.vta                  = VTA()
        self.nacc                 = NucleusAccumbens()
        self.lc                   = LocusCoeruleus()
        self.raphe                = RapheNuclei()
        self.nbm                  = BasalForebrain()
        self.substantia_nigra     = SubstantiaNigra()
        self.pag                  = PeriaqueductalGray()
        self.hypothalamic_crh     = HypothalamicCRH()
        self.homeostasis          = Homeostasis()
        self.reuptake             = ReuptakeSystem()
        self.receptor_adaptation  = ReceptorAdaptation()
        self.gated_projections    = GatedProjectionGraph()
        self.mesolimbic           = MesolimbicCircuit(d_state=cfg.d_sem)
        self.plasticity_gate      = PlasticityGate()

        # NT shape registry (protein-shape matching for receptor banks)
        self.nt_shapes = NTShapeRegistry()

        # Receptor banks per region
        self.rcpt_pfc = ReceptorBank([
            Receptor("DA",   sign=+1, weight=0.6),
            Receptor("5HT",  sign=+1, weight=0.3),
            Receptor("ACh",  sign=+1, weight=0.4),
            Receptor("GABA", sign=-1, weight=0.4),
        ])
        self.rcpt_hippo = ReceptorBank([
            Receptor("ACh", sign=+1, weight=0.5),
            Receptor("Glu", sign=+1, weight=0.4),
        ])
        self.rcpt_bg = ReceptorBank([
            Receptor("DA",   sign=+1, weight=0.7),
            Receptor("GABA", sign=-1, weight=0.5),
        ])
        self.rcpt_thal = ReceptorBank([
            Receptor("NE",   sign=+1, weight=0.5),
            Receptor("GABA", sign=-1, weight=0.3),
        ])
        self.rcpt_lang = ReceptorBank([
            Receptor("ACh", sign=+1, weight=0.3),
            Receptor("eCB", sign=-1, weight=0.3),
        ])
        self.rcpt_dmn = ReceptorBank([
            Receptor("5HT", sign=-1, weight=0.4),
            Receptor("ACh", sign=-1, weight=0.2),
        ])
        for bank in [self.rcpt_pfc, self.rcpt_hippo, self.rcpt_bg,
                     self.rcpt_thal, self.rcpt_lang, self.rcpt_dmn]:
            bank.bind_registry(self.nt_shapes)

        # GPCR metabotropic bank: sustained NT window for slow modulation
        # ACh gate → widens DG sparse_k (more encoding winners under high ACh)
        # NE arousal → raises CALM exit threshold (forces full-depth processing)
        self.gpcr = GPCRBank(
            window_size=getattr(cfg, 'gpcr_window', 16),
            ach_threshold=getattr(cfg, 'gpcr_ach_threshold', 0.55),
            ne_threshold=getattr(cfg, 'gpcr_ne_threshold', 0.55),
        )

        # NT projection connectome (hardcoded SOTA anatomy)
        self.projections = ProjectionGraph([
            Projection("VTA",   "NAcc",  "DA",   release_scale=1.0),
            Projection("VTA",   "PFC",   "DA",   release_scale=0.8),
            Projection("VTA",   "Hippo", "DA",   release_scale=0.5),
            Projection("SNc",   "BG",    "DA",   release_scale=1.0),
            Projection("LC",    "PFC",   "NE",   release_scale=0.7),
            Projection("LC",    "Thal",  "NE",   release_scale=0.6),
            Projection("LC",    "Hippo", "NE",   release_scale=0.4),
            Projection("Raphe", "PFC",   "5HT",  release_scale=0.5),
            Projection("Raphe", "DMN",   "5HT",  release_scale=0.6),
            Projection("Raphe", "Hippo", "5HT",  release_scale=0.4),
            Projection("NBM",   "PFC",   "ACh",  release_scale=0.6),
            Projection("NBM",   "Hippo", "ACh",  release_scale=0.7),
            Projection("NBM",   "Lang",  "ACh",  release_scale=0.4),
            Projection("NAcc",  "VTA",   "DA",   release_scale=0.3),  # feedback
            Projection("PFC",   "BG",    "Glu",  release_scale=0.6),
            Projection("BG",    "Thal",  "GABA", release_scale=0.7),
        ], {r: cfg.d_sem for r in [
            "VTA", "NAcc", "SNc", "LC", "Raphe", "NBM",
            "PFC", "BG", "Hippo", "DMN", "Thal", "Lang",
        ]})
        self.trophic = TrophicSystem(self.projections)

        # ---- Learning ----
        self.learning_layer = LearningLayer(n_inputs=8, hidden=32, init_scale=1.0)
        self.learned_opt    = LearnedBackprop(n_neuromods=cfg.n_neuromods, hidden=32)

        # ---- Virtual environment (sensory grounding) ----
        from .environments.virtual_world import environment_stream
        self._env_stream = environment_stream(seed=42, switch_every=50)
        # Continuous sensory world loop: encode 6-dim frame signal → d_sem grounding
        self.sensory_encoder = SensoryFrameEncoder(cfg.d_sem)

        # ---- Expert cortices ----
        # Legacy mode: vesicle-gated specialist modules at d_sem.
        # SRC-TEH mode: token-level transformer experts at d_hidden,
        # dispatched by ExpertChoiceRouter on the trunk's hidden state.
        # MathCortex: differential-attention over a learnable fact memory
        # ReasoningCortex: Modern Hopfield pattern-completion attractors
        # TopicClassifier: routes sem → expert gate probabilities
        _expert_d_hidden = cfg.d_hidden if _src_teh else None
        _expert_blocks   = int(getattr(cfg, 'expert_n_blocks', 0)) if _src_teh else 0
        _expert_heads    = int(getattr(cfg, 'expert_n_heads', cfg.lang_heads))
        self.math_cortex = MathCortex(
            d_sem=cfg.d_sem,
            n_heads=getattr(cfg, 'expert_heads', 4),
            memory_size=getattr(cfg, 'math_memory_size', 128),
            d_hidden=_expert_d_hidden,
            n_blocks=_expert_blocks,
            max_ctx=cfg.lang_ctx,
        )
        self.reasoning_cortex = ReasoningCortex(
            d_sem=cfg.d_sem,
            n_attractors=getattr(cfg, 'reasoning_attractors', 64),
            base_beta=getattr(cfg, 'reasoning_beta', 4.0),
            d_hidden=_expert_d_hidden,
            n_blocks=_expert_blocks,
            max_ctx=cfg.lang_ctx,
            expert_n_heads=_expert_heads,
            recursive_iters=(int(getattr(cfg, 'recursive_iters', 1))
                             if getattr(cfg, 'recursive_reasoning', False) else 1),
        )
        self.topic_classifier = TopicClassifier(cfg.d_sem)

        # ---- SRC-TEH: Expert-Choice router, Language expert, Latent Bus ----
        # Built unconditionally so checkpoints round-trip even if cfg
        # toggles change between runs.  Inert when enable_src_teh=False.
        from .modules.expert_router import (
            ExpertChoiceRouter, LanguageExpert,
            gather_tokens as _gather_tokens, scatter_add_tokens as _scatter_add_tokens,
        )
        from .intelligence.latent_program_bus import LatentProgramBus
        self._src_teh_enabled = _src_teh
        self._gather_tokens = staticmethod(_gather_tokens).__func__
        self._scatter_add_tokens = staticmethod(_scatter_add_tokens).__func__
        if _src_teh:
            self.expert_router = ExpertChoiceRouter(
                d_hidden=cfg.d_hidden,
                n_experts=int(getattr(cfg, 'n_token_experts', 3)),
                capacity_factor=float(getattr(cfg, 'expert_capacity_factor', 1.5)),
            )
            self.language_expert = LanguageExpert(
                d_hidden=cfg.d_hidden,
                n_blocks=_expert_blocks,
                n_heads=_expert_heads,
                max_ctx=cfg.lang_ctx,
            )
            self.latent_program_bus = LatentProgramBus(
                d_hidden=cfg.d_hidden,
                bus_dim=int(getattr(cfg, 'bus_dim', 16)),
                ema_alpha=float(getattr(cfg, 'bus_ema_alpha', 0.5)),
                d_sem=cfg.d_sem,
            ) if bool(getattr(cfg, 'enable_latent_bus', False)) else None
            # Memory injector: lift d_sem entries → d_hidden K/V rows.
            self.memory_kv_proj = nn.Linear(cfg.d_sem, cfg.d_hidden, bias=False)
            nn.init.normal_(self.memory_kv_proj.weight, std=0.02)
            # Lazy-bowtie EMA caches: slots (B,S,d_sem) and selected (B,d_sem).
            self.register_buffer('_bowtie_ema_slots',
                                 torch.zeros(cfg.gws_slots, cfg.d_sem))
            self.register_buffer('_bowtie_ema_selected',
                                 torch.zeros(cfg.d_sem))
            self.register_buffer('_bowtie_step', torch.zeros(1, dtype=torch.long))
        else:
            self.expert_router = None
            self.language_expert = None
            self.latent_program_bus = None
            self.memory_kv_proj = None

        # ---- State tracking ----
        self.last_nt:            dict | None = None
        self.last_routing:       torch.Tensor | None = None
        self.last_learning_gain: torch.Tensor | None = None
        self.last_action_idx: torch.Tensor | None = None
        self.last_threat: torch.Tensor | None = None
        self.last_survival: torch.Tensor | None = None

        # Register all modules with orchestrator (stage-based topology)
        _S = self._ORCH_STAGES
        _reg = self.orchestrator.register_module_brain
        _reg('sensory',            STAGE_SENSORY,       self.sensory)
        _reg('association',        STAGE_SENSORY,       self.association)
        _reg('thalamus',           STAGE_THALAMUS,      self.thalamus)
        _reg('world',              STAGE_STATE_MODELS,  self.world)
        _reg('self_m',             STAGE_STATE_MODELS,  self.self_m)
        if self.amygdala  is not None: _reg('amygdala',  STAGE_SUBCORTICAL,   self.amygdala)
        if self.insula    is not None: _reg('insula',    STAGE_SUBCORTICAL,   self.insula)
        _reg('qualia',             STAGE_QUALIA,        self.qualia)
        _reg('gws',                STAGE_GWS,           self.gws)
        if cfg.enable_neural_geometry: _reg('neural_geometry', STAGE_GWS, self.neural_geometry)
        _reg('hippo',              STAGE_MEMORY,        self.hippo)
        _reg('entorhinal',         STAGE_MEMORY,        self.entorhinal)
        if cfg.enable_cerebellum:  _reg('cerebellum',   STAGE_MEMORY,        self.cerebellum)
        _reg('pfc',                STAGE_COGNITIVE_CTL, self.pfc)
        if self.acc       is not None: _reg('acc',       STAGE_COGNITIVE_CTL, self.acc)
        _reg('bg',                 STAGE_EXECUTIVE,     self.bg)
        _reg('forward_m',          STAGE_EXECUTIVE,     self.forward_m)
        _reg('evaluator',          STAGE_EXECUTIVE,     self.evaluator)
        _reg('dmn',                STAGE_CONSCIOUSNESS, self.dmn)
        if cfg.enable_thought_transformer: _reg('thought_transformer', STAGE_CONSCIOUSNESS, self.thought_transformer)
        if cfg.enable_claustrum:   _reg('claustrum',    STAGE_CONSCIOUSNESS, self.claustrum)
        _reg('motor',              STAGE_MOTOR,         self.motor)

        # Apply initial enable/disable state from config
        self._sync_module_enables()

    # ------------------------------------------------------------------
    # Module enable / disable sync from config
    # ------------------------------------------------------------------
    def _sync_module_enables(self):
        """Apply cfg.enable_* flags to all BrainModule instances."""
        from .modules.brain_module import BrainModule
        flag_map = {
            "hippo":              "enable_hippocampus",
            "pfc":                "enable_pfc",
            "bg":                 "enable_basal_ganglia",
            "dmn":                "enable_dmn",
            "thalamus":           "enable_thalamus",
            "cerebellum":         "enable_cerebellum",
            "cortical_sheet":     "enable_cortical_sheet",
            "entorhinal":         "enable_entorhinal",
            "claustrum":          "enable_claustrum",
            "critic":             "enable_critic",
            "neural_geometry":    "enable_neural_geometry",
            "qualia":             "enable_qualia",
            "thought_transformer":"enable_thought_transformer",
        }
        for attr, flag in flag_map.items():
            mod = getattr(self, attr, None)
            if isinstance(mod, BrainModule):
                enabled = getattr(self.cfg, flag, True)
                mod.enabled = enabled

    def enable_module(self, name: str):
        """Enable a brain area by attribute name (e.g. 'hippo', 'pfc')."""
        from .modules.brain_module import BrainModule
        mod = getattr(self, name, None)
        if isinstance(mod, BrainModule):
            mod.enable()

    def disable_module(self, name: str):
        """Disable a brain area by attribute name."""
        from .modules.brain_module import BrainModule
        mod = getattr(self, name, None)
        if isinstance(mod, BrainModule):
            mod.disable()

    def module_status(self) -> dict[str, str]:
        """Return enabled/disabled status of all BrainModule instances."""
        from .modules.brain_module import BrainModule
        return {name: mod.status
                for name, mod in self.named_modules()
                if isinstance(mod, BrainModule) and name}

    # ------------------------------------------------------------------
    # NT helpers
    # ------------------------------------------------------------------
    def _nt_dict(self) -> dict[str, float]:
        """Return current NT levels as a plain dict."""
        return {n: float(self.transmitters.get(n).detach().mean())
                for n in NT_NAMES}

    def _release_via_nuclei(self, signals: dict[str, torch.Tensor]):
        nacc_drive, learning_gain = self.nacc(
            signals["novelty"], signals["reward"],
            signals["curiosity"], signals["ecb"])
        vta_in = torch.stack(
            [signals["rpe"], signals["salience"], nacc_drive, signals["valence"]], dim=-1)
        da_demand = self.vta.demand(vta_in)
        self.transmitters.release("DA", da_demand)

        lc_in = torch.stack(
            [signals["uncertainty"], signals["arousal"], signals["novelty"]], dim=-1)
        self.transmitters.release("NE", self.lc.demand(lc_in))

        raphe_in = torch.stack(
            [signals["avg_reward"], signals["time_since_reward"], signals["mood"]], dim=-1)
        self.transmitters.release("5HT", self.raphe.demand(raphe_in))

        nbm_in = torch.stack(
            [signals["attention_demand"], signals["novelty"], signals["surprise"]], dim=-1)
        self.transmitters.release("ACh", self.nbm.demand(nbm_in))

        return learning_gain, nacc_drive, da_demand

    def _release_via_projections(self, activities: dict[str, torch.Tensor]):
        for i, p in enumerate(self.projections.projections):
            if p.src not in activities:
                continue
            amt = self.projections.release_amount(i, activities[p.src])
            self.transmitters.release(p.nt, amt)

    @staticmethod
    def _act_scalar(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x.detach().abs().mean(dim=-1) - 1.0)

    @staticmethod
    def _chunked_ce(logits: torch.Tensor, targets: torch.Tensor,
                    chunk: int = 128, ignore_index: int = -100,
                    label_smoothing: float = 0.0) -> torch.Tensor:
        """Cross-entropy in T-dimension chunks to avoid a huge (B*T, V) allocation."""
        B, T, V = logits.shape
        acc = torch.zeros(B, T, dtype=torch.float32, device=logits.device)
        for t0 in range(0, T, chunk):
            t1 = min(t0 + chunk, T)
            lg = logits[:, t0:t1, :].reshape(-1, V)
            tg = targets[:, t0:t1].reshape(-1)
            losses = F.cross_entropy(lg, tg, ignore_index=ignore_index,
                                     reduction="none",
                                     label_smoothing=label_smoothing)
            acc[:, t0:t1] = losses.reshape(B, t1 - t0).float()
        return acc.mean(dim=1)

    def init_latents(self, batch_size: int, device, dtype=None):
        cfg = self.cfg
        if dtype is None:
            dtype = next(self.parameters()).dtype
        if (self.transmitters.level.size(0) != batch_size or
                self.transmitters.level.device != device):
            self.transmitters.reset(batch_size, device)
        return {
            "floating_thought": torch.zeros(batch_size, cfg.d_sem, device=device, dtype=dtype),
            "last_action":      torch.zeros(batch_size, cfg.bg_action_dim, device=device, dtype=dtype),
            "world_h":          self.world.init_state(batch_size, device),
            "self_h":           self.self_m.init_state(batch_size, device),
            "novelty":          torch.zeros(batch_size, device=device, dtype=dtype),
            "qualia":           torch.zeros(batch_size, cfg.d_sem, device=device, dtype=dtype),
            "prev_action_idx":  torch.full((batch_size,), -1, device=device, dtype=torch.long),
            "thought_valence":  torch.zeros(batch_size, device=device, dtype=dtype),
        }

    # ------------------------------------------------------------------
    # Maturity Index (MAT virtual protein)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_maturity(self, lm_loss: float, l_random: float | None = None) -> float:
        """Update the smoothed Maturity Index from the current LM loss.

        M_t = clamp(1.0 - L_lm / L_random, 0, 1) with L_random ≈ log(vocab).
        Stored as an EMA on `self.maturity`. Drives the continuous fade-in
        of expert cortices, MoD compute, ε-routing strength, and GABA
        tolerance.

        Side-effect: also flips `self._infancy` based on M < 0.3 so legacy
        code paths that still consult the flag continue to work.
        """
        from .neurochem.transmitters import compute_mat, L_RANDOM_DEFAULT
        l_random = l_random if l_random is not None else L_RANDOM_DEFAULT
        m_now = compute_mat(lm_loss, l_random)
        cur = float(self.maturity.item())

        # ── Asymmetric maturity EMA (fix B, revised) ─────────────────────
        # Rise quickly, fall slowly. A transient lm_loss spike barely dents
        # MAT (so the MAT-gated aux weight + pruning thresholds don't whipsaw
        # and amplify a single bad batch), yet a SUSTAINED regression still
        # lowers MAT — preserving the maturity-fall "recovery valve" that lets
        # the aux objectives disengage so the model can refocus on the LM.
        # (A hard monotonic ratchet removed that valve and made a collapse
        #  unrecoverable, so it is opt-in only.)
        alpha_up = float(getattr(self.cfg, 'maturity_ema_alpha',
                                 self._maturity_ema_alpha))
        alpha_dn = float(getattr(self.cfg, 'maturity_fall_alpha', alpha_up))
        alpha = alpha_up if m_now >= cur else alpha_dn
        m_ema = (1.0 - alpha) * cur + alpha * m_now

        if getattr(self.cfg, 'maturity_ratchet', False):
            hwm = getattr(self, '_maturity_hwm', 0.0)
            floor = float(getattr(self.cfg, 'maturity_awaken_floor', 0.3))
            if hwm >= floor:
                m_ema = max(m_ema, hwm)
            self._maturity_hwm = max(hwm, m_ema)
        else:
            # Keep the high-water mark current (used by the pruning freeze, C).
            self._maturity_hwm = max(getattr(self, '_maturity_hwm', 0.0), m_ema)

        self.maturity.fill_(m_ema)
        self._infancy = bool(m_ema < 0.3)
        return m_ema

    def maturity_scalar(self) -> float:
        """Plain-Python float of the current MAT level (avoids autograd)."""
        return float(self.maturity.detach().item())

    def topology_summary(self) -> str:
        """One-line-per-module breakdown of the active brain topology.

        Prints which top-level submodules exist (non-None), their parameter
        count, and the LM-trunk / bio-modules split so the user can verify
        a 'FULL' run isn't silently degenerating into a baseline-equivalent.
        Pure introspection — no side effects.
        """
        if getattr(self, '_baseline', False):
            return ("BASELINE: language cortex only "
                    f"({sum(p.numel() for p in self.parameters())/1e6:.2f}M params)")

        # Top-level nn.Modules that exist on this brain.
        rows = []
        bio_total = lm_total = other_total = 0
        for name, mod in self.named_children():
            if not hasattr(mod, 'parameters'):
                continue
            np = sum(p.numel() for p in mod.parameters())
            if np == 0:
                continue
            rows.append((name, np))
            if name == 'language':
                lm_total += np
            elif name in {'sensory', 'association', 'thalamus', 'world',
                          'self_m', 'gws', 'hippo', 'dmn', 'pfc', 'bg',
                          'forward_m', 'evaluator', 'motor', 'qualia',
                          'thought_transformer', 'cerebellum',
                          'entorhinal', 'claustrum', 'cortical_sheet',
                          'amygdala', 'acc', 'insula', 'lhb',
                          'math_cortex', 'reasoning_cortex',
                          'language_expert', 'expert_router'}:
                bio_total += np
            else:
                other_total += np

        rows.sort(key=lambda x: -x[1])
        lines = [
            "Brain topology:",
            f"  • LM trunk           : {lm_total/1e6:>6.2f}M",
            f"  • bio + bowtie       : {bio_total/1e6:>6.2f}M",
            f"  • neurochem + memory : {other_total/1e6:>6.2f}M",
            f"  • TOTAL              : {(lm_total+bio_total+other_total)/1e6:>6.2f}M",
            f"  • SRC-TEH            : {getattr(self, '_src_teh_enabled', False)}",
            f"  • neural_topology    : {self.cfg.neural_topology}",
            "  • top modules (>=0.5M params):",
        ]
        for name, np in rows:
            if np >= 500_000:
                lines.append(f"      - {name:24s} {np/1e6:>6.2f}M")
        return "\n".join(lines)

    @staticmethod
    def _phase_gate(mat: float, center: float, width: float = 0.10) -> float:
        """Smooth sigmoid 0→1 across a maturity window [center−width, center+width].

        Used to phase in each destabilising subsystem at its own MAT level so
        awakening is no longer a single switch.  Returns ≈0 below `center−width`,
        crosses 0.5 at `center`, and saturates at 1 above `center+width`.
        """
        x = (float(mat) - float(center)) / max(1e-6, float(width))
        # Approximate sigmoid via tanh (cheaper, no exp underflow risk).
        return 0.5 * (1.0 + math.tanh(x))

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------
    def forward_lm(self, ids: torch.Tensor,
                   targets: torch.Tensor | None = None):
        if getattr(self, '_baseline', False):
            logits, sem, h, _pc = self.language(ids)
            out = {"logits": logits}
            if targets is not None:
                B, T = ids.shape
                loss = self._chunked_ce(logits, targets).mean()
                out["loss"]    = loss
                out["lm_loss"] = loss.detach()
            return out

        cfg    = self.cfg
        B, T   = ids.shape
        device = ids.device
        dtype  = next(self.parameters()).dtype
        latents = self.init_latents(B, device, dtype=dtype)
        nt      = self.transmitters.vector().detach()
        nt_d    = {n: float(nt[0, i].item()) for i, n in enumerate(NT_NAMES)}

        # Infancy fast-path: when True, skip heavy decorative/no-grad
        # operations (orchestrator stage recording, phi_proxy, boundary
        # eigensystem, oscillation tracker, consciousness metrics, episodic
        # insight writes, vesicle pool). Their outputs go into auxiliary
        # losses already gated to ~0 by _aux_w_scale, or into log strings —
        # they contribute no useful signal during infancy and dominate
        # wall-clock for a 228M bf16 model.
        _in = bool(getattr(self, '_infancy', False))

        # Reset per-pass orchestrator state. Without this, stage-output buffers
        # accumulate across steps and Φ computations mix stale graph references.
        if hasattr(self, 'orchestrator') and not getattr(self.orchestrator, 'baseline', False):
            self.orchestrator.begin_pass()

        # GPCR observe on initial NT state (will be updated after threat critic too)
        with torch.no_grad():
            self.gpcr.observe(nt)
        _ne_arousal_lang = self.gpcr.ne_arousal()

        # NE arousal → raise CALM base_threshold so fewer tokens exit early under stress
        # (arousal forces full-depth processing — the model "pays attention")
        _calm_heads = []
        if _ne_arousal_lang > 0.5 and not self.training:
            _arousal_boost = 1.0 + _ne_arousal_lang  # 1.0–2.0×
            for blk in self.language.blocks:
                if hasattr(blk, 'calm_head'):
                    _calm_heads.append((blk.calm_head, blk.calm_head.base_threshold))
                    blk.calm_head.base_threshold = min(
                        0.99, blk.calm_head.base_threshold * _arousal_boost)

        # Propagate the current MAT level to every MoD block so its compute
        # gate (skip when M < 0.2) can take effect for this forward pass.
        # Plain Python float → no XLA cost, set once outside the traced graph.
        _mat_for_lang = self.maturity_scalar() if hasattr(self, "maturity_scalar") else 1.0
        for _blk in self.language.blocks:
            if hasattr(_blk, "_maturity_gate"):
                _blk._maturity_gate = _mat_for_lang

        # 1) Language cortex
        # Inject latent program bus as an extra prefix bias on the trunk
        # input (across-step reasoning state).  Zero-init read head so the
        # bias is zero at step 0 — gradient-equivalent to legacy on init.
        lang_thought = self.rcpt_lang.modulate(
            latents["floating_thought"].unsqueeze(1), nt).squeeze(1)
        if getattr(self, '_src_teh_enabled', False) and self.latent_program_bus is not None:
            # Read the across-step bus state → (B, d_sem) bias on the
            # `thought` channel.  Zero-init projection so this is identity
            # on step 0 and grows only as the read_head learns to be useful.
            bus_bias_s = self.latent_program_bus.read_sem(B, device, dtype)
            if bus_bias_s.size(-1) == cfg.d_sem:
                lang_thought = lang_thought + bus_bias_s
        # ReZero gate (λ_thought) on the from_sem thought-conditioning path.
        # `from_sem.weight` is already zero-init (so identity at t=0), but this
        # adds an explicit scalar that LM gradient can grow independently of
        # the projection weights — same self-discovering blend as the other
        # forward injections.
        if getattr(cfg, 'use_rezero_injection_gates', False):
            lang_thought = self.lambda_thought * lang_thought
        # Build memory_kv for RETRO-style injection: EMA of recent bowtie
        # output plus optional top-N consolidated entries (cheap path: just
        # use the EMA slots for now; full retrieval is a TODO with no
        # accuracy regression on bring-up).
        memory_kv = None
        # MAT-gate the memory injection.  At low MAT the bowtie EMA is mostly
        # random-init noise; attending to it pollutes the trunk's later layers
        # with no useful retrieval signal.  We gate the *strength* with a
        # phase factor centred at MAT 0.55 (same window as the token-level
        # expert residual) so the trunk's last 2 blocks ramp into memory
        # consumption alongside the experts learning to produce useful
        # bowtie slots.
        if (getattr(self, '_src_teh_enabled', False)
                and self.memory_kv_proj is not None
                and bool(self._bowtie_step.item() > 0)):
            _mat_now    = self.maturity_scalar() if hasattr(self, 'maturity_scalar') else 1.0
            # ReZero gate (λ_mem) supersedes the maturity-phase scheduler on
            # this forward injection when enabled; scalar starts at 0 and is
            # grown by LM gradient only as far as memory retrieval helps.
            if getattr(cfg, 'use_rezero_injection_gates', False):
                _mem_phase = self.lambda_mem
                _mem_active = True   # let the matmul run; λ controls magnitude
            else:
                _mem_phase = self._phase_gate(_mat_now, center=0.55, width=0.15)
                _mem_active = float(_mem_phase) > 1e-3
            if _mem_active:
                # Clone the buffer view so the EMA in-place update at the end
                # of this forward does not version-bump a tensor that the
                # backward graph still references through `memory_kv_proj`.
                mem = self._bowtie_ema_slots.detach().clone().to(
                    device=device, dtype=dtype)                           # (S, d_sem)
                memory_kv = _mem_phase * self.memory_kv_proj(mem)         # (S, d_hidden)
        _want_tap = bool(getattr(self, '_src_teh_enabled', False))
        if _want_tap:
            logits, sem, h_lang, pred_coding_loss, tap_sem = self.language(
                ids, thought=lang_thought, nt=nt,
                memory_kv=memory_kv, return_tap=True)
        else:
            tap_sem = None
            logits, sem, h_lang, pred_coding_loss = self.language(
                ids, thought=lang_thought, nt=nt)

        # Restore CALM thresholds after language forward
        for _ch, _orig_thresh in _calm_heads:
            _ch.base_threshold = _orig_thresh

        # ── SRC-TEH: Token-level Expert-Choice routing ────────────────────
        # Each expert (Lang / Math / Reason) pulls top-C tokens by affinity
        # and emits a residual enrichment scattered back into h_lang. Then
        # we re-norm + re-emit logits so the LM head sees the enriched
        # hidden state instead of the raw trunk output.
        # Residual strength is MAT-phase-gated (0 below MAT 0.4, full above
        # MAT 0.7) so un-trained experts can't pump noise into the trunk
        # immediately after awakening at MAT 0.3.
        expert_aux_loss = torch.tensor(0.0, device=device)
        if (getattr(self, '_src_teh_enabled', False)
                and self.expert_router is not None):
            _mat = self.maturity_scalar() if hasattr(self, "maturity_scalar") else 1.0
            _expert_phase = self._phase_gate(_mat, center=0.55, width=0.15)
            # Internal expert MAT — pass to inner residual so token-level
            # output is _expert_phase * expert(x) - x; legacy d_sem path
            # still calls forward(..., maturity=_mat) directly.
            _mat_inner = _expert_phase
            route = self.expert_router(h_lang, maturity=_mat)
            assignments = route["assignments"]                  # 3 × (B, C)
            weights     = route["weights"]                      # 3 × (B, C)
            expert_aux_loss = route["aux_loss"]
            # Experts in canonical order: 0=Language, 1=Math, 2=Reasoning.
            experts_fwd = [
                self.language_expert.forward_tokens,
                self.math_cortex.forward_tokens,
                self.reasoning_cortex.forward_tokens,
            ]
            h_enriched = h_lang
            for e, fn in enumerate(experts_fwd):
                if e >= len(assignments):
                    break
                idx = assignments[e]                            # (B, C)
                gate = weights[e].unsqueeze(-1)                 # (B, C, 1)
                tokens = self._gather_tokens(h_lang, idx)       # (B, C, d_hidden)
                enriched = fn(tokens, maturity=_mat_inner)      # (B, C, d_hidden)
                contrib = gate * (enriched - tokens)            # residual
                h_enriched = self._scatter_add_tokens(h_enriched, idx, contrib)
            # Re-norm and re-emit logits using the post-expert hidden state.
            h_post = self.language.norm_f(h_enriched)
            logits = self.language.lm_head(h_post)
            # Refresh `sem` and `h_lang` to track the enriched stream so
            # downstream modules (motor cond. head, bowtie) see it.
            h_lang = h_enriched
            if self.language.use_attention_pool:
                sem = self.language.sem_pool(h_post)
            else:
                sem = self.language.to_sem(h_post.mean(dim=1))
            # Write the new trunk pool into the latent program bus for
            # next-step readback.
            if self.latent_program_bus is not None:
                _summary = h_post.mean(dim=1)                  # (B, d_hidden)
                self.latent_program_bus.write(_summary.detach())

        # ── Trunk gradient isolation (architectural stabilization) ──────────
        # Hand the bio/cognitive pipeline a STOP-GRADIENT copy of the trunk's
        # semantic output. From here on, every module (phase_attn, htm,
        # sensory, association, GWS, world, self, pfc, motor, Φ, cpc, …) and
        # its auxiliary loss reads a FIXED representation: gradients train the
        # modules but cannot flow back into the language trunk. The trunk is
        # left to converge under the LM loss alone (logits come from `h_post`,
        # not `sem`), exactly like the stable infancy phase — eliminating the
        # backward path by which random-init aux heads corrupt the trunk and
        # drive the post-awakening divergence. Forward couplings are intact.
        if getattr(cfg, 'detach_trunk_from_aux', False):
            sem = sem.detach()

        # Novel: phase-coded attention + HTM temporal structure on language output
        novel_aux_loss = torch.tensor(0.0, device=device)
        novel_aux_loss = novel_aux_loss + getattr(cfg, 'w_expert_aux', 0.01) * expert_aux_loss
        if self.phase_attn is not None:
            nt4 = nt[:, :4] if nt.shape[1] >= 4 else None
            sem, _ = self.phase_attn(sem, step_offset=self._global_step, nt_levels=nt4)
        if self.htm is not None:
            sem, _htm_h, htm_seq_loss = self.htm(sem, h_prev=None)
            novel_aux_loss = novel_aux_loss + htm_seq_loss

        # Baseline topology: return here (language only)
        if cfg.neural_topology == "baseline":
            out = {"logits": logits}
            if targets is not None:
                loss = self._chunked_ce(logits, targets).mean()
                out.update({"loss": loss, "lm_loss": loss.detach()})
            return out

        # ── Lazy Bowtie decision ─────────────────────────────────────────
        # When SRC-TEH is on AND the cache has been primed at least once,
        # only run the heavy bowtie modules every `bowtie_period` steps;
        # on off-steps we substitute the cached EMA of slots+selected.
        _bowtie_period = int(getattr(cfg, 'bowtie_period', 1))
        _run_bowtie = True
        if (getattr(self, '_src_teh_enabled', False)
                and _bowtie_period > 1
                and bool(self._bowtie_step.item() > 0)):
            _run_bowtie = (int(self._bowtie_step.item()) % _bowtie_period == 0)

        # 2) Sensory + association
        sens, salience = self.sensory(sem)
        assoc          = self.association([sens])

        # 2b) Topic classification → expert gate probabilities
        # (zero-init so all gates start near 0.25 = uniform; grows with training)
        with torch.no_grad() if not self.training else torch.enable_grad():
            _topic_probs = self.topic_classifier(sem.detach()
                                                 if not self.training else sem)
        # Scalar gates per expert (batch-mean probability for routing)
        _math_gate    = float(_topic_probs[:, TOPIC_MATH].mean().item())
        _reason_gate  = float(_topic_probs[:, TOPIC_REASONING].mean().item())

        # Record sensory output for Φ proxy (skipped during infancy)
        if not _in:
            self.orchestrator.record_stage_output(sem)

        # 3) Thalamic router + re-entry bias (bowtie top-down loop)
        # Re-entry injects the previous step's PFC+GWS representation as a
        # top-down prior — the key mechanism for bidirectional causal closure
        # required by IIT (Dehaene 2011: thalamo-cortical re-entry).
        _reentry = self.orchestrator.get_reentry_bias(B, device)
        _reentry = _reentry.to(dtype=assoc.dtype)
        # Re-entry feeds BOTH the routing decision (top-down bias on the
        # thalamic gate) and the post-routing residual (so the bias survives
        # the softmax gate even when the router prefers a different stream).
        routed, routing_probs = self.thalamus(
            assoc + _reentry, nt, return_routing=True,
            maturity=self.maturity_scalar() if hasattr(self, "maturity_scalar") else None)
        routed = routed + _reentry
        routed = self.rcpt_thal.modulate(routed.unsqueeze(1), nt).squeeze(1)
        self.last_routing = routing_probs.detach()

        # 4) World + self models
        z_world, _wh, world_pred = self.world(routed, latents["world_h"])
        z_self,  _sh             = self.self_m(
            latents["last_action"], nt[:, :cfg.n_neuromods],
            latents["floating_thought"], latents["self_h"])

        # Continuous sensory world loop: pull one frame, encode, ground z_world
        try:
            _frame = next(self._env_stream)
            _frame_vec = _frame.to_vec()
            _frame_emb = self.sensory_encoder.encode_frame(
                _frame_vec, device=device, dtype=z_world.dtype)  # (1, d_sem)
            z_world = z_world + _frame_emb.expand(B, -1)
        except StopIteration:
            pass  # stream exhausted (shouldn't happen — generator is infinite)

        # Novel: fast-weight associative memory — enrich sem with world context
        if self.fast_weight is not None:
            sem, _ = self.fast_weight(sem, context=z_world)

        # 4b) Threat critic
        threat, survival = self.critic.forward_safe(z_world, z_self)
        # Use torch.where instead of Python bool-on-tensor (avoids XLA graph break)
        ne_release = torch.where(survival, torch.full_like(threat, 0.9), torch.zeros_like(threat))
        self.transmitters.release("NE", ne_release)
        nt   = self.transmitters.vector().detach()
        nt_d = {n: float(nt[0, i].item()) for i, n in enumerate(NT_NAMES)}

        # 4b-ext) Amygdala + LHb in training forward pass
        if self.amygdala is not None:
            amyg_tr = self.amygdala(
                z_world, threat=self.critic.forward_safe(z_world, z_self)[0],
                reward=torch.zeros(B, device=device, dtype=dtype))
            for i, nt_name in enumerate(NT_NAMES):
                if i < amyg_tr["nt_demand"].size(-1):
                    self.transmitters.release(nt_name, amyg_tr["nt_demand"][:, i] * 0.2)
            latents["floating_thought"] = (latents["floating_thought"]
                                           + 0.2 * amyg_tr["thought_tint"])
            nt   = self.transmitters.vector().detach()
            nt_d = {n: float(nt[0, i].item()) for i, n in enumerate(NT_NAMES)}

        # 4c) Qualia — emotion modulates floating thought before GWS
        if cfg.enable_qualia:
            q_out = self.qualia(latents["floating_thought"], nt, threat, z_self)
            modulated_thought_lm = q_out["modulated_thought"]
            latents["thought_valence"] = q_out["thought_valence"]
            for i, nt_name in enumerate(NT_NAMES):
                if i < q_out["thought_nt_demand"].size(-1):
                    self.transmitters.release(
                        nt_name, q_out["thought_nt_demand"][:, i] * 0.2)
            nt   = self.transmitters.vector().detach()
            nt_d = {n: float(nt[0, i].item()) for i, n in enumerate(NT_NAMES)}
        else:
            modulated_thought_lm = latents["floating_thought"]

        # 5) GWS — receives qualia-tinted thought
        candidates = torch.stack(
            [routed, z_world, z_self, modulated_thought_lm], dim=1)
        slots = self.gws(candidates, ne_temp=nt[:, NT_NAMES.index("NE")])

        # Novel: relational attention + causal reasoning over GWS slots
        if self.relational_attn is not None:
            slots, _ = self.relational_attn(slots)
        if self.causal_module is not None:
            ci_out = self.causal_module(slots)
            slots = ci_out["out"]
            novel_aux_loss = novel_aux_loss + ci_out["dag_loss"]

        # Record GWS output for Φ proxy (central bottleneck of bowtie)
        # — skipped during infancy. set_gws_broadcast is kept because it
        # feeds within-pass re-entry into expert projections.
        if not _in:
            self.orchestrator.record_stage_output(slots.mean(1))
        self.orchestrator.set_gws_broadcast(slots.mean(1))

        # 5b) Vesicle-typed topic routing → Expert Cortices
        # Synthesise a typed vesicle from the GWS broadcast state.
        # Skipped during infancy — the topic classifier is at random init,
        # so vesicle synthesis is uniform noise; expert gates derived from
        # vesicle concentrations would just toggle the heavy math/reasoning
        # cortices for no useful signal.
        if not _in and self.vesicle_pool is not None:
            _dominant_topic = int(_topic_probs.argmax(-1).float().mean().round().item())
            self.vesicle_pool.synthesize_typed(
                content=slots.detach().mean(0).mean(0),  # (d_sem,) mean over B,S
                type_idx=_dominant_topic,
                source_module=0,
            )
            self.vesicle_pool.migrate()
            self.vesicle_pool.degrade()
            # Update expert gates from vesicle concentrations
            _math_gate   = max(_math_gate,   self.vesicle_pool.expert_gate(TOPIC_MATH))
            _reason_gate = max(_reason_gate, self.vesicle_pool.expert_gate(TOPIC_REASONING))

        # Expert cortex enrichment (additive, parallel, gated by topic probs)
        # Legacy d_sem path. Skipped when SRC-TEH is on — the token-level
        # experts already enriched the trunk's hidden state at d_hidden, so
        # running the same cortices again at d_sem would be double-counting.
        _slots_mean = slots.mean(1)  # (B, d_sem) — batch aggregate for experts
        _mat = self.maturity_scalar() if hasattr(self, "maturity_scalar") else 1.0
        if not getattr(self, '_src_teh_enabled', False):
            _gws_ctx = _slots_mean.detach()   # broadcast as plasticity context
            if _math_gate > 0.02:
                _slots_mean = self.math_cortex(_slots_mean,
                                               vesicle_gate=_math_gate,
                                               gws_context=_gws_ctx,
                                               maturity=_mat)
            if _reason_gate > 0.02:
                _slots_mean = self.reasoning_cortex(_slots_mean,
                                                    vesicle_gate=_reason_gate,
                                                    gws_context=_gws_ctx,
                                                    maturity=_mat)
            # Inject expert enrichment back into slot representations
            if _math_gate > 0.02 or _reason_gate > 0.02:
                slots = slots + 0.2 * _slots_mean.unsqueeze(1)

        # Bowtie EMA cache update — feeds RETRO-style memory_kv on the
        # NEXT forward pass (zero on the first step, then warms up).
        if getattr(self, '_src_teh_enabled', False):
            with torch.no_grad():
                slots_mean_batch = slots.detach().mean(0)        # (S, d_sem)
                alpha = float(getattr(cfg, 'bowtie_ema_alpha', 0.4))
                slots_mean_batch = slots_mean_batch.to(
                    dtype=self._bowtie_ema_slots.dtype,
                    device=self._bowtie_ema_slots.device)
                self._bowtie_ema_slots.mul_(1.0 - alpha).add_(slots_mean_batch, alpha=alpha)
                self._bowtie_step += 1

        # 6) DMN query + hippocampal enrichment
        dmn_query, _stop = self.dmn.forward_safe(slots, latents["floating_thought"], nt_d)
        dmn_query_mod    = self.rcpt_dmn.modulate(dmn_query.unsqueeze(1), nt).squeeze(1)

        # GPCR metabotropic modulation — observe sustained NT levels
        with torch.no_grad():
            self.gpcr.observe(nt)
        _ach_gate = self.gpcr.ach_gate()    # 0–1: high ACh → broader DG encoding
        _ne_arousal = self.gpcr.ne_arousal()  # 0–1: high NE → block CALM early-exit

        # ACh gate: when sustained ACh is high, widen DG winners (more pattern encoding)
        _orig_sparse_k = self.hippo.sparse_k
        self.hippo.sparse_k = max(
            _orig_sparse_k,
            int(_orig_sparse_k * (1.0 + _ach_gate)))

        # Hippocampal enrichment: multi-dimensional recall → enriched GWS
        val_t = latents["thought_valence"]
        slots_enriched, novelty, all_recalls = self.hippo.forward_safe(
            slots, dmn_query_mod, nt_d, valence=val_t)
        recalls = self.rcpt_hippo.modulate(all_recalls, nt)

        # Restore sparse_k after hippocampal call
        self.hippo.sparse_k = _orig_sparse_k

        # Novel: differentiable external memory read/write
        if self.diff_memory is not None:
            slots_enriched, _ = self.diff_memory(
                slots_enriched, pred_error=novelty)

        # 7) PFC — NT-driven selection
        slots_mod = self.rcpt_pfc.modulate(slots_enriched, nt)
        selected, _replace = self.pfc.forward_safe(
            slots_mod, recalls, latents["floating_thought"], nt_d)

        # Record PFC output for Φ proxy + store re-entry signal (bowtie loop)
        # — stage recording skipped during infancy; update_reentry kept so the
        # bowtie loop still seeds thalamic re-entry for the next step.
        if not _in:
            self.orchestrator.record_stage_output(selected)
        self.orchestrator.update_reentry(selected)

        # Novel: active dendrite, neurogenesis, dynamic MoE on selected representation
        if self.active_dendrite is not None:
            selected = self.active_dendrite(
                selected.unsqueeze(1), context=z_world).squeeze(1)
        if self.neurogenesis is not None:
            sel_seq = selected.unsqueeze(1)
            sel_seq, _ = self.neurogenesis(sel_seq, novelty=novelty)
            selected = sel_seq.squeeze(1)
        if self.dynamic_routing_moe is not None:
            moe_out, moe_aux = self.dynamic_routing_moe(selected.unsqueeze(1))
            selected = selected + moe_out.squeeze(1)
            novel_aux_loss = novel_aux_loss + moe_aux

        # 8) BG + forward simulation + safety gate
        selected_bg = self.rcpt_bg.modulate(selected.unsqueeze(1), nt).squeeze(1)
        action, _conf, _probs, commit_ok = self.bg.forward_safe(
            selected_bg, nt_d,
            world_model=self.forward_m,
            z_world=z_world, z_self=z_self,
            critic=self.critic, evaluator=self.evaluator)

        wp, sp = self.forward_m(z_world, z_self, action)
        value  = self.evaluator(wp, sp, nt)

        # 8b) Motor cortex (only commit if BG says safe)
        _mt, motor_lang_bias, action_idx, action_logits, action_probs = \
            self.motor(action, survival=survival)

        # Motor-conditioned logits (one extra matmul, not a full second pass).
        # CRITICAL: the bias is broadcast across ALL token positions via
        # `unsqueeze(1)` — so every token's logits get perturbed, not just
        # the last (sampling) position.  During infancy the motor cortex is
        # at random init, so an unconditional broadcast pumps pure noise
        # into the LM loss — observed cost: lm_loss stuck at the random
        # ceiling (10.7) while the same-preset baseline reached 7.4 at
        # step 210. We now MAT-phase-gate the bias (centre 0.45) so the
        # motor cortex starts modulating the LM only after world/motor
        # losses also engage.  Pre-awakening the model trains as a clean
        # LM trunk, matching baseline throughput.
        _mat_motor   = self.maturity_scalar() if hasattr(self, 'maturity_scalar') else 1.0
        # ReZero gate (λ_motor) on the motor → LM forward injection. Starts at
        # zero (no contribution → no awakening discontinuity); LM gradient
        # grows it only as far as the motor bias actually helps prediction.
        if getattr(cfg, 'use_rezero_injection_gates', False):
            _motor_phase = self.lambda_motor
            _motor_active = True
        else:
            _motor_phase = self._phase_gate(_mat_motor, center=0.45, width=0.10)
            _motor_active = float(_motor_phase) > 1e-3
        if _motor_active:
            h_biased     = h_lang + _motor_phase * motor_lang_bias.unsqueeze(1)
            logits_motor = self.language.lm_head(h_biased)
            del logits
            logits = logits_motor
        else:
            # Phase ≈ 0: skip the second matmul entirely.  The LM trains
            # against the trunk's clean output (`logits` from
            # `self.language(...)` further above).
            logits_motor = logits

        # 9) NT release
        with torch.no_grad():
            zero = torch.zeros(B, device=device, dtype=dtype)
            if targets is not None:
                p = F.softmax(logits_motor.detach(), dim=-1)
                tgt = targets.clamp_min(0)
                correct_p    = p.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                reward_proxy = correct_p.mean(dim=1)
            else:
                reward_proxy = zero
            curiosity    = (world_pred.detach() - z_world.detach()).pow(2).mean(-1).sqrt()
            curiosity    = (curiosity / (curiosity.mean() + 1e-6)).clamp(0, 1)
            arousal      = (salience.detach() + curiosity) * 0.5
            mood         = self.transmitters.get("5HT").detach()
            rpe          = (reward_proxy - mood).clamp(-1, 1).abs()

        signals = dict(
            novelty=novelty.detach(), reward=reward_proxy, curiosity=curiosity,
            ecb=self.transmitters.get("eCB").detach(), rpe=rpe,
            salience=salience.detach(), valence=reward_proxy,
            uncertainty=curiosity, arousal=arousal,
            avg_reward=reward_proxy,
            time_since_reward=(1 - reward_proxy).clamp(0, 1),
            mood=mood, attention_demand=(1 - reward_proxy + curiosity).clamp(0, 1),
            surprise=novelty.detach(),
        )
        learning_gain, nacc_drive, da_release = self._release_via_nuclei(signals)
        activities = {
            "VTA":   da_release.detach(), "NAcc": nacc_drive.detach(),
            "LC":    self._act_scalar(z_self),
            "Raphe": self._act_scalar(slots.mean(1)),
            "NBM":   signals["attention_demand"],
            "PFC":   self._act_scalar(selected),
            "Hippo": novelty.detach(), "BG": self._act_scalar(action),
        }
        self._release_via_projections(activities)
        self.transmitters.step()

        # Hippocampal store + oscillation tracking — pure observability,
        # both skipped during infancy. Hippocampal recalls during infancy
        # would be random-init noise anyway.
        if not _in:
            with torch.no_grad():
                _val_f = signals["valence"].mean().item()
                _sal_f = salience.detach().mean().item()
                self.hippo.store(dmn_query.detach(), selected.detach(),
                                 nt_state=nt.detach(),
                                 valence=_val_f,
                                 salience=_sal_f)

            if cfg.enable_oscillations and hasattr(self, 'oscillation_tracker'):
                with torch.no_grad():
                    _om = {'language': 0, 'pfc': 1, 'hippocampus': 2,
                           'thalamus': 3, 'basal_ganglia': 4, 'dmn': 5,
                           'gws': 6, 'motor': 7}
                    self.oscillation_tracker.record(_om['language'],  sem.detach().mean(1))
                    self.oscillation_tracker.record(_om['pfc'],       selected.detach())
                    self.oscillation_tracker.record(_om['hippocampus'], dmn_query.detach())
                    self.oscillation_tracker.record(_om['thalamus'],  routed.detach())
                    self.oscillation_tracker.record(_om['basal_ganglia'], action.detach())
                    self.oscillation_tracker.record(_om['dmn'],       dmn_query_mod.detach())
                    self.oscillation_tracker.record(_om['gws'],       slots.detach().mean(1))
                    self.oscillation_tracker.record(_om['motor'],     motor_lang_bias.detach())
                    self.oscillation_tracker.tick()

        with torch.no_grad():
            _bdnf_val = reward_proxy.mean().item()

            if _in:
                # Skip Φ proxy + Laplacian eigensystem during infancy — both
                # are expensive (Gaussian-MI over bipartitions; symmetric
                # eigendecomp) and feed only trophic/logging which are also
                # gated off. Use neutral placeholders.
                _phi_val = 0.0
                _fiedler_val = 1.0
            else:
                # Φ: IIT-style MIP estimate (Gaussian-MI lower bound across the
                # bowtie module bipartition).
                _phi_val = self.orchestrator.compute_phi_proxy()

                # Spectral gap via normalised-Laplacian eigensystem.
                _bd_obs = {n: o for n, o in zip(self.orchestrator.module_names,
                                                self.orchestrator._last_stage_outputs)
                           if o is not None}
                if len(_bd_obs) >= 2:
                    _fiedler_val, _fiedler_vec = self.boundary_detector.observe(_bd_obs)
                else:
                    _fiedler_val = 1.0

            self._last_phi     = _phi_val
            self._last_fiedler = _fiedler_val
            # Φ-coupled + Fiedler-gated BDNF: locks high-integration pathways
            # and homeostically rewires near-disconnected module graph edges.
            # Suppressed during infancy: trophic signals are meaningless when
            # phi/fiedler/novelty all come from a random-init network.
            if not getattr(self, '_infancy', False):
                # ── Freeze structural pruning post-maturation (fix C) ──────
                # Pruning fires when MAT ≥ trophic_prune_mat, i.e. exactly when
                # the model has matured and is doing its best work — removing
                # projection capacity there destabilizes it. Once the maturity
                # high-water mark reaches `prune_freeze_mat`, latch pruning off
                # for the rest of the run by passing an unreachable threshold
                # (BDNF growth below is unaffected).
                _prune_mat = float(getattr(cfg, 'trophic_prune_mat', 0.3))
                if getattr(cfg, 'freeze_pruning_after_maturation', True):
                    _hwm = getattr(self, '_maturity_hwm', self.maturity_scalar())
                    if _hwm >= float(getattr(cfg, 'prune_freeze_mat', 0.6)):
                        self._pruning_frozen = True
                    if getattr(self, '_pruning_frozen', False):
                        _prune_mat = 2.0   # MAT can never exceed → no pruning
                self.trophic.update(activities,
                                    bdnf=_bdnf_val,
                                    ngf=novelty.detach().mean().item(),
                                    phi=_phi_val,
                                    fiedler=_fiedler_val,
                                    maturity=self.maturity_scalar(),
                                    prune_mat_threshold=_prune_mat)

                # BDNF structural growth: Φ-triggered rank increase in NeuralGeometryAdapters
                if hasattr(self, 'language') and hasattr(self.language, 'bdnf_grow_all'):
                    self.language.bdnf_grow_all(_bdnf_val, _phi_val)

        out = {
            "logits":        logits,
            "world_pred":    world_pred,
            "value":         value,
            "novelty":       novelty,
            "selected":      selected,
            "learning_gain": learning_gain.detach(),
            "routing":       routing_probs.detach(),
            "action_idx":    action_idx.detach(),
            "action_probs":  action_probs.detach(),
            "threat":        threat.detach(),
            "survival":      survival.detach(),
            "commit_ok":     commit_ok.detach(),
        }

        if targets is not None:
            lm_loss_per = self._chunked_ce(
                logits_motor, targets,
                label_smoothing=float(getattr(cfg, 'label_smoothing', 0.0)))
            # Mesolimbic CE gain: clamped OFF (=1.0) until MAT > 0.55 because
            # `learning_gain` comes from a randomly-initialised LearningLayer
            # whose output is uncorrelated with reward early on — gating it
            # on prevents pre-trained LM gradient from being modulated by
            # uninformed DA noise.  See diagnosis in docs/RFC.md §1.
            _mat_now    = self.maturity_scalar() if hasattr(self, 'maturity_scalar') else 1.0
            _meso_phase = self._phase_gate(_mat_now, center=0.55, width=0.10)
            meso_gain   = (1.0 + 0.5 * _meso_phase * learning_gain.detach() *
                           self.transmitters.get("DA").detach()).clamp(min=1.0)
            lm_loss     = (lm_loss_per * meso_gain).mean()
            world_loss = F.mse_loss(world_pred, z_world.detach())
            fwd_reg    = (wp.pow(2).mean() + sp.pow(2).mean()) * 0.5

            with torch.no_grad():
                p_motor   = F.softmax(logits_motor.detach(), dim=-1)
                del logits_motor  # free (B,T,vocab) before further allocs
                tgt       = targets.clamp_min(0)
                tgt_p     = p_motor.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                mean_conf = tgt_p.mean(dim=1)
                speak_target = (mean_conf > cfg.speak_conf_threshold).long()
                action_target = torch.where(
                    speak_target.bool(),
                    torch.full_like(speak_target, ACTION_INDEX["SPEAK"]),
                    torch.full_like(speak_target, ACTION_INDEX["REMAIN_SILENT"]),
                )
            motor_loss = F.cross_entropy(action_logits, action_target)

            # RSSM KL divergence (zero if plain WorldModel)
            rssm_kl = torch.tensor(0.0, device=device)
            if self._use_rssm and isinstance(latents.get("world_h"), dict):
                from .modules.world_model import RecurrentStateSpaceModel
                rssm_kl = RecurrentStateSpaceModel.kl_loss(latents["world_h"])

            # Novel: multi-scale predictive coding auxiliary loss on language rep
            if self.pred_coding is not None:
                pc_novel, _ = self.pred_coding(sem)
                novel_aux_loss = novel_aux_loss + pc_novel

            # CPC: contrastive predictive coding on GWS slot sequence
            cpc_loss = torch.tensor(0.0, device=device)
            if self.cpc is not None and slots.shape[1] > 1:
                cpc_loss, _ = self.cpc(slots)

            # ── IIT Φ objective (differentiable) ──
            # The orchestrator has been recording (still-live) stage outputs
            # since the start of this pass. phi_tensor() computes the
            # Gaussian-MI lower bound across the MIP bisection — a real
            # integrated-information signal, not a correlation heuristic.
            # We *minimise* `-phi`, so backprop pushes module outputs toward
            # configurations where every bipartition has high MI.
            phi_t = None
            # Skip differentiable Φ during infancy: its loss contribution is
            # aux_w * w_phi ≈ 1e-5 anyway, but the Gaussian-MI bipartition
            # computation is one of the heaviest ops in forward_lm.
            if (not _in
                    and getattr(cfg, 'enable_phi_objective', True)
                    and hasattr(self, 'orchestrator')
                    and not self.orchestrator.baseline):
                try:
                    phi_t = self.orchestrator.phi_tensor(module_outputs={
                        "language": sem,
                        "thalamus": routed,
                        "world":    z_world,
                        "self":     z_self,
                        "gws":      slots.mean(1) if slots.dim() == 3 else slots,
                        "pfc":      selected,
                    })
                except Exception:
                    phi_t = None
            if phi_t is None:
                phi_loss_term = torch.tensor(0.0, device=device, dtype=dtype)
                phi_value = torch.tensor(0.0, device=device, dtype=dtype)
            else:
                # Bound the contribution AND keep useful gradient at high Φ.
                # Old: -tanh(Φ/3)·3 saturates at Φ≈6 (gradient ≈1%); the Φ
                # objective went silent in observed runs where Φ pinned at 7.
                # New: -tanh(Φ/8)·8 keeps the linear regime out to Φ≈16
                # while still bounded — same protection from runaway
                # integration, restored gradient signal.
                phi_value = phi_t.to(dtype=dtype)
                phi_loss_term = -torch.tanh(phi_value / 8.0) * 8.0

            def _safe(t):
                if isinstance(t, torch.Tensor):
                    return t.nan_to_num(0.0, posinf=0.0, neginf=0.0)
                return torch.tensor(float(t) if not (t != t) else 0.0, device=device)

            # ── Phased Maturation Engine ─────────────────────────────────
            # Each auxiliary loss has its own MAT phase window so the
            # awakening event no longer simultaneously activates 8 noisy
            # objectives.  Centres chosen to preserve a clean "language
            # first → world/motor next → bowtie integration → causal Φ"
            # learning curriculum.  `_aux_w_scale` from train.py still
            # multiplies on top (master scale).
            _mat      = self.maturity_scalar() if hasattr(self, 'maturity_scalar') else 1.0
            aux_w     = float(getattr(self, '_aux_w_scale', 1.0))
            ph_pred   = self._phase_gate(_mat, center=0.35, width=0.08)
            ph_world  = self._phase_gate(_mat, center=0.45, width=0.08)
            ph_motor  = self._phase_gate(_mat, center=0.50, width=0.08)
            ph_fwd    = self._phase_gate(_mat, center=0.50, width=0.08)
            ph_novel  = self._phase_gate(_mat, center=0.55, width=0.08)
            ph_cpc    = self._phase_gate(_mat, center=0.55, width=0.08)
            ph_kl     = self._phase_gate(_mat, center=0.60, width=0.08)
            ph_phi    = self._phase_gate(_mat, center=0.60, width=0.08)
            total = (cfg.w_lm * lm_loss
                     + aux_w * ph_world * cfg.w_world       * _safe(world_loss)
                     + aux_w * ph_fwd   * cfg.w_forward     * _safe(fwd_reg) * 0.01
                     + aux_w * ph_motor * cfg.w_motor       * _safe(motor_loss)
                     + aux_w * ph_pred  * cfg.w_pred_coding * _safe(pred_coding_loss)
                     + aux_w * ph_kl    * getattr(cfg, 'w_kl_world', 0.1) * _safe(rssm_kl)
                     + aux_w * ph_novel * 0.05              * _safe(novel_aux_loss)
                     + aux_w * ph_cpc   * getattr(cfg, 'w_cpc', 0.05) * _safe(cpc_loss)
                     + aux_w * ph_phi   * getattr(cfg, 'w_phi', 0.02) * _safe(phi_loss_term))

            # Orchestrator.route runs the cerebellum/entorhinal/claustrum
            # subnetworks for id_drift / neural_calm metrics. Heavy and only
            # contributes scaled aux signal — skip during infancy.
            if (not _in
                    and hasattr(self, 'orchestrator')
                    and not self.orchestrator.baseline):
                orch_out, orch_metrics = self.orchestrator.route(
                    routed, {'world': self.world, 'cerebellum': self.cerebellum,
                             'entorhinal': self.entorhinal, 'claustrum': self.claustrum})
                id_drift = orch_metrics.get('identity_drift', 0.0)
                calm     = orch_metrics.get('neural_calm', 1.0)
                total    = total + aux_w * (0.01 * _safe(id_drift)
                                            + 0.01 * (1.0 - _safe(calm)))

            out.update({
                "loss":                   total,
                "lm_loss":                lm_loss_per.mean().detach(),
                "world_loss":             world_loss.detach(),
                "motor_loss":             motor_loss.detach(),
                "pred_coding_loss":       pred_coding_loss.detach(),
                "motor_speak_target_rate": speak_target.float().mean().detach(),
                "phi":                    phi_value.detach(),
                "phi_loss":               phi_loss_term.detach(),
            })

            # ConsciousnessMetrics — pure observability (phi/gamma/coherence
            # history). Skipped during infancy.
            if not _in:
                try:
                    with torch.no_grad():
                        c_metrics = self.consciousness.update(
                            module_outputs={
                                "pfc":      selected.detach(),
                                "world":    z_world.detach(),
                                "self":     z_self.detach(),
                                "language": sem.detach().mean(1),
                            },
                            gws_slots=slots.detach(),
                            floating_thought=latents["floating_thought"].detach(),
                            novelty=novelty.detach(),
                            routing=routing_probs.detach(),
                        )
                    self._last_phi = float(c_metrics.get("phi", float(phi_value.detach().item())))
                    out["consciousness"] = c_metrics
                except Exception:
                    self._last_phi = float(phi_value.detach().item())

            # ---- Training insights → RelationalMemoryGraph ----
            # Skipped during infancy: surprise/comprehension/valence are all
            # random-init noise; nothing meaningful to store.
            if not _in:
                self._maybe_store_insight(ids, sem, nt, lm_loss_per, targets, device)

            # ---- BRIAN: post-awakening narrative / causal stack ─────────
            # All gated on `not _in` so the heavy bookkeeping is inert
            # during infancy. ActualCausationHead runs in no-grad here;
            # its aux_loss is added to total only if requested by train.
            if not _in:
                try:
                    with torch.no_grad():
                        # Stack the 8 canonical module outputs (mean over
                        # B,S) into a (n_modules, d_sem) tensor for the
                        # actual-causation head.
                        _module_stack = torch.stack([
                            sem.detach().mean(dim=(0, 1)),
                            selected.detach().mean(0),
                            (dmn_query.detach().mean(0)
                                if dmn_query.dim() >= 1 else dmn_query.detach()),
                            routed.detach().mean(0),
                            action.detach().mean(0),
                            (dmn_query_mod.detach().mean(0)
                                if dmn_query_mod.dim() >= 1 else dmn_query_mod.detach()),
                            slots.detach().mean(dim=(0, 1)),
                            motor_lang_bias.detach().mean(0),
                        ], dim=0).to(dtype=torch.float32)

                        if self._prev_module_outputs is not None:
                            try:
                                alpha = self.actual_causation(
                                    self._prev_module_outputs, _module_stack)
                                self._last_causal_edges = \
                                    self.actual_causation.strong_edges(alpha)

                                # κ_cause vesicle emission for high-α edges
                                if (self.vesicle_pool is not None
                                        and self._last_causal_edges):
                                    from .neurochem.vesicles import TOPIC_CAUSE
                                    # Synthesise one vesicle from the strongest
                                    # edge's destination module embedding
                                    _, _j, _ = self._last_causal_edges[0]
                                    _content = _module_stack[_j].detach()
                                    self.vesicle_pool.synthesize_typed(
                                        content=_content.to(
                                            dtype=next(self.vesicle_pool.parameters()).dtype
                                            if any(True for _ in self.vesicle_pool.parameters())
                                            else _content.dtype),
                                        type_idx=TOPIC_CAUSE,
                                        source_module=0,
                                    )
                            except Exception:
                                pass

                        self._prev_module_outputs = _module_stack

                        # Personality: apply trust-driven NT bias if ToM is
                        # tracking active entities right now.
                        active_ents = {}
                        if self.tom is not None and hasattr(self.tom,
                                                            "active_entities"):
                            active_ents = self.tom.active_entities()
                        if active_ents:
                            self.personality.apply_bias(
                                self.transmitters, active_ents)

                        # Sleep-cycle scheduler — fires every N awake steps.
                        if self.sleep_cycle.maybe_sleep(self._global_step):
                            self._run_sleep_cycle()
                except Exception:
                    # BRIAN stack is best-effort; never crash the training
                    # loop because of a memory-system side-effect.
                    pass

            # ---- Periodic consolidation ----
            self._global_step += 1
            if (self._global_step % cfg.consolidate_every == 0):
                with torch.no_grad():
                    self._run_consolidation(float(da_release.mean()))

        self.last_nt            = nt_d
        self.last_learning_gain = learning_gain.detach()
        self.last_action_idx    = action_idx.detach()
        self.last_threat        = threat.detach()
        self.last_survival      = survival.detach()
        self.transmitters.detach_()
        return out

    # ------------------------------------------------------------------
    # Cognitive Closure: embodied grid-frame ingestion
    # ------------------------------------------------------------------
    @torch.no_grad()
    def set_survival(self, energy: float = None,
                      hydration: float = None,
                      integrity: float = None) -> None:
        """Direct mutation of the persistent survival-state vector. Used
        by tests + by Brain.run_continuous when an env frame arrives."""
        if energy is not None:
            self.survival_state[0] = float(energy)
        if hydration is not None:
            self.survival_state[1] = float(hydration)
        if integrity is not None:
            self.survival_state[2] = float(integrity)

    @torch.no_grad()
    def ingest_grid_frame(self, frame) -> torch.Tensor:
        """Encode a GridFrame via SensoryVAE → d_sem latent + update the
        persistent survival_state from the frame's homeostatic vector.

        Returns the (1, d_sem) latent z to be added as a sensory residual.
        Triggers κ_neg vesicle emission if QualiaState reports high aversive
        pressure on the new homeostat values. Used by Brain.run_continuous
        (the embodied loop) — NOT in forward_lm for text-only training.
        """
        # Update survival_state from the frame
        s = torch.as_tensor(frame.homeostatic,
                              dtype=self.survival_state.dtype,
                              device=self.survival_state.device)
        self.survival_state.copy_(s)

        # Encode the frame
        device = self.survival_state.device
        dtype = next(self.sensory_vae.parameters()).dtype
        try:
            z = self.sensory_vae.encode_frame(frame, device=device, dtype=dtype)
        except Exception:
            z = torch.zeros(1, self.cfg.d_sem, device=device, dtype=dtype)

        # Homeostatic warp: pull the QualiaState's broadcast warp toward
        # the new survival state. We expose the aversive pressure scalar
        # for κ_neg emission.
        if hasattr(self, 'qualia'):
            try:
                z_warped = self.qualia.warp_broadcast(z, s.unsqueeze(0))
                z = z_warped
            except Exception:
                pass

        # κ_neg emission gated on aversive pressure threshold
        if hasattr(self, 'qualia') and self.vesicle_pool is not None:
            try:
                pressure = float(self.qualia.aversive_pressure())
                if pressure > 0.10:
                    from .neurochem.vesicles import TOPIC_NEG
                    # Aversive vesicles synthesised at the reasoning cortex
                    # module index — they migrate from there. Module index
                    # 1 = PFC / executive in the canonical 8-module stack,
                    # but for vesicle source-module we use 0 (GWS) so the
                    # vesicle's diffusion T-matrix routes it to ReasoningCortex.
                    self.vesicle_pool.synthesize_typed(
                        content=z.squeeze(0).to(dtype=dtype),
                        type_idx=TOPIC_NEG,
                        source_module=0,
                    )
            except Exception:
                pass

        return z

    @torch.no_grad()
    def tick_homeostasis(self,
                           reward_action: bool = False,
                           reward_value: float = 0.0,
                           decay: float = 0.01) -> None:
        """Apply one tick of survival-variable decay. Called by
        Brain.run_continuous after each env step."""
        if hasattr(self, 'homeostasis'):
            try:
                self.homeostasis.step(
                    self.survival_state,
                    decay=decay,
                    reward_action=reward_action,
                    reward_value=reward_value)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # BRIAN: sleep cycle (predictive-coding distillation + trophic renorm)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _run_sleep_cycle(self) -> None:
        """Run one sleep phase. Called by forward_lm when
        `sleep_cycle.maybe_sleep(step)` returns True. Always best-effort:
        catches and logs exceptions instead of breaking the training loop.
        """
        try:
            from .modules.hippocampus import Hippocampus  # for typing only
            episodes = self.episodic.recent(256) if hasattr(self, "episodic") else []
            # Episode → embedding callable: use the cached vector if any,
            # else zeros.
            d = self.cfg.d_sem

            def _to_embed(ep) -> torch.Tensor:
                vec = ep.get("content_vec") if isinstance(ep, dict) \
                      else getattr(ep, "content_vec", None)
                if vec is None:
                    return torch.zeros(d)
                import numpy as _np
                if isinstance(vec, _np.ndarray):
                    t = torch.from_numpy(vec).float().flatten()
                else:
                    t = torch.as_tensor(vec).float().flatten()
                if t.numel() >= d:
                    return t[:d]
                pad = torch.zeros(d)
                pad[:t.numel()] = t
                return pad

            def _to_known_nll(ep) -> float:
                # If the episode carries a stored surprise, use it; else 0.
                if isinstance(ep, dict):
                    return float(ep.get("surprise", 0.0))
                return float(getattr(ep, "surprise", 0.0))

            report = self.sleep_cycle.sleep(
                step=self._global_step,
                episodes=[ep if isinstance(ep, dict) else getattr(ep, "__dict__", {})
                           for ep in episodes],
                episode_to_embed=_to_embed,
                episode_to_known_nll=_to_known_nll,
                actual_causation=self.actual_causation,
                trophic_system=getattr(self, "trophic", None),
            )
            print(f"[brain] 🛌 sleep cycle @step {report.step}: "
                  f"replays={report.n_replays}  "
                  f"distill={report.distillation_loss:.4f}  "
                  f"ΔMI={report.mi_reduction:+.3f}  "
                  f"pruned={report.pruned_edges}/{report.pruned_edges + report.strengthened_edges}  "
                  f"{report.duration_s:.2f}s", flush=True)
        except Exception as e:
            print(f"[brain] sleep cycle skipped ({e})", flush=True)

    # ------------------------------------------------------------------
    # Awakening propagation — called by train.py once at the maturation
    # transition. Flips internal flags on the BRIAN subcomponents that
    # need to know about it.
    # ------------------------------------------------------------------
    def set_awakened(self, awakened: bool = True) -> None:
        try:
            self.personality.set_awakened(awakened)
        except Exception:
            pass
        try:
            self.sleep_cycle.set_awakened(awakened)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Insight storage: training loss → memory graph
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _maybe_store_insight(self, ids, sem, nt, lm_loss_per,
                              targets, device):
        """Write high-surprise training observations to the memory graph."""
        try:
            # surprise = NLL of first sample (per-token)
            surprise = float(lm_loss_per[0].item())
            if surprise < 0.1:
                return          # not surprising enough

            from .tokenizer import Tokenizer
            tok     = Tokenizer()
            content = tok.decode(ids[0].tolist())[:512]
            vec     = sem[0].mean(0).cpu().numpy()
            nt_snap = nt[0].cpu().numpy()

            # Comprehension: cosine of consecutive positions
            if sem.size(1) >= 2:
                h1 = F.normalize(sem[0, :-1], dim=-1)
                h2 = F.normalize(sem[0, 1:], dim=-1)
                comprehension = float((h1 * h2).sum(-1).mean().item()) * 0.5 + 0.5
            else:
                comprehension = 0.5

            da_mean = float(self.transmitters.get("DA").detach().mean())

            nid = self.relational_memory.store_insight(
                content       = content,
                content_vec   = vec,
                nt_state      = nt_snap,
                surprise      = surprise,
                comprehension = comprehension,
                valence       = da_mean - 0.5,
                da_level      = da_mean,
                causal_parent = self._last_memory_id,
            )
            if nid is not None:
                self._last_memory_id = nid
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------
    def _run_consolidation(self, da_level: float = 0.5):
        try:
            episodes = self.episodic.recent(256)
            stats    = self.consolidator.consolidate(episodes, da_level=da_level)
            self.causal.prune(max_rules=2048)
            return stats
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Continuous sensory world loop (closed-loop active inference)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_continuous(self, n_ticks: int = 1000,
                       seed_ids: torch.Tensor | None = None,
                       on_tick=None) -> dict:
        """Run the brain as a closed-loop perception/action agent.

        Each tick:
          1. Pull a sensory frame from the virtual environment stream
          2. Encode it via the VAE-style sensory front-end
          3. Add the encoding as a prior on floating_thought (active inference)
          4. Run cognitive_step → logits, state, info
          5. Sample next token (greedy) and append to context
          6. Record per-tick Φ / Fiedler / action / salience

        Pass `on_tick(metrics, state)` for online consumption.
        """
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype
        if seed_ids is None:
            seed_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
        state = self.init_latents(seed_ids.size(0), device, dtype=dtype)

        history = {"phi": [], "fiedler": [], "actions": [], "salience": []}
        for t in range(n_ticks):
            try:
                _frame     = next(self._env_stream)
                _frame_vec = _frame.to_vec()
                _frame_emb = self.sensory_encoder.encode_frame(
                    _frame_vec, device=device, dtype=dtype)   # (1, d_sem)
            except StopIteration:
                break

            # Active-inference prior: ground floating_thought in the new sense
            state["floating_thought"] = (
                state["floating_thought"]
                + 0.3 * _frame_emb.expand_as(state["floating_thought"]))

            logits, state, info = self.cognitive_step(seed_ids, state)

            next_tok = int(logits[:, -1, :].argmax(-1).item())
            seed_ids = torch.cat([
                seed_ids,
                torch.full((seed_ids.size(0), 1), next_tok,
                           dtype=torch.long, device=device)
            ], dim=1)
            if seed_ids.size(1) > self.cfg.lang_ctx:
                seed_ids = seed_ids[:, -self.cfg.lang_ctx:]

            phi = float(getattr(self, "_last_phi", 0.0))
            fid = float(getattr(self, "_last_fiedler", 1.0))
            act = int(self.last_action_idx[0].item()) if self.last_action_idx is not None else 0
            sal = float(info.get("salience", 0.0)) if isinstance(info, dict) else 0.0
            history["phi"].append(phi)
            history["fiedler"].append(fid)
            history["actions"].append(act)
            history["salience"].append(sal)

            if on_tick is not None:
                on_tick({"phi": phi, "fiedler": fid, "action": act,
                         "tick": t, "next_tok": next_tok}, state)

        return history

    # ------------------------------------------------------------------
    # Inference cognitive loop
    # ------------------------------------------------------------------
    @torch.no_grad()
    def cognitive_step(self, ids: torch.Tensor, state: dict,
                       allow_emit: bool = True):
        cfg    = self.cfg
        B      = ids.size(0)
        nt     = self.transmitters.vector()
        nt_d   = {n: float(nt[0, i].item()) for i, n in enumerate(NT_NAMES)}

        # 1) Language
        lang_thought = self.rcpt_lang.modulate(
            state["floating_thought"].unsqueeze(1), nt).squeeze(1)
        logits, sem, h_lang, _ = self.language(ids, thought=lang_thought)

        if cfg.neural_topology == "baseline":
            return logits, state, {"nt": nt_d}

        # 2) Sensory + association
        sens, salience = self.sensory(sem)
        assoc          = self.association([sens])

        # 2b) Cortical sheet
        if cfg.enable_cortical_sheet:
            cortical_out = self.cortical_sheet(assoc, state["floating_thought"])
            salience = salience + 0.3 * cortical_out["burst"]
            assoc    = assoc    + 0.5 * cortical_out["output"]
        else:
            cortical_out = {"burst": torch.zeros_like(salience), "output": assoc}

        # 2c) Entorhinal
        if cfg.enable_entorhinal:
            entorh       = self.entorhinal(state["floating_thought"])
            grid_context = entorh["grid_code"]
        else:
            grid_context = torch.zeros_like(state["floating_thought"])
            entorh       = {"velocity": grid_context}

        # 3) Thalamus
        routed, routing = self.thalamus(assoc, nt, return_routing=True)
        routed = self.rcpt_thal.modulate(routed.unsqueeze(1), nt).squeeze(1)

        # 4) World + self
        z_world, state["world_h"], _wp = self.world(routed, state["world_h"])
        z_self,  state["self_h"]       = self.self_m(
            state["last_action"], nt[:, :cfg.n_neuromods],
            state["floating_thought"], state["self_h"])

        # 5) Critic
        threat, survival = self.critic.forward_safe(z_world, z_self)
        if survival.any():
            self.transmitters.release("NE", torch.where(
                survival, torch.full_like(threat, 0.9), torch.zeros_like(threat)))
            nt   = self.transmitters.vector()
            nt_d = {n: float(nt[0, i].item()) for i, n in enumerate(NT_NAMES)}

        # 5b) Amygdala — emotional tagging before qualia
        reward_proxy_for_amyg = torch.zeros(B, device=device, dtype=dtype)
        if self.amygdala is not None:
            amyg_out = self.amygdala(
                z_world, threat=threat, reward=reward_proxy_for_amyg,
                pfc_input=state.get("floating_thought"))
            state["emotional_valence"] = amyg_out["valence"]
            state["arousal"]           = amyg_out["arousal"]
            # CeA NT release
            for i, nt_name in enumerate(NT_NAMES):
                if i < amyg_out["nt_demand"].size(-1):
                    self.transmitters.release(nt_name, amyg_out["nt_demand"][:, i] * 0.25)
            nt   = self.transmitters.vector()
            nt_d = {n: float(nt[0, i].item()) for i, n in enumerate(NT_NAMES)}
            # Amygdala tints the floating thought before qualia
            amyg_tint = amyg_out["thought_tint"]
        else:
            amyg_tint = torch.zeros_like(state["floating_thought"])
            state["emotional_valence"] = torch.zeros(B, device=device, dtype=dtype)
            state["arousal"]           = torch.zeros(B, device=device, dtype=dtype)

        # 5c) LHb — anti-reward: suppress DA when expected reward missed
        if self.lhb is not None:
            lhb_out = self.lhb.update(
                reward_proxy_for_amyg,
                da_level=self.transmitters.get("DA"))
            if lhb_out["lhb_firing"].item() > 0.05:
                for i, nt_name in enumerate(NT_NAMES):
                    delta = lhb_out["nt_delta"][i]
                    if delta < 0:
                        self.transmitters.release(nt_name, torch.abs(delta) * (-1))
                nt   = self.transmitters.vector()
                nt_d = {n: float(nt[0, i].item()) for i, n in enumerate(NT_NAMES)}

        # 6) Qualia — receives amygdala-tinted thought
        thought_for_qualia = state["floating_thought"] + 0.3 * amyg_tint
        if cfg.enable_qualia:
            q_out = self.qualia(thought_for_qualia, nt, threat, z_self)
            state["qualia"]         = q_out["qualia"]
            state["thought_valence"] = q_out["thought_valence"]
            for i, nt_name in enumerate(NT_NAMES):
                if i < q_out["thought_nt_demand"].size(-1):
                    self.transmitters.release(
                        nt_name, q_out["thought_nt_demand"][:, i] * 0.3)
            nt   = self.transmitters.vector()
            nt_d = {n: float(nt[0, i].item()) for i, n in enumerate(NT_NAMES)}
            modulated_thought = q_out["modulated_thought"]
        else:
            modulated_thought = thought_for_qualia
            state["thought_valence"] = state["emotional_valence"]

        # 7) GWS
        candidates = torch.stack(
            [routed, z_world, z_self, modulated_thought, grid_context], dim=1)
        slots = self.gws(candidates, ne_temp=nt[:, NT_NAMES.index("NE")])

        # 7b) Claustrum
        if cfg.enable_claustrum:
            claustrum_out = self.claustrum([
                sens, routed, z_world, z_self,
                modulated_thought, grid_context,
                state["qualia"], cortical_out["output"],
            ])
        else:
            claustrum_out = {"salience": torch.zeros(B, device=device, dtype=dtype),
                             "gestalt":  torch.zeros_like(modulated_thought)}

        # 7c) Neural geometry
        if cfg.enable_neural_geometry:
            geo_out = self.neural_geometry(slots, state["floating_thought"])
            slots   = slots + 0.3 * geo_out["output"].unsqueeze(1).expand_as(slots)
        else:
            geo_out = {"curvature": torch.zeros(B, device=device, dtype=dtype),
                       "stream_gates": torch.zeros(1, device=device, dtype=dtype)}

        # 8) DMN query + hippocampal enrichment
        dmn_query, stop_logit = self.dmn.forward_safe(
            slots, modulated_thought, nt_d)
        dmn_query_mod = self.rcpt_dmn.modulate(dmn_query.unsqueeze(1), nt).squeeze(1)

        slots_enriched, novelty, all_recalls = self.hippo.forward_safe(
            slots, dmn_query_mod, nt_d, valence=state["thought_valence"])
        recalls = self.rcpt_hippo.modulate(all_recalls, nt)

        # 9) Thought transformer — receives qualia-modulated thought, not raw
        if cfg.enable_thought_transformer:
            tt_out = self.thought_transformer(modulated_thought, slots_enriched)
            enhanced_thought = tt_out["transformed_thought"]
        else:
            enhanced_thought = modulated_thought
            tt_out = {"consistency": torch.zeros(1), "reasoning_depth": torch.zeros(1)}

        # 9b) Active inference — precision-weighted predictive coding over GWS
        if self.active_inference is not None:
            nt4 = nt[:, :4] if nt.size(-1) >= 4 else nt
            ai_out = self.active_inference(
                slots_enriched.mean(1), action_probs=None, nt_levels=nt4)
            slots_enriched = slots_enriched + 0.2 * ai_out["posterior"].unsqueeze(1)
            state["epistemic_value"] = ai_out["epistemic_value"]

        # 10) PFC
        slots_mod = self.rcpt_pfc.modulate(slots_enriched, nt)
        selected, replace_gate = self.pfc.forward_safe(
            slots_mod, recalls, enhanced_thought, nt_d)

        # 10b) Theory of Mind — model active entity's mental state
        tom_out = {}
        if self.tom is not None and self._active_entity_id is not None and \
                self.entity_store is not None:
            eid = self._active_entity_id
            style_np = self.entity_store.entity_embedding(eid)
            belief_np = self.entity_store.belief_vector(eid)
            if style_np is not None:
                d_style = self.tom.d_style
                style_t  = torch.tensor(style_np[:d_style],
                                        device=ids.device).unsqueeze(0).expand(B, -1)
                belief_t = torch.tensor(belief_np,
                                        device=ids.device).unsqueeze(0).expand(B, -1)
                n_inter  = float(self.entity_store.get_profile(eid).interaction_count)
                log_n    = torch.full((B, 1), math.log1p(n_inter), device=ids.device)
                tom_out  = self.tom(style_t, belief_t, log_n, selected,
                                    entity_id=eid)
                # Emotional contagion: bleed entity's predicted affect into NTs
                affect = tom_out["affect_bleed"]   # (B, 4): DA, NE, 5HT, ACh
                for i, nt_name in enumerate(["DA", "NE", "5HT", "ACh"]):
                    self.transmitters.release(nt_name, affect[:, i] * 0.15)
                nt   = self.transmitters.vector()
                nt_d = {n: float(nt[0, i].item()) for i, n in enumerate(NT_NAMES)}

        # 10c) Insula — interoception + gut feelings
        if self.insula is not None:
            insula_out = self.insula(
                selected, nt_vec=nt,
                floating_thought=enhanced_thought,
                effort_proxy=novelty)
            # Insula's somatic go/nogo colors the floating thought slightly
            go_sign = insula_out["go_nogo"].unsqueeze(-1)   # (B, 1)
            enhanced_thought = enhanced_thought + 0.1 * go_sign * insula_out["interoceptive"]
            enhanced_thought = insula_out["empathy_state"]
            # Insula salience: can override DMN/replace gate
            insula_salience = insula_out["salience"]
            # NT demand from insula
            for i, nt_name in enumerate(NT_NAMES):
                if i < insula_out["nt_demand"].size(-1):
                    self.transmitters.release(nt_name, insula_out["nt_demand"][:, i] * 0.15)
            nt   = self.transmitters.vector()
            nt_d = {n: float(nt[0, i].item()) for i, n in enumerate(NT_NAMES)}
        else:
            insula_salience = torch.zeros(B, device=device, dtype=dtype)

        # 10d) ACC — conflict monitoring, effort gating, ACh demand
        if self.acc is not None:
            candidates_for_acc = torch.stack([routed, z_world, z_self, selected], dim=1)
            acc_out = self.acc(
                candidates_for_acc,
                prediction_error=novelty,
                pfc_rep=selected)
            # ACh release from conflict
            self.transmitters.release("ACh", acc_out["ach_demand"] * 0.4)
            nt   = self.transmitters.vector()
            nt_d = {n: float(nt[0, i].item()) for i, n in enumerate(NT_NAMES)}
            state["conflict"]       = acc_out["conflict"]
            state["effort_steps"]   = acc_out["effort_steps"]
            # Affect regulation: ACC suppresses amygdala emotional reactivity
            if "arousal" in state:
                acc_reg = acc_out["affect_reg"]
                state["arousal"] = state["arousal"] * (1.0 - 0.3 * acc_reg)

        # Update floating thought
        thought_alpha = cfg.thought_alpha
        ach   = nt[:, NT_NAMES.index("ACh")].unsqueeze(-1)
        # Insula salience can force replace (gut feeling says this is important)
        replace_mask = (replace_gate > 0.5) | (novelty > cfg.novelty_threshold) | (insula_salience > 0.7)
        # Blend qualia-tinted thought into the base so emotion taints persist across ticks
        qualia_weight = 0.25
        emotional_base = (1 - qualia_weight) * enhanced_thought + qualia_weight * modulated_thought
        smooth = ((1 - thought_alpha * ach) * emotional_base
                  + thought_alpha * ach * selected)
        state["floating_thought"] = torch.where(
            replace_mask.unsqueeze(-1), selected, smooth)

        # 11) BG + forward simulation
        selected_bg = self.rcpt_bg.modulate(selected.unsqueeze(1), nt).squeeze(1)
        action, conf, _probs, commit_ok = self.bg.forward_safe(
            selected_bg, nt_d,
            world_model=self.forward_m,
            z_world=z_world, z_self=z_self,
            critic=self.critic, evaluator=self.evaluator)

        wp, sp = self.forward_m(z_world, z_self, action)
        value  = self.evaluator(wp, sp, nt)

        # 11b) Cerebellum
        if cfg.enable_cerebellum:
            cereb_out  = self.cerebellum(state["floating_thought"], selected_bg, actual_next=wp)
            cereb_error = cereb_out["error"]
        else:
            cereb_error = torch.zeros(B, device=device, dtype=dtype)

        # 12) Motor
        _mt, motor_lang_bias, action_idx, action_logits, action_probs = \
            self.motor(action, survival=survival)
        h_biased = h_lang + motor_lang_bias.unsqueeze(1)
        logits2  = self.language.lm_head(h_biased)

        # 13) Mesolimbic reward circuit
        zero      = torch.zeros(B, device=device, dtype=dtype)
        state_sum = slots_enriched.mean(1)
        meso_out  = self.mesolimbic(
            state_vec=state_sum, reward=zero,
            da_level=self.transmitters.get("DA"),
            ecb_level=self.transmitters.get("eCB"),
            gaba_level=self.transmitters.get("GABA"),
            novelty=novelty, salience=salience,
            valence=state["thought_valence"], uncertainty=novelty)
        rpe = meso_out["rpe"]

        # DA reward-tag relational memory
        if self._last_memory_id is not None:
            try:
                self.relational_memory.tag_reward(
                    self._last_memory_id,
                    da_level=float(self.transmitters.get("DA").mean()),
                    reward_signal=float(rpe.mean()))
            except Exception:
                pass

        # 14) NT release
        signals = dict(
            novelty=novelty, reward=zero, curiosity=zero,
            ecb=self.transmitters.get("eCB"), rpe=rpe.detach(),
            salience=salience, valence=state["thought_valence"],
            uncertainty=novelty, arousal=salience, avg_reward=zero,
            time_since_reward=zero, mood=self.transmitters.get("5HT"),
            attention_demand=salience, surprise=novelty,
        )
        self._release_via_nuclei(signals)

        # SNr / nigrostriatal DA
        motor_intent = self._act_scalar(action)
        bg_act       = self._act_scalar(selected_bg)
        d2_fb        = self.mesolimbic.d2_feedback(
            self.transmitters.get("DA"), motor_intent).detach()
        sn_da, sn_gaba = self.substantia_nigra(
            motor_intent, bg_act, d2_fb, zero, zero,
            self.transmitters.get("GABA"))
        self.transmitters.release("DA",   sn_da   * 0.5)
        self.transmitters.release("GABA", sn_gaba * 0.3)
        self.transmitters.release("DA",   meso_out["da_release_demand"])

        # Gated projection releases
        activities = {
            "PFC": self._act_scalar(selected), "BG": self._act_scalar(action),
            "Hippo": novelty, "VTA": self.transmitters.get("DA"),
            "NAcc": meso_out["wanting"].detach(),
            "LC":   self._act_scalar(z_self),
            "Raphe": self._act_scalar(state_sum), "NBM": salience,
            "SNr":   sn_gaba.detach(), "Thalamus": self._act_scalar(routed),
        }
        nt_ref = self.transmitters.vector()
        for nt_name, amt in self.gated_projections.gated_release(nt_ref, activities).items():
            self.transmitters.release(nt_name, amt.clamp(0, 0.5))
        self._release_via_projections(activities)
        self.trophic.update(activities, bdnf=0.0, ngf=float(novelty.mean()),
                            fiedler=float(getattr(self, '_last_fiedler', 1.0)),
                            maturity=self.maturity_scalar() if hasattr(self, 'maturity_scalar') else None,
                            prune_mat_threshold=float(getattr(
                                self.cfg, 'trophic_prune_mat', 0.3)))

        self.reuptake.clear(self.transmitters)
        self.reuptake.adapt_density(self.transmitters)
        self.receptor_adaptation.update(self.transmitters)
        self.transmitters.step()

        self.hippo.store(dmn_query, selected, nt_state=nt,
                         valence=float(state["thought_valence"].mean()),
                         salience=float(salience.mean()))

        # Encode to relational memory
        consol = float(meso_out["consolidation"].mean())
        sal_v  = float(salience.mean())
        if consol > 0.3 or sal_v > 0.3 or float(novelty.mean()) > 0.4:
            try:
                from .tokenizer import Tokenizer
                tok     = Tokenizer()
                content = tok.decode(ids[0].tolist())
                vec     = sem[0].mean(0).cpu().numpy()
                nt_snap = nt[0].cpu().numpy()
                mid = self.relational_memory.encode(
                    content     = content,
                    content_vec = vec,
                    nt_state    = nt_snap,
                    valence     = float(state["thought_valence"][0]),
                    arousal     = float(nt[0, NT_NAMES.index("NE")].item()),
                    salience    = max(sal_v, consol),
                    reward      = float(meso_out["wanting"].mean()),
                    causal_parent = self._last_memory_id,
                )
                self._last_memory_id = mid
            except Exception:
                pass

        state["last_action"]    = action
        state["prev_action_idx"] = action_idx

        # --- Entity tracking (update entities from association cortex if available) ---
        if hasattr(self.association, 'extract_entities'):
            detected_entities = self.association.extract_entities([sens])
            for ent in detected_entities:
                eid = ent.get('id', None)
                if eid is not None:
                    self.entities[eid] = ent
        state['entities'] = self.entities

        # ToM is already handled above (lines ~10c) when _active_entity_id is set

        # --- RSSM world state ---
        if self._use_rssm:
            rssm_out, rssm_state = self.world(z_world, state.get('rssm_state', None))
            state['rssm_state'] = rssm_state
            state['rssm_out'] = rssm_out

        # --- Active inference (Free Energy Principle) ---
        if cfg.enable_active_inference and self.active_inference is not None:
            fe_out = self.active_inference(
                gws_state=slots_enriched.mean(1),
                action_probs=action_probs,
                nt_levels=nt
            )
            state['free_energy'] = fe_out['free_energy']
            state['epistemic_value'] = fe_out['epistemic_value']
            state['pragmatic_value'] = fe_out['pragmatic_value']
            state['uncertainty'] = fe_out['uncertainty']

        # --- Vesicle neuromodulation (slow long-range content packets) ---
        if self.vesicle_pool is not None:
            # Build (B, n_modules, d_sem) module activation tensor for docking
            _mod_acts = torch.stack([
                slots_enriched.mean(1),   # 0: GWS summary
                z_world,                  # 1: world model
                z_self,                   # 2: self state
                dmn_query,                # 3: DMN
                selected,                 # 4: PFC selection
                sens,                     # 5: sensory
                sem.mean(1),              # 6: language
                state["floating_thought"],# 7: floating thought
            ], dim=1)  # (B, 8, d_sem)
            # Compute global activation variance for homeostatic GABA regulation
            # Flattened across batch and modules; σ²_global used to control GABA decay
            global_variance = _mod_acts.reshape(B, -1).var().item()
            surprise_signal = novelty.unsqueeze(-1) * state["floating_thought"]
            _mat_vp = self.maturity_scalar() if hasattr(self, "maturity_scalar") else 0.0
            vesicle_mod = self.vesicle_pool.tick(
                _mod_acts, surprise=surprise_signal,
                global_variance=global_variance, maturity=_mat_vp)
            # Apply GWS slot modulation (module 0 = GWS)
            slots_enriched = slots_enriched + vesicle_mod[:, 0:1, :].expand_as(slots_enriched)

        # Consciousness metrics
        c_metrics = self.consciousness.update(
            module_outputs={
                "pfc": selected, "dmn": dmn_query,
                "world": z_world, "self": z_self,
                "sensory": sens, "language": sem.mean(1),
            },
            gws_slots=slots_enriched,
            floating_thought=state["floating_thought"],
            novelty=novelty, routing=routing,
        )
        # Cache Φ for BDNF structural growth (used in forward_lm between steps)
        self._last_phi = float(c_metrics.get("phi", 0.0))

        # Narrative
        if cfg.enable_narrative:
            self.narrative_system.record_autobiographical(
                state["floating_thought"][0].detach(),
                valence=float(state["thought_valence"][0]),
                salience=sal_v)
            self.narrative_system.record_world(
                z_world[0].detach(), valence=float(rpe.mean()))

        info = {
            "value":       value, "confidence": conf, "novelty": novelty,
            "stop":        torch.sigmoid(stop_logit),
            "routing":     routing, "threat": threat, "survival": survival,
            "commit_ok":   commit_ok,
            "action_idx":  action_idx, "action_probs": action_probs,
            "nt":          {n: float(self.transmitters.get(n).mean()) for n in NT_NAMES},
            "qualia":      state.get("qualia", torch.zeros_like(state["floating_thought"])).detach(),
            "thought_valence": state["thought_valence"].detach(),
            "rel_memory_size": self.relational_memory.size,
            "rpe":         rpe.detach(),
            "wanting":     meso_out["wanting"].detach(),
            "liking":      meso_out["liking"].detach(),
            "consolidation": meso_out["consolidation"].detach(),
            "learning_gain": meso_out["learning_gain"].detach(),
            "consciousness": c_metrics,
            "cerebellar_error": float(cereb_error.mean()),
            "cortical_burst":   float(cortical_out["burst"].mean()),
            "entorhinal_velocity": float(entorh["velocity"].norm(dim=-1).mean()),
            "claustrum_salience":  float(claustrum_out["salience"].mean()),
            "geometry": {
                "curvature":    float(geo_out["curvature"].mean()),
                "stream_gates": geo_out["stream_gates"].mean(0).tolist(),
            },
            "thought_transformer": {
                "consistency": float(tt_out["consistency"].mean()),
                "depth":       float(tt_out["reasoning_depth"].mean()),
            },
        }
        return logits2, state, info

    # ------------------------------------------------------------------
    # Convergent DMN loop
    # ------------------------------------------------------------------
    @torch.no_grad()
    def convergent_think(self, ids, state, max_iters=6, on_step=None):
        prev_action = state.get("prev_action_idx")
        converged   = False
        logits      = None
        info        = {}
        for i in range(max_iters):
            logits, state, info = self.cognitive_step(ids, state)
            cur_action = info.get("action_idx")
            critic_ok  = not info.get("survival", torch.zeros(1)).any()
            if on_step:
                on_step(i, info)
            if prev_action is not None and critic_ok and cur_action is not None:
                if (cur_action == prev_action).all():
                    converged = True
                    break
            prev_action = cur_action
        info["converged"]    = converged
        info["think_iters"]  = i + 1
        return logits, state, info

    # ------------------------------------------------------------------
    # Mind-wandering
    # ------------------------------------------------------------------
    @torch.no_grad()
    def wander(self, ids, state, max_steps=8, on_step=None):
        last_info = {}
        for i in range(max_steps):
            _l, state, info = self.cognitive_step(ids, state, allow_emit=False)
            last_info = info
            if on_step:
                on_step(i, info)
            if info.get("survival", torch.zeros(1)).any():
                break
            if float(info.get("stop", torch.zeros(1)).mean()) > 0.7:
                break
        return last_info

    # ------------------------------------------------------------------
    # Dream
    # ------------------------------------------------------------------
    @torch.no_grad()
    def dream(self, state, max_steps=20, environment="random", seed=42,
              on_step=None):
        from .environments.virtual_world import create_environment
        from .tokenizer import Tokenizer
        env  = create_environment(environment, seed=seed)
        tok  = Tokenizer()
        last = {}
        for i in range(max_steps):
            frame  = env.step()
            text   = frame.to_text()
            ids    = torch.tensor([tok.encode(text)], dtype=torch.long,
                                  device=state["floating_thought"].device)
            ids    = ids[:, :self.cfg.lang_ctx]
            _l, state, info = self.cognitive_step(ids, state, allow_emit=False)
            info["sensory_frame"] = frame.to_dict()
            last = info
            if on_step:
                on_step(i, info)
            if self.cfg.enable_narrative:
                self.narrative_system.record_world(
                    state["floating_thought"][0].detach(),
                    content=text[:200], valence=frame.valence,
                    salience=frame.arousal)
        return last

    # ------------------------------------------------------------------
    # Continuous sensory-motor loop
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_continuous(
        self,
        env=None,
        max_steps: int = 500,
        device=None,
        on_step=None,
        gamma: float = 0.99,
        seed: int = 42,
    ) -> dict:
        """Closed perception-action loop grounded in a virtual environment.

        Each tick:
          1. Read sensory frame from env (or env.current_frame() if first)
          2. Tokenize frame text → ids
          3. Run cognitive_step → GWS broadcast, motor selection, metrics
          4. Map BG action_idx → env action index (cyclic modulo)
          5. Step env with that action → next frame (with reward in valence)
          6. Inject env valence as DA reward signal
          7. Write high-salience frames to hippocampal memory

        Args:
            env:       Environment instance. None → GridWorld (action-capable).
            max_steps: Maximum ticks before returning.
            device:    Torch device string or object. None → infer from model.
            on_step:   Callback(step_idx, frame, logits, info) per tick.
            gamma:     Discount factor for cumulative return tracking.
            seed:      RNG seed used when creating the default GridWorld.

        Returns:
            dict: episode statistics
              total_return    — discounted cumulative valence reward
              steps           — actual ticks executed
              mean_phi        — mean integrated information Φ
              mean_novelty    — mean CA1 novelty signal
              final_nt        — NT levels at episode end
              action_histogram — count of each env action taken
        """
        from .environments.virtual_world import GridWorld, GRID_ACTIONS
        from .tokenizer import Tokenizer

        if device is None:
            device = next(self.parameters()).device

        if env is None:
            env = GridWorld(seed=seed)

        tok = Tokenizer()
        n_env_actions = len(GRID_ACTIONS)

        state   = self.init_latents(1, device)
        total_return   = 0.0
        discount       = 1.0
        phi_history    : list[float] = []
        novelty_history: list[float] = []
        action_hist    = {a: 0 for a in GRID_ACTIONS}

        # Bootstrap: get first frame without sending an action
        frame = env.step(action=None)

        for step in range(max_steps):
            # --- encode observation ---
            text     = frame.to_text()
            ids_list = tok.encode(text)
            ids_list = ids_list[-self.cfg.lang_ctx:]
            if len(ids_list) < 2:
                ids_list = [0, 0]
            ids = torch.tensor([ids_list], dtype=torch.long, device=device)

            # --- full cognitive step (GWS + motor + memory) ---
            logits, state, info = self.cognitive_step(ids, state, allow_emit=False)

            # --- select environment action ---
            # BG action_idx → cyclic map to env's discrete action space
            raw_idx  = int(info.get("action_idx", torch.zeros(1))[0].item())
            env_act  = raw_idx % n_env_actions
            act_name = GRID_ACTIONS[env_act]
            action_hist[act_name] += 1

            # --- step environment ---
            next_frame = env.step(action=env_act)

            # --- reward shaping ---
            # Intrinsic: curiosity (novelty) + DA (reward expectation)
            # Extrinsic: env valence (sparse structured reward)
            da_now = float(self.transmitters.get("DA").mean().item())
            nov    = float(info["novelty"].mean().item())
            intrinsic  = nov * 0.3 + da_now * 0.3
            extrinsic  = next_frame.valence               # env-defined reward
            step_reward = intrinsic + extrinsic * 0.5
            total_return += discount * step_reward
            discount     *= gamma

            # Inject extrinsic reward as DA signal so NT system tracks it
            if next_frame.valence > 0.1:
                self.transmitters.release(
                    "DA",
                    torch.full((1,), min(next_frame.valence, 1.0), device=device),
                )
            if next_frame.arousal > 0.4:
                self.transmitters.release(
                    "NE",
                    torch.full((1,), next_frame.arousal * 0.4, device=device),
                )

            # Store salient frames in hippocampus
            if next_frame.novelty > 0.4 or next_frame.valence > 0.3:
                sem_vec = state["floating_thought"].detach()
                nt_now  = self.transmitters.vector().detach()
                self.hippo.store(
                    sem_vec, sem_vec,
                    nt_state=nt_now,
                    valence=next_frame.valence,
                    salience=next_frame.novelty,
                )

            # --- log consciousness metrics ---
            c_metrics = info.get("consciousness", {})
            phi_history.append(c_metrics.get("phi", 0.0))
            novelty_history.append(float(info["novelty"].mean().item()))

            if on_step is not None:
                on_step(step, next_frame, logits, info)

            frame = next_frame

            # Stop if goal reached (GridWorld sets valence=1.0 on goal)
            if getattr(env, "done", False):
                break
            # Stop if survival mode fires (extreme threat)
            if info.get("survival", torch.zeros(1)).any():
                break

        n = max(len(phi_history), 1)
        return {
            "total_return":     total_return,
            "steps":            step + 1,
            "mean_phi":         sum(phi_history) / n,
            "mean_novelty":     sum(novelty_history) / n,
            "final_nt":         {k: float(self.transmitters.get(k).mean())
                                 for k in ["DA", "NE", "5HT", "ACh"]},
            "action_histogram": action_hist,
        }

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, ids, max_new=64, temperature=1.0, top_k=50,
                 on_tick=None, max_silent_streak=3, use_convergent=True):
        cfg    = self.cfg
        device = ids.device
        state  = self.init_latents(ids.size(0), device)
        silent = 0
        step   = 0
        while step < max_new:
            ctx = ids[:, -cfg.lang_ctx:]
            if use_convergent:
                logits, state, info = self.convergent_think(ctx, state)
            else:
                logits, state, info = self.cognitive_step(ctx, state)
            act = int(info.get("action_idx", torch.zeros(1))[0].item())
            force_speak = silent >= max_silent_streak
            do_emit     = (act == ACTION_INDEX["SPEAK"]) or force_speak
            info["emitted"]      = do_emit
            info["forced_speak"] = force_speak
            info["action_name"]  = ACTION_NAMES[act] if act < len(ACTION_NAMES) else "UNKNOWN"
            if on_tick:
                on_tick(step, info)
            if do_emit:
                nl = logits[:, -1] / max(temperature, 1e-5)
                if top_k:
                    v, _ = nl.topk(top_k)
                    nl[nl < v[:, [-1]]] = -float("inf")
                probs = F.softmax(nl, dim=-1)
                nxt   = torch.multinomial(probs, 1)
                ids   = torch.cat([ids, nxt], dim=1)
                silent = 0
                step  += 1
            else:
                if silent > 0:
                    self.wander(ctx, state, max_steps=2)
                silent += 1
        return ids

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------
    def record_episode(self, content, content_vec=None, nt_state=None,
                       emotion=None, tags=None, context=None):
        try:
            import numpy as np
            vec, predicted_vec, surprise = content_vec, None, 1.0
            if vec is None:
                from .tokenizer import Tokenizer
                tok  = Tokenizer()
                enc  = tok.encode(content)[-self.cfg.lang_ctx:]
                dev  = next(self.language.parameters()).device
                if len(enc) >= 2:
                    ids_t = torch.tensor([enc], dtype=torch.long, device=dev)
                    self.language.eval()
                    with torch.no_grad():
                        logits_r, sem_r, _, _ = self.language(ids_t)
                        log_p       = torch.log_softmax(logits_r[0, -2], dim=-1)
                        surprise    = float(-log_p[ids_t[0, -1]].item())
                        predicted_vec = sem_r[0, -2].cpu().numpy()
                    vec = sem_r.squeeze(0).mean(0).cpu().numpy()
                else:
                    vec = np.zeros(self.cfg.d_sem, dtype=np.float32)
            gate = self.comprehension_gate.evaluate(
                obs_vec=np.asarray(vec).flatten(),
                predicted_vec=predicted_vec,
                surprise=surprise,
                consolidated=self.consolidated)
            if gate["write"]:
                full_tags = list(tags or []) + [
                    f"comprehension={gate['comprehension']:.2f}",
                    f"novelty={gate['novelty']:.2f}"]
                self.episodic.add(content, content_vec=vec, nt_state=nt_state,
                                  emotion=emotion, tags=full_tags, context=context)
            return gate
        except Exception:
            return {"write": False, "score": 0.0, "error": True}

    def tag_memory(self, memory_id: int, reward: float,
                   da_level: float = 0.5, insight=None):
        try:
            self.relational_memory.tag_reward(
                memory_id, da_level=da_level,
                reward_signal=reward, insight=insight)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Entity-aware interaction methods
    # ------------------------------------------------------------------

    def identify_speaker(self, text: str, semantic_emb=None,
                         name_hint: str | None = None) -> tuple:
        """Identify the current speaker from text style + optional embedding.

        Returns (entity_id, confidence). Updates self._active_entity_id.
        Call this at the start of each conversational turn.
        """
        if self.entity_store is None:
            return None, 0.0
        eid, conf = self.entity_store.identify(
            text, semantic_emb=semantic_emb, name_hint=name_hint)
        self._active_entity_id = eid
        # Extract knowledge triples and preferences from turn text
        self.entity_store.extract_preferences(eid, text)
        # Record in hypergraph with entity tag
        if self.hypergraph is not None and semantic_emb is not None:
            import numpy as np
            vec = np.asarray(semantic_emb, dtype=np.float32).flatten()
            da = float(self.transmitters.get("DA").detach().mean()) if hasattr(self, 'transmitters') else 0.5
            nt_snap = (self.transmitters.vector().detach()[0].cpu().numpy()
                       if hasattr(self, 'transmitters') else None)
            self.hypergraph.encode(
                content=text[:256], embedding=vec, entity_ref=eid,
                nt_state=nt_snap, salience=0.6 + 0.4 * conf)
        return eid, conf

    def observe_social_exchange(self, my_action_emb, response_emb,
                                 my_text: str = "", response_text: str = ""):
        """Record an (action → response) pair for Markov social learning."""
        if self.hypergraph is None:
            return
        self.hypergraph.observe_social_action(
            my_action_emb, action_text=my_text,
            entity_id=self._active_entity_id)
        self.hypergraph.observe_social_response(
            response_emb, response_text=response_text,
            entity_id=self._active_entity_id)

    def get_entity_knowledge(self, entity_id: str | None = None) -> dict:
        """Return known facts about an entity (or active entity)."""
        eid = entity_id or self._active_entity_id
        if eid is None or self.entity_store is None:
            return {}
        profile = self.entity_store.get_profile(eid)
        if profile is None:
            return {}
        return {
            "name":         profile.name,
            "interactions": profile.interaction_count,
            "confidence":   profile.style_confidence(),
            "preferences":  [{"pred": p.predicate, "obj": p.object,
                               "conf": p.confidence}
                              for p in profile.top_preferences(8)],
            "beliefs":      profile.belief_state.to_vector().tolist(),
            "narrative":    profile.narrative_summary(),
        }

    def get_social_rules(self) -> list:
        """Return all learned social Markov rules."""
        if self.hypergraph is None:
            return []
        return [{"action": r.action_label, "response": r.response_label,
                 "p": r.probability, "n": r.observations}
                for r in self.hypergraph.social_rules]

    def consolidate_memory(self, threshold: float = 0.85):
        self._run_consolidation(da_level=0.5)

    def save_memory_checkpoint(self, path: str):
        from .memory.store import save_memory
        return save_memory(path, self)

    def load_memory_checkpoint(self, path: str):
        from .memory.store import load_memory
        return load_memory(path, self)

    def update_narratives(self):
        try:
            for ep in self.episodic.recent(32):
                self.narrative_self.update(ep.get('content', ''))
                self.narrative_world.update(ep.get('content', ''))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Partial load + misc
    # ------------------------------------------------------------------
    def load_partial(self, state_dict: dict, verbose: bool = True):
        own   = self.state_dict()
        new_sd = dict(own)
        loaded, skipped = 0, 0
        for k, v in state_dict.items():
            if k in own and own[k].shape == v.shape:
                new_sd[k] = v
                loaded   += 1
            else:
                skipped  += 1
        if verbose:
            print(f"Loaded {loaded} keys, skipped {skipped}.")
        self.load_state_dict(new_sd, strict=False)
        return loaded, skipped

    def num_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location="cpu"))

    # ------------------------------------------------------------------
    # Continuous virtual-world sensory-motor loop
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_continuous(self, env_stream=None, n_steps: int = 1000,
                       device=None, tokenizer=None):
        """Run the brain in a closed-loop sensory-motor cycle.

        Each step:
          1. Pull a SensoryFrame from env_stream (defaults to internal stream)
          2. Encode the 6-dim frame vector → d_sem grounding embedding
          3. Build a minimal token context (BOS token) and run forward_lm
          4. Sample an action token from the motor cortex logits
          5. Store the frame + action to episodic memory
          6. Yield (action_idx, frame, out_dict) for the caller to log

        The GWS broadcast state is the "RL observation" and the action token
        is the RL action — providing the interface for an RL wrapper.

        Args:
            env_stream: iterator of SensoryFrame (None = use self._env_stream)
            n_steps:    max steps to run
            device:     torch device (None = auto-detect from parameters)
            tokenizer:  Tokenizer instance for decoding (optional, for logging)

        Yields:
            (action_idx, frame, metrics_dict)
        """
        import itertools
        self.eval()
        device = device or next(self.parameters()).device

        stream = env_stream if env_stream is not None else self._env_stream
        stream = itertools.islice(stream, n_steps)

        # Minimal single-token context (BOS = 1)
        _bos = torch.ones(1, 1, dtype=torch.long, device=device)

        for step, frame in enumerate(stream):
            # Encode grounding frame
            frame_vec = frame.to_vec() if hasattr(frame, 'to_vec') else list(frame)
            frame_emb = self.sensory_encoder.encode_frame(
                frame_vec, device=device, dtype=torch.float32)  # (1, d_sem)

            # Forward pass (targets=None → inference mode)
            out = self.forward_lm(_bos)

            # Motor action
            action_idx = out.get("action_idx", torch.zeros(1, dtype=torch.long, device=device))

            # Extract GWS summary for logging
            novelty    = out.get("novelty", torch.zeros(1, device=device))
            phi_proxy  = self.orchestrator.compute_phi_proxy()

            # Store to episodic memory
            with torch.no_grad():
                sem_snap = frame_emb.squeeze(0)
                nt_snap  = self.transmitters.vector().detach() if hasattr(self, 'transmitters') else None
                self.record_episode(
                    content=f"world_step_{step}",
                    content_vec=sem_snap,
                    nt_state=nt_snap,
                    emotion={"valence": getattr(frame, 'valence', 0.0),
                             "arousal": getattr(frame, 'arousal', 0.0)},
                    tags=["world", "continuous"],
                    context={"step": step, "phi": phi_proxy},
                )

            yield int(action_idx[0].item()), frame, {
                "step":        step,
                "novelty":     float(novelty[0].item()),
                "phi_proxy":   phi_proxy,
                "reentry_rms": float(self.orchestrator._reentry_state.norm().item()),
            }

    def record_episode(self, content, content_vec=None, nt_state=None,
                       emotion=None, tags=None, context=None):
        """Add one experience to the episodic buffer."""
        try:
            self.episodic.add(
                content=content,
                content_vec=content_vec,
                nt_state=nt_state,
                emotion=emotion,
                tags=tags or [],
                context=context or {},
            )
        except Exception:
            pass

    def to_device(self, device):
        self.to(device)
        self.transmitters.to_device(device)
        self.critic.to_device(device)
        self.trophic.to_device(device)
        self.forward_m.to_device(device)
        self.evaluator.to_device(device)
        self.motor.to_device(device)
        self.language.to_device(device)
        self.sensory.to_device(device)
        self.association.to_device(device)
        self.world.to_device(device)
        self.self_m.to_device(device)
        self.gws.to_device(device)
        self.hippo.to_device(device)
        self.dmn.to_device(device)
        self.bg.to_device(device)
        self.thalamus.to_device(device)
        self.vta.to_device(device)
        self.nacc.to_device(device)
        self.lc.to_device(device)
        self.raphe.to_device(device)
        self.nbm.to_device(device)
        self.homeostasis.to_device(device)
        self.rcpt_pfc.to_device(device)
        self.rcpt_hippo.to_device(device)
        self.rcpt_bg.to_device(device)
        self.rcpt_thal.to_device(device)
        self.rcpt_lang.to_device(device)
        self.rcpt_dmn.to_device(device)
        self.projections.to_device(device)
        self.learned_opt.to(device)
