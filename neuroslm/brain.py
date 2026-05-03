import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BrainConfig
from .modules.language import LanguageCortex
from .modules.sensory import TextSensoryCortex
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
from .modules.consciousness import ConsciousnessMetrics
from .modules.cortical_column import CorticalSheet
from .modules.entorhinal import EntorhinalCortex
from .modules.claustrum import Claustrum
from .modules.cerebellum import Cerebellum
from .modules.neural_geometry import NeuralGeometryEngine

from .neurochem import (
    TransmitterSystem, NT_NAMES,
    ReceptorBank, NTShapeRegistry, Receptor,
    Projection, ProjectionGraph,
    VTA, NucleusAccumbens, LocusCoeruleus, RapheNuclei, BasalForebrain,
    SubstantiaNigra, PeriaqueductalGray, HypothalamicCRH,
    Homeostasis,
    ReuptakeSystem,
    ReceptorAdaptation,
    GatedProjectionGraph,
    MesolimbicCircuit,
    PlasticityGate,
)
from .neurochem.growth import TrophicSystem
from .genome import GenePool, Genome
from .genomes import select_build_genome, apply_build_genome, BUILTIN_BUILDS
from .learning import LearningLayer
from .learned_opt import LearnedBackprop


class Ribosome:
    def __init__(self, dna_vm, base_cfg):
        """Ribosome constructs region modules from DNA. It also detects
        hardware and selects/apply a BuildGenome to scale the base config
        before building regions."""
        from .genomes import select_build_genome, apply_build_genome, BUILTIN_BUILDS
        self.dna_vm = dna_vm
        self.base_cfg = base_cfg
        # select build genome: DNA 'build' profile overrides auto-detection
        build = None
        if 'build' in self.dna_vm and 'profile' in self.dna_vm['build'].env:
            prof = self.dna_vm['build'].env.get('profile')
            if isinstance(prof, str) and prof.startswith("'"):
                prof = prof[1:]
            if isinstance(prof, str) and prof in BUILTIN_BUILDS:
                build = BUILTIN_BUILDS[prof]
        if build is None:
            build = select_build_genome()
        self.build_genome = build
        # apply scaling to produce an operational cfg for module construction
        self.cfg = apply_build_genome(self.base_cfg, build)

        self.region_modules = {}
        self.projections = []
        self.nt_production = {}
        self._build_regions()
        self._build_projections()
        self._build_nt_production()

    def _build_regions(self):
        # Example for PFC, DMN, Hippo, BG; extend as needed
        from .modules.pfc import PrefrontalCortex
        from .modules.dmn import DefaultModeNetwork
        from .modules.hippocampus import Hippocampus
        from .modules.basal_ganglia import BasalGanglia
        cfg = self.cfg
        # PFC
        if 'pfc' in self.dna_vm:
            env = self.dna_vm['pfc'].env
            layers = int(env.get('layers', cfg.pfc_layers))
            learning_rule = str(env.get('learning_rule', 'backprop'))
            self.region_modules['pfc'] = PrefrontalCortex(cfg.d_sem, layers, cfg.pfc_heads, learning_rule=learning_rule)
        # DMN
        if 'dmn' in self.dna_vm:
            env = self.dna_vm['dmn'].env
            layers = int(env.get('layers', cfg.dmn_layers))
            connections = str(env.get('connections', 'skip'))
            learning_rule = str(env.get('learning_rule', 'backprop'))
            self.region_modules['dmn'] = DefaultModeNetwork(cfg.d_sem, cfg.gws_slots, layers)
        # Hippocampus
        if 'hippocampus' in self.dna_vm:
            env = self.dna_vm['hippocampus'].env
            layers = int(env.get('layers', cfg.hippo_layers if hasattr(cfg, 'hippo_layers') else 2))
            learning_rule = str(env.get('learning_rule', 'hebbian'))
            self.region_modules['hippocampus'] = Hippocampus(cfg.d_sem, cfg.hippo_capacity, cfg.hippo_topk, cfg.hippo_sparse_k)
        # Basal Ganglia
        if 'basal_ganglia' in self.dna_vm:
            env = self.dna_vm['basal_ganglia'].env
            layers = int(env.get('layers', cfg.bg_layers if hasattr(cfg, 'bg_layers') else 2))
            learning_rule = str(env.get('learning_rule', 'reinforce'))
            self.region_modules['basal_ganglia'] = BasalGanglia(cfg.d_sem, cfg.bg_action_dim, cfg.bg_n_candidates)

    def _build_projections(self):
        # Collect projections from all region DNA
        for region, vm in self.dna_vm.items():
            env = vm.env
            if 'projections' in env:
                for proj in env['projections']:
                    # Each proj: (projection src tgt type nt condition)
                    self.projections.append(proj)

    def _build_nt_production(self):
        # Collect NT production rules from all region DNA
        for region, vm in self.dna_vm.items():
            env = vm.env
            if 'nt_production' in env:
                self.nt_production[region] = env['nt_production']

    def get_module(self, region):
        return self.region_modules.get(region)

    def get_projections(self):
        return self.projections

    def get_nt_production(self, region):
        return self.nt_production.get(region, [])


class Brain(nn.Module):
    def __init__(self, cfg: BrainConfig):
        super().__init__()
        # keep original config and allow a scaled build cfg to be applied via Ribosome
        self.base_cfg = cfg

        # ---- Baseline mode: vanilla transformer only ----
        if getattr(cfg, 'baseline', False):
            self.cfg = cfg
            self.language = LanguageCortex(cfg.vocab_size, cfg.d_hidden, cfg.d_sem,
                                          cfg.lang_layers, cfg.lang_heads, cfg.lang_ctx,
                                          n_kv_heads=cfg.lang_kv_heads,
                                          gradient_checkpointing=cfg.gradient_checkpointing)
            self._baseline = True
            return
        self._baseline = False

        # ---- DNA-driven construction (load templates) ----
        from .dna.dsl import LispVM
        self.dna_vm = {}
        dna_dir = os.path.join(os.path.dirname(__file__), 'dna', 'templates')
        if not os.path.isdir(dna_dir):
            os.makedirs(dna_dir, exist_ok=True)
        for fname in os.listdir(dna_dir):
            if not fname.endswith('.lisp'):
                continue
            region = fname[:-5]
            path = os.path.join(dna_dir, fname)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    code = f.read()
                vm = LispVM()
                vm.run(code)
                self.dna_vm[region] = vm
            except Exception:
                # skip malformed or unreadable templates; they will be logged elsewhere
                continue

        # --- DNA-driven ribosome construction ---
        # Ribosome will choose/apply build genome and expose self.ribosome.cfg
        self.ribosome = Ribosome(self.dna_vm, self.base_cfg)
        self.build_genome = self.ribosome.build_genome
        # Use ribosome's scaled cfg for module construction
        self.cfg = self.ribosome.cfg

        # ---- core cortex & models (built using scaled cfg) ----
        from .neurochem.transmitters import N_NT
        self.language = LanguageCortex(self.cfg.vocab_size, self.cfg.d_hidden, self.cfg.d_sem,
                                      self.cfg.lang_layers, self.cfg.lang_heads, self.cfg.lang_ctx,
                                      n_kv_heads=self.cfg.lang_kv_heads,
                                      n_nt=N_NT,
                                      hebbian_rank=getattr(self.cfg, 'hebbian_rank', 8),
                                      gradient_checkpointing=self.cfg.gradient_checkpointing)
        self.sensory = TextSensoryCortex(self.cfg.d_sem)
        self.association = AssociationCortex(self.cfg.d_sem)
        self.gws = GlobalWorkspace(self.cfg.d_sem, self.cfg.gws_slots, self.cfg.gws_heads)
        self.thalamus = Thalamus(self.cfg.d_sem, self.cfg.d_hidden)
        self.world = WorldModel(self.cfg.d_sem, self.cfg.d_hidden, self.cfg.world_layers)
        self.self_m = SelfModel(self.cfg.d_sem, self.cfg.bg_action_dim, self.cfg.n_neuromods,
                                self.cfg.d_hidden, self.cfg.self_layers)

        # Retrieve DNA-built region modules (if available)
        self.pfc = self.ribosome.get_module('pfc')
        self.dmn = self.ribosome.get_module('dmn')
        self.hippo = self.ribosome.get_module('hippocampus')
        self.bg = self.ribosome.get_module('basal_ganglia')

        # Projections and NT production rules from DNA
        self.dna_projections = self.ribosome.get_projections()
        self.dna_nt_production = self.ribosome.nt_production

        # Forward / evaluator / motor built on scaled dims
        self.forward_m = ForwardModel(self.cfg.d_sem, self.cfg.bg_action_dim, self.cfg.forward_layers)
        self.evaluator = Evaluator(self.cfg.d_sem, len(NT_NAMES))
        self.motor = MotorCortex(self.cfg.bg_action_dim, self.cfg.d_sem, self.cfg.d_hidden)

        # ---- memory systems ----
        from .memory.episodic import EpisodicMemory
        from .memory.consolidated import ConsolidatedMemory
        from .memory.narrative import NarrativeBuffer, NarrativeSystem
        from .memory.mesolimbic import MesolimbicTagger
        from .memory.hippocampal import HippocampalEnrichment
        from .memory.relational_graph import RelationalMemoryGraph
        from .memory.causal import CausalRuleStore
        from .memory.comprehension_gate import ComprehensionGate
        self.episodic = EpisodicMemory(maxlen=2048)
        self.consolidated = ConsolidatedMemory()
        self.narrative_self = NarrativeBuffer(maxlen=2048)
        self.narrative_world = NarrativeBuffer(maxlen=2048)
        self.narrative_system = NarrativeSystem(self.cfg.d_sem)
        self.mesolimbic = MesolimbicTagger()
        self.hippocampal = HippocampalEnrichment(self.consolidated)
        # Relational memory graph — multidimensional associative memory
        # encoding associativity, causality, temporality, patterns, NT state
        self.relational_memory = RelationalMemoryGraph(max_nodes=8192)
        self._last_memory_id: int | None = None  # for causal chaining

        # Causal pattern store — generalizations of (action, ctx) → outcome.
        # Persists across runs via the .mem checkpoint format.
        self.causal = CausalRuleStore(merge_threshold=0.86, min_support=2)

        # Comprehension gate — decides which observations become episodes.
        # Tuned for ~10% write rate so QA training produces concept memories,
        # not verbatim copies.
        self.comprehension_gate = ComprehensionGate(
            threshold=0.05, target_write_rate=0.10)

        # Spontaneous self-reflection + theory-of-mind heads
        from .intelligence.reflection import SpontaneousReflection
        self.reflection = SpontaneousReflection(self.cfg.d_sem)

        # Quantitative consciousness/intelligence metrics
        from .intelligence.metrics import IntelligenceMetrics
        self.metrics = IntelligenceMetrics()

        # ---- Qualia state module ----
        self.qualia = QualiaState(self.cfg.d_sem, len(NT_NAMES))

        # ---- Thought Transformer (meta-learnable reasoning amplifier) ----
        self.thought_transformer = ThoughtTransformer(
            d_sem=self.cfg.d_sem, n_thought_tokens=4, n_layers=2, n_heads=4)

        # ---- Consciousness metrics (oscillations, Φ, binding, ignition) ----
        self.consciousness = ConsciousnessMetrics(d_sem=self.cfg.d_sem)

        # ---- Neuroanatomical structures ----
        # Cortical columns: predictive processing (apical/basal dendrites)
        self.cortical_sheet = CorticalSheet(
            self.cfg.d_sem, n_columns=4, n_minicolumns=8)
        # Entorhinal cortex: grid/place cells for conceptual navigation
        self.entorhinal = EntorhinalCortex(
            self.cfg.d_sem, n_modules=4, cells_per_module=32, n_places=64)
        # Claustrum: cross-modal binding hub
        self.claustrum = Claustrum(self.cfg.d_sem, n_modalities=8)
        # Cerebellum: fast predictive error-driven forward model
        self.cerebellum = Cerebellum(self.cfg.d_sem, expansion=4)

        # ---- Novel intelligence geometry ----
        # Hyperdimensional VSA + fractal attention + manifold geodesic flow
        self.neural_geometry = NeuralGeometryEngine(
            self.cfg.d_sem, n_fractal_levels=3)

        # ---- Virtual environment (sensory grounding) ----
        from .environments.virtual_world import environment_stream, SensoryFrame
        self._env_stream = environment_stream(seed=42, switch_every=50)
        self._last_sensory_frame: SensoryFrame | None = None

        # ---- neurochemistry ----
        self.transmitters = TransmitterSystem()
        self.vta   = VTA()
        self.nacc  = NucleusAccumbens()
        self.lc    = LocusCoeruleus()
        self.raphe = RapheNuclei()
        self.nbm   = BasalForebrain()
        self.substantia_nigra = SubstantiaNigra()
        self.pag   = PeriaqueductalGray()
        self.hypothalamic_crh = HypothalamicCRH()
        self.homeostasis = Homeostasis()
        self.reuptake = ReuptakeSystem()
        self.receptor_adaptation = ReceptorAdaptation()
        self.gated_projections = GatedProjectionGraph()
        self.mesolimbic = MesolimbicCircuit(d_state=self.cfg.d_sem)
        self.plasticity_gate = PlasticityGate()

        # Subconscious threat critic — fast survival circuit
        self.critic = SubconsciousCritic(self.cfg.d_sem)

        # Learning layer: observes neuromodulators and suggests an LR multiplier
        # (legacy scalar gating)
        self.learning_layer = LearningLayer(n_inputs=8, hidden=32, init_scale=1.0)

        # Learned backprop module (meta-optimizer). Transforms per-parameter
        # gradients conditioned on neuromodulatory state. This module is
        # intended to be meta-trained by unrolling inner updates.
        self.learned_opt = LearnedBackprop(n_neuromods=self.cfg.n_neuromods, hidden=32)

        # Gene pool — the DMN's algorithmic CFG, evolved over training
        self.gene_pool = GenePool(pool_size=4, tournament_period=200)

        # ── NT Shape Registry ────────────────────────────────────────
        # Shared latent "protein shapes" for every neurotransmitter.
        # NTs are no longer matched by name — they're matched by shape.
        # Receptors have binding-pocket shapes; affinity = cosine_sim(pocket, nt).
        # This is fully differentiable and evolvable.
        self.nt_shapes = NTShapeRegistry()

        # Receptor banks per region (gain modulators on key streams)
        self.rcpt_pfc = ReceptorBank([
            Receptor("DA",   sign=+1, weight=0.6),  # D1 — enhances PFC working mem
            Receptor("5HT",  sign=+1, weight=0.3),  # 5HT2A — cortical gain
            Receptor("ACh",  sign=+1, weight=0.4),  # M1 — signal-to-noise
            Receptor("GABA", sign=-1, weight=0.4),
        ])
        self.rcpt_hippo = ReceptorBank([
            Receptor("ACh",  sign=+1, weight=0.5),
            Receptor("Glu",  sign=+1, weight=0.4),  # NMDA — plasticity gate
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
            Receptor("ACh",  sign=+1, weight=0.3),
            Receptor("eCB",  sign=-1, weight=0.3),  # CB1 retrograde suppression
        ])
        self.rcpt_dmn = ReceptorBank([
            Receptor("5HT",  sign=-1, weight=0.4),  # 5HT suppresses DMN
            Receptor("ACh",  sign=-1, weight=0.2),
        ])

        # Bind the latent shape registry to all receptor banks
        # This inits each receptor's pocket shape from the NT it canonically binds
        for bank in [self.rcpt_pfc, self.rcpt_hippo, self.rcpt_bg,
                     self.rcpt_thal, self.rcpt_lang, self.rcpt_dmn]:
            bank.bind_registry(self.nt_shapes)

        # Projections graph — built FROM DNA projection genome
        # The projection genome encodes the entire NT connectome:
        # which nuclei project to which regions, carrying which NT.
        # This was previously hardcoded; now it stems from evolvable DNA.
        from .dna.structural_genome import default_projection_genome
        _proj_genome = default_projection_genome()
        _all_region_names = set()
        for pg in _proj_genome.projections:
            _all_region_names.add(pg.src)
            _all_region_names.add(pg.dst)
        region_dims = {r: self.cfg.d_sem for r in _all_region_names}
        self.projections = ProjectionGraph([
            Projection(pg.src, pg.dst, pg.nt,
                       release_scale=pg.release_scale,
                       carries_signal=pg.carries_signal)
            for pg in _proj_genome.projections
        ], region_dims)

        # Neurotrophic system — grows / prunes projections
        self.trophic = TrophicSystem(self.projections)

        # ---- Neural Orchestrator (flow routing with homeostatic gates) ----
        from .intelligence.orchestrator import NeuralOrchestrator
        orchestrator_modules = [
            'world', 'self_m', 'pfc', 'dmn', 'hippo',
            'cerebellum', 'entorhinal', 'claustrum',
        ]
        self.orchestrator = NeuralOrchestrator(
            self.cfg.d_sem, orchestrator_modules,
            n_heads=4, baseline=False)

        # ---- Neural Oscillation Tracker (multi-band spectral analysis) ----
        from .intelligence.oscillations import NeuralOscillationTracker
        self.oscillation_tracker = NeuralOscillationTracker(
            self.cfg.d_sem, n_regions=8, window_size=64)
        self.oscillation_tracker.register_regions([
            'language', 'pfc', 'dmn', 'hippo', 'world',
            'cerebellum', 'gws', 'motor',
        ])

        # ---- Latent Program System (differentiable Lisp algorithms) ----
        from .dna.latent_program import LatentProgramSystem
        self.latent_programs = LatentProgramSystem(
            self.cfg.d_sem, d_latent=128, max_tokens=256, max_steps=16)
        # Initialize programs from DNA templates if available
        for region, vm in self.dna_vm.items():
            try:
                path = os.path.join(
                    os.path.dirname(__file__), 'dna', 'templates', f'{region}.lisp')
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        source = f.read()
                    self.latent_programs.register_program(region, source)
            except Exception:
                pass

        # ── Genome Compiler: Genome → Latent → Lisp → Execute ──
        # This is the central pipeline: each module's behavior is defined by
        # its genome, compiled to a latent embedding, decompiled to readable
        # Lisp, then executed by the DSL interpreter.
        from .dna.compiler import GenomeCompiler, ModuleGenomePool, ALL_REGIONS
        from .dna.structural_genome import BrainDNA
        self.genome_compiler = GenomeCompiler(max_steps=16)
        self.module_genomes = ModuleGenomePool(
            ALL_REGIONS, pool_size=4, tournament_period=200)
        # BrainDNA: structural genomes (architecture/receptors/NT modifiers)
        # + projection genome (NT wiring topology)
        self.brain_dna = BrainDNA.default(ALL_REGIONS)
        # Initial compilation: compile all genomes → extract params → push to modules
        self._recompile_all_genomes()

        # ---- exposure for inspectability ----
        self.last_nt: dict | None = None
        self.last_routing: torch.Tensor | None = None
        self.last_learning_gain: torch.Tensor | None = None
        self.last_action_idx: torch.Tensor | None = None
        self.last_threat: torch.Tensor | None = None
        self.last_survival: torch.Tensor | None = None
        self.last_genome: dict | None = None

    # ====================================================================
    # Helpers
    # ====================================================================
    def _recompile_all_genomes(self):
        """Compile all module genomes and push params into live modules.

        Pipeline:
          1. Algorithmic genome alleles → opcode params → Lisp → DSL Execute → env
          2. Structural genome → receptors, NT modifiers, architecture params
          3. Push combined env + structural into each module's configure_from_genome()

        This is where the genome ACTUALLY controls behavior.
        """
        genomes = self.module_genomes.active_all()
        self._compiled_envs = self.genome_compiler.compile_batch(genomes)

        # Push genome-compiled params + structural genome into live modules
        _module_map = {
            'hippocampus': getattr(self, 'hippo', None),
            'pfc':         getattr(self, 'pfc', None),
            'dmn':         getattr(self, 'dmn', None),
            'basal_ganglia': getattr(self, 'bg', None),
            'critic':      getattr(self, 'critic', None),
            'thalamus':    getattr(self, 'thalamus', None),
            'cerebellum':  getattr(self, 'cerebellum', None),
            'entorhinal':  getattr(self, 'entorhinal', None),
            'claustrum':   getattr(self, 'claustrum', None),
            'cortical_sheet': getattr(self, 'cortical_sheet', None),
            'neural_geometry': getattr(self, 'neural_geometry', None),
        }
        for region, env in self._compiled_envs.items():
            mod = _module_map.get(region)
            if mod is not None and hasattr(mod, 'configure_from_genome'):
                structural = self.brain_dna.get_structural(region)
                mod.configure_from_genome(env, structural=structural)
                # Bind NT shape registry for protein-shape modifier matching
                if hasattr(mod, 'bind_nt_shapes'):
                    mod.bind_nt_shapes(self.nt_shapes)

        # Also sync latents if latent_programs exists (backward compat)
        for region, genome in genomes.items():
            if hasattr(self, 'latent_programs'):
                if region not in self.latent_programs._programs:
                    self.latent_programs.register_program(region)

    def _apply_nt_modifiers_all(self, nt_levels_dict: dict[str, float]):
        """Apply NT modifier rules to ALL genome-configured modules.

        Called each forward tick. NT levels activate modifier rules defined
        in each module's StructuralGenome, shifting their behavioral params.

        Example: high 5HT → PFC select_gate *= 0.5 (safer selection)
        """
        _module_map = {
            'hippocampus': getattr(self, 'hippo', None),
            'pfc':         getattr(self, 'pfc', None),
            'dmn':         getattr(self, 'dmn', None),
            'basal_ganglia': getattr(self, 'bg', None),
            'critic':      getattr(self, 'critic', None),
            'thalamus':    getattr(self, 'thalamus', None),
            'cerebellum':  getattr(self, 'cerebellum', None),
        }
        for region, mod in _module_map.items():
            if mod is not None and hasattr(mod, 'apply_nt_modifiers'):
                mod.apply_nt_modifiers(nt_levels_dict)

    def get_module_lisp(self, region: str) -> str:
        """Get the current compiled Lisp for any module — full inspectability."""
        return self.genome_compiler.get_lisp(region)

    def get_all_module_lisp(self) -> dict[str, str]:
        """Get compiled Lisp for ALL modules."""
        return self.genome_compiler.get_all_lisp()

    def genome_compilation_report(self) -> str:
        """Full report of what each module's genome compiled to."""
        return self.genome_compiler.compilation_report()

    def init_latents(self, batch_size: int, device):
        cfg = self.cfg
        # Only reset transmitters if batch size changed or first call.
        # Preserving NT state across forward passes lets dynamics evolve.
        if (self.transmitters.level.size(0) != batch_size or
                self.transmitters.level.device != device):
            self.transmitters.reset(batch_size, device)
        return {
            "floating_thought": torch.zeros(batch_size, cfg.d_sem, device=device),
            "last_action": torch.zeros(batch_size, cfg.bg_action_dim, device=device),
            "world_h": self.world.init_state(batch_size, device),
            "self_h": self.self_m.init_state(batch_size, device),
            "novelty": torch.zeros(batch_size, device=device),
            "qualia": torch.zeros(batch_size, cfg.d_sem, device=device),
            "prev_action_idx": torch.full((batch_size,), -1, device=device, dtype=torch.long),
            "thought_valence": torch.zeros(batch_size, device=device),
        }

    @staticmethod
    def _act_scalar(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x.detach().abs().mean(dim=-1) - 1.0)

    def _release_via_nuclei(self, signals: dict[str, torch.Tensor]):
        nacc_drive, learning_gain = self.nacc(
            signals["novelty"], signals["reward"],
            signals["curiosity"], signals["ecb"]
        )
        vta_in = torch.stack(
            [signals["rpe"], signals["salience"], nacc_drive, signals["valence"]], dim=-1)
        da_demand = self.vta.demand(vta_in)
        self.transmitters.release("DA", da_demand)

        lc_in = torch.stack(
            [signals["uncertainty"], signals["arousal"], signals["novelty"]], dim=-1)
        ne_demand = self.lc.demand(lc_in)
        self.transmitters.release("NE", ne_demand)

        raphe_in = torch.stack(
            [signals["avg_reward"], signals["time_since_reward"], signals["mood"]], dim=-1)
        ht_demand = self.raphe.demand(raphe_in)
        self.transmitters.release("5HT", ht_demand)

        nbm_in = torch.stack(
            [signals["attention_demand"], signals["novelty"], signals["surprise"]], dim=-1)
        ach_demand = self.nbm.demand(nbm_in)
        self.transmitters.release("ACh", ach_demand)

        return learning_gain, nacc_drive, da_demand

    def _release_via_projections(self, activities: dict[str, torch.Tensor]):
        for i, p in enumerate(self.projections.projections):
            if p.src not in activities:
                continue
            amt = self.projections.release_amount(i, activities[p.src])
            self.transmitters.release(p.nt, amt)

    # ====================================================================
    # Pretraining forward
    # ====================================================================
    def forward_lm(self, ids: torch.Tensor, targets: torch.Tensor | None = None):
        # ---- Baseline: pure language model, no bio modules ----
        if getattr(self, '_baseline', False):
            logits, sem, h, _pc = self.language(ids)
            out = {"logits": logits}
            if targets is not None:
                B, T = ids.shape
                loss = F.cross_entropy(
                    logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1),
                    ignore_index=-100)
                out["loss"] = loss
                out["lm_loss"] = loss.detach()
            return out

        # ---- Full model with bio modules ----
        cfg = self.cfg
        B, T = ids.shape
        device = ids.device
        latents = self.init_latents(B, device)
        # Detach NT vector: meta-training only needs gradients through
        # language params, not transmitter state. Prevents in-place
        # modification errors during create_graph=True backward.
        nt = self.transmitters.vector().detach()          # (B, N_NT)

        # Apply NT modifier rules to all genome-configured modules.
        # This is where neurochemistry ACTUALLY controls module behavior:
        #   high 5HT → PFC safer selection, high NE → PFC urgent tasks, etc.
        with torch.no_grad():
            nt_mean = {n: float(nt[:, i].mean())
                       for i, n in enumerate(NT_NAMES)}
            self._apply_nt_modifiers_all(nt_mean)

        # 1) Language cortex — modulated by ACh / eCB + NT-gated attention
        lang_in_thought = self.rcpt_lang.modulate(
            latents["floating_thought"].unsqueeze(1), nt).squeeze(1)
        logits, sem, h_lang, pred_coding_loss = self.language(
            ids, thought=lang_in_thought, nt=nt)

        # 2) Sensory + association
        sens, salience = self.sensory(sem)
        assoc = self.association([sens])

        # 3) Thalamic router — content-aware MoE-style gate
        routed, routing_probs = self.thalamus(assoc, nt, return_routing=True)
        routed = self.rcpt_thal.modulate(routed.unsqueeze(1), nt).squeeze(1)
        self.last_routing = routing_probs.detach()

        # 3b) Neural Orchestrator — homeostatic pre/post processing
        # The orchestrator wraps each module connection with learnable
        # transformer blocks that maintain stable signal flow. Its gates
        # are meta-trainable: identity coherence + signal stability.
        orch_modules = {
            'world': self.world,
            'cerebellum': self.cerebellum,
            'entorhinal': self.entorhinal,
            'claustrum': self.claustrum,
        }
        orch_out, orch_metrics = self.orchestrator.route(
            routed, orch_modules)
        # Blend orchestrator output with raw routed signal (residual)
        routed = routed + 0.1 * (orch_out - routed)

        # 4) World + self models
        z_world, _wh, world_pred = self.world(routed, latents["world_h"])
        z_self, _sh = self.self_m(latents["last_action"], nt[:, :cfg.n_neuromods],
                                  latents["floating_thought"], latents["self_h"])

        # 4b) Subconscious threat critic — survival circuit
        threat, survival = self.critic(z_world, z_self)
        # If any sample is in survival mode, force NE surge BEFORE the GWS
        # competition (so attention sharpens this very tick).
        if survival.any():
            self.transmitters.release("NE", torch.where(
                survival, torch.full_like(threat, 0.9), torch.zeros_like(threat)))
            nt = self.transmitters.vector().detach()  # refresh after release

        # 5) Global workspace
        candidates = torch.stack(
            [routed, z_world, z_self, latents["floating_thought"]], dim=1)
        slots = self.gws(candidates, ne_temp=nt[:, NT_NAMES.index("NE")])

        # 5b) Read DNA / genome to parameterize the DMN this tick
        genome = self.gene_pool.active()
        novelty_thresh = 0.2 + 0.6 * genome.get("novelty_threshold")
        thought_alpha  = 0.05 + 0.6 * genome.get("thought_alpha")
        replace_bias   = (genome.get("replace_bias") - 0.5) * 2.0

        # 6) DMN + hippocampus (ACh-modulated)
        dmn_query, _stop = self.dmn(slots, latents["floating_thought"])
        dmn_query_mod = self.rcpt_dmn.modulate(dmn_query.unsqueeze(1), nt).squeeze(1)
        recalls, novelty = self.hippo.recall(dmn_query_mod)
        recalls = self.rcpt_hippo.modulate(recalls, nt)

        # 7) PFC selection — modulated by D1/5HT2A/M1
        slots_mod = self.rcpt_pfc.modulate(slots, nt)
        selected, _replace = self.pfc(slots_mod, recalls, latents["floating_thought"])

        # 8) BG + forward + evaluator
        selected_bg = self.rcpt_bg.modulate(selected.unsqueeze(1), nt).squeeze(1)
        action, _conf, _probs = self.bg(selected_bg, nt[:, NT_NAMES.index("DA")])
        wp, sp = self.forward_m(z_world, z_self, action)
        value = self.evaluator(wp, sp, nt)

        # 8b) Motor cortex — discrete action + lang_bias for emission
        _motor_thought, motor_lang_bias, action_idx, action_logits, action_probs = \
            self.motor(action, survival=survival)

        # ---- Single-pass motor conditioning. Re-use hidden states `h` from
        # the first language pass and apply the LM head a second time with
        # the motor bias added. Cost is one extra (d_hidden -> vocab) matmul,
        # vs a full second transformer pass. Bias is added to ALL positions
        # so the motor head receives gradient at every step.
        h_biased = h_lang + motor_lang_bias.unsqueeze(1)
        logits_motor = self.language.lm_head(h_biased)

        # 9) Compute neurochemical signals & release for next tick
        with torch.no_grad():
            zero = torch.zeros(B, device=device)
            if targets is not None:
                p = F.softmax(logits.detach(), dim=-1)
                tgt = targets.clamp_min(0)
                correct_p = p.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                reward_proxy = correct_p.mean(dim=1)             # (B,)
            else:
                reward_proxy = zero
            curiosity = (world_pred.detach() - z_world.detach()).pow(2).mean(-1).sqrt()
            curiosity = (curiosity / (curiosity.mean() + 1e-6)).clamp(0.0, 1.0)
            uncertainty = curiosity
            surprise = novelty.detach()
            arousal = (salience.detach() + curiosity) * 0.5
            mood = self.transmitters.get("5HT").detach()
            avg_reward = reward_proxy
            time_since_reward = (1.0 - reward_proxy).clamp(0.0, 1.0)
            attention_demand = (1.0 - reward_proxy + curiosity).clamp(0.0, 1.0)
            ecb_lvl = self.transmitters.get("eCB").detach()
            valence = reward_proxy
            rpe = (reward_proxy - mood).clamp(-1.0, 1.0).abs()

        signals = dict(
            novelty=novelty.detach(), reward=reward_proxy, curiosity=curiosity,
            ecb=ecb_lvl, rpe=rpe, salience=salience.detach(), valence=valence,
            uncertainty=uncertainty, arousal=arousal,
            avg_reward=avg_reward, time_since_reward=time_since_reward, mood=mood,
            attention_demand=attention_demand, surprise=surprise,
        )
        learning_gain, nacc_drive, da_release = self._release_via_nuclei(signals)

        activities = {
            "VTA":   da_release.detach(),
            "NAcc":  nacc_drive.detach(),
            "LC":    self._act_scalar(z_self),
            "Raphe": self._act_scalar(slots.mean(1)),
            "NBM":   attention_demand,
            "PFC":   self._act_scalar(selected),
            "Hippo": novelty.detach(),
            "BG":    self._act_scalar(action),
        }
        self._release_via_projections(activities)
        self.transmitters.step()

        # ---- Oscillation tracking (record regional activations) ----
        if hasattr(self, 'oscillation_tracker'):
            with torch.no_grad():
                _osc_map = {'language': 0, 'pfc': 1, 'hippocampus': 2,
                            'thalamus': 3, 'basal_ganglia': 4, 'dmn': 5,
                            'gws': 6, 'motor': 7}
                self.oscillation_tracker.record(_osc_map.get('language', 0), sem.detach().mean(1))
                self.oscillation_tracker.record(_osc_map.get('pfc', 1), selected.detach().mean(1) if selected.dim() == 3 else selected.detach())
                self.oscillation_tracker.record(_osc_map.get('hippocampus', 2), dmn_query.detach())
                self.oscillation_tracker.record(_osc_map.get('thalamus', 3), routed.detach())
                self.oscillation_tracker.record(_osc_map.get('basal_ganglia', 4), action.detach())
                self.oscillation_tracker.record(_osc_map.get('dmn', 5), dmn_query_mod.detach())
                self.oscillation_tracker.record(_osc_map.get('gws', 6), slots.detach().mean(1))
                self.oscillation_tracker.record(_osc_map.get('motor', 7), motor_lang_bias.detach())
                # MUST call tick() every forward pass to advance the write pointer
                self.oscillation_tracker.tick()

        # Trophic update: BDNF rises with reward, NGF rises with novelty.
        with torch.no_grad():
            bdnf = float(reward_proxy.mean())
            ngf  = float(novelty.detach().mean())
            self.trophic.update(activities, bdnf=bdnf, ngf=ngf)

        with torch.no_grad():
            self.hippo.store(dmn_query.detach(), selected.detach())

        out = {
            "logits": logits, "world_pred": world_pred, "value": value,
            "novelty": novelty, "selected": selected,
            "learning_gain": learning_gain.detach(),
            "routing": routing_probs.detach(),
            "action_idx": action_idx.detach(),
            "action_probs": action_probs.detach(),
            "threat": threat.detach(),
            "survival": survival.detach(),
        }
        if hasattr(self, 'oscillation_tracker'):
            out["oscillation_snapshot"] = self.oscillation_tracker.compute_spectrum().as_dict()

        if targets is not None:
            # Use motor-conditioned logits — gradient now flows into motor head.
            lm_loss_per = F.cross_entropy(
                logits_motor.reshape(-1, cfg.vocab_size), targets.reshape(-1),
                ignore_index=-100, reduction="none",
            ).reshape(B, T).mean(dim=1)                     # (B,)
            # Mesolimbic gain: modulates per-example loss.  Floor at 1.0 so
            # total_loss >= raw lm_loss (bio gain can only *amplify*).
            meso_gain = (1.0 + 0.5 * learning_gain.detach() *
                         self.transmitters.get("DA").detach()).clamp(min=1.0)
            lm_loss = (lm_loss_per * meso_gain).mean()
            world_loss = F.mse_loss(world_pred, z_world.detach())
            fwd_reg = (wp.pow(2).mean() + sp.pow(2).mean()) * 0.5

            # ---- Motor SPEAK auxiliary loss ----
            # Target: SPEAK if mean next-token confidence on this batch is
            # above threshold, else REMAIN_SILENT. This teaches the motor
            # head to gate emission by predictive confidence: if the model
            # has something useful to say, fire SPEAK; else suggest more
            # thinking.
            with torch.no_grad():
                p_motor = F.softmax(logits_motor.detach(), dim=-1)
                tgt = targets.clamp_min(0)
                tgt_p = p_motor.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                mean_conf = tgt_p.mean(dim=1)                   # (B,)
                speak_target = (mean_conf > cfg.speak_conf_threshold).long()
                from .modules.motor import ACTION_INDEX as _AI
                # Map binary target -> SPEAK index or REMAIN_SILENT index.
                action_target = torch.where(
                    speak_target.bool(),
                    torch.full_like(speak_target, _AI["SPEAK"]),
                    torch.full_like(speak_target, _AI["REMAIN_SILENT"]),
                )
            motor_loss = F.cross_entropy(action_logits, action_target)

            total = (cfg.w_lm * lm_loss
                     + cfg.w_world * world_loss
                     + cfg.w_forward * fwd_reg * 0.01
                     + cfg.w_motor * motor_loss
                     + cfg.w_pred_coding * pred_coding_loss)

            # ── Homeostatic stability loss (meta-training the orchestrator) ──
            # Penalizes identity drift + signal instability so the
            # pre/post processing gates learn to maintain coherent signalling.
            if hasattr(self, 'orchestrator') and not self.orchestrator.baseline:
                identity_drift = orch_metrics.get('identity_drift', 0.0)
                neural_calm = orch_metrics.get('neural_calm', 1.0)
                # We want low drift and high calm
                stability_loss = 0.01 * identity_drift + 0.01 * (1.0 - neural_calm)
                total = total + stability_loss
                out["stability_loss"] = stability_loss
            out["loss"] = total
            out["lm_loss"] = lm_loss_per.mean().detach()
            out["world_loss"] = world_loss.detach()
            out["motor_loss"] = motor_loss.detach()
            out["pred_coding_loss"] = pred_coding_loss.detach()
            out["motor_speak_target_rate"] = speak_target.float().mean().detach()
            # Gene pool fitness = the unweighted LM loss
            self.gene_pool.report(float(lm_loss_per.mean().detach()))

            # Module genome evolution: report fitness, advance clock,
            # recompile genomes when evolution produces new champions
            if hasattr(self, 'module_genomes'):
                self.module_genomes.report_all(float(lm_loss_per.mean().detach()))
                old_step = self.module_genomes.steps
                self.module_genomes.step()
                # Recompile if a tournament just ran (new genome may have won)
                if (self.module_genomes.steps %
                        self.module_genomes.tournament_period == 0 and
                        self.module_genomes.steps != old_step):
                    self._recompile_all_genomes()

        self.last_nt = {n: float(self.transmitters.get(n).detach().mean()) for n in NT_NAMES}
        self.last_learning_gain = learning_gain.detach()
        self.last_action_idx = action_idx.detach()
        self.last_threat = threat.detach()
        self.last_survival = survival.detach()
        self.last_genome = self.gene_pool.active().to_dict()
        self.transmitters.detach_()
        return out

    def lm_logprob(self, ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Compute per-sample log-probability of sequences `ids` under the LM.

        ids: (B, T) token ids (padded with 0). lengths: optional (B,) true lengths
        Returns: (B,) summed log-probabilities (natural log)
        """
        device = ids.device
        B, T = ids.shape
        # Compute logits from language cortex (no conditioning thought)
        # We run the language forward to get logits for all positions.
        logits, _, _, _ = self.language(ids)
        # Compute log softmax across vocab
        logp = F.log_softmax(logits, dim=-1)  # (B, T, V)
        # Gather token log-probs for each position
        token_logp = logp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)  # (B, T)
        if lengths is None:
            # Assume all tokens are valid
            seq_logp = token_logp.sum(dim=1)
        else:
            # lengths is tensor of true lengths per sample
            mask = (torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)).to(token_logp.dtype)
            seq_logp = (token_logp * mask).sum(dim=1)
        return seq_logp

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Return the number of parameters in the Brain.

        If trainable_only is True, only count parameters with requires_grad==True.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    # ====================================================================
    # Inference-time cognitive loop (single tick at ~10 Hz)
    # ====================================================================
    @torch.no_grad()
    def cognitive_step(self, ids: torch.Tensor, state: dict,
                       allow_emit: bool = True):
        """One cognitive tick. Processes sensory input, updates world/self
        models, computes qualia, runs DMN query + hippocampal recall,
        PFC selection, BG action, forward-model simulation, and feeds
        the predicted outcome back into the GWS for the next tick.
        Also drives thought→NT feedback."""
        cfg = self.cfg
        B = ids.size(0)
        nt = self.transmitters.vector()

        # 1) Language cortex — modulated by qualia + ACh/eCB
        lang_in_thought = self.rcpt_lang.modulate(
            state["floating_thought"].unsqueeze(1), nt).squeeze(1)
        logits, sem, h_lang, _ = self.language(ids, thought=lang_in_thought)

        # 2) Sensory + association
        sens, salience = self.sensory(sem)
        assoc = self.association([sens])

        # 2b) Cortical column: predictive processing
        cortical_out = self.cortical_sheet(assoc, state["floating_thought"])
        salience = salience + 0.3 * cortical_out["burst"]
        assoc = assoc + 0.5 * cortical_out["output"]

        # 2c) Entorhinal cortex: conceptual navigation
        entorh = self.entorhinal(state["floating_thought"])
        grid_context = entorh["grid_code"]

        # 3) Thalamic router
        routed, routing = self.thalamus(assoc, nt, return_routing=True)
        routed = self.rcpt_thal.modulate(routed.unsqueeze(1), nt).squeeze(1)

        # 4) World + self models
        z_world, state["world_h"], _wp = self.world(routed, state["world_h"])
        z_self, state["self_h"] = self.self_m(
            state["last_action"], nt[:, :cfg.n_neuromods],
            state["floating_thought"], state["self_h"])

        # 5) Subconscious threat critic
        threat, survival = self.critic(z_world, z_self)
        if survival.any():
            self.transmitters.release("NE", torch.where(
                survival, torch.full_like(threat, 0.9), torch.zeros_like(threat)))
            nt = self.transmitters.vector()

        # 6) Qualia state — produces qualia embedding, modulates thought,
        #    and generates thought→NT feedback
        q_out = self.qualia(state["floating_thought"], nt, threat, z_self)
        state["qualia"] = q_out["qualia"]
        state["thought_valence"] = q_out["thought_valence"]
        # Apply thought→NT feedback: thought content drives NT release
        thought_nt = q_out["thought_nt_demand"]  # (B, n_nt)
        for i, nt_name in enumerate(NT_NAMES):
            if i < thought_nt.size(-1):
                self.transmitters.release(nt_name, thought_nt[:, i] * 0.3)
        nt = self.transmitters.vector()  # refresh after thought→NT

        # 7) GWS — include qualia-modulated thought + grid context as candidates
        modulated_thought = q_out["modulated_thought"]
        candidates = torch.stack(
            [routed, z_world, z_self, modulated_thought, grid_context], dim=1)
        slots = self.gws(candidates, ne_temp=nt[:, NT_NAMES.index("NE")])

        # 7b) Claustrum: cross-modal binding
        claustrum_out = self.claustrum([
            sens, routed, z_world, z_self,
            modulated_thought, grid_context,
            state["qualia"], cortical_out["output"],
        ])
        # Claustral gestalt enriches GWS slots
        gestalt = claustrum_out["gestalt"]  # (B, D)

        # 7c) Neural geometry engine: VSA + fractal + manifold
        geo_out = self.neural_geometry(slots, state["floating_thought"])
        # Geometry-enriched reasoning integrates with GWS
        slots = slots + 0.3 * geo_out["output"].unsqueeze(1).expand_as(slots)

        # 8) DMN query
        dmn_query, stop_logit = self.dmn(slots, modulated_thought)
        dmn_query_mod = self.rcpt_dmn.modulate(dmn_query.unsqueeze(1), nt).squeeze(1)

        # 9) Hippocampal recall — query relational memory graph
        recalls, novelty = self.hippo.recall(dmn_query_mod)
        recalls = self.rcpt_hippo.modulate(recalls, nt)

        # Also query relational memory for associative enrichment
        try:
            query_np = dmn_query_mod[0].cpu().numpy()
            nt_np = nt[0].cpu().numpy()
            rel_nodes = self.relational_memory.query_associative(
                query_np, topk=3, nt_filter=nt_np)
            # If we have a recent memory, do spreading activation from it
            if self._last_memory_id is not None:
                spread_nodes = self.relational_memory.spreading_activation(
                    self._last_memory_id, hops=2, topk=3)
                rel_nodes = rel_nodes + spread_nodes
        except Exception:
            rel_nodes = []

        # 10) Thought Transformer — amplify reasoning on slots + thought
        tt_out = self.thought_transformer(
            state["floating_thought"], slots)
        # tt_out["transformed_thought"] is (B, D), blend into floating thought before PFC
        enhanced_thought = tt_out["transformed_thought"]

        # 11) PFC selection
        slots_mod = self.rcpt_pfc.modulate(slots, nt)
        selected, replace_gate = self.pfc(slots_mod, recalls, enhanced_thought)

        # 12) Update floating thought (genome-driven + qualia-modulated)
        genome = self.gene_pool.active()
        novelty_thresh = 0.2 + 0.6 * genome.get("novelty_threshold")
        thought_alpha = 0.05 + 0.6 * genome.get("thought_alpha")
        replace_mask = (replace_gate > 0.5) | (novelty > novelty_thresh)
        ach = nt[:, NT_NAMES.index("ACh")].unsqueeze(-1)
        smooth = (1 - thought_alpha * ach) * enhanced_thought \
                 + thought_alpha * ach * selected
        state["floating_thought"] = torch.where(
            replace_mask.unsqueeze(-1), selected, smooth)

        # 12) BG action selection
        selected_bg = self.rcpt_bg.modulate(selected.unsqueeze(1), nt).squeeze(1)
        action, conf, _ = self.bg(selected_bg, nt[:, NT_NAMES.index("DA")])

        # 13) Forward model — predict outcome of action
        wp, sp = self.forward_m(z_world, z_self, action)
        value = self.evaluator(wp, sp, nt)

        # 13b) Cerebellum: fast error-driven forward model
        cereb_out = self.cerebellum(
            state["floating_thought"], selected_bg, actual_next=wp)
        # Cerebellar prediction error feeds back as additional learning signal
        cereb_error = cereb_out["error"]  # (B,) scalar error magnitude

        # 14) Motor cortex
        _motor_thought, motor_lang_bias, action_idx, action_logits, action_probs = \
            self.motor(action, survival=survival)
        h_biased = h_lang + motor_lang_bias.unsqueeze(1)
        logits2 = self.language.lm_head(h_biased)

        # 15) Mesolimbic reward circuit — RPE, wanting/liking, consolidation
        zero = torch.zeros(B, device=ids.device)
        state_summary = slots.mean(1)  # (B, d_sem) compressed GWS state
        meso_out = self.mesolimbic(
            state_vec=state_summary,
            reward=zero,  # explicit reward is zero during inference; intrinsic only
            da_level=self.transmitters.get("DA"),
            ecb_level=self.transmitters.get("eCB"),
            gaba_level=self.transmitters.get("GABA"),
            novelty=novelty, salience=salience,
            valence=state["thought_valence"],
            uncertainty=novelty,
        )
        rpe = meso_out["rpe"]

        # 16) NT release (standard nuclei pathway)
        signals = dict(
            novelty=novelty, reward=zero, curiosity=zero,
            ecb=self.transmitters.get("eCB"),
            rpe=rpe.detach(), salience=salience,
            valence=state["thought_valence"],
            uncertainty=novelty, arousal=salience, avg_reward=zero,
            time_since_reward=zero, mood=self.transmitters.get("5HT"),
            attention_demand=salience, surprise=novelty,
        )
        self._release_via_nuclei(signals)

        # 16b) Substantia Nigra: nigrostriatal DA + thalamic GABA gate
        motor_intent = self._act_scalar(action)
        bg_act = self._act_scalar(selected_bg)
        d2_fb = self.mesolimbic.d2_feedback(
            self.transmitters.get("DA"), motor_intent).detach()
        sn_da, sn_gaba = self.substantia_nigra(
            motor_intent, bg_act, d2_fb,
            zero, zero, self.transmitters.get("GABA"))
        self.transmitters.release("DA", sn_da * 0.5)
        self.transmitters.release("GABA", sn_gaba * 0.3)

        # 16c) Mesolimbic DA release (VTA→NAcc, gated by D2 + CB1)
        self.transmitters.release("DA", meso_out["da_release_demand"])

        # 16d) Receptor-gated projections — NT flow based on receptor activity
        activities = {
            "PFC": self._act_scalar(selected), "BG": self._act_scalar(action),
            "Hippo": novelty, "VTA": self.transmitters.get("DA"),
            "NAcc": meso_out["wanting"].detach(),
            "LC": self._act_scalar(z_self),
            "Raphe": self._act_scalar(state_summary), "NBM": salience,
            "SNr": sn_gaba.detach(), "Thalamus": self._act_scalar(routed),
        }
        nt_refreshed = self.transmitters.vector()
        gated_releases = self.gated_projections.gated_release(nt_refreshed, activities)
        for nt_name, amount in gated_releases.items():
            self.transmitters.release(nt_name, amount.clamp(0.0, 0.5))

        # 16e) Standard projections + trophic update
        self._release_via_projections(activities)
        self.trophic.update(activities, bdnf=0.0, ngf=float(novelty.mean()))

        # 17) Reuptake: transporters clear excess NT from synaptic cleft
        self.reuptake.clear(self.transmitters)
        # Adapt transporter density over long timescales
        self.reuptake.adapt_density(self.transmitters)

        # 18) Receptor desensitization / sensitization
        self.receptor_adaptation.update(self.transmitters)

        # 19) Transmitter decay + vesicle replenishment
        self.transmitters.step()
        self.hippo.store(dmn_query, selected)

        # 20) Encode to relational memory if salient enough
        # Use mesolimbic consolidation strength to decide
        consol = float(meso_out["consolidation"].mean())
        try:
            sal_val = float(salience.mean())
            if consol > 0.3 or sal_val > 0.3 or float(novelty.mean()) > 0.4:
                from .tokenizer import Tokenizer
                tok = Tokenizer()
                content = tok.decode(ids[0].tolist())
                mem_vec = sem[0].cpu().numpy()
                nt_snap = nt[0].cpu().numpy()
                valence = float(state["thought_valence"][0])
                mid = self.relational_memory.encode(
                    content, mem_vec, nt_snap, valence=valence,
                    salience=max(sal_val, consol),
                    causal_parent=self._last_memory_id)
                self._last_memory_id = mid
        except Exception:
            pass

        state["last_action"] = action
        state["prev_action_idx"] = action_idx

        # Consciousness metrics — oscillations, binding, ignition, Φ
        c_metrics = self.consciousness.update(
            module_outputs={
                "pfc": selected, "dmn": dmn_query,
                "world": z_world, "self": z_self,
                "sensory": sens, "language": sem.mean(1),
            },
            gws_slots=slots,
            floating_thought=state["floating_thought"],
            novelty=novelty,
            routing=routing,
        )

        # Narrative recording — autobiographical (thought + valence)
        self.narrative_system.record_autobiographical(
            state["floating_thought"][0].detach(),
            valence=float(state["thought_valence"][0]),
            salience=float(salience.mean()))
        # World narrative from world model
        self.narrative_system.record_world(
            z_world[0].detach(),
            valence=float(rpe.mean()))

        info = {
            "value": value, "confidence": conf, "novelty": novelty,
            "stop": torch.sigmoid(stop_logit), "routing": routing,
            "threat": threat, "survival": survival,
            "action_idx": action_idx, "action_probs": action_probs,
            "nt": {n: float(self.transmitters.get(n).mean()) for n in NT_NAMES},
            "genome_id": genome.id,
            "qualia": q_out["qualia"].detach(),
            "thought_valence": q_out["thought_valence"].detach(),
            "rel_memory_size": self.relational_memory.size,
            "rpe": rpe.detach(),
            "wanting": meso_out["wanting"].detach(),
            "liking": meso_out["liking"].detach(),
            "consolidation": meso_out["consolidation"].detach(),
            "learning_gain": meso_out["learning_gain"].detach(),
            "receptor_sensitivity": self.receptor_adaptation.info(),
            "consciousness": c_metrics,
            "thought_transformer": {
                "consistency": float(tt_out["consistency"].mean()),
                "depth": float(tt_out["reasoning_depth"].mean()),
            },
            "narrative": self.narrative_system.info(),
            "cortical_burst": float(cortical_out["burst"].mean()),
            "entorhinal_velocity": float(entorh["velocity"].norm(dim=-1).mean()),
            "claustrum_salience": float(claustrum_out["salience"].mean()),
            "cerebellar_error": float(cereb_error.mean()),
            "geometry": {
                "curvature": float(geo_out["curvature"].mean()),
                "stream_gates": geo_out["stream_gates"].mean(0).tolist(),
            },
        }
        return logits2, state, info

    # ----------------------------------------------------------------
    # Convergent DMN loop — iterates cognitive_step until BG action
    # stabilizes (same action 2x in a row) AND critic does not block,
    # or max_iters is reached.
    # ----------------------------------------------------------------
    @torch.no_grad()
    def convergent_think(self, ids: torch.Tensor, state: dict,
                         max_iters: int = 6, on_step=None) -> tuple:
        """Run the DMN reasoning loop until action converges or max_iters."""
        prev_action = state.get("prev_action_idx", None)
        converged = False
        logits = None
        info = {}
        for i in range(max_iters):
            logits, state, info = self.cognitive_step(ids, state)
            current_action = info["action_idx"]
            critic_ok = not info["survival"].any()
            if on_step:
                on_step(i, info)
            # Check convergence: same action as last iteration + critic OK
            if prev_action is not None and critic_ok:
                same = (current_action == prev_action).all()
                if same:
                    converged = True
                    break
            prev_action = current_action
        info["converged"] = converged
        info["think_iters"] = i + 1
        return logits, state, info

    # ----------------------------------------------------------------
    # Mind wandering — runs when no sensory input is given. The DMN
    # generates queries from floating thought + qualia, hippocampus
    # recalls associated memories, and thought evolves. Produces
    # coherent internal narrative driven by memory + qualia.
    # ----------------------------------------------------------------
    @torch.no_grad()
    def wander(self, ids: torch.Tensor, state: dict, max_steps: int = 8,
               on_step=None) -> dict:
        """Mind wandering loop. Floating thought drifts through memory
        associations, modulated by qualia. No tokens are emitted."""
        last_info = {}
        for i in range(max_steps):
            _logits, state, info = self.cognitive_step(ids, state, allow_emit=False)
            last_info = info
            if on_step is not None:
                on_step(i, info)
            # Break if a threat interrupts mind wandering (survival mode)
            if info["survival"].any():
                break
            # Also break if stop signal is high (DMN says enough thinking)
            if float(info["stop"].mean()) > 0.7:
                break
        return last_info

    @torch.no_grad()
    def dream(self, state: dict, max_steps: int = 20,
              environment: str = "random", seed: int = 42,
              on_step=None) -> dict:
        """Embodied dreaming: the brain processes sensory input from a
        simulated virtual environment. Grounds cognition in experience.

        The virtual world feeds sensory frames to the cortex each tick,
        driving mind wandering, narrative formation, and memory encoding.
        """
        from .environments.virtual_world import create_environment
        from .tokenizer import Tokenizer
        env = create_environment(environment, seed=seed)
        tok = Tokenizer()
        last_info = {}
        for i in range(max_steps):
            # Get sensory frame from virtual world
            frame = env.step()
            self._last_sensory_frame = frame
            # Encode sensory text to tokens for the language cortex
            text = frame.to_text()
            ids = torch.tensor([tok.encode(text)], dtype=torch.long,
                               device=state["floating_thought"].device)
            ids = ids[:, :self.cfg.lang_ctx]  # truncate to context
            # Run cognitive step with virtual sensory input
            _logits, state, info = self.cognitive_step(ids, state, allow_emit=False)
            info["sensory_frame"] = frame.to_dict()
            last_info = info
            if on_step:
                on_step(i, info)
            # Record to narrative system
            self.narrative_system.record_world(
                state["floating_thought"][0].detach(),
                content=text[:200],
                valence=frame.valence, salience=frame.arousal)
        return last_info

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new: int = 64,
                 temperature: float = 1.0, top_k: int = 50,
                 on_tick=None, max_silent_streak: int = 3,
                 use_convergent: bool = True):
        """Generate tokens. Uses convergent DMN loop by default — the model
        thinks until its action stabilises before emitting each token."""
        cfg = self.cfg
        device = ids.device
        state = self.init_latents(ids.size(0), device)
        silent_streak = 0
        step = 0
        while step < max_new:
            ctx = ids[:, -cfg.lang_ctx:]
            if use_convergent:
                logits, state, info = self.convergent_think(ctx, state)
            else:
                logits, state, info = self.cognitive_step(ctx, state)
            from .modules.motor import ACTION_NAMES, ACTION_INDEX
            act = int(info["action_idx"][0].item())
            force_speak = silent_streak >= max_silent_streak
            do_emit = (act == ACTION_INDEX["SPEAK"]) or force_speak
            info["emitted"] = do_emit
            info["forced_speak"] = force_speak
            info["action_name"] = ACTION_NAMES[act]
            if on_tick is not None:
                on_tick(step, info)
            if do_emit:
                next_logits = logits[:, -1] / max(temperature, 1e-5)
                if top_k:
                    v, _ = next_logits.topk(top_k)
                    next_logits[next_logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(next_logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
                ids = torch.cat([ids, nxt], dim=1)
                silent_streak = 0
                step += 1
            else:
                # Mind wander during silent streaks
                if silent_streak > 0:
                    self.wander(ctx, state, max_steps=2)
                silent_streak += 1
        return ids

    # ====================================================================
    # Partial checkpoint loading (transfer v0.1 weights)
    # ====================================================================
    def load_partial(self, state_dict: dict, verbose: bool = True) -> tuple[int, int]:
        own = self.state_dict()
        loaded, skipped = 0, 0
        new_sd = dict(own)
        for k, v in state_dict.items():
            if k in own and own[k].shape == v.shape:
                new_sd[k] = v
                loaded += 1
            else:
                skipped += 1
        if verbose:
            print(f"Loaded {loaded} keys, skipped {skipped}.")
        self.load_state_dict(new_sd, strict=False)
        return loaded, skipped

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location="cpu"))

    def to_device(self, device):
        for m in self.modules():
            m.to(device)
        self.transmitters.to_device(device)
        self.critic.to_device(device)
        self.gene_pool.to_device(device)
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

    # ------------------ Memory helpers ------------------
    def record_episode(self, content, content_vec=None, nt_state=None, emotion=None, tags=None, context=None):
        """Record an episodic memory through the comprehension gate.

        Most observations are *not* stored — only those that score high
        on (surprise × comprehension × novelty). This is what turns QA
        training into concept-memory formation rather than rote copying.

        Returns the gate evaluation dict (use it to log gate behavior).
        """
        try:
            import numpy as np
            vec = content_vec
            predicted_vec = None
            surprise = 1.0
            if vec is None:
                from .tokenizer import Tokenizer
                tok = Tokenizer()
                ids = tok.encode(content)
                ids = ids[-self.cfg.lang_ctx:]
                import torch
                device = next(self.language.parameters()).device if any(True for _ in self.language.parameters()) else torch.device('cpu')
                if len(ids) >= 2:
                    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
                    self.language.eval()
                    with torch.no_grad():
                        logits, sem, _, _ = self.language(ids_t)
                        # surprise: NLL of last token under prefix
                        log_p = torch.log_softmax(logits[0, -2], dim=-1)
                        surprise = float(-log_p[ids_t[0, -1]].item())
                        # predicted "next" embedding (mean of last two)
                        predicted_vec = sem[0, -2].cpu().numpy()
                    vec = sem.squeeze(0).mean(0).cpu().numpy()
                else:
                    vec = np.zeros(self.cfg.d_sem, dtype=np.float32)
            # Comprehension gate decision
            gate = self.comprehension_gate.evaluate(
                obs_vec=np.asarray(vec).flatten(),
                predicted_vec=predicted_vec,
                surprise=surprise,
                consolidated=self.consolidated,
            )
            if gate["write"]:
                tags = list(tags or []) + [
                    f"comprehension={gate['comprehension']:.2f}",
                    f"novelty={gate['novelty']:.2f}",
                ]
                self.episodic.add(content, content_vec=vec, nt_state=nt_state,
                                  emotion=emotion, tags=tags, context=context)
            return gate
        except Exception:
            return {"write": False, "score": 0.0, "error": True}

    def tag_memory(self, memory_id: int, reward: float, insight=None):
        try:
            self.mesolimbic.tag(memory_id, reward, insight=insight)
        except Exception:
            return

    def consolidate_memory(self, threshold: float = 0.85):
        """Consolidate recent episodes into the graph AND extract causal
        rules of the form (action_ctx) → outcome_valence.

        Causal extraction reads the mesolimbic tag (reward) attached to
        each consolidated episode and treats consecutive episodes as
        (context_t, action_t) → outcome_t+1.
        """
        try:
            import numpy as np
            episodes = self.episodic.recent(256)
            for ep in episodes:
                if 'content_vec' not in ep:
                    ep['content_vec'] = getattr(ep, 'content_vec', None) or (0.0,)
            self.consolidated.consolidate(episodes, threshold=threshold)

            # Causal extraction: pair (prev → curr) and use mesolimbic
            # reward tags as outcome valence. Generalizes "action made
            # entity happy/sad" patterns over many examples.
            for i in range(1, len(episodes)):
                prev, curr = episodes[i - 1], episodes[i]
                ctx = np.asarray(prev.get('content_vec', [0.0])).flatten()
                act = np.asarray(curr.get('content_vec', [0.0])).flatten()
                # outcome valence: prefer mesolimbic tag, fall back to NT
                outcome = 0.0
                try:
                    tag = self.mesolimbic.get_tag(i) if hasattr(self.mesolimbic, 'get_tag') else None
                    if tag is not None:
                        outcome = float(tag.get('reward', 0.0))
                except Exception:
                    pass
                if outcome == 0.0:
                    nt = curr.get('nt_state')
                    if nt is not None:
                        nt = np.asarray(nt).flatten()
                        # crude: DA-rich = positive; CRH/cortisol = negative
                        outcome = float(nt[:1].mean() - nt[-1:].mean()) if nt.size >= 2 else 0.0
                if abs(outcome) > 0.05:
                    self.causal.observe(act, ctx, outcome,
                                        step=getattr(self, '_global_step', 0))
            self.causal.prune(max_rules=2048)
        except Exception:
            return

    # ------------------ Memory checkpoint (Git-LFS shippable) ------------------
    def save_memory_checkpoint(self, path):
        """Save consolidated graph + narratives + causal rules to a `.mem`
        file. Independent of model weights — transferable to a fresh
        model with matching d_sem.
        """
        from .memory.store import save_memory
        return save_memory(path, self)

    def load_memory_checkpoint(self, path):
        """Restore memory state from a `.mem` file (does not touch model
        weights). Embeddings of differing dim are zero-padded/truncated.
        """
        from .memory.store import load_memory
        return load_memory(path, self)

    def update_narratives(self):
        try:
            # Simple heuristic: collect recent episodes and append to narrative buffers
            recent = self.episodic.recent(32)
            for ep in recent:
                self.narrative_self.update(ep.get('content', ''))
                self.narrative_world.update(ep.get('content', ''))
        except Exception:
            return