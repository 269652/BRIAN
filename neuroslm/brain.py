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

from .neurochem import (
    TransmitterSystem, NT_NAMES,
    ReceptorBank,
    Projection, ProjectionGraph,
    VTA, NucleusAccumbens, LocusCoeruleus, RapheNuclei, BasalForebrain,
    Homeostasis,
)
from .neurochem.growth import TrophicSystem
from .neurochem.receptors import Receptor
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
        self.language = LanguageCortex(self.cfg.vocab_size, self.cfg.d_hidden, self.cfg.d_sem,
                                      self.cfg.lang_layers, self.cfg.lang_heads, self.cfg.lang_ctx)
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
        from .memory.narrative import NarrativeBuffer
        from .memory.mesolimbic import MesolimbicTagger
        from .memory.hippocampal import HippocampalEnrichment
        from .memory.relational_graph import RelationalMemoryGraph
        self.episodic = EpisodicMemory(maxlen=2048)
        self.consolidated = ConsolidatedMemory()
        self.narrative_self = NarrativeBuffer(maxlen=2048)
        self.narrative_world = NarrativeBuffer(maxlen=2048)
        self.mesolimbic = MesolimbicTagger()
        self.hippocampal = HippocampalEnrichment(self.consolidated)
        # Relational memory graph — multidimensional associative memory
        # encoding associativity, causality, temporality, patterns, NT state
        self.relational_memory = RelationalMemoryGraph(max_nodes=8192)
        self._last_memory_id: int | None = None  # for causal chaining

        # ---- Qualia state module ----
        self.qualia = QualiaState(self.cfg.d_sem, len(NT_NAMES))

        # ---- neurochemistry ----
        self.transmitters = TransmitterSystem()
        self.vta   = VTA()
        self.nacc  = NucleusAccumbens()
        self.lc    = LocusCoeruleus()
        self.raphe = RapheNuclei()
        self.nbm   = BasalForebrain()
        self.homeostasis = Homeostasis()

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

        # Projections graph — moves NT (and optionally signals) between regions
        region_dims = {
            "VTA":   self.cfg.d_sem, "NAcc": self.cfg.d_sem, "LC": self.cfg.d_sem,
            "Raphe": self.cfg.d_sem, "NBM":  self.cfg.d_sem,
            "PFC":   self.cfg.d_sem, "Hippo": self.cfg.d_sem, "BG": self.cfg.d_sem,
            "Thalamus": self.cfg.d_sem, "Language": self.cfg.d_sem, "DMN": self.cfg.d_sem,
        }
        self.projections = ProjectionGraph([
            Projection("VTA",   "NAcc",     "DA",  release_scale=1.0, carries_signal=False),
            Projection("VTA",   "PFC",      "DA",  release_scale=0.8, carries_signal=False),
            Projection("VTA",   "BG",       "DA",  release_scale=1.0, carries_signal=False),
            Projection("NAcc",  "VTA",      "Glu", release_scale=0.5, carries_signal=False),
            Projection("LC",    "PFC",      "NE",  release_scale=0.7, carries_signal=False),
            Projection("LC",    "Thalamus", "NE",  release_scale=0.7, carries_signal=False),
            Projection("Raphe", "DMN",      "5HT", release_scale=0.6, carries_signal=False),
            Projection("Raphe", "PFC",      "5HT", release_scale=0.5, carries_signal=False),
            Projection("NBM",   "Language", "ACh", release_scale=0.6, carries_signal=False),
            Projection("NBM",   "Hippo",    "ACh", release_scale=0.6, carries_signal=False),
            Projection("PFC",   "VTA",      "Glu", release_scale=0.4, carries_signal=False),
            Projection("Hippo", "NAcc",     "Glu", release_scale=0.5, carries_signal=False),
            Projection("PFC",   "PFC",      "eCB", release_scale=0.3, carries_signal=False),
        ], region_dims)

        # Neurotrophic system — grows / prunes projections
        self.trophic = TrophicSystem(self.projections)

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
    def init_latents(self, batch_size: int, device):
        cfg = self.cfg
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
        cfg = self.cfg
        B, T = ids.shape
        device = ids.device
        latents = self.init_latents(B, device)
        nt = self.transmitters.vector()                  # (B, N_NT)

        # 1) Language cortex — modulated by ACh / eCB
        lang_in_thought = self.rcpt_lang.modulate(
            latents["floating_thought"].unsqueeze(1), nt).squeeze(1)
        logits, sem, h_lang = self.language(ids, thought=lang_in_thought)

        # 2) Sensory + association
        sens, salience = self.sensory(sem)
        assoc = self.association([sens])

        # 3) Thalamic router — content-aware MoE-style gate
        routed, routing_probs = self.thalamus(assoc, nt, return_routing=True)
        routed = self.rcpt_thal.modulate(routed.unsqueeze(1), nt).squeeze(1)
        self.last_routing = routing_probs.detach()

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
            # Suppress 5HT (less patience under threat) by zeroing release demand.
            nt = self.transmitters.vector()  # refresh after release

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

        if targets is not None:
            # Use motor-conditioned logits — gradient now flows into motor head.
            lm_loss_per = F.cross_entropy(
                logits_motor.reshape(-1, cfg.vocab_size), targets.reshape(-1),
                ignore_index=-100, reduction="none",
            ).reshape(B, T).mean(dim=1)                     # (B,)
            # Mesolimbic gain: scales each example's loss by 0.5 + DA·NAcc-gain.
            meso_gain = 0.5 + 0.5 * learning_gain.detach() * \
                              self.transmitters.get("DA").detach()
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
                     + cfg.w_motor * motor_loss)
            out["loss"] = total
            out["lm_loss"] = lm_loss_per.mean().detach()
            out["world_loss"] = world_loss.detach()
            out["motor_loss"] = motor_loss.detach()
            out["motor_speak_target_rate"] = speak_target.float().mean().detach()
            # Gene pool fitness = the unweighted LM loss
            self.gene_pool.report(float(lm_loss_per.mean()))

        self.last_nt = {n: float(self.transmitters.get(n).mean()) for n in NT_NAMES}
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
        logits, _, _ = self.language(ids)
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
        logits, sem, h_lang = self.language(ids, thought=lang_in_thought)

        # 2) Sensory + association
        sens, salience = self.sensory(sem)
        assoc = self.association([sens])

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

        # 7) GWS — include qualia-modulated thought as a candidate
        modulated_thought = q_out["modulated_thought"]
        candidates = torch.stack(
            [routed, z_world, z_self, modulated_thought], dim=1)
        slots = self.gws(candidates, ne_temp=nt[:, NT_NAMES.index("NE")])

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

        # 10) PFC selection
        slots_mod = self.rcpt_pfc.modulate(slots, nt)
        selected, replace_gate = self.pfc(slots_mod, recalls, modulated_thought)

        # 11) Update floating thought (genome-driven + qualia-modulated)
        genome = self.gene_pool.active()
        novelty_thresh = 0.2 + 0.6 * genome.get("novelty_threshold")
        thought_alpha = 0.05 + 0.6 * genome.get("thought_alpha")
        replace_mask = (replace_gate > 0.5) | (novelty > novelty_thresh)
        ach = nt[:, NT_NAMES.index("ACh")].unsqueeze(-1)
        smooth = (1 - thought_alpha * ach) * modulated_thought \
                 + thought_alpha * ach * selected
        state["floating_thought"] = torch.where(
            replace_mask.unsqueeze(-1), selected, smooth)

        # 12) BG action selection
        selected_bg = self.rcpt_bg.modulate(selected.unsqueeze(1), nt).squeeze(1)
        action, conf, _ = self.bg(selected_bg, nt[:, NT_NAMES.index("DA")])

        # 13) Forward model — predict outcome of action
        wp, sp = self.forward_m(z_world, z_self, action)
        value = self.evaluator(wp, sp, nt)

        # 14) Motor cortex
        _motor_thought, motor_lang_bias, action_idx, action_logits, action_probs = \
            self.motor(action, survival=survival)
        h_biased = h_lang + motor_lang_bias.unsqueeze(1)
        logits2 = self.language.lm_head(h_biased)

        # 15) NT release (standard nuclei pathway)
        zero = torch.zeros(B, device=ids.device)
        signals = dict(
            novelty=novelty, reward=zero, curiosity=zero,
            ecb=self.transmitters.get("eCB"),
            rpe=zero, salience=salience, valence=state["thought_valence"],
            uncertainty=novelty, arousal=salience, avg_reward=zero,
            time_since_reward=zero, mood=self.transmitters.get("5HT"),
            attention_demand=salience, surprise=novelty,
        )
        self._release_via_nuclei(signals)
        activities = {
            "PFC": self._act_scalar(selected), "BG": self._act_scalar(action),
            "Hippo": novelty, "VTA": self.transmitters.get("DA"),
            "NAcc": novelty, "LC": self._act_scalar(z_self),
            "Raphe": self._act_scalar(slots.mean(1)), "NBM": salience,
        }
        self._release_via_projections(activities)
        self.trophic.update(activities, bdnf=0.0, ngf=float(novelty.mean()))
        self.transmitters.step()
        self.hippo.store(dmn_query, selected)

        # 16) Encode to relational memory if salient enough
        try:
            sal_val = float(salience.mean())
            if sal_val > 0.3 or float(novelty.mean()) > 0.4:
                from .tokenizer import Tokenizer
                tok = Tokenizer()
                content = tok.decode(ids[0].tolist())
                mem_vec = sem[0].cpu().numpy()
                nt_snap = nt[0].cpu().numpy()
                valence = float(state["thought_valence"][0])
                mid = self.relational_memory.encode(
                    content, mem_vec, nt_snap, valence=valence,
                    salience=sal_val,
                    causal_parent=self._last_memory_id)
                self._last_memory_id = mid
        except Exception:
            pass

        state["last_action"] = action
        state["prev_action_idx"] = action_idx

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
        """Record an episodic memory. If content_vec is None, compute a semantic
        embedding using the model's LanguageCortex (in eval mode). This is a
        best-effort operation — failures are swallowed so training isn't
        interrupted by memory logging issues."""
        try:
            vec = content_vec
            if vec is None:
                # Lazily import tokenizer to avoid circular imports at module load
                from .tokenizer import Tokenizer
                tok = Tokenizer()
                ids = tok.encode(content)
                # Truncate/pad to context length
                ids = ids[-self.cfg.lang_ctx:]
                import torch
                device = next(self.language.parameters()).device if any(True for _ in self.language.parameters()) else torch.device('cpu')
                ids_t = torch.tensor([ids], dtype=torch.long, device=device)
                # Run LM in eval mode to get semantic vector
                self.language.eval()
                with torch.no_grad():
                    _, sem, _ = self.language(ids_t)
                vec = sem.squeeze(0).cpu().numpy()
            self.episodic.add(content, content_vec=vec, nt_state=nt_state, emotion=emotion, tags=tags, context=context)
        except Exception:
            return

    def tag_memory(self, memory_id: int, reward: float, insight=None):
        try:
            self.mesolimbic.tag(memory_id, reward, insight=insight)
        except Exception:
            return

    def consolidate_memory(self, threshold: float = 0.85):
        try:
            episodes = self.episodic.recent(256)
            # Each episode should have a content_vec field; fall back to zero vectors
            for ep in episodes:
                if 'content_vec' not in ep:
                    ep['content_vec'] = getattr(ep, 'content_vec', None) or (0.0,)
            self.consolidated.consolidate(episodes, threshold=threshold)
        except Exception:
            return

    def update_narratives(self):
        try:
            # Simple heuristic: collect recent episodes and append to narrative buffers
            recent = self.episodic.recent(32)
            for ep in recent:
                self.narrative_self.update(ep.get('content', ''))
                self.narrative_world.update(ep.get('content', ''))
        except Exception:
            return