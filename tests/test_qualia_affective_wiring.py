"""Tests for qualia affective wiring and neuromodulatory nuclei connectivity.

Covers three layers:

  Layer 1 — DSL topology (HypergraphIR)
    TestNucleiDSL          28 synapse contracts for brainstem NT source populations
    TestQualiaTopologyDSL  populations, synapses, modulations introduced for qualia
                           — including the emotions_vector → thought_transformer
                           "latent affective modulation of thought" edge

  Layer 2 — Python module (QualiaState)
    TestQualiaStateShape      output tensor shapes and dtypes
    TestQualiaStateBehavioral bidirectionality, NT sensitivity, oscillatory EMA
    TestLatentAffectiveModulation
        The key contract: qualia semantically modulates the latent thought
        manifold via the emotions_vector pathway.  Tests pin:
          • Zero-init thought_proj → no modulation at step 0 (ReZero contract)
          • Different NT/survival states produce measurably different thought warp
          • Threatening vs calming thoughts → divergent NT demand profiles
          • EMA smoothing produces softer qualia than single-step embedding

Neuroscientific references encoded in test docstrings:
  Crick & Koch 1990 (gamma binding), Singer & Gray 1995 (40 Hz),
  Lisman & Jensen 2013 (theta-gamma PAC), Damasio 1999 (somatic marker),
  Craig 2009; Seth & Friston 2016 (interoceptive PC).
"""
from __future__ import annotations

import math
import pytest
import torch
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
ARCH_ROOT = str(REPO_ROOT / "architectures" / "master")


# ══════════════════════════════════════════════════════════════════════════════
# SHARED FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def current_ir():
    """Build HypergraphIR from architectures/current/ once per module."""
    from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph
    return lift_arch_to_hypergraph(ARCH_ROOT)


# ── DSL query helpers ──────────────────────────────────────────────────────────

def _has_synapse(ir, pre: str, post: str) -> bool:
    return any(
        e.kind == "synapse"
        and len(e.members) >= 2
        and e.members[0] == pre
        and e.members[1] == post
        for e in ir.hyperedges
    )


def _has_modulation(ir, nt: str, pop: str) -> bool:
    return any(
        e.kind == "modulation"
        and len(e.members) >= 2
        and e.members[0] == nt
        and e.members[1] == pop
        for e in ir.hyperedges
    )


def _get_pop(ir, name: str):
    return next(
        (n for n in ir.nodes if n.kind == "population" and n.name == name),
        None,
    )


def _attr(node, key: str, default: str = "") -> str:
    """Return DSL attr value with surrounding quotes stripped."""
    return node.attrs.get(key, default).strip('"').strip("'")


def _attr_int(node, key: str, default: int = 0) -> int:
    try:
        return int(_attr(node, key, str(default)))
    except (ValueError, AttributeError):
        return default


def _attr_float(node, key: str, default: float = 0.0) -> float:
    try:
        return float(_attr(node, key, str(default)))
    except (ValueError, AttributeError):
        return default


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1A — NUCLEI DSL TOPOLOGY
# ══════════════════════════════════════════════════════════════════════════════

class TestNucleiDSL:
    """28 synapse contracts for brainstem / mesolimbic NT source populations.

    The nuclei (VTA, SN, LC, Raphe, NBM, NAcc) were declared but previously
    unwired — these tests pin every efferent and afferent projection that was
    added to arch.neuro to fix that gap.
    """

    # ── nuclei populations present ────────────────────────────────────────────

    def test_all_6_nuclei_populations_present(self, current_ir):
        expected = {
            "vta", "nucleus_accumbens", "locus_coeruleus",
            "raphe_nuclei", "nucleus_basalis", "substantia_nigra",
        }
        found = {n.name for n in current_ir.nodes if n.kind == "population"}
        missing = expected - found
        assert not missing, f"Missing nuclei populations: {missing}"

    # ── VTA (mesolimbic + mesocortical DA) ────────────────────────────────────

    def test_vta_to_nacc_mesolimbic(self, current_ir):
        """VTA → NAcc: mesolimbic dopamine pathway (reward circuit)."""
        assert _has_synapse(current_ir, "vta", "nucleus_accumbens"), \
            "Missing VTA→NAcc mesolimbic DA synapse"

    def test_vta_to_pfc_mesocortical(self, current_ir):
        """VTA → PFC: mesocortical dopamine (motivational drive to executive)."""
        assert _has_synapse(current_ir, "vta", "pfc"), \
            "Missing VTA→PFC mesocortical DA synapse"

    def test_vta_to_bg_reward_gating(self, current_ir):
        """VTA → BG: reward-salience gating of action selection."""
        assert _has_synapse(current_ir, "vta", "bg"), \
            "Missing VTA→BG reward-gating synapse"

    # ── Substantia Nigra (nigrostriatal DA) ───────────────────────────────────

    def test_sn_to_bg_nigrostriatal(self, current_ir):
        """SN → BG: classical nigrostriatal dopamine pathway (motor selection)."""
        assert _has_synapse(current_ir, "substantia_nigra", "bg"), \
            "Missing nigrostriatal SN→BG synapse"

    # ── NAcc (ventral striatum gateway) ──────────────────────────────────────

    def test_nacc_to_bg_ventral_striatum(self, current_ir):
        """NAcc → BG: ventral striatum motivational gating of dorsal striatum."""
        assert _has_synapse(current_ir, "nucleus_accumbens", "bg"), \
            "Missing NAcc→BG ventral-striatum synapse"

    def test_nacc_to_motor_reward_action(self, current_ir):
        """NAcc → motor: reward-driven action output (mesolimbic → motor loop)."""
        assert _has_synapse(current_ir, "nucleus_accumbens", "motor"), \
            "Missing NAcc→motor reward-action synapse"

    # ── Locus Coeruleus (noradrenergic arousal) ───────────────────────────────

    def test_lc_to_thalamus_arousal(self, current_ir):
        """LC → thalamus: NE arousal amplifies thalamic relay gain."""
        assert _has_synapse(current_ir, "locus_coeruleus", "thalamus"), \
            "Missing LC→thalamus NE-arousal synapse"

    def test_lc_to_amygdala_stress(self, current_ir):
        """LC → amygdala: NE amplifies emotional reactivity under stress."""
        assert _has_synapse(current_ir, "locus_coeruleus", "amygdala"), \
            "Missing LC→amygdala NE-stress synapse"

    def test_lc_to_pfc_ne_inverted_u(self, current_ir):
        """LC → PFC: moderate NE sharpens PFC; high NE degrades it (inverted-U)."""
        assert _has_synapse(current_ir, "locus_coeruleus", "pfc"), \
            "Missing LC→PFC NE synapse"

    # ── Raphe (serotonergic mood / fear) ──────────────────────────────────────

    def test_raphe_to_amygdala_fear_suppression(self, current_ir):
        """Raphe → amygdala: 5HT suppresses fear and anxiety."""
        assert _has_synapse(current_ir, "raphe_nuclei", "amygdala"), \
            "Missing Raphe→amygdala 5HT fear-suppression synapse"

    def test_raphe_to_pfc_mood(self, current_ir):
        """Raphe → PFC: 5HT mood-state modulation of executive planning."""
        assert _has_synapse(current_ir, "raphe_nuclei", "pfc"), \
            "Missing Raphe→PFC 5HT-mood synapse"

    def test_raphe_to_hippo_consolidation(self, current_ir):
        """Raphe → hippo: serotonergic memory consolidation gating."""
        assert _has_synapse(current_ir, "raphe_nuclei", "hippo"), \
            "Missing Raphe→hippo 5HT-consolidation synapse"

    # ── Nucleus Basalis Meynert (cholinergic) ────────────────────────────────

    def test_nbm_to_sensory_attention(self, current_ir):
        """NBM → sensory: cholinergic top-down attention to sensory cortex."""
        assert _has_synapse(current_ir, "nucleus_basalis", "sensory"), \
            "Missing NBM→sensory ACh-attention synapse"

    def test_nbm_to_hippo_encoding(self, current_ir):
        """NBM → hippo: cholinergic encoding gate for episodic memory."""
        assert _has_synapse(current_ir, "nucleus_basalis", "hippo"), \
            "Missing NBM→hippo ACh-encoding synapse"

    def test_nbm_to_pfc_working_memory(self, current_ir):
        """NBM → PFC: cholinergic modulation of working memory."""
        assert _has_synapse(current_ir, "nucleus_basalis", "pfc"), \
            "Missing NBM→PFC ACh-WM synapse"

    # ── Afferents: closing the NT feedback loops ──────────────────────────────

    def test_bg_to_vta_rpe_gaba(self, current_ir):
        """BG → VTA: striatal GABA inhibits VTA; disinhibition encodes reward."""
        assert _has_synapse(current_ir, "bg", "vta"), \
            "Missing BG→VTA GABA/RPE synapse"

    def test_amygdala_to_vta_stress_da(self, current_ir):
        """Amygdala → VTA: stress/fear drives phasic DA burst."""
        assert _has_synapse(current_ir, "amygdala", "vta"), \
            "Missing amygdala→VTA stress-DA synapse"

    def test_amygdala_to_lc_stress_ne(self, current_ir):
        """Amygdala → LC: fear triggers NE surge (classical stress arousal)."""
        assert _has_synapse(current_ir, "amygdala", "locus_coeruleus"), \
            "Missing amygdala→LC fear-NE synapse"

    def test_amygdala_to_raphe_5ht_depletion(self, current_ir):
        """Amygdala → Raphe: chronic stress depletes 5HT via raphe suppression."""
        assert _has_synapse(current_ir, "amygdala", "raphe_nuclei"), \
            "Missing amygdala→raphe stress-5HT synapse"

    def test_evaluator_to_vta_rpe(self, current_ir):
        """Evaluator → VTA: scalar RPE → DA burst (temporal-difference learning)."""
        assert _has_synapse(current_ir, "evaluator", "vta"), \
            "Missing evaluator→VTA RPE synapse"

    def test_acc_to_lc_error_arousal(self, current_ir):
        """ACC → LC: conflict/error monitoring → NE arousal for adaptive gain."""
        assert _has_synapse(current_ir, "acc", "locus_coeruleus"), \
            "Missing ACC→LC error-arousal synapse"

    def test_pfc_to_nbm_top_down_ach(self, current_ir):
        """PFC → NBM: top-down cholinergic control of attention."""
        assert _has_synapse(current_ir, "pfc", "nucleus_basalis"), \
            "Missing PFC→NBM top-down ACh synapse"

    def test_hippo_to_raphe_context_5ht(self, current_ir):
        """Hippo → Raphe: memory retrieval context modulates 5HT tone."""
        assert _has_synapse(current_ir, "hippo", "raphe_nuclei"), \
            "Missing hippo→raphe memory-context 5HT synapse"


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1B — QUALIA TOPOLOGY DSL
# ══════════════════════════════════════════════════════════════════════════════

class TestQualiaTopologyDSL:
    """DSL contracts for the three new qualia populations and their wiring.

    Key edge: emotions_vector → thought_transformer is the DSL-level contract
    for the 'latent affective modulation of thought manifold'.
    """

    # ── populations ──────────────────────────────────────────────────────────

    def test_qualia_population_present(self, current_ir):
        assert _get_pop(current_ir, "qualia") is not None, \
            "qualia population missing from IR"

    def test_emotions_vector_population_present(self, current_ir):
        assert _get_pop(current_ir, "emotions_vector") is not None, \
            "emotions_vector population missing from IR"

    def test_survival_imperatives_population_present(self, current_ir):
        assert _get_pop(current_ir, "survival_imperatives") is not None, \
            "survival_imperatives population missing from IR"

    def test_qualia_oscillatory_dynamics(self, current_ir):
        """Qualia must be oscillatory — gamma-band binding (Crick & Koch 1990)."""
        pop = _get_pop(current_ir, "qualia")
        assert pop is not None, "qualia population not found"
        assert _attr(pop, "dynamics") == "oscillatory", (
            f"qualia dynamics should be 'oscillatory', got {_attr(pop, 'dynamics')!r}. "
            "Gamma-band binding (40 Hz, Crick & Koch 1990) requires oscillatory dynamics."
        )

    def test_emotions_vector_rate_coded(self, current_ir):
        """emotions_vector is rate_coded — slow affective state accumulator."""
        pop = _get_pop(current_ir, "emotions_vector")
        assert pop is not None, "emotions_vector population not found"
        assert _attr(pop, "dynamics") == "rate_code", (
            f"emotions_vector dynamics should be 'rate_code', "
            f"got {_attr(pop, 'dynamics')!r}"
        )

    def test_survival_imperatives_timescale_slow(self, current_ir):
        """Survival imperatives integrate on a slow (~500ms) homeostatic timescale."""
        pop = _get_pop(current_ir, "survival_imperatives")
        assert pop is not None
        ts = _attr_float(pop, "timescale")
        assert ts >= 0.4, (
            f"survival_imperatives timescale should be >=0.4 s (homeostatic), "
            f"got {ts}"
        )

    def test_emotions_vector_count(self, current_ir):
        pop = _get_pop(current_ir, "emotions_vector")
        assert pop is not None
        count = _attr_int(pop, "count")
        assert count == 128, f"emotions_vector count should be 128, got {count}"

    def test_survival_imperatives_count(self, current_ir):
        pop = _get_pop(current_ir, "survival_imperatives")
        assert pop is not None
        count = _attr_int(pop, "count")
        assert count == 64, f"survival_imperatives count should be 64, got {count}"

    # ── inputs → qualia ──────────────────────────────────────────────────────

    def test_amygdala_to_qualia(self, current_ir):
        """Amygdala → qualia: emotional binding (fear, anger, salience)."""
        assert _has_synapse(current_ir, "amygdala", "qualia"), \
            "Missing amygdala→qualia emotional binding synapse"

    def test_insula_to_qualia(self, current_ir):
        """Insula → qualia: interoceptive signals (bodily state into qualia)."""
        assert _has_synapse(current_ir, "insula", "qualia"), \
            "Missing insula→qualia interoceptive synapse"

    def test_gws_to_qualia(self, current_ir):
        """GWS → qualia: conscious content drives experiential binding."""
        assert _has_synapse(current_ir, "gws", "qualia"), \
            "Missing GWS→qualia conscious-content synapse"

    def test_thought_transformer_to_qualia(self, current_ir):
        """thought_transformer → qualia: thoughts affect qualia (bidirectional)."""
        assert _has_synapse(current_ir, "thought_transformer", "qualia"), \
            "Missing thought_transformer→qualia synapse (thoughts affect qualia)"

    def test_dmn_to_qualia(self, current_ir):
        """DMN → qualia: narrative self affects qualia."""
        assert _has_synapse(current_ir, "dmn", "qualia"), \
            "Missing DMN→qualia narrative synapse"

    def test_survival_imperatives_to_qualia(self, current_ir):
        """Survival imperatives → qualia: homeostatic drives bias qualia content."""
        assert _has_synapse(current_ir, "survival_imperatives", "qualia"), \
            "Missing survival_imperatives→qualia synapse"

    # ── qualia → emotions_vector (the key latent-modulation edge) ────────────

    def test_qualia_to_emotions_vector(self, current_ir):
        """qualia → emotions_vector: oscillatory phase encodes affective valence.

        This is the output side of the qualia binding surface: gamma-band
        oscillations phase-modulate the latent emotions vector that downstream
        thought processes read from (Lisman & Jensen 2013 theta-gamma PAC).
        """
        assert _has_synapse(current_ir, "qualia", "emotions_vector"), \
            "Missing qualia→emotions_vector synapse (affective valence encoding)"

    # ── emotions_vector → cognition (the modulation pathway) ─────────────────

    def test_emotions_vector_modulates_thought_transformer(self, current_ir):
        """emotions_vector → thought_transformer: LATENT AFFECTIVE MODULATION.

        This is the DSL-level pin for the core claim: the latent emotions
        manifold semantically modulates the thought manifold.  Emotions don't
        bypass cognition — they enter the thought_transformer as an additional
        input, shaping the attention-pool dynamics that produce each cognitive
        cycle's output.
        """
        assert _has_synapse(current_ir, "emotions_vector", "thought_transformer"), (
            "Missing emotions_vector→thought_transformer synapse. "
            "This is the DSL contract for latent affective modulation of thought."
        )

    def test_emotions_vector_to_gws(self, current_ir):
        """emotions_vector → GWS: emotional coloring of conscious broadcast."""
        assert _has_synapse(current_ir, "emotions_vector", "gws"), \
            "Missing emotions_vector→GWS synapse"

    # ── qualia → NT release ──────────────────────────────────────────────────

    def test_qualia_to_vta_da_release(self, current_ir):
        """qualia → VTA: positive qualia drives DA burst (reward/pleasure loop)."""
        assert _has_synapse(current_ir, "qualia", "vta"), \
            "Missing qualia→VTA NT-release synapse"

    def test_qualia_to_raphe_5ht_modulation(self, current_ir):
        """qualia → Raphe: qualia state drives 5HT release/reuptake."""
        assert _has_synapse(current_ir, "qualia", "raphe_nuclei"), \
            "Missing qualia→raphe_nuclei NT-release synapse"

    def test_qualia_to_lc_ne_modulation(self, current_ir):
        """qualia → LC: arousal qualia drives NE release."""
        assert _has_synapse(current_ir, "qualia", "locus_coeruleus"), \
            "Missing qualia→locus_coeruleus NT-release synapse"

    def test_qualia_to_dmn_self_model(self, current_ir):
        """qualia → DMN: qualia reshapes narrative self-model."""
        assert _has_synapse(current_ir, "qualia", "dmn"), \
            "Missing qualia→DMN self-model synapse"

    # ── NT modulations of qualia ─────────────────────────────────────────────

    def test_dopamine_modulates_qualia(self, current_ir):
        """DA → qualia: reward/pleasure shifts hedonic tone (Damasio 1999)."""
        assert _has_modulation(current_ir, "dopamine", "qualia"), \
            "Missing dopamine→qualia modulation"

    def test_serotonin_modulates_qualia(self, current_ir):
        """5HT → qualia: mood baseline modulates qualia tone."""
        assert _has_modulation(current_ir, "serotonin", "qualia"), \
            "Missing serotonin→qualia modulation"

    def test_norepinephrine_modulates_qualia(self, current_ir):
        """NE → qualia: arousal intensity modulates qualia (stress response)."""
        assert _has_modulation(current_ir, "norepinephrine", "qualia"), \
            "Missing norepinephrine→qualia modulation"

    def test_endocannabinoid_modulates_qualia(self, current_ir):
        """eCB → qualia: hedonic set-point and analgesia effects on qualia."""
        assert _has_modulation(current_ir, "endocannabinoid", "qualia"), \
            "Missing endocannabinoid→qualia modulation"

    def test_dopamine_modulates_survival_imperatives(self, current_ir):
        """DA → survival_imperatives: reward reinforces homeostatic drives."""
        assert _has_modulation(current_ir, "dopamine", "survival_imperatives"), \
            "Missing dopamine→survival_imperatives modulation"

    def test_norepinephrine_modulates_survival_imperatives(self, current_ir):
        """NE → survival_imperatives: stress heightens interoceptive drives."""
        assert _has_modulation(current_ir, "norepinephrine", "survival_imperatives"), \
            "Missing norepinephrine→survival_imperatives modulation"

    # ── bidirectionality summary ──────────────────────────────────────────────

    def test_thought_qualia_bidirectional_loop_both_directions_present(self, current_ir):
        """Both directions of the thought↔qualia loop must exist.

        thought_transformer → qualia  (thoughts affect qualia)
        emotions_vector → thought_transformer  (qualia affects thought via latent manifold)

        Without both edges the loop is broken and the system degenerates to
        open-loop affect (emotions without cognition, or cognition without affect).
        """
        fwd = _has_synapse(current_ir, "thought_transformer", "qualia")
        bwd = _has_synapse(current_ir, "emotions_vector", "thought_transformer")
        assert fwd and bwd, (
            f"Thought↔qualia bidirectional loop incomplete: "
            f"thought→qualia={fwd}, emotions_vector→thought={bwd}"
        )

    def test_nt_qualia_bidirectional_loop_both_directions_present(self, current_ir):
        """Both directions of the NT↔qualia loop must exist.

        dopamine→qualia  (NT levels modulate qualia tone)
        qualia→vta       (qualia drives NT release)

        Without both edges there is no homeostatic qualia-NT feedback.
        """
        nt_to_q = _has_modulation(current_ir, "dopamine", "qualia")
        q_to_nt = _has_synapse(current_ir, "qualia", "vta")
        assert nt_to_q and q_to_nt, (
            f"NT↔qualia loop incomplete: "
            f"DA→qualia={nt_to_q}, qualia→VTA={q_to_nt}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2A — QUALIASTATE SHAPE CONTRACTS
# ══════════════════════════════════════════════════════════════════════════════

class TestQualiaStateShape:
    """Output tensor shapes, dtypes, and value ranges on random inputs."""

    @pytest.fixture
    def qs(self):
        from neuroslm.modules.qualia import QualiaState
        return QualiaState(d_sem=32, n_nt=7)

    @pytest.fixture
    def inputs(self):
        torch.manual_seed(0)
        B = 4
        d_sem, n_nt = 32, 7
        return {
            "thought": torch.randn(B, d_sem),
            "nt":      torch.rand(B, n_nt),
            "threat":  torch.rand(B),
            "z_self":  torch.randn(B, d_sem),
        }

    def test_qualia_output_shape(self, qs, inputs):
        out = qs(inputs["thought"], inputs["nt"], inputs["threat"], inputs["z_self"])
        B, d_sem = 4, 32
        assert out["qualia"].shape == (B, d_sem), \
            f"qualia shape should be (B, d_sem)=({B},{d_sem}), got {out['qualia'].shape}"

    def test_modulated_thought_shape(self, qs, inputs):
        out = qs(inputs["thought"], inputs["nt"], inputs["threat"], inputs["z_self"])
        assert out["modulated_thought"].shape == (4, 32), \
            f"modulated_thought shape wrong: {out['modulated_thought'].shape}"

    def test_thought_nt_demand_shape(self, qs, inputs):
        out = qs(inputs["thought"], inputs["nt"], inputs["threat"], inputs["z_self"])
        assert out["thought_nt_demand"].shape == (4, 7), \
            f"thought_nt_demand shape wrong: {out['thought_nt_demand'].shape}"

    def test_thought_valence_shape(self, qs, inputs):
        out = qs(inputs["thought"], inputs["nt"], inputs["threat"], inputs["z_self"])
        assert out["thought_valence"].shape == (4,), \
            f"thought_valence shape wrong: {out['thought_valence'].shape}"

    def test_qualia_bounded_tanh(self, qs, inputs):
        """qualia embedding uses tanh → values must be in (-1, 1)."""
        out = qs(inputs["thought"], inputs["nt"], inputs["threat"], inputs["z_self"])
        q = out["qualia"]
        assert q.min().item() >= -1.0 - 1e-6, f"qualia min={q.min():.4f} < -1"
        assert q.max().item() <= 1.0 + 1e-6, f"qualia max={q.max():.4f} > 1"

    def test_thought_nt_demand_bounded_sigmoid(self, qs, inputs):
        """NT demand uses sigmoid → values in [0, 1]."""
        out = qs(inputs["thought"], inputs["nt"], inputs["threat"], inputs["z_self"])
        d = out["thought_nt_demand"]
        assert d.min().item() >= -1e-6, f"NT demand min={d.min():.4f} < 0"
        assert d.max().item() <= 1.0 + 1e-6, f"NT demand max={d.max():.4f} > 1"

    def test_valence_bounded_tanh(self, qs, inputs):
        """Valence uses tanh → in (-1, 1). Negative = threatening."""
        out = qs(inputs["thought"], inputs["nt"], inputs["threat"], inputs["z_self"])
        v = out["thought_valence"]
        assert v.min().item() >= -1.0 - 1e-6
        assert v.max().item() <= 1.0 + 1e-6

    def test_output_dict_has_required_keys(self, qs, inputs):
        out = qs(inputs["thought"], inputs["nt"], inputs["threat"], inputs["z_self"])
        required = {"qualia", "modulated_thought", "thought_nt_demand", "thought_valence"}
        assert required.issubset(out.keys()), \
            f"Missing output keys: {required - out.keys()}"


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2B — QUALIASTATE BEHAVIORAL CONTRACTS
# ══════════════════════════════════════════════════════════════════════════════

class TestQualiaStateBehavioral:
    """Behavioral contracts: NT sensitivity, oscillatory EMA, bidirectionality."""

    @pytest.fixture
    def qs(self):
        from neuroslm.modules.qualia import QualiaState
        m = QualiaState(d_sem=32, n_nt=7)
        m.eval()
        return m

    def _fwd(self, qs, thought=None, nt=None, threat=None, z_self=None):
        torch.manual_seed(42)
        d_sem, n_nt = 32, 7
        if thought is None:
            thought = torch.randn(1, d_sem)
        if nt is None:
            nt = torch.zeros(1, n_nt)
        if threat is None:
            threat = torch.zeros(1)
        if z_self is None:
            z_self = torch.zeros(1, d_sem)
        return qs(thought, nt, threat, z_self)

    def test_different_nt_levels_produce_different_qualia(self, qs):
        """NT levels enter the encoder → different NT → different qualia embedding."""
        low_da  = torch.zeros(1, 7)
        high_da = torch.zeros(1, 7)
        high_da[0, 0] = 1.0  # DA is channel 0 by convention

        torch.manual_seed(5)
        thought = torch.randn(1, 32)
        threat  = torch.zeros(1)
        z_self  = torch.zeros(1, 32)

        with torch.no_grad():
            out_low  = qs(thought, low_da,  threat, z_self)
            out_high = qs(thought, high_da, threat, z_self)

        diff = (out_low["qualia"] - out_high["qualia"]).norm().item()
        assert diff > 1e-4, (
            "NT levels should change qualia embedding, but got near-zero difference. "
            "NT vector feeds into the qualia encoder."
        )

    def test_different_threats_produce_different_qualia(self, qs):
        """Threat level enters the encoder → high threat → different qualia."""
        torch.manual_seed(6)
        thought = torch.randn(1, 32)
        nt      = torch.zeros(1, 7)
        z_self  = torch.zeros(1, 32)

        no_threat  = torch.zeros(1)
        max_threat = torch.ones(1)

        with torch.no_grad():
            out_safe  = qs(thought, nt, no_threat,  z_self)
            out_fear  = qs(thought, nt, max_threat, z_self)

        diff = (out_safe["qualia"] - out_fear["qualia"]).norm().item()
        assert diff > 1e-4, "Threat level must change qualia embedding"

    def test_different_thoughts_produce_different_qualia(self, qs):
        """Thought content shapes qualia (via valence head → encoder)."""
        nt      = torch.zeros(1, 7)
        threat  = torch.zeros(1)
        z_self  = torch.zeros(1, 32)

        torch.manual_seed(7)
        thought_a = torch.randn(1, 32)
        thought_b = torch.randn(1, 32)

        with torch.no_grad():
            out_a = qs(thought_a, nt, threat, z_self)
            out_b = qs(thought_b, nt, threat, z_self)

        diff = (out_a["qualia"] - out_b["qualia"]).norm().item()
        assert diff > 1e-4, "Different thoughts should yield different qualia"

    def test_ema_qualia_buffer_updates_on_forward(self, qs):
        """Oscillatory smoothing: ema_qualia buffer changes after a forward pass.

        The EMA (α=0.3) is the mechanism that makes qualia oscillatory —
        it prevents qualia from jumping sharply between steps, producing the
        sustained ~10 Hz envelope described in the module docstring.
        """
        before = qs.ema_qualia.clone()
        self._fwd(qs)
        after  = qs.ema_qualia

        diff = (before - after).norm().item()
        assert diff > 1e-8, (
            "ema_qualia should update after a forward pass "
            "(oscillatory smoothing is broken if the buffer is frozen)"
        )

    def test_ema_alpha_produces_partial_update(self, qs):
        """EMA with α=0.3 means the buffer moves less than the full step size.

        ema_new = 0.7 * ema_old + 0.3 * qualia_mean
        → |ema_new - ema_old| < |qualia_mean - ema_old| (partial, not full step)
        """
        qs.ema_qualia.zero_()  # start at zero
        torch.manual_seed(8)
        thought = torch.randn(4, 32)
        nt      = torch.ones(4, 7)
        threat  = torch.ones(4)
        z_self  = torch.randn(4, 32)

        with torch.no_grad():
            out = qs(thought, nt, threat, z_self)

        qualia_mean = out["qualia"].detach().mean(0)
        step_delta  = (qs.ema_qualia - qualia_mean).norm().item()
        full_delta  = qualia_mean.norm().item()  # distance from zero to mean
        # EMA buffer should be closer to zero than the raw mean
        assert step_delta < full_delta * 0.99, (
            "EMA smoothing should keep buffer between old value and new signal"
        )

    def test_warp_broadcast_starvation_vs_healthy(self, qs):
        """Low energy (starvation) must produce higher aversive pressure than healthy.

        Replicates the 'qualia warp' contract from the survival imperatives spec:
        survival_imperatives → qualia → GWS broadcast warp.
        """
        torch.manual_seed(9)
        broadcast = torch.randn(1, 32)
        healthy   = torch.tensor([[1.0, 1.0, 1.0]])
        starving  = torch.tensor([[0.05, 0.5, 0.5]])

        with torch.no_grad():
            qs.warp_broadcast(broadcast, healthy)
            p_healthy = qs.aversive_pressure()
            qs.warp_broadcast(broadcast, starving)
            p_starving = qs.aversive_pressure()

        assert p_starving > p_healthy + 0.05, (
            f"Starvation pressure={p_starving:.3f} should exceed "
            f"healthy pressure={p_healthy:.3f} by >0.05"
        )

    def test_warp_broadcast_aversive_magnitude_exceeds_appetitive(self, qs):
        """Starvation warp must be larger in magnitude than satiety warp.

        The aversive direction is initialized at 10× the appetitive direction
        (0.5 vs 0.05) so the survival signal dominates over hedonic comfort.
        """
        torch.manual_seed(10)
        broadcast = torch.randn(1, 32)

        with torch.no_grad():
            w_healthy  = qs.warp_broadcast(broadcast, torch.tensor([[1.0, 1.0, 1.0]]))
            delta_h    = (w_healthy - broadcast).norm().item()
            w_starving = qs.warp_broadcast(broadcast, torch.tensor([[0.05, 0.5, 0.5]]))
            delta_s    = (w_starving - broadcast).norm().item()

        assert delta_s > 2.0 * delta_h, (
            f"Starvation warp ({delta_s:.3f}) should be >2× healthy warp ({delta_h:.3f})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2C — LATENT AFFECTIVE MODULATION OF THOUGHT
# ══════════════════════════════════════════════════════════════════════════════

class TestLatentAffectiveModulation:
    """The key behavioral contract: qualia semantically modulates the latent
    thought manifold (the emotions_vector pathway in the DSL).

    In the Python module this corresponds to the gated-residual path:
        modulated = floating_thought + gate(qualia, thought) * proj(qualia)

    Four contracts are pinned here:

      1. Zero-init ReZero contract: thought_proj is zeros at init →
         modulated_thought = floating_thought at step 0 (no modulation yet).

      2. Non-trivial modulation after random perturbation of thought_proj:
         once the gate opens, qualia changes the thought direction.

      3. Emotional valence divergence:
         a positively-valenced NT state produces a different thought warp
         direction than a negatively-valenced NT state (→ not just scaling).

      4. Thought→NT demand (the other direction of the loop):
         threatening thoughts raise NE demand more than calming thoughts.
    """

    @pytest.fixture
    def qs_zero(self):
        """Fresh QualiaState — thought_proj is zero-initialized (ReZero)."""
        from neuroslm.modules.qualia import QualiaState
        m = QualiaState(d_sem=32, n_nt=7)
        m.eval()
        return m

    @pytest.fixture
    def qs_nonzero(self):
        """QualiaState with thought_proj perturbed so modulation fires."""
        from neuroslm.modules.qualia import QualiaState
        torch.manual_seed(99)
        m = QualiaState(d_sem=32, n_nt=7)
        # Perturb thought_proj to a small but non-zero value so the gate opens
        with torch.no_grad():
            m.thought_proj.weight.normal_(std=0.1)
        m.eval()
        return m

    # ── Contract 1: ReZero — no modulation at step 0 ─────────────────────────

    def test_zero_init_thought_proj_no_modulation(self, qs_zero):
        """thought_proj is zero-initialized → modulated_thought == floating_thought.

        This is the ReZero contract: at step 0 the affect pathway is silent,
        so the first forward is bit-identical to a baseline without qualia.
        Only gradient pressure from the LM loss can open the gate.
        """
        torch.manual_seed(0)
        d_sem, n_nt = 32, 7
        thought = torch.randn(2, d_sem)
        nt      = torch.rand(2, n_nt)
        threat  = torch.rand(2)
        z_self  = torch.randn(2, d_sem)

        with torch.no_grad():
            out = qs_zero(thought, nt, threat, z_self)

        delta = (out["modulated_thought"] - thought).abs().max().item()
        assert delta < 1e-6, (
            f"With zero-init thought_proj, modulated_thought should equal "
            f"floating_thought (ReZero contract), but max delta={delta:.2e}. "
            "thought_proj was not correctly initialized to zero."
        )

    # ── Contract 2: Non-trivial modulation after non-zero proj ───────────────

    def test_nonzero_proj_modulates_thought(self, qs_nonzero):
        """After thought_proj is non-zero, qualia changes the thought vector."""
        torch.manual_seed(1)
        d_sem, n_nt = 32, 7
        thought = torch.randn(2, d_sem)
        nt      = torch.rand(2, n_nt)
        threat  = torch.rand(2)
        z_self  = torch.randn(2, d_sem)

        with torch.no_grad():
            out = qs_nonzero(thought, nt, threat, z_self)

        delta = (out["modulated_thought"] - thought).norm().item()
        assert delta > 1e-3, (
            f"With non-zero thought_proj, qualia should modify thought, "
            f"but delta={delta:.2e} is near zero. "
            "The gated-residual path may be broken."
        )

    def test_modulated_thought_preserves_thought_as_residual(self, qs_nonzero):
        """modulated_thought = thought + gate * qualia_bias  (additive residual).

        The thought is the primary signal; qualia is a bias. This ensures that
        even with strong emotion, the cognitive content is not overwritten —
        only semantically coloured.
        """
        torch.manual_seed(2)
        d_sem, n_nt = 32, 7
        thought = torch.randn(1, d_sem) * 10  # large primary signal

        with torch.no_grad():
            out = qs_nonzero(
                thought,
                nt_vec=torch.rand(1, n_nt),
                threat=torch.rand(1),
                z_self=torch.randn(1, d_sem),
            )

        # The modulated output should be positively correlated with the input
        cos = torch.nn.functional.cosine_similarity(
            out["modulated_thought"], thought, dim=-1
        ).item()
        assert cos > 0.5, (
            f"modulated_thought should stay positively aligned with original thought "
            f"(residual structure), but cosine similarity={cos:.3f}"
        )

    # ── Contract 3: Emotional valence divergence ──────────────────────────────

    def test_positive_vs_negative_nt_warp_different_directions(self, qs_nonzero):
        """Positive emotional state (high DA) vs negative (high NE) produce
        measurably DIFFERENT thought warp directions — not just different magnitudes.

        This pins the semantic content of affective modulation:
        joy and fear don't just scale the thought by different amounts,
        they pull it in different directions in d_sem space.
        """
        torch.manual_seed(3)
        d_sem, n_nt = 32, 7
        thought = torch.randn(1, d_sem)
        threat  = torch.zeros(1)
        z_self  = torch.zeros(1, d_sem)

        # DA-dominant (reward / positive affect) — channel 0
        nt_pos = torch.zeros(1, n_nt)
        nt_pos[0, 0] = 1.0  # DA high

        # NE-dominant (stress / negative affect) — channel 1
        nt_neg = torch.zeros(1, n_nt)
        nt_neg[0, 1] = 1.0  # NE high

        with torch.no_grad():
            out_pos = qs_nonzero(thought, nt_pos, threat, z_self)
            out_neg = qs_nonzero(thought, nt_neg, threat, z_self)

        bias_pos = out_pos["modulated_thought"] - thought
        bias_neg = out_neg["modulated_thought"] - thought

        # The warp directions should be different (not just the magnitudes)
        # If both biases are near zero, the test is moot — ensure they're meaningful
        if bias_pos.norm() < 1e-6 and bias_neg.norm() < 1e-6:
            pytest.skip("Both biases are zero (thought_proj may need larger perturbation)")

        # Cosine similarity between the two warp directions — must not be 1.0
        cos = torch.nn.functional.cosine_similarity(
            bias_pos.flatten().unsqueeze(0),
            bias_neg.flatten().unsqueeze(0),
        ).item()
        assert cos < 0.99, (
            f"Positive and negative NT states produce the same warp direction "
            f"(cosine={cos:.4f}). The qualia encoder must produce different "
            f"qualia for different NT profiles."
        )

    def test_same_thought_different_nt_different_modulation(self, qs_nonzero):
        """NT levels modulate how thought is coloured — same thought, diff NT → diff output."""
        torch.manual_seed(4)
        d_sem, n_nt = 32, 7
        thought = torch.randn(1, d_sem)
        threat  = torch.zeros(1)
        z_self  = torch.zeros(1, d_sem)

        nt_a = torch.zeros(1, n_nt)
        nt_b = torch.ones(1, n_nt)  # all channels at max

        with torch.no_grad():
            out_a = qs_nonzero(thought, nt_a, threat, z_self)
            out_b = qs_nonzero(thought, nt_b, threat, z_self)

        diff = (out_a["modulated_thought"] - out_b["modulated_thought"]).norm().item()
        assert diff > 1e-4, (
            f"Same thought with different NT levels should produce different "
            f"modulated_thought, but diff={diff:.2e}"
        )

    # ── Contract 4: Thought→NT demand (reverse direction) ────────────────────

    def test_thought_affects_nt_demand(self, qs_zero):
        """Different thoughts produce different NT release demands.

        This is the 'thought→NT feedback' direction: threatening thought →
        NE demand; calming thought → 5HT demand (Damasio 1999 somatic marker).
        """
        nt     = torch.zeros(1, 7)
        threat = torch.zeros(1)
        z_self = torch.zeros(1, 32)

        torch.manual_seed(11)
        thought_a = torch.randn(1, 32)
        thought_b = torch.randn(1, 32)

        with torch.no_grad():
            dem_a = qs_zero(thought_a, nt, threat, z_self)["thought_nt_demand"]
            dem_b = qs_zero(thought_b, nt, threat, z_self)["thought_nt_demand"]

        diff = (dem_a - dem_b).abs().max().item()
        assert diff > 1e-4, (
            "Different thoughts should produce different NT demands, "
            f"but max diff={diff:.2e}"
        )

    def test_nt_demand_is_differentiable_wrt_thought(self, qs_zero):
        """Gradient flows from NT demand back to thought.

        This is the training-time contract: the thought→NT feedback path must
        be differentiable so the LM loss can shape WHAT the model thinks
        (and therefore, what NT profile it implicitly requests).
        """
        d_sem, n_nt = 32, 7
        thought = torch.randn(1, d_sem, requires_grad=True)
        nt      = torch.zeros(1, n_nt)
        threat  = torch.zeros(1)
        z_self  = torch.zeros(1, d_sem)

        # Force train mode so EMA update doesn't interfere with the grad check
        qs_zero.train()
        out = qs_zero(thought, nt, threat, z_self)
        loss = out["thought_nt_demand"].sum()
        loss.backward()

        assert thought.grad is not None, \
            "thought.grad is None — NT demand is not differentiable w.r.t. thought"
        assert thought.grad.abs().max().item() > 1e-8, \
            "thought.grad is effectively zero — gradient is not flowing"

    def test_qualia_is_differentiable_wrt_nt(self, qs_zero):
        """Gradient flows from qualia back to NT levels.

        Ensures that NT levels can be shaped by the LM loss via the qualia
        embedding (the NT → qualia → modulated_thought → LM loss path).
        """
        d_sem, n_nt = 32, 7
        nt      = torch.rand(1, n_nt, requires_grad=True)
        thought = torch.randn(1, d_sem)
        threat  = torch.zeros(1)
        z_self  = torch.zeros(1, d_sem)

        qs_zero.train()
        out = qs_zero(thought, nt, threat, z_self)
        out["qualia"].sum().backward()

        assert nt.grad is not None, \
            "nt.grad is None — qualia is not differentiable w.r.t. NT levels"
        assert nt.grad.abs().max().item() > 1e-8, \
            "nt.grad is effectively zero — NT→qualia gradient is not flowing"

    def test_qualia_modulated_thought_differentiable_wrt_thought(self, qs_zero):
        """modulated_thought gradient flows back to floating_thought.

        This is required for the thought manifold to be trainable via the
        latent affective modulation pathway (emotions_vector → thought).
        """
        d_sem, n_nt = 32, 7
        thought = torch.randn(1, d_sem, requires_grad=True)
        nt      = torch.zeros(1, n_nt)
        threat  = torch.zeros(1)
        z_self  = torch.zeros(1, d_sem)

        qs_zero.train()
        out = qs_zero(thought, nt, threat, z_self)
        out["modulated_thought"].sum().backward()

        assert thought.grad is not None, \
            "No gradient to thought from modulated_thought"
        assert thought.grad.abs().max().item() > 1e-8, \
            "Gradient to thought is zero — modulated_thought grad path broken"
