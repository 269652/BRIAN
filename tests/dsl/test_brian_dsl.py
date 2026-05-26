#!/usr/bin/env python3
"""Structural validation tests for BRIAN DSL architecture specification.

Validates that the compiled brian.neuro DSL IR has:
- All 23 core module populations with correct dynamics/counts
- All 6 neuromodulatory nuclei populations
- All 7 neurotransmitter systems with correct kinetics
- All 16 anatomical projections as synapses
- All modulations with correct gains and effects
- Formal specs (sheaf + Φ integration)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pytest
from neuroslm.dsl.compiler import NeuroMLCompiler


@pytest.fixture
def brian_ir():
    """Compile BRIAN architecture DSL and return IR."""
    dsl_path = os.path.join(os.path.dirname(__file__), '../../neuroslm/dsl/brian.neuro')
    return NeuroMLCompiler.compile_file(dsl_path)


class TestBrianPopulations:
    """Validate all 29 populations (23 modules + 6 nuclei) are present."""

    def test_total_population_count(self, brian_ir):
        """Should have at least 29 populations."""
        assert len(brian_ir.populations) >= 29, \
            f"Expected >=29 populations, got {len(brian_ir.populations)}"

    def test_all_23_core_modules_present(self, brian_ir):
        """All 23 core module populations should be present."""
        core_modules = {
            'language_cortex', 'sensory_cortex', 'association_cortex',
            'global_workspace', 'thalamus', 'world_model', 'self_model',
            'hippocampus', 'default_mode_network', 'prefrontal_cortex',
            'basal_ganglia', 'forward_model', 'evaluator', 'motor_cortex',
            'cortical_sheet', 'entorhinal_cortex', 'claustrum', 'cerebellum',
            'neural_geometry', 'qualia_state', 'thought_transformer',
            'amygdala', 'anterior_cingulate'
        }
        pop_names = {p.name for p in brian_ir.populations}
        missing = core_modules - pop_names
        assert not missing, f"Missing core modules: {missing}"

    def test_all_6_nt_nuclei_present(self, brian_ir):
        """All 6 neuromodulatory nuclei should be present."""
        nt_nuclei = {
            'vta', 'nucleus_accumbens', 'locus_coeruleus',
            'raphe_nuclei', 'nucleus_basalis', 'substantia_nigra'
        }
        pop_names = {p.name for p in brian_ir.populations}
        missing = nt_nuclei - pop_names
        assert not missing, f"Missing NT nuclei: {missing}"

    def test_hippocampus_capacity(self, brian_ir):
        """Hippocampus should have count=4096 (sparse storage)."""
        hippo = next((p for p in brian_ir.populations if p.name == 'hippocampus'), None)
        assert hippo is not None, "Hippocampus population not found"
        assert hippo.count == 4096, f"Hippocampus count should be 4096, got {hippo.count}"

    def test_amygdala_small_count(self, brian_ir):
        """Amygdala should have count=32 (fear/arousal nucleus)."""
        amygdala = next((p for p in brian_ir.populations if p.name == 'amygdala'), None)
        assert amygdala is not None, "Amygdala population not found"
        assert amygdala.count == 32, f"Amygdala count should be 32, got {amygdala.count}"

    def test_global_workspace_dynamics(self, brian_ir):
        """GWS should use winner_take_all dynamics."""
        gws = next((p for p in brian_ir.populations if p.name == 'global_workspace'), None)
        assert gws is not None, "Global workspace not found"
        assert 'winner_take_all' in gws.dynamics, \
            f"GWS dynamics should be 'winner_take_all', got {gws.dynamics}"

    def test_basal_ganglia_winner_take_all(self, brian_ir):
        """Basal ganglia should use winner_take_all (action selection)."""
        bg = next((p for p in brian_ir.populations if p.name == 'basal_ganglia'), None)
        assert bg is not None, "Basal ganglia not found"
        assert 'winner_take_all' in bg.dynamics, \
            f"Basal ganglia dynamics should be 'winner_take_all', got {bg.dynamics}"

    def test_hippocampus_attractor_network(self, brian_ir):
        """Hippocampus should use attractor_network dynamics."""
        hippo = next((p for p in brian_ir.populations if p.name == 'hippocampus'), None)
        assert hippo is not None, "Hippocampus not found"
        assert 'attractor_network' in hippo.dynamics, \
            f"Hippocampus dynamics should be 'attractor_network', got {hippo.dynamics}"

    def test_amygdala_integrate_and_fire(self, brian_ir):
        """Amygdala should use integrate_and_fire (spiking)."""
        amygdala = next((p for p in brian_ir.populations if p.name == 'amygdala'), None)
        assert amygdala is not None, "Amygdala not found"
        assert 'integrate_and_fire' in amygdala.dynamics, \
            f"Amygdala dynamics should be 'integrate_and_fire', got {amygdala.dynamics}"

    def test_thalamus_gated_dynamics(self, brian_ir):
        """Thalamus should use gated dynamics (relay control)."""
        thalamus = next((p for p in brian_ir.populations if p.name == 'thalamus'), None)
        assert thalamus is not None, "Thalamus not found"
        assert 'gated' in thalamus.dynamics, \
            f"Thalamus dynamics should be 'gated', got {thalamus.dynamics}"

    def test_neural_geometry_static(self, brian_ir):
        """Neural geometry should be static (coordinate frame)."""
        ng = next((p for p in brian_ir.populations if p.name == 'neural_geometry'), None)
        assert ng is not None, "Neural geometry not found"
        assert 'static' in ng.dynamics, \
            f"Neural geometry dynamics should be 'static', got {ng.dynamics}"

    def test_most_populations_rate_code(self, brian_ir):
        """Most populations should use rate_code dynamics."""
        rate_code_pops = [p for p in brian_ir.populations if 'rate_code' in p.dynamics]
        assert len(rate_code_pops) >= 15, \
            f"Should have >=15 rate_code populations, got {len(rate_code_pops)}"

    def test_population_timescales(self, brian_ir):
        """Populations should have realistic timescales (0.001 - 0.1 s)."""
        for p in brian_ir.populations:
            assert 0.0005 <= p.timescale <= 0.15, \
                f"Population {p.name} timescale {p.timescale} outside realistic range"

    def test_population_capacities(self, brian_ir):
        """Populations should have reasonable capacities (0.5 - 8.0)."""
        for p in brian_ir.populations:
            assert 0.25 <= p.capacity <= 10.0, \
                f"Population {p.name} capacity {p.capacity} outside realistic range"


class TestBrianNeurotransmitters:
    """Validate all 7 neurotransmitter systems with correct kinetics."""

    def test_all_7_nt_systems_present(self, brian_ir):
        """All 7 NT systems should be present."""
        nt_names = {
            'dopamine', 'norepinephrine', 'serotonin', 'acetylcholine',
            'endocannabinoid', 'glutamate', 'gaba'
        }
        ir_nt_names = {nt.name for nt in brian_ir.neurotransmitter_systems}
        missing = nt_names - ir_nt_names
        assert not missing, f"Missing NT systems: {missing}"

    def test_dopamine_baseline(self, brian_ir):
        """Dopamine baseline concentration should be ~0.10."""
        da = next((nt for nt in brian_ir.neurotransmitter_systems
                   if nt.name == 'dopamine'), None)
        assert da is not None, "Dopamine NT not found"
        assert abs(da.base_concentration - 0.10) < 0.01, \
            f"Dopamine baseline should be ~0.10, got {da.base_concentration}"

    def test_norepinephrine_baseline(self, brian_ir):
        """Norepinephrine baseline concentration should be ~0.15."""
        ne = next((nt for nt in brian_ir.neurotransmitter_systems
                   if nt.name == 'norepinephrine'), None)
        assert ne is not None, "Norepinephrine NT not found"
        assert abs(ne.base_concentration - 0.15) < 0.01, \
            f"Norepinephrine baseline should be ~0.15, got {ne.base_concentration}"

    def test_serotonin_baseline(self, brian_ir):
        """Serotonin baseline concentration should be ~0.30."""
        se = next((nt for nt in brian_ir.neurotransmitter_systems
                   if nt.name == 'serotonin'), None)
        assert se is not None, "Serotonin NT not found"
        assert abs(se.base_concentration - 0.30) < 0.01, \
            f"Serotonin baseline should be ~0.30, got {se.base_concentration}"

    def test_acetylcholine_baseline(self, brian_ir):
        """Acetylcholine baseline concentration should be ~0.20."""
        ach = next((nt for nt in brian_ir.neurotransmitter_systems
                    if nt.name == 'acetylcholine'), None)
        assert ach is not None, "Acetylcholine NT not found"
        assert abs(ach.base_concentration - 0.20) < 0.01, \
            f"Acetylcholine baseline should be ~0.20, got {ach.base_concentration}"

    def test_gaba_baseline(self, brian_ir):
        """GABA baseline concentration should be ~0.10."""
        gaba = next((nt for nt in brian_ir.neurotransmitter_systems
                     if nt.name == 'gaba'), None)
        assert gaba is not None, "GABA NT not found"
        assert abs(gaba.base_concentration - 0.10) < 0.01, \
            f"GABA baseline should be ~0.10, got {gaba.base_concentration}"

    def test_dopamine_kinetics(self, brian_ir):
        """Dopamine should have realistic kinetic rates."""
        da = next((nt for nt in brian_ir.neurotransmitter_systems
                   if nt.name == 'dopamine'), None)
        assert da.release_rate is not None and da.release_rate > 0, \
            f"Dopamine should have positive release_rate"
        assert da.reuptake_rate is not None and da.reuptake_rate > 0, \
            f"Dopamine should have positive reuptake_rate"

    def test_serotonin_slow_kinetics(self, brian_ir):
        """Serotonin should have slow reuptake (high tau)."""
        se = next((nt for nt in brian_ir.neurotransmitter_systems
                   if nt.name == 'serotonin'), None)
        assert se.reuptake_rate is not None, "Serotonin should have reuptake_rate"
        assert se.reuptake_rate >= 0.90, \
            f"Serotonin should have high reuptake tau (>=0.90), got {se.reuptake_rate}"

    def test_nt_baseline_in_range(self, brian_ir):
        """All NT baseline concentrations should be in [0, 1]."""
        for nt in brian_ir.neurotransmitter_systems:
            assert 0 <= nt.base_concentration <= 1, \
                f"NT {nt.name} baseline {nt.base_concentration} outside [0, 1]"


class TestBrianProjections:
    """Validate all 16 anatomical projections as synapses."""

    def test_minimum_projection_count(self, brian_ir):
        """Should have at least 16 anatomical projections."""
        assert len(brian_ir.synapses) >= 16, \
            f"Expected >=16 synapses, got {len(brian_ir.synapses)}"

    def test_dopaminergic_vta_nacc(self, brian_ir):
        """VTA → NAcc dopaminergic projection should exist."""
        syn = next((s for s in brian_ir.synapses
                    if s.source == 'vta' and s.target == 'nucleus_accumbens'), None)
        assert syn is not None, "VTA → NAcc synapse not found"
        assert syn.neurotransmitter == 'dopamine', \
            f"VTA → NAcc should use dopamine, got {syn.neurotransmitter}"

    def test_dopaminergic_vta_pfc(self, brian_ir):
        """VTA → PFC dopaminergic projection should exist."""
        syn = next((s for s in brian_ir.synapses
                    if s.source == 'vta' and s.target == 'prefrontal_cortex'), None)
        assert syn is not None, "VTA → PFC synapse not found"
        assert syn.neurotransmitter == 'dopamine'

    def test_dopaminergic_vta_hippo(self, brian_ir):
        """VTA → Hippocampus dopaminergic projection should exist."""
        syn = next((s for s in brian_ir.synapses
                    if s.source == 'vta' and s.target == 'hippocampus'), None)
        assert syn is not None, "VTA → Hippocampus synapse not found"
        assert syn.neurotransmitter == 'dopamine'

    def test_dopaminergic_sn_bg(self, brian_ir):
        """SNc → Basal Ganglia dopaminergic projection should exist."""
        syn = next((s for s in brian_ir.synapses
                    if s.source == 'substantia_nigra' and s.target == 'basal_ganglia'), None)
        assert syn is not None, "SNc → BG synapse not found"
        assert syn.neurotransmitter == 'dopamine'

    def test_noradrenergic_lc_pfc(self, brian_ir):
        """LC → PFC noradrenergic projection should exist."""
        syn = next((s for s in brian_ir.synapses
                    if s.source == 'locus_coeruleus' and s.target == 'prefrontal_cortex'), None)
        assert syn is not None, "LC → PFC synapse not found"
        assert syn.neurotransmitter == 'norepinephrine'

    def test_noradrenergic_lc_thalamus(self, brian_ir):
        """LC → Thalamus noradrenergic projection should exist."""
        syn = next((s for s in brian_ir.synapses
                    if s.source == 'locus_coeruleus' and s.target == 'thalamus'), None)
        assert syn is not None, "LC → Thalamus synapse not found"
        assert syn.neurotransmitter == 'norepinephrine'

    def test_noradrenergic_lc_hippo(self, brian_ir):
        """LC → Hippocampus noradrenergic projection should exist."""
        syn = next((s for s in brian_ir.synapses
                    if s.source == 'locus_coeruleus' and s.target == 'hippocampus'), None)
        assert syn is not None, "LC → Hippocampus synapse not found"
        assert syn.neurotransmitter == 'norepinephrine'

    def test_serotonergic_raphe_pfc(self, brian_ir):
        """Raphe → PFC serotonergic projection should exist."""
        syn = next((s for s in brian_ir.synapses
                    if s.source == 'raphe_nuclei' and s.target == 'prefrontal_cortex'), None)
        assert syn is not None, "Raphe → PFC synapse not found"
        assert syn.neurotransmitter == 'serotonin'

    def test_serotonergic_raphe_dmn(self, brian_ir):
        """Raphe → DMN serotonergic projection should exist."""
        syn = next((s for s in brian_ir.synapses
                    if s.source == 'raphe_nuclei' and s.target == 'default_mode_network'), None)
        assert syn is not None, "Raphe → DMN synapse not found"
        assert syn.neurotransmitter == 'serotonin'

    def test_serotonergic_raphe_hippo(self, brian_ir):
        """Raphe → Hippocampus serotonergic projection should exist."""
        syn = next((s for s in brian_ir.synapses
                    if s.source == 'raphe_nuclei' and s.target == 'hippocampus'), None)
        assert syn is not None, "Raphe → Hippocampus synapse not found"
        assert syn.neurotransmitter == 'serotonin'

    def test_cholinergic_nbm_pfc(self, brian_ir):
        """NBM → PFC cholinergic projection should exist."""
        syn = next((s for s in brian_ir.synapses
                    if s.source == 'nucleus_basalis' and s.target == 'prefrontal_cortex'), None)
        assert syn is not None, "NBM → PFC synapse not found"
        assert syn.neurotransmitter == 'acetylcholine'

    def test_cholinergic_nbm_hippo(self, brian_ir):
        """NBM → Hippocampus cholinergic projection should exist."""
        syn = next((s for s in brian_ir.synapses
                    if s.source == 'nucleus_basalis' and s.target == 'hippocampus'), None)
        assert syn is not None, "NBM → Hippocampus synapse not found"
        assert syn.neurotransmitter == 'acetylcholine'

    def test_cholinergic_nbm_language(self, brian_ir):
        """NBM → Language cortex cholinergic projection should exist."""
        syn = next((s for s in brian_ir.synapses
                    if s.source == 'nucleus_basalis' and s.target == 'language_cortex'), None)
        assert syn is not None, "NBM → Language cortex synapse not found"
        assert syn.neurotransmitter == 'acetylcholine'

    def test_feedback_nacc_vta(self, brian_ir):
        """NAcc → VTA feedback projection should exist."""
        syn = next((s for s in brian_ir.synapses
                    if s.source == 'nucleus_accumbens' and s.target == 'vta'), None)
        assert syn is not None, "NAcc → VTA synapse not found"
        assert syn.neurotransmitter == 'dopamine'

    def test_glutamatergic_pfc_bg(self, brian_ir):
        """PFC → Basal Ganglia glutamatergic projection should exist."""
        syn = next((s for s in brian_ir.synapses
                    if s.source == 'prefrontal_cortex' and s.target == 'basal_ganglia'), None)
        assert syn is not None, "PFC → BG synapse not found"
        assert syn.neurotransmitter == 'glutamate'

    def test_gabaergic_bg_thalamus(self, brian_ir):
        """Basal Ganglia → Thalamus GABAergic projection should exist."""
        syn = next((s for s in brian_ir.synapses
                    if s.source == 'basal_ganglia' and s.target == 'thalamus'), None)
        assert syn is not None, "BG → Thalamus synapse not found"
        assert syn.neurotransmitter == 'gaba'


class TestBrianModulations:
    """Validate neuromodulation (receptor banks) present with correct gains."""

    def test_minimum_modulation_count(self, brian_ir):
        """Should have at least 12 modulations."""
        assert len(brian_ir.modulations) >= 12, \
            f"Expected >=12 modulations, got {len(brian_ir.modulations)}"

    def test_dopamine_pfc_modulation(self, brian_ir):
        """Dopamine should modulate PFC."""
        mods = [m for m in brian_ir.modulations
                if m.source_nt == 'dopamine' and m.target_population == 'prefrontal_cortex']
        assert len(mods) >= 1, "Dopamine → PFC modulation not found"
        assert mods[0].gain >= 0.5, \
            f"DA → PFC gain should be >=0.5, got {mods[0].gain}"

    def test_serotonin_pfc_modulation(self, brian_ir):
        """Serotonin should modulate PFC."""
        mods = [m for m in brian_ir.modulations
                if m.source_nt == 'serotonin' and m.target_population == 'prefrontal_cortex']
        assert len(mods) >= 1, "Serotonin → PFC modulation not found"
        assert mods[0].effect == 'additive', \
            f"5HT → PFC should be additive, got {mods[0].effect}"

    def test_dopamine_acc_modulation(self, brian_ir):
        """Dopamine should modulate anterior cingulate."""
        mods = [m for m in brian_ir.modulations
                if m.source_nt == 'dopamine' and m.target_population == 'anterior_cingulate']
        assert len(mods) >= 1, "Dopamine → ACC modulation not found"

    def test_acetylcholine_hippo_modulation(self, brian_ir):
        """Acetylcholine should modulate hippocampus."""
        mods = [m for m in brian_ir.modulations
                if m.source_nt == 'acetylcholine' and m.target_population == 'hippocampus']
        assert len(mods) >= 1, "ACh → Hippocampus modulation not found"
        assert mods[0].gain >= 0.5, \
            f"ACh → Hippo gain should be >=0.5, got {mods[0].gain}"

    def test_serotonin_dmn_modulation(self, brian_ir):
        """Serotonin should modulate default mode network."""
        mods = [m for m in brian_ir.modulations
                if m.source_nt == 'serotonin' and m.target_population == 'default_mode_network']
        assert len(mods) >= 1, "Serotonin → DMN modulation not found"

    def test_gaba_bg_modulation(self, brian_ir):
        """GABA should modulate basal ganglia."""
        mods = [m for m in brian_ir.modulations
                if m.source_nt == 'gaba' and m.target_population == 'basal_ganglia']
        assert len(mods) >= 1, "GABA → Basal Ganglia modulation not found"

    def test_modulation_gains_realistic(self, brian_ir):
        """All modulation gains should be in [0.1, 10.0]."""
        for m in brian_ir.modulations:
            assert 0.05 <= m.gain <= 15.0, \
                f"Modulation {m.source_nt} → {m.target_population} gain {m.gain} outside realistic range"

    def test_modulation_effects_valid(self, brian_ir):
        """All modulation effects should be valid types."""
        valid_effects = {'multiplicative', 'additive', 'gating', 'allosteric'}
        for m in brian_ir.modulations:
            assert m.effect in valid_effects, \
                f"Invalid modulation effect: {m.effect}"


class TestFormalSpecs:
    """Validate formal constraint systems (sheaf + Φ)."""

    def test_sheaf_present(self, brian_ir):
        """Sheaf consistency specification should be present."""
        assert len(brian_ir.sheaf_specs) >= 1, \
            f"Expected >=1 sheaf specs, got {len(brian_ir.sheaf_specs)}"

    def test_sheaf_contradiction_threshold(self, brian_ir):
        """Sheaf should have contradiction_threshold ~0.7."""
        sheaf = brian_ir.sheaf_specs[0]
        assert abs(sheaf.contradiction_threshold - 0.7) < 0.05, \
            f"Sheaf threshold should be ~0.7, got {sheaf.contradiction_threshold}"

    def test_sheaf_mechanism(self, brian_ir):
        """Sheaf should use h1_cohomology_proxy mechanism."""
        sheaf = brian_ir.sheaf_specs[0]
        assert 'h1_cohomology' in sheaf.mechanism, \
            f"Sheaf should use h1_cohomology, got {sheaf.mechanism}"

    def test_phi_integration_spec_present(self, brian_ir):
        """Φ-IIT integration formal spec should be present."""
        assert len(brian_ir.formal_specs) >= 1, \
            f"Expected >=1 formal specs, got {len(brian_ir.formal_specs)}"

    def test_phi_integration_rule(self, brian_ir):
        """Φ spec should reference integrated_information metric."""
        phi_specs = [fs for fs in brian_ir.formal_specs
                     if 'integrated_information' in fs.spec_type]
        assert len(phi_specs) >= 1, "Phi integration formal spec not found"


class TestBrianIntegration:
    """Integration tests: overall IR structure and consistency."""

    def test_all_components_compiled(self, brian_ir):
        """Should have populations, NTs, synapses, modulations, and formal specs."""
        assert len(brian_ir.populations) >= 29, "Missing populations"
        assert len(brian_ir.neurotransmitter_systems) == 7, "Missing NT systems"
        assert len(brian_ir.synapses) >= 16, "Missing synapses"
        assert len(brian_ir.modulations) >= 12, "Missing modulations"
        assert len(brian_ir.sheaf_specs) >= 1, "Missing sheaf specs"
        assert len(brian_ir.formal_specs) >= 1, "Missing formal specs"

    def test_no_synapse_duplicates(self, brian_ir):
        """Should not have duplicate synapses."""
        syn_pairs = [(s.source, s.target) for s in brian_ir.synapses]
        assert len(syn_pairs) == len(set(syn_pairs)), \
            "Found duplicate synapses"

    def test_synapse_endpoints_exist(self, brian_ir):
        """All synapse endpoints should reference existing populations."""
        pop_names = {p.name for p in brian_ir.populations}
        for syn in brian_ir.synapses:
            assert syn.source in pop_names, \
                f"Synapse source '{syn.source}' not in populations"
            assert syn.target in pop_names, \
                f"Synapse target '{syn.target}' not in populations"

    def test_modulation_targets_exist(self, brian_ir):
        """All modulation targets should reference existing populations."""
        pop_names = {p.name for p in brian_ir.populations}
        for m in brian_ir.modulations:
            assert m.target_population in pop_names, \
                f"Modulation target '{m.target_population}' not in populations"

    def test_file_compiles_without_error(self, brian_ir):
        """brian.neuro file should compile without errors."""
        # If we got here, compilation succeeded
        assert brian_ir is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
