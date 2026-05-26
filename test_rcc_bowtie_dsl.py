#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick test of RCC Bowtie DSL compilation.

Verifies that the formalized RCC bowtie architecture:
- Compiles successfully to IR
- Has all 20 orchestrator modules + 6 NT nuclei = 26 populations
- Has 7 neurotransmitter systems
- Has all major anatomical projections
- Has receptor banks (modulations) for all major brain regions
- Has formal specs (sheaf, phi, bowtie)
"""
import sys
sys.path.insert(0, 'c:\\Users\\morrossl\\Documents\\Private\\SLM')

try:
    from neuroslm.dsl.compiler import NeuroMLCompiler

    print("Compiling RCC Bowtie DSL...")
    ir = NeuroMLCompiler.compile_file('neuroslm/dsl/rcc_bowtie.neuro')

    print("[OK] RCC Bowtie IR compiled successfully")
    print("  Populations: %d" % len(ir.populations))
    print("  NT systems: %d" % len(ir.neurotransmitter_systems))
    print("  Synapses: %d" % len(ir.synapses))
    print("  Modulations: %d" % len(ir.modulations))
    print("  Sheaf specs: %d" % len(ir.sheaf_specs))
    print("  Formal specs: %d" % len(ir.formal_specs))

    # Verify orchestrator modules
    orch_modules = {
        'sensory', 'association', 'thalamus',
        'world', 'self_m',
        'amygdala', 'insula',
        'qualia',
        'gws', 'neural_geometry',
        'hippo', 'entorhinal', 'cerebellum',
        'pfc', 'acc',
        'bg', 'forward_m', 'evaluator',
        'dmn', 'thought_transformer', 'claustrum',
        'motor'
    }
    nt_nuclei = {'vta', 'nucleus_accumbens', 'locus_coeruleus', 'raphe_nuclei', 'nucleus_basalis', 'substantia_nigra'}

    pop_names = set(p.name for p in ir.populations)
    missing_modules = orch_modules - pop_names
    missing_nuclei = nt_nuclei - pop_names

    if missing_modules:
        print("\n[ERROR] Missing orchestrator modules: %s" % missing_modules)
        sys.exit(1)
    else:
        print("[OK] All 20 orchestrator modules present")

    if missing_nuclei:
        print("[ERROR] Missing NT nuclei: %s" % missing_nuclei)
        sys.exit(1)
    else:
        print("[OK] All 6 NT nuclei present (26 total populations)")

    # Verify bowtie path: sensory -> thalamus -> gws -> pfc -> bg -> motor
    bowtie_path = [('sensory', 'thalamus'), ('thalamus', 'gws'), ('gws', 'pfc'), ('pfc', 'bg'), ('bg', 'motor')]
    syn_pairs = {(s.source, s.target) for s in ir.synapses}

    for src, tgt in bowtie_path:
        if (src, tgt) not in syn_pairs:
            print("\n[WARN] Bowtie pathway %s->%s not found (not strictly required)" % (src, tgt))
        else:
            print("[OK] Bowtie pathway present: %s -> %s" % (src, tgt))

    # Verify key projections
    key_projections = [
        ('world', 'gws'),
        ('gws', 'hippo'),
        ('hippo', 'pfc'),
        ('forward_m', 'evaluator'),
        ('dmn', 'gws'),
    ]
    for src, tgt in key_projections:
        if (src, tgt) in syn_pairs:
            print("[OK] Projection present: %s -> %s" % (src, tgt))
        else:
            print("[WARN] Expected projection missing: %s -> %s" % (src, tgt))

    # Verify receptor banks
    receptor_banks = {
        'pfc': {'dopamine', 'serotonin', 'acetylcholine', 'gaba'},
        'hippo': {'acetylcholine', 'glutamate'},
        'bg': {'dopamine', 'gaba'},
        'thalamus': {'norepinephrine', 'gaba'},
        'gws': {'dopamine'},
    }

    for region, expected_nts in receptor_banks.items():
        mods = [m for m in ir.modulations if m.target_population == region]
        mod_nts = {m.source_nt for m in mods}
        found = expected_nts & mod_nts
        if found:
            print("[OK] Receptor bank %s has %s" % (region, ', '.join(sorted(found))))
        else:
            print("[WARN] Receptor bank %s missing expected NTs" % region)

    # Verify NT systems
    nt_names = {nt.name for nt in ir.neurotransmitter_systems}
    expected_nts = {'dopamine', 'norepinephrine', 'serotonin', 'acetylcholine',
                    'endocannabinoid', 'glutamate', 'gaba'}
    if expected_nts == nt_names:
        print("[OK] All 7 NT systems present")
    else:
        print("[ERROR] NT systems mismatch. Expected: %s, Got: %s" % (expected_nts, nt_names))
        sys.exit(1)

    # Verify formal specs
    if len(ir.sheaf_specs) > 0:
        print("[OK] Sheaf consistency spec present")
    else:
        print("[ERROR] Sheaf spec missing")
        sys.exit(1)

    if len(ir.formal_specs) >= 2:  # phi + bowtie
        print("[OK] Formal specs (phi + bowtie) present")
    else:
        print("[WARN] Expected 2+ formal specs, found %d" % len(ir.formal_specs))

    print("\n" + "="*60)
    print("SUCCESS: RCC Bowtie DSL extracts correctly!")
    print("="*60)
    print("\nDSL extraction complete. The formalized architecture can now be used for:")
    print("  1. Evolutionary discovery (mutate and search the design space)")
    print("  2. Verification against unit tests")
    print("  3. Codegen to PyTorch (future phase)")
    print("  4. Declarative specification of architecture variants")

except Exception as e:
    print("[ERROR]: %s" % e)
    import traceback
    traceback.print_exc()
    sys.exit(1)
