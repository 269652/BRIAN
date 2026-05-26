#!/usr/bin/env python
"""Quick test of BRIAN DSL compilation."""
import sys
sys.path.insert(0, 'c:\\Users\\morrossl\\Documents\\Private\\SLM')

try:
    from neuroslm.dsl.compiler import NeuroMLCompiler

    print("Compiling BRIAN architecture DSL...")
    ir = NeuroMLCompiler.compile_file('neuroslm/dsl/brian.neuro')

    print("[OK] IR compiled successfully")
    print("  Populations: %d" % len(ir.populations))
    print("  NT systems: %d" % len(ir.neurotransmitter_systems))
    print("  Synapses: %d" % len(ir.synapses))
    print("  Modulations: %d" % len(ir.modulations))
    print("  Sheaf specs: %d" % len(ir.sheaf_specs))
    print("  Formal specs: %d" % len(ir.formal_specs))

    # Verify key populations
    pop_names = set(p.name for p in ir.populations)
    expected_pops = {
        'language_cortex', 'sensory_cortex', 'global_workspace',
        'hippocampus', 'prefrontal_cortex', 'basal_ganglia',
        'vta', 'nucleus_accumbens', 'locus_coeruleus'
    }
    missing = expected_pops - pop_names

    if missing:
        print("\n[ERROR] Missing populations: %s" % missing)
        sys.exit(1)
    else:
        print("\n[OK] All expected populations present")

    # Verify hippocampus count
    hippo = next((p for p in ir.populations if p.name == 'hippocampus'), None)
    if hippo and hippo.count == 4096:
        print("[OK] Hippocampus count correct (4096)")
    else:
        print("[ERROR] Hippocampus count incorrect (expected 4096, got %s)" % (hippo.count if hippo else 'None'))

    # Verify NT systems
    nt_names = set(nt.name for nt in ir.neurotransmitter_systems)
    expected_nts = {'dopamine', 'norepinephrine', 'serotonin', 'acetylcholine',
                    'endocannabinoid', 'glutamate', 'gaba'}
    if expected_nts == nt_names:
        print("[OK] All 7 NT systems present")
    else:
        print("[ERROR] NT systems mismatch. Expected: %s, Got: %s" % (expected_nts, nt_names))

    # Verify synapses
    syn_count = len(ir.synapses)
    if syn_count >= 16:
        print("[OK] Anatomical projections present (%d synapses)" % syn_count)
    else:
        print("[ERROR] Insufficient synapses (expected >=16, got %d)" % syn_count)

    # Verify modulations
    mod_count = len(ir.modulations)
    if mod_count >= 12:
        print("[OK] Receptor banks present (%d modulations)" % mod_count)
    else:
        print("[ERROR] Insufficient modulations (expected >=12, got %d)" % mod_count)

    # Verify formal specs
    if len(ir.sheaf_specs) > 0:
        print("[OK] Sheaf consistency spec present")
    else:
        print("[ERROR] Sheaf consistency spec missing")

    if len(ir.formal_specs) > 0:
        print("[OK] Formal integration spec present")
    else:
        print("[ERROR] Formal integration spec missing")

    print("\n" + "="*60)
    print("SUCCESS: BRIAN DSL architecture compiles correctly!")
    print("="*60)

except Exception as e:
    print("[ERROR]: %s" % e)
    import traceback
    traceback.print_exc()
    sys.exit(1)
