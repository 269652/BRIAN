# THSD Refactoring Plan — Full Implementation with TDD

**Status:** Starting Phase 1  
**Target:** Complete THSD support with parser, compiler, hypergraph IR, and comprehensive tests

---

## Overview

Transform RCC Bowtie from imperative DSL to formal **Topological Hyper-Sheaf Dynamics (THSD)** notation:
- Replace `population` with `complex` (simplicial complexes σᵈᵢ)
- Add sheaf-stalk definitions (local representation + Fisher-Information)
- Formalize topological invariants (H¹, λ₁, Φ)
- Compile to hypergraph intermediate representation
- Full parser + compiler + test coverage

---

## Phase 1: THSD DSL Syntax & Parser (THIS SESSION)

### What We're Building

New DSL blocks in arch.neuro:

```neuro
# Simplicial complex declaration (replaces population)
complex LanguageCortex {
    # Sheaf stalk definition
    stalk {
        representation_dim: 512,
        fisher_information_metric: "information_geometry",
        local_constraints: ["predictive_consistency"]
    },
    
    # Topology specification
    topology {
        kind: "Tonnetz",
        spectral_gap: 0.3,      # λ₁ (Fiedler value) > 0.3
        dimension: 8,
        coherence_threshold: 0.95
    },
    
    # Cohomological constraints
    formal_spec {
        # H¹(K; F) ≈ 0 (minimize hallucinations)
        cohomology_floor: 0.01,
        
        # IIT 4.0: maximize Φ (integrated information)
        phi_target: 0.8,
        phi_method: "geometric_IIT4",
        
        # Information bottleneck (NEMORI)
        information_bottleneck: {
            enabled: true,
            compression_ratio: 0.7,
            prediction_lower_bound: 0.95
        }
    },
    
    # Dynamic operators (vesicles, mutations)
    dynamics {
        # Emission kernel: P_emit
        emission {
            trigger: "surprise_head(threshold=0.8)",
            payload_dim: 64,
            lifetime_steps: 100
        },
        
        # Release operator: R_rule  
        release {
            rule: "rank_one_update",
            learning_rate: 0.001,
            target: "parameter_counts"
        },
        
        # Predictive forgetting
        nemori {
            enabled: true,
            consolidation_interval: 1000,
            forgetting_floor: 0.01
        }
    }
}

# Sheaf (constraint bundle)
sheaf LanguageSheath {
    base_complex: "LanguageCortex",
    sections: [
        {name: "syntactic_layer", dimension: 256},
        {name: "semantic_layer", dimension: 256}
    ],
    consistency_check: "fisher_information_divergence < 0.05"
}

# Grand unified loss function
formal_spec {
    loss_equation: """
    min_θ,V L_LM + λL_FE - βΦ(K) + γ‖H¹(F)‖
    
    where:
      L_LM = cross_entropy(logits, targets)
      L_FE = free_energy(PCT hierarchy)
      Φ(K) = integrated_information(simplicial_complex)
      H¹(F) = first_cohomology_group(sheaf_bundle)
    
    with weights:
      λ = 0.02  (free energy coupling)
      β = 0.5   (Φ maximization strength)
      γ = 1.0   (cohomology penalty)
    """,
    
    # Convergence criteria
    convergence {
        phi_min: 0.75,
        cohomology_max: 0.02,
        gap_ratio_max: 2.0,
        steps_to_verify: 500
    }
}
```

### Files to Create (TDD)

**Test files created FIRST, then implementation:**

1. `tests/dsl/test_thsd_parser_complex.py` — test `complex` block parsing
2. `tests/dsl/test_thsd_parser_sheaf.py` — test `sheaf` block parsing
3. `tests/dsl/test_thsd_parser_formal_spec.py` — test `formal_spec` block parsing
4. `tests/dsl/test_thsd_ir.py` — test IR construction from parsed blocks

**Source files:**

1. `neuroslm/dsl/thsd_ir.py` — IR dataclasses (SimplexIR, SheafIR, CohomologyIR, etc.)
2. `neuroslm/dsl/compiler.py` — extend NeuroMLCompiler with THSD parsing
3. `neuroslm/dsl/thsd_parser.py` — dedicated THSD block parser (helper module)

---

## Phase 2: Hypergraph Intermediate Representation

Create hypergraph IR that represents simplicial complexes + sheaves:

**File:** `neuroslm/dsl/hypergraph_ir.py`

```python
@dataclass
class SimplexNode:
    """A simplex σᵈᵢ in the complex K"""
    id: str
    dimension: int      # 0=vertex, 1=edge, 2=face, ...
    name: str
    stalk: SheafStalk   # F(σ)
    
@dataclass
class HypergraphEdge:
    """Edge between simplices (face relation)"""
    src_simplex: str
    dst_simplex: str
    kind: str           # "faces" | "coboundary" | "coupling"
    weight: float
    
@dataclass
class HypergraphIR:
    """Complete hypergraph representation"""
    simplices: Dict[str, SimplexNode]
    edges: Dict[str, HypergraphEdge]
    sheaf_bundle: Optional[SheafIR]
    invariants: TopologicalInvariants
    loss_equation: str
```

---

## Phase 3: Compiler Extensions

Update `CodeGenerator` to:
1. Compile THSD IR → hypergraph
2. Validate topological invariants (spectral gap, cohomology)
3. Emit PyTorch modules that enforce constraints

---

## Phase 4: Integration Testing

End-to-end: arch.neuro → THSD IR → hypergraph → compiled module → forward pass

---

## Test Structure (TDD Order)

### Phase 1 Tests (This Session)

```
tests/dsl/test_thsd_parser_complex.py
├─ test_complex_block_parses_basic
├─ test_complex_stalk_definition
├─ test_complex_topology_tonnetz
├─ test_complex_formal_spec_cohomology
├─ test_complex_dynamics_emission
├─ test_complex_dynamics_release
└─ test_complex_full_integration

tests/dsl/test_thsd_parser_sheaf.py
├─ test_sheaf_block_parses
├─ test_sheaf_sections_defined
├─ test_sheaf_consistency_check
└─ test_sheaf_references_complex

tests/dsl/test_thsd_parser_formal_spec.py
├─ test_formal_spec_loss_equation
├─ test_formal_spec_convergence_criteria
└─ test_formal_spec_weights_normalized

tests/dsl/test_thsd_ir.py
├─ test_complex_ir_construction
├─ test_sheaf_ir_construction
├─ test_topological_invariants_extracted
└─ test_ir_roundtrip_fidelity
```

---

## Critical Success Criteria

- ✅ All Phase 1 tests GREEN before moving to Phase 2
- ✅ Parser correctly extracts all THSD blocks
- ✅ IR fully captures mathematical semantics
- ✅ No regressions in existing tests
- ✅ Hypergraph IR validates topological constraints
- ✅ End-to-end compilation produces valid PyTorch modules

---

## Timeline

- **Phase 1 (today):** Parser + IR + tests
- **Phase 2 (next):** Hypergraph representation + validation
- **Phase 3:** Compiler extensions
- **Phase 4:** Integration + full test suite
- **Phase 5:** Documentation sync + final commit

