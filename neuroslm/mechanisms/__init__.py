# -*- coding: utf-8 -*-
"""Novel architectural mechanisms grounded in the THSD cellular-sheaf
framework (docs/formal_framework.md). Each module ships as a discrete
unit so a NaN regression can be bisected to a single mechanism.

Current roster:
  topo_charge          -- Pontryagin / Hopfion-lite topological-charge
                          diagnostic (Phase 1).
  liouville_symplectic -- Symplectic leapfrog residual block, det(J)=1
                          by construction (Phase 2, planned).
  kjpla                -- Kuramoto-Josephson Phase Lattice Attention
                          (Phase 3, planned).
"""
