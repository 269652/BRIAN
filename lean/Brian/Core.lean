/-
  Brian.Core — umbrella import surface for the Brian Lean library.

  Hypothesis proofs at `hypothesis/proofs/H###_*.lean` should
  import this single module to bring the entire THSD vocabulary
  into scope; per CLAUDE.md §12.1 they may also import narrower
  `Brian.Thsd.*` / `Brian.Verification.*` / `Brian.Statistics.*`
  submodules instead.

  Layout (mirrors `docs/formal_framework.md` §10.2):

    Brian.Postulate              -- empirical-admission namespace
    Brian.Postulate.Cdga         -- H002 contraction postulate
    Brian.Postulate.Coboundary   -- H004 H¹/Fiedler admissions
    Brian.Postulate.Symbolic     -- H003 Gumbel-Softmax collapse
    Brian.Postulate.Welch        -- H005 Type-I error bound (handle)
    Brian.Postulate.Nfo          -- H017 Swift–Hohenberg Lyapunov
    Brian.Thsd.Simplex           -- SimplexComplex K
    Brian.Thsd.Sheaf             -- CellularSheaf F, Coupling, ⊕
    Brian.Thsd.Coboundary        -- H1Vanishes, LambdaPositive (opaque)
    Brian.Thsd.Phi               -- IIT 4.0 Φ proxy + monotonicity
    Brian.Thsd.Symbolic          -- SymbolicSimplex / AnnealedUnit
    Brian.Cdga                   -- CdgaRegularizer + gap_monotone
    Brian.Nfo                    -- H015/H016/H018 NFO vocabulary
    Brian.NN                     -- LinearLayer, CoherenceGate,
                                 --   ResidualUpdate (neural net notation)
    Brian.Verification.TripleGuard
    Brian.Statistics.Welch       -- ImprovementGate Welch spec
-/
import Brian.Postulate
import Brian.Postulate.Cdga
import Brian.Postulate.Coboundary
import Brian.Postulate.Symbolic
import Brian.Postulate.Welch
import Brian.Postulate.Nfo
import Brian.Thsd.Simplex
import Brian.Thsd.Sheaf
import Brian.Thsd.Coboundary
import Brian.Thsd.Phi
import Brian.Thsd.Symbolic
import Brian.Cdga
import Brian.Nfo
import Brian.NN
import Brian.Verification.TripleGuard
import Brian.Statistics.Welch
