/-
  Brian.Thsd.Simplex — Simplicial complex K.

  Mirrors `neuroslm/thsd/engine.py::SimplexComplex`. See
  `docs/formal_framework.md` §2 ("THSD primitives").

  A simplicial complex K is a finite collection of simplices σ
  organized by dimension d ∈ {0, 1, …, dimMax}. We do not need the
  general k-face boundary apparatus for the current hypothesis set
  (H001 needs only vertex/edge counts via the sheaf Laplacian); we
  expose just enough structure to define `Sheaf` on top.
-/
namespace Brian.Thsd

/-- A simplicial complex with vertex- and edge-counts and an upper
    dimension cap. Higher-dimensional cells (triangles, …) are
    tracked abstractly by `numCellsAt` so future hypothesis proofs
    can extend the structure without refactoring downstream files. -/
structure SimplexComplex where
  /-- Highest simplex dimension declared in K. -/
  dimMax : Nat
  /-- Number of 0-simplices (vertices). -/
  numVertices : Nat
  /-- Number of 1-simplices (edges). -/
  numEdges : Nat
  /-- Number of k-cells for k ≥ 2; default zero. -/
  numCellsAt : Nat → Nat := fun _ => 0
  deriving Inhabited

namespace SimplexComplex

/-- Total cell count across all dimensions ≤ dimMax. -/
def totalCells (K : SimplexComplex) : Nat :=
  K.numVertices + K.numEdges +
    (List.range (K.dimMax + 1)).foldl
      (fun acc d => if d ≥ 2 then acc + K.numCellsAt d else acc) 0

/-- The number of simplices at a given dimension. -/
def numAt (K : SimplexComplex) : Nat → Nat
  | 0 => K.numVertices
  | 1 => K.numEdges
  | (n+2) => K.numCellsAt (n+2)

/-- Adding a vertex strictly increases the vertex count. -/
theorem numVertices_succ_lt (K : SimplexComplex) :
    K.numVertices < K.numVertices + 1 :=
  Nat.lt_succ_self K.numVertices

/-- Adding an edge strictly increases the edge count. -/
theorem numEdges_succ_lt (K : SimplexComplex) :
    K.numEdges < K.numEdges + 1 :=
  Nat.lt_succ_self K.numEdges

end SimplexComplex

end Brian.Thsd
