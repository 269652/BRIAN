"""Hyperdimensional computing + neural geometry for compositional intelligence.

Implements three research-grade novel architectures:

1. **Hyperdimensional Vector Symbolic Architecture (VSA)**
   Based on Kanerva (2009), Frady et al. (2021). Uses 
   high-dimensional holographic vectors where:
   - Binding = element-wise multiplication (bundling concepts)
   - Bundling = element-wise addition (superposition)
   - Permutation = role assignment (sequence/structure)
   This gives algebraic compositionality: DOG*CHASES + CAT*CHASED_BY
   
2. **Fractal Recursive Attention**
   Self-similar attention at multiple scales. The same attention pattern
   repeats fractally: micro-attention within tokens, meso-attention within
   chunks, macro-attention across chunks. Meta-learnable depth and branching.
   Inspired by the fractal organization of cortical connectivity.

3. **Neural Manifold Geodesic Flow**
   Thought moves along geodesics on a learned Riemannian manifold.
   Instead of straight-line interpolation between concepts, thought
   follows the natural curvature of semantic space. Implements:
   - Learned metric tensor (how distances work in concept space)
   - Geodesic integration (thought follows curvature)
   - Parallel transport (maintaining structure during thought flow)
   - Curvature detection (regions of high conceptual complexity)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. Hyperdimensional Vector Symbolic Architecture
# =====================================================================
class HyperdimensionalVSA(nn.Module):
    """Holographic reduced representation for compositional reasoning.
    
    Operations in hyperspace:
      bind(A, B)   = A ⊙ B        (element-wise multiply — creates compound)
      bundle(A, B) = A + B         (superposition — creates set)
      permute(A)   = roll(A, 1)    (structural role assignment)
      unbind(AB, B)= AB ⊙ B*      (retrieve A from compound)
      similarity   = cos(A, B)     (associative retrieval)
    """

    def __init__(self, d_hyper: int = 512, n_roles: int = 8):
        super().__init__()
        self.d_hyper = d_hyper
        self.n_roles = n_roles

        # Learned role vectors (structural positions)
        self.roles = nn.Parameter(torch.randn(n_roles, d_hyper) / math.sqrt(d_hyper))

        # Cleanup memory: auto-associative network to denoise
        self.cleanup = nn.Sequential(
            nn.Linear(d_hyper, d_hyper * 2),
            nn.GELU(),
            nn.Linear(d_hyper * 2, d_hyper),
        )

        # Resonator network: iterative factorization (Frady et al. 2020)
        self.resonator_steps = 3
        self.resonator_proj = nn.Linear(d_hyper, d_hyper, bias=False)

    @staticmethod
    def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Binding: element-wise multiply. Produces compound representation."""
        return a * b

    @staticmethod
    def bundle(*vecs: torch.Tensor) -> torch.Tensor:
        """Bundling: superposition via addition + normalization."""
        s = sum(vecs)
        return F.normalize(s, dim=-1) * math.sqrt(vecs[0].size(-1))

    @staticmethod
    def permute(v: torch.Tensor, shift: int = 1) -> torch.Tensor:
        """Role assignment via circular permutation."""
        return torch.roll(v, shifts=shift, dims=-1)

    def unbind(self, compound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Retrieve value from compound given key. Uses approximate inverse."""
        # For bipolar vectors, inverse ≈ identity. For real-valued, use cleanup.
        raw = compound * key  # approximate unbinding
        return self.cleanup(raw)

    def encode_sequence(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode a sequence into a single hyperdimensional vector.
        
        tokens: (B, T, D) → (B, D) holographic sequence encoding
        """
        B, T, D = tokens.shape
        result = torch.zeros(B, D, device=tokens.device)
        for t in range(min(T, self.n_roles)):
            role = self.roles[t]  # (D,)
            bound = self.bind(tokens[:, t], role)
            result = result + bound
        return F.normalize(result, dim=-1) * math.sqrt(D)

    def resonator_factorize(self, compound: torch.Tensor,
                            codebook: torch.Tensor) -> torch.Tensor:
        """Resonator network: factorize compound into components.
        
        compound: (B, D) bound representation
        codebook: (N, D) known atomic vectors
        
        Returns: (B, D) best matching factorization
        """
        estimate = compound.clone()
        for _ in range(self.resonator_steps):
            # Compare with codebook
            sims = F.cosine_similarity(
                estimate.unsqueeze(1), codebook.unsqueeze(0), dim=-1)
            # Soft retrieval from codebook
            weights = F.softmax(sims * 10, dim=-1)  # (B, N)
            retrieved = torch.mm(weights, codebook)  # (B, D)
            # Update estimate
            estimate = self.resonator_proj(retrieved) + 0.5 * compound
            estimate = F.normalize(estimate, dim=-1) * math.sqrt(estimate.size(-1))
        return estimate

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: (B, T, D) input sequence.
        
        Returns: holographic encoding + cleaned representation.
        """
        encoded = self.encode_sequence(x)
        cleaned = self.cleanup(encoded)
        return {
            "holographic": encoded,    # (B, D) raw holographic code
            "cleaned": cleaned,        # (B, D) denoised
        }


# =====================================================================
# 2. Fractal Recursive Attention
# =====================================================================
class FractalAttentionBlock(nn.Module):
    """Self-similar attention at one fractal level."""

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.ff_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        a, _ = self.attn(h, h, h)
        x = x + a
        x = x + self.ff(self.ff_norm(x))
        return x


class FractalAttention(nn.Module):
    """Fractal recursive attention: self-similar processing at multiple scales.
    
    Level 0 (micro): attention within local windows of tokens
    Level 1 (meso): attention between window summaries
    Level 2 (macro): attention between chunk summaries of summaries
    
    Each level uses the same architectural pattern (self-similarity)
    but with meta-learned depth weights controlling contribution.
    The fractal structure naturally captures hierarchical composition.
    """

    def __init__(self, d_model: int, n_levels: int = 3,
                 window_size: int = 4, n_heads: int = 4):
        super().__init__()
        self.n_levels = n_levels
        self.window_size = window_size

        # Same attention pattern at each level (fractal self-similarity)
        self.levels = nn.ModuleList([
            FractalAttentionBlock(d_model, n_heads) for _ in range(n_levels)
        ])

        # Pool between levels (summarize windows)
        self.pool_projs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_levels - 1)
        ])

        # Unpool: broadcast summary back to original resolution
        self.unpool_projs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_levels - 1)
        ])

        # Meta-learned level weights: how much each fractal level contributes
        self.level_gates = nn.Parameter(torch.ones(n_levels) / n_levels)

        # Cross-level residual
        self.cross_level = nn.Linear(d_model * n_levels, d_model)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: (B, T, D) → multi-scale fractal processing.
        
        Returns dict with output and per-level representations.
        """
        B, T, D = x.shape
        gates = F.softmax(self.level_gates, dim=0)
        level_outputs = []
        current = x

        representations = [current]

        for level in range(self.n_levels):
            # Apply attention at this level
            processed = self.levels[level](current)
            level_outputs.append(processed)

            # Pool for next level (if not last)
            if level < self.n_levels - 1:
                T_cur = processed.size(1)
                # Pad to multiple of window_size
                pad_len = (self.window_size - T_cur % self.window_size) % self.window_size
                if pad_len > 0:
                    processed_padded = F.pad(processed, (0, 0, 0, pad_len))
                else:
                    processed_padded = processed
                # Reshape and pool
                T_padded = processed_padded.size(1)
                n_windows = T_padded // self.window_size
                windowed = processed_padded.view(B, n_windows, self.window_size, D)
                pooled = windowed.mean(dim=2)  # (B, n_windows, D)
                current = self.pool_projs[level](pooled)
                representations.append(current)

        # Combine levels: gate-weighted sum at original resolution
        # Unpool higher levels back to original T
        combined = gates[0] * level_outputs[0]
        for level in range(1, self.n_levels):
            lo = level_outputs[level]  # (B, T', D)
            T_level = lo.size(1)
            # Repeat to match original T
            ratio = max(1, T // max(T_level, 1))
            unpooled = lo.repeat_interleave(ratio, dim=1)[:, :T, :]
            if unpooled.size(1) < T:
                unpooled = F.pad(unpooled, (0, 0, 0, T - unpooled.size(1)))
            unpooled = self.unpool_projs[level - 1](unpooled)
            combined = combined + gates[level] * unpooled

        return {
            "output": combined,                    # (B, T, D)
            "level_representations": representations,
            "level_gates": gates,
        }


# =====================================================================
# 3. Neural Manifold Geodesic Flow
# =====================================================================
class NeuralManifold(nn.Module):
    """Learned Riemannian manifold for thought flow.

    Thought doesn't move in straight lines through concept space —
    it follows the curvature of learned semantic geometry.

    Implements:
      - Metric tensor g(x): defines local distances and angles
      - Christoffel symbols Γ(x): defines how parallel lines curve  
      - Geodesic flow: thought follows shortest paths on manifold
      - Curvature scalar R(x): complexity of local concept space
      - Parallel transport: maintaining coherence during thought flow
    """

    def __init__(self, d_model: int, n_chart_params: int = 32):
        super().__init__()
        self.d_model = d_model

        # Metric tensor network: position → local metric
        # g(x) is a d×d positive-definite matrix, parametrized via Cholesky
        self.metric_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_chart_params),
        )

        # From chart params → Cholesky factor L (lower triangular)
        # g = L @ L^T ensures positive definiteness
        self.n_chart = n_chart_params
        tri_size = d_model  # we'll use a low-rank approximation
        self.cholesky_proj = nn.Linear(n_chart_params, tri_size)

        # Christoffel symbol approximation: how the metric changes
        self.connection_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
        )

        # Curvature estimator
        self.curvature_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # Geodesic integrator: Euler steps along the manifold
        self.n_steps = 4
        self.step_size = nn.Parameter(torch.tensor(0.1))

        # Exponential map: tangent vector → manifold point
        self.exp_map = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
        )

    def metric_at(self, x: torch.Tensor) -> torch.Tensor:
        """Compute local metric tensor at position x.
        
        Returns a low-rank approximation: g(x) ≈ I + L(x)L(x)^T
        This ensures positive definiteness.
        """
        chart = self.metric_net(x)  # (B, n_chart)
        L = self.cholesky_proj(chart)  # (B, d_model) — low rank factor
        # g = I + L⊗L (rank-1 update to identity)
        # For efficiency, we don't form the full matrix
        return L  # return the factor; operations use it implicitly

    def geodesic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Approximate geodesic distance between x and y on the manifold.
        
        Uses the metric tensor to warp Euclidean distance.
        """
        diff = y - x
        L = self.metric_at(x)
        # d²(x,y) ≈ ||y-x||² + (L·(y-x))²  (metric-warped distance)
        euclidean = (diff ** 2).sum(-1)
        projected = (L * diff).sum(-1) ** 2
        return (euclidean + projected).sqrt()

    def parallel_transport(self, vector: torch.Tensor, 
                          start: torch.Tensor,
                          end: torch.Tensor) -> torch.Tensor:
        """Transport a vector from start to end along the manifold.
        
        Maintains the "meaning" of a direction as thought moves.
        """
        connection = self.connection_net(torch.cat([start, end], dim=-1))
        # First-order approximation: v_transported ≈ v + Γ(path) · v
        transported = vector + 0.1 * connection * vector
        return transported

    def geodesic_step(self, position: torch.Tensor,
                      velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One step of geodesic integration.
        
        Updates position and velocity according to manifold curvature.
        """
        # Christoffel correction to velocity (curvature bends the path)
        connection = self.connection_net(
            torch.cat([position, velocity], dim=-1))
        # Geodesic equation: dv/dt = -Γ·v·v (simplified)
        velocity_correction = -0.1 * connection * velocity
        new_velocity = velocity + self.step_size * velocity_correction

        # Exponential map: move on manifold
        new_position = self.exp_map(
            torch.cat([position, new_velocity], dim=-1))
        # Residual connection to ensure stability
        new_position = position + self.step_size * new_position

        return new_position, new_velocity

    def flow(self, start: torch.Tensor,
             direction: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow along geodesic from start in given direction.
        
        Args:
            start: (B, D) starting position
            direction: (B, D) initial velocity (unnormalized)
            
        Returns dict with trajectory, final position, curvature.
        """
        pos = start
        vel = direction
        trajectory = [pos]

        for _ in range(self.n_steps):
            pos, vel = self.geodesic_step(pos, vel)
            trajectory.append(pos)

        # Curvature at final position
        curvature = self.curvature_head(pos).squeeze(-1)

        return {
            "position": pos,               # (B, D) final position
            "velocity": vel,               # (B, D) final velocity
            "trajectory": trajectory,      # list of (B, D)
            "curvature": curvature,        # (B,) local curvature scalar
        }


# =====================================================================
# Combined Intelligence Module
# =====================================================================
class NeuralGeometryEngine(nn.Module):
    """Integrates all three novel architectures into a unified intelligence layer.
    
    Pipeline:
      1. VSA encodes compositional structure
      2. Fractal attention processes at multiple scales
      3. Manifold flow guides thought along learned geometry
      4. Output combines all three for enriched reasoning
    """

    def __init__(self, d_model: int, n_fractal_levels: int = 3):
        super().__init__()
        self.vsa = HyperdimensionalVSA(d_hyper=d_model)
        self.fractal = FractalAttention(
            d_model, n_levels=n_fractal_levels, window_size=4)
        self.manifold = NeuralManifold(d_model)

        # Integration: combine VSA + fractal + manifold
        self.integrate = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        # Meta-learned weighting of the three streams
        self.stream_gate = nn.Sequential(
            nn.Linear(d_model, 3),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor,
                thought: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, D) sequence of hidden states
            thought: (B, D) current floating thought (direction for manifold)
            
        Returns dict with enriched representation and component outputs.
        """
        B = x.size(0)

        # 1. Hyperdimensional VSA: compositional encoding
        vsa_out = self.vsa(x)
        holographic = vsa_out["cleaned"]  # (B, D)

        # 2. Fractal attention: multi-scale processing
        fractal_out = self.fractal(x)
        fractal_repr = fractal_out["output"].mean(1)  # (B, D) pooled

        # 3. Manifold geodesic flow: thought follows curvature
        direction = thought - holographic  # direction from structure to current thought
        manifold_out = self.manifold.flow(thought, direction)
        manifold_repr = manifold_out["position"]  # (B, D)

        # Meta-learned stream gating
        gate_input = (holographic + fractal_repr + manifold_repr) / 3
        gates = self.stream_gate(gate_input)  # (B, 3)

        # Weighted combination
        weighted = (gates[:, 0:1] * holographic +
                    gates[:, 1:2] * fractal_repr +
                    gates[:, 2:3] * manifold_repr)

        # Full integration with cross-stream interactions
        full = self.integrate(torch.cat(
            [holographic, fractal_repr, manifold_repr], dim=-1))

        # Final: gated residual
        output = 0.5 * weighted + 0.5 * full

        return {
            "output": output,                      # (B, D) enriched
            "holographic": holographic,            # (B, D) VSA
            "fractal": fractal_repr,               # (B, D) multi-scale
            "manifold_position": manifold_repr,    # (B, D) geodesic endpoint
            "curvature": manifold_out["curvature"],  # (B,) local complexity
            "stream_gates": gates,                 # (B, 3) which stream dominates
            "fractal_level_gates": fractal_out["level_gates"],
        }
