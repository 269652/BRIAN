# -*- coding: utf-8 -*-
"""Observability metrics for the DSL model (Phases N7/N8).

Every metric is computed from the DSL model's *own* activations and
computation graph — Φ (integrated information), λ₁ (Fiedler / algebraic
connectivity), GWS ignition, oscillation bands, the neurotransmitter
system, trophic state, and meso learning gain. This is the IIT /
hypershape-aligned approach (docs/dsl_nn_language.md §N10–N11): the model
is a typed graph, and these are genuine spectral / information-theoretic
measures over it, not copies of Brain's module-specific values.

They populate the native `train.py` log columns so a DSL training run
reports the same metrics as the hand-written trainer.
"""
from __future__ import annotations
import math
from collections import deque
from typing import Dict, List, Optional

import torch


# ── Φ proxy — integrated information over the representation ────────────

def phi_proxy(layer_acts: List[torch.Tensor]) -> float:
    """Gaussian mutual information between the two halves of the final
    representation — a tractable integrated-information proxy.

    Φ ≈ H(A) + H(B) − H(A,B) for a bipartition of the feature dims into
    halves A,B, using Gaussian differential entropy
    H = ½·logdet(2πe·Σ). High when the halves are statistically
    integrated (can't be factorised), zero when independent.
    """
    if not layer_acts:
        return 0.0
    h = layer_acts[-1]                       # (B, T, D)
    x = h.reshape(-1, h.shape[-1]).float()   # (N, D)
    N, D = x.shape
    if D < 2 or N < 2:
        return 0.0
    half = D // 2
    x = x - x.mean(dim=0, keepdim=True)

    def _gauss_entropy(mat: torch.Tensor) -> float:
        # ½ logdet(2πe Σ); regularise Σ for numerical stability.
        d = mat.shape[1]
        cov = (mat.T @ mat) / max(1, mat.shape[0] - 1)
        cov = cov + 1e-4 * torch.eye(d, device=cov.device)
        sign, logdet = torch.linalg.slogdet(cov)
        if sign <= 0:
            return 0.0
        return 0.5 * (logdet.item() + d * math.log(2 * math.pi * math.e))

    h_a = _gauss_entropy(x[:, :half])
    h_b = _gauss_entropy(x[:, half:])
    h_ab = _gauss_entropy(x)
    # Per-dimension integration density (scale-invariant across model
    # widths) — the "intelligence density" handle for the hypershape goal.
    return max(0.0, (h_a + h_b - h_ab) / half)


# ── Fiedler λ₁ — algebraic connectivity of the layer graph ─────────────

def fiedler_lambda(n_layers: int, attention: bool = True) -> float:
    """Second-smallest eigenvalue of the normalized Laplacian of the
    architecture's layer-connectivity graph.

    A depth-L residual transformer is a chain of L nodes (residual stream),
    each with a self-loop (self-attention). The Fiedler value measures how
    well-connected / hard-to-bipartition the computation graph is — a
    direct graph-theoretic handle for the hypershape analysis (N11).
    """
    if n_layers <= 1:
        return 0.0
    # Chain adjacency (residual stream): node i connects to i-1, i+1.
    n = n_layers
    A = torch.zeros(n, n)
    for i in range(n - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    if attention:
        A += torch.eye(n) * 0.5   # self-attention self-loops
    deg = A.sum(dim=1)
    d_inv_sqrt = torch.diag((deg + 1e-8).rsqrt())
    L = torch.eye(n) - d_inv_sqrt @ A @ d_inv_sqrt
    eig = torch.linalg.eigvalsh(L)            # ascending
    # Fiedler = second-smallest eigenvalue
    return float(eig[1].clamp(min=0.0))


# ── GWS ignition — sparsity / peakiness of conscious access ────────────

def gws_ignition(act: torch.Tensor) -> float:
    """Fraction of 'ignited' units: how peaked the representation is.

    Computed as the mean (over batch/time) max-softmax probability scaled
    to [0,1] — high when a few units dominate (ignition), low when activity
    is diffuse. Mirrors the winner-take-all ignition gate of the GWS.
    """
    x = act.reshape(-1, act.shape[-1]).float()
    if x.numel() == 0:
        return 0.0
    p = torch.softmax(x, dim=-1)
    peak = p.max(dim=-1).values.mean().item()   # in [1/D, 1]
    D = x.shape[-1]
    # Normalise: uniform → 0, fully peaked → 1
    return float(max(0.0, min(1.0, (peak - 1.0 / D) / (1.0 - 1.0 / D))))


# ── Oscillation bands — δ/θ/γ power of activation dynamics ─────────────

class OscillationTracker:
    """Tracks a scalar activation signal over time and reports its power
    in three frequency bands (δ low, θ mid, γ high) via rFFT — the neural
    'oscillation' readout in the native log."""

    def __init__(self, window: int = 64):
        self.window = window
        self.history: deque = deque(maxlen=window)

    def observe(self, act: torch.Tensor) -> None:
        self.history.append(float(act.float().abs().mean().item()))

    def bands(self) -> Dict[str, float]:
        if len(self.history) < 4:
            return {"δ": 0.0, "θ": 0.0, "γ": 0.0}
        sig = torch.tensor(list(self.history), dtype=torch.float32)
        sig = sig - sig.mean()
        spec = torch.fft.rfft(sig).abs()      # (F,)
        spec = spec[1:]                       # drop DC
        if spec.numel() == 0:
            return {"δ": 0.0, "θ": 0.0, "γ": 0.0}
        third = max(1, spec.numel() // 3)
        total = spec.sum().item() + 1e-8
        delta = spec[:third].sum().item() / total
        theta = spec[third:2 * third].sum().item() / total
        gamma = spec[2 * third:].sum().item() / total
        return {"δ": float(delta), "θ": float(theta), "γ": float(gamma)}


# ── Neurotransmitter system — 7-NT homeostatic state ───────────────────

class NTSystem:
    """Seven-neurotransmitter state with first-order homeostatic kinetics,
    initialised from arch.neuro base concentrations.

    Levels drift toward their baselines with a small activity-driven
    release term — a compact, stateful analogue of Brain's transmitter
    system, sufficient to populate the NT[...] log column."""

    _BASELINES = {"DA": 0.10, "NE": 0.15, "5HT": 0.30, "ACh": 0.20,
                  "eCB": 0.05, "Glu": 0.40, "GABA": 0.10}

    def __init__(self, baselines: Optional[Dict[str, float]] = None,
                 reuptake: float = 0.1):
        self._level = dict(baselines or self._BASELINES)
        self._baseline = dict(self._level)
        self.reuptake = reuptake

    def step(self, activity: float = 0.0) -> None:
        # Small activity-driven release, strong reuptake → levels hover
        # near baseline with mild activity-dependent modulation (Brain's
        # NT columns are near-static around their baselines).
        rel = 0.02 * math.tanh(max(0.0, activity))     # bounded release
        for k in self._level:
            v = self._level[k]
            v = v + rel - 0.4 * (v - self._baseline[k])
            self._level[k] = float(max(0.0, min(1.0, v)))

    def levels(self) -> Dict[str, float]:
        return dict(self._level)


# ── Trophic system — per-projection BDNF-like state ────────────────────

class TrophicSystem:
    """Per-layer trophic (BDNF) state: an EMA of activation magnitude per
    projection. Reports n_active / n_projections and the mean trophic
    level — the troph a/b μX log column."""

    def __init__(self, n_projections: int, decay: float = 0.95,
                 active_thresh: float = 0.05):
        self.n_projections = n_projections
        self.decay = decay
        self.active_thresh = active_thresh
        self._troph = [0.0] * n_projections

    def step(self, layer_acts: List[torch.Tensor]) -> None:
        for i in range(min(self.n_projections, len(layer_acts))):
            mag = float(layer_acts[i].float().abs().mean().item())
            self._troph[i] = self.decay * self._troph[i] + (1 - self.decay) * mag

    def stats(self) -> Dict:
        n_active = sum(1 for t in self._troph if t > self.active_thresh)
        mean = sum(self._troph) / max(1, len(self._troph))
        return {"n_projections": self.n_projections, "n_active": n_active,
                "trophic_mean": float(mean)}


# ── Meso learning gain — smoothed loss-improvement rate ────────────────

class MesoLearningGain:
    """EMA of the normalised step-to-step loss improvement — the mesoLG
    column. Clamped to [0,1]; ~0.5 at steady descent."""

    def __init__(self, decay: float = 0.95):
        self.decay = decay
        self._prev = None
        self._lg = 0.5

    def step(self, loss: float) -> float:
        if self._prev is not None and self._prev > 1e-6:
            improve = (self._prev - loss) / self._prev      # >0 if improving
            inst = 0.5 + 5.0 * improve                       # center 0.5
            inst = max(0.0, min(1.0, inst))
            self._lg = self.decay * self._lg + (1 - self.decay) * inst
        self._prev = loss
        return self._lg


# ── Bundle: MetricObserver ─────────────────────────────────────────────

class MetricObserver:
    """Bundles all observers and produces the metric dict that
    train_dsl logs in the native format.

    Setting ``enable_emergent=True`` swaps in the C1–C6 telemetry layer
    from `neuroslm.emergent` (see docs/EMERGENT_TOPOLOGY.md). When
    enabled the returned metric dict contains *both* the legacy keys
    (for back-compat) and additional ``nt_driven``, ``ign_rate``,
    ``pc_residual``, ``Q_total``, ``Q_walls``, ``Q_plateau_len``,
    ``lattice_spec``, ``pac`` keys. The legacy ``nt`` and ``ignition``
    values are *replaced* by their driven counterparts when emergent is
    on, so the existing log column line is reused.

    When ``enable_emergent=False`` (default) ``observe()`` returns a
    dict byte-identical to the legacy behaviour.
    """

    def __init__(self, n_layers: int,
                 nt_baselines: Optional[Dict[str, float]] = None,
                 enable_emergent: bool = True,
                 emergent_dim: Optional[int] = None,
                 nt_w_trainable: bool = False):
        self.n_layers = n_layers
        self.nt = NTSystem(nt_baselines)
        self.trophic = TrophicSystem(n_projections=n_layers)
        self.osc = OscillationTracker()
        self.meso = MesoLearningGain()
        self._fiedler = fiedler_lambda(n_layers)   # static — graph property
        self.enable_emergent = bool(enable_emergent)
        self._emergent = None
        # Autograd-tracked PC residual stashed by observe(); read by
        # harness.compute_loss() when training_config.pc_reentry_weight > 0.
        # Tensor with requires_grad=True OR None.
        self.last_pc_residual_diff = None
        if self.enable_emergent:
            # Lazy import so the emergent package is optional.
            from neuroslm.emergent import (
                DrivenNTSystem, MetastableIgnition, PCReentryProbe,
                TopologicalChargeProbe, BowtieLatticeProbe, PACBindingProbe,
            )
            self._emergent = {
                # Item 6: when nt_w_trainable=True the OU coupling matrix
                # is exposed as a (7, 5) nn.Parameter. The float OU
                # dynamics in step_full() are unchanged either way, so
                # `levels()` is bit-identical — only `predict_nt_tensor()`
                # carries the gradient path that lets the optimiser
                # refine W.
                "nt":      DrivenNTSystem(
                    baselines=nt_baselines,
                    trainable_W=bool(nt_w_trainable),
                ),
                "ign":     MetastableIgnition(),
                "pc":      None,                # constructed on first call (need dim)
                "topo":    None,                # constructed on first call
                "lattice": None,                # constructed on first call
                "pac":     PACBindingProbe(),
            }
            self._emergent_dim_hint = emergent_dim
            # Class refs we'll need for lazy construction.
            self._PC = PCReentryProbe
            self._TOPO = TopologicalChargeProbe
            self._LATTICE = BowtieLatticeProbe

    def observe(self,
                layer_acts: List[torch.Tensor],
                loss: float,
                grad_norm: Optional[float] = None,
                h_motor: Optional[torch.Tensor] = None,
                h_sensory: Optional[torch.Tensor] = None,
                attn_entropy_norm: Optional[float] = None,
                class_label: Optional[int] = None) -> Dict:
        # Update stateful subsystems
        last = layer_acts[-1] if layer_acts else torch.zeros(1, 1, 1)
        activity = float(last.float().abs().mean().item())
        self.nt.step(activity=activity)
        self.trophic.step(layer_acts)
        self.osc.observe(last)
        lg = self.meso.step(loss)

        troph = self.trophic.stats()
        phi = phi_proxy(layer_acts)
        ign_legacy = gws_ignition(last)

        out = {
            "phi": phi,
            "fiedler": self._fiedler,
            "ignition": ign_legacy,
            "meso_lg": lg,
            "troph_active": troph["n_active"],
            "troph_total": troph["n_projections"],
            "troph_mean": troph["trophic_mean"],
            "nt": self.nt.levels(),
            "osc": self.osc.bands(),
        }

        if not self.enable_emergent:
            return out

        em = self._emergent
        # ── C2: metastable ignition (peak computed from `last`) ────
        peak = em["ign"].peak(last)
        # NE for the ignition coupling comes from the driven NTs
        # *after* we step them — but we need ignition_rate as a driver
        # for GABA. Resolve the loop by using last-step's stored values.
        ign_stats = em["ign"].step(peak, ne=em["nt"].levels().get("NE", 0.0))

        # ── C1: driven NTs (full closed-loop) ──────────────────────
        em["nt"].step_full(
            loss=float(loss),
            grad_norm=grad_norm,
            activation=activity,
            ignition_rate=ign_stats["ign_rate"],
            attn_entropy_norm=attn_entropy_norm,
        )
        nt_driven = em["nt"].levels()
        # Replace the legacy NT column with the driven one so the log
        # line is unchanged in *shape* but alive in *values*.
        out["nt"] = nt_driven
        out["nt_driven"] = nt_driven
        out.update({k: v for k, v in ign_stats.items()})
        # And replace `ignition` with the metastable strength so the
        # existing log column has the same name but the live value.
        out["ignition"] = ign_stats["ign_strength"]

        # ── C3: PC reentry (needs motor / sensory) ─────────────────
        dim = None
        if h_motor is not None:
            dim = h_motor.shape[-1]
        elif h_sensory is not None:
            dim = h_sensory.shape[-1]
        elif self._emergent_dim_hint is not None:
            dim = int(self._emergent_dim_hint)
        if dim is not None and em["pc"] is None:
            em["pc"] = self._PC(dim=dim)
        if em["pc"] is not None:
            pc_stats = em["pc"].step(h_motor, h_sensory)
            out.update(pc_stats)
            # Autograd-tracked residual for the NT-gated trunk loss
            # (TrainingConfig.pc_reentry_weight > 0). The probe keeps
            # its internal SGD on a frozen-W copy; this term only
            # touches the trunk activations. None when shapes/inputs
            # are missing — harness guards.
            try:
                self.last_pc_residual_diff = em["pc"].residual_diff(
                    h_motor, h_sensory
                )
            except Exception:  # pragma: no cover — best-effort
                self.last_pc_residual_diff = None
        else:
            self.last_pc_residual_diff = None

        # ── C4: topological charge over the last layer activation ──
        topo_dim = last.shape[-1] if last.dim() >= 1 else 0
        if topo_dim >= 2:
            if em["topo"] is None or em["topo"].dim != topo_dim:
                em["topo"] = self._TOPO(dim=topo_dim)
            topo_stats = em["topo"].step(last)
            out.update(topo_stats)

        # ── C5: bowtie-lattice probe (needs class label) ───────────
        if class_label is not None and topo_dim >= 4:
            # Use K=4 by default; require dim % 4 == 0
            K = 4
            if topo_dim % K != 0:
                K = max(1, topo_dim // (topo_dim // 4))
                # Best-effort: skip if it still doesn't divide cleanly
            if topo_dim % K == 0:
                if em["lattice"] is None or em["lattice"].dim != topo_dim:
                    em["lattice"] = self._LATTICE(dim=topo_dim, K=K)
                lat_stats = em["lattice"].step(last, class_label=class_label)
                out.update(lat_stats)

        # ── C6: PAC over the same oscillation buffer ───────────────
        # Re-use the legacy OscillationTracker's history (already
        # appended above via self.osc.observe). For the probe we
        # feed the same scalar.
        em["pac"].observe(activity)
        pac_stats = em["pac"].compute()
        out.update(pac_stats)

        return out

