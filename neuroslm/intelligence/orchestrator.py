"""NeuralOrchestrator — Defines the complete brain wiring topology for NeuroSLM.

The orchestrator is the connective tissue between all brain areas. Rather than
hard-coding `output_X = module_X(output_Y)` in the forward pass, the
orchestrator:

  1. Registers all modules with their stage and connection topology
  2. Routes signals through HomeostaticGates (stability control) at every edge
  3. Executes modules in defined stage order
  4. Fuses cross-stage signals with learned attention weights
  5. Returns stability metrics and identity coherence scores

Routing graph (biological analogy → forward-pass stage):
  Stage 0  — Sensory input: TextSensoryCortex, Association
  Stage 1  — Thalamic routing: Thalamus gates cortical access
  Stage 2  — World / Self models: current state representation
  Stage 3  — Subcortical affect: Amygdala, LHb, Insula (gut feelings)
  Stage 4  — Qualia / emotional integration: QualiaState
  Stage 5  — Global Workspace: GWS integrates across all prior stages
  Stage 6  — Memory: Hippocampus, EntorhinalCortex, HyperGraph
  Stage 7  — Cognitive control: PFC, ACC (conflict monitoring)
  Stage 8  — Executive: BG action selection, ForwardModel, Cerebellum
  Stage 9  — Consciousness / Narrative: DMN, ThoughtTransformer, Claustrum
  Stage 10 — Motor output: MotorCortex

This topology is declared once in __init__ and the forward pass just calls
orchestrator.route_stage(n, ctx) at each stage boundary.

HomeostaticGate at each edge:
  Every signal crossing a module boundary passes through a HomeostaticGate
  that maintains stable signal magnitude. This prevents runaway activations
  and models thalamo-cortical gain control.

The orchestrator also tracks:
  - Per-module firing rates (are modules being used?)
  - Identity coherence (is the "self" representation stable across ticks?)
  - Neural calm score (low variance = calm; high = aroused/stressed)
  - Conflict escalation: when ACC detects high conflict, effort_steps > 0
    triggers additional cognitive cycles through stages 6-8

References:
  Dehaene et al. (2011): Experimental and Theoretical Approaches to Conscious Processing
  Baars (2005): Global Workspace Theory: A Cognitive Architecture
  Doya (1999): What are the computations of the cerebellum, basal ganglia, and cerebral cortex?
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any, Callable
from ..modules.fast_weight import FastWeightLayer


class HomeostaticGate(nn.Module):
    """Stabilises neural signal magnitude at each brain area boundary.

    Implements homeostatic plasticity: adapts gain online to keep the signal
    RMS near target_magnitude.  Also provides a small transformer refinement
    step (pre-synaptic gating) to condition the signal before/after routing.
    """

    def __init__(self, d_model: int, n_heads: int = 4,
                 target_magnitude: float = 1.0,
                 adaptation_rate: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.target_magnitude = target_magnitude
        self.adaptation_rate = adaptation_rate

        n_h = max(1, n_heads)
        while d_model % n_h != 0 and n_h > 1:
            n_h -= 1

        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_h, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.ff_norm = nn.LayerNorm(d_model)

        self.gain = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

        self.register_buffer('running_mean', torch.zeros(d_model))
        self.register_buffer('running_var',  torch.ones(d_model))
        self.register_buffer('n_updates',    torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)

        h = self.norm(x.float()).to(dtype=x.dtype)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        h = x + attn_out
        h = h + self.ff(self.ff_norm(h.float()).to(dtype=h.dtype))

        # Snapshot running stats BEFORE the in-place EMA update so the
        # autograd-tracked normalisation below does not pin a tensor whose
        # version is about to be bumped (lerp_ would otherwise raise the
        # "modified by an inplace operation" error in backward).
        rm = self.running_mean.detach().clone()
        rv = self.running_var.detach().clone()

        with torch.no_grad():
            bm = h.mean((0, 1))
            # unbiased=False avoids the "dof <= 0" warning at B=T=1 and
            # gives the population-variance estimate that the EMA assumes.
            bv = h.var((0, 1), unbiased=False)
            bm = bm.to(dtype=self.running_mean.dtype, device=self.running_mean.device)
            bv = bv.to(dtype=self.running_var.dtype, device=self.running_var.device)
            a  = self.adaptation_rate
            self.running_mean.lerp_(bm, a)
            self.running_var.lerp_(bv, a)
            self.n_updates += 1

        rms = (rv + 1e-8).sqrt()
        h   = (h - rm) / rms * self.target_magnitude
        h   = h * self.gain + self.bias

        return h.squeeze(1) if squeeze else h

    def stability_metrics(self) -> dict:
        return {
            'gain_mean':   float(self.gain.detach().mean()),
            'gain_std':    float(self.gain.detach().std()),
            'running_rms': float(self.running_var.sqrt().mean()),
            'n_updates':   int(self.n_updates.item()),
        }


class LateralGridMixer(nn.Module):
    """Horizontal lateral connections between co-active modules at the same stage.

    Implements grid-like spatial binding via multi-head attention across
    all modules in a stage.  A per-slot gate allows excitatory/inhibitory
    modulation.  Grid architectures are noted to yield high Φ_max by
    providing a spatial framework for integration.
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        n_h = n_heads
        while d_model % n_h != 0 and n_h > 1:
            n_h -= 1
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_h, batch_first=True)
        self.gate = nn.Linear(d_model, 1)   # excitatory/inhibitory lateral gate

    def forward(self, slot_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """slot_outputs: list of N tensors each (B, D).  Returns N mixed tensors."""
        n = len(slot_outputs)
        if n < 2:
            return slot_outputs
        stacked = torch.stack(slot_outputs, dim=1)          # (B, N, D)
        normed  = self.norm(stacked.float()).to(dtype=stacked.dtype)
        mixed, _ = self.attn(normed, normed, normed, need_weights=False)
        gate = torch.sigmoid(self.gate(stacked))            # (B, N, 1)
        result = stacked + gate * mixed
        return [result[:, i] for i in range(n)]


class ModuleRegistration:
    """Registration record for a brain module in the orchestrator."""
    __slots__ = ('name', 'stage', 'callable_ref', 'call_fn', 'pre_gate', 'post_gate')

    def __init__(self, name: str, stage: int,
                 callable_ref: Any,
                 call_fn: Optional[Callable] = None,
                 pre_gate: Optional[HomeostaticGate] = None,
                 post_gate: Optional[HomeostaticGate] = None):
        self.name         = name
        self.stage        = stage
        self.callable_ref = callable_ref  # the nn.Module or object
        self.call_fn      = call_fn       # custom call wrapper (None = pass through)
        self.pre_gate     = pre_gate
        self.post_gate    = post_gate


# Stage constants — document the intended topology
STAGE_SENSORY       = 0
STAGE_THALAMUS      = 1
STAGE_STATE_MODELS  = 2
STAGE_SUBCORTICAL   = 3   # amygdala, LHb, insula
STAGE_QUALIA        = 4
STAGE_GWS           = 5
STAGE_MEMORY        = 6
STAGE_COGNITIVE_CTL = 7   # PFC, ACC
STAGE_EXECUTIVE     = 8   # BG, ForwardModel, Cerebellum
STAGE_CONSCIOUSNESS = 9   # DMN, ThoughtTransformer, Claustrum
STAGE_MOTOR         = 10

STAGE_NAMES = {
    STAGE_SENSORY:       "sensory",
    STAGE_THALAMUS:      "thalamus",
    STAGE_STATE_MODELS:  "state_models",
    STAGE_SUBCORTICAL:   "subcortical",
    STAGE_QUALIA:        "qualia",
    STAGE_GWS:           "gws",
    STAGE_MEMORY:        "memory",
    STAGE_COGNITIVE_CTL: "cognitive_control",
    STAGE_EXECUTIVE:     "executive",
    STAGE_CONSCIOUSNESS: "consciousness",
    STAGE_MOTOR:         "motor",
}


class NeuralOrchestrator(nn.Module):
    """Routes neural signals through all brain areas with homeostatic control.

    Bowtie topology (Dehaene 2011):
      Narrowing path:  sensory → thalamus → GWS (bottleneck)
      Widening path:   GWS → PFC → memory → executive → motor
      Re-entry loop:   PFC output stored as _reentry_state, injected
                       into thalamus on the NEXT forward call.

    This bidirectional causal structure satisfies the IIT requirement
    that every part has both causes AND effects within the system,
    moving the integrated information proxy Φ from ≈0 toward >0.5.

    Usage:
        orch = NeuralOrchestrator(d_sem, module_names, n_heads)
        orch.register("amygdala",  STAGE_SUBCORTICAL, brain.amygdala)
        orch.register("pfc",       STAGE_COGNITIVE_CTL, brain.pfc)

        # In forward pass:
        signals = orch.route_stage(STAGE_SUBCORTICAL, base_signal, ctx)

    The orchestrator can also be used in the simpler legacy mode:
        orch_out, metrics = orch.route(sem, modules_dict)
    """

    def __init__(self, d_sem: int, module_names: List[str],
                 n_heads: int = 4, baseline: bool = False):
        super().__init__()
        self.d_sem        = d_sem
        self.baseline     = baseline
        self.module_names = list(module_names)

        # Per-stage signal fusion: combine stage outputs
        self.stage_fusions: nn.ModuleDict = nn.ModuleDict()
        self.pre_gates:     nn.ModuleDict = nn.ModuleDict()
        self.post_gates:    nn.ModuleDict = nn.ModuleDict()

        if not baseline:
            for name in module_names:
                n_h = max(1, min(n_heads, d_sem // 16))
                self.pre_gates[name]  = HomeostaticGate(d_sem, n_h)
                self.post_gates[name] = HomeostaticGate(d_sem, n_h)

            # Global workspace fusion (cross-stage attention)
            n_h = max(1, min(n_heads, d_sem // 16))
            self.fusion_norm = nn.LayerNorm(d_sem)
            self.fusion_attn = nn.MultiheadAttention(d_sem, n_h, batch_first=True)
            self.fusion_proj = nn.Linear(d_sem, d_sem)

            # Stage-to-stage bridging: each stage emits a summary that feeds forward
            n_stages = 11
            self.stage_bridges = nn.ModuleList([
                nn.Sequential(nn.Linear(d_sem, d_sem), nn.LayerNorm(d_sem), nn.GELU())
                for _ in range(n_stages)
            ])

            # Identity coherence tracking
            self.register_buffer('_identity_baseline', torch.zeros(d_sem))
            self.register_buffer('_identity_count',    torch.zeros(1))

            # ── Re-entry (bowtie): carry GWS+PFC output to next thalamus step ──
            # This creates the backward causal loop needed for IIT Φ > 0.
            # EMA with α=0.15 to stabilise across steps (prevents echo).
            self.register_buffer('_reentry_state', torch.zeros(d_sem))
            self.register_buffer('_reentry_count', torch.zeros(1))
            # Learnable mixing coefficient: how strongly re-entry modulates thalamus
            self.reentry_mix = nn.Parameter(torch.tensor(0.05))  # starts small

            # ── Φ proxy accumulator (Integrated Information) ──
            # Stores the most recent stage outputs for computing inter-module MI
            self._last_stage_outputs: List[torch.Tensor] = []

            # ── Per-module GWS feedback projections (re-entrant backward loop) ──
            # Zero-initialised: starts inactive, learns to the extent feedback helps.
            # Provides backward causal link GWS → expert module, satisfying IIT
            # intrinsicality: every part has both causes and effects.
            self.gws_feedback_projs: nn.ModuleDict = nn.ModuleDict()

            # ── Per-stage lateral grid mixers ──
            # Horizontal connections between co-active modules at the same stage.
            self.lateral_mixers: nn.ModuleDict = nn.ModuleDict({
                str(s): LateralGridMixer(d_sem, max(1, min(n_heads, d_sem // 16)))
                for s in range(11)
            })

            # ── HFW at expert cortex outputs (stages 7-8) ──
            # Added dynamically in register_module_brain when stage ∈ {7, 8}.
            self.hfw_layers: nn.ModuleDict = nn.ModuleDict()

            # Current-pass GWS broadcast (set after Stage 5 completes)
            self.register_buffer('_gws_broadcast', torch.zeros(d_sem))
            self.register_buffer('_gws_broadcast_ready',
                                 torch.zeros(1, dtype=torch.bool))

            # Fast-weight carry-over per module (not Parameters; reset per sequence)
            self._hfw_states: Dict[str, torch.Tensor] = {}

        # Dynamic registration (populated at runtime, not nn.Parameters)
        self._registrations: Dict[str, ModuleRegistration] = {}
        self._stage_signals: Dict[int, List[torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Module registration
    # ------------------------------------------------------------------

    def register_module_brain(self, name: str, stage: int,
                               callable_ref: Any,
                               call_fn: Optional[Callable] = None):
        """Register a brain module at a given stage.

        name        : unique name (must match a key in pre_gates/post_gates)
        stage       : STAGE_* constant (determines execution order)
        callable_ref: the nn.Module or object to call
        call_fn     : optional wrapper fn(module, signal, **ctx) → tensor
        """
        if name not in self.pre_gates and not self.baseline:
            # Auto-create gates for dynamically registered modules
            n_h = max(1, min(4, self.d_sem // 16))
            self.pre_gates[name]  = HomeostaticGate(self.d_sem, n_h)
            self.post_gates[name] = HomeostaticGate(self.d_sem, n_h)
            if name not in self.module_names:
                self.module_names.append(name)

        if not self.baseline:
            # GWS backward feedback projection — zero-init (starts inactive)
            if name not in self.gws_feedback_projs:
                proj = nn.Linear(self.d_sem, self.d_sem, bias=False)
                nn.init.zeros_(proj.weight)
                self.gws_feedback_projs[name] = proj

            # HFW for expert cortex stages only
            if stage in (STAGE_COGNITIVE_CTL, STAGE_EXECUTIVE) and name not in self.hfw_layers:
                n_h = max(1, min(4, self.d_sem // 16))
                while self.d_sem % n_h != 0 and n_h > 1:
                    n_h -= 1
                self.hfw_layers[name] = FastWeightLayer(
                    self.d_sem, decay=0.95, base_eta=0.05, n_heads=n_h)

        # nn.ModuleDict does not implement .get(), so use membership check
        pre  = (self.pre_gates[name] if name in self.pre_gates else None) if not self.baseline else None
        post = (self.post_gates[name] if name in self.post_gates else None) if not self.baseline else None

        self._registrations[name] = ModuleRegistration(
            name=name, stage=stage,
            callable_ref=callable_ref,
            call_fn=call_fn,
            pre_gate=pre, post_gate=post,
        )

    # ------------------------------------------------------------------
    # Stage-based routing
    # ------------------------------------------------------------------

    def begin_pass(self):
        """Reset per-pass state: stage signals, Φ history, and GWS broadcast."""
        self._stage_signals.clear()
        if not self.baseline:
            self._last_stage_outputs.clear()
            self._gws_broadcast_ready.fill_(False)

    def route_stage(self, stage: int,
                    signal: torch.Tensor,
                    context: Optional[Dict[str, Any]] = None,
                   ) -> Tuple[torch.Tensor, dict]:
        """Route signal through all modules registered at `stage`.

        signal:  (B, D) or (B, T, D) — input signal for this stage
        context: optional dict of kwargs passed to each module's call_fn

        Returns:
          output:  (B, D) or (B, T, D) — signal after all stage modules
          metrics: dict of stability/activity metrics
        """
        if self.baseline:
            return signal, {'stage': stage, 'mode': 'baseline'}

        context  = context or {}
        outputs  = []
        metrics  = {'stage': stage, 'stage_name': STAGE_NAMES.get(stage, str(stage))}
        squeeze  = signal.dim() == 2

        # Collect modules for this stage in registration order
        stage_modules = [(n, r) for n, r in self._registrations.items()
                         if r.stage == stage and r.callable_ref is not None]

        for name, reg in stage_modules:
            try:
                # Pre-gate: condition the input signal
                s = signal if not squeeze else signal.unsqueeze(1)
                pre_out = reg.pre_gate(s)
                pre_out = pre_out.squeeze(1) if squeeze else pre_out

                # Re-entrant GWS feedback: backward projection GWS → module input.
                # Satisfies IIT causal reciprocity — every part has causes AND effects.
                if (not self.baseline
                        and name in self.gws_feedback_projs
                        and bool(self._gws_broadcast_ready.item())):
                    gws_b = self._gws_broadcast.to(dtype=pre_out.dtype,
                                                   device=pre_out.device)
                    fb = self.gws_feedback_projs[name](gws_b)  # (d_sem,)
                    fb = fb.unsqueeze(0).expand(pre_out.shape[0], -1)
                    if pre_out.dim() == 3:
                        fb = fb.unsqueeze(1)
                    pre_out = pre_out + fb

                # Call module
                if reg.call_fn is not None:
                    out = reg.call_fn(reg.callable_ref, pre_out, **context)
                else:
                    out = reg.callable_ref(pre_out)

                # Normalize output shape
                if isinstance(out, dict):
                    # Module returns dict — look for a tensor output
                    for k in ('out', 'output', 'hidden', 'z', 'embedding'):
                        if k in out and torch.is_tensor(out[k]):
                            out = out[k]; break
                    else:
                        continue  # no recognised tensor key

                if isinstance(out, tuple):
                    out = out[0]

                if out.dim() == 3 and squeeze:
                    out = out.mean(1)
                if out.shape[-1] != self.d_sem:
                    continue

                # Post-gate: stabilise output
                post_out = reg.post_gate(
                    out.unsqueeze(1) if squeeze else out)
                post_out = post_out.squeeze(1) if squeeze else post_out

                # Hebbian Fast Weights at expert cortex output (stages 7-8).
                # Increases state differentiation (key IIT Φ component).
                if not self.baseline and name in self.hfw_layers:
                    hfw_in = post_out.unsqueeze(1) if post_out.dim() == 2 else post_out
                    ctx_vec = None
                    if bool(self._gws_broadcast_ready.item()):
                        ctx_vec = (self._gws_broadcast
                                   .to(dtype=hfw_in.dtype, device=hfw_in.device)
                                   .unsqueeze(0).expand(hfw_in.shape[0], -1))
                    hfw_out, W_new = self.hfw_layers[name](
                        hfw_in, context=ctx_vec, W_fast=self._hfw_states.get(name))
                    self._hfw_states[name] = W_new
                    post_out = hfw_out.squeeze(1) if squeeze else hfw_out

                outputs.append(post_out)

            except Exception:
                continue

        if not outputs:
            return signal, metrics

        # Lateral grid mixing: horizontal binding across co-active stage modules.
        # Grid-like topologies yield high Φ_max by integrating neighbouring slots.
        if not self.baseline and len(outputs) >= 2 and str(stage) in self.lateral_mixers:
            outputs = self.lateral_mixers[str(stage)](outputs)

        # Fuse all stage outputs with the original signal
        stacked = torch.stack(outputs, dim=1)             # (B, N, D) or fuse
        if stacked.dim() == 3:
            s_2d = signal if signal.dim() == 2 else signal.mean(1)
            fused_attn, _ = self.fusion_attn(
                s_2d.unsqueeze(1),
                self.fusion_norm(stacked),
                self.fusion_norm(stacked),
                need_weights=False)
            fused = self.fusion_proj(fused_attn.squeeze(1))
        else:
            fused = stacked.mean(0)

        # Residual: original signal + stage contribution
        output = signal + (fused.unsqueeze(1) if not squeeze and signal.dim() == 3
                           else fused)

        # Stage bridge: smooth transition to next stage
        if STAGE_SENSORY <= stage < STAGE_MOTOR:
            bridge_out = self.stage_bridges[stage](
                output if output.dim() == 2 else output.mean(1))
            # Accumulate bridge signal for use by next stage
            self._stage_signals.setdefault(stage, []).append(bridge_out)

        # Identity coherence
        with torch.no_grad():
            out_mean = (output.mean(1) if output.dim() == 3 else output).mean(0)
            # Detach and cast to baseline buffer dtype/device to support AMP/mixed-precision
            out_mean_det = out_mean.detach().to(dtype=self._identity_baseline.dtype,
                                                 device=self._identity_baseline.device)
            alpha    = min(1.0, 1.0 / (self._identity_count.item() + 1))
            self._identity_baseline.lerp_(out_mean_det, alpha)
            self._identity_count += 1
            metrics['identity_drift'] = F.mse_loss(out_mean_det, self._identity_baseline).item()

        metrics['n_active']      = len(outputs)
        metrics['stage_modules'] = [n for n, _ in stage_modules]
        return output, metrics

    def get_stage_context(self, stage: int) -> Optional[torch.Tensor]:
        """Retrieve the bridge signal from a completed stage (for cross-stage skip)."""
        sigs = self._stage_signals.get(stage, [])
        if not sigs:
            return None
        return torch.stack(sigs, 0).mean(0)

    # ------------------------------------------------------------------
    # Legacy interface (used by forward_lm)
    # ------------------------------------------------------------------

    def route(self, sem: torch.Tensor,
              modules: Dict[str, Any],
              module_kwargs: Optional[Dict[str, dict]] = None,
             ) -> Tuple[torch.Tensor, dict]:
        """Legacy flat routing interface (backward compatible).

        Routes sem through each named module in registration order
        without stage grouping.
        """
        if self.baseline:
            return sem, {'mode': 'baseline', 'stability': 1.0}

        module_kwargs  = module_kwargs or {}
        outputs        = []
        stability_vals = []

        for name in self.module_names:
            if name not in modules or modules[name] is None:
                continue
            if name not in self.pre_gates:
                continue

            pre  = self.pre_gates[name](sem)
            mod  = modules[name]
            kw   = module_kwargs.get(name, {})
            try:
                out = mod(pre, **kw)
                if isinstance(out, tuple): out = out[0]
                if out.dim() == 3:         out = out.mean(1)
                if out.shape[-1] != self.d_sem: continue
            except Exception:
                continue

            post = self.post_gates[name](out)
            outputs.append(post)
            stability_vals.append(
                (self.pre_gates[name].stability_metrics()['running_rms'] +
                 self.post_gates[name].stability_metrics()['running_rms']) / 2
            )

        if not outputs:
            return sem, {'mode': 'full', 'stability': 0.0, 'n_active': 0}

        stacked = torch.stack(outputs, dim=1)
        fused, _ = self.fusion_attn(
            sem.unsqueeze(1),
            self.fusion_norm(stacked),
            self.fusion_norm(stacked),
            need_weights=False)
        fused  = self.fusion_proj(fused.squeeze(1))
        output = sem + fused

        with torch.no_grad():
            out_mean = output.mean(0)
            alpha    = min(1.0, 1.0 / (self._identity_count.item() + 1))
            self._identity_baseline.lerp_(out_mean.detach(), alpha)
            self._identity_count += 1
            id_drift = F.mse_loss(out_mean.detach(),
                                  self._identity_baseline).item()

        avg_stab = sum(stability_vals) / max(len(stability_vals), 1)
        return output, {
            'mode': 'full',
            'stability': avg_stab,
            'neural_calm': 1.0 / (1.0 + avg_stab),
            'identity_drift': id_drift,
            'n_active': len(outputs),
        }

    # ------------------------------------------------------------------
    # Re-entry: bidirectional loop for IIT Φ compliance
    # ------------------------------------------------------------------

    def get_reentry_bias(self, B: int, device: torch.device) -> torch.Tensor:
        """Return the re-entry signal (B, d_sem) for thalamus injection.

        This is the GWS+PFC output from the PREVIOUS forward call.
        On the very first call _reentry_state is all-zeros (neutral).
        """
        if self.baseline:
            return torch.zeros(B, self.d_sem, device=device)
        # Clone the buffer view before any graph-tracking op: `update_reentry`
        # later does an inplace `lerp_` on the underlying _reentry_state,
        # which would otherwise version-bump the saved-for-backward tensor
        # and raise the "modified by inplace operation" error.
        bias = self._reentry_state.to(device).clone().unsqueeze(0).expand(B, -1)
        mix  = torch.sigmoid(self.reentry_mix)   # ∈ (0, 1) — learnable gate
        return mix * bias

    @torch.no_grad()
    def update_reentry(self, pfc_gws_signal: torch.Tensor) -> None:
        """Store PFC+GWS output for injection into thalamus next step.

        pfc_gws_signal: (B, d_sem) — typically the PFC-selected representation.
        Uses EMA to smooth across the batch and across time steps.
        """
        if self.baseline:
            return
        mean_signal = pfc_gws_signal.detach().mean(0)  # (d_sem,)
        mean_signal = mean_signal.to(dtype=self._reentry_state.dtype,
                                     device=self._reentry_state.device)
        # EMA: α = 0.15 gives ~6-step effective window (100ms at 60fps ≈ thalamo-cortical loop)
        self._reentry_state.lerp_(mean_signal, 0.15)
        self._reentry_count += 1

    def set_gws_broadcast(self, gws_out: torch.Tensor) -> None:
        """Store the GWS stage output for within-pass re-entrant feedback.

        Call this immediately after Stage 5 (GWS) completes.  All subsequent
        route_stage() calls in the same pass will inject this as a backward
        feedback residual into each registered module (GWS → expert).
        """
        if self.baseline:
            return
        mean_out = (gws_out.mean(1) if gws_out.dim() == 3 else gws_out).detach().mean(0)
        mean_out = mean_out.to(dtype=self._gws_broadcast.dtype,
                               device=self._gws_broadcast.device)
        self._gws_broadcast.copy_(mean_out)
        self._gws_broadcast_ready.fill_(True)

    def reset_fast_weights(self) -> None:
        """Clear HFW carry-over states between sequences."""
        self._hfw_states.clear()

    # ------------------------------------------------------------------
    # Integrated Information proxy (Φ)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # IIT 4.0 Φ via Gaussian-MI minimum information partition (MIP).
    # We expose two entry points:
    #   compute_phi_proxy()  → float, no_grad, fast (used for logging + trophic gates)
    #   phi_tensor()         → torch.Tensor, differentiable (used as loss term)
    # Both share the same estimator: cov-of-Gram → Fiedler bisection → MI(A;B).
    # ------------------------------------------------------------------

    def _stack_module_outputs(self, *, detach: bool) -> Optional[torch.Tensor]:
        """Build M ∈ R^{n × d}: one mean-pooled, batch-mean row per module."""
        outputs = self._last_stage_outputs
        if len(outputs) < 2:
            return None
        vecs: List[torch.Tensor] = []
        for o in outputs:
            if o is None:
                continue
            t = o
            if detach:
                t = t.detach()
            t = t.float()
            if t.dim() == 3:
                t = t.mean(1)        # (B, d)
            if t.dim() == 2:
                t = t.mean(0)        # (d,)
            t = t.flatten()
            vecs.append(t)
        if len(vecs) < 2:
            return None
        d = min(int(v.size(0)) for v in vecs)
        d = min(d, 256)
        if d < 2:
            return None
        return torch.stack([v[:d] for v in vecs], dim=0)   # (n, d)

    @staticmethod
    def _phi_from_M(M: torch.Tensor) -> torch.Tensor:
        """Return a non-negative scalar Φ ≈ min over bipartitions of MI(A;B).

        Implementation:
          1. Mean-centre M; compute n×n Gram covariance.
          2. Build similarity W from |Σ| / (σ_i σ_j); zero diagonal.
          3. Normalised Laplacian L = I - D^{-1/2} W D^{-1/2}.
          4. For n ≤ 8: enumerate all 2^(n-1)-1 bipartitions (true MIP).
             For n  > 8: spectral bisection via Fiedler vector of L.
          5. MI(A;B) = ½(logdet Σ_A + logdet Σ_B − logdet Σ_AB), clamped ≥ 0.

        Differentiable: uses torch.linalg.slogdet which has a defined gradient
        for full-rank, symmetric, positive-definite inputs (jittered with eps).
        """
        n, d = M.shape
        device = M.device
        dtype  = M.dtype
        Mc  = M - M.mean(dim=1, keepdim=True)
        cov = (Mc @ Mc.T) / max(d - 1, 1)                  # (n, n)
        eps_n = 1e-6 * torch.eye(n, device=device, dtype=dtype)
        ld_full = torch.linalg.slogdet(cov + eps_n).logabsdet

        if n <= 8:
            best = None
            # Iterate half the partitions (symmetry); skip empty splits.
            for mask_int in range(1, 1 << (n - 1)):
                a_idx = [i for i in range(n) if (mask_int >> i) & 1]
                b_idx = [i for i in range(n) if not (mask_int >> i) & 1]
                if not a_idx or not b_idx:
                    continue
                ai = torch.tensor(a_idx, device=device)
                bi = torch.tensor(b_idx, device=device)
                cov_A = cov.index_select(0, ai).index_select(1, ai)
                cov_B = cov.index_select(0, bi).index_select(1, bi)
                eps_A = 1e-6 * torch.eye(len(a_idx), device=device, dtype=dtype)
                eps_B = 1e-6 * torch.eye(len(b_idx), device=device, dtype=dtype)
                ld_A = torch.linalg.slogdet(cov_A + eps_A).logabsdet
                ld_B = torch.linalg.slogdet(cov_B + eps_B).logabsdet
                mi = 0.5 * (ld_A + ld_B - ld_full)
                mi = torch.clamp(mi, min=0.0)
                best = mi if best is None else torch.minimum(best, mi)
            return best if best is not None else torch.zeros((), device=device, dtype=dtype)

        # n > 8 — spectral bisection (Fiedler) gives a single, differentiable
        # cut. We use slogdet on the resulting bipartition.
        diag = cov.diagonal().clamp(min=1e-8).sqrt()
        W = (cov.abs() / (diag.unsqueeze(1) * diag.unsqueeze(0))).clamp(0.0, 1.0)
        # Zero diagonal without a non-differentiable mutation
        W = W * (1.0 - torch.eye(n, device=device, dtype=dtype))
        deg = W.sum(-1).clamp(min=1e-8)
        D_inv_sqrt = deg.pow(-0.5)
        L = torch.eye(n, device=device, dtype=dtype) \
            - D_inv_sqrt.unsqueeze(1) * W * D_inv_sqrt.unsqueeze(0)
        # eigh works best in float32; cast for stability and back.
        with torch.no_grad():
            eigvecs = torch.linalg.eigh(L.float())[1]
            fiedler = eigvecs[:, 1]
        a_mask = fiedler >= 0
        b_mask = ~a_mask
        if not bool(a_mask.any().item()) or not bool(b_mask.any().item()):
            return torch.zeros((), device=device, dtype=dtype)
        a_idx = a_mask.nonzero(as_tuple=True)[0]
        b_idx = b_mask.nonzero(as_tuple=True)[0]
        cov_A = cov.index_select(0, a_idx).index_select(1, a_idx)
        cov_B = cov.index_select(0, b_idx).index_select(1, b_idx)
        eps_A = 1e-6 * torch.eye(int(a_idx.numel()), device=device, dtype=dtype)
        eps_B = 1e-6 * torch.eye(int(b_idx.numel()), device=device, dtype=dtype)
        ld_A = torch.linalg.slogdet(cov_A + eps_A).logabsdet
        ld_B = torch.linalg.slogdet(cov_B + eps_B).logabsdet
        mi = 0.5 * (ld_A + ld_B - ld_full)
        return torch.clamp(mi, min=0.0)

    def phi_tensor(self,
                   module_outputs: Optional[Dict[str, torch.Tensor]] = None
                  ) -> Optional[torch.Tensor]:
        """Differentiable Φ. Returns None when fewer than 2 module outputs.

        When `module_outputs` is provided, computes Φ directly from those
        (still graph-connected) tensors. This is the path used for the
        training-time Φ objective in brain.forward_lm.

        When called with no argument, falls back to the detached buffer of
        stage outputs — i.e. the same data the no_grad proxy sees, so the
        result is float-equivalent but the gradient will be empty.
        """
        if self.baseline:
            return None
        if module_outputs is not None:
            # Build M from the supplied (live) tensors.
            vecs: List[torch.Tensor] = []
            for v in module_outputs.values():
                if v is None or not torch.is_tensor(v):
                    continue
                t = v.float()
                if t.dim() == 3:
                    t = t.mean(1)
                if t.dim() == 2:
                    t = t.mean(0)
                vecs.append(t.flatten())
            if len(vecs) < 2:
                return None
            d = min(int(v.size(0)) for v in vecs)
            d = min(d, 256)
            if d < 2:
                return None
            M = torch.stack([v[:d] for v in vecs], dim=0)
        else:
            M = self._stack_module_outputs(detach=False)
            if M is None:
                return None
        try:
            return self._phi_from_M(M)
        except Exception:
            return None

    @torch.no_grad()
    def compute_phi_proxy(self) -> float:
        """Real IIT MIP estimate as a python float — for logging / gating.

        Returns 0.0 if the MIP could not be computed (e.g. <2 stages logged
        this pass, or numerical failure).
        """
        if self.baseline:
            return 0.0
        M = self._stack_module_outputs(detach=True)
        if M is None:
            return 0.0
        try:
            phi = self._phi_from_M(M)
            v = float(phi.item())
            if v != v or v == float('inf') or v == float('-inf'):
                return 0.0
            return max(0.0, v)
        except Exception:
            return 0.0

    def record_stage_output(self, signal: torch.Tensor) -> None:
        """Accumulate a *detached* stage output for the next Φ proxy.

        Detached because downstream modules may mutate slices of the same
        tensor in-place; storing a graph-connected reference here would
        poison the backward graph. The differentiable Φ path (`phi_tensor`)
        receives still-live tensors directly from the forward pass via its
        `module_outputs` argument.
        """
        if self.baseline:
            return
        if len(self._last_stage_outputs) >= 16:
            self._last_stage_outputs.pop(0)
        self._last_stage_outputs.append(signal.detach())

    def stability_report(self) -> dict:
        if self.baseline:
            return {'mode': 'baseline'}
        return {
            f'{n}_pre':  self.pre_gates[n].stability_metrics()
            for n in self.module_names if n in self.pre_gates
        } | {
            f'{n}_post': self.post_gates[n].stability_metrics()
            for n in self.module_names if n in self.post_gates
        }
