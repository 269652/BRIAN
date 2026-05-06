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

        h = self.norm(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        h = x + attn_out
        h = h + self.ff(self.ff_norm(h))

        with torch.no_grad():
            bm = h.mean((0, 1))
            bv = h.var((0, 1))
            # Ensure stats have the same dtype and device as the running buffers
            bm = bm.to(dtype=self.running_mean.dtype, device=self.running_mean.device)
            bv = bv.to(dtype=self.running_var.dtype, device=self.running_var.device)
            a  = self.adaptation_rate
            self.running_mean.lerp_(bm, a)
            self.running_var.lerp_(bv, a)
            self.n_updates += 1

        rms = (self.running_var + 1e-8).sqrt()
        h   = (h - self.running_mean) / rms * self.target_magnitude
        h   = h * self.gain + self.bias

        return h.squeeze(1) if squeeze else h

    def stability_metrics(self) -> dict:
        return {
            'gain_mean':   float(self.gain.detach().mean()),
            'gain_std':    float(self.gain.detach().std()),
            'running_rms': float(self.running_var.sqrt().mean()),
            'n_updates':   int(self.n_updates.item()),
        }


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
        """Reset per-pass stage signal accumulator."""
        self._stage_signals.clear()

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
                outputs.append(post_out)

            except Exception:
                continue

        if not outputs:
            return signal, metrics

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
