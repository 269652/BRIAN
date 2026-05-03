"""NeuralOrchestrator — the flow layer that wires all bio modules together.

This is the central routing layer that determines how neural signals flow
between the language cortex and all bio modules. It has two modes:

  - FULL: Routes through all modules with homeostatic pre/post processing
  - BASELINE: Pure passthrough from language cortex (vanilla transformer)

Each module connection has a small transformer block that pre-processes
the incoming signal and post-processes the output, optimized for stable
neural signalling (low qualia variance, high identity coherence).

Scientific basis:
  - Thalamocortical loops: thalamus gates cortical information flow
  - Homeostatic plasticity: neurons maintain stable firing rates
  - Allostatic regulation: predictive regulation of internal state
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HomeostaticGate(nn.Module):
    """A small transformer block + gain controller that maintains stable
    neural signalling between two modules.

    Implements homeostatic plasticity: if the signal is too hot (high variance)
    it dampens; if too cold (low magnitude) it amplifies. The target is
    a calm, stable qualia state — balanced neural activity.

    Args:
        d_model: embedding dimension
        n_heads: attention heads (must divide d_model)
        target_magnitude: desired RMS of output signal
        adaptation_rate: how quickly gain adapts (0=frozen, 1=instant)
    """

    def __init__(self, d_model: int, n_heads: int = 4,
                 target_magnitude: float = 1.0,
                 adaptation_rate: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.target_magnitude = target_magnitude
        self.adaptation_rate = adaptation_rate

        # Small 1-layer transformer for pre/post processing
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.ff_norm = nn.LayerNorm(d_model)

        # Learnable gain + bias for homeostatic control
        self.gain = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

        # Running stats for adaptation (not gradient-trained)
        self.register_buffer('running_mean', torch.zeros(d_model))
        self.register_buffer('running_var', torch.ones(d_model))
        self.register_buffer('n_updates', torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) or (B, D)"""
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True

        # Transformer attention + FF
        h = self.norm(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        h = x + attn_out
        h = h + self.ff(self.ff_norm(h))

        # Homeostatic gain control
        with torch.no_grad():
            batch_mean = h.mean(dim=(0, 1))
            batch_var = h.var(dim=(0, 1))
            alpha = self.adaptation_rate
            self.running_mean.lerp_(batch_mean, alpha)
            self.running_var.lerp_(batch_var, alpha)
            self.n_updates += 1

        # Normalize toward target magnitude
        rms = (self.running_var + 1e-8).sqrt()
        h = (h - self.running_mean) / rms * self.target_magnitude
        h = h * self.gain + self.bias

        if squeeze:
            h = h.squeeze(1)
        return h

    def stability_metrics(self) -> dict:
        """Return metrics about signal stability."""
        return {
            'gain_mean': float(self.gain.mean()),
            'gain_std': float(self.gain.std()),
            'running_rms': float(self.running_var.sqrt().mean()),
            'n_updates': int(self.n_updates.item()),
        }


class ModuleSlot:
    """Describes a module's position in the neural flow."""

    def __init__(self, name: str, module: nn.Module | None,
                 pre_gate: HomeostaticGate, post_gate: HomeostaticGate):
        self.name = name
        self.module = module
        self.pre_gate = pre_gate
        self.post_gate = post_gate


class NeuralOrchestrator(nn.Module):
    """Routes neural signals through bio modules with homeostatic control.

    In FULL mode:
        lang_sem → [pre_gate → module → post_gate] × N_modules → fused output

    In BASELINE mode:
        lang_sem → identity (no bio modules)

    The orchestrator also computes a "neural stability" score that measures
    how calm/stable the overall signal flow is — this can be used as an
    auxiliary training objective.
    """

    def __init__(self, d_sem: int, module_names: list[str],
                 n_heads: int = 4, baseline: bool = False):
        super().__init__()
        self.d_sem = d_sem
        self.baseline = baseline
        self.module_names = module_names

        if not baseline:
            # Pre/post homeostatic gates for each module
            self.pre_gates = nn.ModuleDict()
            self.post_gates = nn.ModuleDict()
            for name in module_names:
                n_h = max(1, min(n_heads, d_sem // 16))  # ensure valid
                # Make n_heads divide d_sem
                while d_sem % n_h != 0 and n_h > 1:
                    n_h -= 1
                self.pre_gates[name] = HomeostaticGate(d_sem, n_h)
                self.post_gates[name] = HomeostaticGate(d_sem, n_h)

            # Fusion: combine all module outputs into single representation
            self.fusion_norm = nn.LayerNorm(d_sem)
            self.fusion_attn = nn.MultiheadAttention(
                d_sem, max(1, min(n_heads, d_sem // 16)), batch_first=True)
            self.fusion_proj = nn.Linear(d_sem, d_sem)

            # Identity coherence: tracks how consistent the signal is
            self.register_buffer('_identity_baseline', torch.zeros(d_sem))
            self.register_buffer('_identity_count', torch.zeros(1))

    def route(self, sem: torch.Tensor,
              modules: dict[str, nn.Module],
              module_kwargs: dict[str, dict] | None = None,
              ) -> tuple[torch.Tensor, dict]:
        """Route semantic embedding through modules.

        Args:
            sem: (B, D) semantic embedding from language cortex
            modules: dict mapping module name → nn.Module (callable)
            module_kwargs: optional kwargs per module

        Returns:
            output: (B, D) fused output
            metrics: dict of stability/coherence metrics
        """
        if self.baseline:
            return sem, {'mode': 'baseline', 'stability': 1.0}

        module_kwargs = module_kwargs or {}
        outputs = []
        stability_scores = []

        for name in self.module_names:
            if name not in modules or modules[name] is None:
                continue

            # Pre-process: homeostatic gate conditions the input
            pre = self.pre_gates[name](sem)

            # Route through the module
            mod = modules[name]
            kwargs = module_kwargs.get(name, {})
            try:
                out = mod(pre, **kwargs)
                # Handle modules that return tuples
                if isinstance(out, tuple):
                    out = out[0]
                # Ensure output is (B, D)
                if out.dim() == 3:
                    out = out.mean(dim=1)
                if out.shape[-1] != self.d_sem:
                    # Dimension mismatch — skip this module
                    continue
            except Exception:
                continue

            # Post-process: homeostatic gate stabilizes output
            post = self.post_gates[name](out)
            outputs.append(post)

            # Track stability
            pre_stability = self.pre_gates[name].stability_metrics()
            post_stability = self.post_gates[name].stability_metrics()
            stability_scores.append(
                (pre_stability['running_rms'] + post_stability['running_rms']) / 2
            )

        if not outputs:
            return sem, {'mode': 'full', 'stability': 0.0, 'n_active': 0}

        # Fuse: stack module outputs and attend over them
        stacked = torch.stack(outputs, dim=1)  # (B, N_modules, D)
        fused = self.fusion_norm(stacked)
        fused, _ = self.fusion_attn(
            sem.unsqueeze(1), fused, fused, need_weights=False)
        fused = self.fusion_proj(fused.squeeze(1))

        # Residual connection: output = sem + fused_module_signal
        output = sem + fused

        # Identity coherence tracking
        with torch.no_grad():
            output_mean = output.mean(0)
            alpha = min(1.0, 1.0 / (self._identity_count.item() + 1))
            self._identity_baseline.lerp_(output_mean, alpha)
            self._identity_count += 1
            identity_drift = F.mse_loss(output_mean, self._identity_baseline).item()

        # Compute overall stability
        avg_stability = sum(stability_scores) / max(len(stability_scores), 1)
        neural_calm = 1.0 / (1.0 + avg_stability)  # lower RMS = calmer

        metrics = {
            'mode': 'full',
            'stability': avg_stability,
            'neural_calm': neural_calm,
            'identity_drift': identity_drift,
            'n_active': len(outputs),
        }
        return output, metrics

    def stability_report(self) -> dict:
        """Full stability report across all gates."""
        if self.baseline:
            return {'mode': 'baseline'}
        report = {}
        for name in self.module_names:
            if name in self.pre_gates:
                report[f'{name}_pre'] = self.pre_gates[name].stability_metrics()
                report[f'{name}_post'] = self.post_gates[name].stability_metrics()
        return report
