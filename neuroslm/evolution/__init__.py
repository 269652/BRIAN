# -*- coding: utf-8 -*-
"""Evolution subsystem: training heatmap, hot-path identification, and the
epigenetic mutation pipeline (heatmap -> propose -> gate -> Lean proof).
"""
from neuroslm.evolution.heatmap import TrainingHeatmap, HeatEntry
from neuroslm.evolution.publisher import HeatmapPublisher
from neuroslm.evolution.grad_heat import (
    parameter_grad_norms, signals_from_grad_norms, update_heatmap,
)

__all__ = [
    "TrainingHeatmap", "HeatEntry", "HeatmapPublisher",
    "parameter_grad_norms", "signals_from_grad_norms", "update_heatmap",
]
