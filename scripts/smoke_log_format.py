"""Show the operator-facing training log line at calm vs. runaway moments.

The signature is:
    _format_metrics_line(step, avg_loss, avg_lm, gnorm, lr, tok_per_s, metrics)
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))

from neuroslm.train_dsl import _format_metrics_line

# Snapshot from PHASE 3 at step 200 (chronic stress, dampers ENGAGED)
metrics_at_runaway = {
    "lm": 12.40,
    "allostasis_load": 0.572,
    "allostasis_cort": 0.555,
    "allostasis_ne_mult": 0.611,
    "allostasis_trophic_mult": 0.445,
    "allostasis_lr_mult": 0.722,
    "ne_level": 0.93,
    "gaba_level": 0.60,
}
print("LOG LINE at runaway moment (step 520, dampers engaged):")
print(_format_metrics_line(
    step=520,
    avg_loss=14.85,
    avg_lm=12.40,
    gnorm=24.0,
    lr=8.0e-5,
    tok_per_s=4200.0,
    metrics=metrics_at_runaway,
))
print()

# Calm baseline snapshot
metrics_calm = {
    "lm": 4.32,
    "allostasis_load": 0.05,
    "allostasis_cort": 0.005,
    "allostasis_ne_mult": 0.996,
    "allostasis_trophic_mult": 0.994,
    "allostasis_lr_mult": 0.997,
    "ne_level": 0.20,
    "gaba_level": 0.13,
}
print("LOG LINE during calm training (step 300, dampers off):")
print(_format_metrics_line(
    step=300,
    avg_loss=4.32,
    avg_lm=4.32,
    gnorm=2.1,
    lr=2.0e-4,
    tok_per_s=4600.0,
    metrics=metrics_calm,
))
