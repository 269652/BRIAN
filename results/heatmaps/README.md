# results/heatmaps/

Incremental training-heatmap JSON artifacts produced by
`neuroslm.evolution.harness_hook.HeatmapHook` during long training runs.

Each file is named `<arch>.heatmap.json` and is:

- **Tracked** in git (intentionally, so the HeatmapPublisher's
  `git commit/push` cadence has a stable target on vast/Colab).
- Updated incrementally every `heatmap_update_every_n` steps.
- Committed + pushed every `heatmap_commit_every_n` steps when the
  publisher is enabled.

Format: `TrainingHeatmap.to_dict()` round-trips through
`TrainingHeatmap.from_dict()` — see `neuroslm/evolution/heatmap.py`.
