"""Contract tests for the three metric-observability fixes shipped on
2026-06-18 to answer the user's three questions about the Lightning
deploy log::

    "Why is OOD PPL missing every five hundred steps and why are all
    OOD interventions off? Is NFO even activated? Please make sure
    the full metrics appear and make it into the log. Also set
    checkpoint saving to every 2.5k"

Each test pins ONE behaviour so a regression is immediately obvious.

Group A — ProjectConfig surfaces ``default_ood_every``
    A1. Default = 0 (no opinion when brian.toml is silent → matches
        the legacy train_dsl.py default of 0).
    A2. ``[defaults].ood_every = 500`` parses to 500.
    A3. ``BRIAN_DEFAULT_OOD_EVERY`` env var wins over the file.
    A4. ``BRIAN_DEFAULT_OOD_EVERY="0"`` explicitly disables the probe.

Group B — ``cmd_deploy`` propagates ``cfg.arch`` to ``config.arch``
    B1. When no ``--arch`` and no ``--dna``, ``config.arch`` gets set
        to ``cfg.arch`` from brian.toml. This is the bug that made
        the trainer use the stale ``architectures/current`` folder
        instead of the configured ``architectures/SmolLM``.
    B2. CLI ``--arch`` still wins over brian.toml.
    B3. DNA mode skips arch propagation (workspace.arch_root takes
        precedence later in the flow).

Group C — ``cmd_deploy`` propagates ``cfg.default_ood_every``
    C1. CLI ``--ood N`` wins.
    C2. When CLI flag is absent / 0, ``cfg.default_ood_every`` is used.
    C3. Both 0 → ``config.ood_every == 0`` (probe disabled).

Group D — NFO telemetry surfaces in the train log
    D1. ``_collect_nfo_metrics`` returns {} for a harness without
        NFO (no NeuralFieldOscillator in the module tree).
    D2. ``_collect_nfo_metrics`` returns a populated dict with
        ``nfo_R_mean``/``nfo_kappa``/``nfo_phi_kappa`` keys after
        a forward pass through an NFO block.
    D3. ``_format_metrics_line`` includes the ``nfo[...]`` segment
        when nfo_* keys are in the metrics dict.
    D4. ``_format_metrics_line`` omits the segment when no nfo_*
        key is present (legacy archs see no change).

Group E — brian.toml checkpoint cadence
    E1. ``save_every == 2500`` (per the user's request).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neuroslm.project_config import (
    ProjectConfig,
    _DEFAULT_OOD_EVERY,
    load_project_config,
)
from neuroslm.train_dsl import (
    _collect_nfo_metrics,
    _format_metrics_line,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ──────────────────────────────────────────────────────────────────
# Group A — ProjectConfig.default_ood_every
# ──────────────────────────────────────────────────────────────────

class TestProjectConfigOodEvery:
    """``[defaults].ood_every`` → ``cfg.default_ood_every`` plumbing."""

    def _write_brian_toml(self, tmp_path: Path, body: str) -> Path:
        (tmp_path / "brian.toml").write_text(body, encoding="utf-8")
        return tmp_path

    def test_A1_default_is_zero(self, tmp_path, monkeypatch):
        """Empty file → default 0 (matches train_dsl.py argparse default)."""
        monkeypatch.delenv("BRIAN_DEFAULT_OOD_EVERY", raising=False)
        root = self._write_brian_toml(tmp_path, "")
        cfg = load_project_config(start=root)
        assert cfg.default_ood_every == 0

    def test_A2_brian_toml_overrides_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BRIAN_DEFAULT_OOD_EVERY", raising=False)
        root = self._write_brian_toml(
            tmp_path, "[defaults]\nood_every = 500\n",
        )
        cfg = load_project_config(start=root)
        assert cfg.default_ood_every == 500

    def test_A3_env_var_wins_over_brian_toml(self, tmp_path, monkeypatch):
        root = self._write_brian_toml(
            tmp_path, "[defaults]\nood_every = 500\n",
        )
        monkeypatch.setenv("BRIAN_DEFAULT_OOD_EVERY", "1000")
        cfg = load_project_config(start=root)
        assert cfg.default_ood_every == 1000

    def test_A4_env_var_zero_disables_probe(self, tmp_path, monkeypatch):
        """Explicit "0" must turn the probe off — not be treated as
        "no opinion" the way an empty string is."""
        root = self._write_brian_toml(
            tmp_path, "[defaults]\nood_every = 500\n",
        )
        monkeypatch.setenv("BRIAN_DEFAULT_OOD_EVERY", "0")
        cfg = load_project_config(start=root)
        assert cfg.default_ood_every == 0

    def test_A5_default_constant_export(self):
        """The module-level constant must stay 0 (changing it would
        flip every legacy run that doesn't have a brian.toml)."""
        assert _DEFAULT_OOD_EVERY == 0


# ──────────────────────────────────────────────────────────────────
# Group B / C — cmd_deploy propagation (arch + ood_every)
# ──────────────────────────────────────────────────────────────────

class _DeployFixture:
    """Minimal scaffolding to call ``cmd_deploy`` with a stub connector.

    Avoids the full pytest fixture machinery so each test reads
    bottom-up — the connector stub records the ``DeployConfig`` it
    received and that's what every assertion inspects.
    """

    @staticmethod
    def make_args(**overrides):
        from argparse import Namespace
        base = dict(
            platform=None, steps=None, branch=None, scale=None,
            label=None, dna=None, arch=None, ood=0, machine=None,
            teamspace=None, resume=None, latest=False,
        )
        base.update(overrides)
        return Namespace(**base)

    @staticmethod
    def run(cfg, args, *, captured: list):
        """Patch ``get_connector`` + ``load_project_config`` and invoke
        ``cmd_deploy``. The connector stub appends each launched
        ``DeployConfig`` to *captured*.

        ``cmd_deploy`` imports ``load_project_config`` lazily inside
        the function body, so the patch target is the source module
        (``neuroslm.project_config``) — patching the cli module would
        miss the late import."""
        from neuroslm import cli as cli_module
        connector = MagicMock()
        connector.launch = MagicMock(side_effect=lambda c: captured.append(c) or 0)
        with patch(
            "neuroslm.project_config.load_project_config",
            return_value=cfg,
        ):
            with patch(
                "neuroslm.connectors.get_connector",
                return_value=connector,
            ):
                rc = cli_module.cmd_deploy(args)
        return rc


def _make_cfg(**overrides):
    """ProjectConfig with sane defaults for the deploy-flow tests."""
    base = dict(
        repo_root=REPO_ROOT,
        arch="architectures/SmolLM",
        dna="",
        default_steps=1000,
        default_log_every=20,
        default_save_every=500,
        default_push_every=500,
        default_ood_every=0,
        default_platform="lightning",
        default_machine="",
        default_teamspace="",
    )
    base.update(overrides)
    return ProjectConfig(**base)


class TestCmdDeployPropagatesArch:
    """Deploy → connector contract for ``config.arch``."""

    def test_B1_no_cli_no_dna_uses_brian_toml_arch(self):
        """The bug this fixes: previously ``config.arch`` was left
        unset and the Lightning connector fell back to
        ``architectures/current`` (a stale folder), silently
        ignoring brian.toml's ``[current].arch``."""
        cfg = _make_cfg(arch="architectures/SmolLM")
        args = _DeployFixture.make_args()
        captured: list = []
        rc = _DeployFixture.run(cfg, args, captured=captured)
        assert rc == 0
        assert len(captured) == 1
        assert captured[0].arch == "architectures/SmolLM"

    def test_B2_cli_arch_wins_over_brian_toml(self, tmp_path):
        """``brian deploy --arch architectures/other`` must win."""
        other = tmp_path / "architectures" / "other"
        other.mkdir(parents=True)
        cfg = _make_cfg(arch="architectures/SmolLM")
        args = _DeployFixture.make_args(arch=str(other))
        captured: list = []
        _DeployFixture.run(cfg, args, captured=captured)
        assert captured[0].arch == str(other)

    def test_B3_dna_mode_skips_brian_toml_arch(self, tmp_path, monkeypatch):
        """When DNA mode is active, ``prepare_run_workspace`` sets
        ``config.arch`` to the freshly-compiled workspace path. The
        brian.toml ``[current].arch`` propagation MUST NOT clobber
        that — DNA mode wins."""
        cfg = _make_cfg(arch="architectures/SmolLM")
        args = _DeployFixture.make_args(dna="some/path.dna")

        # Stub the workspace prep to return a deterministic arch path.
        workspace = MagicMock()
        workspace.arch_root = tmp_path / "compiled_workspace"
        workspace.source_kind = "dna"
        workspace.source_path = "some/path.dna"
        workspace.hypergraph_ir.nodes = []
        workspace.hypergraph_ir.hyperedges = []

        captured: list = []
        with patch(
            "neuroslm.compiler.run_workspace.prepare_run_workspace",
            return_value=workspace,
        ):
            _DeployFixture.run(cfg, args, captured=captured)
        # DNA workspace path wins — NOT the brian.toml arch.
        assert captured[0].arch == str(workspace.arch_root)


class TestCmdDeployPropagatesOodEvery:
    """Deploy → connector contract for ``config.ood_every``."""

    def test_C1_cli_ood_wins(self):
        cfg = _make_cfg(default_ood_every=500)
        args = _DeployFixture.make_args(ood=2000)
        captured: list = []
        _DeployFixture.run(cfg, args, captured=captured)
        assert captured[0].ood_every == 2000

    def test_C2_brian_toml_ood_every_used_when_no_cli(self):
        """Without ``--ood`` on the CLI, ``brian.toml [defaults]
        .ood_every`` must flow through. This is the fix for the
        user's question "why is OOD PPL missing every 500 steps?"
        — it was missing because nothing wired the cadence."""
        cfg = _make_cfg(default_ood_every=500)
        args = _DeployFixture.make_args(ood=0)
        captured: list = []
        _DeployFixture.run(cfg, args, captured=captured)
        assert captured[0].ood_every == 500

    def test_C3_both_zero_means_probe_disabled(self):
        cfg = _make_cfg(default_ood_every=0)
        args = _DeployFixture.make_args(ood=0)
        captured: list = []
        _DeployFixture.run(cfg, args, captured=captured)
        assert captured[0].ood_every == 0


# ──────────────────────────────────────────────────────────────────
# Group D — NFO telemetry visible in the per-step log
# ──────────────────────────────────────────────────────────────────

class TestNfoMetricCollector:
    """``_collect_nfo_metrics`` walks the harness for NFO blocks."""

    def test_D1_no_nfo_block_returns_empty_dict(self):
        """A harness with no NeuralFieldOscillator returns {} — no
        legacy log format changes."""
        import torch.nn as nn
        # Plain linear module — no NFO anywhere.
        harness = nn.Linear(4, 4)
        out = _collect_nfo_metrics(harness)
        assert out == {}

    def test_D2_nfo_block_with_state_returns_scalar_keys(self):
        """After a forward pass the NFO block populates ``last_state``;
        the collector flattens that to ``nfo_*`` scalar keys."""
        import torch
        from neuroslm.modules.neural_field_oscillator import (
            NeuralFieldOscillator, NFOConfig,
        )
        d_model = 16
        block = NeuralFieldOscillator(
            d_model=d_model,
            cfg=NFOConfig(n_osc=4, expose_phi_lower_bound=True),
        )
        x = torch.randn(2, 8, d_model)
        with torch.no_grad():
            _ = block(x)

        out = _collect_nfo_metrics(block)
        # The block exposes at minimum R, A, κ, dt — all in last_state.
        assert "nfo_R_mean" in out
        assert "nfo_kappa" in out
        # Per the H016 bipartition expose_phi_lower_bound=True flag,
        # the block must publish Φκ as well.
        assert "nfo_phi_kappa" in out
        # All returned values must be plain Python floats (no tensors
        # leaking through — that would break JSON serialisation in
        # the log-pusher downstream).
        for k, v in out.items():
            assert isinstance(v, float), f"{k} → {type(v).__name__}"

    def test_D3_collector_aggregates_multiple_blocks(self):
        """A model with two NFO blocks (e.g. two cortex columns)
        averages the scalar metrics so the log line stays single-row."""
        import torch
        import torch.nn as nn
        from neuroslm.modules.neural_field_oscillator import (
            NeuralFieldOscillator, NFOConfig,
        )
        d_model = 16
        cfg = NFOConfig(n_osc=4)

        class TwoColumn(nn.Module):
            def __init__(self):
                super().__init__()
                self.nfo_a = NeuralFieldOscillator(d_model=d_model, cfg=cfg)
                self.nfo_b = NeuralFieldOscillator(d_model=d_model, cfg=cfg)

            def forward(self, x):
                return self.nfo_a(x) + self.nfo_b(x)

        m = TwoColumn()
        x = torch.randn(1, 4, d_model)
        with torch.no_grad():
            _ = m(x)

        out = _collect_nfo_metrics(m)
        # Both blocks populated state → aggregated mean is finite.
        assert "nfo_R_mean" in out
        import math
        assert math.isfinite(out["nfo_R_mean"])


class TestNfoFormatterAppearsInLog:
    """``_format_metrics_line`` emits the ``nfo[...]`` segment when
    any ``nfo_*`` key is in the metrics dict."""

    def _base_metrics(self):
        # Minimum keys ``_format_metrics_line`` reads (no required
        # ones — every block guards with .get(k) — but supply
        # realistic values so the line looks like a real log row).
        return {
            "phi": 0.9,
            "fidelity_lambda1": 0.14,
        }

    def test_D4_nfo_block_appears_when_keys_present(self):
        m = self._base_metrics()
        m.update({
            "nfo_R_mean": 0.42,
            "nfo_R_max": 0.78,
            "nfo_kappa": 0.32,
            "nfo_alpha": 0.08,
            "nfo_phi_kappa": 0.22,
        })
        line = _format_metrics_line(
            step=100, avg_loss=6.5, avg_lm=6.3,
            gnorm=2.0, lr=1e-4, tok_per_s=650.0, metrics=m,
        )
        assert "nfo[" in line
        assert "R=0.42" in line
        assert "R★=0.78" in line
        assert "κ=0.32" in line
        assert "α=0.080" in line
        assert "Φκ=0.22" in line

    def test_D5_nfo_block_omitted_when_no_keys(self):
        """Legacy arches without NFO see ZERO change to the log format."""
        line = _format_metrics_line(
            step=100, avg_loss=6.5, avg_lm=6.3,
            gnorm=2.0, lr=1e-4, tok_per_s=650.0,
            metrics=self._base_metrics(),
        )
        assert "nfo[" not in line, (
            "Empty NFO segment leaked into the log — legacy arches "
            "would see a spurious `nfo[]` block.")


# ──────────────────────────────────────────────────────────────────
# Group E — checkpoint cadence
# ──────────────────────────────────────────────────────────────────

class TestBrianTomlCheckpointCadence:
    """brian.toml at repo root must reflect the user's chosen cadence."""

    def test_E1_save_every_is_2500(self):
        """User asked for "checkpoint saving every 2.5k" on 2026-06-18."""
        cfg = load_project_config()
        assert cfg.default_save_every == 2500, (
            f"brian.toml [defaults].save_every must be 2500 "
            f"(got {cfg.default_save_every}). Update brian.toml.")

    def test_E2_ood_every_is_500(self):
        """brian.toml ships with ood_every=500 so deploys get a
        real WikiText probe every 500 steps."""
        cfg = load_project_config()
        assert cfg.default_ood_every == 500, (
            f"brian.toml [defaults].ood_every must be 500 "
            f"(got {cfg.default_ood_every}). Update brian.toml.")


# ──────────────────────────────────────────────────────────────────
# Group F — `brian ps` Lightning table surfaces OOD-PPL
# ──────────────────────────────────────────────────────────────────
#
# Follow-up to the user's question on 2026-06-18 16:39:
#   "why is brian ps not showing ood ppl?"
#
# Root cause: `_render_lightning_section` only parsed `_STEP_RE`
# (training ppl), never `_MID_OOD_RE`. The vast.ai render had the
# OOD-PPL column, but Lightning didn't. Now `_summarise_lightning_tail`
# returns `(step, ppl, ood, last_line)` and the table includes the
# column between PPL and LAST.

class TestLightningPsOodPplColumn:
    """`brian ps` Lightning table must surface `[mid-ood]` ppl."""

    def test_F1_empty_tail_returns_dashes_for_ood(self):
        """No log content → OOD column shows '-' (not a crash)."""
        from neuroslm.cli import _summarise_lightning_tail
        step, ppl, ood, last = _summarise_lightning_tail("")
        assert (step, ppl, ood, last) == ("-", "-", "-", "")

    def test_F2_tail_with_only_step_lines_has_dash_ood(self):
        """Training has started but OOD eval hasn't fired yet —
        STEP/PPL populated, OOD still '-'."""
        from neuroslm.cli import _summarise_lightning_tail
        tail = (
            "step    20 | loss 6.98 | lm 6.32 | ppl 559.0 | gnorm 6.1 "
            "| lr 1.00e-05 | 594 tok/s | other stuff\n"
            "step    40 | loss 6.89 | lm 6.41 | ppl 612.4 | gnorm 5.7 "
            "| lr 2.00e-05 | 643 tok/s | other stuff\n"
        )
        step, ppl, ood, last = _summarise_lightning_tail(tail)
        assert step == "40", f"latest step should be 40, got {step!r}"
        assert ppl == "612.4", f"latest ppl should be 612.4, got {ppl!r}"
        assert ood == "-", (
            f"no [mid-ood] line in tail → OOD column must be '-', "
            f"got {ood!r}")

    def test_F3_tail_with_mid_ood_populates_column(self):
        """When `[mid-ood] step N: wikitext ppl=X` is in the tail,
        OOD column shows '<ppl>@<step>'."""
        from neuroslm.cli import _summarise_lightning_tail
        tail = (
            "step   480 | loss 6.5 | lm 6.0 | ppl 403.4 | gnorm 4.1 "
            "| lr 1.00e-04 | 700 tok/s | other\n"
            "[mid-ood] step 500: wikitext ppl=1550.1 gap_ratio=3.84 "
            "(train_ppl=403.4) heldout=64 batches\n"
            "step   500 | loss 6.4 | lm 6.0 | ppl 398.2 | gnorm 4.0 "
            "| lr 1.00e-04 | 695 tok/s | other\n"
        )
        step, ppl, ood, last = _summarise_lightning_tail(tail)
        assert step == "500"
        assert ppl == "398.2"
        assert ood == "1550@500", (
            f"OOD column must show '1550@500', got {ood!r}")

    def test_F4_latest_mid_ood_wins_over_earlier(self):
        """Multiple [mid-ood] blocks → table shows the most recent."""
        from neuroslm.cli import _summarise_lightning_tail
        tail = (
            "[mid-ood] step 500: wikitext ppl=1550.1 heldout=64 batches\n"
            "step   600 | loss 6.3 | lm 5.9 | ppl 365.0 | gnorm 3.8 "
            "| lr 1.00e-04 | 700 tok/s | other\n"
            "[mid-ood] step 1000: wikitext ppl=1280.5 heldout=64 batches\n"
            "step  1020 | loss 6.2 | lm 5.8 | ppl 330.0 | gnorm 3.6 "
            "| lr 1.00e-04 | 700 tok/s | other\n"
        )
        _, _, ood, _ = _summarise_lightning_tail(tail)
        assert ood == "1280@1000", (
            f"latest [mid-ood] (step 1000) must win — got {ood!r}")

    def test_F5_render_uses_n_500_tail(self):
        """The render loop must request n=500 lines (not the old n=20)
        so the parser actually catches [mid-ood] lines that sit ~25
        step-lines back when log_every=20."""
        import inspect
        from neuroslm import cli as cli_module
        src = inspect.getsource(cli_module._render_lightning_section)
        assert "tail_logs(j.job_id, n=500)" in src, (
            "Lightning render must tail 500 lines so [mid-ood] blocks "
            "(which fire every ood_every=500 steps with log_every=20, "
            "i.e. ~25 step lines apart) actually appear in the tail.")

    def test_F6_header_contains_ood_ppl_column(self):
        """The Lightning ps header must include an OOD-PPL column."""
        import inspect
        from neuroslm import cli as cli_module
        src = inspect.getsource(cli_module._render_lightning_section)
        assert "'OOD-PPL'" in src, (
            "Lightning ps header must include the OOD-PPL column.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
