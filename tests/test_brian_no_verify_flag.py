"""Contracts for the global ``--no-verify`` flag on ``brian``.

Like ``git commit --no-verify``, the flag short-circuits any pre-flight
hook that would otherwise run before a CLI command. Today only
``brian deploy`` calls a hook (``pre-deploy``), but the flag is
parsed at the top level so any future hook-calling command picks it
up uniformly.

Semantics pinned here:

  * ``brian --no-verify deploy ...``  -> skip the pre-deploy hook
  * ``brian deploy --no-verify ...``  -> same (positional flexibility,
                                          since argparse honors the
                                          parent parser's flags before
                                          OR after the subcommand)
  * ``brian deploy ...``              -> hook still runs (default)
  * The flag must be IDEMPOTENT (passing twice is fine)
  * The flag must be VISIBLE in ``brian --help`` AND ``brian deploy --help``
  * The flag must NOT change ANY other behaviour — only the hook is
    skipped; the deploy itself still runs.

Without this contract, agents and humans hitting an over-eager
pre-deploy hook (e.g. when the working tree is intentionally dirty for
a quick experiment) had to either disable the hook globally or edit
the YAML — both of which are footguns. ``--no-verify`` is the
narrowly-scoped, per-invocation escape hatch.
"""
from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest


# ──────────────────────────────────────────────────────────────────────
# Parser surface
# ──────────────────────────────────────────────────────────────────────


class TestNoVerifyFlagParser:
    def test_top_level_no_verify_flag_exists(self):
        """``brian --no-verify deploy ...`` must parse without error."""
        from neuroslm.cli import _build_parser

        parser = _build_parser()
        ns = parser.parse_args(["--no-verify", "deploy"])
        assert getattr(ns, "no_verify", None) is True, (
            "expected top-level --no-verify to set ns.no_verify=True; "
            f"got namespace={ns}"
        )

    def test_no_verify_defaults_to_false(self):
        """If the user doesn't pass ``--no-verify``, the attribute must
        still be present and set to ``False`` so call sites can do
        ``if not args.no_verify:`` without ``hasattr`` guards."""
        from neuroslm.cli import _build_parser

        parser = _build_parser()
        ns = parser.parse_args(["deploy"])
        assert hasattr(ns, "no_verify"), (
            "ns.no_verify must always exist (default False) so call "
            "sites don't need hasattr guards"
        )
        assert ns.no_verify is False

    def test_no_verify_after_subcommand_also_works(self):
        """argparse normally rejects parent flags after the subcommand
        name, but ``--no-verify`` should be tolerated in BOTH positions
        (matching ``git commit --no-verify`` ergonomics)."""
        from neuroslm.cli import _build_parser

        parser = _build_parser()
        ns = parser.parse_args(["deploy", "--no-verify"])
        assert ns.no_verify is True, (
            "expected `brian deploy --no-verify` to set "
            "ns.no_verify=True; got namespace={ns}"
        )

    def test_no_verify_is_idempotent(self):
        """Passing the flag twice must NOT raise."""
        from neuroslm.cli import _build_parser

        parser = _build_parser()
        ns = parser.parse_args(["--no-verify", "deploy", "--no-verify"])
        assert ns.no_verify is True

    def test_no_verify_appears_in_top_level_help(self):
        """``brian --help`` must advertise the flag so users can
        discover it without reading the source."""
        from neuroslm.cli import _build_parser

        parser = _build_parser()
        helptext = parser.format_help()
        assert "--no-verify" in helptext, (
            f"top-level --help missing --no-verify; got:\n{helptext}"
        )


# ──────────────────────────────────────────────────────────────────────
# Behavioural contract — cmd_deploy must skip the hook when set
# ──────────────────────────────────────────────────────────────────────


class TestDeploySkipsHookWhenNoVerify:
    def test_deploy_calls_hook_by_default(self):
        """Sanity baseline: with NO ``--no-verify``, the pre-deploy
        hook IS called."""
        from neuroslm import cli as cli_mod

        # Stub everything past the hook so the deploy doesn't actually
        # provision a vast box during a unit test.
        with patch.object(cli_mod, "_run_hook", return_value=0) as hook_mock, \
             patch.object(cli_mod, "_run", return_value=0):
            # Build the simplest possible Namespace cmd_deploy will accept.
            ns = argparse.Namespace(
                steps=None, branch=None, scale=None, dna=None,
                label=None, ood=None,
                no_verify=False,
            )
            try:
                cli_mod.cmd_deploy(ns)
            except SystemExit:
                pass
            except Exception:
                # Other failures (e.g. brian.toml missing) are fine —
                # we only care that the hook WAS called before they
                # surfaced.
                pass
        assert hook_mock.called, (
            "cmd_deploy with no_verify=False must invoke _run_hook"
        )

    def test_deploy_skips_hook_when_no_verify_true(self):
        """``brian --no-verify deploy ...`` must NOT call _run_hook."""
        from neuroslm import cli as cli_mod

        with patch.object(cli_mod, "_run_hook", return_value=0) as hook_mock, \
             patch.object(cli_mod, "_run", return_value=0):
            ns = argparse.Namespace(
                steps=None, branch=None, scale=None, dna=None,
                label=None, ood=None,
                no_verify=True,
            )
            try:
                cli_mod.cmd_deploy(ns)
            except SystemExit:
                pass
            except Exception:
                # Same indulgence as above — we only care about the hook.
                pass
        assert not hook_mock.called, (
            "cmd_deploy with no_verify=True must NOT invoke _run_hook "
            "(the entire point of --no-verify is to skip hooks)"
        )

    def test_deploy_emits_skip_notice_when_no_verify(self, capsys):
        """When ``--no-verify`` is set the user must see an explicit
        notice that the hook was skipped — silent omission would be a
        usability footgun ('did the hook run?')."""
        from neuroslm import cli as cli_mod

        with patch.object(cli_mod, "_run_hook", return_value=0), \
             patch.object(cli_mod, "_run", return_value=0):
            ns = argparse.Namespace(
                steps=None, branch=None, scale=None, dna=None,
                label=None, ood=None,
                no_verify=True,
            )
            try:
                cli_mod.cmd_deploy(ns)
            except SystemExit:
                pass
            except Exception:
                pass
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        # The notice must mention both 'pre-deploy' and 'skip' so it's
        # grep-able and unambiguous.
        assert "pre-deploy" in combined.lower() and "skip" in combined.lower(), (
            "cmd_deploy with --no-verify must print a notice that the "
            "pre-deploy hook was skipped; got:\n"
            f"--- stdout ---\n{captured.out}\n--- stderr ---\n{captured.err}"
        )


# ──────────────────────────────────────────────────────────────────────
# End-to-end argv routing (CLI -> Namespace -> dispatcher)
# ──────────────────────────────────────────────────────────────────────


class TestDeployYoloAlias:
    """``brian deploy yolo`` is sugar for ``brian deploy --no-verify``."""

    def test_yolo_parses_as_arch_yolo(self):
        """Argparse gives us arch='yolo'; cmd_deploy must intercept it."""
        from neuroslm.cli import _build_parser

        parser = _build_parser()
        ns = parser.parse_args(["deploy", "yolo"])
        assert ns.arch == "yolo"

    def _fake_cfg(self):
        """Return a minimal project config mock that prevents real I/O."""
        from unittest.mock import MagicMock
        cfg = MagicMock()
        cfg.default_platform = "vast"
        cfg.default_steps = 0
        cfg.default_branch = None
        cfg.is_dna_mode = False
        cfg.dna = None
        cfg.arch = None
        cfg.default_ood_every = 0
        cfg.default_log_every = 100
        cfg.default_save_every = 1000
        cfg.default_push_every = 1000
        cfg.default_push_backend = "hf"
        cfg.default_hf_repo_id = "moritzroessler/BRIAN"
        cfg.default_push_optimizer = False
        cfg.default_machine = None
        cfg.default_teamspace = None
        return cfg

    def _yolo_ns(self, **overrides) -> argparse.Namespace:
        defaults = dict(
            arch="yolo", steps=None, branch=None, scale=None,
            dna=None, label=None, ood=None, no_verify=False,
            resume=None, latest=False, hf_repo=None, hf_prefix=None,
            platform=None, machine=None, teamspace=None,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_yolo_skips_hook(self):
        """``brian deploy yolo`` must NOT call _run_hook."""
        from neuroslm import cli as cli_mod

        # get_connector and load_project_config are local imports inside
        # cmd_deploy, so patch at their source modules.
        with patch.object(cli_mod, "_run_hook", return_value=0) as hook_mock, \
             patch("neuroslm.project_config.load_project_config",
                   return_value=self._fake_cfg()), \
             patch("neuroslm.connectors.get_connector") as mock_gc:
            mock_gc.return_value.launch.return_value = 0
            cli_mod.cmd_deploy(self._yolo_ns())

        assert not hook_mock.called, (
            "brian deploy yolo must skip the pre-deploy hook"
        )

    def test_yolo_clears_arch(self):
        """``brian deploy yolo`` must not pass 'yolo' as the arch path."""
        from neuroslm import cli as cli_mod
        from neuroslm.connectors import DeployConfig

        launched_configs: list[DeployConfig] = []

        class FakeConnector:
            def launch(self, config: DeployConfig) -> int:
                launched_configs.append(config)
                return 0

        with patch.object(cli_mod, "_run_hook", return_value=0), \
             patch("neuroslm.project_config.load_project_config",
                   return_value=self._fake_cfg()), \
             patch("neuroslm.connectors.get_connector",
                   return_value=FakeConnector()):
            cli_mod.cmd_deploy(self._yolo_ns())

        assert launched_configs, "connector.launch must have been called"
        assert launched_configs[0].arch != "yolo", (
            "the yolo alias must not forward 'yolo' as the arch path; "
            f"got config.arch={launched_configs[0].arch!r}"
        )

    def test_end_to_end_yolo_argv(self):
        """Full argv: ``brian deploy yolo`` -> hook skipped."""
        from neuroslm.cli import _build_parser
        from neuroslm import cli as cli_mod

        parser = _build_parser()
        ns = parser.parse_args(["deploy", "yolo"])

        with patch.object(cli_mod, "_run_hook", return_value=0) as hook_mock, \
             patch("neuroslm.project_config.load_project_config",
                   return_value=self._fake_cfg()), \
             patch("neuroslm.connectors.get_connector") as mock_gc:
            mock_gc.return_value.launch.return_value = 0
            try:
                ns.func(ns)
            except (SystemExit, Exception):
                pass
        assert not hook_mock.called, (
            "end-to-end 'brian deploy yolo' must skip the pre-deploy hook"
        )


class TestArgvRouting:
    def test_top_level_no_verify_reaches_cmd_deploy(self):
        """End-to-end: ``brian --no-verify deploy`` -> cmd_deploy sees
        ``ns.no_verify == True`` -> hook NOT called."""
        from neuroslm.cli import _build_parser
        from neuroslm import cli as cli_mod

        parser = _build_parser()
        ns = parser.parse_args(["--no-verify", "deploy"])

        with patch.object(cli_mod, "_run_hook", return_value=0) as hook_mock, \
             patch.object(cli_mod, "_run", return_value=0):
            try:
                ns.func(ns)
            except (SystemExit, Exception):
                pass
        assert not hook_mock.called, (
            "argv-level --no-verify must propagate end-to-end to "
            "cmd_deploy and skip the hook"
        )
