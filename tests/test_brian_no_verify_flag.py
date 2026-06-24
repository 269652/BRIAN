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
