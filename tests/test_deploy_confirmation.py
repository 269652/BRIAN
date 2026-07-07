# -*- coding: utf-8 -*-
"""The deploy confirmation gate: humans (TTY or Colab/Jupyter) pass, agents block.

The gate exists to stop AI agents / CI / piped subprocesses from launching paid
cloud instances. Colab-from-a-phone is a *human* flow: stdin is not a TTY, but
`input()` renders a prompt box a person types into. This must be allowed, while a
plain script / piped subprocess (agents) stays blocked.
"""
import builtins

import pytest

from neuroslm import cli


class TestConfirmationChannel:
    def test_tty_is_a_human_channel(self, monkeypatch):
        monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)
        assert cli._deploy_confirm_is_human() is True

    def test_colab_is_a_human_channel(self, monkeypatch):
        monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False, raising=False)
        monkeypatch.setattr("neuroslm.utils.secrets.detect_environment", lambda: "colab")
        assert cli._deploy_confirm_is_human() is True

    def test_jupyter_is_a_human_channel(self, monkeypatch):
        monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False, raising=False)
        monkeypatch.setattr("neuroslm.utils.secrets.detect_environment", lambda: "jupyter")
        assert cli._deploy_confirm_is_human() is True

    def test_plain_script_is_not_a_human_channel(self, monkeypatch):
        monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False, raising=False)
        monkeypatch.setattr("neuroslm.utils.secrets.detect_environment", lambda: "script")
        assert cli._deploy_confirm_is_human() is False


class TestGate:
    def test_script_context_blocks(self, monkeypatch):
        # non-TTY, non-notebook → SystemExit(1), no input() consulted
        monkeypatch.setattr(cli, "_deploy_confirm_is_human", lambda: False)
        with pytest.raises(SystemExit) as ei:
            cli._require_human_confirmation("vast", 10000)
        assert ei.value.code == 1

    def test_colab_human_types_deploy_passes(self, monkeypatch):
        monkeypatch.setattr(cli, "_deploy_confirm_is_human", lambda: True)
        monkeypatch.setattr(builtins, "input", lambda *a, **k: "deploy")
        # should NOT raise
        cli._require_human_confirmation("vast", 10000)

    def test_colab_human_types_other_aborts(self, monkeypatch):
        monkeypatch.setattr(cli, "_deploy_confirm_is_human", lambda: True)
        monkeypatch.setattr(builtins, "input", lambda *a, **k: "nope")
        with pytest.raises(SystemExit) as ei:
            cli._require_human_confirmation("vast", 10000)
        assert ei.value.code == 1

    def test_eof_aborts(self, monkeypatch):
        monkeypatch.setattr(cli, "_deploy_confirm_is_human", lambda: True)
        def _raise(*a, **k):
            raise EOFError
        monkeypatch.setattr(builtins, "input", _raise)
        with pytest.raises(SystemExit):
            cli._require_human_confirmation("vast", 10000)
