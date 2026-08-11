# -*- coding: utf-8 -*-
"""MindServer — the always-on mind as a network service (§14.5).

`brian chat --serve` wraps the ChatDaemon in a newline-delimited-JSON
TCP protocol so a laptop can talk to a mind running on a vast.ai box
through an SSH tunnel. Security model: the server binds 127.0.0.1
ONLY — vast boxes are internet-exposed, so SSH (key-auth'd port
forward) is the transport and the auth layer; no public port ever.

Protocol (one JSON object per line, both directions):
  → {"op": "ping"}                      ← {"ok": true, "pong": true}
  → {"op": "say", "text": "..."}        ← {"ok": true, "reply": "..."}
  → {"op": "think"}                     ← {"ok": true, "thought": ...}
  → {"op": "render"}                    ← {"ok": true, "render": "..."}
  → anything else                       ← {"ok": false, "error": "..."}
"""
from __future__ import annotations

import json
import socket

import pytest

from neuroslm.chat_daemon import ChatDaemon, ChatDaemonConfig


class _EchoGen:
    def __call__(self, prompt: str, max_new_tokens: int) -> str:
        return "echo-reply"


@pytest.fixture
def daemon():
    return ChatDaemon(_EchoGen(), ChatDaemonConfig(), use_color=False)


@pytest.fixture
def server(daemon):
    from neuroslm.cognition.server import MindServer
    s = MindServer(daemon, host="127.0.0.1", port=0)  # 0 → ephemeral
    port = s.start()
    yield s, port
    s.stop()


def _rpc(port: int, payload: dict, n: int = 1):
    with socket.create_connection(("127.0.0.1", port), timeout=5) as c:
        f = c.makefile("rw", encoding="utf-8", newline="\n")
        out = []
        for _ in range(n):
            f.write(json.dumps(payload) + "\n")
            f.flush()
            out.append(json.loads(f.readline()))
        return out if n > 1 else out[0]


class TestMindServer:
    def test_start_returns_bound_port(self, server):
        _, port = server
        assert isinstance(port, int) and port > 0

    def test_ping(self, server):
        _, port = server
        assert _rpc(port, {"op": "ping"}) == {"ok": True, "pong": True}

    def test_say_routes_through_daemon_respond(self, server, daemon):
        _, port = server
        res = _rpc(port, {"op": "say", "text": "hello"})
        assert res["ok"] is True
        assert res["reply"] == "echo-reply"
        kinds = [e.kind for e in daemon.memory.recent(8)]
        assert "user" in kinds and "reply" in kinds

    def test_think_and_render(self, server):
        _, port = server
        t = _rpc(port, {"op": "think"})
        assert t["ok"] is True and t["thought"] == "echo-reply"
        r = _rpc(port, {"op": "render"})
        assert r["ok"] is True and "BRIAN" in r["render"]

    def test_unknown_op_is_an_error_not_a_crash(self, server):
        _, port = server
        res = _rpc(port, {"op": "frobnicate"})
        assert res["ok"] is False and "error" in res
        # server still alive afterwards
        assert _rpc(port, {"op": "ping"})["ok"] is True

    def test_malformed_json_is_an_error_not_a_crash(self, server):
        _, port = server
        with socket.create_connection(("127.0.0.1", port), timeout=5) as c:
            f = c.makefile("rw", encoding="utf-8", newline="\n")
            f.write("this is not json\n")
            f.flush()
            res = json.loads(f.readline())
        assert res["ok"] is False

    def test_binds_localhost_only_by_default(self, daemon):
        from neuroslm.cognition.server import MindServer
        s = MindServer(daemon, port=0)
        try:
            s.start()
            assert s.host == "127.0.0.1", (
                "vast boxes are internet-exposed — the mind must never "
                "listen on a public interface; SSH tunnels are the "
                "transport")
        finally:
            s.stop()


class _FakeMind:
    """Duck-typed mind: canned TickResults, no real CognitiveRuntime
    machinery needed to test the server/client telemetry plumbing."""

    def __init__(self, results):
        self._results = list(results)
        self._i = 0

    def tick(self):
        r = self._results[min(self._i, len(self._results) - 1)]
        self._i += 1
        return r


def _tick_result(**kw):
    from neuroslm.cognition.runtime import TickResult
    base = dict(
        thought="a thought", candidates=["a thought"], scores=[],
        recalled=[{"content": "x"}], stored=True, inhibited=False,
        nt_levels={"DA": 0.15, "NE": 0.20, "5HT": 0.50, "ACh": 0.30,
                  "eCB": 0.10, "Glu": 0.45, "GABA": 0.15},
        phi_proxy=0.40)
    base.update(kw)
    return TickResult(**base)


def _mk_daemon_with_mind(results):
    from neuroslm.chat_daemon import ChatDaemon, ChatDaemonConfig
    mind = _FakeMind(results)
    return (ChatDaemon(_EchoGen(), ChatDaemonConfig(), use_color=False,
                       mind=mind),
            mind)


class TestServerTelemetry:
    """§14.5: a connected client — laptop or `brian logs` — must be
    able to see WHY a tick did what it did, not just its text."""

    def test_think_response_includes_telemetry_summary(self):
        from neuroslm.cognition.server import MindServer
        daemon, _ = _mk_daemon_with_mind([_tick_result()])
        s = MindServer(daemon, host="127.0.0.1", port=0)
        port = s.start()
        try:
            res = _rpc(port, {"op": "think"})
            assert res["ok"] is True
            tel = res.get("telemetry")
            assert tel is not None
            assert "Φ=" in tel["summary"] and "GABA=" in tel["summary"]
            assert tel["recalled"] == 1
            assert tel["stored"] is True
            assert tel["inhibited"] is False
        finally:
            s.stop()

    def test_status_peeks_without_forcing_a_new_tick(self):
        from neuroslm.cognition.server import MindServer
        daemon, mind = _mk_daemon_with_mind([_tick_result()])
        s = MindServer(daemon, host="127.0.0.1", port=0)
        port = s.start()
        try:
            before = _rpc(port, {"op": "status"})
            assert before["ok"] is True and before["telemetry"] is None, (
                "no tick has happened yet")
            _rpc(port, {"op": "think"})
            after = _rpc(port, {"op": "status"})
            assert after["telemetry"] is not None
            assert mind._i == 1, "status must not itself trigger a tick"
        finally:
            s.stop()

    def test_daemon_without_mind_has_null_telemetry(self, server):
        _, port = server
        res = _rpc(port, {"op": "think"})
        assert res["ok"] is True and res.get("telemetry") is None


class TestConnectClientTelemetry:
    def test_think_command_prints_inner_state(self):
        import io
        from neuroslm.cognition.server import MindServer, connect_repl
        daemon, _ = _mk_daemon_with_mind([_tick_result()])
        s = MindServer(daemon, host="127.0.0.1", port=0)
        port = s.start()
        try:
            out_stream = io.StringIO()
            connect_repl("127.0.0.1", port,
                         in_stream=io.StringIO("/think\n/quit\n"),
                         out_stream=out_stream)
            out = out_stream.getvalue()
            assert "a thought" in out
            assert "GABA=" in out, (
                "basal-ganglia/hippocampus/NT telemetry must reach the "
                "laptop, not just the thought text")
        finally:
            s.stop()

    def test_status_command_peeks_last_tick(self):
        import io
        from neuroslm.cognition.server import MindServer, connect_repl
        daemon, _ = _mk_daemon_with_mind([_tick_result()])
        s = MindServer(daemon, host="127.0.0.1", port=0)
        port = s.start()
        try:
            out_stream = io.StringIO()
            connect_repl(
                "127.0.0.1", port,
                in_stream=io.StringIO("/status\n/think\n/status\n/quit\n"),
                out_stream=out_stream)
            out = out_stream.getvalue()
            assert "no ticks yet" in out.lower() or "none" in out.lower()
            assert out.count("GABA=") >= 1
        finally:
            s.stop()


class TestConnectClient:
    def test_client_repl_talks_to_a_live_server(self, server):
        import io
        from neuroslm.cognition.server import connect_repl
        _, port = server
        in_stream = io.StringIO("hello there\n/quit\n")
        out_stream = io.StringIO()
        rc = connect_repl("127.0.0.1", port,
                          in_stream=in_stream, out_stream=out_stream)
        assert rc == 0
        out = out_stream.getvalue()
        assert "echo-reply" in out

    def test_client_survives_connection_refused(self):
        import io
        from neuroslm.cognition.server import connect_repl
        rc = connect_repl("127.0.0.1", 1,  # nothing listens on port 1
                          in_stream=io.StringIO(""),
                          out_stream=io.StringIO())
        assert rc != 0


class TestCliWiring:
    def test_serve_flags_parse(self):
        from neuroslm.cli import _build_parser
        args = _build_parser().parse_args(
            ["chat", "--expert", "--mind", "--serve", "--port", "7999"])
        assert args.serve is True and args.port == 7999

    def test_serve_defaults(self):
        from neuroslm.cli import _build_parser
        args = _build_parser().parse_args(["chat"])
        assert args.serve is False and args.port == 7861

    def test_connect_subcommand_parses(self):
        from neuroslm.cli import _build_parser
        args = _build_parser().parse_args(
            ["chat", "connect", "--host", "127.0.0.1", "--port", "7999"])
        assert args.ckpt == "connect"
        assert args.connect_host == "127.0.0.1"
        # --port feeds both server and client; --connect-port overrides
        assert args.port == 7999 and args.connect_port is None
        args = _build_parser().parse_args(
            ["chat", "connect", "--connect-port", "8001"])
        assert args.connect_port == 8001

    def test_tunnel_passes_identity_file(self):
        from neuroslm.cognition.server import open_vast_tunnel
        seen = {}

        def spawner(argv):
            seen["argv"] = argv
            return object()

        open_vast_tunnel("123", 7861,
                         resolver=lambda iid: "ssh://root@h.vast.ai:2222",
                         spawner=spawner,
                         identity="C:/Users/x/.ssh/id")
        argv = seen["argv"]
        assert "-i" in argv
        assert argv[argv.index("-i") + 1] == "C:/Users/x/.ssh/id", (
            "nonstandard key filenames (e.g. ~/.ssh/id) are never "
            "auto-offered by ssh — the tunnel must pass -i explicitly")
        # sanity on the rest of the argv shape
        assert argv[argv.index("-p") + 1] == "2222"
        assert "root@h.vast.ai" in argv

    def test_deploy_mind_parses(self):
        from neuroslm.cli import _build_parser
        args = _build_parser().parse_args(["deploy-mind"])
        assert args.expert == "smollm2_360m"
        assert args.no_mind is False and args.port == 7861

    def test_deploy_mind_defaults_to_a100(self):
        """2026-08-11: user preference — inference deploys should also
        target A100, not the cheap-card default."""
        from neuroslm.connectors.vast_mind import MindDeployConfig
        assert "A100" in MindDeployConfig().gpu_query

    def test_launch_falls_back_to_env_file_when_shell_env_is_empty(
            self, monkeypatch, tmp_path):
        """2026-08-11 incident: `brian deploy-mind` run from a shell
        session with no GH_TOKEN exported produced a box whose onstart
        printed '✗ GH_TOKEN token not set' and never started the
        server — the token was sitting in .env the whole time but
        vast_mind.py only read raw os.environ. Fix: bootstrap_secrets
        (the same .env walker lightning.py already uses) runs before
        the onstart is built."""
        import neuroslm.connectors.vast_mind as vm

        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("GH_TOKEN=ghp_fromdotenv\nHF_TOKEN=hf_fromdotenv\n")
        monkeypatch.chdir(tmp_path)

        captured = {}

        def fake_find_bash():
            return "bash"

        def fake_call(argv, cwd, env, stdin):
            onstart_path = env["ONSTART_FILE"]
            captured["onstart"] = open(onstart_path, encoding="utf-8").read()
            return 0

        monkeypatch.setattr(vm.VastMindConnector, "_find_bash",
                            staticmethod(fake_find_bash))
        monkeypatch.setattr(vm.subprocess, "call", fake_call)

        rc = vm.VastMindConnector().launch(vm.MindDeployConfig())
        assert rc == 0
        assert "ghp_fromdotenv" in captured["onstart"], (
            ".env must be walked into the onstart when the shell env "
            "is empty — the token must not silently ship as ''")

    def test_missing_token_warns_before_spending_money(
            self, monkeypatch, tmp_path, capsys):
        """§8.1: a missing token must surface at deploy-time, not
        silently produce a broken box that bills anyway."""
        import neuroslm.connectors.vast_mind as vm

        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.chdir(tmp_path)  # no .env here either
        monkeypatch.setattr(vm.VastMindConnector, "_find_bash",
                            staticmethod(lambda: "bash"))
        monkeypatch.setattr(vm.subprocess, "call", lambda *a, **k: 0)

        vm.VastMindConnector().launch(vm.MindDeployConfig())
        err = capsys.readouterr().err
        assert "GH_TOKEN" in err and "missing" in err.lower()

    def test_mind_onstart_has_no_self_destroy_and_a_restart_loop(self):
        from neuroslm.connectors.vast_mind import build_mind_onstart
        s = build_mind_onstart({"EXPERT": "smollm2_360m", "MIND": True,
                                "PORT": 7861, "BRANCH": "master"})
        assert "destroy instance" not in s, (
            "a mind box is always-on BY DESIGN — destroyed only via "
            "`brian destroy <id>`, never by its own onstart")
        assert "while true" in s and "--serve" in s and "--mind" in s
        assert "--port 7861" in s
