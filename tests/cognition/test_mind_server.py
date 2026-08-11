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

    def test_deploy_mind_parses(self):
        from neuroslm.cli import _build_parser
        args = _build_parser().parse_args(["deploy-mind"])
        assert args.expert == "smollm2_360m"
        assert args.no_mind is False and args.port == 7861

    def test_mind_onstart_has_no_self_destroy_and_a_restart_loop(self):
        from neuroslm.connectors.vast_mind import build_mind_onstart
        s = build_mind_onstart({"EXPERT": "smollm2_360m", "MIND": True,
                                "PORT": 7861, "BRANCH": "master"})
        assert "destroy instance" not in s, (
            "a mind box is always-on BY DESIGN — destroyed only via "
            "`brian destroy <id>`, never by its own onstart")
        assert "while true" in s and "--serve" in s and "--mind" in s
        assert "--port 7861" in s
