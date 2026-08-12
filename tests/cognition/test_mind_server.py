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
  → {"op": "observe_sensory",
     "modality": "visual", "vec": [...],
     "source": "isaac_sim"}             ← {"ok": true, "attended": bool,
                                            "novelty": float}
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
    from neuroslm.cognition.runtime import TickResult, ThoughtScore
    base = dict(
        thought="a thought", candidates=["a thought"],
        scores=[ThoughtScore(mean_nll=2.5, entropy_norm=0.4)],
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


class TestPatternsOp:
    """'It should learn causal and temporal relations' — the wire-
    level surface for neuroslm/cognition/patterns.py's association
    mining, so a connected client can actually see what's been
    detected, not just the box's own process state."""

    def _daemon_with_history(self):
        from neuroslm.cognition.runtime import CognitiveRuntime, MindConfig
        from neuroslm.memory.episodic import EpisodicMemory

        def score_fn(text):
            from neuroslm.cognition.runtime import ThoughtScore
            return ThoughtScore(mean_nll=2.0, entropy_norm=0.5)

        rt = CognitiveRuntime(
            generate_fn=_EchoGen(), score_fn=score_fn,
            embed_fn=lambda t: [1.0, 0.0],
            memory=EpisodicMemory(maxlen=64),
            cfg=MindConfig(n_candidates=1))
        for _ in range(4):
            rt.observe("You suck")
            rt.observe("No, that's wrong")
        return ChatDaemon(_EchoGen(), ChatDaemonConfig(), use_color=False,
                          mind=rt)

    def test_patterns_op_returns_mined_rules(self):
        from neuroslm.cognition.server import MindServer
        daemon = self._daemon_with_history()
        s = MindServer(daemon, host="127.0.0.1", port=0)
        port = s.start()
        try:
            res = _rpc(port, {"op": "patterns", "min_confidence": 0.0})
            assert res["ok"] is True
            rules = res["rules"]
            assert any(r["antecedent"] == "insult"
                      and r["consequent"] == "disagreement" for r in rules)
        finally:
            s.stop()

    def test_patterns_op_without_mind_is_a_clean_error(self):
        daemon = ChatDaemon(_EchoGen(), ChatDaemonConfig(), use_color=False)
        from neuroslm.cognition.server import MindServer
        s = MindServer(daemon, host="127.0.0.1", port=0)
        port = s.start()
        try:
            res = _rpc(port, {"op": "patterns"})
            assert res["ok"] is False
        finally:
            s.stop()


class TestObserveSensoryOp:
    """§15 remote bridge: a sensory source (e.g. a SensoryBridge
    running wherever Isaac Sim actually runs — a different box than
    the mind server) pushes an already-embedded percept over the SAME
    SSH-tunnelled wire protocol every other op uses, instead of
    calling CognitiveRuntime.observe_sensory() in-process. This is
    what makes a REMOTE bridge possible without changing
    SensoryBridge itself (see RemoteMindProxy)."""

    def _daemon_with_mind(self):
        from neuroslm.cognition.runtime import CognitiveRuntime, MindConfig
        from neuroslm.memory.episodic import EpisodicMemory

        def score_fn(text):
            from neuroslm.cognition.runtime import ThoughtScore
            return ThoughtScore(mean_nll=2.0, entropy_norm=0.5)

        rt = CognitiveRuntime(
            generate_fn=_EchoGen(), score_fn=score_fn,
            embed_fn=lambda t: [1.0, 0.0],
            memory=EpisodicMemory(maxlen=64),
            cfg=MindConfig(n_candidates=1))
        return ChatDaemon(_EchoGen(), ChatDaemonConfig(), use_color=False,
                          mind=rt), rt

    def test_novel_percept_is_attended_and_stored(self):
        from neuroslm.cognition.server import MindServer
        daemon, rt = self._daemon_with_mind()
        s = MindServer(daemon, host="127.0.0.1", port=0)
        port = s.start()
        try:
            res = _rpc(port, {"op": "observe_sensory", "modality": "visual",
                              "vec": [1.0, 0.0], "source": "isaac_sim"})
            assert res["ok"] is True
            assert res["attended"] is True
            assert 0.0 <= res["novelty"] <= 1.0
            assert any((e.get("context") or {}).get("modality") == "visual"
                      for e in rt.memory.all())
        finally:
            s.stop()

    def test_habituated_repeat_percept_reports_not_attended(self):
        from neuroslm.cognition.server import MindServer
        daemon, rt = self._daemon_with_mind()
        s = MindServer(daemon, host="127.0.0.1", port=0)
        port = s.start()
        try:
            msg = {"op": "observe_sensory", "modality": "visual",
                  "vec": [1.0, 0.0]}
            first = _rpc(port, msg)
            second = _rpc(port, msg)
            assert first["attended"] is True
            assert second["attended"] is False
        finally:
            s.stop()

    def test_missing_mind_is_a_clean_error(self):
        from neuroslm.cognition.server import MindServer
        daemon = ChatDaemon(_EchoGen(), ChatDaemonConfig(), use_color=False)
        s = MindServer(daemon, host="127.0.0.1", port=0)
        port = s.start()
        try:
            res = _rpc(port, {"op": "observe_sensory", "modality": "visual",
                              "vec": [1.0, 0.0]})
            assert res["ok"] is False
        finally:
            s.stop()

    def test_missing_modality_or_vec_is_a_clean_error(self):
        from neuroslm.cognition.server import MindServer
        daemon, rt = self._daemon_with_mind()
        s = MindServer(daemon, host="127.0.0.1", port=0)
        port = s.start()
        try:
            res = _rpc(port, {"op": "observe_sensory", "vec": [1.0, 0.0]})
            assert res["ok"] is False
            res2 = _rpc(port, {"op": "observe_sensory", "modality": "visual"})
            assert res2["ok"] is False
        finally:
            s.stop()


class TestEmbedDimOp:
    """RemoteMindProxy's other half: a remote sensory bridge sizes its
    cortices' projection heads by asking the SERVER what dimension the
    mind's own text embed_fn uses — the same probe embed_dim() does
    for an in-process SensoryBridge."""

    def _daemon_with_mind(self):
        from neuroslm.cognition.runtime import CognitiveRuntime, MindConfig
        from neuroslm.memory.episodic import EpisodicMemory

        def score_fn(text):
            from neuroslm.cognition.runtime import ThoughtScore
            return ThoughtScore(mean_nll=2.0, entropy_norm=0.5)

        rt = CognitiveRuntime(
            generate_fn=_EchoGen(), score_fn=score_fn,
            embed_fn=lambda t: [0.0] * 6,
            memory=EpisodicMemory(maxlen=64),
            cfg=MindConfig(n_candidates=1))
        return ChatDaemon(_EchoGen(), ChatDaemonConfig(), use_color=False,
                          mind=rt)

    def test_returns_the_minds_embed_dim(self):
        from neuroslm.cognition.server import MindServer
        daemon = self._daemon_with_mind()
        s = MindServer(daemon, host="127.0.0.1", port=0)
        port = s.start()
        try:
            res = _rpc(port, {"op": "embed_dim"})
            assert res["ok"] is True
            assert res["dim"] == 6
        finally:
            s.stop()

    def test_missing_mind_is_a_clean_error(self):
        from neuroslm.cognition.server import MindServer
        daemon = ChatDaemon(_EchoGen(), ChatDaemonConfig(), use_color=False)
        s = MindServer(daemon, host="127.0.0.1", port=0)
        port = s.start()
        try:
            res = _rpc(port, {"op": "embed_dim"})
            assert res["ok"] is False
        finally:
            s.stop()


class TestRemoteMindProxy:
    """The client-side counterpart: makes a mind running on a
    DIFFERENT box/process look, to SensoryBridge, exactly like a
    local CognitiveRuntime — same three touch points
    (embed_dim/observe_sensory/last_sensory_novelty), nothing else of
    CognitiveRuntime is proxied. This is what lets a SensoryBridge
    running wherever Isaac Sim actually is (an RTX-capable box) drive
    a mind deployed on a different (e.g. vast.ai A100) box."""

    def _daemon_with_mind(self, embed_fn=None):
        from neuroslm.cognition.runtime import CognitiveRuntime, MindConfig
        from neuroslm.memory.episodic import EpisodicMemory

        def score_fn(text):
            from neuroslm.cognition.runtime import ThoughtScore
            return ThoughtScore(mean_nll=2.0, entropy_norm=0.5)

        rt = CognitiveRuntime(
            generate_fn=_EchoGen(), score_fn=score_fn,
            embed_fn=embed_fn or (lambda t: [1.0, 0.0]),
            memory=EpisodicMemory(maxlen=64),
            cfg=MindConfig(n_candidates=1))
        return ChatDaemon(_EchoGen(), ChatDaemonConfig(), use_color=False,
                          mind=rt), rt

    def test_embed_dim_matches_the_remote_minds_embedding(self):
        from neuroslm.cognition.server import MindServer, RemoteMindProxy
        daemon, rt = self._daemon_with_mind(embed_fn=lambda t: [0.0] * 4)
        s = MindServer(daemon, host="127.0.0.1", port=0)
        port = s.start()
        try:
            proxy = RemoteMindProxy(host="127.0.0.1", port=port)
            assert proxy.embed_dim() == 4
        finally:
            s.stop()

    def test_observe_sensory_round_trips_to_the_remote_mind(self):
        from neuroslm.cognition.server import MindServer, RemoteMindProxy
        daemon, rt = self._daemon_with_mind()
        s = MindServer(daemon, host="127.0.0.1", port=0)
        port = s.start()
        try:
            proxy = RemoteMindProxy(host="127.0.0.1", port=port)
            attended = proxy.observe_sensory("visual", [1.0, 0.0],
                                             source="isaac_sim")
            assert attended is True
            assert proxy.last_sensory_novelty is not None
            assert any((e.get("context") or {}).get("modality") == "visual"
                      for e in rt.memory.all())
        finally:
            s.stop()

    def test_sensory_bridge_works_transparently_against_the_proxy(self):
        """The actual point: SensoryBridge is unmodified — a
        RemoteMindProxy just fills the 'runtime' slot."""
        import numpy as np
        from neuroslm.cognition.server import MindServer, RemoteMindProxy
        from neuroslm.connectors.isaac_sim import SensoryBridge

        daemon, rt = self._daemon_with_mind(embed_fn=lambda t: [0.0] * 8)
        s = MindServer(daemon, host="127.0.0.1", port=0)
        port = s.start()
        try:
            proxy = RemoteMindProxy(host="127.0.0.1", port=port)

            class _FakeClient:
                def get_frame(self):
                    return np.ones((8, 8, 3), dtype="uint8") * 150

                def get_joint_state(self):
                    return None

            def fake_visual_cortex(frame):
                return [1.0] + [0.0] * 7

            bridge = SensoryBridge(proxy, _FakeClient(),
                                   visual_cortex=fake_visual_cortex)
            attended = bridge.pump()
            assert attended["visual"] is True
            assert any((e.get("context") or {}).get("modality") == "visual"
                      for e in rt.memory.all())
        finally:
            s.stop()


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

    def test_think_response_includes_full_debug_trace_and_tick_n(self):
        """2026-08-12 live incident: the connected laptop client never
        saw the DMN loop's autonomous thoughts OR the basal-ganglia
        debug trace at all — both only ever reached the box's own
        stdout (brian logs). The wire payload must carry the SAME
        info format_debug_trace prints server-side."""
        from neuroslm.cognition.server import MindServer
        daemon, _ = _mk_daemon_with_mind([_tick_result(tick_n=7)])
        s = MindServer(daemon, host="127.0.0.1", port=0)
        port = s.start()
        try:
            res = _rpc(port, {"op": "think"})
            tel = res["telemetry"]
            assert tel["tick_n"] == 7
            assert tel["thought"] == "a thought"
            assert "BG deliberation" in tel["debug"]
            assert "SELECTED" in tel["debug"]
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


class TestBackgroundTickPolling:
    """2026-08-12 live incident: 'still no wandering thoughts or
    introspection also no debug logs' when connected via
    `brian chat connect` — the DMN loop WAS running server-side
    (confirmed in `brian logs`), but the wire protocol is pure
    request/response, so a connected client only ever saw something
    when IT initiated a `/think`. A background poller (its own
    connection, so it never contends with the daemon's inference
    lock or the main REPL socket) watches for tick_n advancing and
    prints new thoughts as they happen — this is what makes the
    always-on mind's own thinking visible live, not just on request.
    """

    def test_poll_prints_a_new_tick(self):
        import io
        import threading
        from neuroslm.cognition.server import MindServer, _poll_new_ticks

        long_thought = "a wandering idea about " + ("x" * 200)
        daemon, _ = _mk_daemon_with_mind([_tick_result(
            thought=long_thought, candidates=[long_thought],
            tick_n=1)])
        s = MindServer(daemon, host="127.0.0.1", port=0)
        port = s.start()
        try:
            _rpc(port, {"op": "think"})  # server-side tick happens
            out = io.StringIO()
            _poll_new_ticks("127.0.0.1", port, out, threading.Lock(),
                            threading.Event(), poll_interval=0.01,
                            max_polls=1)
            text = out.getvalue()
            assert long_thought in text, (
                "the winning thought must render in full to a "
                "connected client, not truncated")
            assert "BG deliberation" in text, (
                "the debug trace must reach the client too, not just "
                "the compact summary")
        finally:
            s.stop()

    def test_poll_does_not_reprint_the_same_tick(self):
        import io
        import threading
        from neuroslm.cognition.server import MindServer, _poll_new_ticks

        daemon, _ = _mk_daemon_with_mind([_tick_result(
            thought="same idea", candidates=["same idea"], tick_n=3)])
        s = MindServer(daemon, host="127.0.0.1", port=0)
        port = s.start()
        try:
            _rpc(port, {"op": "think"})
            out = io.StringIO()
            _poll_new_ticks("127.0.0.1", port, out, threading.Lock(),
                            threading.Event(), poll_interval=0.01,
                            max_polls=3)
            assert out.getvalue().count("same idea") == 1, (
                "a tick already shown must not be printed again on "
                "every subsequent poll of the same tick_n")
        finally:
            s.stop()

    def test_poll_stops_when_event_is_set(self):
        import io
        import threading
        from neuroslm.cognition.server import MindServer, _poll_new_ticks

        daemon, _ = _mk_daemon_with_mind([_tick_result()])
        s = MindServer(daemon, host="127.0.0.1", port=0)
        port = s.start()
        try:
            stop = threading.Event()
            stop.set()  # already stopped — must return immediately
            t0 = __import__("time").time()
            _poll_new_ticks("127.0.0.1", port, io.StringIO(),
                            threading.Lock(), stop, poll_interval=5.0)
            assert __import__("time").time() - t0 < 1.0
        finally:
            s.stop()

    def test_connect_repl_starts_and_cleanly_stops_the_poller(self):
        import io
        from neuroslm.cognition.server import MindServer, connect_repl

        daemon, _ = _mk_daemon_with_mind([_tick_result()])
        s = MindServer(daemon, host="127.0.0.1", port=0)
        port = s.start()
        try:
            rc = connect_repl(
                "127.0.0.1", port,
                in_stream=io.StringIO("/quit\n"), out_stream=io.StringIO(),
                poll_interval=0.05)
            assert rc == 0
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

    def test_isaac_relay_opens_a_forward_tunnel_to_the_mind(self):
        """§15 remote bridge: connecting an Isaac Sim box (RTX,
        wherever it actually is) to a mind box (A100, vast.ai) without
        either box ever holding the operator's SSH private key —
        BOTH tunnels originate from the operator's own machine, the
        SAME security model as brian chat connect."""
        from neuroslm.cognition.server import open_isaac_relay
        seen = []

        def spawner(argv):
            seen.append(argv)
            return object()

        def resolver(iid):
            return {"mind-1": "ssh://root@mind.vast.ai:2222",
                   "isaac-1": "ssh://root@isaac.vast.ai:3333"}[iid]

        open_isaac_relay("isaac-1", "mind-1", 7861,
                         resolver=resolver, spawner=spawner,
                         identity="~/.ssh/id")
        to_mind = next(a for a in seen if "root@mind.vast.ai" in a)
        assert "-L" in to_mind
        assert to_mind[to_mind.index("-L") + 1] == "7861:127.0.0.1:7861"
        assert to_mind[to_mind.index("-p") + 1] == "2222"
        assert "-i" in to_mind

    def test_isaac_relay_opens_a_reverse_tunnel_to_the_isaac_box(self):
        from neuroslm.cognition.server import open_isaac_relay
        seen = []

        def spawner(argv):
            seen.append(argv)
            return object()

        def resolver(iid):
            return {"mind-1": "ssh://root@mind.vast.ai:2222",
                   "isaac-1": "ssh://root@isaac.vast.ai:3333"}[iid]

        open_isaac_relay("isaac-1", "mind-1", 7861,
                         resolver=resolver, spawner=spawner)
        to_isaac = next(a for a in seen if "root@isaac.vast.ai" in a)
        assert "-R" in to_isaac, (
            "the isaac-side leg must be a REMOTE forward (-R) — the "
            "isaac box's own 127.0.0.1:port relays back through this "
            "connection to the operator's machine, which the OTHER "
            "leg forwards on to the mind box"
        )
        assert to_isaac[to_isaac.index("-R") + 1] == "7861:127.0.0.1:7861"
        assert to_isaac[to_isaac.index("-p") + 1] == "3333"

    def test_isaac_relay_returns_both_processes(self):
        from neuroslm.cognition.server import open_isaac_relay

        def resolver(iid):
            return "ssh://root@h.vast.ai:22"

        procs = open_isaac_relay("isaac-1", "mind-1", 7861,
                                 resolver=resolver,
                                 spawner=lambda argv: object())
        assert len(procs) == 2

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

    def test_deploy_mind_defaults_to_cuda_device(self):
        """2026-08-12 live incident companion fix: the model-placement
        bug (build_runtime_from_hf_lm never called .to(device)) meant
        even a correct --device cuda flag wouldn't have helped without
        that fix, but the onstart ALSO never passed --device at all —
        `brian chat`'s own CLI default is cpu. Both must be fixed for
        an A100 mind deploy to actually use the GPU."""
        from neuroslm.connectors.vast_mind import MindDeployConfig
        assert MindDeployConfig().device == "cuda"

    def test_mind_onstart_requests_cuda_device(self):
        from neuroslm.connectors.vast_mind import build_mind_onstart
        s = build_mind_onstart({"EXPERT": "smollm2_360m", "MIND": True,
                                "PORT": 7861, "BRANCH": "master",
                                "DEVICE": "cuda"})
        assert "--device 'cuda'" in s

    def test_mind_onstart_has_no_self_destroy_and_a_restart_loop(self):
        from neuroslm.connectors.vast_mind import build_mind_onstart
        s = build_mind_onstart({"EXPERT": "smollm2_360m", "MIND": True,
                                "PORT": 7861, "BRANCH": "master"})
        assert "destroy instance" not in s, (
            "a mind box is always-on BY DESIGN — destroyed only via "
            "`brian destroy <id>`, never by its own onstart")
        assert "while true" in s and "--serve" in s and "--mind" in s
        assert "--port 7861" in s


class TestIsaacDeployCliWiring:
    """§15 remote bridge: `brian deploy-isaac-sim` — a sensory-source
    box, separate from the mind box, running on an RTX-class GPU.

    Per CLAUDE.md §1's own exemption, vast.ai deploy scripts are
    'verified by deploying, not unit-tested' — the actual Isaac Sim
    boot sequence (pip install, scene setup, SimulationApp) is
    UNVERIFIED against real hardware (documented explicitly in
    docs/findings.md). What IS pinned here is everything testable
    without a GPU: config defaults, CLI parsing, onstart templating,
    and the secret-handling safety net — the same class of contract
    TestCliWiring already pins for the mind box."""

    def test_deploy_isaac_sim_parses(self):
        from neuroslm.cli import _build_parser
        args = _build_parser().parse_args(["deploy-isaac-sim"])
        assert args.func is not None
        assert args.port == 7861

    def test_bridge_isaac_subcommand_parses(self):
        from neuroslm.cli import _build_parser
        args = _build_parser().parse_args(
            ["chat", "bridge-isaac", "--tunnel", "mind-1",
             "--isaac-tunnel", "isaac-1"])
        assert args.ckpt == "bridge-isaac"
        assert args.tunnel == "mind-1"
        assert args.isaac_tunnel == "isaac-1"

    def test_deploy_isaac_sim_defaults_to_an_rtx_card(self):
        """A100 (the mind's own default) has no RT cores — Isaac
        Sim's renderer needs an RTX-class card."""
        from neuroslm.connectors.vast_isaac import IsaacSimDeployConfig
        assert "RTX" in IsaacSimDeployConfig().gpu_query
        assert "A100" not in IsaacSimDeployConfig().gpu_query

    def test_isaac_onstart_pins_the_isaac_sim_version(self):
        from neuroslm.connectors.vast_isaac import build_isaac_onstart
        s = build_isaac_onstart({"BRANCH": "master", "PORT": 7861,
                                 "ISAAC_SIM_VERSION": "4.5.0"})
        assert "isaacsim[all,extscache]==4.5.0" in s
        assert "pypi.nvidia.com" in s

    def test_isaac_onstart_has_no_self_destroy_and_a_restart_loop(self):
        from neuroslm.connectors.vast_isaac import build_isaac_onstart
        s = build_isaac_onstart({"BRANCH": "master", "PORT": 7861,
                                 "ISAAC_SIM_VERSION": "4.5.0"})
        assert "destroy instance" not in s, (
            "an isaac-sim sensory source is an always-on companion "
            "to the mind box — destroyed only via `brian destroy "
            "<id>`, never by its own onstart")
        assert "while true" in s

    def test_isaac_onstart_answers_the_eula_prompt_non_interactively(self):
        """Live incident (2026-08-12): omni.kit_app's check_eula() calls
        input() at import time — in a non-interactive onstart context
        that's an immediate EOFError, and the crash-restart loop just
        hammers the same prompt every 10s forever, never reaching the
        sensor loop. ACCEPT_EULA=Y alone does not satisfy THIS specific
        gate (it's a bare input() call, not an env-var check) — the
        python invocation must feed it an answer via stdin."""
        from neuroslm.connectors.vast_isaac import build_isaac_onstart
        s = build_isaac_onstart({"BRANCH": "master", "PORT": 7861,
                                 "ISAAC_SIM_VERSION": "4.5.0"})
        assert "yes 'Yes'" in s or 'yes "Yes"' in s, (
            "the sensor loop's stdin must be fed an EULA answer — "
            "env vars alone don't satisfy omni.kit_app's input() gate")

    def test_isaac_onstart_captures_the_real_python_exit_code(self):
        """Live incident (2026-08-12): `... | tee -a log; echo rc=$?`
        reports tee's own exit code (always 0), not the python
        process's — every crash logged 'process exited rc=0', masking
        the actual failure. Must read PIPESTATUS[0] instead."""
        from neuroslm.connectors.vast_isaac import build_isaac_onstart
        s = build_isaac_onstart({"BRANCH": "master", "PORT": 7861,
                                 "ISAAC_SIM_VERSION": "4.5.0"})
        assert "PIPESTATUS[0]" in s

    def test_isaac_onstart_writes_the_sensor_loop_script(self):
        from neuroslm.connectors.vast_isaac import build_isaac_onstart
        s = build_isaac_onstart({"BRANCH": "master", "PORT": 7861,
                                 "ISAAC_SIM_VERSION": "4.5.0"})
        assert "SimulationApp" in s
        assert "RemoteMindProxy" in s
        assert "SensoryBridge" in s
        assert "port=7861" in s, "the port placeholder must be substituted"

    def test_isaac_launch_falls_back_to_env_file_when_shell_env_is_empty(
            self, monkeypatch, tmp_path):
        import neuroslm.connectors.vast_isaac as vi

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

        monkeypatch.setattr(vi.VastIsaacConnector, "_find_bash",
                            staticmethod(fake_find_bash))
        monkeypatch.setattr(vi.subprocess, "call", fake_call)

        rc = vi.VastIsaacConnector().launch(vi.IsaacSimDeployConfig())
        assert rc == 0
        assert "ghp_fromdotenv" in captured["onstart"]

    def test_isaac_missing_token_warns_before_spending_money(
            self, monkeypatch, tmp_path, capsys):
        import neuroslm.connectors.vast_isaac as vi

        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(vi.VastIsaacConnector, "_find_bash",
                            staticmethod(lambda: "bash"))
        monkeypatch.setattr(vi.subprocess, "call", lambda *a, **k: 0)

        vi.VastIsaacConnector().launch(vi.IsaacSimDeployConfig())
        err = capsys.readouterr().err
        assert "GH_TOKEN" in err and "missing" in err.lower()

    def test_isaac_launch_uses_the_configured_gpu_query_and_label(
            self, monkeypatch, tmp_path):
        import neuroslm.connectors.vast_isaac as vi

        monkeypatch.setenv("GH_TOKEN", "ghp_x")
        monkeypatch.chdir(tmp_path)
        captured = {}

        def fake_call(argv, cwd, env, stdin):
            captured["env"] = env
            return 0

        monkeypatch.setattr(vi.VastIsaacConnector, "_find_bash",
                            staticmethod(lambda: "bash"))
        monkeypatch.setattr(vi.subprocess, "call", fake_call)

        cfg = vi.IsaacSimDeployConfig(label="my-isaac-box",
                                      gpu_query="gpu_name=RTX_3090")
        vi.VastIsaacConnector().launch(cfg)
        assert captured["env"]["VAST_LABEL"] == "my-isaac-box"
        assert captured["env"]["GPU_QUERY"] == "gpu_name=RTX_3090"
