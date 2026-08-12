# -*- coding: utf-8 -*-
"""MindServer + connect client — the always-on mind over a socket (§14.5).

Server side (on the vast.ai box / any host)::

    brian chat --serve --expert --mind          # localhost:7861

Client side (laptop), through an SSH tunnel::

    ssh -N -L 7861:127.0.0.1:7861 root@<box>    # (or `brian chat connect <ID>`)
    brian chat connect                          # talks to 127.0.0.1:7861

Security model: the server binds **127.0.0.1 only**. Vast boxes are
internet-exposed; SSH key auth + local port forward IS the transport
and the auth layer. No public listener, no token scheme to get wrong.

Protocol: newline-delimited JSON, one object per line each way — see
tests/cognition/test_mind_server.py for the pinned contract.
"""
from __future__ import annotations

import json
import socket
import socketserver
import subprocess
import sys
import threading
from typing import Any, Optional

DEFAULT_PORT = 7861


class MindServer:
    """Thread-per-connection TCP wrapper around a :class:`ChatDaemon`.

    The daemon's own inference lock already serialises generation, so
    concurrent clients are safe — they just queue on the lock like the
    thought thread does. ``port=0`` binds an ephemeral port (tests).
    """

    def __init__(self, daemon: Any, host: str = "127.0.0.1",
                 port: int = DEFAULT_PORT) -> None:
        self.daemon = daemon
        self.host = host
        self.port = port
        self._server: Optional[socketserver.ThreadingTCPServer] = None
        self._thread: Optional[threading.Thread] = None

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self) -> int:
        """Bind + serve in a background thread; returns the bound port."""
        daemon = self.daemon

        class _Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                while True:
                    raw = self.rfile.readline()
                    if not raw:
                        return
                    try:
                        msg = json.loads(raw.decode("utf-8"))
                        resp = _dispatch(daemon, msg)
                    except Exception as e:  # malformed JSON / op crash
                        resp = {"ok": False,
                                "error": f"{type(e).__name__}: {e}"}
                    self.wfile.write(
                        (json.dumps(resp) + "\n").encode("utf-8"))
                    self.wfile.flush()

        class _Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._server = _Server((self.host, self.port), _Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True,
            name="brian-mind-server")
        self._thread.start()
        return self.port

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def _tick_telemetry(daemon: Any) -> Optional[dict]:
    """§14.5: the inner-state summary of the daemon's last cognitive
    tick — basal-ganglia pick, hippocampal recall/write, NT snapshot,
    Φ, and the FULL deliberation trace (every candidate + score, not
    just the compact summary — 2026-08-12: a connected client had no
    way to see this, only `brian logs` on the box did). ``None`` when
    no mind is attached or no tick has happened yet (peekable via the
    ``status`` op without forcing a new tick).
    """
    t = getattr(daemon, "last_tick", None)
    if t is None:
        return None
    from neuroslm.cognition.runtime import format_debug_trace, format_introspection
    return {
        "summary": format_introspection(t),
        "debug": format_debug_trace(t),
        "tick_n": t.tick_n,
        "thought": t.thought,
        "nt_levels": t.nt_levels,
        "phi_proxy": t.phi_proxy,
        "recalled": len(t.recalled),
        "candidates": len(t.candidates),
        "stored": t.stored,
        "inhibited": t.inhibited,
    }


def _dispatch(daemon: Any, msg: dict) -> dict:
    op = msg.get("op")
    if op == "ping":
        return {"ok": True, "pong": True}
    if op == "say":
        return {"ok": True, "reply": daemon.respond(str(msg.get("text", "")))}
    if op == "think":
        thought = daemon.think_once()
        return {"ok": True, "thought": thought,
                "telemetry": _tick_telemetry(daemon)}
    if op == "status":
        # Peek only — must NOT trigger a tick (a client checking in
        # should never itself cause the mind to think or go silent).
        return {"ok": True, "telemetry": _tick_telemetry(daemon)}
    if op == "patterns":
        mind = getattr(daemon, "_mind", None)
        if mind is None or not hasattr(mind, "detect_patterns"):
            return {"ok": False,
                    "error": "no mind attached — patterns need episodic "
                            "history to mine"}
        rules = mind.detect_patterns(
            window=int(msg.get("window", 1)),
            min_support=float(msg.get("min_support", 0.0)),
            min_confidence=float(msg.get("min_confidence", 0.3)))
        return {"ok": True, "rules": [
            {"antecedent": r.antecedent, "consequent": r.consequent,
             "support": r.support, "confidence": r.confidence,
             "lift": r.lift, "evidence_count": r.evidence_count,
             "grounded": r.grounded,
             "self_referential_only": r.self_referential_only}
            for r in rules]}
    if op == "render":
        return {"ok": True, "render": daemon.render()}
    if op == "embed_dim":
        # RemoteMindProxy's other half: a remote sensory bridge sizes
        # its cortices' projection heads by asking the server what
        # dimension the mind's text embed_fn uses.
        mind = getattr(daemon, "_mind", None)
        if mind is None or not hasattr(mind, "embed_dim"):
            return {"ok": False,
                    "error": "no mind attached — embed_dim needs the "
                            "cognitive runtime"}
        return {"ok": True, "dim": mind.embed_dim()}
    if op == "observe_sensory":
        # §15 remote bridge: a sensory source running wherever it
        # actually needs to (e.g. Isaac Sim on an RTX-capable box,
        # separate from wherever this mind server is deployed) pushes
        # an already-embedded percept over the SAME SSH-tunnelled
        # connection every other op uses — RemoteMindProxy is the
        # client-side counterpart that makes this look, to
        # SensoryBridge, exactly like calling
        # CognitiveRuntime.observe_sensory() in-process.
        mind = getattr(daemon, "_mind", None)
        if mind is None or not hasattr(mind, "observe_sensory"):
            return {"ok": False,
                    "error": "no mind attached — sensory input needs "
                            "the cognitive runtime, not the legacy "
                            "chat-only daemon"}
        modality = msg.get("modality")
        vec = msg.get("vec")
        if not modality or not isinstance(vec, list):
            return {"ok": False,
                    "error": "observe_sensory requires 'modality' "
                            "(str) and 'vec' (list of floats)"}
        source = str(msg.get("source") or "remote_sensor")
        attended = mind.observe_sensory(str(modality), vec, source=source)
        return {"ok": True, "attended": attended,
                "novelty": getattr(mind, "last_sensory_novelty", None)}
    return {"ok": False, "error": f"unknown op {op!r}"}


# ── Client ───────────────────────────────────────────────────────────

DEFAULT_POLL_INTERVAL = 4.0


class RemoteMindProxy:
    """§15 remote sensory bridge — makes a mind running on a
    DIFFERENT box/process (reached over the SAME SSH-tunnelled
    connection every other op uses) look, to
    :class:`~neuroslm.connectors.isaac_sim.SensoryBridge`, exactly
    like an in-process :class:`~neuroslm.cognition.runtime.
    CognitiveRuntime`. Implements only the three touch points
    ``SensoryBridge`` actually calls — ``embed_dim()``,
    ``observe_sensory()``, and the ``last_sensory_novelty`` readback —
    nothing else of ``CognitiveRuntime`` is proxied, and nothing about
    ``SensoryBridge`` needs to change to use one.

    This is what makes "run Isaac Sim on an RTX-capable box and the
    mind on a vast.ai A100" possible: construct a ``SensoryBridge``
    on the Isaac Sim side with a ``RemoteMindProxy`` in the
    ``runtime`` slot, tunnelled to the deployed mind server exactly
    like ``connect_repl``/``brian chat connect`` already are.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                *, timeout: float = 10.0):
        self._conn = socket.create_connection((host, port), timeout=timeout)
        self._f = self._conn.makefile("rw", encoding="utf-8", newline="\n")
        self.last_sensory_novelty: Optional[float] = None
        self._dim: Optional[int] = None

    def _rpc(self, msg: dict) -> dict:
        self._f.write(json.dumps(msg) + "\n")
        self._f.flush()
        line = self._f.readline()
        if not line:
            raise ConnectionError(
                "RemoteMindProxy: connection closed by the mind server")
        return json.loads(line)

    def embed_dim(self) -> int:
        if self._dim is None:
            res = self._rpc({"op": "embed_dim"})
            if not res.get("ok"):
                raise RuntimeError(
                    f"RemoteMindProxy.embed_dim: {res.get('error')}")
            self._dim = int(res["dim"])
        return self._dim

    def observe_sensory(self, modality: str, content_vec, *,
                        source: str = "sensor") -> bool:
        res = self._rpc({"op": "observe_sensory", "modality": modality,
                         "vec": list(content_vec), "source": source})
        if not res.get("ok"):
            raise RuntimeError(
                f"RemoteMindProxy.observe_sensory: {res.get('error')}")
        self.last_sensory_novelty = res.get("novelty")
        return bool(res["attended"])

    def close(self) -> None:
        self._conn.close()


def _poll_new_ticks(host: str, port: int, out_stream, write_lock,
                    stop_event, poll_interval: float = DEFAULT_POLL_INTERVAL,
                    max_polls: Optional[int] = None) -> None:
    """Background: watch for the DMN loop's own ticks and print them
    live. 2026-08-12 live incident: the mind's autonomous thinking WAS
    running server-side (confirmed in `brian logs`), but the wire
    protocol is pure request/response — a connected client only ever
    saw a thought when IT asked for one via ``/think``. This polls
    ``status`` (a peek — never touches the daemon's inference lock or
    triggers a tick) on its OWN connection, so it never contends with
    the main REPL socket, and prints the full deliberation trace
    whenever ``tick_n`` advances past what was last shown.

    ``max_polls`` is a test seam — production callers leave it unset
    and rely on ``stop_event`` to end the loop.
    """
    seen_tick_n = -1
    polls = 0
    while not stop_event.is_set():
        res = None
        try:
            with socket.create_connection((host, port), timeout=5) as c:
                f = c.makefile("rw", encoding="utf-8", newline="\n")
                f.write(json.dumps({"op": "status"}) + "\n")
                f.flush()
                line = f.readline()
                res = json.loads(line) if line else None
        except (OSError, ValueError):
            res = None  # box unreachable / mid-restart — just retry later

        tel = (res or {}).get("telemetry") if res and res.get("ok") else None
        if tel and tel.get("thought") and tel.get("tick_n", -1) != seen_tick_n:
            seen_tick_n = tel["tick_n"]
            with write_lock:
                out_stream.write(f"\n[mind] {tel['summary']}\n")
                if tel.get("debug"):
                    out_stream.write(tel["debug"] + "\n")
                out_stream.write("> ")
                out_stream.flush()

        polls += 1
        if max_polls is not None and polls >= max_polls:
            return
        stop_event.wait(poll_interval)


def connect_repl(host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                 *, in_stream=None, out_stream=None,
                 poll_interval: float = DEFAULT_POLL_INTERVAL) -> int:
    """Interactive REPL against a remote MindServer.

    Slash commands mirror the local REPL: ``/think``, ``/render``,
    ``/quit``. Everything else is a user turn. Returns 0 on clean
    exit, 1 when the server is unreachable.
    """
    in_stream = in_stream if in_stream is not None else sys.stdin
    out_stream = out_stream if out_stream is not None else sys.stdout
    try:
        conn = socket.create_connection((host, port), timeout=10)
    except OSError as e:
        out_stream.write(
            f"[connect] ✗ cannot reach {host}:{port} — {e}\n"
            f"[connect]   is the tunnel up?  "
            f"ssh -N -L {port}:127.0.0.1:{port} root@<box>\n")
        return 1
    conn.settimeout(600)  # a slow CPU thought must not drop the line
    f = conn.makefile("rw", encoding="utf-8", newline="\n")

    def _rpc(payload: dict) -> dict:
        f.write(json.dumps(payload) + "\n")
        f.flush()
        line = f.readline()
        if not line:
            raise ConnectionError("server closed the connection")
        return json.loads(line)

    # §14.5: the DMN loop keeps thinking whether or not anyone asks —
    # a background poller (its own connection) shows those ticks live
    # instead of only via `brian logs` on the box. write_lock keeps
    # its prints from interleaving with the main loop's.
    write_lock = threading.Lock()
    stop_event = threading.Event()
    poller = threading.Thread(
        target=_poll_new_ticks,
        args=(host, port, out_stream, write_lock, stop_event, poll_interval),
        daemon=True, name="brian-connect-poller")

    try:
        hello = _rpc({"op": "ping"})
        if not hello.get("ok"):
            out_stream.write(f"[connect] ✗ bad handshake: {hello}\n")
            return 1
        poller.start()
        with write_lock:
            out_stream.write(f"[connect] mind online @ {host}:{port} — "
                             f"/think /status /render /quit\n> ")
            out_stream.flush()
        while True:
            line = in_stream.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                with write_lock:
                    out_stream.write("> ")
                    out_stream.flush()
                continue
            if line in ("/quit", "/exit", ":q"):
                break
            if line == "/think":
                res = _rpc({"op": "think"})
                tel = res.get("telemetry")
                with write_lock:
                    out_stream.write(f"[thought] {res.get('thought')}\n")
                    if tel:
                        out_stream.write(f"  {tel['summary']}\n")
                        if tel.get("debug"):
                            out_stream.write(tel["debug"] + "\n")
            elif line == "/status":
                res = _rpc({"op": "status"})
                tel = res.get("telemetry")
                with write_lock:
                    if tel:
                        out_stream.write(f"  {tel['summary']}\n")
                        if tel.get("debug"):
                            out_stream.write(tel["debug"] + "\n")
                    else:
                        out_stream.write("  (no ticks yet)\n")
            elif line == "/patterns":
                res = _rpc({"op": "patterns", "min_confidence": 0.3})
                with write_lock:
                    if not res.get("ok"):
                        out_stream.write(f"  {res.get('error')}\n")
                    else:
                        rules = res.get("rules") or []
                        if not rules:
                            out_stream.write(
                                "  (no associations mined yet — "
                                "needs more episode history)\n")
                        for r in rules:
                            flag = ("⚠ self-talk only" if r["self_referential_only"]
                                   else "grounded")
                            out_stream.write(
                                f"  {r['antecedent']} -> {r['consequent']}  "
                                f"conf={r['confidence']:.2f} "
                                f"lift={r['lift']:.2f} "
                                f"support={r['support']:.2f} "
                                f"n={r['evidence_count']}  [{flag}]\n")
            elif line == "/render":
                res = _rpc({"op": "render"})
                with write_lock:
                    out_stream.write(str(res.get("render", "")) + "\n")
            else:
                with write_lock:
                    out_stream.write("[generating…]\n")
                    out_stream.flush()
                res = _rpc({"op": "say", "text": line})
                with write_lock:
                    out_stream.write(f"{res.get('reply')}\n")
            with write_lock:
                out_stream.write("> ")
                out_stream.flush()
    except (ConnectionError, OSError) as e:
        out_stream.write(f"\n[connect] ✗ connection lost: {e}\n")
        return 1
    finally:
        stop_event.set()
        if poller.is_alive():
            poller.join(timeout=2.0)
        try:
            conn.close()
        except OSError:
            pass
    return 0


def _resolve_vast_ssh(instance_id: str, resolver=None):
    """``resolver(instance_id) -> "ssh://root@host:sshport"`` defaults
    to ``vastai ssh-url``. Returns ``(userhost, sshport)``."""
    if resolver is None:
        def resolver(iid):
            out = subprocess.run(["vastai", "ssh-url", str(iid)],
                                 capture_output=True, text=True,
                                 timeout=30)
            if out.returncode != 0:
                raise RuntimeError(
                    f"vastai ssh-url {iid} failed: {out.stderr.strip()}")
            return out.stdout.strip()
    url = resolver(instance_id)
    # ssh://root@HOST:PORT → user@host + -p PORT
    body = url.split("://", 1)[-1]
    userhost, _, sshport = body.rpartition(":")
    return userhost, sshport


def open_vast_tunnel(instance_id: str, port: int = DEFAULT_PORT,
                     *, resolver=None, spawner=None,
                     identity: Optional[str] = None):
    """Open ``ssh -N -L port:127.0.0.1:port`` to a vast instance.

    ``resolver(instance_id) -> "ssh://root@host:sshport"`` defaults to
    ``vastai ssh-url``; ``spawner(argv) -> Popen`` defaults to
    ``subprocess.Popen``. Returns the tunnel process (caller owns it).

    ``identity``: path to the private key to offer (``ssh -i``).
    Needed whenever the key filename is nonstandard (e.g. ``~/.ssh/id``)
    — ssh only auto-offers id_rsa/id_ecdsa/id_ed25519-style names.
    """
    userhost, sshport = _resolve_vast_ssh(instance_id, resolver)
    argv = ["ssh", "-N",
            "-o", "StrictHostKeyChecking=accept-new",
            "-p", sshport,
            "-L", f"{port}:127.0.0.1:{port}"]
    if identity:
        argv += ["-i", str(identity)]
    argv.append(userhost)
    if spawner is None:
        spawner = subprocess.Popen
    return spawner(argv)


def open_isaac_relay(isaac_instance_id: str, mind_instance_id: str,
                     port: int = DEFAULT_PORT, *, resolver=None,
                     spawner=None, identity: Optional[str] = None):
    """§15 remote bridge: connect an Isaac Sim box (RTX-class GPU,
    wherever it actually is) to a mind box's wire protocol (A100,
    vast.ai) WITHOUT either rented box ever holding the operator's SSH
    private key — the security model this project has used since
    §14.5 (SSH is the transport and auth layer, no public listener)
    extends to two boxes exactly the way it already covers one:

    Two ordinary tunnels, BOTH initiated from the operator's own
    machine, chained through it:

      1. FORWARD to the mind:  ``ssh -L port:127.0.0.1:port <mind>``
         (identical to :func:`open_vast_tunnel` — laptop:port now
         reaches the mind server).
      2. REVERSE to the isaac box: ``ssh -R port:127.0.0.1:port
         <isaac>`` — tells sshd on the isaac box to bind its OWN
         127.0.0.1:port and relay any connection back through this
         tunnel to the operator's machine, which leg 1 is already
         forwarding on to the mind. A ``SensoryBridge``/
         ``RemoteMindProxy`` running on the isaac box then just
         connects to its own ``127.0.0.1:port`` — same-host, no
         different from the in-process case as far as that code knows.

    Returns ``(to_mind_process, to_isaac_process)`` — caller owns both
    and must terminate them together.
    """
    if spawner is None:
        spawner = subprocess.Popen
    mind_userhost, mind_sshport = _resolve_vast_ssh(mind_instance_id, resolver)
    isaac_userhost, isaac_sshport = _resolve_vast_ssh(isaac_instance_id,
                                                       resolver)

    to_mind = ["ssh", "-N",
              "-o", "StrictHostKeyChecking=accept-new",
              "-p", mind_sshport,
              "-L", f"{port}:127.0.0.1:{port}"]
    if identity:
        to_mind += ["-i", str(identity)]
    to_mind.append(mind_userhost)

    to_isaac = ["ssh", "-N",
               "-o", "StrictHostKeyChecking=accept-new",
               "-p", isaac_sshport,
               "-R", f"{port}:127.0.0.1:{port}"]
    if identity:
        to_isaac += ["-i", str(identity)]
    to_isaac.append(isaac_userhost)

    return spawner(to_mind), spawner(to_isaac)
