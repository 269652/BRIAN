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


def _dispatch(daemon: Any, msg: dict) -> dict:
    op = msg.get("op")
    if op == "ping":
        return {"ok": True, "pong": True}
    if op == "say":
        return {"ok": True, "reply": daemon.respond(str(msg.get("text", "")))}
    if op == "think":
        return {"ok": True, "thought": daemon.think_once()}
    if op == "render":
        return {"ok": True, "render": daemon.render()}
    return {"ok": False, "error": f"unknown op {op!r}"}


# ── Client ───────────────────────────────────────────────────────────


def connect_repl(host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                 *, in_stream=None, out_stream=None) -> int:
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

    try:
        hello = _rpc({"op": "ping"})
        if not hello.get("ok"):
            out_stream.write(f"[connect] ✗ bad handshake: {hello}\n")
            return 1
        out_stream.write(f"[connect] mind online @ {host}:{port} — "
                         f"/think /render /quit\n> ")
        out_stream.flush()
        while True:
            line = in_stream.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                out_stream.write("> ")
                out_stream.flush()
                continue
            if line in ("/quit", "/exit", ":q"):
                break
            if line == "/think":
                res = _rpc({"op": "think"})
                out_stream.write(f"[thought] {res.get('thought')}\n")
            elif line == "/render":
                res = _rpc({"op": "render"})
                out_stream.write(str(res.get("render", "")) + "\n")
            else:
                out_stream.write("[generating…]\n")
                out_stream.flush()
                res = _rpc({"op": "say", "text": line})
                out_stream.write(f"{res.get('reply')}\n")
            out_stream.write("> ")
            out_stream.flush()
    except (ConnectionError, OSError) as e:
        out_stream.write(f"\n[connect] ✗ connection lost: {e}\n")
        return 1
    finally:
        try:
            conn.close()
        except OSError:
            pass
    return 0


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
