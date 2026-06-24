# -*- coding: utf-8 -*-
"""Brian Studio — REST + MCP server.

Easter-egg ports:
  PORT        = 1984   (Orwell — the language model that controls language)
  CLIENT_PORT = 2049   (Blade Runner 2049 — near-future AI)
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PORT = 1984
CLIENT_PORT = 3141  # π — because language models approximate the infinite

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO))

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Brian Studio",
    description=(
        "Visual language model editor. "
        f"Server port {PORT} (Orwell 1984). "
        f"Client port {CLIENT_PORT} (Blade Runner 2049)."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        f"http://localhost:{CLIENT_PORT}",
        f"http://127.0.0.1:{CLIENT_PORT}",
        "http://localhost:3000",
        "http://localhost:3141",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

from studio.server.routers.architectures import router as arch_router
from studio.server.routers.mechanics import router as mech_router
from studio.server.routers.deploy import router as deploy_router
from studio.server.routers.inference import router as infer_router

app.include_router(arch_router, prefix="/api/architectures")
app.include_router(mech_router, prefix="/api/mechanics")
app.include_router(deploy_router, prefix="/api/deploy")
app.include_router(infer_router, prefix="/api/inference")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "port": PORT, "client_port": CLIENT_PORT}


# ---------------------------------------------------------------------------
# MCP server (SSE transport at /mcp)
# ---------------------------------------------------------------------------

try:
    from mcp.server.fastmcp import FastMCP
    from studio.server.neuro_parser import parse_arch
    from studio.server.routers.architectures import _list_arch_dirs, _ARCH_ROOT

    mcp = FastMCP(
        "Brian Studio",
        instructions=(
            "Brian Studio MCP — tools for browsing, editing, and deploying "
            "language model architectures defined in the .neuro DSL."
        ),
    )

    @mcp.tool()
    def list_architectures() -> list[str]:
        """Return names of all available architectures."""
        return [p.name for p in _list_arch_dirs()]

    @mcp.tool()
    def get_architecture_source(name: str) -> str:
        """Return the raw .neuro source for a named architecture."""
        f = _ARCH_ROOT / name / "arch.neuro"
        if not f.exists():
            return f"ERROR: architecture '{name}' not found"
        return f.read_text(encoding="utf-8")

    @mcp.tool()
    def get_architecture_nodes(name: str) -> dict:
        """Return parsed nodes + edges for the visual editor."""
        f = _ARCH_ROOT / name / "arch.neuro"
        if not f.exists():
            return {"error": f"architecture '{name}' not found"}
        return parse_arch(f.read_text(encoding="utf-8"), name)

    @mcp.tool()
    def save_architecture(name: str, source: str) -> dict:
        """Save updated .neuro source for an architecture."""
        arch_dir = _ARCH_ROOT / name
        arch_dir.mkdir(parents=True, exist_ok=True)
        (arch_dir / "arch.neuro").write_text(source, encoding="utf-8")
        return {"saved": True, "name": name}

    @mcp.tool()
    def list_mechanics() -> list[str]:
        """Return names of all available mechanics, structures, and dynamics."""
        from studio.server.routers.mechanics import _get_specs
        return [s["name"] for s in _get_specs()]

    @mcp.tool()
    def get_mechanic(name: str) -> dict:
        """Return full specification for a named mechanic."""
        from studio.server.routers.mechanics import _get_specs
        specs = _get_specs()
        match = next((s for s in specs if s["name"] == name), None)
        return match or {"error": f"mechanic '{name}' not found"}

    @mcp.tool()
    def deploy_architecture(arch: str, steps: int = 10000, label: str = "") -> dict:
        """Deploy an architecture to vast.ai for training. Requires user confirmation."""
        import subprocess
        cmd = [sys.executable, "-m", "neuroslm.cli", "deploy"]
        if arch:
            cmd.append(arch)
        cmd += ["--steps", str(steps)]
        if label:
            cmd += ["--label", label]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=_REPO, timeout=120)
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-1000:],
        }

    @mcp.tool()
    def run_inference(prompt: str, arch: str = "gpt2", max_new_tokens: int = 128) -> str:
        """Run text generation on a pretrained model."""
        from studio.server.routers.inference import InferRequest, _build_infer_script
        import subprocess
        body = InferRequest(prompt=prompt, arch=arch, max_new_tokens=max_new_tokens)
        script = _build_infer_script(body)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, cwd=_REPO, timeout=120
        )
        return result.stdout or result.stderr

    # Mount MCP SSE endpoint
    app.mount("/mcp", mcp.sse_app())

except ImportError:
    pass  # mcp not installed — REST API still works fine
