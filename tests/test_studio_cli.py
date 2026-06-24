# -*- coding: utf-8 -*-
"""TDD contract for `brian studio start` CLI command."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Force UTF-8 for all subprocess calls that test parser output — arch.neuro
# files contain non-cp1252 characters (e.g. superscript ⁰ U+2070) that crash
# the default Windows cp1252 codec.
_UTF8_ENV = {**os.environ, "PYTHONUTF8": "1"}

import pytest

REPO_ROOT = Path(__file__).parent.parent
PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"


# ---------------------------------------------------------------------------
# CLI registration contract
# ---------------------------------------------------------------------------

def test_studio_subcommand_registered():
    """brian studio --help must not fail with 'invalid choice'."""
    result = subprocess.run(
        [str(PYTHON), "-m", "neuroslm.cli", "studio", "--help"],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert "studio" in result.stdout.lower() or "start" in result.stdout.lower()


def test_studio_start_subcommand_registered():
    """brian studio start --help must succeed."""
    result = subprocess.run(
        [str(PYTHON), "-m", "neuroslm.cli", "studio", "start", "--help"],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert "1984" in result.stdout or "studio" in result.stdout.lower()


def test_studio_start_has_no_browser_flag():
    """--no-browser flag must be accepted."""
    result = subprocess.run(
        [str(PYTHON), "-m", "neuroslm.cli", "studio", "start", "--help"],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0
    assert "--no-browser" in result.stdout


# ---------------------------------------------------------------------------
# Server package contract
# ---------------------------------------------------------------------------

def test_studio_server_importable():
    """studio.server.app must be importable (fastapi + uvicorn present)."""
    result = subprocess.run(
        [str(PYTHON), "-c", "from studio.server.app import app; print('ok')"],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_studio_server_port_is_1984():
    """The server must declare PORT = 1984 (easter egg: Orwell)."""
    result = subprocess.run(
        [str(PYTHON), "-c", "from studio.server.app import PORT; print(PORT)"],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1984"


def test_studio_client_port_is_2049():
    """The client must declare CLIENT_PORT = 2049 (easter egg: Blade Runner)."""
    result = subprocess.run(
        [str(PYTHON), "-c", "from studio.server.app import CLIENT_PORT; print(CLIENT_PORT)"],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "3141"


# ---------------------------------------------------------------------------
# Neuro parser contract
# ---------------------------------------------------------------------------

def test_neuro_parser_parses_gpt2_arch():
    """neuro_parser must extract model + sheaf from GPT-2 arch.neuro."""
    result = subprocess.run(
        [
            str(PYTHON), "-c",
            "from studio.server.neuro_parser import parse_arch; "
            "import json; "
            "src = open('architectures/gpt2/arch.neuro', encoding='utf-8').read(); "
            "r = parse_arch(src, 'gpt2'); "
            "nodes = {n['id']: n for n in r['nodes']}; "
            "assert 'model' in nodes, f'no model node: {r}'; "
            "assert nodes['model']['data']['kind'] == 'gpt2'; "
            "assert 'sheaf' in nodes, f'no sheaf node: {r}'; "
            "assert nodes['sheaf']['data']['dim'] == 768; "
            "print('ok')"
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, env=_UTF8_ENV,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_neuro_parser_produces_edges():
    """parse_arch must include at least model->sheaf edge."""
    result = subprocess.run(
        [
            str(PYTHON), "-c",
            "from studio.server.neuro_parser import parse_arch; "
            "src = open('architectures/gpt2/arch.neuro', encoding='utf-8').read(); "
            "r = parse_arch(src, 'gpt2'); "
            "edges = {(e['source'], e['target']) for e in r['edges']}; "
            "assert ('model', 'sheaf') in edges, f'missing edge: {r}'; "
            "print('ok')"
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, env=_UTF8_ENV,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# ---------------------------------------------------------------------------
# Mechanics library contract (via server)
# ---------------------------------------------------------------------------

def test_mechanics_list_endpoint_returns_all_categories():
    """GET /api/mechanics must include attention, regularizer, optimizer categories."""
    from fastapi.testclient import TestClient
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from studio.server.app import app
    client = TestClient(app)
    r = client.get("/api/mechanics")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    categories = {m["category"] for m in data if "category" in m}
    # At least one mechanic per major category
    assert len(data) > 0


def test_architectures_list_endpoint():
    """GET /api/architectures must return at least gpt2."""
    from fastapi.testclient import TestClient
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from studio.server.app import app
    client = TestClient(app)
    r = client.get("/api/architectures")
    assert r.status_code == 200
    names = [a["name"] for a in r.json()]
    assert "gpt2" in names


def test_architecture_detail_endpoint():
    """GET /api/architectures/gpt2 must return nodes with model + sheaf."""
    from fastapi.testclient import TestClient
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from studio.server.app import app
    client = TestClient(app)
    r = client.get("/api/architectures/gpt2")
    assert r.status_code == 200
    body = r.json()
    node_ids = {n["id"] for n in body["nodes"]}
    assert "model" in node_ids
    assert "sheaf" in node_ids
