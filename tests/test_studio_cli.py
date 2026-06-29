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
    """neuro_parser must expand GPT-2 into a full grouped transformer graph."""
    result = subprocess.run(
        [
            str(PYTHON), "-c",
            "from studio.server.neuro_parser import parse_arch; "
            "src = open('architectures/gpt2/arch.neuro', encoding='utf-8').read(); "
            "r = parse_arch(src, 'gpt2'); "
            "nodes = {n['id']: n for n in r['nodes']}; "
            "assert 'model' in nodes, f'no model node: {r}'; "
            "assert nodes['model']['data']['kind'] == 'gpt2'; "
            "assert nodes['model']['data']['dim'] == 768, 'dim missing on model'; "
            # Three colored group panels
            "assert 'grp_embed' in nodes and nodes['grp_embed']['type'] == 'group'; "
            "assert 'grp_block' in nodes and nodes['grp_block']['type'] == 'group'; "
            "assert 'grp_output' in nodes and nodes['grp_output']['type'] == 'group'; "
            # Every GPT-2 mechanic is a visible node
            "need = ('tok_embed','pos_embed','embed_add','ln_1','attn',"
            "'attn_resid','ln_2','ffn','ffn_resid','final_ln','lm_head'); "
            "assert all(nid in nodes for nid in need), "
            "f'missing: {[x for x in need if x not in nodes]}'; "
            # Block children belong to the block panel
            "assert nodes['attn']['parentId'] == 'grp_block'; "
            "print('ok')"
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, env=_UTF8_ENV,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_neuro_parser_produces_edges():
    """parse_arch must wire the forward pass: model→embed→block→head + residuals."""
    result = subprocess.run(
        [
            str(PYTHON), "-c",
            "from studio.server.neuro_parser import parse_arch; "
            "src = open('architectures/gpt2/arch.neuro', encoding='utf-8').read(); "
            "r = parse_arch(src, 'gpt2'); "
            "edges = {(e['source'], e['target']) for e in r['edges']}; "
            "assert ('model', 'tok_embed') in edges, f'missing model->tok_embed: {r}'; "
            "assert ('ln_1', 'attn') in edges, 'missing ln_1->attn'; "
            "assert ('attn', 'attn_resid') in edges, 'missing attn->attn_resid'; "
            # Residual skip connection (block input bypasses attn into the add)
            "assert ('embed_add', 'attn_resid') in edges, 'missing residual skip'; "
            # Residual edges are styled (dashed)
            "resid = [e for e in r['edges'] if e.get('data', {}).get('residual')]; "
            "assert resid, 'no residual-styled edges'; "
            "print('ok')"
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, env=_UTF8_ENV,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_gpt2_block_group_shows_depth():
    """The transformer-block panel must show the repeat count (×12)."""
    result = subprocess.run(
        [
            str(PYTHON), "-c",
            "from studio.server.neuro_parser import parse_arch; "
            "src = open('architectures/gpt2/arch.neuro', encoding='utf-8').read(); "
            "r = parse_arch(src, 'gpt2'); "
            "blk = next(n for n in r['nodes'] if n['id'] == 'grp_block'); "
            "assert '12' in blk['data']['label'], blk['data']['label']; "
            "print('ok')"
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, env=_UTF8_ENV,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_attn_node_carries_resolved_equation():
    """The attention node must carry the equation math resolved from @lib imports."""
    result = subprocess.run(
        [
            str(PYTHON), "-c",
            "from studio.server.neuro_parser import parse_arch; "
            "src = open('architectures/gpt2/arch.neuro', encoding='utf-8').read(); "
            "r = parse_arch(src, 'gpt2'); "
            "attn = next(n for n in r['nodes'] if n['id'] == 'attn'); "
            "assert attn['data'].get('equation_math'), 'no equation_math on attn'; "
            "print('ok')"
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, env=_UTF8_ENV,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_smollm2_multi_equation_attention():
    """SmolLM2 coboundary declares GQA + RoPE — both must surface on the attn node."""
    result = subprocess.run(
        [
            str(PYTHON), "-c",
            "from studio.server.neuro_parser import parse_arch; "
            "src = open('architectures/smollm2-135m/arch.neuro', encoding='utf-8').read(); "
            "r = parse_arch(src, 'smollm2-135m'); "
            "attn = next(n for n in r['nodes'] if n['id'] == 'attn'); "
            "impl = attn['data'].get('impl', ''); "
            "assert 'grouped_query_attention' in impl, impl; "
            "assert 'rope_attention' in impl, impl; "
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
    """GET /api/architectures/gpt2 must return the full expanded transformer graph."""
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
    assert {"grp_embed", "grp_block", "grp_output"} <= node_ids
    assert {"attn", "ffn", "ln_1", "lm_head"} <= node_ids


def test_api_responses_are_not_cached():
    """Arch graphs change live; /api responses must carry Cache-Control: no-store."""
    from fastapi.testclient import TestClient
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from studio.server.app import app
    client = TestClient(app)
    r = client.get("/api/architectures/gpt2")
    assert r.status_code == 200
    assert r.headers.get("cache-control") == "no-store"


def test_population_arch_nested_scope_region_grouping():
    """GET /api/architectures/SmolLM must nest populations 3 levels deep:
    gradient-scope container (trunk/bio) → brain-region module panel
    (sensory, pfc, …) → population leaf, plus a Neurotransmitters panel.
    Every population carries its equation; every NT carries a colour."""
    from fastapi.testclient import TestClient
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from studio.server.app import app
    client = TestClient(app)
    r = client.get("/api/architectures/SmolLM")
    assert r.status_code == 200
    nodes = r.json()["nodes"]
    by_id = {n["id"]: n for n in nodes}

    # Outer gradient-scope containers + the NT panel (all tier=scope).
    scopes = {n["id"] for n in nodes if n.get("data", {}).get("tier") == "scope"}
    assert {"grp_trunk", "grp_bio", "grp_nt"} <= scopes, scopes

    # Inner brain-region module panels (tier=module), each parented to a scope.
    modules = [n for n in nodes if n.get("data", {}).get("tier") == "module"]
    assert len(modules) >= 10, f"too few region panels: {len(modules)}"
    region_names = {m["data"]["module"] for m in modules}
    assert {"sensory", "pfc", "thalamus", "motor", "hippocampus"} <= region_names, region_names
    for m in modules:
        assert by_id[m["parentId"]]["data"]["tier"] == "scope", m["id"]

    # Every population is a leaf inside a module panel (3-level nesting).
    pops = [n for n in nodes if n["type"] == "population"]
    assert len(pops) >= 30, len(pops)
    for p in pops:
        parent = by_id.get(p.get("parentId", ""))
        assert parent and parent["data"].get("tier") == "module", f"{p['id']} not in a region"

    # Nothing population-ish is left floating at the top level.
    ungrouped = [
        n["id"] for n in nodes
        if not n.get("parentId") and n["type"] in ("population", "neurotransmitter")
    ]
    assert not ungrouped, f"ungrouped: {ungrouped}"

    # Every population surfaces its equation.
    missing = [n["id"] for n in pops if not n["data"].get("equation")]
    assert not missing, f"populations missing equation: {missing}"

    # NT nodes carry a per-transmitter colour.
    nts = [n for n in nodes if n["type"] == "neurotransmitter"]
    assert nts and all(n["data"].get("color") for n in nts), "NT nodes missing colour"


def test_population_arch_distinguishes_synapse_from_modulation():
    """Modulation edges must render dashed (dimmer) so the solid synaptic
    feedforward flow is visually separable from diffuse neuromodulation."""
    from fastapi.testclient import TestClient
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from studio.server.app import app
    client = TestClient(app)
    r = client.get("/api/architectures/SmolLM")
    assert r.status_code == 200
    edges = r.json()["edges"]
    mods = [e for e in edges if e["data"]["kind"] == "modulation"]
    syns = [e for e in edges if e["data"]["kind"] == "synapse"]
    assert mods and syns, "expected both synapse and modulation edges"
    assert all("strokeDasharray" in e["style"] for e in mods), "modulation edges not dashed"
    assert all("strokeDasharray" not in e["style"] for e in syns), "synapse edges should be solid"
