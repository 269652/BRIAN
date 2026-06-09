# -*- coding: utf-8 -*-
"""TDD: ``HypothesisStore`` + ``DiscoveryStore`` — file-backed CRUD.

The store is the on-disk authority for hypothesis / discovery records.
Layout it must maintain:

    hypothesis/
      H001_phi_monotone.md
      H002_ood_gap_decrease.md
      index.json                ← machine-readable list of every record

    discoveries/
      D001_add_dopamine_pfc.md
      index.json

Every ``save()`` rewrites the targeted ``.md`` file AND rewrites the
sibling ``index.json`` so the index never drifts from the on-disk
record set. ``load_all()`` reads the directory authoritatively (the
index is a cache, the ``.md`` files are the source of truth).
"""
from __future__ import annotations

from pathlib import Path

import pytest


# ───────────────────────────────────────────────────────────────────
# 1. HypothesisStore — basic CRUD
# ───────────────────────────────────────────────────────────────────

class TestHypothesisStoreCRUD:
    """``HypothesisStore(root)`` must persist records as
    front-matter Markdown files and keep an ``index.json`` in sync."""

    def test_save_then_load_by_id(self, tmp_path: Path):
        from neuroslm.discoveries.records import HypothesisRecord
        from neuroslm.discoveries.store import HypothesisStore
        store = HypothesisStore(tmp_path)
        h = HypothesisRecord(
            id="H001", title="Φ monotone",
            statement_md=r"$\Phi(\theta') \ge \Phi(\theta)$",
            theorem_name="Brian.PhiMonotone",
        )
        store.save(h)
        loaded = store.get("H001")
        assert loaded == h

    def test_save_writes_markdown_file_at_expected_path(self, tmp_path: Path):
        from neuroslm.discoveries.records import HypothesisRecord
        from neuroslm.discoveries.store import HypothesisStore
        store = HypothesisStore(tmp_path)
        h = HypothesisRecord(
            id="H007", title="Trunk gradient isolation",
            statement_md="trunk grad invariant to aux weights",
            theorem_name="Brian.TrunkIsolation",
        )
        store.save(h)
        # Expected filename: H007_trunk_gradient_isolation.md
        expected = tmp_path / "H007_trunk_gradient_isolation.md"
        assert expected.exists(), \
            f"expected .md at {expected}, got dir: {list(tmp_path.iterdir())}"
        text = expected.read_text(encoding="utf-8")
        assert "Brian.TrunkIsolation" in text

    def test_save_updates_index_json(self, tmp_path: Path):
        import json
        from neuroslm.discoveries.records import HypothesisRecord
        from neuroslm.discoveries.store import HypothesisStore
        store = HypothesisStore(tmp_path)
        h = HypothesisRecord(
            id="H010", title="Improvement gate Welch",
            statement_md=r"$p < \alpha \wedge |\mathrm{eff}| \ge \mathrm{min}$",
            theorem_name="Brian.ImprovementGateWelch",
        )
        store.save(h)
        idx_path = tmp_path / "index.json"
        assert idx_path.exists()
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        ids = {entry["id"] for entry in idx["records"]}
        assert "H010" in ids

    def test_list_all_returns_every_record(self, tmp_path: Path):
        from neuroslm.discoveries.records import HypothesisRecord
        from neuroslm.discoveries.store import HypothesisStore
        store = HypothesisStore(tmp_path)
        for i in range(3):
            store.save(HypothesisRecord(
                id=f"H00{i+1}", title=f"hyp {i+1}",
                statement_md="x",
                theorem_name=f"Brian.H{i+1}",
            ))
        all_ids = sorted(r.id for r in store.list_all())
        assert all_ids == ["H001", "H002", "H003"]

    def test_directory_is_source_of_truth(self, tmp_path: Path):
        """Hand-edit the ``.md`` file and re-load: changes are picked
        up even though ``index.json`` is stale."""
        from neuroslm.discoveries.records import HypothesisRecord
        from neuroslm.discoveries.store import HypothesisStore
        store = HypothesisStore(tmp_path)
        h = HypothesisRecord(id="H020", title="t", statement_md="x",
                             theorem_name="Brian.T")
        store.save(h)
        # Hand-edit body of the md file
        md_path = tmp_path / "H020_t.md"
        text = md_path.read_text(encoding="utf-8")
        text = text.replace("Brian.T", "Brian.T_v2")
        md_path.write_text(text, encoding="utf-8")
        # Re-instantiate the store and ask
        store2 = HypothesisStore(tmp_path)
        loaded = store2.get("H020")
        assert loaded.theorem_name == "Brian.T_v2"

    def test_next_id_finds_next_free_slot(self, tmp_path: Path):
        from neuroslm.discoveries.records import HypothesisRecord
        from neuroslm.discoveries.store import HypothesisStore
        store = HypothesisStore(tmp_path)
        store.save(HypothesisRecord(id="H001", title="t", statement_md="x",
                                    theorem_name="Brian.T1"))
        store.save(HypothesisRecord(id="H002", title="t", statement_md="x",
                                    theorem_name="Brian.T2"))
        # H003 should be next free
        assert store.next_id() == "H003"

    def test_get_missing_raises(self, tmp_path: Path):
        from neuroslm.discoveries.store import HypothesisStore
        store = HypothesisStore(tmp_path)
        with pytest.raises(KeyError, match="H999"):
            store.get("H999")


# ───────────────────────────────────────────────────────────────────
# 2. DiscoveryStore — same contract, different record type
# ───────────────────────────────────────────────────────────────────

class TestDiscoveryStoreCRUD:
    def _make(self, did="D001"):
        from neuroslm.discoveries.records import DiscoveryRecord
        return DiscoveryRecord(
            id=did, title=f"discovery {did}",
            mechanism_md="discovered a thing",
            mutation_chain=["add_modulation"],
            parent_dna_id="parent",
            fitness_before={"ood_ppl": 250.0},
            fitness_after={"ood_ppl": 230.0},
            generation=3,
            theorem_name=f"Brian.Discoveries.{did}_X",
        )

    def test_save_and_load(self, tmp_path: Path):
        from neuroslm.discoveries.store import DiscoveryStore
        store = DiscoveryStore(tmp_path)
        d = self._make("D001")
        store.save(d)
        loaded = store.get("D001")
        assert loaded == d

    def test_save_updates_index(self, tmp_path: Path):
        import json
        from neuroslm.discoveries.store import DiscoveryStore
        store = DiscoveryStore(tmp_path)
        store.save(self._make("D005"))
        idx = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
        ids = {entry["id"] for entry in idx["records"]}
        assert "D005" in ids

    def test_promote_persists_state(self, tmp_path: Path):
        """After a successful DNA splice, the on-disk record must
        reflect the ``dna_integrated`` flag — auditability requires
        the bit to survive a process restart."""
        from neuroslm.discoveries.store import DiscoveryStore
        store = DiscoveryStore(tmp_path)
        d = self._make("D009")
        d.proof_status = "verified"
        store.save(d)
        d2 = store.get("D009")
        d2.promote_to_dna(at="2026-06-09T11:00:00Z")
        store.save(d2)
        # Restart
        store_fresh = DiscoveryStore(tmp_path)
        loaded = store_fresh.get("D009")
        assert loaded.dna_integrated is True
        assert loaded.dna_integrated_at == "2026-06-09T11:00:00Z"

    def test_next_id_skips_used_slots(self, tmp_path: Path):
        from neuroslm.discoveries.store import DiscoveryStore
        store = DiscoveryStore(tmp_path)
        store.save(self._make("D001"))
        store.save(self._make("D003"))  # gap intentional
        # Engine policy: hand back the next monotonically-increasing free id,
        # never re-use a hole (gives a stable temporal sort).
        assert store.next_id() == "D004"
