# -*- coding: utf-8 -*-
"""Full research-mechanics catalog — every `*.neuro` mechanic/dynamic/structure.

The discovery loop needs to know *what already exists* so it only reports
genuinely novel mechanics. The repo already carries ~74 mechanic specs (rich
`when_to_use` / `not_for` / `properties` blocks — the human-facing semantic
description language) under `mechanics/`, `dynamics/`, and `structures/`.
`MechanicCatalog` loads all of them through the existing `mechanic_parser`, so
"all currently existing research mechanics" is a live enumeration, not a
hand-maintained list of 13.
"""
from neuroslm.genetic.catalog import (
    MechanicCatalog, load_catalog, catalog_names,
)


class TestCatalogLoads:
    def test_loads_the_full_mechanic_set(self):
        cat = load_catalog()
        # 54 mechanics + 11 dynamics + 9 structures ≈ 70+
        assert len(cat) >= 60

    def test_has_headline_mechanics(self):
        names = set(load_catalog().names())
        for n in ("rope", "swiglu", "rmsnorm", "alibi", "gqa", "mamba_ssm",
                  "flash_attention"):
            assert n in names, n

    def test_includes_dynamics_and_structures(self):
        names = set(load_catalog().names())
        assert "adamw" in names or "lion" in names       # dynamics/
        assert "prenorm_block" in names or "moe_block" in names  # structures/


class TestCatalogQuery:
    def test_get_returns_a_spec(self):
        spec = load_catalog().get("rmsnorm")
        assert spec is not None
        assert spec.category == "normalization"
        assert spec.summary

    def test_get_missing_returns_none(self):
        assert load_catalog().get("does_not_exist_xyz") is None

    def test_categories_are_populated(self):
        cats = load_catalog().categories()
        assert "normalization" in cats
        assert len(cats) >= 3

    def test_by_category_groups_specs(self):
        grouped = load_catalog().by_category()
        norm = grouped.get("normalization", [])
        assert any(s.name == "rmsnorm" for s in norm)


class TestCatalogDescribe:
    def test_describe_is_human_readable(self):
        text = load_catalog().describe("rope")
        assert isinstance(text, str) and len(text) > 20
        assert "rope" in text.lower()

    def test_describe_missing_is_graceful(self):
        text = load_catalog().describe("nope_not_here")
        assert "unknown" in text.lower() or text == ""


class TestPrepopulated2024_2026:
    """The web-verified 2024-2026 mechanics prepopulated into the catalog."""

    NEW = [
        "native_sparse_attention", "moba", "selective_attention",
        "forgetting_attention", "softpick", "yarn", "longrope", "deltanet",
        "gated_deltanet", "titans", "xlstm", "rwkv7", "ngpt", "qk_clip",
        "soap", "schedule_free", "adam_mini", "galore", "grokfast",
        "loss_free_balancing", "fine_grained_experts", "mup",
    ]

    def test_all_present(self):
        names = set(load_catalog().names())
        missing = [n for n in self.NEW if n not in names]
        assert not missing, missing

    def test_each_has_summary_category_and_reference(self):
        cat = load_catalog()
        for n in self.NEW:
            s = cat.get(n)
            assert s is not None, n
            assert s.category and s.summary, n
            assert s.references, n          # every entry cites its source

    def test_catalog_spans_the_new_families(self):
        cat = load_catalog()
        assert cat.get("mamba_ssm") is not None      # legacy still there
        assert cat.get("native_sparse_attention").category == "attention"
        assert cat.get("galore").category == "optimizer"


class TestPriorArtNames:
    def test_catalog_names_are_prior_art(self):
        names = catalog_names()
        assert "rope" in names and "swiglu" in names
        assert len(names) >= 60

    def test_catalog_names_matches_loaded_catalog(self):
        assert catalog_names() == set(load_catalog().names())
