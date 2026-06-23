# -*- coding: utf-8 -*-
"""Contract for `neuroslm.dsl.mechanic_parser`.

The parser was shipped without tests; these pin the load-bearing behaviour
the mechanic catalog relies on. The headline contract: a comma INSIDE a
quoted scalar value (e.g. a citation "Ye et al., 2024") must not terminate
the value — bare scalars and flat string-dicts were silently truncating
at the first comma, leaking the tail into `extra`.
"""
from __future__ import annotations

from neuroslm.dsl.mechanic_parser import parse_mechanic_file


def _one(src: str):
    specs = parse_mechanic_file(src)
    assert len(specs) == 1
    return specs[0]


class TestBasicParse:
    def test_name_and_export(self):
        spec = _one('export mechanic foo { category: "attention" }')
        assert spec.name == "foo"
        assert spec.exported is True
        assert spec.category == "attention"

    def test_triple_quoted_equation(self):
        spec = _one('mechanic foo { equation: """a = b + c""" }')
        assert spec.equation == "a = b + c"

    def test_param_default_typed(self):
        spec = _one(
            'mechanic foo { params: { lr: { default: 0.02, type: "float", '
            'doc: "step size" } } }'
        )
        assert spec.params["lr"].default == 0.02
        assert spec.params["lr"].type_hint == "float"


class TestCommaInsideQuotes:
    def test_bare_scalar_summary_keeps_comma(self):
        # `summary` is a single-line quoted scalar with an internal comma.
        spec = _one('mechanic foo { summary: "noise-cancelling, dual-softmax" }')
        assert spec.summary == "noise-cancelling, dual-softmax"

    def test_str_dict_value_keeps_comma(self):
        spec = _one(
            'mechanic foo { empirical_evidence: { '
            'source: "Ye et al. (2024) Differential Transformer, Microsoft" } }'
        )
        assert spec.empirical_evidence["source"] == (
            "Ye et al. (2024) Differential Transformer, Microsoft"
        )

    def test_property_value_keeps_comma(self):
        spec = _one(
            'mechanic foo { properties: { '
            'note: "A and B, then subtract" } }'
        )
        assert spec.properties["note"] == "A and B, then subtract"

    def test_reference_citation_with_commas(self):
        spec = _one(
            'mechanic foo { references: [ '
            '"Ye, Dong, Jia et al. (2024) Differential Transformer", '
            '"Michel, Levy, Neubig (2019)" ] }'
        )
        assert spec.references == [
            "Ye, Dong, Jia et al. (2024) Differential Transformer",
            "Michel, Levy, Neubig (2019)",
        ]

    def test_no_tail_leaks_into_extra(self):
        spec = _one('mechanic foo { summary: "a, b, c" }')
        assert spec.summary == "a, b, c"
        assert not spec.extra
