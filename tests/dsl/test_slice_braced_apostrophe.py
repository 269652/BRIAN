"""Regression: stray apostrophe in a `#` comment must not swallow `}` closers.

Without the fix, `_slice_braced` treats `'` as a string opener and absorbs
the rest of the file, dropping all subsequent decls from the export table.
"""
from neuroslm.dsl.multifile import _slice_braced, parse_module
from pathlib import Path


def test_slice_braced_ignores_apostrophe_in_comment():
    src = '{\n    a: { x: 1 }, # there\'s a stray apostrophe\n    b: { y: 2 }\n}'
    inner, end = _slice_braced(src, 0)
    assert end == len(src)
    assert 'b: { y: 2 }' in inner


def test_slice_braced_ignores_brace_in_comment():
    src = '{\n    a: 1, # closing } in a comment must not decrement depth\n}'
    inner, end = _slice_braced(src, 0)
    assert end == len(src)


def test_slice_braced_still_balances_normally():
    src = '{ a: { b: { c: 1 } } }'
    inner, end = _slice_braced(src, 0)
    assert end == len(src)


def test_parse_module_extracts_formal_phi_integration():
    """End-to-end: the real constraints.neuro must expose all 3 formal specs."""
    path = Path('architectures/rcc_bowtie/lib/constraints.neuro')
    if not path.exists():
        return  # skip when running in a stripped repo
    m = parse_module(path.read_text(encoding='utf-8'), path)
    assert 'sheaf_narrative_consistency' in m.exports
    assert 'formal_phi_integration' in m.exports
    assert 'formal_bowtie_topology' in m.exports
