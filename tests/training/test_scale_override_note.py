# -*- coding: utf-8 -*-
"""The scale block silently discards CLI --seq_len/--batch — warn about it."""
from neuroslm.train_dsl import _scale_override_note


def test_no_note_when_cli_matches_scale():
    assert _scale_override_note("s", 512, 1, 512, 1) is None


def test_note_flags_both_overrides():
    note = _scale_override_note("30m_p4", cli_seq=2048, cli_batch=16,
                                eff_seq=512, eff_batch=1)
    assert note is not None
    assert "2048→512" in note and "16→1" in note
    assert "30m_p4" in note
    assert "SEQ_LEN" in note and "BATCH_SIZE" in note


def test_note_flags_only_the_changed_field():
    note = _scale_override_note("s", cli_seq=512, cli_batch=16, eff_seq=512, eff_batch=1)
    assert note is not None
    assert "--batch 16→1" in note
    assert "seq_len" not in note        # seq unchanged → not mentioned
