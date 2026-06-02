import torch

from neuroslm.dsl.nn_lang import build_dsl_language_cortex, build_language_model


def test_build_language_model_stashes_last_hidden() -> None:
    m = build_language_model(
        vocab=257,
        d_model=64,
        depth=2,
        n_heads=4,
        max_ctx=64,
    )
    ids = torch.randint(0, 257, (2, 16))
    logits = m(ids)

    assert logits.shape == (2, 16, 257)
    assert hasattr(m, "_last_hidden")
    assert m._last_hidden is not None
    assert m._last_hidden.shape == (2, 16, 64)
    # Must stay attached for PR2 aux gradients.
    assert m._last_hidden.requires_grad


def test_build_dsl_language_cortex_stashes_last_hidden() -> None:
    m = build_dsl_language_cortex(
        vocab=257,
        d_model=64,
        depth=2,
        n_heads=4,
        max_ctx=64,
        dropout=0.0,
    )
    ids = torch.randint(0, 257, (2, 16))
    logits = m(ids)

    assert logits.shape == (2, 16, 257)
    assert hasattr(m, "_last_hidden")
    assert m._last_hidden is not None
    assert m._last_hidden.shape == (2, 16, 64)
    # Must stay attached for PR2 aux gradients.
    assert m._last_hidden.requires_grad
