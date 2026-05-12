"""Web backend dispatch + config sanity."""

import pytest


def test_all_expected_backends_registered():
    from web.app import _BACKENDS
    assert set(_BACKENDS.keys()) == {"words", "trocr", "trocr-whole", "claude"}


def test_each_backend_is_callable():
    from web.app import _BACKENDS
    for name, fn in _BACKENDS.items():
        assert callable(fn), f"backend {name!r} is not callable"


def test_seed_weights_path_format():
    from config import seed_weights_path
    p = seed_weights_path(42)
    assert p.endswith("crnn_gnhk_seed42_best.pth")
    assert "checkpoints" in p


def test_charset_consistency():
    """encode_text must produce indices that round-trip via IDX2CHAR."""
    from config import CHARS, CHAR2IDX, IDX2CHAR, encode_text
    assert len(CHARS) == len(CHAR2IDX) == len(IDX2CHAR)
    for c in CHARS:
        idx = CHAR2IDX[c]
        assert IDX2CHAR[idx] == c
    # Unknown chars get dropped, not raise
    assert encode_text("xyz!!ÿ") == [CHAR2IDX[c] for c in "xyz!!" if c in CHAR2IDX]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
