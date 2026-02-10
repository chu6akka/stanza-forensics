from app.core.backends.pymorphy_backend import PymorphyBackend


def test_offsets_are_consistent():
    text = "Привет, мир!"
    result = PymorphyBackend().analyze(text)
    for tok in result.tokens:
        assert 0 <= tok.start <= tok.end <= len(text)
        assert text[tok.start:tok.end] == tok.text
