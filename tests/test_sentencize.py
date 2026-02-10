from app.core.backends.pymorphy_backend import PymorphyBackend


def test_sentencize_abbreviations():
    text = "г. Москва, т.к. это важно. Дата: 01.01.2026."
    result = PymorphyBackend().analyze(text)
    assert len(result.sentences) >= 2
