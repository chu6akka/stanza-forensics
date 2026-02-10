from collections import Counter

from app.core.backends.interface import Token


def pos_profile(tokens: list[Token]) -> dict[str, dict[str, float | int]]:
    words = [t for t in tokens if any(ch.isalpha() for ch in t.text)]
    total = max(len(words), 1)
    counts = Counter(t.pos for t in words)
    return {k: {"количество": v, "коэффициент": round(v / total, 4)} for k, v in counts.items()}


def pos_ngrams(tokens: list[Token], n: int = 2) -> dict[str, int]:
    seq = [t.pos for t in tokens if any(ch.isalpha() for ch in t.text)]
    grams = Counter(" ".join(seq[i : i + n]) for i in range(max(len(seq) - n + 1, 0)))
    return dict(grams)
