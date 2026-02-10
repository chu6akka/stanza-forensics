from collections import Counter

from app.core.backends.interface import Token


def lex_metrics(tokens: list[Token]) -> dict[str, float]:
    words = [t for t in tokens if any(ch.isalpha() for ch in t.text)]
    n = max(len(words), 1)
    lemmas = [t.lemma.lower() for t in words]
    hapax = sum(1 for _, c in Counter(lemmas).items() if c == 1)
    return {
        "Лексическое разнообразие (TTR)": round(len(set(t.text.lower() for t in words)) / n, 4),
        "Лемматическое разнообразие": round(len(set(lemmas)) / n, 4),
        "Доля hapax-лемм": round(hapax / n, 4),
    }
