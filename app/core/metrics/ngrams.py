from collections import Counter


def char_ngrams(text: str, n_min: int = 3, n_max: int = 5, top_k: int = 50) -> dict[str, list[tuple[str, int]]]:
    text = " ".join(text.lower().split())
    out: dict[str, list[tuple[str, int]]] = {}
    for n in range(n_min, n_max + 1):
        grams = Counter(text[i : i + n] for i in range(max(len(text) - n + 1, 0)))
        out[f"char_{n}"] = grams.most_common(top_k)
    return out
