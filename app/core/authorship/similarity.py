import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compare_char_ngrams(text_a: str, text_b: str) -> float:
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
    x = vec.fit_transform([text_a, text_b])
    return float(cosine_similarity(x[0], x[1])[0, 0])


def compare_frequency_distributions(freq_a: dict[str, float], freq_b: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(freq_a) | set(freq_b))
    a = np.array([freq_a.get(k, 0.0) for k in keys], dtype=float)
    b = np.array([freq_b.get(k, 0.0) for k in keys], dtype=float)
    if a.sum() == 0 or b.sum() == 0:
        return {"cosine": 0.0, "js_divergence": 1.0}
    a /= a.sum()
    b /= b.sum()
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    js = float(jensenshannon(a, b, base=2.0))
    return {"cosine": cosine, "js_divergence": js}
