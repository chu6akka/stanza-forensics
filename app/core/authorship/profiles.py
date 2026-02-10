from sklearn.feature_extraction.text import TfidfVectorizer

from app.core.backends.interface import Token
from app.core.metrics.pos import pos_ngrams


def build_profile(text: str, tokens: list[Token]) -> dict:
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
    x = vec.fit_transform([text])
    pos2 = pos_ngrams(tokens, 2)
    pos3 = pos_ngrams(tokens, 3)
    return {
        "char_vocab": vec.get_feature_names_out().tolist(),
        "char_weights": x.toarray()[0].tolist(),
        "pos2": pos2,
        "pos3": pos3,
    }
