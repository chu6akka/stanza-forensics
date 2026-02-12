from __future__ import annotations

from collections import Counter

import regex
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.authorship.similarity import compare_char_ngrams, compare_frequency_distributions

WORD_RE = regex.compile(r"\p{L}+[\p{L}\p{M}'’-]*", flags=regex.IGNORECASE)
SENT_RE = regex.compile(r"[^.!?\n]+[.!?…]*", flags=regex.IGNORECASE)
PUNCT_RE = regex.compile(r"[.!?,:;—\-()\[\]{}…\"'«»]+")

LAT_TO_CYR = str.maketrans(
    {
        "a": "а",
        "b": "б",
        "c": "с",
        "d": "д",
        "e": "е",
        "f": "ф",
        "g": "г",
        "h": "х",
        "i": "и",
        "j": "й",
        "k": "к",
        "l": "л",
        "m": "м",
        "n": "н",
        "o": "о",
        "p": "п",
        "q": "к",
        "r": "р",
        "s": "с",
        "t": "т",
        "u": "у",
        "v": "в",
        "w": "в",
        "x": "кс",
        "y": "ы",
        "z": "з",
    }
)


def _normalize_word(word: str) -> str:
    return word.lower().replace("ё", "е")


def _words(text: str) -> list[str]:
    return [_normalize_word(w) for w in WORD_RE.findall(text)]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_RE.findall(text) if s.strip()]


def _build_word_ngrams(words: list[str], n: int) -> Counter[str]:
    if len(words) < n:
        return Counter()
    return Counter(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))


def _find_exact_fragments(text_a: str, text_b: str, min_len: int = 3, top_k: int = 12) -> list[dict]:
    wa = _words(text_a)
    wb = _words(text_b)
    found: dict[str, int] = {}
    for n in range(8, min_len - 1, -1):
        a_ngrams = _build_word_ngrams(wa, n)
        b_ngrams = _build_word_ngrams(wb, n)
        common = set(a_ngrams) & set(b_ngrams)
        for phrase in common:
            if any(phrase in existed for existed in found):
                continue
            found[phrase] = min(a_ngrams[phrase], b_ngrams[phrase])
    ranked = sorted(found.items(), key=lambda item: (-len(item[0].split()), -item[1], item[0]))[:top_k]
    return [{"fragment": fragment, "matches": count, "words": len(fragment.split())} for fragment, count in ranked]


def _punctuation_profile(text: str) -> Counter[str]:
    profile = Counter(ch for ch in text if ch in ".!?;,:-—…()\"'«»")
    profile["triple_dots"] = len(regex.findall(r"\.\.\.|…", text))
    profile["multi_q"] = len(regex.findall(r"\?{2,}", text))
    profile["multi_e"] = len(regex.findall(r"!{2,}", text))
    profile["caps_words"] = len(regex.findall(r"\b\p{Lu}{2,}\b", text))
    return profile


def _punctuation_sequences(text: str) -> Counter[str]:
    chunks = [c for c in PUNCT_RE.findall(text) if c]
    return Counter(chunks)


def _transliteration_candidates(words: list[str]) -> set[str]:
    candidates = set()
    for word in words:
        if regex.search(r"[a-z]", word) and regex.search(r"[а-я]", word):
            candidates.add(word)
        elif regex.search(r"[a-z]", word):
            translit = word.translate(LAT_TO_CYR)
            if translit != word:
                candidates.add(f"{word}→{translit}")
    return candidates


def _paraphrase_pairs(text_a: str, text_b: str, top_k: int = 8) -> list[dict]:
    sa = _sentences(text_a)
    sb = _sentences(text_b)
    if not sa or not sb:
        return []

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    matrix = vec.fit_transform(sa + sb)
    ma = matrix[: len(sa)]
    mb = matrix[len(sa) :]
    sims = cosine_similarity(ma, mb)

    pairs = []
    for i, row in enumerate(sims):
        j = int(row.argmax())
        score = float(row[j])
        token_overlap = fuzz.token_set_ratio(sa[i], sb[j]) / 100.0
        if 0.45 <= score < 0.95 and token_overlap < 0.92:
            pairs.append(
                {
                    "source": sa[i],
                    "target": sb[j],
                    "semantic_similarity": round(score, 4),
                    "lexical_overlap": round(token_overlap, 4),
                }
            )
    pairs.sort(key=lambda item: (-item["semantic_similarity"], item["lexical_overlap"]))
    return pairs[:top_k]


def _style_indicators(text: str) -> dict[str, float]:
    words = _words(text)
    word_count = max(len(words), 1)
    sent_count = max(len(_sentences(text)), 1)
    emoticons = len(regex.findall(r"[:;=8xX][-^']?[)D(P/\\|oO]", text))
    mixed_script = len(regex.findall(r"\b(?=\w*[a-z])(?=\w*[а-я]).+?\b", text.lower()))
    return {
        "avg_sentence_len": round(len(words) / sent_count, 2),
        "comma_per_100_words": round(text.count(",") * 100 / word_count, 2),
        "dash_per_100_words": round((text.count("-") + text.count("—")) * 100 / word_count, 2),
        "exclamation_per_100_words": round(text.count("!") * 100 / word_count, 2),
        "question_per_100_words": round(text.count("?") * 100 / word_count, 2),
        "uppercase_ratio": round(sum(1 for ch in text if ch.isupper()) / max(len(text), 1), 4),
        "emoji_like": emoticons,
        "mixed_script_tokens": mixed_script,
    }


def _compare_numeric_profiles(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(a) | set(b))
    diffs = [abs(a.get(key, 0.0) - b.get(key, 0.0)) for key in keys]
    avg_diff = sum(diffs) / max(len(diffs), 1)
    bounded = 1.0 / (1.0 + avg_diff)
    return {"style_similarity": round(bounded, 4), "avg_feature_gap": round(avg_diff, 4)}


def compare_texts_detailed(text_a: str, text_b: str) -> dict:
    text_a = text_a or ""
    text_b = text_b or ""

    exact_fragments = _find_exact_fragments(text_a, text_b)
    paraphrases = _paraphrase_pairs(text_a, text_b)

    punct_a = _punctuation_profile(text_a)
    punct_b = _punctuation_profile(text_b)
    punct_dist = compare_frequency_distributions(dict(punct_a), dict(punct_b))

    punct_seq_a = _punctuation_sequences(text_a)
    punct_seq_b = _punctuation_sequences(text_b)
    punct_seq_dist = compare_frequency_distributions(dict(punct_seq_a), dict(punct_seq_b))

    style_a = _style_indicators(text_a)
    style_b = _style_indicators(text_b)
    style_cmp = _compare_numeric_profiles(style_a, style_b)

    translit_overlap = sorted(_transliteration_candidates(_words(text_a)) & _transliteration_candidates(_words(text_b)))
    translit_unique = sorted(_transliteration_candidates(_words(text_b)) - _transliteration_candidates(_words(text_a)))

    char_cosine = compare_char_ngrams(text_a, text_b)
    exact_intensity = min(sum(item["words"] * item["matches"] for item in exact_fragments) / 30.0, 1.0)
    paraphrase_intensity = min(len(paraphrases) / 8.0, 1.0)
    punctuation_similarity = (punct_dist["cosine"] + punct_seq_dist["cosine"]) / 2

    final_score = 0.35 * char_cosine + 0.25 * exact_intensity + 0.2 * paraphrase_intensity + 0.2 * punctuation_similarity

    return {
        "scores": {
            "char_ngram_cosine": round(char_cosine, 4),
            "exact_fragment_intensity": round(exact_intensity, 4),
            "paraphrase_intensity": round(paraphrase_intensity, 4),
            "punctuation_similarity": round(punctuation_similarity, 4),
            "style_similarity": style_cmp["style_similarity"],
            "overall_similarity": round(final_score, 4),
        },
        "exact_fragments": exact_fragments,
        "paraphrase_candidates": paraphrases,
        "punctuation": {
            "distribution": punct_dist,
            "sequence_distribution": punct_seq_dist,
            "profile_a": dict(punct_a),
            "profile_b": dict(punct_b),
        },
        "style": {
            "text_a": style_a,
            "text_b": style_b,
            "comparison": style_cmp,
        },
        "transliteration": {
            "shared_patterns": translit_overlap,
            "patterns_in_examined_only": translit_unique,
        },
        "conclusion_draft": _build_conclusion_draft(
            final_score=final_score,
            exact_fragments=exact_fragments,
            paraphrases=paraphrases,
            punct_similarity=punctuation_similarity,
            style_similarity=style_cmp["style_similarity"],
            translit_overlap=translit_overlap,
        ),
    }


def _build_conclusion_draft(
    *,
    final_score: float,
    exact_fragments: list[dict],
    paraphrases: list[dict],
    punct_similarity: float,
    style_similarity: float,
    translit_overlap: list[str],
) -> list[str]:
    level = "низкий"
    if final_score >= 0.7:
        level = "высокий"
    elif final_score >= 0.45:
        level = "средний"

    notes = [f"Интегральный показатель сходства — {final_score:.3f} ({level} уровень)."]
    notes.append(f"Обнаружено полностью идентичных фрагментов: {len(exact_fragments)}.")
    notes.append(f"Выявлено вероятных перефразов/синонимических замен: {len(paraphrases)}.")
    notes.append(f"Сходство пунктуационной манеры: {punct_similarity:.3f}; сходство стилевых индикаторов: {style_similarity:.3f}.")
    if translit_overlap:
        notes.append(f"Совпадают признаки транслитерации/смешанной графики: {', '.join(translit_overlap[:5])}.")
    else:
        notes.append("Совпадающие признаки транслитерации не выявлены или выражены слабо.")
    notes.append("Вывод является автоматизированной подсказкой и требует экспертной автороведческой интерпретации.")
    return notes
