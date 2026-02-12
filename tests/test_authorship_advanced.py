from app.core.authorship.advanced_analysis import compare_texts_detailed


def test_detects_exact_fragments_and_scores_shape() -> None:
    sample = "Я бегала сегодня утром по набережной и слушала музыку. Потом пила кофе!"
    examined = "Сегодня я бегала утром по набережной и слушала музыку. Потом пила кофе!!!"

    result = compare_texts_detailed(sample, examined)

    assert "scores" in result
    assert "exact_fragments" in result
    assert "paraphrase_candidates" in result
    assert result["scores"]["overall_similarity"] > 0
    assert any("потом пила кофе" in item["fragment"] for item in result["exact_fragments"])


def test_detects_transliteration_patterns() -> None:
    sample = "Это был krutoi день, потом done и go home"
    examined = "Да, krutoi вайб, только что done"

    result = compare_texts_detailed(sample, examined)

    shared = result["transliteration"]["shared_patterns"]
    assert any("krutoi" in item for item in shared)
    assert any("done" in item for item in shared)
