from app.core.backends.interface import AnalysisResult


def quality_flags(result: AnalysisResult) -> list[str]:
    flags: list[str] = []
    words = [t for t in result.tokens if any(ch.isalpha() for ch in t.text)]
    if len(words) < 80:
        flags.append("Малый объем текста: менее 80 слов.")
    if len(result.sentences) < 5:
        flags.append("Недостаточно предложений для устойчивых оценок.")
    list_like = sum(1 for s in result.sentences if ";" in s.text or "\t" in s.text)
    if result.sentences and list_like / len(result.sentences) > 0.4:
        flags.append("Возможна табличность/списочность материала.")
    return flags
