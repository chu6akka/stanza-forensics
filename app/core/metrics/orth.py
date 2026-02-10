import regex as re


def orth_metrics(text: str) -> dict[str, float | int | list[dict[str, str]]]:
    low = text.lower()
    yo = low.count("ё")
    e = low.count("е")
    tsya_contexts = []
    for m in re.finditer(r"\b\p{L}+т(ь)?ся\b", low):
        a = max(0, m.start() - 20)
        b = min(len(text), m.end() + 20)
        tsya_contexts.append({"форма": m.group(0), "контекст": text[a:b]})
    return {
        "Доля ё среди е+ё": round(yo / (yo + e), 4) if (yo + e) else 0.0,
        "Повторы букв (3+)": len(re.findall(r"(\p{L})\1{2,}", low)),
        "Повторы знаков": len(re.findall(r"([!?.,])\1+", text)),
        "Кандидаты тся/ться": tsya_contexts,
    }
