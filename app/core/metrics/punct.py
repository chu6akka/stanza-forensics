import regex as re


def punct_metrics(text: str) -> dict[str, float | int]:
    words = max(len(re.findall(r"\p{L}+", text)), 1)
    return {
        "Тире em-dash": text.count("—"),
        "Дефис": text.count("-") - text.count("—"),
        "Троеточие символ": text.count("…"),
        "Троеточие три точки": text.count("..."),
        "Комбинация ?!": len(re.findall(r"\?!|!\?", text)),
        "Пробел перед запятой": len(re.findall(r"\s,", text)),
        "Плотность пунктуации на 100 слов": round(len(re.findall(r"[.,!?;:—-]", text)) / words * 100, 2),
    }
