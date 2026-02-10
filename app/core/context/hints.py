import regex as re


MODAL_PATTERNS = [r"\bдолжен\b", r"\bнадо\b", r"\bнельзя\b", r"\bследует\b", r"\bможно\b"]
ADDRESS_PATTERNS = [r"\bты\b", r"\bвы\b", r"\bтебя\b", r"\bвам\b"]
DISTANCE_PATTERNS = [r"\bякобы\b", r"\bтак называем\w*\b"]


def contextual_hints(text: str) -> dict[str, list[str]]:
    low = text.lower()
    hints = {"модальность": [], "обращение": [], "дистанцирование": []}
    for p in MODAL_PATTERNS:
        if re.search(p, low):
            hints["модальность"].append(p)
    for p in ADDRESS_PATTERNS:
        if re.search(p, low):
            hints["обращение"].append(p)
    for p in DISTANCE_PATTERNS:
        if re.search(p, low):
            hints["дистанцирование"].append(p)
    return hints
