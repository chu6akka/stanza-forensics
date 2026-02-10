from dataclasses import dataclass


@dataclass
class Span:
    start: int
    end: int
    tags: list[str]
    notes: str


def create_span(start: int, end: int, tags: list[str], notes: str = "") -> Span:
    if end <= start:
        raise ValueError("Некорректные границы фрагмента")
    return Span(start=start, end=end, tags=tags, notes=notes)
