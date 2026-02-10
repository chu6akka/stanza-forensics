from dataclasses import dataclass


@dataclass
class Token:
    text: str
    lemma: str
    pos: str
    feats: dict[str, str]
    start: int
    end: int


@dataclass
class Sentence:
    text: str
    start: int
    end: int


@dataclass
class AnalysisResult:
    backend: str
    tokens: list[Token]
    sentences: list[Sentence]
    warnings: list[str]


class Backend:
    name = "base"

    def analyze(self, text: str) -> AnalysisResult:
        raise NotImplementedError
