from pymorphy3 import MorphAnalyzer
from razdel import sentenize, tokenize

from app.core.backends.interface import AnalysisResult, Backend, Sentence, Token


POS_MAP = {
    "NOUN": "NOUN",
    "ADJF": "ADJ",
    "ADJS": "ADJ",
    "COMP": "ADV",
    "VERB": "VERB",
    "INFN": "VERB",
    "PRTF": "PARTICIPLE",
    "PRTS": "PARTICIPLE",
    "GRND": "DEEPRICHASTIE",
    "NUMR": "NUM",
    "ADVB": "ADV",
    "NPRO": "PRON",
    "PRED": "ADV",
    "PREP": "ADP",
    "CONJ": "CCONJ",
    "PRCL": "PART",
    "INTJ": "INTJ",
}


class PymorphyBackend(Backend):
    name = "pymorphy3"

    def __init__(self) -> None:
        self.morph = MorphAnalyzer()

    def analyze(self, text: str) -> AnalysisResult:
        tokens: list[Token] = []
        for tok in tokenize(text):
            p = self.morph.parse(tok.text)[0]
            pos = POS_MAP.get(str(p.tag.POS), "X")
            feats = {"OpenCorpora": str(p.tag)}
            tokens.append(Token(tok.text, p.normal_form, pos, feats, tok.start, tok.stop))
        sents = [Sentence(s.text, s.start, s.stop) for s in sentenize(text)]
        return AnalysisResult(backend=self.name, tokens=tokens, sentences=sents, warnings=[])
