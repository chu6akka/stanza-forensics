from natasha import Doc, MorphVocab, NewsEmbedding, NewsMorphTagger, Segmenter
from razdel import sentenize

from app.core.backends.interface import AnalysisResult, Backend, Sentence, Token


class NatashaBackend(Backend):
    name = "natasha"

    def __init__(self) -> None:
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.morph_tagger = NewsMorphTagger(NewsEmbedding())

    def analyze(self, text: str) -> AnalysisResult:
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        tokens: list[Token] = []
        for tok in doc.tokens:
            tok.lemmatize(self.morph_vocab)
            feats = tok.feats if isinstance(tok.feats, dict) else {}
            tokens.append(
                Token(
                    text=tok.text,
                    lemma=tok.lemma or tok.text.lower(),
                    pos=(tok.pos or "X").upper(),
                    feats=feats,
                    start=tok.start,
                    end=tok.stop,
                )
            )
        sents = [Sentence(s.text, s.start, s.stop) for s in sentenize(text)]
        return AnalysisResult(backend=self.name, tokens=tokens, sentences=sents, warnings=[])
