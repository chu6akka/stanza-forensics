#!/usr/bin/env python3
"""POS forensics CLI using Stanza (default) or Trankit."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class TokenAnalysis:
    sentence_id: int
    token_id: int
    text: str
    lemma: str | None
    upos: str | None
    xpos: str | None
    feats: dict[str, str]
    head_id: int | None
    deprel: str | None
    head_text: str | None
    context: str
    explanation: str


class BaseBackend:
    def analyze(self, text: str) -> list[TokenAnalysis]:
        raise NotImplementedError


class StanzaBackend(BaseBackend):
    def __init__(self, lang: str, processors: str = "tokenize,mwt,pos,lemma,depparse") -> None:
        import stanza

        self.lang = lang
        self.processors = processors

        try:
            self.nlp = stanza.Pipeline(
                lang=lang,
                processors=processors,
                use_gpu=False,
                verbose=False,
            )
        except Exception:
            stanza.download(lang, processors=processors, verbose=False)
            self.nlp = stanza.Pipeline(
                lang=lang,
                processors=processors,
                use_gpu=False,
                verbose=False,
            )

    def analyze(self, text: str) -> list[TokenAnalysis]:
        doc = self.nlp(text)
        result: list[TokenAnalysis] = []

        for s_idx, sentence in enumerate(doc.sentences, start=1):
            id_to_word = {w.id: w.text for w in sentence.words}
            sent_text = " ".join(word.text for word in sentence.words)

            for word in sentence.words:
                feats = _parse_feats(word.feats)
                head_text = id_to_word.get(word.head) if word.head and word.head != 0 else "ROOT"
                result.append(
                    TokenAnalysis(
                        sentence_id=s_idx,
                        token_id=word.id,
                        text=word.text,
                        lemma=word.lemma,
                        upos=word.upos,
                        xpos=word.xpos,
                        feats=feats,
                        head_id=word.head if word.head != 0 else 0,
                        deprel=word.deprel,
                        head_text=head_text,
                        context=sent_text,
                        explanation=_build_explanation(
                            token=word.text,
                            upos=word.upos,
                            deprel=word.deprel,
                            feats=feats,
                            head=head_text,
                        ),
                    )
                )

        return result


class TrankitBackend(BaseBackend):
    def __init__(self, lang: str) -> None:
        import trankit

        self.pipeline = trankit.Pipeline(lang=lang)

    def analyze(self, text: str) -> list[TokenAnalysis]:
        parsed = self.pipeline.posdep(text)
        result: list[TokenAnalysis] = []

        for s_idx, sentence in enumerate(parsed.get("sentences", []), start=1):
            tokens = sentence.get("tokens", [])
            id_to_word = {tok.get("id"): tok.get("text") for tok in tokens}
            sent_text = " ".join(tok.get("text", "") for tok in tokens).strip()

            for tok in tokens:
                feats = _parse_feats(tok.get("feats"))
                head_id = tok.get("head", 0)
                head_text = id_to_word.get(head_id) if head_id else "ROOT"
                result.append(
                    TokenAnalysis(
                        sentence_id=s_idx,
                        token_id=tok.get("id", -1),
                        text=tok.get("text", ""),
                        lemma=tok.get("lemma"),
                        upos=tok.get("upos"),
                        xpos=tok.get("xpos"),
                        feats=feats,
                        head_id=head_id,
                        deprel=tok.get("deprel"),
                        head_text=head_text,
                        context=sent_text,
                        explanation=_build_explanation(
                            token=tok.get("text", ""),
                            upos=tok.get("upos"),
                            deprel=tok.get("deprel"),
                            feats=feats,
                            head=head_text,
                        ),
                    )
                )

        return result


def _parse_feats(raw_feats: str | None) -> dict[str, str]:
    if not raw_feats:
        return {}
    out: dict[str, str] = {}
    for pair in raw_feats.split("|"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            out[k] = v
    return out


def _build_explanation(
    token: str,
    upos: str | None,
    deprel: str | None,
    feats: dict[str, str],
    head: str | None,
) -> str:
    features = ", ".join(f"{k}={v}" for k, v in sorted(feats.items())) or "без явных морфопризнаков"
    relation = deprel or "не указана"
    pos = upos or "не определена"
    head_text = head or "неизвестно"
    return f"{token}: часть речи={pos}; связь={relation}; главный токен={head_text}; признаки={features}."


def _render_pretty(rows: list[TokenAnalysis]) -> str:
    headers = [
        "sent",
        "id",
        "token",
        "lemma",
        "upos",
        "xpos",
        "deprel",
        "head",
        "feats",
    ]

    table: list[list[str]] = [headers]
    for r in rows:
        table.append(
            [
                str(r.sentence_id),
                str(r.token_id),
                r.text,
                r.lemma or "",
                r.upos or "",
                r.xpos or "",
                r.deprel or "",
                r.head_text or "",
                ",".join(f"{k}={v}" for k, v in r.feats.items()),
            ]
        )

    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]

    def fmt(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    lines = [fmt(table[0]), "-+-".join("-" * w for w in widths)]
    lines.extend(fmt(r) for r in table[1:])
    return "\n".join(lines)


def _read_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text.strip()
    if args.input_file:
        return Path(args.input_file).read_text(encoding="utf-8").strip()
    raise ValueError("Передайте --text или --input-file")


def _create_backend(name: str, lang: str) -> BaseBackend:
    if name == "stanza":
        return StanzaBackend(lang=lang)
    if name == "trankit":
        return TrankitBackend(lang=lang)
    raise ValueError(f"Неизвестный backend: {name}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Морфологический и синтаксический forensic-анализ текста с нейросетевыми моделями."
    )
    parser.add_argument("--text", help="Текст для анализа")
    parser.add_argument("--input-file", help="Путь к файлу с текстом")
    parser.add_argument("--lang", default="ru", help="Код языка (например: ru, en, de)")
    parser.add_argument(
        "--backend",
        choices=["stanza", "trankit"],
        default="stanza",
        help="Нейросетевой backend",
    )
    parser.add_argument(
        "--format",
        choices=["json", "pretty"],
        default="pretty",
        help="Формат вывода",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    text = _read_text(args)
    backend = _create_backend(args.backend, args.lang)
    rows = backend.analyze(text)

    if args.format == "json":
        print(json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2))
    else:
        print(_render_pretty(rows))
        print("\nКонтекстные объяснения:")
        for r in rows:
            print(f"- [{r.sentence_id}:{r.token_id}] {r.explanation}")


if __name__ == "__main__":
    main()
