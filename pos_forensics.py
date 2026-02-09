#!/usr/bin/env python3
"""Simple GUI for Russian forensic POS analysis with Stanza + fallback model."""

from __future__ import annotations

import re
import statistics
import tkinter as tk
from collections import Counter
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Iterable

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё-]+")

UPOS_RU = {
    "NOUN": "Существительное",
    "PROPN": "Имя собственное",
    "VERB": "Глагол",
    "AUX": "Вспомогательный глагол",
    "ADJ": "Прилагательное",
    "ADV": "Наречие",
    "PRON": "Местоимение",
    "NUM": "Числительное",
    "DET": "Определительное",
    "ADP": "Предлог",
    "PART": "Частица",
    "CCONJ": "Сочинительный союз",
    "SCONJ": "Подчинительный союз",
    "INTJ": "Междометие",
    "PUNCT": "Пунктуация",
    "X": "Другое",
}

PYMORPHY_POS_TO_UPOS = {
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

STYLE_MARKERS = {
    "частицы": ["же", "ли", "вот", "ну"],
    "связки": ["в общем", "короче", "на самом деле"],
    "союзы": ["потому что", "так как", "однако"],
}


@dataclass
class TokenInfo:
    text: str
    lemma: str
    pos: str
    pos_label: str
    feats: str


class Analyzer:
    """Stanza analyzer with lightweight fallback backend."""

    def __init__(self, lang: str = "ru") -> None:
        self.lang = lang
        self.backend_name: str | None = None
        self._backend = None

    def _ensure_backend(self) -> None:
        if self._backend is not None:
            return

        errors: list[str] = []

        try:
            import stanza

            try:
                self._backend = stanza.Pipeline(
                    lang=self.lang,
                    processors="tokenize,mwt,pos,lemma,depparse",
                    use_gpu=False,
                    verbose=False,
                )
            except Exception:
                stanza.download(self.lang, processors="tokenize,mwt,pos,lemma,depparse", verbose=False)
                self._backend = stanza.Pipeline(
                    lang=self.lang,
                    processors="tokenize,mwt,pos,lemma,depparse",
                    use_gpu=False,
                    verbose=False,
                )
            self.backend_name = "stanza"
            return
        except Exception as exc:  # noqa: BLE001
            errors.append(f"stanza: {exc}")

        try:
            import pymorphy3
            from razdel import tokenize

            self._backend = {"morph": pymorphy3.MorphAnalyzer(), "tokenize": tokenize}
            self.backend_name = "pymorphy3"
            return
        except Exception as exc:  # noqa: BLE001
            errors.append(f"pymorphy3: {exc}")

        raise RuntimeError(
            "Не удалось инициализировать модели анализа. "
            "Установите stanza или fallback-зависимости pymorphy3/razdel. "
            + " | ".join(errors)
        )

    def analyze(self, text: str) -> list[TokenInfo]:
        self._ensure_backend()
        if self.backend_name == "stanza":
            return self._analyze_stanza(text)
        return self._analyze_pymorphy(text)

    def _analyze_stanza(self, text: str) -> list[TokenInfo]:
        doc = self._backend(text)
        items: list[TokenInfo] = []
        for sentence in doc.sentences:
            for word in sentence.words:
                pos = _normalize_pos(word.upos, word.feats)
                items.append(TokenInfo(word.text, word.lemma or word.text.lower(), pos, _pos_label_ru(pos), word.feats or ""))
        return items

    def _analyze_pymorphy(self, text: str) -> list[TokenInfo]:
        morph = self._backend["morph"]
        tokenize = self._backend["tokenize"]

        items: list[TokenInfo] = []
        for t in tokenize(text):
            token = t.text
            if not WORD_RE.search(token):
                items.append(TokenInfo(token, token, "PUNCT", "Пунктуация", ""))
                continue

            p = morph.parse(token)[0]
            gram_pos = str(p.tag.POS) if p.tag.POS else "X"
            upos = PYMORPHY_POS_TO_UPOS.get(gram_pos, "X")
            upos = _normalize_pos(upos, f"VerbForm=Conv" if gram_pos == "GRND" else "")
            items.append(TokenInfo(token, p.normal_form, upos, _pos_label_ru(upos), str(p.tag)))

        return items


def _feats_to_dict(feats: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not feats:
        return out
    for pair in feats.split("|"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            out[k] = v
    return out


def _normalize_pos(upos: str, feats_raw: str | None) -> str:
    upos = (upos or "X").upper()
    if upos == "GERUND":
        return "VERB"

    feats = _feats_to_dict(feats_raw or "")
    if feats.get("VerbForm") == "Part":
        return "PARTICIPLE"
    if feats.get("VerbForm") == "Conv":
        return "DEEPRICHASTIE"
    return upos


def _pos_label_ru(pos: str) -> str:
    if pos == "PARTICIPLE":
        return "Причастие"
    if pos == "DEEPRICHASTIE":
        return "Деепричастие"
    return UPOS_RU.get(pos, pos)


def _word_tokens(tokens: Iterable[TokenInfo]) -> list[TokenInfo]:
    return [t for t in tokens if WORD_RE.search(t.text) and t.pos != "PUNCT"]


def _count_style_markers(text: str) -> dict[str, dict[str, int]]:
    lowered = f" {text.lower()} "
    out: dict[str, dict[str, int]] = {}
    for group, markers in STYLE_MARKERS.items():
        out[group] = {}
        for marker in markers:
            pattern = rf"(?<!\w){re.escape(marker)}(?!\w)"
            out[group][marker] = len(re.findall(pattern, lowered))
    return out


def _char_ngrams(text: str, n_values: tuple[int, ...] = (3, 4, 5), top_k: int = 10) -> dict[str, list[tuple[str, int]]]:
    prepared = re.sub(r"\s+", " ", text.lower())
    out: dict[str, list[tuple[str, int]]] = {}
    for n in n_values:
        grams = [prepared[i : i + n] for i in range(max(len(prepared) - n + 1, 0))]
        out[f"char_{n}gram"] = Counter(grams).most_common(top_k)
    return out


def _pos_ngrams(tokens: list[TokenInfo], top_k: int = 10) -> dict[str, list[tuple[str, int]]]:
    pos_seq = [t.pos_label for t in _word_tokens(tokens)]
    result: dict[str, list[tuple[str, int]]] = {}
    for n in (2, 3):
        grams = [" ".join(pos_seq[i : i + n]) for i in range(max(len(pos_seq) - n + 1, 0))]
        result[f"pos_{n}gram"] = Counter(grams).most_common(top_k)
    return result


def _punctuation_patterns(text: str) -> dict[str, float | int]:
    words = max(len(WORD_RE.findall(text)), 1)
    dash = text.count("—")
    hyphen = text.count("-") - dash
    ellipsis_unicode = text.count("…")
    ellipsis_three = text.count("...")
    return {
        "тире_эмдеш": dash,
        "дефис": max(hyphen, 0),
        "доля_тире_среди_тире_и_дефисов": round(dash / (dash + max(hyphen, 0)), 4) if (dash + max(hyphen, 0)) else 0.0,
        "троеточие_символ": ellipsis_unicode,
        "троеточие_три_точки": ellipsis_three,
        "комбо_!?": len(re.findall(r"!\?", text)),
        "комбо_??": len(re.findall(r"\?\?", text)),
        "комбо_!!": len(re.findall(r"!!", text)),
        "пробел_перед_запятой": len(re.findall(r"\s,", text)),
        "пробел_перед_точкой": len(re.findall(r"\s\.", text)),
        "знаков_препинания_на_100_слов": round((len(re.findall(r"[.,!?;:—\-…]", text)) / words) * 100, 2),
    }


def _orthography_flags(text: str) -> dict[str, float | int]:
    words = WORD_RE.findall(text)
    words_cnt = max(len(words), 1)
    lower = text.lower()
    yo = len(re.findall("ё", lower))
    e = len(re.findall("е", lower))
    return {
        "доля_ё_среди_е_ё": round(yo / (yo + e), 4) if (yo + e) else 0.0,
        "ALL_CAPS_слов": len([w for w in words if len(w) > 1 and w.isupper()]),
        "повтор_знаков_3plus": len(re.findall(r"([!?.,])\1{2,}", text)),
        "подозрительные_тся_ться": len(re.findall(r"\b\w+т(ь)?ся\b", lower)),
        "доля_caps_слов": round(len([w for w in words if len(w) > 1 and w.isupper()]) / words_cnt, 4),
    }


def calculate_metrics(tokens: list[TokenInfo], text: str) -> dict[str, object]:
    words = _word_tokens(tokens)
    total = len(words)
    if total == 0:
        return {"freq": {}, "additional": {"Всего слов": 0, "Комментарий": "Недостаточно данных для расчета показателей."}}

    pos_counts = Counter(t.pos_label for t in words)
    freq = {pos: {"count": cnt, "coefficient": round(cnt / total, 4)} for pos, cnt in sorted(pos_counts.items(), key=lambda x: (-x[1], x[0]))}

    lemmas = [t.lemma.lower() for t in words]
    unique_words = len(set(w.text.lower() for w in words))
    unique_lemmas = len(set(lemmas))
    hapax = sum(1 for _, c in Counter(lemmas).items() if c == 1)

    sentence_chunks = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sent_lens = [len(WORD_RE.findall(s)) for s in sentence_chunks if WORD_RE.findall(s)]
    avg_sent_len = round(sum(sent_lens) / len(sent_lens), 2) if sent_lens else 0.0
    sent_var = round(statistics.pvariance(sent_lens), 2) if len(sent_lens) > 1 else 0.0

    content_set = {"Существительное", "Имя собственное", "Глагол", "Причастие", "Деепричастие", "Прилагательное", "Наречие"}
    function_set = {"Предлог", "Частица", "Сочинительный союз", "Подчинительный союз", "Местоимение", "Определительное"}
    content = sum(pos_counts.get(p, 0) for p in content_set)
    function = sum(pos_counts.get(p, 0) for p in function_set)

    noun_cnt = pos_counts.get("Существительное", 0) + pos_counts.get("Имя собственное", 0)
    verb_cnt = pos_counts.get("Глагол", 0) + pos_counts.get("Вспомогательный глагол", 0)

    additional = {
        "Всего слов": total,
        "Лексическое разнообразие (TTR)": round(unique_words / total, 4),
        "Лемматическое разнообразие": round(unique_lemmas / total, 4),
        "Доля hapax-лемм": round(hapax / total, 4),
        "Средняя длина предложения (слов)": avg_sent_len,
        "Дисперсия длины предложений": sent_var,
        "Коэф. содержательные/служебные": round(content / function, 4) if function else "inf",
        "Коэф. существительные/глаголы": round(noun_cnt / verb_cnt, 4) if verb_cnt else "inf",
        "Доля причастий": round(pos_counts.get("Причастие", 0) / total, 4),
        "Доля деепричастий": round(pos_counts.get("Деепричастие", 0) / total, 4),
    }

    return {
        "freq": freq,
        "additional": additional,
        "service_profile": _count_style_markers(text),
        "char_ngrams": _char_ngrams(text),
        "pos_ngrams": _pos_ngrams(tokens),
        "punctuation_patterns": _punctuation_patterns(text),
        "orthography_flags": _orthography_flags(text),
    }


class ForensicsApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Forensic POS Analyzer (RU)")
        self.root.geometry("1080x720")

        self.analyzer = Analyzer(lang="ru")
        self._build_ui()
        self._bind_hotkeys()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="both", expand=True)

        ttk.Label(top, text="Введите русский текст для анализа:").pack(anchor="w")
        self.text_input = tk.Text(top, height=9, wrap="word")
        self.text_input.pack(fill="x", pady=(4, 10))
        self._attach_context_menu(self.text_input)

        btns = ttk.Frame(top)
        btns.pack(fill="x", pady=(0, 8))
        ttk.Button(btns, text="Анализировать (Ctrl+Enter)", command=self.run_analysis).pack(side="left")
        ttk.Button(btns, text="Очистить", command=self.clear_all).pack(side="left", padx=8)

        self.backend_var = tk.StringVar(value="backend: не инициализирован")
        ttk.Label(btns, textvariable=self.backend_var).pack(side="right")

        self.tokens_table = ttk.Treeview(top, columns=("token", "lemma", "pos", "feats"), show="headings", height=12)
        for col, width, title in [("token", 200, "Словоформа"), ("lemma", 220, "Начальная форма"), ("pos", 180, "Часть речи"), ("feats", 420, "Морфологические признаки")]:
            self.tokens_table.heading(col, text=title)
            self.tokens_table.column(col, width=width, anchor="w")
        self.tokens_table.pack(fill="both", expand=True)

        ttk.Label(top, text="Частотные коэффициенты и показатели для автороведческого анализа:").pack(anchor="w", pady=(10, 4))
        self.report = tk.Text(top, height=11, wrap="word")
        self.report.pack(fill="both", expand=True)
        self.report.configure(state="disabled")
        self._attach_context_menu(self.report)

    def _bind_hotkeys(self) -> None:
        self.root.bind_all("<Control-Return>", lambda _: self.run_analysis())
        self.text_input.bind("<Control-v>", self._paste_event)
        self.text_input.bind("<Control-V>", self._paste_event)

    def _paste_event(self, _: tk.Event) -> str:
        try:
            self.text_input.insert("insert", self.root.clipboard_get())
        except tk.TclError:
            pass
        return "break"

    def _attach_context_menu(self, widget: tk.Text) -> None:
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="Вырезать", command=lambda: widget.event_generate("<<Cut>>"))
        menu.add_command(label="Копировать", command=lambda: widget.event_generate("<<Copy>>"))
        menu.add_command(label="Вставить", command=lambda: widget.event_generate("<<Paste>>"))
        menu.add_separator()
        menu.add_command(label="Выделить всё", command=lambda: widget.tag_add("sel", "1.0", "end-1c"))
        widget.bind("<Button-3>", lambda event: menu.tk_popup(event.x_root, event.y_root))

    def clear_all(self) -> None:
        self.text_input.delete("1.0", "end")
        for row in self.tokens_table.get_children():
            self.tokens_table.delete(row)
        self._set_report("")

    def _set_report(self, text: str) -> None:
        self.report.configure(state="normal")
        self.report.delete("1.0", "end")
        self.report.insert("1.0", text)
        self.report.configure(state="disabled")

    def run_analysis(self) -> None:
        text = self.text_input.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Нет текста", "Введите текст для анализа.")
            return

        try:
            tokens = self.analyzer.analyze(text)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка моделей", f"Не удалось выполнить анализ: {exc}\n\nУбедитесь, что stanza установлена; fallback использует pymorphy3/razdel.")
            return

        self.backend_var.set(f"backend: {self.analyzer.backend_name}")
        for row in self.tokens_table.get_children():
            self.tokens_table.delete(row)

        for tok in tokens:
            if WORD_RE.search(tok.text):
                self.tokens_table.insert("", "end", values=(tok.text, tok.lemma, tok.pos_label, tok.feats))

        m = calculate_metrics(tokens, text)
        lines = ["Частотные коэффициенты частей речи (count / общее количество слов):"]
        for pos, vals in m["freq"].items():
            lines.append(f"- {pos}: {vals['count']} / coef={vals['coefficient']}")

        lines.append("\nДополнительные показатели (ориентиры для автороведческой экспертизы):")
        for k, v in m["additional"].items():
            lines.append(f"- {k}: {v}")

        lines.append("\nПрофиль служебных слов/дискурсивных маркеров:")
        for group, vals in m["service_profile"].items():
            joined = ", ".join(f"{k}={v}" for k, v in vals.items())
            lines.append(f"- {group}: {joined}")

        lines.append("\nPOS n-граммы (top):")
        for n_name, grams in m["pos_ngrams"].items():
            lines.append(f"- {n_name}: " + "; ".join(f"'{g}'={c}" for g, c in grams[:7]))

        lines.append("\nСимвольные n-граммы 3-5 (top):")
        for n_name, grams in m["char_ngrams"].items():
            lines.append(f"- {n_name}: " + "; ".join(f"'{g}'={c}" for g, c in grams[:7]))

        lines.append("\nПунктуационные паттерны:")
        for k, v in m["punctuation_patterns"].items():
            lines.append(f"- {k}: {v}")

        lines.append("\nОрфографические признаки:")
        for k, v in m["orthography_flags"].items():
            lines.append(f"- {k}: {v}")

        self._set_report("\n".join(lines))

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    ForensicsApp().run()
