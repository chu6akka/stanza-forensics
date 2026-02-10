#!/usr/bin/env python3
"""GUI for Russian forensic POS analysis with Natasha + pymorphy3 backends."""

from __future__ import annotations

import colorsys
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
    "DET": "Определительное слово",
    "ADP": "Предлог",
    "PART": "Частица",
    "CCONJ": "Сочинительный союз",
    "SCONJ": "Подчинительный союз",
    "INTJ": "Междометие",
    "PUNCT": "Пунктуация",
    "X": "Другое",
}

FEAT_NAME_RU = {
    "Animacy": "Одушевлённость",
    "Aspect": "Вид",
    "Case": "Падеж",
    "Degree": "Степень сравнения",
    "Gender": "Род",
    "Mood": "Наклонение",
    "Number": "Число",
    "Person": "Лицо",
    "Tense": "Время",
    "VerbForm": "Форма глагола",
    "Voice": "Залог",
}

FEAT_VALUE_RU = {
    "Anim": "одушевлённое",
    "Inan": "неодушевлённое",
    "Imp": "несовершенный",
    "Perf": "совершенный",
    "Nom": "именительный",
    "Gen": "родительный",
    "Dat": "дательный",
    "Acc": "винительный",
    "Ins": "творительный",
    "Loc": "предложный",
    "Sing": "единственное",
    "Plur": "множественное",
    "Masc": "мужской",
    "Fem": "женский",
    "Neut": "средний",
    "Pres": "настоящее",
    "Past": "прошедшее",
    "Fut": "будущее",
    "Ind": "изъявительное",
    "Part": "причастие",
    "Conv": "деепричастие",
    "Act": "действительный",
    "Pass": "страдательный",
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

# Научно-точные русские соответствия OpenCorpora-граммемам.
PYMORPHY_GRAMMEME_RU = {
    "NOUN": "существительное",
    "ADJF": "прилагательное (полная форма)",
    "ADJS": "прилагательное (краткая форма)",
    "COMP": "компаратив",
    "VERB": "глагол",
    "INFN": "инфинитив",
    "PRTF": "причастие (полное)",
    "PRTS": "причастие (краткое)",
    "GRND": "деепричастие",
    "NUMR": "числительное",
    "ADVB": "наречие",
    "NPRO": "местоимение-существительное",
    "PRED": "предикатив",
    "PREP": "предлог",
    "CONJ": "союз",
    "PRCL": "частица",
    "INTJ": "междометие",
    "anim": "одушевлённое",
    "inan": "неодушевлённое",
    "masc": "мужской род",
    "femn": "женский род",
    "neut": "средний род",
    "sing": "единственное число",
    "plur": "множественное число",
    "nomn": "именительный падеж",
    "gent": "родительный падеж",
    "datv": "дательный падеж",
    "accs": "винительный падеж",
    "ablt": "творительный падеж",
    "loct": "предложный падеж",
    "voct": "звательный падеж",
    "gen1": "первый родительный",
    "gen2": "второй родительный",
    "acc2": "второй винительный",
    "loc1": "первый предложный",
    "loc2": "второй предложный",
    "perf": "совершенный вид",
    "impf": "несовершенный вид",
    "tran": "переходный",
    "intr": "непереходный",
    "pres": "настоящее время",
    "past": "прошедшее время",
    "futr": "будущее время",
    "indc": "изъявительное наклонение",
    "impr": "повелительное наклонение",
    "incl": "включительная форма",
    "excl": "исключительная форма",
    "actv": "действительный залог",
    "pssv": "страдательный залог",
    "1per": "1-е лицо",
    "2per": "2-е лицо",
    "3per": "3-е лицо",
}

STYLE_MARKERS = {
    "частицы": ["же", "ли", "вот", "ну"],
    "связки": ["в общем", "короче", "на самом деле"],
    "союзы": ["потому что", "так как", "однако"],
}

INTENTIONAL_TYPO_MARKERS = [
    "щас",
    "счас",
    "ваще",
    "кароч",
    "чо",
    "тока",
    "шо",
    "канеш",
    "превед",
]


@dataclass
class TokenInfo:
    text: str
    lemma: str
    pos: str
    pos_label: str
    feats: str


class Analyzer:
    def __init__(self, lang: str = "ru") -> None:
        self.lang = lang
        self.backend_name: str | None = None
        self._backend = None
        self.last_errors: dict[str, str] = {}

    def _init_natasha(self):
        from natasha import Doc, MorphVocab, NewsEmbedding, NewsMorphTagger, Segmenter

        return {
            "Doc": Doc,
            "segmenter": Segmenter(),
            "morph_vocab": MorphVocab(),
            "morph_tagger": NewsMorphTagger(NewsEmbedding()),
        }

    def _init_pymorphy(self):
        import pymorphy3
        from razdel import tokenize

        return {"morph": pymorphy3.MorphAnalyzer(), "tokenize": tokenize}

    def _ensure_backend(self, preferred: str = "natasha") -> tuple[str | None, str | None]:
        """Return (preferred_backend_name, fail_reason_if_preferred_failed)."""
        if self._backend is not None and self.backend_name == preferred:
            return preferred, None

        orders = {
            "natasha": ["natasha", "pymorphy3"],
            "pymorphy3": ["pymorphy3", "natasha"],
        }
        order = orders.get(preferred, orders["natasha"])
        self.last_errors = {}

        for name in order:
            try:
                if name == "natasha":
                    self._backend = self._init_natasha()
                else:
                    self._backend = self._init_pymorphy()
                self.backend_name = name
                fail_reason = self.last_errors.get(preferred)
                return preferred, fail_reason
            except Exception as exc:  # noqa: BLE001
                self.last_errors[name] = str(exc)

        raise RuntimeError("Не удалось инициализировать бэкенды Natasha и pymorphy3. " + " | ".join(f"{k}: {v}" for k, v in self.last_errors.items()))

    def analyze(self, text: str, preferred_backend: str = "natasha") -> tuple[list[TokenInfo], str | None]:
        _, fail_reason = self._ensure_backend(preferred_backend)
        if self.backend_name == "natasha":
            return self._analyze_natasha(text), fail_reason
        return self._analyze_pymorphy(text), fail_reason

    def _analyze_natasha(self, text: str) -> list[TokenInfo]:
        Doc = self._backend["Doc"]
        doc = Doc(text)
        doc.segment(self._backend["segmenter"])
        doc.tag_morph(self._backend["morph_tagger"])

        out: list[TokenInfo] = []
        for t in doc.tokens:
            t.lemmatize(self._backend["morph_vocab"])
            if not WORD_RE.search(t.text):
                out.append(TokenInfo(t.text, t.text, "PUNCT", "Пунктуация", "—"))
                continue
            feats = getattr(t, "feats", {}) or {}
            feats_ud = "|".join(f"{k}={v}" for k, v in feats.items()) if isinstance(feats, dict) else ""
            pos = _normalize_pos(getattr(t, "pos", "X"), feats_ud)
            out.append(TokenInfo(t.text, t.lemma or t.text.lower(), pos, _pos_label_ru(pos), _format_feats_ru(feats_ud)))
        return out

    def _analyze_pymorphy(self, text: str) -> list[TokenInfo]:
        out: list[TokenInfo] = []
        for t in self._backend["tokenize"](text):
            if not WORD_RE.search(t.text):
                out.append(TokenInfo(t.text, t.text, "PUNCT", "Пунктуация", "—"))
                continue
            p = self._backend["morph"].parse(t.text)[0]
            gram_pos = str(p.tag.POS) if p.tag.POS else "X"
            upos = _normalize_pos(PYMORPHY_POS_TO_UPOS.get(gram_pos, "X"), "VerbForm=Conv" if gram_pos == "GRND" else "")
            out.append(TokenInfo(t.text, p.normal_form, upos, _pos_label_ru(upos), _format_pymorphy_tag_ru(str(p.tag))))
        return out


def _feats_to_dict(feats: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in feats.split("|") if feats else []:
        if "=" in pair:
            k, v = pair.split("=", 1)
            out[k] = v
    return out


def _format_feats_ru(feats: str) -> str:
    d = _feats_to_dict(feats)
    return "—" if not d else "; ".join(f"{FEAT_NAME_RU.get(k, k)}: {FEAT_VALUE_RU.get(v, v)}" for k, v in d.items())


def _format_pymorphy_tag_ru(tag: str) -> str:
    grammemes = [g for g in re.split(r"[\s,]+", tag.strip()) if g]
    if not grammemes:
        return "—"
    translated = [PYMORPHY_GRAMMEME_RU.get(g, g) for g in grammemes]
    return "; ".join(translated)


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
    return "Причастие" if pos == "PARTICIPLE" else "Деепричастие" if pos == "DEEPRICHASTIE" else UPOS_RU.get(pos, pos)




def _diagnose_backend_error(reason: str) -> str:
    r = (reason or "").lower()
    tips: list[str] = []
    if "no module named 'pkg_resources'" in r:
        tips.append("Не найден модуль pkg_resources (обычно отсутствует setuptools).")
        tips.append("Решение: pip install setuptools")
    if "no module named" in r and "natasha" in r:
        tips.append("Не установлен пакет Natasha.")
        tips.append("Решение: pip install natasha")
    if "no module named" in r and "pymorphy3" in r:
        tips.append("Не установлен пакет pymorphy3.")
        tips.append("Решение: pip install pymorphy3 razdel")
    if "cannot import" in r or "dll" in r:
        tips.append("Возможна проблема бинарных зависимостей/окружения Python.")
        tips.append("Решение: создать новое venv и переустановить зависимости.")
    if not tips:
        tips.append("Проверьте зависимости и окружение Python (venv).")
    return "\n".join(f"• {t}" for t in tips)

def _word_tokens(tokens: Iterable[TokenInfo]) -> list[TokenInfo]:
    return [t for t in tokens if WORD_RE.search(t.text) and t.pos != "PUNCT"]


def _count_style_markers(text: str) -> dict[str, dict[str, int]]:
    lowered = text.lower()
    return {g: {m: len(re.findall(rf"(?<!\w){re.escape(m)}(?!\w)", lowered)) for m in marks} for g, marks in STYLE_MARKERS.items()}


def _register_profile(text: str) -> dict[str, float | int]:
    words = [w for w in re.findall(r"[A-Za-zА-Яа-яЁё]+", text) if w]
    total = max(len(words), 1)
    upper = sum(1 for w in words if len(w) > 1 and w.isupper())
    lower = sum(1 for w in words if w.islower())
    mixed = total - upper - lower
    return {
        "Слов всего": len(words),
        "Доля ВЕРХНЕГО РЕГИСТРА": round(upper / total, 4),
        "Доля нижнего регистра": round(lower / total, 4),
        "Доля смешанного регистра": round(mixed / total, 4),
    }


def _duplications_and_typos(text: str) -> dict[str, float | int | str]:
    lowered = text.lower()
    repeated_letters = len(re.findall(r"([а-яёa-z])\1{2,}", lowered))
    repeated_punct = len(re.findall(r"([!?.,])\1{1,}", text))
    typo_hits = {m: len(re.findall(rf"(?<!\w){re.escape(m)}(?!\w)", lowered)) for m in INTENTIONAL_TYPO_MARKERS}
    typo_total = sum(typo_hits.values())
    return {
        "Повторы букв (>=3 подряд)": repeated_letters,
        "Повторы знаков препинания (>=2 подряд)": repeated_punct,
        "Намеренные разговорные/искажённые формы (всего)": typo_total,
        "Найденные формы": ", ".join(f"{k}:{v}" for k, v in typo_hits.items() if v > 0) or "не обнаружены",
    }


def calculate_metrics(tokens: list[TokenInfo], text: str) -> dict[str, object]:
    words = _word_tokens(tokens)
    total = len(words)
    if total == 0:
        return {
            "частоты": {},
            "дополнительно": {"Всего слов": 0, "Комментарий": "Недостаточно данных для расчета показателей."},
            "профиль_служебных_слов": {},
            "регистровый_профиль": _register_profile(text),
            "повторы_и_опечатки": _duplications_and_typos(text),
        }

    pos_counts = Counter(t.pos_label for t in words)
    freq = {p: {"количество": c, "коэффициент": round(c / total, 4)} for p, c in sorted(pos_counts.items(), key=lambda x: (-x[1], x[0]))}

    lemmas = [t.lemma.lower() for t in words]
    sent_lens = [len(WORD_RE.findall(s)) for s in re.split(r"[.!?]+", text) if WORD_RE.findall(s)]

    content_set = {"Существительное", "Имя собственное", "Глагол", "Причастие", "Деепричастие", "Прилагательное", "Наречие"}
    function_set = {"Предлог", "Частица", "Сочинительный союз", "Подчинительный союз", "Местоимение", "Определительное слово"}
    content = sum(pos_counts.get(p, 0) for p in content_set)
    function = sum(pos_counts.get(p, 0) for p in function_set)
    noun_cnt = pos_counts.get("Существительное", 0) + pos_counts.get("Имя собственное", 0)
    verb_cnt = pos_counts.get("Глагол", 0) + pos_counts.get("Вспомогательный глагол", 0)

    additional = {
        "Всего слов": total,
        "Лексическое разнообразие (TTR)": round(len(set(w.text.lower() for w in words)) / total, 4),
        "Лемматическое разнообразие": round(len(set(lemmas)) / total, 4),
        "Доля hapax-лемм": round(sum(1 for _, c in Counter(lemmas).items() if c == 1) / total, 4),
        "Средняя длина предложения (слов)": round(sum(sent_lens) / len(sent_lens), 2) if sent_lens else 0.0,
        "Дисперсия длины предложений": round(statistics.pvariance(sent_lens), 2) if len(sent_lens) > 1 else 0.0,
        "Коэффициент содержательные/служебные": round(content / function, 4) if function else "∞",
        "Коэффициент существительные/глаголы": round(noun_cnt / verb_cnt, 4) if verb_cnt else "∞",
    }

    return {
        "частоты": freq,
        "дополнительно": additional,
        "профиль_служебных_слов": _count_style_markers(text),
        "регистровый_профиль": _register_profile(text),
        "повторы_и_опечатки": _duplications_and_typos(text),
    }


class ForensicsApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Forensic POS Analyzer (RU)")
        self.root.geometry("1200x850")

        self.analyzer = Analyzer(lang="ru")
        self._row_to_span: dict[str, tuple[int, int]] = {}
        self._last_freq: dict[str, dict[str, float | int]] = {}

        self._build_ui()
        self._bind_hotkeys()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="both", expand=True)

        ttk.Label(top, text="Введите русский текст для анализа:").pack(anchor="w")

        controls = ttk.Frame(top)
        controls.pack(fill="x", pady=(2, 4))
        self.expand_input = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Расширить поле ввода", variable=self.expand_input, command=self._toggle_input_size).pack(side="left")

        self.text_input = tk.Text(top, height=16, wrap="word")
        self.text_input.pack(fill="x", pady=(2, 10))
        self.text_input.tag_configure("hover_token", background="#fff176")
        self._attach_context_menu(self.text_input)

        btns = ttk.Frame(top)
        btns.pack(fill="x", pady=(0, 8))
        ttk.Button(btns, text="Анализировать (Ctrl+Enter)", command=self.run_analysis).pack(side="left")
        ttk.Button(btns, text="Показать круговую диаграмму", command=self.show_pie_chart).pack(side="left", padx=8)
        ttk.Label(btns, text="Режим:").pack(side="left", padx=(12, 4))

        self.backend_choice = tk.StringVar(value="natasha")
        ttk.Combobox(btns, width=18, textvariable=self.backend_choice, state="readonly", values=["natasha", "pymorphy3"]).pack(side="left")

        self.backend_var = tk.StringVar(value="бэкенд: не инициализирован")
        self.backend_hint_var = tk.StringVar(value="По умолчанию используется Natasha. Резерв — pymorphy3.")
        ttk.Label(btns, textvariable=self.backend_var).pack(side="right")
        ttk.Label(top, textvariable=self.backend_hint_var, foreground="#444").pack(anchor="w")

        self.tokens_table = ttk.Treeview(top, columns=("token", "lemma", "pos", "feats"), show="headings", height=12)
        for col, width, title in [
            ("token", 170, "Словоформа"),
            ("lemma", 210, "Начальная форма"),
            ("pos", 190, "Часть речи"),
            ("feats", 560, "Морфологические признаки (рус.)"),
        ]:
            self.tokens_table.heading(col, text=title)
            self.tokens_table.column(col, width=width, anchor="w")
        self.tokens_table.pack(fill="both", expand=True)
        self.tokens_table.bind("<Motion>", self._on_table_hover)
        self.tokens_table.bind("<Leave>", lambda _: self.text_input.tag_remove("hover_token", "1.0", "end"))

        self.report = tk.Text(top, height=10, wrap="word")
        self.report.pack(fill="both", expand=True, pady=(8, 0))
        self.report.configure(state="disabled")
        self._attach_context_menu(self.report)

    def _toggle_input_size(self) -> None:
        self.text_input.configure(height=24 if self.expand_input.get() else 8)

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

    def _set_report(self, text: str) -> None:
        self.report.configure(state="normal")
        self.report.delete("1.0", "end")
        self.report.insert("1.0", text)
        self.report.configure(state="disabled")

    def _on_table_hover(self, event: tk.Event) -> None:
        row_id = self.tokens_table.identify_row(event.y)
        self.text_input.tag_remove("hover_token", "1.0", "end")
        if row_id in self._row_to_span:
            a, b = self._row_to_span[row_id]
            self.text_input.tag_add("hover_token", f"1.0+{a}c", f"1.0+{b}c")

    def _map_rows_to_text_spans(self, tokens: list[TokenInfo], text: str) -> None:
        self._row_to_span.clear()
        lower = text.lower()
        cursor = 0
        for tok in tokens:
            if not WORD_RE.search(tok.text):
                continue
            found = lower.find(tok.text.lower(), cursor)
            if found == -1:
                found = lower.find(tok.text.lower())
                if found == -1:
                    continue
            row = self.tokens_table.insert("", "end", values=(tok.text, tok.lemma, tok.pos_label, tok.feats))
            self._row_to_span[row] = (found, found + len(tok.text))
            cursor = found + len(tok.text)

    def _set_backend_hint(self, name: str) -> None:
        hints = {
            "natasha": "Используется Natasha (основной бэкенд).",
            "pymorphy3": "Используется pymorphy3 (резерв), потому что Natasha не инициализировалась в этом запуске.",
        }
        self.backend_hint_var.set(hints.get(name, ""))

    def run_analysis(self) -> None:
        text = self.text_input.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Нет текста", "Введите текст для анализа.")
            return
        preferred = self.backend_choice.get()
        try:
            tokens, preferred_fail_reason = self.analyzer.analyze(text, preferred_backend=preferred)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка моделей", f"Не удалось выполнить анализ: {exc}")
            return

        self.backend_var.set(f"бэкенд: {self.analyzer.backend_name}")
        self._set_backend_hint(self.analyzer.backend_name or "natasha")

        if preferred_fail_reason and self.analyzer.backend_name != preferred:
            diagnostics = _diagnose_backend_error(preferred_fail_reason)
            messagebox.showwarning(
                "Бэкенд не загрузился",
                f"Выбранный бэкенд «{preferred}» не инициализировался.\n"
                f"Причина: {preferred_fail_reason}\n\n"
                f"Автоматически использован «{self.analyzer.backend_name}».\n\n"
                f"Диагностика:\n{diagnostics}"
            )

        for r in self.tokens_table.get_children():
            self.tokens_table.delete(r)
        self._map_rows_to_text_spans(tokens, text)

        m = calculate_metrics(tokens, text)
        self._last_freq = m["частоты"]

        lines = ["Частотные коэффициенты частей речи:"]
        lines += [f"- {p}: {v['количество']} / коэффициент={v['коэффициент']}" for p, v in m["частоты"].items()]
        lines.append("\nДополнительные лингвостатистические показатели:")
        lines += [f"- {k}: {v}" for k, v in m["дополнительно"].items()]
        lines.append("\nПрофиль служебных слов:")
        for g, vals in m["профиль_служебных_слов"].items():
            lines.append(f"- {g}: " + ", ".join(f"{k}={v}" for k, v in vals.items()))
        lines.append("\nПрофиль регистра:")
        lines += [f"- {k}: {v}" for k, v in m["регистровый_профиль"].items()]
        lines.append("\nПовторы, намеренные опечатки и искажения:")
        lines += [f"- {k}: {v}" for k, v in m["повторы_и_опечатки"].items()]

        self._set_report("\n".join(lines))

    def show_pie_chart(self) -> None:
        if not self._last_freq:
            messagebox.showinfo("Нет данных", "Сначала выполните анализ текста.")
            return

        win = tk.Toplevel(self.root)
        win.title("Круговая диаграмма частот частей речи")
        win.geometry("860x580")

        legend_text = (
            "Пояснение:\n"
            "• Каждый сектор — доля конкретной части речи в тексте.\n"
            "• Цвет сектора соответствует строке в легенде справа.\n"
            "• Подпись формата «Часть речи: 0.123 (12.3%)»."
        )
        ttk.Label(win, text=legend_text, justify="left").pack(anchor="w", padx=10, pady=6)

        canvas = tk.Canvas(win, bg="white")
        canvas.pack(fill="both", expand=True)

        entries = [(k, float(v["коэффициент"])) for k, v in self._last_freq.items() if float(v["коэффициент"]) > 0]
        total = sum(v for _, v in entries) or 1.0
        cx, cy, r, angle = 250, 310, 180, 0.0

        for i, (_, val) in enumerate(entries):
            extent = 360.0 * (val / total)
            rgb = colorsys.hsv_to_rgb((i / max(len(entries), 1)) % 1.0, 0.65, 0.95)
            color = "#%02x%02x%02x" % tuple(int(c * 255) for c in rgb)
            canvas.create_arc(cx - r, cy - r, cx + r, cy + r, start=angle, extent=extent, fill=color, outline="white")
            angle += extent

        y = 90
        canvas.create_text(520, 60, text="Легенда", anchor="w", font=("Arial", 11, "bold"))
        for i, (label, val) in enumerate(entries):
            rgb = colorsys.hsv_to_rgb((i / max(len(entries), 1)) % 1.0, 0.65, 0.95)
            color = "#%02x%02x%02x" % tuple(int(c * 255) for c in rgb)
            canvas.create_rectangle(520, y, 540, y + 14, fill=color, outline=color)
            canvas.create_text(548, y + 7, text=f"{label}: {val:.3f} ({val*100:.1f}%)", anchor="w")
            y += 24

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    ForensicsApp().run()
