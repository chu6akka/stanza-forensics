#!/usr/bin/env python3
"""GUI for Russian forensic POS analysis with Stanza primary backend."""

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
    "Degree": "Степень",
    "Gender": "Род",
    "Mood": "Наклонение",
    "Number": "Число",
    "Person": "Лицо",
    "Tense": "Время",
    "VerbForm": "Форма глагола",
    "Voice": "Залог",
}

FEAT_VALUE_RU = {
    "Anim": "одуш.",
    "Inan": "неодуш.",
    "Imp": "несов.",
    "Perf": "сов.",
    "Nom": "им.",
    "Gen": "род.",
    "Dat": "дат.",
    "Acc": "вин.",
    "Ins": "твор.",
    "Loc": "предл.",
    "Sing": "ед.",
    "Plur": "мн.",
    "Masc": "муж.",
    "Fem": "жен.",
    "Neut": "ср.",
    "Pres": "наст.",
    "Past": "прош.",
    "Fut": "буд.",
    "Ind": "изъяв.",
    "Part": "причастие",
    "Conv": "деепричастие",
    "Act": "действ.",
    "Pass": "страд.",
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
    def __init__(self, lang: str = "ru") -> None:
        self.lang = lang
        self.backend_name: str | None = None
        self._backend = None

    def _init_stanza(self):
        import stanza

        try:
            return stanza.Pipeline(lang=self.lang, processors="tokenize,mwt,pos,lemma,depparse", use_gpu=False, verbose=False)
        except Exception:
            stanza.download(self.lang, processors="tokenize,mwt,pos,lemma,depparse", verbose=False)
            return stanza.Pipeline(lang=self.lang, processors="tokenize,mwt,pos,lemma,depparse", use_gpu=False, verbose=False)

    def _init_natasha(self):
        from natasha import Doc, MorphVocab, NewsEmbedding, NewsMorphTagger, Segmenter

        return {"Doc": Doc, "segmenter": Segmenter(), "morph_vocab": MorphVocab(), "morph_tagger": NewsMorphTagger(NewsEmbedding())}

    def _init_pymorphy(self):
        import pymorphy3
        from razdel import tokenize

        return {"morph": pymorphy3.MorphAnalyzer(), "tokenize": tokenize}

    def _ensure_backend(self, preferred: str = "stanza") -> None:
        if self._backend is not None and self.backend_name == preferred:
            return

        orders = {
            "stanza": ["stanza", "natasha", "pymorphy3"],
            "natasha": ["natasha", "pymorphy3", "stanza"],
            "pymorphy3": ["pymorphy3", "natasha", "stanza"],
        }
        order = orders.get(preferred, orders["stanza"])
        errors: list[str] = []

        for name in order:
            try:
                if name == "stanza":
                    self._backend = self._init_stanza()
                elif name == "natasha":
                    self._backend = self._init_natasha()
                else:
                    self._backend = self._init_pymorphy()
                self.backend_name = name
                return
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}: {exc}")

        raise RuntimeError("Не удалось инициализировать бэкенды stanza/natasha/pymorphy3. " + " | ".join(errors))

    def analyze(self, text: str, preferred_backend: str = "stanza") -> list[TokenInfo]:
        self._ensure_backend(preferred_backend)
        if self.backend_name == "stanza":
            return self._analyze_stanza(text)
        if self.backend_name == "natasha":
            return self._analyze_natasha(text)
        return self._analyze_pymorphy(text)

    def _analyze_stanza(self, text: str) -> list[TokenInfo]:
        doc = self._backend(text)
        return [
            TokenInfo(w.text, w.lemma or w.text.lower(), _normalize_pos(w.upos, w.feats), _pos_label_ru(_normalize_pos(w.upos, w.feats)), _format_feats_ru(w.feats or ""))
            for s in doc.sentences
            for w in s.words
        ]

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
            out.append(TokenInfo(t.text, p.normal_form, upos, _pos_label_ru(upos), str(p.tag)))
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


def _word_tokens(tokens: Iterable[TokenInfo]) -> list[TokenInfo]:
    return [t for t in tokens if WORD_RE.search(t.text) and t.pos != "PUNCT"]


def _count_style_markers(text: str) -> dict[str, dict[str, int]]:
    lowered = text.lower()
    return {g: {m: len(re.findall(rf"(?<!\w){re.escape(m)}(?!\w)", lowered)) for m in marks} for g, marks in STYLE_MARKERS.items()}


def calculate_metrics(tokens: list[TokenInfo], text: str) -> dict[str, object]:
    words = _word_tokens(tokens)
    total = len(words)
    if total == 0:
        return {"freq": {}, "additional": {"Всего слов": 0, "Комментарий": "Недостаточно данных для расчета показателей."}, "service_profile": {}}
    pos_counts = Counter(t.pos_label for t in words)
    freq = {p: {"count": c, "coefficient": round(c / total, 4)} for p, c in sorted(pos_counts.items(), key=lambda x: (-x[1], x[0]))}
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
        "Коэф. содержательные/служебные": round(content / function, 4) if function else "inf",
        "Коэф. существительные/глаголы": round(noun_cnt / verb_cnt, 4) if verb_cnt else "inf",
    }
    return {"freq": freq, "additional": additional, "service_profile": _count_style_markers(text)}


class ForensicsApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Forensic POS Analyzer (RU)")
        self.root.geometry("1120x760")
        self.analyzer = Analyzer(lang="ru")
        self._row_to_span: dict[str, tuple[int, int]] = {}
        self._last_freq: dict[str, dict[str, float | int]] = {}
        self._build_ui()
        self._bind_hotkeys()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="both", expand=True)
        ttk.Label(top, text="Введите русский текст для анализа:").pack(anchor="w")
        self.text_input = tk.Text(top, height=8, wrap="word")
        self.text_input.pack(fill="x", pady=(4, 10))
        self.text_input.tag_configure("hover_token", background="#fff176")
        self._attach_context_menu(self.text_input)

        btns = ttk.Frame(top)
        btns.pack(fill="x", pady=(0, 8))
        ttk.Button(btns, text="Анализировать (Ctrl+Enter)", command=self.run_analysis).pack(side="left")
        ttk.Button(btns, text="Показать круговую диаграмму", command=self.show_pie_chart).pack(side="left", padx=8)
        ttk.Label(btns, text="Режим:").pack(side="left", padx=(12, 4))
        self.backend_choice = tk.StringVar(value="stanza")
        ttk.Combobox(btns, width=18, textvariable=self.backend_choice, state="readonly", values=["stanza", "natasha", "pymorphy3"]).pack(side="left")

        self.backend_var = tk.StringVar(value="backend: не инициализирован")
        self.backend_hint_var = tk.StringVar(value="По умолчанию используется Stanza. Резерв: Natasha → pymorphy3.")
        ttk.Label(btns, textvariable=self.backend_var).pack(side="right")
        ttk.Label(top, textvariable=self.backend_hint_var, foreground="#444").pack(anchor="w")

        self.tokens_table = ttk.Treeview(top, columns=("token", "lemma", "pos", "feats"), show="headings", height=11)
        for col, width, title in [("token", 170, "Словоформа"), ("lemma", 200, "Начальная форма"), ("pos", 180, "Часть речи"), ("feats", 520, "Морфологические признаки")]:
            self.tokens_table.heading(col, text=title)
            self.tokens_table.column(col, width=width, anchor="w")
        self.tokens_table.pack(fill="both", expand=True)
        self.tokens_table.bind("<Motion>", self._on_table_hover)
        self.tokens_table.bind("<Leave>", lambda _: self.text_input.tag_remove("hover_token", "1.0", "end"))

        self.report = tk.Text(top, height=10, wrap="word")
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
            "stanza": "Используется Stanza (основной). Если не запускается — автопереход на Natasha, затем pymorphy3.",
            "natasha": "Используется Natasha (резерв №1), потому что Stanza не инициализировалась в этом запуске.",
            "pymorphy3": "Используется pymorphy3 (резерв №2), потому что Stanza и Natasha не инициализировались.",
        }
        self.backend_hint_var.set(hints.get(name, ""))

    def run_analysis(self) -> None:
        text = self.text_input.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Нет текста", "Введите текст для анализа.")
            return
        try:
            tokens = self.analyzer.analyze(text, preferred_backend=self.backend_choice.get())
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка моделей", f"Не удалось выполнить анализ: {exc}")
            return

        self.backend_var.set(f"backend: {self.analyzer.backend_name}")
        self._set_backend_hint(self.analyzer.backend_name or "stanza")
        for r in self.tokens_table.get_children():
            self.tokens_table.delete(r)
        self._map_rows_to_text_spans(tokens, text)

        m = calculate_metrics(tokens, text)
        self._last_freq = m["freq"]
        lines = ["Частотные коэффициенты частей речи:"]
        lines += [f"- {p}: {v['count']} / coef={v['coefficient']}" for p, v in m["freq"].items()]
        lines.append("\nДополнительные показатели:")
        lines += [f"- {k}: {v}" for k, v in m["additional"].items()]
        lines.append("\nПрофиль служебных слов:")
        for g, vals in m["service_profile"].items():
            lines.append(f"- {g}: " + ", ".join(f"{k}={v}" for k, v in vals.items()))
        self._set_report("\n".join(lines))

    def show_pie_chart(self) -> None:
        if not self._last_freq:
            messagebox.showinfo("Нет данных", "Сначала выполните анализ текста.")
            return
        win = tk.Toplevel(self.root)
        win.title("Круговая диаграмма")
        win.geometry("760x540")
        canvas = tk.Canvas(win, bg="white")
        canvas.pack(fill="both", expand=True)
        entries = [(k, float(v["coefficient"])) for k, v in self._last_freq.items() if float(v["coefficient"]) > 0]
        total = sum(v for _, v in entries) or 1.0
        cx, cy, r, angle = 250, 250, 180, 0.0
        for i, (label, val) in enumerate(entries):
            extent = 360.0 * (val / total)
            rgb = colorsys.hsv_to_rgb((i / max(len(entries), 1)) % 1.0, 0.65, 0.95)
            color = "#%02x%02x%02x" % tuple(int(c * 255) for c in rgb)
            canvas.create_arc(cx - r, cy - r, cx + r, cy + r, start=angle, extent=extent, fill=color, outline="white")
            angle += extent

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    ForensicsApp().run()
