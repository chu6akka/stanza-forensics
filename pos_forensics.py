#!/usr/bin/env python3
"""GUI for Russian forensic POS analysis with Natasha backend and optional Stanza."""

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
            return stanza.Pipeline(
                lang=self.lang,
                processors="tokenize,mwt,pos,lemma,depparse",
                use_gpu=False,
                verbose=False,
            )
        except Exception:
            stanza.download(self.lang, processors="tokenize,mwt,pos,lemma,depparse", verbose=False)
            return stanza.Pipeline(
                lang=self.lang,
                processors="tokenize,mwt,pos,lemma,depparse",
                use_gpu=False,
                verbose=False,
            )

    def _init_natasha(self):
        from natasha import Doc, MorphVocab, NewsEmbedding, NewsMorphTagger, Segmenter

        return {
            "Doc": Doc,
            "segmenter": Segmenter(),
            "morph_vocab": MorphVocab(),
            "morph_tagger": NewsMorphTagger(NewsEmbedding()),
        }

    def _ensure_backend(self, preferred: str = "natasha") -> None:
        if self._backend is not None and self.backend_name == preferred:
            return

        errors: list[str] = []

        if preferred == "stanza":
            try:
                self._backend = self._init_stanza()
                self.backend_name = "stanza"
                return
            except Exception as exc:  # noqa: BLE001
                errors.append(f"stanza: {exc}")

            try:
                self._backend = self._init_natasha()
                self.backend_name = "natasha"
                return
            except Exception as exc:  # noqa: BLE001
                errors.append(f"natasha: {exc}")
        else:
            try:
                self._backend = self._init_natasha()
                self.backend_name = "natasha"
                return
            except Exception as exc:  # noqa: BLE001
                errors.append(f"natasha: {exc}")

            try:
                self._backend = self._init_stanza()
                self.backend_name = "stanza"
                return
            except Exception as exc:  # noqa: BLE001
                errors.append(f"stanza: {exc}")

        raise RuntimeError("Не удалось инициализировать ни Natasha, ни Stanza. " + " | ".join(errors))

    def analyze(self, text: str, preferred_backend: str = "natasha") -> list[TokenInfo]:
        self._ensure_backend(preferred_backend)
        if self.backend_name == "stanza":
            return self._analyze_stanza(text)
        return self._analyze_natasha(text)

    def _analyze_stanza(self, text: str) -> list[TokenInfo]:
        doc = self._backend(text)
        out: list[TokenInfo] = []
        for sent in doc.sentences:
            for word in sent.words:
                pos = _normalize_pos(word.upos, word.feats)
                out.append(TokenInfo(word.text, word.lemma or word.text.lower(), pos, _pos_label_ru(pos), _format_feats_ru(word.feats or "")))
        return out

    def _analyze_natasha(self, text: str) -> list[TokenInfo]:
        Doc = self._backend["Doc"]
        segmenter = self._backend["segmenter"]
        morph_tagger = self._backend["morph_tagger"]
        morph_vocab = self._backend["morph_vocab"]

        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)

        out: list[TokenInfo] = []
        for token in doc.tokens:
            token.lemmatize(morph_vocab)
            if not WORD_RE.search(token.text):
                out.append(TokenInfo(token.text, token.text, "PUNCT", "Пунктуация", "—"))
                continue
            pos_raw = getattr(token, "pos", None) or "X"
            feats_dict = getattr(token, "feats", {}) or {}
            feats_ud = "|".join(f"{k}={v}" for k, v in feats_dict.items()) if isinstance(feats_dict, dict) else ""
            pos = _normalize_pos(pos_raw, feats_ud)
            out.append(TokenInfo(token.text, token.lemma or token.text.lower(), pos, _pos_label_ru(pos), _format_feats_ru(feats_ud)))
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
    if not d:
        return "—"
    return "; ".join(f"{FEAT_NAME_RU.get(k, k)}: {FEAT_VALUE_RU.get(v, v)}" for k, v in d.items())


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
    lowered = text.lower()
    return {
        group: {m: len(re.findall(rf"(?<!\w){re.escape(m)}(?!\w)", lowered)) for m in markers}
        for group, markers in STYLE_MARKERS.items()
    }


def _char_ngrams(text: str, n_values: tuple[int, ...] = (3, 4, 5), top_k: int = 10) -> dict[str, list[tuple[str, int]]]:
    source = re.sub(r"\s+", " ", text.lower())
    out: dict[str, list[tuple[str, int]]] = {}
    for n in n_values:
        grams = [source[i : i + n] for i in range(max(len(source) - n + 1, 0))]
        out[f"char_{n}gram"] = Counter(grams).most_common(top_k)
    return out


def _pos_ngrams(tokens: list[TokenInfo], top_k: int = 10) -> dict[str, list[tuple[str, int]]]:
    seq = [t.pos_label for t in _word_tokens(tokens)]
    out: dict[str, list[tuple[str, int]]] = {}
    for n in (2, 3):
        grams = [" ".join(seq[i : i + n]) for i in range(max(len(seq) - n + 1, 0))]
        out[f"pos_{n}gram"] = Counter(grams).most_common(top_k)
    return out


def _punctuation_patterns(text: str) -> dict[str, float | int]:
    words = max(len(WORD_RE.findall(text)), 1)
    em_dash = text.count("—")
    hyphen = max(text.count("-") - em_dash, 0)
    return {
        "тире_эмдеш": em_dash,
        "дефис": hyphen,
        "доля_тире_среди_тире_и_дефисов": round(em_dash / (em_dash + hyphen), 4) if em_dash + hyphen else 0.0,
        "троеточие_символ": text.count("…"),
        "троеточие_три_точки": text.count("..."),
        "комбо_!?": len(re.findall(r"!\?", text)),
        "комбо_??": len(re.findall(r"\?\?", text)),
        "комбо_!!": len(re.findall(r"!!", text)),
        "пробел_перед_запятой": len(re.findall(r"\s,", text)),
        "пробел_перед_точкой": len(re.findall(r"\s\.", text)),
        "знаков_препинания_на_100_слов": round((len(re.findall(r"[.,!?;:—\-…]", text)) / words) * 100, 2),
    }


def _orthography_flags(text: str) -> dict[str, float | int]:
    words = WORD_RE.findall(text)
    n = max(len(words), 1)
    lower = text.lower()
    yo = lower.count("ё")
    e = lower.count("е")
    caps = len([w for w in words if len(w) > 1 and w.isupper()])
    return {
        "доля_ё_среди_е_ё": round(yo / (yo + e), 4) if (yo + e) else 0.0,
        "ALL_CAPS_слов": caps,
        "повтор_знаков_3plus": len(re.findall(r"([!?.,])\1{2,}", text)),
        "подозрительные_тся_ться": len(re.findall(r"\b\w+т(ь)?ся\b", lower)),
        "доля_caps_слов": round(caps / n, 4),
    }


def calculate_metrics(tokens: list[TokenInfo], text: str) -> dict[str, object]:
    words = _word_tokens(tokens)
    total = len(words)
    if total == 0:
        return {"freq": {}, "additional": {"Всего слов": 0, "Комментарий": "Недостаточно данных для расчета показателей."}}

    pos_counts = Counter(t.pos_label for t in words)
    freq = {p: {"count": c, "coefficient": round(c / total, 4)} for p, c in sorted(pos_counts.items(), key=lambda x: (-x[1], x[0]))}

    lemmas = [t.lemma.lower() for t in words]
    unique_words = len(set(w.text.lower() for w in words))
    unique_lemmas = len(set(lemmas))
    hapax = sum(1 for _, c in Counter(lemmas).items() if c == 1)
    sentence_chunks = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sent_lens = [len(WORD_RE.findall(s)) for s in sentence_chunks if WORD_RE.findall(s)]

    content_set = {"Существительное", "Имя собственное", "Глагол", "Причастие", "Деепричастие", "Прилагательное", "Наречие"}
    function_set = {"Предлог", "Частица", "Сочинительный союз", "Подчинительный союз", "Местоимение", "Определительное слово"}
    content = sum(pos_counts.get(p, 0) for p in content_set)
    function = sum(pos_counts.get(p, 0) for p in function_set)
    noun_cnt = pos_counts.get("Существительное", 0) + pos_counts.get("Имя собственное", 0)
    verb_cnt = pos_counts.get("Глагол", 0) + pos_counts.get("Вспомогательный глагол", 0)

    additional = {
        "Всего слов": total,
        "Лексическое разнообразие (TTR)": round(unique_words / total, 4),
        "Лемматическое разнообразие": round(unique_lemmas / total, 4),
        "Доля hapax-лемм": round(hapax / total, 4),
        "Средняя длина предложения (слов)": round(sum(sent_lens) / len(sent_lens), 2) if sent_lens else 0.0,
        "Дисперсия длины предложений": round(statistics.pvariance(sent_lens), 2) if len(sent_lens) > 1 else 0.0,
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
        self.root.geometry("1120x780")

        self.analyzer = Analyzer(lang="ru")
        self._row_to_span: dict[str, tuple[int, int]] = {}
        self._last_freq: dict[str, dict[str, float | int]] = {}

        self._build_ui()
        self._bind_hotkeys()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="both", expand=True)

        ttk.Label(top, text="Введите русский текст для анализа:").pack(anchor="w")
        self.text_input = tk.Text(top, height=9, wrap="word")
        self.text_input.pack(fill="x", pady=(4, 10))
        self.text_input.tag_configure("hover_token", background="#fff176")
        self._attach_context_menu(self.text_input)

        btns = ttk.Frame(top)
        btns.pack(fill="x", pady=(0, 8))
        ttk.Button(btns, text="Анализировать (Ctrl+Enter)", command=self.run_analysis).pack(side="left")
        ttk.Button(btns, text="Показать круговую диаграмму", command=self.show_pie_chart).pack(side="left", padx=8)
        ttk.Button(btns, text="Очистить", command=self.clear_all).pack(side="left", padx=8)

        ttk.Label(btns, text="Режим:").pack(side="left", padx=(12, 4))
        self.backend_choice = tk.StringVar(value="natasha")
        combo = ttk.Combobox(btns, width=18, textvariable=self.backend_choice, state="readonly", values=["natasha", "stanza"])
        combo.pack(side="left")

        self.backend_var = tk.StringVar(value="backend: не инициализирован")
        self.backend_hint_var = tk.StringVar(value="По умолчанию используется Natasha. Stanza можно включить через переключатель.")
        ttk.Label(btns, textvariable=self.backend_var).pack(side="right")
        ttk.Label(top, textvariable=self.backend_hint_var, foreground="#444").pack(anchor="w")

        self.tokens_table = ttk.Treeview(top, columns=("token", "lemma", "pos", "feats"), show="headings", height=12)
        for col, width, title in [
            ("token", 170, "Словоформа"),
            ("lemma", 200, "Начальная форма"),
            ("pos", 180, "Часть речи"),
            ("feats", 520, "Морфологические признаки"),
        ]:
            self.tokens_table.heading(col, text=title)
            self.tokens_table.column(col, width=width, anchor="w")
        self.tokens_table.pack(fill="both", expand=True)
        self.tokens_table.bind("<Motion>", self._on_table_hover)
        self.tokens_table.bind("<Leave>", lambda _: self.text_input.tag_remove("hover_token", "1.0", "end"))

        ttk.Label(top, text="Частотные коэффициенты и показатели для автороведческого анализа:").pack(anchor="w", pady=(10, 4))
        self.report = tk.Text(top, height=12, wrap="word")
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
            start, end = self._row_to_span[row_id]
            self.text_input.tag_add("hover_token", f"1.0+{start}c", f"1.0+{end}c")

    def _map_rows_to_text_spans(self, tokens: list[TokenInfo], text: str) -> None:
        self._row_to_span.clear()
        lower = text.lower()
        cursor = 0
        for tok in tokens:
            if not WORD_RE.search(tok.text):
                continue
            needle = tok.text.lower()
            found = lower.find(needle, cursor)
            if found == -1:
                found = lower.find(needle)
                if found == -1:
                    continue
            end = found + len(tok.text)
            row_id = self.tokens_table.insert("", "end", values=(tok.text, tok.lemma, tok.pos_label, tok.feats))
            self._row_to_span[row_id] = (found, end)
            cursor = end

    def _set_backend_hint(self, backend_name: str) -> None:
        if backend_name == "stanza":
            self.backend_hint_var.set("Используется Stanza. Если снова появится fallback — проверьте установку модели и совместимость Python/Torch.")
        else:
            self.backend_hint_var.set("Используется Natasha (основной стабильный режим). Если выбрана Stanza и не запустилась — приложение автоматически вернется к Natasha.")

    def clear_all(self) -> None:
        self.text_input.delete("1.0", "end")
        self.text_input.tag_remove("hover_token", "1.0", "end")
        self.backend_var.set("backend: не инициализирован")
        self.backend_hint_var.set("")
        self._last_freq = {}
        self._row_to_span.clear()
        for row in self.tokens_table.get_children():
            self.tokens_table.delete(row)
        self._set_report("")

    def run_analysis(self) -> None:
        text = self.text_input.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Нет текста", "Введите текст для анализа.")
            return

        preferred = self.backend_choice.get()
        try:
            tokens = self.analyzer.analyze(text, preferred_backend=preferred)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка моделей", f"Не удалось выполнить анализ: {exc}")
            return

        self.backend_var.set(f"backend: {self.analyzer.backend_name}")
        self._set_backend_hint(self.analyzer.backend_name or "natasha")

        for row in self.tokens_table.get_children():
            self.tokens_table.delete(row)
        self._map_rows_to_text_spans(tokens, text)

        m = calculate_metrics(tokens, text)
        self._last_freq = m["freq"]

        lines = ["Частотные коэффициенты частей речи (count / общее количество слов):"]
        for pos, vals in m["freq"].items():
            lines.append(f"- {pos}: {vals['count']} / coef={vals['coefficient']}")

        lines.append("\nДополнительные показатели:")
        for k, v in m["additional"].items():
            lines.append(f"- {k}: {v}")

        lines.append("\nПрофиль служебных слов/маркеров:")
        for group, vals in m["service_profile"].items():
            lines.append(f"- {group}: " + ", ".join(f"{k}={v}" for k, v in vals.items()))

        self._set_report("\n".join(lines))

    def show_pie_chart(self) -> None:
        if not self._last_freq:
            messagebox.showinfo("Нет данных", "Сначала выполните анализ текста.")
            return

        win = tk.Toplevel(self.root)
        win.title("Круговая диаграмма частот частей речи")
        win.geometry("760x540")

        canvas = tk.Canvas(win, bg="white")
        canvas.pack(fill="both", expand=True)

        entries = [(k, float(v["coefficient"])) for k, v in self._last_freq.items() if float(v["coefficient"]) > 0]
        total = sum(v for _, v in entries) or 1.0

        cx, cy, r = 250, 250, 180
        angle = 0.0
        for i, (label, val) in enumerate(entries):
            extent = 360.0 * (val / total)
            hue = (i / max(len(entries), 1)) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
            color = "#%02x%02x%02x" % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            canvas.create_arc(cx - r, cy - r, cx + r, cy + r, start=angle, extent=extent, fill=color, outline="white")
            angle += extent

        y = 35
        canvas.create_text(520, 15, text="Легенда", anchor="w", font=("Arial", 11, "bold"))
        for i, (label, val) in enumerate(entries):
            hue = (i / max(len(entries), 1)) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
            color = "#%02x%02x%02x" % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            canvas.create_rectangle(520, y, 540, y + 14, fill=color, outline=color)
            canvas.create_text(548, y + 7, text=f"{label}: {val:.3f}", anchor="w")
            y += 22

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    ForensicsApp().run()
