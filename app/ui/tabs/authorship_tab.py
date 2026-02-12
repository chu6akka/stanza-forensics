import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from docx import Document


class AuthorshipTab(ttk.Frame):
    def __init__(self, parent, app) -> None:
        super().__init__(parent)
        self.app = app
        self.main_text = ""
        self.sample_text = ""
        self.last_result: dict = {}

        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=8, pady=4)
        ttk.Button(bar, text="Загрузить исследуемый", command=self.load_main).pack(side="left")
        ttk.Button(bar, text="Загрузить образец", command=self.load_sample).pack(side="left", padx=6)
        ttk.Button(bar, text="Сравнить", command=self.compare).pack(side="left", padx=6)

        self.out = tk.Text(self, height=30, wrap="word")
        self.out.pack(fill="both", expand=True, padx=8, pady=4)
        self._attach_context_menu(self.out)

    def _attach_context_menu(self, widget: tk.Text) -> None:
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="Копировать", command=lambda: widget.event_generate("<<Copy>>"))
        menu.add_command(label="Вставить", command=lambda: widget.event_generate("<<Paste>>"))
        menu.add_separator()
        menu.add_command(label="Выделить всё", command=lambda: widget.tag_add("sel", "1.0", "end-1c"))
        widget.bind("<Button-3>", lambda event: menu.tk_popup(event.x_root, event.y_root))

    def _load_text_file(self) -> str:
        p = filedialog.askopenfilename(filetypes=[("Text", "*.txt *.docx")])
        if not p:
            return ""
        path = Path(p)
        if path.suffix.lower() == ".docx":
            doc = Document(path)
            return "\n".join(par.text for par in doc.paragraphs)
        return path.read_text(encoding="utf-8", errors="ignore")

    def load_main(self) -> None:
        text = self._load_text_file()
        if text:
            self.main_text = text

    def load_sample(self) -> None:
        text = self._load_text_file()
        if text:
            self.sample_text = text

    def compare(self) -> None:
        if not self.main_text or not self.sample_text:
            return
        self.last_result = self.app.compare_with_sample(self.main_text, self.sample_text)
        scores = self.last_result.get("scores", {})
        exact = self.last_result.get("exact_fragments", [])
        paraphrases = self.last_result.get("paraphrase_candidates", [])
        translit = self.last_result.get("transliteration", {}).get("shared_patterns", [])

        self.out.delete("1.0", "end")
        self.out.insert("1.0", "Автоматизированный автороведческий анализ\n")
        self.out.insert("end", "=" * 72 + "\n")
        self.out.insert("end", json.dumps(scores, ensure_ascii=False, indent=2) + "\n\n")

        self.out.insert("end", "Идентичные фрагменты и речевые обороты:\n")
        if exact:
            for item in exact[:12]:
                self.out.insert("end", f" • [{item['matches']}x, {item['words']} слов] {item['fragment']}\n")
        else:
            self.out.insert("end", " • Совпадающих фрагментов достаточной длины не найдено.\n")

        self.out.insert("end", "\nКандидаты на синонимию/перефраз:\n")
        if paraphrases:
            for item in paraphrases[:8]:
                self.out.insert(
                    "end",
                    f" • sim={item['semantic_similarity']}, overlap={item['lexical_overlap']}\n"
                    f"   A: {item['source']}\n"
                    f"   B: {item['target']}\n",
                )
        else:
            self.out.insert("end", " • Явные кандидаты не обнаружены.\n")

        self.out.insert("end", "\nТранслитерация/смешанная графика:\n")
        if translit:
            self.out.insert("end", " • Совпавшие паттерны: " + ", ".join(translit[:10]) + "\n")
        else:
            self.out.insert("end", " • Совпадающие паттерны не выявлены.\n")

        self.out.insert("end", "\nЧерновик вывода эксперта:\n")
        for line in self.last_result.get("conclusion_draft", []):
            self.out.insert("end", f" • {line}\n")

    def get_comparison_summary(self) -> dict:
        if not self.last_result:
            return {}
        return {
            "scores": self.last_result.get("scores", {}),
            "conclusion_draft": self.last_result.get("conclusion_draft", []),
            "exact_fragments_count": len(self.last_result.get("exact_fragments", [])),
            "paraphrase_candidates_count": len(self.last_result.get("paraphrase_candidates", [])),
            "transliteration_shared": self.last_result.get("transliteration", {}).get("shared_patterns", []),
        }
