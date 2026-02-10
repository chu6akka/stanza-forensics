import tkinter as tk
from tkinter import filedialog, ttk


class AuthorshipTab(ttk.Frame):
    def __init__(self, parent, app) -> None:
        super().__init__(parent)
        self.app = app
        self.main_text = ""
        self.sample_text = ""
        self.last_score = 0.0

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

    def load_main(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("Text", "*.txt")])
        if p:
            self.main_text = open(p, encoding="utf-8", errors="ignore").read()

    def load_sample(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("Text", "*.txt")])
        if p:
            self.sample_text = open(p, encoding="utf-8", errors="ignore").read()

    def compare(self) -> None:
        if not self.main_text or not self.sample_text:
            return
        self.last_score = self.app.compare_with_sample(self.main_text, self.sample_text)
        self.out.delete("1.0", "end")
        self.out.insert("1.0", f"Сходство char n-gram профилей (cosine): {self.last_score:.4f}\n")
        self.out.insert("end", "Интерпретация: это показатель сходства профилей, а не автоматический вывод об авторстве.")

    def get_comparison_summary(self) -> dict:
        return {"cosine_char_ngrams": round(self.last_score, 4)}
