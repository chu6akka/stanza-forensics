import tkinter as tk
from tkinter import ttk

from app.core.context.spans import create_span


class ContextTab(ttk.Frame):
    def __init__(self, parent, app) -> None:
        super().__init__(parent)
        self.app = app
        self.spans = []

        ttk.Label(self, text="Разметка фрагментов по offsets").pack(anchor="w", padx=8, pady=4)
        self.text = tk.Text(self, height=18, wrap="word")
        self.text.pack(fill="both", expand=True, padx=8, pady=4)

        tools = ttk.Frame(self)
        tools.pack(fill="x", padx=8, pady=4)
        self.tag_entry = ttk.Entry(tools)
        self.tag_entry.insert(0, "утверждение")
        self.tag_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(tools, text="Добавить фрагмент по выделению", command=self.add_fragment).pack(side="left", padx=6)

        self.table = tk.Listbox(self)
        self.table.pack(fill="both", expand=True, padx=8, pady=4)

    def add_fragment(self) -> None:
        try:
            start = int(self.text.count("1.0", "sel.first", "chars")[0])
            end = int(self.text.count("1.0", "sel.last", "chars")[0])
        except Exception:
            return
        tag = self.tag_entry.get().strip() or "фрагмент"
        span = create_span(start, end, [tag], "")
        self.spans.append(span)
        self.table.insert("end", f"{start}-{end} [{tag}]")

    def get_fragments_text(self) -> list[str]:
        return [f"{s.start}-{s.end}: {','.join(s.tags)}" for s in self.spans]
