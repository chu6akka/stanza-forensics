from pathlib import Path
from tkinter import filedialog, messagebox, ttk


class ReportTab(ttk.Frame):
    def __init__(self, parent, app) -> None:
        super().__init__(parent)
        self.app = app

        ttk.Label(self, text="Сборка отчёта (DOCX/JSON/CSV)").pack(anchor="w", padx=8, pady=8)
        ttk.Button(self, text="Экспортировать пакет отчёта", command=self.export).pack(anchor="w", padx=8)

    def export(self) -> None:
        payload = self.app.stat_tab.current_payload
        if not payload:
            messagebox.showwarning("Нет результатов", "Сначала выполните анализ в режиме 1")
            return
        d = filedialog.askdirectory()
        if not d:
            return
        self.app.export_all(payload, Path(d))
        messagebox.showinfo("Готово", "Экспорт завершен")
