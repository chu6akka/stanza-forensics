import tkinter as tk
from tkinter import messagebox, ttk


class StatisticalTab(ttk.Frame):
    def __init__(self, parent, app) -> None:
        super().__init__(parent)
        self.app = app
        self.current_payload: dict | None = None

        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.pack(fill="x", padx=8, pady=4)

        self.text = tk.Text(self, height=18, wrap="word")
        self.text.pack(fill="both", expand=False, padx=8, pady=4)

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=8, pady=4)
        ttk.Button(btns, text="Запустить анализ", command=self.run_analysis).pack(side="left")
        ttk.Button(btns, text="Отмена", command=self.cancel).pack(side="left", padx=6)

        self.report = tk.Text(self, height=20, wrap="word")
        self.report.pack(fill="both", expand=True, padx=8, pady=4)

    def set_text(self, text: str) -> None:
        self.text.delete("1.0", "end")
        self.text.insert("1.0", text)

    def cancel(self) -> None:
        self.app.cancel_event.set()

    def run_analysis(self) -> None:
        text = self.text.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Пустой текст", "Введите текст для анализа")
            return
        self.progress.start(10)
        self.app.run_analysis_background("mode1", text, self._on_done)

    def _on_done(self, payload: dict | None, err: str | None) -> None:
        self.progress.stop()
        if err:
            messagebox.showerror("Ошибка", err)
            return
        self.current_payload = payload
        metrics = payload["metrics"]
        lines = ["=== ПАСПОРТ ТЕКСТА ==="]
        lines.append(f"SHA256: {payload['manifest']['sha256']}")
        lines.append(f"Бэкенд: {payload['manifest']['backend']}")
        lines.append("\n=== POS ПРОФИЛЬ ===")
        for k, v in metrics["pos_profile"].items():
            lines.append(f"{k}: {v}")
        lines.append("\n=== ЛЕКСИКА ===")
        for k, v in metrics["lex"].items():
            lines.append(f"{k}: {v}")
        lines.append("\n=== КАЧЕСТВО МАТЕРИАЛА ===")
        for w in payload["manifest"]["warnings"]:
            lines.append(f"- {w}")
        self.report.delete("1.0", "end")
        self.report.insert("1.0", "\n".join(lines))
