# Forensic Linguistics Desktop v1.0

Офлайн desktop-приложение (Tkinter) для лингвистического и автороведческого анализа текста.

## Дисклеймер

Программа **не устанавливает автора и смысл автоматически**. Результаты являются ориентирующими и требуют экспертной интерпретации.

## Архитектура

- `app/ui` — GUI (4 вкладки/режима)
- `app/core/backends` — бэкенды `natasha` + `pymorphy3`
- `app/core/metrics` — POS/лексика/пунктуация/орфография/ngrams/quality
- `app/core/context` — фрагменты (span offsets) + подсказки
- `app/core/authorship` — профили и метрики сходства
- `app/core/reporting` — экспорт `DOCX/JSON/CSV`
- `app/storage` — SQLite-хранилище проектов (`case.db`)

## Режимы

1. **Статистический профиль**
2. **Контекстный анализ и частные признаки**
3. **Автоворедение: сравнение с образцами**
4. **Мастер отчёта**

## Проекты и воспроизводимость

Данные сохраняются в `project/`:
- `project/case.db` — SQLite
- `project/raw/` — исходные файлы
- `project/results/` — JSON-результаты запусков

В `manifest` запуска фиксируются:
- `sha256` текста
- backend
- версия Python
- предупреждения качества
- режим анализа

## Установка

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск

```bash
python -m app
# или
python pos_forensics.py
```

## Экспорт

Вкладка «Мастер отчёта» экспортирует:
- `report.docx`
- `report.json`
- `tokens.csv`

## Диагностика окружения

В шапке интерфейса есть кнопка «Диагностика окружения».

## Сборка в .exe (Windows, PyInstaller)

Пример:

```bash
pyinstaller --name forensic-desktop --onefile --windowed pos_forensics.py
```

(при необходимости добавьте data-файлы моделей/ресурсов ключом `--add-data`).
