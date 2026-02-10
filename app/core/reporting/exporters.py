import csv
import json
from pathlib import Path
from typing import Any


def export_json(data: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def export_tokens_csv(tokens: list[dict[str, Any]], path: Path) -> None:
    if not tokens:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(tokens[0].keys()))
        writer.writeheader()
        writer.writerows(tokens)
