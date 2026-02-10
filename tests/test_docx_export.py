from pathlib import Path

import pytest

pytest.importorskip("docx")

from app.core.reporting.docx_report import build_docx_report


def test_docx_export(tmp_path: Path):
    out = tmp_path / "r.docx"
    build_docx_report({"reproducibility": {"sha256": "x"}}, out)
    assert out.exists()
