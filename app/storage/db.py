import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS cases(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL,
  created_at TEXT NOT NULL,
  notes TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS documents(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  case_id INTEGER NOT NULL,
  role TEXT NOT NULL,
  title TEXT NOT NULL,
  source_path TEXT DEFAULT '',
  text_sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(case_id) REFERENCES cases(id)
);
CREATE TABLE IF NOT EXISTS fragments(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id INTEGER NOT NULL,
  start INTEGER NOT NULL,
  end INTEGER NOT NULL,
  tags_json TEXT NOT NULL,
  notes TEXT DEFAULT '',
  created_at TEXT NOT NULL,
  FOREIGN KEY(document_id) REFERENCES documents(id)
);
CREATE TABLE IF NOT EXISTS analysis_runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id INTEGER NOT NULL,
  mode TEXT NOT NULL,
  params_json TEXT NOT NULL,
  backend TEXT NOT NULL,
  versions_json TEXT NOT NULL,
  warnings_json TEXT NOT NULL,
  result_json_path TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(document_id) REFERENCES documents(id)
);
"""


class ProjectStore:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.project_dir / "raw"
        self.results_dir = self.project_dir / "results"
        self.raw_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.db_path = self.project_dir / "case.db"
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        with self._lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def now(self) -> str:
        return datetime.utcnow().isoformat(timespec="seconds")

    def create_case(self, title: str, notes: str = "") -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO cases(title, created_at, notes) VALUES (?,?,?)",
                (title, self.now(), notes),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def add_document(self, case_id: int, role: str, title: str, source_path: str, text_sha256: str) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO documents(case_id, role, title, source_path, text_sha256, created_at) VALUES (?,?,?,?,?,?)",
                (case_id, role, title, source_path, text_sha256, self.now()),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def add_fragment(self, document_id: int, start: int, end: int, tags: list[str], notes: str) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO fragments(document_id,start,end,tags_json,notes,created_at) VALUES (?,?,?,?,?,?)",
                (document_id, start, end, json.dumps(tags, ensure_ascii=False), notes, self.now()),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def add_run(
        self,
        document_id: int,
        mode: str,
        params: dict[str, Any],
        backend: str,
        versions: dict[str, str],
        warnings: list[str],
        result_json_path: str,
    ) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO analysis_runs(document_id,mode,params_json,backend,versions_json,warnings_json,result_json_path,created_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    document_id,
                    mode,
                    json.dumps(params, ensure_ascii=False),
                    backend,
                    json.dumps(versions, ensure_ascii=False),
                    json.dumps(warnings, ensure_ascii=False),
                    result_json_path,
                    self.now(),
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def list_documents(self, case_id: int) -> list[tuple]:
        with self._lock:
            cur = self.conn.execute(
                "SELECT id, role, title, source_path, text_sha256, created_at FROM documents WHERE case_id=? ORDER BY id DESC",
                (case_id,),
            )
            return list(cur.fetchall())

    def list_runs(self, document_id: int) -> list[tuple]:
        with self._lock:
            cur = self.conn.execute(
                "SELECT id, mode, backend, created_at, result_json_path FROM analysis_runs WHERE document_id=? ORDER BY id DESC",
                (document_id,),
            )
            return list(cur.fetchall())
