from dataclasses import dataclass


@dataclass
class Case:
    id: int
    title: str
    created_at: str
    notes: str


@dataclass
class Document:
    id: int
    case_id: int
    role: str
    title: str
    source_path: str
    text_sha256: str
    created_at: str
