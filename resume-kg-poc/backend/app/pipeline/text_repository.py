"""Step 1 — Text Repository (paper Fig.2 Dataflow).

Temporarily holds raw input text/documents before NLP processing.
For resumes: holds PDF-extracted text + metadata.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class TextDocument:
    doc_id: str
    raw_text: str
    source: str = "resume_pdf"  # resume_pdf | resume_text | sample
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    char_count: int = 0
    sentence_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.raw_text)


class TextRepository:
    """In-memory text repository (research-friendly, no DB per requirements)."""

    def __init__(self):
        self._store: dict[str, TextDocument] = {}

    def add(self, doc: TextDocument) -> TextDocument:
        self._store[doc.doc_id] = doc
        return doc

    def get(self, doc_id: str) -> TextDocument:
        return self._store[doc_id]

    def all(self) -> list[TextDocument]:
        return list(self._store.values())
