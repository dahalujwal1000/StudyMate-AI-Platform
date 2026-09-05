"""Document processing pipeline: extract -> summarize -> chunk -> index."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from ..config import UPLOAD_DIR
from ..models import Chunk, Document
from . import llm, parser, rag
from .llm import extractive_summary, top_keywords

SYSTEM_SUMMARY = (
    "You are an academic assistant. Summarize study material clearly for a university student. "
    "Be concise, structured and factual. Use only the provided material."
)


def save_upload(doc: Document, raw: bytes) -> Path:
    path = UPLOAD_DIR / f"{doc.user_id}_{doc.id}{Path(doc.filename).suffix.lower()}"
    path.write_bytes(raw)
    return path


def process_document(db: Session, doc: Document) -> None:
    """Parse, summarize, chunk and index a document. Sets status ready/error."""
    try:
        path = Path(doc.stored_path)
        extracted = parser.extract_chunks(path, doc.mime_type)
        if not extracted:
            raise ValueError("No readable text found in file")

        full_text = "\n\n".join(t for _, t in extracted)

        # --- chunks ---
        doc.page_count = len(extracted)
        db.query(Chunk).filter(Chunk.document_id == doc.id).delete()
        for i, (label, text) in enumerate(extracted):
            for piece in rag.chunk_text(text):
                db.add(Chunk(document_id=doc.id, idx=i, label=label, content=piece))
        db.flush()

        # --- summary (LLM if available, else extractive) ---
        summary = llm.generate(
            f"Summarize the following study material in 5-8 bullet points "
            f"followed by one line 'Focus areas:' listing 3 key topics.\n\n"
            f"{full_text[:8000]}",
            system=SYSTEM_SUMMARY,
            temperature=0.3,
        )
        doc.summary = summary.strip() if summary else extractive_summary(full_text)
        doc.key_topics = top_keywords(full_text, k=6)
        doc.status = "ready"
        db.commit()
        rag.rebuild_index(db)
    except Exception as exc:  # noqa: BLE001
        doc.status = "error"
        doc.error = str(exc)[:480]
        db.commit()
