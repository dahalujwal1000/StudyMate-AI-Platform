import threading

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import Document, User
from ..services import pipeline
from ..services.rag import rebuild_index

router = APIRouter(prefix="/api/documents")

ALLOWED = {".pdf", ".pptx", ".docx", ".txt", ".md"}


@router.post("/upload")
def upload(file: UploadFile = File(...), db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED))}")
    raw = file.file.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 20 MB)")
    doc = Document(
        user_id=user.id,
        title=file.filename.rsplit(".", 1)[0],
        filename=file.filename,
        mime_type=file.content_type or "",
        size_bytes=len(raw),
        status="processing",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    path = pipeline.save_upload(doc, raw)
    doc.stored_path = str(path)
    db.commit()

    # process in background so upload returns instantly (status polls to 'ready')
    db.expunge(doc)
    threading.Thread(target=_process, args=(doc.id,), daemon=True).start()

    return {"id": doc.id, "status": "processing"}


def _process(doc_id: int):
    from ..database import SessionLocal

    db = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        if doc:
            pipeline.process_document(db, doc)
        db.close()
        index_db = SessionLocal()
        try:
            rebuild_index(index_db)
        finally:
            index_db.close()
    except Exception as exc:  # noqa: BLE001 - never crash the worker thread
        print(f"[process] doc {doc_id} failed: {exc}")
        db.close()



@router.get("")
def list_docs(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    docs = (
        db.query(Document)
        .filter(Document.user_id == user.id)
        .order_by(Document.created_at.desc())
        .all()
    )
    return [
        {
            "id": d.id, "title": d.title, "filename": d.filename, "status": d.status,
            "size_bytes": d.size_bytes, "page_count": d.page_count, "summary": d.summary,
            "key_topics": d.key_topics, "error": d.error,
            "created_at": d.created_at.isoformat(),
        }
        for d in docs
    ]


@router.delete("/{doc_id}")
def delete_doc(doc_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    doc = db.get(Document, doc_id)
    if not doc or doc.user_id != user.id:
        raise HTTPException(404, "Document not found")
    db.delete(doc)
    db.commit()
    rebuild_index(db)
    return {"ok": True}
