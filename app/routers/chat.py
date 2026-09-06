import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import ChatMessage, Chunk, Document, User
from ..services import llm
from ..services.rag import get_index

router = APIRouter(prefix="/api/chat")

SYSTEM_CHAT = (
    "You are StudyMate, a helpful academic tutor. You are given the document's AI summary and "
    "usually numbered excerpts [1], [2]… taken from the study material — all of this counts as "
    "the study material. Answer the student's question using this material and cite excerpts "
    "like [1], [2] when you use them. If the question is about the summary, the overall topic, "
    "or asks you to explain/simplify something, base your answer on the summary. Never claim "
    "that no material or excerpts were provided. Only if the material genuinely lacks the "
    "specific fact asked about, say so honestly and suggest what to check. Be concise and clear."
)


class ChatIn(BaseModel):
    document_id: int
    message: str


class AskIn(BaseModel):
    message: str


@router.post("")
def chat(body: ChatIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    doc = db.get(Document, body.document_id)
    if not doc or doc.user_id != user.id:
        raise HTTPException(404, "Document not found")

    user_msg = ChatMessage(user_id=user.id, document_id=doc.id, role="user", content=body.message)
    db.add(user_msg)

    hits = get_index().search(body.message, k=4, document_id=doc.id)

    # Meta/vague questions ("explain the summary", "what is this about?") often share no
    # keywords with the chunks, so TF-IDF retrieval can return nothing. Fall back to the
    # document's opening chunks so the model always has real material to answer from.
    overview: list[Chunk] = []
    if not hits:
        overview = (
            db.query(Chunk)
            .filter(Chunk.document_id == doc.id)
            .order_by(Chunk.idx)
            .limit(2)
            .all()
        )

    answer, citations = _answer(doc.title, body.message, hits, doc.summary or "", overview, _prior_history(db, user.id, doc.id))

    ai_msg = ChatMessage(user_id=user.id, document_id=doc.id, role="assistant",
                         content=answer, citations=citations)
    db.add(ai_msg)
    db.commit()
    db.refresh(ai_msg)
    return {
        "answer": answer,
        "citations": citations,
        "message_id": ai_msg.id,
        "grounded": bool(hits or overview or doc.summary),
    }


def _prior_history(db: Session, user_id: int, document_id: int) -> list[tuple[str, str]]:
    """Return the last N turns (user+assistant) so the model can follow context."""
    msgs = (
        db.query(ChatMessage)
        .filter(ChatMessage.user_id == user_id, ChatMessage.document_id == document_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(6)
        .all()
    )
    pairs: list[tuple[str, str]] = []
    for m in reversed(msgs):
        if m.role in ("user", "assistant") and m.content:
            pairs.append((m.role, m.content))
    return pairs[-6:]


def _answer(title: str, question: str, hits, summary: str = "", overview=(), history=None) -> tuple[str, list[dict]]:
    # hits = keyword-retrieved excerpts; overview = opening chunks used when retrieval
    # found nothing. history = prior [(role, content), ...] turns for multi-turn context.
    history = history or []
    excerpts = list(hits) if hits else list(overview)

    parts: list[str] = []
    if summary:
        parts.append(f"AI summary of '{title}':\n{summary}")
    parts.extend(f"[{i + 1}] ({h.label}) {h.content}" for i, h in enumerate(excerpts))
    context = "\n\n".join(parts)

    if llm.llm_available() and context:
        prompt = (
            f"Study material from '{title}':\n\n{context}\n\n"
            f"Student question: {question}"
        )
        out = llm.generate_chat(prompt, system=SYSTEM_CHAT, history=history, temperature=0.3)
        if out:
            cited = _used_citations(out, len(excerpts))
            return out.strip(), [
                {"n": i + 1, "label": excerpts[i].label, "excerpt": excerpts[i].content[:280]}
                for i in cited
            ]
    # fallback: grounded extractive answer
    if not excerpts:
        if summary:
            return f"Here's the summary of '{title}':\n\n{summary}", []
        return (
            "I couldn't find anything about that in this document yet. Try rephrasing, or upload "
            "material that covers this topic.",
            [],
        )
    top = excerpts[0]
    sentences = re.split(r"(?<=[.!?])\s+", top.content)
    best = sentences[:3]
    answer = f"Here's what your document says ({top.label}):\n\n" + " ".join(best)
    if len(excerpts) > 1:
        answer += f"\n\nRelated excerpt from {excerpts[1].label}: {excerpts[1].content[:200]}…"
    citations = [
        {"n": i + 1, "label": h.label, "excerpt": h.content[:280]} for i, h in enumerate(excerpts[:2])
    ]
    return answer, citations


def _used_citations(answer: str, n: int) -> list[int]:
    found = {int(m) - 1 for m in re.findall(r"\[(\d+)\]", answer) if 0 < int(m) <= n}
    return sorted(found)[:4]


@router.get("/{document_id}")
def history(document_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    msgs = (
        db.query(ChatMessage)
        .filter(ChatMessage.user_id == user.id, ChatMessage.document_id == document_id)
        .order_by(ChatMessage.created_at)
        .all()
    )
    return [
        {"role": m.role, "content": m.content, "citations": m.citations}
        for m in msgs
    ]
