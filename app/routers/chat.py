import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import ChatMessage, Document, User
from ..services import llm
from ..services.rag import get_index

router = APIRouter(prefix="/api/chat")

SYSTEM_CHAT = (
    "You are StudyMate, a helpful academic tutor. Answer the student's question using ONLY the "
    "provided study material excerpts. Cite excerpts like [1], [2] when used. If the material does "
    "not contain the answer, say so honestly and suggest what to check. Be concise and clear."
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
    answer, citations = _answer(doc.title, body.message, hits)

    ai_msg = ChatMessage(user_id=user.id, document_id=doc.id, role="assistant",
                         content=answer, citations=citations)
    db.add(ai_msg)
    db.commit()
    db.refresh(ai_msg)
    return {
        "answer": answer,
        "citations": citations,
        "message_id": ai_msg.id,
        "grounded": bool(hits),
    }


def _answer(title: str, question: str, hits) -> tuple[str, list[dict]]:
    context = "\n\n".join(f"[{i + 1}] ({h.label}) {h.content}" for i, h in enumerate(hits))
    if llm.llm_available():
        prompt = (
            f"Study material from '{title}':\n\n{context or '(no matching excerpts found)'}\n\n"
            f"Student question: {question}"
        )
        out = llm.generate(prompt, system=SYSTEM_CHAT, temperature=0.3)
        if out:
            cited = _used_citations(out, len(hits))
            return out.strip(), [
                {"n": i + 1, "label": hits[i].label, "excerpt": hits[i].content[:280]}
                for i in cited
            ]
    # fallback: grounded extractive answer
    if not hits:
        return (
            "I couldn't find anything about that in this document yet. Try rephrasing, or upload "
            "material that covers this topic.",
            [],
        )
    top = hits[0]
    sentences = re.split(r"(?<=[.!?])\s+", top.content)
    best = sentences[:3]
    answer = f"Here's what your document says ({top.label}):\n\n" + " ".join(best)
    if len(hits) > 1:
        answer += f"\n\nRelated excerpt from {hits[1].label}: {hits[1].content[:200]}…"
    citations = [
        {"n": i + 1, "label": h.label, "excerpt": h.content[:280]} for i, h in enumerate(hits[:2])
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
