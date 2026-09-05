from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import Chunk, Document, Quiz, QuizAttempt, QuizQuestion, User
from ..services.quiz_gen import generate_flashcards, generate_mcqs

router = APIRouter(prefix="/api/quiz")


class GenerateIn(BaseModel):
    document_id: int
    kind: str = "mcq"  # mcq | flashcards
    count: int = 5


@router.post("/generate")
def generate(body: GenerateIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    doc = db.get(Document, body.document_id)
    if not doc or doc.user_id != user.id:
        raise HTTPException(404, "Document not found")
    if doc.status != "ready":
        raise HTTPException(400, f"Document is {doc.status}; try again once it's ready")

    chunks = [c.content for c in db.query(Chunk).filter(Chunk.document_id == doc.id).all()]
    if not chunks:
        raise HTTPException(400, "No content indexed for this document")

    count = max(3, min(body.count, 12))
    if body.kind == "flashcards":
        cards = generate_flashcards(doc.title, chunks, n=count)
        if not cards:
            raise HTTPException(500, "Could not generate flashcards")
        quiz = Quiz(user_id=user.id, document_id=doc.id, kind="flashcards",
                    title=f"{doc.title} — Flashcards")
        db.add(quiz)
        db.flush()
        for i, card in enumerate(cards):
            db.add(QuizQuestion(quiz_id=quiz.id, position=i, question=card["front"],
                                answer_text=card["back"]))
        db.commit()
        return {"quiz_id": quiz.id, "kind": "flashcards", "count": len(cards)}

    mcqs = generate_mcqs(doc.title, chunks, n=count)
    if not mcqs:
        raise HTTPException(500, "Could not generate questions")
    quiz = Quiz(user_id=user.id, document_id=doc.id, kind="mcq", title=f"{doc.title} — Quiz")
    db.add(quiz)
    db.flush()
    for i, q in enumerate(mcqs):
        db.add(QuizQuestion(
            quiz_id=quiz.id, position=i, question=q["question"], options=q["options"],
            answer_idx=q["answer_idx"], explanation=q["explanation"],
        ))
    db.commit()
    return {"quiz_id": quiz.id, "kind": "mcq", "count": len(mcqs)}


@router.get("")
def list_quizzes(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    quizzes = db.query(Quiz).filter(Quiz.user_id == user.id).order_by(Quiz.created_at.desc()).all()
    out = []
    for q in quizzes:
        attempts = q.attempts
        best = max((a.score for a in attempts), default=None)
        out.append({
            "id": q.id, "title": q.title, "kind": q.kind,
            "document_id": q.document_id, "question_count": len(q.questions),
            "attempts": len(attempts),
            "best_score": best,
            "created_at": q.created_at.isoformat(),
        })
    return out


class SubmitIn(BaseModel):
    answers: list[int]  # selected option index per question (-1 = skipped)


@router.post("/{quiz_id}/submit")
def submit(quiz_id: int, body: SubmitIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    quiz = db.get(Quiz, quiz_id)
    if not quiz or quiz.user_id != user.id:
        raise HTTPException(404, "Quiz not found")
    questions = quiz.questions
    score = 0
    per_q = []
    for i, q in enumerate(questions):
        given = body.answers[i] if i < len(body.answers) else -1
        correct = given == q.answer_idx
        score += int(correct)
        per_q.append({"question": q.question, "given": given, "answer_idx": q.answer_idx,
                      "correct": correct, "explanation": q.explanation})
    attempt = QuizAttempt(quiz_id=quiz.id, user_id=user.id, score=score,
                          total=len(questions), answers=per_q)
    db.add(attempt)
    db.commit()
    return {"attempt_id": attempt.id, "score": score, "total": len(questions), "review": per_q}
