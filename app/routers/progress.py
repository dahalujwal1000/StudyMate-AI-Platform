from collections import defaultdict

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import Document, Quiz, QuizAttempt, StudyTask, User

router = APIRouter(prefix="/api/progress")


@router.get("")
def progress(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    docs = db.query(Document).filter(Document.user_id == user.id).order_by(Document.created_at).all()
    quizzes = db.query(Quiz).filter(Quiz.user_id == user.id).all()
    attempts = (
        db.query(QuizAttempt)
        .filter(QuizAttempt.user_id == user.id)
        .order_by(QuizAttempt.created_at)
        .all()
    )
    tasks = db.query(StudyTask).filter(StudyTask.user_id == user.id).all()

    # topic mastery = best score per quiz document
    per_doc: dict[int, dict] = {}
    for q in quizzes:
        entry = per_doc.setdefault(q.document_id, {"scores": [], "title": q.title, "kind": q.kind})
        for a in q.attempts:
            if a.total:
                entry["scores"].append(round(a.score / a.total * 100))
    topics = [
        {"title": v["title"], "mastery": max(v["scores"], default=0),
         "attempts": len(v["scores"])}
        for v in per_doc.values()
    ]
    topics.sort(key=lambda t: t["mastery"])
    weak = topics[:3]

    score_history = [
        {"date": a.created_at.date().isoformat(), "pct": round(a.score / a.total * 100) if a.total else 0}
        for a in attempts if a.total
    ]

    tasks_done = sum(1 for t in tasks if t.status == "done")
    by_course = defaultdict(lambda: {"total": 0, "done": 0})
    for t in tasks:
        key = t.course or "General"
        by_course[key]["total"] += 1
        by_course[key]["done"] += int(t.status == "done")

    minutes_studied = sum(a.total for a in attempts) * 4  # ~4 min per question estimate

    return {
        "documents": len(docs),
        "ready_docs": sum(1 for d in docs if d.status == "ready"),
        "topics": [{"title": t["title"], "mastery": t["mastery"]} for t in topics],
        "weak_topics": weak,
        "score_history": score_history,
        "avg_score": round(sum(h["pct"] for h in score_history) / len(score_history)) if score_history else 0,
        "quiz_attempts": len(attempts),
        "tasks_total": len(tasks),
        "tasks_done": tasks_done,
        "tasks_by_course": dict(by_course),
        "minutes_studied": minutes_studied,
    }
