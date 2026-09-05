from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import Document, Quiz, StudyTask, User

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))

NAV = [
    {"href": "/dashboard", "label": "Dashboard", "icon": "space_dashboard"},
    {"href": "/upload", "label": "Upload", "icon": "upload_file"},
    {"href": "/quiz", "label": "Quiz", "icon": "quiz"},
    {"href": "/planner", "label": "Planner", "icon": "calendar_month"},
    {"href": "/progress", "label": "Progress", "icon": "monitoring"},
]


def render(request: Request, template: str, user: User, active: str, **ctx):
    return templates.TemplateResponse(request, template, {
        "user": user, "nav": NAV, "active": active, **ctx,
    })


def _redirect_if_anon(user: User | None) -> RedirectResponse | None:
    if user is None:
        return RedirectResponse("/login", status_code=302)
    return None


@router.get("/")
def landing(request: Request, db: Session = Depends(get_db)):
    from ..routers.auth_routes import current_user_optional

    user = current_user_optional(request, db)
    if user:
        return RedirectResponse("/dashboard", status_code=302)
    error = request.query_params.get("error")
    return templates.TemplateResponse(request, "landing.html", {"error": error})


@router.get("/login")
def login_page():
    return RedirectResponse("/auth/login", status_code=302)


@router.get("/dashboard")
def dashboard(request: Request, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    docs = (
        db.query(Document).filter(Document.user_id == user.id)
        .order_by(Document.created_at.desc()).limit(5).all()
    )
    tasks = (
        db.query(StudyTask).filter(StudyTask.user_id == user.id, StudyTask.status == "todo")
        .order_by(StudyTask.due_date).limit(5).all()
    )
    quizzes = db.query(Quiz).filter(Quiz.user_id == user.id).order_by(Quiz.created_at.desc()).limit(3).all()
    ready = sum(1 for d in docs if d.status == "ready")
    return render(request, "dashboard.html", user, "dashboard",
                  docs=docs, tasks=tasks, quizzes=quizzes, doc_count=ready)


@router.get("/upload")
def upload_page(request: Request, user: User = Depends(get_current_user)):
    return render(request, "upload.html", user, "upload")


@router.get("/documents/{doc_id}")
def viewer(doc_id: int, request: Request, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    doc = db.get(Document, doc_id)
    if not doc or doc.user_id != user.id:
        raise HTTPException(404, "Document not found")
    return render(request, "viewer.html", user, "dashboard", doc=doc)


@router.get("/quiz")
def quiz_list(request: Request, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    docs = db.query(Document).filter(Document.user_id == user.id, Document.status == "ready").all()
    quizzes = db.query(Quiz).filter(Quiz.user_id == user.id).order_by(Quiz.created_at.desc()).all()
    return render(request, "quiz.html", user, "quiz", docs=docs, quizzes=quizzes)


@router.get("/quiz/{quiz_id}")
def quiz_take(quiz_id: int, request: Request, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    quiz = db.get(Quiz, quiz_id)
    if not quiz or quiz.user_id != user.id:
        raise HTTPException(404, "Quiz not found")
    return render(request, "quiz_take.html", user, "quiz", quiz=quiz)


@router.get("/planner")
def planner(request: Request, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    tasks = (
        db.query(StudyTask).filter(StudyTask.user_id == user.id)
        .order_by(StudyTask.status, StudyTask.due_date, StudyTask.id).all()
    )
    return render(request, "planner.html", user, "planner", tasks=tasks)


@router.get("/progress")
def progress_page(request: Request, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    docs = db.query(Document).filter(Document.user_id == user.id).all()
    quizzes = db.query(Quiz).filter(Quiz.user_id == user.id).all()
    attempts = []
    for q in quizzes:
        attempts.extend(q.attempts)
    attempts.sort(key=lambda a: a.created_at)
    tasks = db.query(StudyTask).filter(StudyTask.user_id == user.id).all()

    topics = {}
    for q in quizzes:
        entry = topics.setdefault(q.title, [])
        for a in q.attempts:
            if a.total:
                entry.append(round(a.score / a.total * 100))
    topic_rows = sorted(
        ({"title": t, "mastery": max(scores, default=0)} for t, scores in topics.items()),
        key=lambda x: x["mastery"],
    )
    return render(request, "progress.html", user, "progress",
                  docs=docs, attempts=attempts, tasks=tasks,
                  topics=topic_rows)


@router.get("/settings")
def settings_page(request: Request, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    from ..config import settings as cfg
    from ..services import llm

    return render(request, "settings.html", user, "settings",
                  llm_provider=cfg.llm_provider, llm_ready=llm.llm_available(),
                  llm_chain=llm._provider_order(),
                  oauth_ready=bool(cfg.google_client_id))


@router.get("/health")
def health():
    return {"ok": True}
