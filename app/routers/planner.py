from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import StudyTask, User

router = APIRouter(prefix="/api/tasks")


class TaskIn(BaseModel):
    title: str
    course: str = ""
    task_type: str = "reading"
    due_date: str = ""
    duration_min: int = 30


@router.get("")
def list_tasks(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    tasks = (
        db.query(StudyTask)
        .filter(StudyTask.user_id == user.id)
        .order_by(StudyTask.status, StudyTask.due_date, StudyTask.id)
        .all()
    )
    return [
        {"id": t.id, "title": t.title, "course": t.course, "task_type": t.task_type,
         "due_date": t.due_date, "duration_min": t.duration_min, "status": t.status}
        for t in tasks
    ]


@router.post("")
def create_task(body: TaskIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    if not body.title.strip():
        raise HTTPException(400, "Title required")
    task = StudyTask(user_id=user.id, title=body.title.strip(), course=body.course.strip(),
                     task_type=body.task_type, due_date=body.due_date, duration_min=body.duration_min)
    db.add(task)
    db.commit()
    db.refresh(task)
    return {"id": task.id, "title": task.title, "course": task.course, "task_type": task.task_type,
            "due_date": task.due_date, "duration_min": task.duration_min, "status": task.status}


class TaskPatch(BaseModel):
    status: str | None = None
    title: str | None = None
    due_date: str | None = None


@router.patch("/{task_id}")
def update_task(task_id: int, body: TaskPatch, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    task = db.get(StudyTask, task_id)
    if not task or task.user_id != user.id:
        raise HTTPException(404, "Task not found")
    if body.status is not None:
        if body.status not in ("todo", "done"):
            raise HTTPException(400, "status must be todo|done")
        task.status = body.status
    if body.title is not None and body.title.strip():
        task.title = body.title.strip()
    if body.due_date is not None:
        task.due_date = body.due_date
    db.commit()
    return {"id": task.id, "status": task.status, "title": task.title, "due_date": task.due_date}


@router.delete("/{task_id}")
def delete_task(task_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    task = db.get(StudyTask, task_id)
    if not task or task.user_id != user.id:
        raise HTTPException(404, "Task not found")
    db.delete(task)
    db.commit()
    return {"ok": True}
