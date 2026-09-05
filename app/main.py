from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from .config import BASE_DIR, settings
from .database import init_db
from .routers import auth_routes, chat, documents, pages, planner, progress, quiz

app = FastAPI(title="StudyMate AI", version="1.0.0")

app.add_middleware(SessionMiddleware, secret_key=settings.secret_key,
                   max_age=settings.session_max_age)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")

init_db()

app.include_router(auth_routes.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(quiz.router)
app.include_router(planner.router)
app.include_router(progress.router)
app.include_router(pages.router)


@app.exception_handler(404)
async def not_found(request: Request, exc):
    if request.url.path.startswith("/api/"):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "Not found"}, status_code=404)
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/", status_code=302)
