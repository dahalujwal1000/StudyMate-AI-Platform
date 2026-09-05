from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from ..auth import google_oauth_enabled, get_or_create_user, login_user, logout_user, oauth
from ..config import settings
from ..database import get_db
from ..models import User

router = APIRouter(prefix="/auth")


@router.get("/login")
async def login(request: Request):
    """Start Google OAuth; falls back to demo login when not configured."""
    if google_oauth_enabled():
        base = settings.oauth_redirect_base.rstrip("/")
        redirect_uri = f"{base}/auth/callback"
        return await oauth.google.authorize_redirect(request, redirect_uri)
    return RedirectResponse("/auth/demo")


@router.get("/callback")
async def callback(request: Request, db: Session = Depends(get_db)):
    if request.query_params.get("error"):
        # e.g. user denied consent: ?error=access_denied
        return RedirectResponse("/?error=denied", status_code=302)
    if not request.query_params.get("code"):
        return RedirectResponse("/?error=missing_code", status_code=302)
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception:  # CSRF state mismatch / token exchange failure
        return RedirectResponse("/?error=oauth_failed", status_code=302)
    userinfo = token.get("userinfo") or {}
    email = userinfo.get("email")
    if not email:
        return RedirectResponse("/?error=no_email", status_code=302)
    user = get_or_create_user(
        db,
        email=email,
        name=userinfo.get("name", ""),
        avatar_url=userinfo.get("picture", ""),
        google_sub=userinfo.get("sub", ""),
    )
    login_user(request, user)
    return RedirectResponse("/dashboard", status_code=302)


@router.get("/demo")
def demo_login(request: Request, db: Session = Depends(get_db)):
    """One-click demo account for local testing without OAuth credentials."""
    user = get_or_create_user(db, email="demo@studymate.ai", name="Alex", is_demo=True)
    login_user(request, user)
    return RedirectResponse("/dashboard")


@router.get("/logout")
def logout(request: Request):
    logout_user(request)
    return RedirectResponse("/")


def current_user_optional(request: Request, db: Session = Depends(get_db)) -> User | None:
    uid = request.session.get("user_id")
    return db.get(User, uid) if uid else None
