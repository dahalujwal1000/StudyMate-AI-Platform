"""Authentication: Google OAuth (authlib) + demo login fallback."""
from __future__ import annotations

from functools import wraps

from authlib.integrations.starlette_client import OAuth
from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from .config import settings
from .database import get_db
from .models import User

oauth = OAuth()
if settings.google_client_id:
    oauth.register(
        name="google",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

SESSION_KEY = "user_id"


def google_oauth_enabled() -> bool:
    return bool(settings.google_client_id and settings.google_client_secret)


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    user_id = request.session.get(SESSION_KEY)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not signed in")
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="Not signed in")
    return user


def login_user(request: Request, user: User) -> None:
    request.session[SESSION_KEY] = user.id
    request.session["max_age"] = settings.session_max_age


def logout_user(request: Request) -> None:
    request.session.pop(SESSION_KEY, None)


def get_or_create_user(db: Session, email: str, name: str = "", avatar_url: str = "",
                       google_sub: str = "", is_demo: bool = False) -> User:
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        user = User(email=email, name=name or email.split("@")[0],
                    avatar_url=avatar_url, google_sub=google_sub, is_demo=is_demo)
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        changed = False
        if name and user.name != name:
            user.name, changed = name, True
        if avatar_url and user.avatar_url != avatar_url:
            user.avatar_url, changed = avatar_url, True
        if google_sub and user.google_sub != google_sub:
            user.google_sub, changed = google_sub, True
        if changed:
            db.commit()
    return user
