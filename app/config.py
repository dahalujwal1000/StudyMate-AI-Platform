from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "dev"
    secret_key: str = "dev-secret-change-me"
    database_url: str = f"sqlite:///{DATA_DIR / 'studymate.db'}"

    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    groq_api_key: str = ""

    google_client_id: str = ""
    google_client_secret: str = ""
    oauth_redirect_base: str = "http://127.0.0.1:8000"

    session_max_age: int = 60 * 60 * 24 * 14  # 14 days


settings = Settings()

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
