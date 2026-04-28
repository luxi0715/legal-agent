"""Application configuration."""

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Settings(BaseModel):
    """All app settings, loaded from environment variables."""

    # DeepSeek
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # FastAPI
    app_host: str = "127.0.0.1"
    app_port: int = 8000

    # PostgreSQL
    postgres_dsn: str

    # Redis
    redis_url: str = "redis://localhost:6379/0"


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings(
        deepseek_api_key=os.environ["DEEPSEEK_API_KEY"],
        deepseek_base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        deepseek_model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        app_host=os.environ.get("APP_HOST", "127.0.0.1"),
        app_port=int(os.environ.get("APP_PORT", "8000")),
        postgres_dsn=os.environ["POSTGRES_DSN"],
        redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
    )
