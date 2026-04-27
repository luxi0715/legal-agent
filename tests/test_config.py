"""Test configuration loading."""

from legal_agent.core.config import get_settings


def test_settings_loaded() -> None:
    """Settings should load from .env without errors."""
    settings = get_settings()
    assert settings.deepseek_api_key.startswith("sk-")
    assert settings.deepseek_base_url == "https://api.deepseek.com"
    assert settings.deepseek_model == "deepseek-chat"
    assert settings.app_port == 8000
