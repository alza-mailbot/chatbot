"""Unit tests for application settings."""

import pytest
from pydantic import ValidationError

from chatbot.config import Settings, get_settings


class TestSettings:
    """Tests for the Settings model."""

    def test_defaults_without_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify default values apply when no optional environment variables are set."""
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        monkeypatch.delenv("PORT", raising=False)

        settings = Settings(_env_file=None)

        assert settings.log_level == "INFO"
        assert settings.port == 8080
        assert settings.gcp_location == "global"
        assert settings.gemini_model == "gemini-2.5-flash"
        assert settings.system_prompt
        assert settings.max_attachment_bytes == 15 * 1024 * 1024

    def test_missing_project_id_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify settings fail fast when the required GCP project id is absent."""
        monkeypatch.delenv("GCP_PROJECT_ID", raising=False)

        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_environment_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify an environment variable overrides the default value."""
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        settings = Settings(_env_file=None)

        assert settings.log_level == "DEBUG"

    def test_invalid_port_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify a non-numeric port raises a validation error."""
        monkeypatch.setenv("PORT", "not-a-number")

        with pytest.raises(ValidationError):
            Settings(_env_file=None)


class TestWebSearchSettings:
    """Tests for the web search feature configuration."""

    def test_disabled_by_default_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify web search is off and keyless unless explicitly configured."""
        monkeypatch.delenv("WEB_SEARCH_ENABLED", raising=False)
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)

        settings = Settings(_env_file=None)

        assert settings.web_search_enabled is False
        assert settings.brave_api_key is None
        assert settings.brave_max_results == 5
        assert settings.agent_max_iterations == 6
        assert settings.agent_deadline_seconds == 90

    def test_enabled_with_key_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify the flag and key are read from the environment together."""
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")

        settings = Settings(_env_file=None)

        assert settings.web_search_enabled is True
        assert settings.brave_api_key == "test-key"

    def test_enabled_without_key_fails_fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify a misconfiguration is rejected at startup, not at request time."""
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)

        with pytest.raises(ValidationError):
            Settings(_env_file=None)


class TestGetSettings:
    """Tests for the cached settings accessor."""

    def test_returns_cached_instance(self) -> None:
        """Verify repeated calls return the same Settings instance."""
        get_settings.cache_clear()

        assert get_settings() is get_settings()
