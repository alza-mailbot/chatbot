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


class TestGetSettings:
    """Tests for the cached settings accessor."""

    def test_returns_cached_instance(self) -> None:
        """Verify repeated calls return the same Settings instance."""
        get_settings.cache_clear()

        assert get_settings() is get_settings()
