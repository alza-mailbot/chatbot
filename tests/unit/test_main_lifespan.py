"""Unit tests for the application lifespan. The Gemini client class is mocked."""

from unittest.mock import patch

from chatbot.main import app, lifespan


class TestLifespan:
    """Tests for startup and shutdown behaviour."""

    async def test_creates_shared_resources_on_startup(self) -> None:
        """Verify the Gemini client and settings are stored in app state."""
        with patch("chatbot.main.GeminiClient") as mock_cls:
            async with lifespan(app):
                assert app.state.gemini is mock_cls.return_value
                assert app.state.settings is not None
