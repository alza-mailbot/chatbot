"""Unit tests for the application lifespan. The Gemini client class is mocked."""

from unittest.mock import patch

from chatbot.main import app, lifespan


class TestLifespan:
    """Tests for startup and shutdown behaviour."""

    async def test_creates_gemini_client_on_startup(self) -> None:
        """Verify the shared Gemini client is created and stored in app state."""
        with patch("chatbot.main.GeminiClient") as mock_cls:
            async with lifespan(app):
                assert app.state.gemini is mock_cls.return_value
