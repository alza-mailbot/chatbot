"""Unit tests for the Gemini LLM client. The google-genai SDK is mocked."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chatbot.config import Settings
from chatbot.core.llm.gemini import GeminiClient, LLMError


def _make_settings() -> Settings:
    """Return settings with deterministic values for assertions."""
    return Settings(
        _env_file=None,
        gcp_project_id="test-project",
        gcp_location="test-location",
        gemini_model="gemini-test",
        system_prompt="Test persona",
    )


def _make_client(
    response_text: str | None = "Generated reply",
) -> tuple[GeminiClient, AsyncMock, MagicMock]:
    """Build a GeminiClient whose SDK is fully mocked.

    Returns:
        tuple: The client, the mocked generate_content coroutine and the
            mocked genai module.
    """
    with patch("chatbot.core.llm.gemini.genai") as mock_genai:
        sdk_client = MagicMock()
        mock_genai.Client.return_value = sdk_client
        client = GeminiClient(_make_settings())
    generate = AsyncMock(return_value=SimpleNamespace(text=response_text))
    sdk_client.aio.models.generate_content = generate
    return client, generate, mock_genai


class TestInit:
    """Tests for GeminiClient construction."""

    def test_creates_vertex_ai_client_from_settings(self) -> None:
        """Verify the SDK client is created for Vertex AI with configured project."""
        _, _, mock_genai = _make_client()

        mock_genai.Client.assert_called_once_with(
            vertexai=True, project="test-project", location="test-location"
        )


class TestGenerateReply:
    """Tests for GeminiClient.generate_reply."""

    async def test_returns_model_text(self) -> None:
        """Verify the reply is the text returned by the model."""
        client, _, _ = _make_client(response_text="Hello there")

        reply = await client.generate_reply(subject="Hi", body="Question?")

        assert reply == "Hello there"

    async def test_passes_model_and_system_prompt(self) -> None:
        """Verify the configured model name and system prompt reach the SDK call."""
        client, generate, _ = _make_client()

        await client.generate_reply(subject="Hi", body="Question?")

        kwargs = generate.call_args.kwargs
        assert kwargs["model"] == "gemini-test"
        assert kwargs["config"].system_instruction == "Test persona"

    async def test_contents_include_subject_and_body(self) -> None:
        """Verify both subject and body are part of the prompt contents."""
        client, generate, _ = _make_client()

        await client.generate_reply(subject="Warranty claim", body="My laptop broke.")

        contents = str(generate.call_args.kwargs["contents"])
        assert "Warranty claim" in contents
        assert "My laptop broke." in contents

    async def test_sdk_error_raises_llm_error(self) -> None:
        """Verify an SDK failure surfaces as LLMError with the original cause."""
        client, generate, _ = _make_client()
        generate.side_effect = RuntimeError("boom")

        with pytest.raises(LLMError) as exc_info:
            await client.generate_reply(subject="Hi", body="Question?")

        assert isinstance(exc_info.value.__cause__, RuntimeError)

    async def test_empty_response_raises_llm_error(self) -> None:
        """Verify a response without text is treated as a failure."""
        client, _, _ = _make_client(response_text=None)

        with pytest.raises(LLMError):
            await client.generate_reply(subject="Hi", body="Question?")
