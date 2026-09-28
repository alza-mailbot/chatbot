"""Unit tests for the Gemini LLM client. The google-genai SDK is mocked."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types

from chatbot.config import Settings
from chatbot.core.attachments import Attachment
from chatbot.core.llm.gemini import GeminiClient, LLMError, build_contents
from chatbot.models.chat import ThreadMessage


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
        assert kwargs["config"].system_instruction.startswith("Test persona")

    async def test_contents_include_subject_and_body(self) -> None:
        """Verify both subject and body are part of the prompt contents."""
        client, generate, _ = _make_client()

        await client.generate_reply(subject="Warranty claim", body="My laptop broke.")

        contents = str(generate.call_args.kwargs["contents"])
        assert "Warranty claim" in contents
        assert "My laptop broke." in contents

    async def test_attachments_become_parts_before_text(self) -> None:
        """Verify attachments are inline parts of the current turn, with the text last."""
        client, generate, _ = _make_client()
        attachment = Attachment(filename="doc.pdf", mime_type="application/pdf", data=b"%PDF")

        await client.generate_reply(subject="S", body="B", attachments=[attachment])

        current = generate.call_args.kwargs["contents"][-1]
        assert current.parts[0].inline_data.mime_type == "application/pdf"
        assert current.parts[0].inline_data.data == b"%PDF"
        assert "S" in current.parts[-1].text
        assert "B" in current.parts[-1].text

    async def test_no_attachments_sends_single_user_turn(self) -> None:
        """Verify the prompt is one user turn when no attachments or history come in."""
        client, generate, _ = _make_client()

        await client.generate_reply(subject="S", body="B")

        contents = generate.call_args.kwargs["contents"]
        assert len(contents) == 1
        assert contents[0].role == "user"

    async def test_thread_history_becomes_role_turns(self) -> None:
        """Verify prior messages precede the current turn with mapped roles."""
        client, generate, _ = _make_client()
        thread = [
            ThreadMessage(role="user", text="Q1"),
            ThreadMessage(role="assistant", text="A1"),
        ]

        await client.generate_reply(subject="S", body="Follow-up", thread=thread)

        contents = generate.call_args.kwargs["contents"]
        assert len(contents) == 3
        assert (contents[0].role, contents[0].parts[0].text) == ("user", "Q1")
        assert (contents[1].role, contents[1].parts[0].text) == ("model", "A1")
        assert contents[2].role == "user"
        assert "Follow-up" in contents[2].parts[-1].text


def _turn(text: str) -> types.Content:
    """Build one user turn with the given text."""
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


class TestGenerate:
    """Tests for the low-level GeminiClient.generate call."""

    async def test_passthrough_of_prebuilt_contents(self) -> None:
        """Verify generate sends the given contents unchanged to the SDK."""
        client, generate, _ = _make_client()
        contents = [_turn("turn-1"), _turn("turn-2")]

        await client.generate(contents)

        assert generate.call_args.kwargs["contents"] is contents

    async def test_returns_full_response_object(self) -> None:
        """Verify the caller gets the whole response, not just its text."""
        client, generate, _ = _make_client()

        response = await client.generate([_turn("turn")])

        assert response is generate.return_value

    async def test_tools_reach_the_config(self) -> None:
        """Verify declared tools travel inside the request config."""
        client, generate, _ = _make_client()
        tools = [types.Tool(function_declarations=[types.FunctionDeclaration(name="t")])]

        await client.generate([_turn("turn")], tools=tools)

        config = generate.call_args.kwargs["config"]
        assert config.tools == tools
        assert config.system_instruction.startswith("Test persona")

    async def test_todays_date_follows_the_persona(self) -> None:
        """Verify every call tells the model what day it is."""
        client, generate, _ = _make_client()

        await client.generate([_turn("turn")])

        instruction = generate.call_args.kwargs["config"].system_instruction
        assert instruction.startswith("Test persona")
        assert date.today().isoformat() in instruction

    async def test_no_tools_by_default(self) -> None:
        """Verify the config declares no tools unless some are passed."""
        client, generate, _ = _make_client()

        await client.generate([_turn("turn")])

        assert generate.call_args.kwargs["config"].tools is None

    async def test_generate_sdk_error_raises_llm_error(self) -> None:
        """Verify an SDK failure in the low-level call surfaces as LLMError."""
        client, generate, _ = _make_client()
        generate.side_effect = RuntimeError("boom")

        with pytest.raises(LLMError):
            await client.generate([_turn("turn")])


class TestBuildContents:
    """Tests for the prompt builder shared by the plain path and the agent."""

    def test_thread_then_current_turn_with_attachments(self) -> None:
        """Verify history precedes the current turn and attachments precede text."""
        thread = [
            ThreadMessage(role="user", text="Q1"),
            ThreadMessage(role="assistant", text="A1"),
        ]
        attachment = Attachment(filename="doc.pdf", mime_type="application/pdf", data=b"%PDF")

        contents = build_contents(subject="S", body="B", attachments=[attachment], thread=thread)

        assert [content.role for content in contents] == ["user", "model", "user"]
        parts = contents[-1].parts
        assert parts is not None
        assert parts[0].inline_data is not None
        assert parts[0].inline_data.mime_type == "application/pdf"
        assert parts[-1].text is not None
        assert "S" in parts[-1].text
        assert "B" in parts[-1].text


class TestGenerateReplyErrors:
    """Tests for error mapping of GeminiClient.generate_reply."""

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
