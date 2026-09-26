"""Unit tests for chat API models and thread parsing."""

from typing import Literal

import pytest
from pydantic import ValidationError

from chatbot.models.chat import ThreadMessage, parse_thread


class TestThreadMessage:
    """Tests for the ThreadMessage model."""

    @pytest.mark.parametrize("role", ["user", "assistant"])
    def test_known_roles_are_accepted(self, role: Literal["user", "assistant"]) -> None:
        """Verify both conversation roles validate."""
        message = ThreadMessage(role=role, text="Hello")

        assert message.role == role
        assert message.text == "Hello"

    def test_unknown_role_is_rejected(self) -> None:
        """Verify a role outside the allowed pair fails validation at runtime."""
        with pytest.raises(ValidationError):
            ThreadMessage(role="system", text="Hello")  # ty: ignore[invalid-argument-type]

    def test_missing_text_is_rejected(self) -> None:
        """Verify a message without text fails validation at runtime."""
        with pytest.raises(ValidationError):
            ThreadMessage(role="user")  # ty: ignore[missing-argument]


class TestParseThread:
    """Tests for parse_thread."""

    def test_parses_json_array_in_order(self) -> None:
        """Verify a JSON array becomes ThreadMessage objects in order."""
        raw = '[{"role": "user", "text": "Q1"}, {"role": "assistant", "text": "A1"}]'

        thread = parse_thread(raw)

        assert thread == [
            ThreadMessage(role="user", text="Q1"),
            ThreadMessage(role="assistant", text="A1"),
        ]

    @pytest.mark.parametrize("raw", [None, ""])
    def test_missing_value_means_empty_thread(self, raw: str | None) -> None:
        """Verify absent input parses as an empty history."""
        assert parse_thread(raw) == []

    def test_invalid_json_is_rejected(self) -> None:
        """Verify malformed JSON raises a ValueError."""
        with pytest.raises(ValueError):
            parse_thread("not json")

    def test_invalid_schema_is_rejected(self) -> None:
        """Verify a valid JSON array with a wrong shape raises a ValueError."""
        with pytest.raises(ValueError):
            parse_thread('[{"role": "user"}]')
