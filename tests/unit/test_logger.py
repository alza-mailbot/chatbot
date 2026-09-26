"""Unit tests for the central application logger."""

import logging

from chatbot.utils.logger import logger


class TestLogger:
    """Tests for the shared logger instance."""

    def test_logger_uses_application_name(self) -> None:
        """Verify the shared logger is named after the application."""
        assert logger.name == "chatbot"

    def test_logger_has_a_handler(self) -> None:
        """Verify the logger has at least one handler attached."""
        assert logger.handlers

    def test_format_includes_level_name_and_message(self) -> None:
        """Verify formatted records carry the level, logger name and message."""
        formatter = logger.handlers[0].formatter
        assert formatter is not None
        record = logging.LogRecord(
            name="chatbot",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)

        assert "[INFO]" in formatted
        assert "chatbot" in formatted
        assert "hello" in formatted
