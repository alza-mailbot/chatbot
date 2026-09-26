"""Unit tests for attachment validation."""

import pytest

from chatbot.core.attachments import (
    SUPPORTED_MIME_TYPES,
    Attachment,
    AttachmentError,
    AttachmentTooLargeError,
    UnsupportedAttachmentError,
    validate_attachment,
)

_MAX_BYTES = 1024


class TestValidateAttachment:
    """Tests for validate_attachment."""

    @pytest.mark.parametrize("mime", sorted(SUPPORTED_MIME_TYPES))
    def test_supported_types_pass(self, mime: str) -> None:
        """Verify every supported mime type is accepted and normalized."""
        attachment = validate_attachment("file.bin", mime, b"data", max_bytes=_MAX_BYTES)

        assert attachment == Attachment(filename="file.bin", mime_type=mime, data=b"data")

    def test_wav_alias_is_normalized(self) -> None:
        """Verify the common audio/x-wav alias maps to audio/wav."""
        attachment = validate_attachment("sound.wav", "audio/x-wav", b"data", max_bytes=_MAX_BYTES)

        assert attachment.mime_type == "audio/wav"

    def test_unsupported_type_is_rejected(self) -> None:
        """Verify an unsupported type raises with filename and supported list."""
        with pytest.raises(UnsupportedAttachmentError) as exc_info:
            validate_attachment("archive.zip", "application/zip", b"data", max_bytes=_MAX_BYTES)

        message = str(exc_info.value)
        assert "archive.zip" in message
        assert "application/pdf" in message

    def test_oversized_attachment_is_rejected(self) -> None:
        """Verify data above the size limit raises AttachmentTooLargeError."""
        with pytest.raises(AttachmentTooLargeError) as exc_info:
            validate_attachment(
                "big.pdf", "application/pdf", b"x" * (_MAX_BYTES + 1), max_bytes=_MAX_BYTES
            )

        assert "big.pdf" in str(exc_info.value)

    def test_size_exactly_at_limit_passes(self) -> None:
        """Verify data exactly at the size limit is accepted."""
        attachment = validate_attachment(
            "ok.pdf", "application/pdf", b"x" * _MAX_BYTES, max_bytes=_MAX_BYTES
        )

        assert len(attachment.data) == _MAX_BYTES

    def test_empty_attachment_is_rejected(self) -> None:
        """Verify an empty file raises AttachmentError."""
        with pytest.raises(AttachmentError):
            validate_attachment("empty.pdf", "application/pdf", b"", max_bytes=_MAX_BYTES)
