"""Validation of incoming email attachments."""

from dataclasses import dataclass

SUPPORTED_MIME_TYPES = frozenset(
    {
        "application/pdf",
        "image/jpeg",
        "image/png",
        "audio/mpeg",
        "audio/wav",
    }
)

_MIME_ALIASES = {"audio/x-wav": "audio/wav"}


class AttachmentError(Exception):
    """Raised when an attachment cannot be accepted."""


class UnsupportedAttachmentError(AttachmentError):
    """Raised for attachments of a type the service cannot process."""


class AttachmentTooLargeError(AttachmentError):
    """Raised for attachments exceeding the configured size limit."""


@dataclass(frozen=True)
class Attachment:
    """Validated attachment content.

    Attributes:
        filename: Original file name of the attachment.
        mime_type: Normalized mime type, guaranteed to be supported.
        data: Raw attachment bytes.
    """

    filename: str
    mime_type: str
    data: bytes


def validate_attachment(
    filename: str, mime_type: str, data: bytes, *, max_bytes: int
) -> Attachment:
    """Validate one attachment and normalize its mime type.

    Args:
        filename: Original file name, used in error messages.
        mime_type: Declared mime type of the file.
        data: Raw attachment bytes.
        max_bytes: Maximum accepted size in bytes.

    Returns:
        Attachment: The validated, normalized attachment.

    Raises:
        UnsupportedAttachmentError: If the mime type is not supported.
        AttachmentTooLargeError: If the data exceeds ``max_bytes``.
        AttachmentError: If the attachment is empty.
    """
    mime = _MIME_ALIASES.get(mime_type, mime_type)
    if mime not in SUPPORTED_MIME_TYPES:
        supported = ", ".join(sorted(SUPPORTED_MIME_TYPES))
        raise UnsupportedAttachmentError(
            f"Attachment {filename!r} has unsupported type {mime_type!r}; "
            f"supported types: {supported}"
        )
    if not data:
        raise AttachmentError(f"Attachment {filename!r} is empty")
    if len(data) > max_bytes:
        raise AttachmentTooLargeError(
            f"Attachment {filename!r} exceeds the size limit of {max_bytes} bytes"
        )
    return Attachment(filename=filename, mime_type=mime, data=data)
