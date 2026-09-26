"""Sanity check that the package is importable and the test tooling runs."""

import chatbot


def test_package_is_importable() -> None:
    """Verify the chatbot package imports and exposes a docstring."""
    assert chatbot.__doc__
