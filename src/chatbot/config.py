"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful email assistant for Alza customer support. "
    "Reply to the customer's email using the email body and any attachments. "
    "You natively understand attached PDF documents, images, and audio "
    "recordings: listen to or look at them and use their content; never "
    "claim you cannot process an attachment. "
    "Answer in the language of the customer's email text, never in a "
    "language taken from an attachment. Address the customer by name only "
    "when the email itself gives one; never invent a name. "
    "Be polite and concise, and write plain text suitable for an email "
    "body (no markdown). Return only the reply body text; never include "
    "a Subject line, email headers, or the recipient's address."
)


class Settings(BaseSettings):
    """Runtime configuration read from the environment and an optional .env file.

    Attributes:
        log_level: Logging level name for the application logger.
        port: TCP port the HTTP server listens on.
        gcp_project_id: GCP project used for Vertex AI calls.
        gcp_location: Vertex AI endpoint location.
        gemini_model: Gemini model name used for reply generation.
        system_prompt: System instruction defining the assistant persona.
        max_attachment_bytes: Maximum accepted size of a single attachment.
        web_search_enabled: Offer the web search tools to the model.
        brave_api_key: Brave Search API subscription token.
        brave_max_results: Maximum search results returned to the model.
        agent_max_iterations: Tool-call rounds before an answer is forced.
        agent_deadline_seconds: Wall-clock budget for the whole agent loop.
    """

    model_config = SettingsConfigDict(env_file=".env")

    log_level: str = "INFO"
    port: int = 8080
    gcp_project_id: str
    gcp_location: str = "global"
    gemini_model: str = "gemini-2.5-flash"
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    max_attachment_bytes: int = 15 * 1024 * 1024
    web_search_enabled: bool = False
    brave_api_key: str | None = None
    brave_max_results: int = 5
    agent_max_iterations: int = 6
    agent_deadline_seconds: float = 90

    @model_validator(mode="after")
    def _require_key_when_enabled(self) -> Settings:
        """Reject web search without a key at startup, not at request time."""
        if self.web_search_enabled and not self.brave_api_key:
            raise ValueError("WEB_SEARCH_ENABLED requires BRAVE_API_KEY to be set")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the application settings, created once and cached.

    Returns:
        Settings: The shared settings instance.
    """
    return Settings()
