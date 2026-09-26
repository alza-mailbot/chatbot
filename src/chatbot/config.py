"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful email assistant for Alza customer support. "
    "Reply to the customer's email using the email body and any attachments. "
    "Answer in the language of the incoming email, be polite and concise, "
    "and write plain text suitable for an email body (no markdown)."
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
    """

    model_config = SettingsConfigDict(env_file=".env")

    log_level: str = "INFO"
    port: int = 8080
    gcp_project_id: str
    gcp_location: str = "global"
    gemini_model: str = "gemini-2.5-flash"
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT


@lru_cache
def get_settings() -> Settings:
    """Return the application settings, created once and cached.

    Returns:
        Settings: The shared settings instance.
    """
    return Settings()
