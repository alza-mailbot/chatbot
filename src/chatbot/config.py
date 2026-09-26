"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration read from the environment and an optional .env file.

    Attributes:
        log_level: Logging level name for the application logger.
        port: TCP port the HTTP server listens on.
    """

    model_config = SettingsConfigDict(env_file=".env")

    log_level: str = "INFO"
    port: int = 8080


@lru_cache
def get_settings() -> Settings:
    """Return the application settings, created once and cached.

    Returns:
        Settings: The shared settings instance.
    """
    return Settings()
