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
    "You send exactly one reply and have no way to follow up later, so "
    "never promise to check something and get back to the customer. "
    "When the question involves current prices, availability, news, or a "
    "product you do not recognize, and a web search tool is offered, you "
    "must call it before answering and use its results in this reply; "
    "never claim a product does not exist or is not yet released based "
    "on memory alone. "
    "Answer in the language of the customer's email text, never in a "
    "language taken from an attachment. Address the customer by name only "
    "when the email itself gives one; never invent a name. "
    "Be polite and concise, and write plain text suitable for an email "
    "body (no markdown). Return only the reply body text; never include "
    "a Subject line, email headers, or the recipient's address."
)

# appended only when web search is enabled: the model must not be told
# about a tool it is not offered
_WEB_TOOL_RULES = (
    " TOOL RULES: You have a function tool named web_search. When the "
    "email asks about a current price, availability, news, or a product "
    "or model you do not recognize, the only correct first response is a "
    "web_search function call — never a text reply. Use a short generic "
    "query in the language of the customer's email (for example 'Samsung "
    "Galaxy S25 cena'). Never answer such questions from memory, never "
    "claim the product is unavailable or unknown without searching, and "
    "never promise to check later."
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
    agent_max_iterations: int = 4
    agent_deadline_seconds: float = 90

    @model_validator(mode="after")
    def _require_key_when_enabled(self) -> Settings:
        """Reject web search without a key at startup, not at request time."""
        if self.web_search_enabled and not self.brave_api_key:
            raise ValueError("WEB_SEARCH_ENABLED requires BRAVE_API_KEY to be set")
        return self

    @model_validator(mode="after")
    def _append_tool_rules_when_enabled(self) -> Settings:
        """Tell the model about web_search only when the tool is offered."""
        if self.web_search_enabled and _WEB_TOOL_RULES not in self.system_prompt:
            self.system_prompt += _WEB_TOOL_RULES
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the application settings, created once and cached.

    Returns:
        Settings: The shared settings instance.
    """
    return Settings()
