"""LangChain chat model abstraction for Phase 3.

The central lesson in this phase is that application code can do this:

    llm = get_llm()
    response = llm.invoke(messages)

without knowing whether the backend is OpenAI, Groq, or Ollama. LangChain gives
us a common chat model interface and standard message objects.
"""

from __future__ import annotations

from typing import Any

from travel_weather_agent.config.settings import Settings, get_settings


SYSTEM_INSTRUCTION = "You are a concise travel assistant."


class LLMConfigurationError(RuntimeError):
    """Raised when model configuration is missing or invalid."""


class MissingAPIKeyError(LLMConfigurationError):
    """Raised when a selected remote provider has no API key configured."""


class UnsupportedProviderError(LLMConfigurationError):
    """Raised when LLM_PROVIDER is not one of the supported providers."""


def get_llm(settings: Settings | None = None) -> Any:
    """Return the configured LangChain chat model.

    The calling code receives one object with the standard LangChain chat model
    interface. Provider-specific details stay inside this factory.
    """
    resolved_settings = settings or get_settings()
    provider = resolved_settings.llm_provider.strip().lower()

    if provider == "openai":
        _require_api_key(
            resolved_settings.openai_api_key,
            "OPENAI_API_KEY is required when LLM_PROVIDER=openai.",
        )
        ChatOpenAI = _load_chat_openai()
        return ChatOpenAI(
            model=resolved_settings.openai_model,
            api_key=resolved_settings.openai_api_key,
            temperature=0,
            timeout=resolved_settings.http_timeout_seconds,
            max_retries=resolved_settings.http_max_retries,
        )

    if provider == "groq":
        _require_api_key(
            resolved_settings.groq_api_key,
            "GROQ_API_KEY is required when LLM_PROVIDER=groq.",
        )
        ChatGroq = _load_chat_groq()
        return ChatGroq(
            model=resolved_settings.groq_model,
            groq_api_key=resolved_settings.groq_api_key,
            temperature=0,
            request_timeout=resolved_settings.http_timeout_seconds,
            max_retries=resolved_settings.http_max_retries,
        )

    if provider == "ollama":
        ChatOllama = _load_chat_ollama()
        return ChatOllama(
            model=resolved_settings.ollama_model,
            base_url=resolved_settings.ollama_base_url,
            temperature=0,
        )

    raise UnsupportedProviderError(
        f"Unsupported LLM_PROVIDER '{provider}'. Use 'openai', 'groq', or 'ollama'."
    )


def ask_with_langchain(prompt: str) -> str:
    """Ask the configured LangChain chat model and return plain text.

    This intentionally uses message objects directly. Prompt templates, tools,
    agents, and structured output come later.
    """
    SystemMessage, HumanMessage = _load_message_classes()
    messages = [
        SystemMessage(content=SYSTEM_INSTRUCTION),
        HumanMessage(content=prompt),
    ]

    llm = get_llm()
    response = llm.invoke(messages)
    content = getattr(response, "content", None)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        return _stringify_content_blocks(content)

    raise LLMConfigurationError("LangChain response did not include text content.")


def _require_api_key(value: str | None, message: str) -> None:
    """Validate a provider API key without logging or exposing the key."""
    if not value or not value.strip():
        raise MissingAPIKeyError(message)


def _stringify_content_blocks(content: list[Any]) -> str:
    """Convert LangChain content blocks into plain text for the CLI demo."""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])

    text = "\n".join(part.strip() for part in parts if part.strip())
    if not text:
        raise LLMConfigurationError("LangChain response content blocks had no text.")
    return text


def _load_chat_openai() -> Any:
    """Import ChatOpenAI lazily so tests can mock provider classes."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI


def _load_chat_groq() -> Any:
    """Import ChatGroq lazily so the dependency is only needed for Groq."""
    from langchain_groq import ChatGroq

    return ChatGroq


def _load_chat_ollama() -> Any:
    """Import ChatOllama lazily so local-model support remains optional at runtime."""
    from langchain_ollama import ChatOllama

    return ChatOllama


def _load_message_classes() -> tuple[Any, Any]:
    """Load LangChain's standard chat message classes."""
    from langchain_core.messages import HumanMessage, SystemMessage

    return SystemMessage, HumanMessage

