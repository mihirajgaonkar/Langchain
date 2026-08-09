"""Optional LangSmith observability helpers."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from travel_weather_agent.config.settings import Settings, get_settings


def is_langsmith_enabled(settings: Settings | None = None) -> bool:
    """Return whether LangSmith tracing should be enabled for this run."""
    resolved = settings or get_settings()
    return (
        str(resolved.langsmith_tracing).strip().lower() == "true"
        and bool(resolved.langsmith_api_key)
    )


def trace_metadata(settings: Settings | None = None, *, thread_id: str) -> dict[str, Any]:
    """Build non-secret trace metadata for LangChain/LangGraph runs."""
    resolved = settings or get_settings()
    provider = resolved.llm_provider.strip().lower()
    return {
        "application": "travel-weather-agent",
        "environment": "local",
        "thread_id": thread_id,
        "provider": provider,
        "model": _model_name_for_provider(resolved, provider),
    }


def langsmith_tracing_context(settings: Settings | None = None):
    """Return a LangSmith tracing context when available, otherwise a no-op.

    LangChain/LangGraph natively honor the LangSmith environment variables. This
    helper only makes selective tracing explicit while keeping LangSmith fully
    optional.
    """
    resolved = settings or get_settings()
    if not is_langsmith_enabled(resolved):
        return nullcontext()

    try:
        import langsmith as ls
    except ImportError:
        return nullcontext()

    return ls.tracing_context(
        enabled=True,
        project_name=resolved.langsmith_project,
    )


def _model_name_for_provider(settings: Settings, provider: str) -> str:
    """Return the configured model name without exposing secrets."""
    if provider == "openai":
        return settings.openai_model
    if provider == "groq":
        return settings.groq_model
    if provider == "ollama":
        return settings.ollama_model
    return "unknown"

