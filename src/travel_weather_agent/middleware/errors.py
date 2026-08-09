"""Error normalization for user-facing agent responses."""

from __future__ import annotations

from travel_weather_agent.llm.model import LLMConfigurationError
from travel_weather_agent.middleware.validation import InputValidationError
from travel_weather_agent.services.open_meteo import WeatherAPIError

try:
    from langgraph.errors import GraphRecursionError
except ImportError:  # pragma: no cover - compatibility with older LangGraph builds
    GraphRecursionError = RecursionError


def normalize_error(error: Exception) -> str:
    """Convert expected failures into concise user-facing messages."""
    if isinstance(error, InputValidationError):
        return str(error)
    if isinstance(error, WeatherAPIError):
        return "I couldn't retrieve weather data for that location."
    if isinstance(error, LLMConfigurationError):
        return "The model provider is not configured correctly."
    if isinstance(error, (RecursionError, GraphRecursionError)):
        return "I could not finish because the tool loop reached its safety limit."
    return "Something went wrong while handling that request."
