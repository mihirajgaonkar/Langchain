"""Deterministic trip scoring logic and LangChain tool wrapper."""

from __future__ import annotations

from langchain_core.tools import tool

from travel_weather_agent.schemas.responses import TripScore


def calculate_trip_score_value(
    temperature_max: float,
    precipitation_probability: float,
    wind_speed: float,
) -> TripScore:
    """Score travel weather from 0 to 100 using deterministic Python logic."""
    temperature_score = _temperature_component(temperature_max)
    precipitation_score = _clamp(100 - precipitation_probability)
    wind_score = _wind_component(wind_speed)

    score = round(
        (temperature_score * 0.50)
        + (precipitation_score * 0.35)
        + (wind_score * 0.15),
        1,
    )

    explanation = _score_explanation(
        temperature_score=temperature_score,
        precipitation_score=precipitation_score,
        wind_score=wind_score,
    )

    return TripScore(
        score=score,
        temperature_score=round(temperature_score, 1),
        precipitation_score=round(precipitation_score, 1),
        wind_score=round(wind_score, 1),
        explanation=explanation,
    )


@tool
def calculate_trip_score(
    temperature_max: float,
    precipitation_probability: float,
    wind_speed: float,
) -> dict:
    """Calculate a deterministic travel-weather score.

    Use this tool when a user asks whether weather is good for a trip or when
    comparing destinations. The LLM decides that a score is useful, but Python
    performs the arithmetic. Inputs are high temperature in Fahrenheit, rain
    probability from 0 to 100, and wind speed in mph.
    """
    return calculate_trip_score_value(
        temperature_max=temperature_max,
        precipitation_probability=precipitation_probability,
        wind_speed=wind_speed,
    ).model_dump()


def _temperature_component(temperature_max: float) -> float:
    """Return 100 for ideal 65-80 F highs, penalizing distance outside it."""
    if 65 <= temperature_max <= 80:
        return 100.0
    if temperature_max < 65:
        return _clamp(100 - ((65 - temperature_max) * 3))
    return _clamp(100 - ((temperature_max - 80) * 3))


def _wind_component(wind_speed: float) -> float:
    """Return a wind comfort score, with stronger penalties above 15 mph."""
    if wind_speed <= 10:
        return 100.0
    if wind_speed <= 25:
        return _clamp(100 - ((wind_speed - 10) * 4))
    return _clamp(40 - ((wind_speed - 25) * 2))


def _score_explanation(
    *, temperature_score: float, precipitation_score: float, wind_score: float
) -> str:
    """Create a compact explanation for the component score mix."""
    if min(temperature_score, precipitation_score, wind_score) >= 75:
        return "Comfortable temperature with low rain risk and manageable wind."
    if precipitation_score < 50:
        return "Rain risk is the main drawback for travel comfort."
    if temperature_score < 50:
        return "Temperature is far from the ideal travel range."
    if wind_score < 50:
        return "Wind may make outdoor plans less comfortable."
    return "Mixed conditions with some tradeoffs for travel comfort."


def _clamp(value: float, minimum: float = 0, maximum: float = 100) -> float:
    """Clamp a score to a fixed range."""
    return max(minimum, min(maximum, value))

