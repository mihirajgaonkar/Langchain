"""LangChain tool wrapper for weather forecasts."""

from __future__ import annotations

from langchain_core.tools import tool

from travel_weather_agent.services.open_meteo import (
    WeatherAPIError,
    geocode_city,
    get_forecast,
)


@tool
def get_weather(city: str, forecast_days: int = 5) -> dict:
    """Get a compact weather forecast for a city.

    Use this tool when a user asks about weather, trip timing, or whether a city
    is pleasant to visit. Input is a city name and optional number of forecast
    days. The tool geocodes the city, fetches Open-Meteo daily weather, and
    returns only compact forecast fields suitable for an LLM.
    """
    try:
        location = geocode_city(city)
        forecast = get_forecast(location, forecast_days=forecast_days)
    except WeatherAPIError as exc:
        return {"error": str(exc)}

    return {
        "city": forecast.location.name,
        "country": forecast.location.country,
        "timezone": forecast.location.timezone,
        "forecast": [day.model_dump() for day in forecast.days],
    }

