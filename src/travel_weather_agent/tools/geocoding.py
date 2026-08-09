"""LangChain tool wrapper for geocoding."""

from __future__ import annotations

from langchain_core.tools import tool

from travel_weather_agent.services.open_meteo import WeatherAPIError, geocode_city


@tool
def geocode_location(city: str) -> dict:
    """Find latitude, longitude, country, and timezone for a city.

    Use this tool whenever you need coordinates or timezone information for a
    city before requesting weather data. Input should be a city name such as
    "Boston" or "Washington DC". Returns a compact JSON-serializable dict.
    """
    try:
        location = geocode_city(city)
    except WeatherAPIError as exc:
        return {"error": str(exc)}

    return location.model_dump()

