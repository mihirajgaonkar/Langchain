"""Open-Meteo API client functions.

This layer owns HTTP communication and payload cleanup. LangChain tools wrap
these functions later, but the API client itself has no LangChain dependency.
"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import ValidationError

from travel_weather_agent.config.settings import get_settings
from travel_weather_agent.schemas.weather import (
    DailyWeather,
    LocationInfo,
    WeatherForecast,
)


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherAPIError(RuntimeError):
    """Raised when Open-Meteo data cannot be fetched or parsed cleanly."""


def geocode_city(city: str) -> LocationInfo:
    """Find the best Open-Meteo location match for a city name."""
    normalized_city = city.strip()
    if not normalized_city:
        raise WeatherAPIError("City name is required.")

    data = _get_json(
        GEOCODING_URL,
        params={
            "name": normalized_city,
            "count": 1,
            "language": "en",
            "format": "json",
        },
        error_context="Geocoding request failed",
    )

    results = data.get("results")
    if not isinstance(results, list) or not results:
        raise WeatherAPIError(f"No location found for '{normalized_city}'.")

    best_match = results[0]
    if not isinstance(best_match, dict):
        raise WeatherAPIError("Geocoding response had an invalid result.")

    try:
        return LocationInfo(
            name=best_match["name"],
            country=best_match.get("country"),
            latitude=best_match["latitude"],
            longitude=best_match["longitude"],
            timezone=best_match.get("timezone"),
        )
    except (KeyError, TypeError, ValidationError) as exc:
        raise WeatherAPIError("Geocoding response was missing required fields.") from exc


def get_forecast(location: LocationInfo, forecast_days: int = 5) -> WeatherForecast:
    """Fetch and normalize a daily weather forecast for a location."""
    days = max(1, min(forecast_days, 16))
    data = _get_json(
        FORECAST_URL,
        params={
            "latitude": location.latitude,
            "longitude": location.longitude,
            "daily": ",".join(
                [
                    "weather_code",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_probability_max",
                    "wind_speed_10m_max",
                ]
            ),
            "temperature_unit": get_settings().temperature_unit,
            "wind_speed_unit": "mph",
            "timezone": location.timezone or "auto",
            "forecast_days": days,
        },
        error_context="Forecast request failed",
    )

    daily = data.get("daily")
    if not isinstance(daily, dict):
        raise WeatherAPIError("Forecast response did not include daily weather data.")

    try:
        forecast_days_data = _parse_daily_weather(daily)
    except (KeyError, IndexError, TypeError, ValidationError, WeatherAPIError) as exc:
        raise WeatherAPIError("Forecast response had an invalid daily payload.") from exc

    return WeatherForecast(location=location, days=forecast_days_data)


def weather_code_to_condition(weather_code: int | None) -> str:
    """Convert Open-Meteo weather codes into readable conditions."""
    if weather_code is None:
        return "Unknown"
    if weather_code == 0:
        return "Clear sky"
    if weather_code in {1, 2}:
        return "Mainly clear"
    if weather_code == 3:
        return "Overcast"
    if weather_code in {45, 48}:
        return "Fog"
    if weather_code in {51, 53, 55, 56, 57}:
        return "Drizzle"
    if weather_code in {61, 63, 65, 66, 67, 80, 81, 82}:
        return "Rain"
    if weather_code in {71, 73, 75, 77, 85, 86}:
        return "Snow"
    if weather_code in {95, 96, 99}:
        return "Thunderstorm"
    return "Unknown"


def _get_json(
    url: str, *, params: dict[str, Any], error_context: str
) -> dict[str, Any]:
    """Call an HTTP JSON endpoint and normalize common API errors."""
    timeout = get_settings().http_timeout_seconds
    try:
        response = httpx.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except httpx.TimeoutException as exc:
        raise WeatherAPIError(f"{error_context}: request timed out.") from exc
    except httpx.HTTPStatusError as exc:
        raise WeatherAPIError(
            f"{error_context}: service returned HTTP {exc.response.status_code}."
        ) from exc
    except httpx.HTTPError as exc:
        raise WeatherAPIError(f"{error_context}: network error.") from exc
    except ValueError as exc:
        raise WeatherAPIError(f"{error_context}: invalid JSON response.") from exc

    if not isinstance(data, dict):
        raise WeatherAPIError(f"{error_context}: invalid API response.")
    return data


def _parse_daily_weather(daily: dict[str, Any]) -> list[DailyWeather]:
    """Transform Open-Meteo daily arrays into compact Pydantic objects."""
    dates = daily["time"]
    max_temps = daily["temperature_2m_max"]
    min_temps = daily["temperature_2m_min"]
    precip = daily.get("precipitation_probability_max", [])
    wind = daily.get("wind_speed_10m_max", [])
    codes = daily.get("weather_code", [])

    if not isinstance(dates, list) or not dates:
        raise WeatherAPIError("Forecast response did not include forecast dates.")

    days: list[DailyWeather] = []
    for index, date in enumerate(dates):
        weather_code = _optional_list_value(codes, index)
        days.append(
            DailyWeather(
                date=date,
                temperature_max=max_temps[index],
                temperature_min=min_temps[index],
                precipitation_probability=_optional_list_value(precip, index),
                wind_speed_max=_optional_list_value(wind, index),
                weather_code=weather_code,
                condition=weather_code_to_condition(weather_code),
            )
        )

    return days


def _optional_list_value(values: Any, index: int) -> Any:
    """Return a list value when present; otherwise return None."""
    if isinstance(values, list) and index < len(values):
        return values[index]
    return None
