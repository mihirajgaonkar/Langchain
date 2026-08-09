"""Tests for the Open-Meteo service layer."""

from __future__ import annotations

import httpx
import pytest

from travel_weather_agent.schemas.weather import LocationInfo
from travel_weather_agent.services import open_meteo
from travel_weather_agent.services.open_meteo import WeatherAPIError


class MockHTTPResponse:
    """Small httpx response double used by Open-Meteo tests."""

    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://api.open-meteo.test")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError(
                "mock HTTP failure",
                request=request,
                response=response,
            )


def test_geocoding_successful_result(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(*args, **kwargs) -> MockHTTPResponse:
        return MockHTTPResponse(
            {
                "results": [
                    {
                        "name": "Boston",
                        "country": "United States",
                        "latitude": 42.3584,
                        "longitude": -71.0598,
                        "timezone": "America/New_York",
                    }
                ]
            }
        )

    monkeypatch.setattr(open_meteo.httpx, "get", fake_get)

    location = open_meteo.geocode_city("Boston")

    assert location.name == "Boston"
    assert location.country == "United States"
    assert location.latitude == 42.3584
    assert location.longitude == -71.0598
    assert location.timezone == "America/New_York"


def test_geocoding_empty_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        open_meteo.httpx,
        "get",
        lambda *args, **kwargs: MockHTTPResponse({"results": []}),
    )

    with pytest.raises(WeatherAPIError, match="No location found"):
        open_meteo.geocode_city("Notacity")


def test_geocoding_http_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        open_meteo.httpx,
        "get",
        lambda *args, **kwargs: MockHTTPResponse({"error": True}, status_code=500),
    )

    with pytest.raises(WeatherAPIError, match="HTTP 500"):
        open_meteo.geocode_city("Boston")


def test_forecast_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        open_meteo.httpx,
        "get",
        lambda *args, **kwargs: MockHTTPResponse(
            {
                "daily": {
                    "time": ["2026-08-09"],
                    "weather_code": [2],
                    "temperature_2m_max": [82.0],
                    "temperature_2m_min": [67.0],
                    "precipitation_probability_max": [20],
                    "wind_speed_10m_max": [13.0],
                }
            }
        ),
    )
    location = LocationInfo(
        name="Boston",
        country="United States",
        latitude=42.3584,
        longitude=-71.0598,
        timezone="America/New_York",
    )

    forecast = open_meteo.get_forecast(location, forecast_days=1)

    assert forecast.location.name == "Boston"
    assert len(forecast.days) == 1
    assert forecast.days[0].temperature_max == 82.0
    assert forecast.days[0].condition == "Mainly clear"


def test_weather_code_conversion() -> None:
    assert open_meteo.weather_code_to_condition(0) == "Clear sky"
    assert open_meteo.weather_code_to_condition(61) == "Rain"
    assert open_meteo.weather_code_to_condition(71) == "Snow"
    assert open_meteo.weather_code_to_condition(95) == "Thunderstorm"
    assert open_meteo.weather_code_to_condition(None) == "Unknown"


def test_invalid_api_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        open_meteo.httpx,
        "get",
        lambda *args, **kwargs: MockHTTPResponse({"daily": {"time": []}}),
    )
    location = LocationInfo(name="Boston", latitude=42.3584, longitude=-71.0598)

    with pytest.raises(WeatherAPIError, match="invalid daily payload"):
        open_meteo.get_forecast(location, forecast_days=1)

