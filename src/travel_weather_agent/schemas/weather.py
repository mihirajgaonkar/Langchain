"""Pydantic schemas for weather and location data."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LocationInfo(BaseModel):
    """Compact geocoding result used throughout the application."""

    name: str
    country: str | None = None
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str | None = None


class DailyWeather(BaseModel):
    """One day of normalized Open-Meteo forecast data."""

    date: str
    temperature_max: float
    temperature_min: float
    precipitation_probability: float | None = Field(default=None, ge=0, le=100)
    wind_speed_max: float | None = Field(default=None, ge=0)
    weather_code: int | None = None
    condition: str | None = None


class WeatherForecast(BaseModel):
    """Weather forecast for a geocoded location."""

    location: LocationInfo
    days: list[DailyWeather]

