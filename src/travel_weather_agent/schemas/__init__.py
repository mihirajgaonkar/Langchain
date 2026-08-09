"""Pydantic schemas for structured application data."""

from travel_weather_agent.schemas.responses import (
    DestinationActivities,
    TravelRecommendation,
    TripScore,
)
from travel_weather_agent.schemas.weather import DailyWeather, LocationInfo, WeatherForecast

__all__ = [
    "DailyWeather",
    "DestinationActivities",
    "LocationInfo",
    "TravelRecommendation",
    "TripScore",
    "WeatherForecast",
]
