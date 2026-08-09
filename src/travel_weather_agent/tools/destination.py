"""Destination activity data and LangChain tool wrapper."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

from langchain_core.tools import tool

from travel_weather_agent.schemas.responses import DestinationActivities


SEVERE_WEATHER_CODES = {
    61,
    63,
    65,
    66,
    67,
    71,
    73,
    75,
    77,
    80,
    81,
    82,
    85,
    86,
    95,
    96,
    99,
}


def find_destination_activities(city: str) -> DestinationActivities | None:
    """Return local activity data for a city, if available."""
    destinations = _load_destinations()
    match = _find_city_key(city, destinations)
    if match is None:
        return None

    data = destinations[match]
    return DestinationActivities(
        city=match,
        indoor=list(data["indoor"]),
        outdoor=list(data["outdoor"]),
    )


@tool
def get_destination_activities(city: str, activity_type: str = "all") -> dict:
    """Get local indoor and/or outdoor activity ideas for a supported city.

    Use this tool when a user asks what to do in a destination. Input is a city
    name and activity_type of "indoor", "outdoor", or "all". It reads a small
    local dataset rather than calling a paid places API.
    """
    activities = find_destination_activities(city)
    if activities is None:
        return {
            "city": city,
            "error": "No local activity data is available for this city.",
        }

    normalized_type = activity_type.strip().lower()
    if normalized_type == "indoor":
        return {"city": activities.city, "indoor": activities.indoor}
    if normalized_type == "outdoor":
        return {"city": activities.city, "outdoor": activities.outdoor}
    if normalized_type == "all":
        return activities.model_dump()

    return {
        "city": activities.city,
        "error": "activity_type must be 'indoor', 'outdoor', or 'all'.",
    }


def recommend_activity_type(
    precipitation_probability: float,
    weather_code: int | None,
) -> str:
    """Choose indoor or outdoor activities using deterministic weather rules."""
    if weather_code in SEVERE_WEATHER_CODES:
        return "indoor"
    if precipitation_probability >= 50:
        return "indoor"
    return "outdoor"


@lru_cache(maxsize=1)
def _load_destinations() -> dict:
    """Load the bundled demo destination dataset."""
    data_file = resources.files("travel_weather_agent.data").joinpath(
        "destinations.json"
    )
    return json.loads(data_file.read_text(encoding="utf-8"))


def _find_city_key(city: str, destinations: dict) -> str | None:
    """Find a dataset key with case-insensitive city matching."""
    normalized_city = city.strip().lower()
    for destination in destinations:
        if destination.lower() == normalized_city:
            return destination
    return None

