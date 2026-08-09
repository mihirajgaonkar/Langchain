"""Pydantic schemas for travel recommendations and tool results."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DestinationActivities(BaseModel):
    """Indoor and outdoor activity options for a city."""

    city: str
    indoor: list[str]
    outdoor: list[str]


class TripScore(BaseModel):
    """Deterministic travel-weather score with component breakdown."""

    score: float = Field(ge=0, le=100)
    temperature_score: float = Field(ge=0, le=100)
    precipitation_score: float = Field(ge=0, le=100)
    wind_score: float = Field(ge=0, le=100)
    explanation: str


class TravelRecommendation(BaseModel):
    """Structured LLM recommendation for a simple weather-aware city visit."""

    city: str
    weather_summary: str
    recommended_activity_type: Literal["indoor", "outdoor", "mixed"]
    recommended_activities: list[str]
    reasoning: str

