"""Tests for Phase 4 deterministic tools and structured output."""

from __future__ import annotations

from travel_weather_agent.schemas.responses import DestinationActivities, TravelRecommendation
from travel_weather_agent.schemas.weather import DailyWeather, LocationInfo, WeatherForecast
from travel_weather_agent.services import recommendations
from travel_weather_agent.tools.destination import (
    find_destination_activities,
    get_destination_activities,
    recommend_activity_type,
)
from travel_weather_agent.tools.scoring import calculate_trip_score_value


def test_trip_score_ideal_weather() -> None:
    score = calculate_trip_score_value(
        temperature_max=72,
        precipitation_probability=5,
        wind_speed=8,
    )

    assert score.score >= 95
    assert score.temperature_score == 100
    assert score.wind_score == 100


def test_trip_score_bad_weather() -> None:
    score = calculate_trip_score_value(
        temperature_max=100,
        precipitation_probability=90,
        wind_speed=35,
    )

    assert score.score < 40
    assert "drawback" in score.explanation or "tradeoffs" in score.explanation


def test_score_stays_between_0_and_100() -> None:
    for temperature in [-20, 72, 140]:
        score = calculate_trip_score_value(
            temperature_max=temperature,
            precipitation_probability=150,
            wind_speed=100,
        )
        assert 0 <= score.score <= 100
        assert 0 <= score.temperature_score <= 100
        assert 0 <= score.precipitation_score <= 100
        assert 0 <= score.wind_score <= 100


def test_destination_indoor_activities() -> None:
    result = get_destination_activities.invoke(
        {"city": "Boston", "activity_type": "indoor"}
    )

    assert result["city"] == "Boston"
    assert "Museum of Fine Arts" in result["indoor"]
    assert "outdoor" not in result


def test_destination_outdoor_activities() -> None:
    result = get_destination_activities.invoke(
        {"city": "Boston", "activity_type": "outdoor"}
    )

    assert result["city"] == "Boston"
    assert "Freedom Trail" in result["outdoor"]
    assert "indoor" not in result


def test_unknown_destination_handling() -> None:
    result = get_destination_activities.invoke(
        {"city": "Atlantis", "activity_type": "all"}
    )

    assert result["city"] == "Atlantis"
    assert "error" in result


def test_find_destination_activities_case_insensitive() -> None:
    activities = find_destination_activities("boston")

    assert activities is not None
    assert activities.city == "Boston"


def test_weather_aware_activity_selection() -> None:
    assert recommend_activity_type(20, 2) == "outdoor"
    assert recommend_activity_type(75, 2) == "indoor"
    assert recommend_activity_type(20, 95) == "indoor"


class FakeStructuredModel:
    def invoke(self, messages):
        return TravelRecommendation(
            city="Boston",
            weather_summary="Mild and mostly clear.",
            recommended_activity_type="outdoor",
            recommended_activities=["Freedom Trail"],
            reasoning="The forecast is pleasant with low rain risk.",
        )


class FakeLLM:
    def with_structured_output(self, schema):
        assert schema is TravelRecommendation
        return FakeStructuredModel()


def test_structured_recommendation_response_parsing(monkeypatch) -> None:
    monkeypatch.setattr(recommendations, "get_llm", lambda: FakeLLM())
    weather = WeatherForecast(
        location=LocationInfo(name="Boston", latitude=42.3584, longitude=-71.0598),
        days=[
            DailyWeather(
                date="2026-08-09",
                temperature_max=72,
                temperature_min=62,
                precipitation_probability=10,
                wind_speed_max=8,
                weather_code=2,
                condition="Mainly clear",
            )
        ],
    )
    activities = DestinationActivities(
        city="Boston",
        indoor=["Museum of Fine Arts"],
        outdoor=["Freedom Trail"],
    )

    recommendation = recommendations.generate_structured_recommendation(
        "Boston",
        weather,
        activities,
    )

    assert recommendation.city == "Boston"
    assert recommendation.recommended_activity_type == "outdoor"
    assert recommendation.recommended_activities == ["Freedom Trail"]

