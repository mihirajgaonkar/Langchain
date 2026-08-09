"""Structured recommendation demo for Phase 4."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from travel_weather_agent.llm.model import get_llm
from travel_weather_agent.schemas.responses import (
    DestinationActivities,
    TravelRecommendation,
)
from travel_weather_agent.schemas.weather import WeatherForecast
from travel_weather_agent.tools.destination import recommend_activity_type


def generate_structured_recommendation(
    city: str,
    weather: WeatherForecast,
    activities: DestinationActivities,
) -> TravelRecommendation:
    """Generate a Pydantic travel recommendation with LangChain structured output.

    This is a narrow demonstration of `with_structured_output`. The weather and
    activities are prepared by deterministic code first; the LLM only writes the
    final recommendation in the requested schema.
    """
    first_day = weather.days[0]
    activity_type = recommend_activity_type(
        precipitation_probability=first_day.precipitation_probability or 0,
        weather_code=first_day.weather_code,
    )
    candidate_activities = (
        activities.indoor if activity_type == "indoor" else activities.outdoor
    )

    llm = get_llm()
    structured_llm = llm.with_structured_output(TravelRecommendation)
    result = structured_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a concise travel assistant. Return a practical "
                    "weather-aware recommendation."
                )
            ),
            HumanMessage(
                content=(
                    f"City: {city}\n"
                    f"Forecast date: {first_day.date}\n"
                    f"High: {first_day.temperature_max} F\n"
                    f"Low: {first_day.temperature_min} F\n"
                    f"Rain probability: {first_day.precipitation_probability}%\n"
                    f"Wind: {first_day.wind_speed_max} mph\n"
                    f"Condition: {first_day.condition}\n"
                    f"Recommended activity type from rules: {activity_type}\n"
                    f"Candidate activities: {candidate_activities}"
                )
            ),
        ]
    )

    if isinstance(result, TravelRecommendation):
        return result
    if isinstance(result, dict):
        return TravelRecommendation.model_validate(result)

    raise TypeError("Structured recommendation did not return the expected schema.")

