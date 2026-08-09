"""Command-line entry point for the Travel + Weather AI Agent."""

from __future__ import annotations

import argparse

from travel_weather_agent.agents.travel_agent import run_agent
from travel_weather_agent.graph.hitl import DEFAULT_CANDIDATE_CITIES, run_hitl_demo
from travel_weather_agent.graph.travel_graph import get_travel_graph_mermaid
from travel_weather_agent.llm.direct import DirectLLMError, call_direct_llm
from travel_weather_agent.llm.model import LLMConfigurationError, ask_with_langchain
from travel_weather_agent.services.open_meteo import (
    WeatherAPIError,
    geocode_city,
    get_forecast,
)
from travel_weather_agent.services.recommendations import (
    generate_structured_recommendation,
)
from travel_weather_agent.tools.destination import (
    find_destination_activities,
    get_destination_activities,
    recommend_activity_type,
)
from travel_weather_agent.tools.scoring import calculate_trip_score_value


def main() -> None:
    """Run the CLI application."""
    parser = argparse.ArgumentParser(
        description="Travel + Weather Agent learning project."
    )
    parser.add_argument(
        "--raw-llm-demo",
        action="store_true",
        help="Run the Phase 2 direct LLM API example without LangChain.",
    )
    parser.add_argument(
        "--langchain-demo",
        action="store_true",
        help="Run the Phase 3 LangChain chat model example.",
    )
    parser.add_argument(
        "--weather-demo",
        metavar="CITY",
        help="Run the Phase 4 Open-Meteo weather demo for a city.",
    )
    parser.add_argument(
        "--activities-demo",
        metavar="CITY",
        help="Run the Phase 4 local destination activities demo for a city.",
    )
    parser.add_argument(
        "--recommendation-demo",
        metavar="CITY",
        help="Run the Phase 4 structured weather recommendation demo for a city.",
    )
    parser.add_argument(
        "--agent-demo",
        action="store_true",
        help="Run the Phase 5 tool-calling agent demo.",
    )
    parser.add_argument(
        "--thread-id",
        default="default",
        help="Conversation memory thread id for the agent demo.",
    )
    parser.add_argument(
        "--show-tool-calls",
        action="store_true",
        help="Show concise tool-call debug lines during the agent demo.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show high-level execution trace output for learner demos.",
    )
    parser.add_argument(
        "--hitl-demo",
        action="store_true",
        help="Run the Phase 6 LangGraph human-in-the-loop demo.",
    )
    parser.add_argument(
        "--show-graph",
        action="store_true",
        help="Print a Mermaid diagram of the agent/tool graph.",
    )
    parser.add_argument(
        "--prompt",
        default="In one sentence, explain why weather matters when planning travel.",
        help="Prompt to send when using a demo flag.",
    )
    args = parser.parse_args()

    print("Travel + Weather Agent")

    if args.raw_llm_demo:
        try:
            response = call_direct_llm(args.prompt)
        except DirectLLMError as exc:
            print(f"Raw LLM demo failed: {exc}")
            return

        print(f"Provider: {response.provider}")
        print(f"Model: {response.model}")
        print()
        print(response.text)
        return

    if args.langchain_demo:
        try:
            response_text = ask_with_langchain(args.prompt)
        except LLMConfigurationError as exc:
            print(f"LangChain demo failed: {exc}")
            return

        print("LangChain demo")
        print()
        print(response_text)
        return

    if args.weather_demo:
        _run_weather_demo(args.weather_demo)
        return

    if args.activities_demo:
        _run_activities_demo(args.activities_demo)
        return

    if args.recommendation_demo:
        _run_recommendation_demo(args.recommendation_demo)
        return

    if args.agent_demo:
        _run_agent_demo(
            prompt=args.prompt,
            thread_id=args.thread_id,
            show_tool_calls=args.show_tool_calls,
            debug=args.debug,
        )
        return

    if args.hitl_demo:
        _run_hitl_demo()
        return

    if args.show_graph:
        print(get_travel_graph_mermaid())
        return

    print(
        "Final project is ready. Run --raw-llm-demo, --langchain-demo, "
        "--weather-demo CITY, --activities-demo CITY, or "
        "--recommendation-demo CITY, --agent-demo, --hitl-demo, or --show-graph."
    )


def _run_weather_demo(city: str) -> None:
    """Print a compact live Open-Meteo forecast for a city."""
    try:
        location = geocode_city(city)
        forecast = get_forecast(location, forecast_days=3)
    except WeatherAPIError as exc:
        print(f"Weather demo failed: {exc}")
        return

    print(f"Weather for {forecast.location.name}")
    print()
    for day in forecast.days:
        print(day.date)
        print(f"High: {day.temperature_max} F")
        print(f"Low: {day.temperature_min} F")
        print(f"Rain: {day.precipitation_probability}%")
        print(f"Wind: {day.wind_speed_max} mph")
        print(f"Condition: {day.condition}")
        print()


def _run_activities_demo(city: str) -> None:
    """Print local demo activities for a city."""
    result = get_destination_activities.invoke({"city": city, "activity_type": "all"})
    if "error" in result:
        print(f"Activities demo failed: {result['error']}")
        return

    print(f"Activities for {result['city']}")
    print()
    print("Indoor:")
    for activity in result["indoor"]:
        print(f"- {activity}")
    print()
    print("Outdoor:")
    for activity in result["outdoor"]:
        print(f"- {activity}")


def _run_recommendation_demo(city: str) -> None:
    """Run a deterministic pipeline and ask the LLM for structured output."""
    try:
        location = geocode_city(city)
        forecast = get_forecast(location, forecast_days=1)
    except WeatherAPIError as exc:
        print(f"Recommendation demo failed: {exc}")
        return

    activities = find_destination_activities(city)
    if activities is None:
        print("Recommendation demo failed: no local activity data for this city.")
        return

    first_day = forecast.days[0]
    activity_type = recommend_activity_type(
        precipitation_probability=first_day.precipitation_probability or 0,
        weather_code=first_day.weather_code,
    )
    score = calculate_trip_score_value(
        temperature_max=first_day.temperature_max,
        precipitation_probability=first_day.precipitation_probability or 0,
        wind_speed=first_day.wind_speed_max or 0,
    )

    try:
        recommendation = generate_structured_recommendation(
            city=activities.city,
            weather=forecast,
            activities=activities,
        )
    except (LLMConfigurationError, TypeError, ValueError) as exc:
        print(f"Recommendation demo failed: {exc}")
        return

    print(f"Structured recommendation for {recommendation.city}")
    print()
    print(f"Weather: {recommendation.weather_summary}")
    print(f"Activity type: {recommendation.recommended_activity_type}")
    print(f"Rule-selected activity type: {activity_type}")
    print(f"Trip score: {score.score}/100")
    print(f"Score explanation: {score.explanation}")
    print()
    print("Recommended activities:")
    for activity in recommendation.recommended_activities:
        print(f"- {activity}")
    print()
    print(f"Reasoning: {recommendation.reasoning}")


def _run_agent_demo(
    prompt: str,
    thread_id: str,
    show_tool_calls: bool,
    debug: bool,
) -> None:
    """Run the Phase 5 agent in one-shot or interactive mode."""
    default_prompt = (
        "In one sentence, explain why weather matters when planning travel."
    )
    if prompt != default_prompt:
        print("Agent:")
        print(
            run_agent(
                prompt,
                thread_id=thread_id,
                show_tool_calls=show_tool_calls or debug,
                debug=debug,
            )
        )
        return

    print()
    print("Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if user_query.lower() in {"exit", "quit"}:
            return
        if not user_query:
            continue

        answer = run_agent(
            user_query,
            thread_id=thread_id,
            show_tool_calls=show_tool_calls or debug,
            debug=debug,
        )
        print(f"Agent: {answer}")
        print()


def _run_hitl_demo() -> None:
    """Run the LangGraph interrupt/resume destination-selection example."""
    print("Travel + Weather Agent - Human-in-the-loop Demo")
    print()
    interrupted = run_hitl_demo(thread_id="hitl-cli-demo")
    payload = _extract_interrupt_payload(interrupted)
    candidates = payload.get("candidate_cities", DEFAULT_CANDIDATE_CITIES)

    print("Candidate destinations:")
    for index, city in enumerate(candidates, start=1):
        print(f"{index}. {city}")

    selection = input("Choose a destination: ").strip()
    print()
    print("Resuming graph...")
    result = run_hitl_demo(selection=selection, thread_id="hitl-cli-demo")
    print()
    print(result.get("final_recommendation", "No recommendation was produced."))


def _extract_interrupt_payload(result: dict) -> dict:
    """Extract a JSON-serializable interrupt payload from LangGraph output."""
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return {"candidate_cities": DEFAULT_CANDIDATE_CITIES}
    first = interrupts[0]
    value = getattr(first, "value", first)
    return value if isinstance(value, dict) else {"candidate_cities": DEFAULT_CANDIDATE_CITIES}


if __name__ == "__main__":
    main()
