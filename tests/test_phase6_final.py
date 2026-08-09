"""Final Phase 6 tests for observability, HITL, and learner UX helpers."""

from __future__ import annotations

from langgraph.types import Command

from travel_weather_agent.agents import travel_agent
from travel_weather_agent.config.observability import (
    is_langsmith_enabled,
    trace_metadata,
)
from travel_weather_agent.config.settings import Settings
from travel_weather_agent.graph.hitl import build_hitl_graph
from travel_weather_agent.graph.travel_graph import get_travel_graph_mermaid


def test_langsmith_disabled_without_api_key() -> None:
    settings = Settings(LANGSMITH_TRACING="true", LANGSMITH_API_KEY="")

    assert is_langsmith_enabled(settings) is False


def test_langsmith_enabled_with_api_key() -> None:
    settings = Settings(LANGSMITH_TRACING="true", LANGSMITH_API_KEY="test-key")

    assert is_langsmith_enabled(settings) is True


def test_trace_metadata_contains_no_secrets() -> None:
    settings = Settings(
        LLM_PROVIDER="groq",
        GROQ_API_KEY="secret-value",
        GROQ_MODEL="llama-3.3-70b-versatile",
    )

    metadata = trace_metadata(settings, thread_id="demo-thread")

    assert metadata["application"] == "travel-weather-agent"
    assert metadata["thread_id"] == "demo-thread"
    assert metadata["provider"] == "groq"
    assert metadata["model"] == "llama-3.3-70b-versatile"
    assert "secret-value" not in str(metadata)


def test_hitl_interrupt_and_resume() -> None:
    graph = build_hitl_graph()
    config = {"configurable": {"thread_id": "hitl-test"}}

    interrupted = graph.invoke(
        {"candidate_cities": ["Boston", "New York", "Washington DC"]},
        config=config,
    )

    assert "__interrupt__" in interrupted
    payload = interrupted["__interrupt__"][0].value
    assert payload["candidate_cities"] == ["Boston", "New York", "Washington DC"]

    resumed = graph.invoke(Command(resume="1"), config=config)

    assert resumed["selected_city"] == "Boston"
    assert "Recommendation for Boston" in resumed["final_recommendation"]


def test_graph_mermaid_contains_agent_tool_loop() -> None:
    mermaid = get_travel_graph_mermaid()

    assert "START --> Agent" in mermaid
    assert "Tools --> Agent" in mermaid
    assert "Decision" in mermaid


def test_agent_debug_flag_reaches_graph(monkeypatch) -> None:
    captured = {}

    def fake_invoke(**kwargs):
        captured.update(kwargs)
        return "debug answer"

    monkeypatch.setattr(travel_agent, "invoke_travel_graph", fake_invoke)

    answer = travel_agent.run_agent(
        "What is the weather in Boston?",
        thread_id="debug-test",
        show_tool_calls=True,
        debug=True,
    )

    assert answer == "debug answer"
    assert captured["show_tool_calls"] is True
    assert captured["debug"] is True

