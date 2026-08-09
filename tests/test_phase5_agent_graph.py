"""Tests for Phase 5 agent, middleware, graph, and memory."""

from __future__ import annotations

import logging

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

try:
    from langgraph.errors import GraphRecursionError
except ImportError:  # pragma: no cover - compatibility with older LangGraph builds
    GraphRecursionError = Exception

from travel_weather_agent.agents import travel_agent
from travel_weather_agent.graph.travel_graph import build_travel_graph, invoke_travel_graph
from travel_weather_agent.middleware.logging import log_request, new_request_context


class ScriptedToolCallingModel:
    """Fake chat model that requests configured tools, then returns a final answer."""

    def __init__(self, tool_calls: list[dict], final_text: str) -> None:
        self.tool_calls = tool_calls
        self.final_text = final_text
        self.invocations = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.invocations.append(messages)
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content=self.final_text)
        if not self.tool_calls:
            return AIMessage(content=self.final_text)
        return AIMessage(content="", tool_calls=self.tool_calls)


class NeverEndingToolModel:
    """Fake model that always requests another tool call."""

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "mock_weather",
                    "args": {"city": "Boston"},
                    "id": "loop",
                }
            ],
        )


class MemoryAwareModel:
    """Fake model that reveals which thread history it can see."""

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        human_text = "\n".join(
            str(message.content)
            for message in messages
            if getattr(message, "type", None) == "human"
        )
        if "warm weather" in human_text and "Should I visit" in human_text:
            return AIMessage(content="I remember you prefer warm weather.")
        return AIMessage(content=f"History: {human_text}")


@tool
def mock_weather(city: str, forecast_days: int = 5) -> dict:
    """Return mocked weather for a city."""
    return {
        "city": city,
        "forecast": [
            {
                "temperature_max": 72,
                "precipitation_probability": 10,
                "wind_speed_max": 8,
            }
        ],
    }


@tool
def mock_score(
    temperature_max: float,
    precipitation_probability: float,
    wind_speed: float,
) -> dict:
    """Return a mocked deterministic trip score."""
    return {"score": 95}


@tool
def mock_activities(city: str, activity_type: str = "all") -> dict:
    """Return mocked activity ideas."""
    raise AssertionError("Activity tool should not have been called.")


@tool
def broken_tool(city: str) -> dict:
    """Raise an error to test tool failure normalization inside the loop."""
    raise RuntimeError("provider exploded")


def test_agent_can_answer_direct_weather_question_using_weather_tool() -> None:
    model = ScriptedToolCallingModel(
        [
            {
                "name": "mock_weather",
                "args": {"city": "Boston"},
                "id": "weather-1",
            }
        ],
        "Boston is mild and mostly clear.",
    )
    graph = build_travel_graph(
        llm=model,
        tools=[mock_weather],
        checkpointer=InMemorySaver(),
    )

    answer = invoke_travel_graph(
        user_query="What is the weather in Boston?",
        thread_id="weather-test",
        graph=graph,
    )

    assert answer == "Boston is mild and mostly clear."
    assert len(model.invocations) == 2


def test_agent_can_call_multiple_tools_for_comparison() -> None:
    model = ScriptedToolCallingModel(
        [
            {
                "name": "mock_weather",
                "args": {"city": "Boston"},
                "id": "weather-boston",
            },
            {
                "name": "mock_score",
                "args": {
                    "temperature_max": 72,
                    "precipitation_probability": 10,
                    "wind_speed": 8,
                },
                "id": "score-boston",
            },
        ],
        "Boston has the better trip score.",
    )
    graph = build_travel_graph(
        llm=model,
        tools=[mock_weather, mock_score],
        checkpointer=InMemorySaver(),
    )

    answer = invoke_travel_graph(
        user_query="Should I visit Boston or Miami?",
        thread_id="comparison-test",
        graph=graph,
    )

    assert "Boston" in answer
    second_call_messages = model.invocations[1]
    tool_messages = [
        message for message in second_call_messages if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 2


def test_agent_does_not_call_activity_tool_for_weather_only_request() -> None:
    model = ScriptedToolCallingModel(
        [
            {
                "name": "mock_weather",
                "args": {"city": "Boston"},
                "id": "weather-1",
            }
        ],
        "Here is the weather only.",
    )
    graph = build_travel_graph(
        llm=model,
        tools=[mock_weather, mock_activities],
        checkpointer=InMemorySaver(),
    )

    answer = invoke_travel_graph(
        user_query="What is the weather in Boston?",
        thread_id="weather-only-test",
        graph=graph,
    )

    assert answer == "Here is the weather only."


def test_tool_errors_are_converted_into_clean_failures() -> None:
    model = ScriptedToolCallingModel(
        [
            {
                "name": "broken_tool",
                "args": {"city": "Boston"},
                "id": "broken-1",
            }
        ],
        "I couldn't retrieve weather data for that location.",
    )
    graph = build_travel_graph(
        llm=model,
        tools=[broken_tool],
        checkpointer=InMemorySaver(),
    )

    answer = invoke_travel_graph(
        user_query="What is the weather in Boston?",
        thread_id="tool-error-test",
        graph=graph,
    )

    assert "couldn't retrieve weather data" in answer


def test_empty_input_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        travel_agent,
        "invoke_travel_graph",
        lambda **kwargs: "should not run",
    )

    answer = travel_agent.run_agent("   ")

    assert "Please enter" in answer


def test_request_logging_middleware_does_not_crash_execution(caplog) -> None:
    caplog.set_level(logging.INFO, logger="travel_weather_agent")
    context = new_request_context(thread_id="log-test", user_query="Hello")

    with log_request(context):
        pass

    assert "request start" in caplog.text
    assert "request finished" in caplog.text


def test_graph_routes_agent_to_tools_then_agent() -> None:
    model = ScriptedToolCallingModel(
        [
            {
                "name": "mock_weather",
                "args": {"city": "Boston"},
                "id": "weather-1",
            }
        ],
        "Done after tool.",
    )
    graph = build_travel_graph(
        llm=model,
        tools=[mock_weather],
        checkpointer=InMemorySaver(),
    )

    answer = invoke_travel_graph(
        user_query="Weather?",
        thread_id="route-test",
        graph=graph,
    )

    assert answer == "Done after tool."
    assert len(model.invocations) == 2


def test_graph_terminates_when_no_tool_calls_remain() -> None:
    model = ScriptedToolCallingModel([], "No tools needed.")
    graph = build_travel_graph(
        llm=model,
        tools=[mock_weather],
        checkpointer=InMemorySaver(),
    )

    answer = invoke_travel_graph(
        user_query="Say hello.",
        thread_id="terminate-test",
        graph=graph,
    )

    assert answer == "No tools needed."
    assert len(model.invocations) == 1


def test_graph_respects_recursion_limit() -> None:
    graph = build_travel_graph(
        llm=NeverEndingToolModel(),
        tools=[mock_weather],
        checkpointer=InMemorySaver(),
    )

    with pytest.raises(GraphRecursionError):
        invoke_travel_graph(
            user_query="Loop please.",
            thread_id="recursion-test",
            graph=graph,
            recursion_limit=3,
        )


def test_same_thread_id_preserves_message_history() -> None:
    graph = build_travel_graph(
        llm=MemoryAwareModel(),
        tools=[mock_weather],
        checkpointer=InMemorySaver(),
    )

    invoke_travel_graph(
        user_query="I prefer warm weather.",
        thread_id="memory-a",
        graph=graph,
    )
    answer = invoke_travel_graph(
        user_query="Should I visit Boston or Miami?",
        thread_id="memory-a",
        graph=graph,
    )

    assert "remember" in answer.lower()


def test_different_thread_ids_remain_isolated() -> None:
    graph = build_travel_graph(
        llm=MemoryAwareModel(),
        tools=[mock_weather],
        checkpointer=InMemorySaver(),
    )

    invoke_travel_graph(
        user_query="I prefer warm weather.",
        thread_id="memory-a",
        graph=graph,
    )
    answer = invoke_travel_graph(
        user_query="Should I visit Boston or Miami?",
        thread_id="memory-b",
        graph=graph,
    )

    assert "warm weather" not in answer.lower()
