"""LangGraph workflow for the Phase 5 tool-calling agent."""

from __future__ import annotations

import json
import time
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from travel_weather_agent.agents.prompts import TRAVEL_AGENT_SYSTEM_PROMPT
from travel_weather_agent.config.observability import trace_metadata
from travel_weather_agent.graph.state import TravelAgentState
from travel_weather_agent.llm.model import get_llm, _stringify_content_blocks
from travel_weather_agent.middleware.logging import log_tool_result, sanitize_tool_args
from travel_weather_agent.tools.destination import get_destination_activities
from travel_weather_agent.tools.geocoding import geocode_location
from travel_weather_agent.tools.scoring import calculate_trip_score
from travel_weather_agent.tools.weather import get_weather


DEFAULT_RECURSION_LIMIT = 12


def get_agent_tools() -> list[BaseTool]:
    """Return the tools available to the Phase 5 agent."""
    return [
        get_weather,
        geocode_location,
        get_destination_activities,
        calculate_trip_score,
    ]


def build_travel_graph(
    *,
    llm: Any | None = None,
    tools: list[BaseTool] | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Build the explicit LangGraph agent/tool loop.

    The graph is intentionally visible:
    agent -> tools when the model asks for tools, otherwise agent -> END.
    """
    resolved_tools = tools or get_agent_tools()
    model = llm or get_llm()
    model_with_tools = model.bind_tools(resolved_tools)
    tools_by_name = {tool.name: tool for tool in resolved_tools}

    def call_agent(state: TravelAgentState) -> dict[str, list[AIMessage]]:
        if state.get("debug", False):
            print("[agent] invoking model")
        messages = [
            SystemMessage(content=TRAVEL_AGENT_SYSTEM_PROMPT),
            *state["messages"],
        ]
        response = model_with_tools.invoke(messages)
        if state.get("debug", False):
            tool_calls = getattr(response, "tool_calls", []) or []
            if tool_calls:
                print(f"[agent] requested {len(tool_calls)} tool call(s)")
            else:
                print("[agent] final response")
        return {"messages": [response]}

    def call_tools(state: TravelAgentState) -> dict[str, Any]:
        last_message = state["messages"][-1]
        tool_messages: list[ToolMessage] = []
        show_tool_calls = state.get("show_tool_calls", False)

        for tool_call in getattr(last_message, "tool_calls", []) or []:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            if not isinstance(tool_args, dict):
                tool_args = {}
            tool_call_id = tool_call.get("id", tool_name)
            start = time.perf_counter()
            success = False

            if show_tool_calls:
                print(f"[tool] {tool_name}({_format_tool_args(tool_args)})")

            try:
                tool = tools_by_name[tool_name]
                result = tool.invoke(tool_args)
                success = not (isinstance(result, dict) and "error" in result)
                content = _tool_result_to_text(result)
                if state.get("debug", False):
                    print(f"[tool] {tool_name} result: {_summarize_tool_result(result)}")
            except Exception as exc:
                content = f"Tool error: {exc}"
                if state.get("debug", False):
                    print(f"[tool] {tool_name} failed: {exc}")

            duration = time.perf_counter() - start
            log_tool_result(
                tool_name=tool_name,
                args=tool_args,
                duration=duration,
                success=success,
            )
            tool_messages.append(
                ToolMessage(content=content, tool_call_id=tool_call_id)
            )

        return {
            "messages": tool_messages,
            "tool_call_count": state.get("tool_call_count", 0) + len(tool_messages),
        }

    def should_continue(state: TravelAgentState) -> Literal["tools", "__end__"]:
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    builder = StateGraph(TravelAgentState)
    builder.add_node("agent", call_agent)
    builder.add_node("tools", call_tools)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer)


_CHECKPOINTER = InMemorySaver()
_GRAPH = None


def get_compiled_travel_graph() -> Any:
    """Return a process-wide graph with in-memory conversation checkpointing."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_travel_graph(checkpointer=_CHECKPOINTER)
    return _GRAPH


def invoke_travel_graph(
    *,
    user_query: str,
    thread_id: str,
    show_tool_calls: bool = False,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    graph: Any | None = None,
    debug: bool = False,
) -> str:
    """Invoke the travel graph and return the final assistant text."""
    compiled_graph = graph or get_compiled_travel_graph()
    result = compiled_graph.invoke(
        {
            "messages": [HumanMessage(content=user_query)],
            "thread_id": thread_id,
            "show_tool_calls": show_tool_calls,
            "debug": debug,
            "tool_call_count": 0,
        },
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": recursion_limit,
            "metadata": trace_metadata(thread_id=thread_id),
            "tags": ["travel-weather-agent", "phase-6"],
        },
    )
    return _extract_final_text(result["messages"])


def _extract_final_text(messages: list[Any]) -> str:
    """Extract the final AI message content from graph output."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = _message_content_to_text(message.content)
            if text:
                return text
    return ""


def _message_content_to_text(content: Any) -> str:
    """Normalize LangChain message content into plain text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        try:
            return _stringify_content_blocks(content).strip()
        except Exception:
            return ""
    return ""


def _tool_result_to_text(result: Any) -> str:
    """Serialize a tool result into ToolMessage content."""
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=True)


def _format_tool_args(args: dict[str, Any]) -> str:
    """Format sanitized tool args for the optional CLI debug display."""
    sanitized = sanitize_tool_args(args)
    return ", ".join(f"{key}={value!r}" for key, value in sanitized.items())


def _summarize_tool_result(result: Any) -> str:
    """Return a compact tool result summary for learner debug output."""
    text = _tool_result_to_text(result)
    return text if len(text) <= 160 else f"{text[:157]}..."


def get_travel_graph_mermaid() -> str:
    """Return an educational Mermaid diagram for the agent/tool loop."""
    return """flowchart TD
    START --> Agent
    Agent --> Decision{Tool call?}
    Decision -->|tool call| Tools
    Tools --> Agent
    Decision -->|no tool call| END
"""
