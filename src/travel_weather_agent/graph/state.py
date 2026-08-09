"""LangGraph state for the travel/weather agent."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class TravelAgentState(TypedDict, total=False):
    """Minimal graph state.

    Graph state is data for the current execution. The `messages` field also
    becomes conversation memory when persisted by a LangGraph checkpointer.
    """

    messages: Annotated[list[Any], add_messages]
    thread_id: str
    show_tool_calls: bool
    debug: bool
    tool_call_count: int
