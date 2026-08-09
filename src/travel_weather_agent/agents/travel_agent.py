"""Public API for the Phase 5 travel/weather tool-calling agent."""

from __future__ import annotations

from travel_weather_agent.config.observability import langsmith_tracing_context
from travel_weather_agent.graph.travel_graph import invoke_travel_graph
from travel_weather_agent.middleware.errors import normalize_error
from travel_weather_agent.middleware.logging import log_request, new_request_context
from travel_weather_agent.middleware.retry import retry_call
from travel_weather_agent.middleware.validation import validate_user_query


class AgentRunError(RuntimeError):
    """Raised when the agent cannot produce a usable response."""


def run_agent(
    user_query: str,
    thread_id: str = "default",
    *,
    show_tool_calls: bool = False,
    debug: bool = False,
) -> str:
    """Run the travel/weather agent for one user query.

    The caller does not need to know about model binding, tool orchestration,
    graph routing, or checkpointed memory. Those concerns live behind this API.
    """
    try:
        normalized_query = validate_user_query(user_query)
        context = new_request_context(thread_id=thread_id, user_query=normalized_query)
        with log_request(context), langsmith_tracing_context():
            if debug:
                print("[agent] received request")
            return retry_call(
                lambda: invoke_travel_graph(
                    user_query=normalized_query,
                    thread_id=thread_id,
                    show_tool_calls=show_tool_calls,
                    debug=debug,
                ),
                attempts=2,
                retry_exceptions=(RuntimeError,),
            )
    except Exception as exc:
        return normalize_error(exc)
