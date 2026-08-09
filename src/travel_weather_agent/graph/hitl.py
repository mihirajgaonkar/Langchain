"""Human-in-the-loop LangGraph example for Phase 6."""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from travel_weather_agent.tools.destination import find_destination_activities


DEFAULT_CANDIDATE_CITIES = ["Boston", "New York", "Washington DC"]


class HumanReviewState(TypedDict, total=False):
    """Minimal state for the HITL destination-selection demo."""

    candidate_cities: list[str]
    selected_city: str | None
    final_recommendation: str


def build_hitl_graph():
    """Build a graph that pauses for a human destination choice."""

    def ask_human(state: HumanReviewState) -> dict[str, str | None]:
        candidates = state.get("candidate_cities", DEFAULT_CANDIDATE_CITIES)
        selection = interrupt(
            {
                "question": "Which destination should I explore further?",
                "candidate_cities": candidates,
            }
        )
        return {"selected_city": _normalize_selection(selection, candidates)}

    def recommend(state: HumanReviewState) -> dict[str, str]:
        city = state.get("selected_city") or DEFAULT_CANDIDATE_CITIES[0]
        activities = find_destination_activities(city)
        if activities is None:
            recommendation = (
                f"Recommendation for {city}: I do not have local activity data "
                "for that destination yet."
            )
        else:
            recommendation = (
                f"Recommendation for {activities.city}: start with "
                f"{activities.outdoor[0]} if the weather is pleasant, or "
                f"{activities.indoor[0]} if conditions turn poor."
            )
        return {"final_recommendation": recommendation}

    builder = StateGraph(HumanReviewState)
    builder.add_node("ask_human", ask_human)
    builder.add_node("recommend", recommend)
    builder.add_edge(START, "ask_human")
    builder.add_edge("ask_human", "recommend")
    builder.add_edge("recommend", END)
    return builder.compile(checkpointer=InMemorySaver())


_HITL_GRAPH = None


def get_hitl_graph():
    """Return a process-wide HITL graph so interrupts can be resumed."""
    global _HITL_GRAPH
    if _HITL_GRAPH is None:
        _HITL_GRAPH = build_hitl_graph()
    return _HITL_GRAPH


def run_hitl_demo(selection: str | None = None, *, thread_id: str = "hitl-demo") -> dict:
    """Run or resume the HITL graph.

    If `selection` is omitted, the graph pauses and returns `__interrupt__`.
    If `selection` is provided, the graph resumes from the saved checkpoint.
    """
    graph = get_hitl_graph()
    config = {"configurable": {"thread_id": thread_id}}
    if selection is None:
        return graph.invoke({"candidate_cities": DEFAULT_CANDIDATE_CITIES}, config=config)
    return graph.invoke(Command(resume=selection), config=config)


def _normalize_selection(selection: object, candidates: list[str]) -> str:
    """Normalize a numeric or textual human destination selection."""
    text = str(selection).strip()
    if text.isdigit():
        index = int(text) - 1
        if 0 <= index < len(candidates):
            return candidates[index]

    for city in candidates:
        if city.lower() == text.lower():
            return city

    return candidates[0]
