"""Input validation middleware for public agent entry points."""

from __future__ import annotations


MAX_QUERY_LENGTH = 2000


class InputValidationError(ValueError):
    """Raised when a user query is too empty or too large to process safely."""


def validate_user_query(user_query: str) -> str:
    """Validate and normalize a user query before invoking the agent."""
    normalized = user_query.strip()
    if not normalized:
        raise InputValidationError("Please enter a travel or weather question.")
    if len(normalized) > MAX_QUERY_LENGTH:
        raise InputValidationError(
            f"Please keep your question under {MAX_QUERY_LENGTH} characters."
        )
    return normalized

