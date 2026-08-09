"""Tests for direct/raw LLM provider calls."""

from __future__ import annotations

import httpx
import pytest

from travel_weather_agent.llm import direct
from travel_weather_agent.llm.direct import DirectLLMError, RawLLMResponse


class MockHTTPResponse:
    """Small response double with the subset of httpx.Response used by direct.py."""

    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://api.groq.com")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError(
                "mock HTTP failure",
                request=request,
                response=response,
            )


def test_successful_groq_response_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Groq chat completion text is normalized into RawLLMResponse."""
    captured_request: dict = {}

    def fake_post(*args, **kwargs) -> MockHTTPResponse:
        captured_request["url"] = args[0]
        captured_request["headers"] = kwargs["headers"]
        captured_request["json"] = kwargs["json"]
        return MockHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Pack a light rain jacket.",
                        }
                    }
                ]
            }
        )

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setattr(direct.httpx, "post", fake_post)

    response = direct.call_groq_chat_completions_api("Give me a travel tip.")

    assert response == RawLLMResponse(
        provider="groq",
        model="llama-3.3-70b-versatile",
        text="Pack a light rain jacket.",
    )
    assert captured_request["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured_request["headers"]["Authorization"] == "Bearer test-key"
    assert captured_request["json"] == {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "user",
                "content": "Give me a travel tip.",
            }
        ],
    }


def test_missing_groq_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Groq requires an API key and reports a clean configuration error."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(DirectLLMError, match="GROQ_API_KEY is required"):
        direct.call_groq_chat_completions_api("Hello")


def test_groq_http_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Groq HTTP errors are wrapped in DirectLLMError."""

    def fake_post(*args, **kwargs) -> MockHTTPResponse:
        return MockHTTPResponse({"error": {"message": "bad request"}}, status_code=400)

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(direct.httpx, "post", fake_post)

    with pytest.raises(DirectLLMError, match="Groq request failed"):
        direct.call_groq_chat_completions_api("Hello")


def test_unsupported_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider routing rejects unknown provider names."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")

    with pytest.raises(DirectLLMError, match="Unsupported LLM_PROVIDER"):
        direct.call_direct_llm("Hello")

