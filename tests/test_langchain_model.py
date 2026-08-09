"""Tests for the Phase 3 LangChain model abstraction."""

from __future__ import annotations

import pytest

from travel_weather_agent.config.settings import Settings
from travel_weather_agent.llm import model
from travel_weather_agent.llm.model import (
    MissingAPIKeyError,
    UnsupportedProviderError,
)


class FakeChatModel:
    """Records constructor args so provider selection can be tested."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class FakeAIMessage:
    """Tiny AIMessage stand-in for testing content extraction."""

    def __init__(self, content) -> None:
        self.content = content


class FakeInvokableModel:
    """Fake LangChain chat model with an invoke method."""

    def __init__(self, response_content: str) -> None:
        self.response_content = response_content
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return FakeAIMessage(self.response_content)


class FakeSystemMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeHumanMessage:
    def __init__(self, content: str) -> None:
        self.content = content


def test_get_llm_selects_groq_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model, "_load_chat_groq", lambda: FakeChatModel)
    settings = Settings(
        LLM_PROVIDER="groq",
        GROQ_API_KEY="test-key",
        GROQ_MODEL="llama-3.3-70b-versatile",
    )

    llm = model.get_llm(settings)

    assert isinstance(llm, FakeChatModel)
    assert llm.kwargs["model"] == "llama-3.3-70b-versatile"
    assert llm.kwargs["groq_api_key"] == "test-key"


def test_get_llm_selects_openai_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model, "_load_chat_openai", lambda: FakeChatModel)
    settings = Settings(
        LLM_PROVIDER="openai",
        OPENAI_API_KEY="test-key",
        OPENAI_MODEL="gpt-4o-mini",
    )

    llm = model.get_llm(settings)

    assert isinstance(llm, FakeChatModel)
    assert llm.kwargs["model"] == "gpt-4o-mini"
    assert llm.kwargs["api_key"] == "test-key"


def test_get_llm_selects_ollama_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model, "_load_chat_ollama", lambda: FakeChatModel)
    settings = Settings(
        LLM_PROVIDER="ollama",
        OLLAMA_MODEL="llama3.1",
        OLLAMA_BASE_URL="http://localhost:11434",
    )

    llm = model.get_llm(settings)

    assert isinstance(llm, FakeChatModel)
    assert llm.kwargs["model"] == "llama3.1"
    assert llm.kwargs["base_url"] == "http://localhost:11434"


def test_unsupported_provider_raises_clean_error() -> None:
    settings = Settings(LLM_PROVIDER="anthropic")

    with pytest.raises(UnsupportedProviderError, match="Unsupported LLM_PROVIDER"):
        model.get_llm(settings)


def test_missing_groq_api_key_raises_clean_error() -> None:
    settings = Settings(LLM_PROVIDER="groq", GROQ_API_KEY="")

    with pytest.raises(MissingAPIKeyError, match="GROQ_API_KEY is required"):
        model.get_llm(settings)


def test_missing_openai_api_key_raises_clean_error() -> None:
    settings = Settings(LLM_PROVIDER="openai", OPENAI_API_KEY="")

    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY is required"):
        model.get_llm(settings)


def test_ask_with_langchain_extracts_ai_message_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = FakeInvokableModel("Bring layers for changing weather.")
    monkeypatch.setattr(model, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        model,
        "_load_message_classes",
        lambda: (FakeSystemMessage, FakeHumanMessage),
    )

    text = model.ask_with_langchain("Give me one travel tip.")

    assert text == "Bring layers for changing weather."
    assert fake_llm.messages[0].content == model.SYSTEM_INSTRUCTION
    assert fake_llm.messages[1].content == "Give me one travel tip."

