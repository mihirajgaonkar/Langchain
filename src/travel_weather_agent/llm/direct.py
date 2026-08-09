"""Direct LLM API calls without LangChain.

This module is intentionally small and explicit. It shows the lower-level work
that LangChain will abstract in the next phase: choosing a provider, building
HTTP payloads, sending requests, checking errors, and extracting text.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_OLLAMA_MODEL = "llama3.1"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class RawLLMResponse:
    """Minimal normalized response from a direct LLM API call."""

    provider: str
    model: str
    text: str


class DirectLLMError(RuntimeError):
    """Raised when a direct provider API call cannot return usable text."""


def call_direct_llm(prompt: str) -> RawLLMResponse:
    """Call the configured LLM provider directly, without LangChain.

    Environment variables:
        LLM_PROVIDER: "openai", "groq", or "ollama".
        OPENAI_API_KEY: Required for OpenAI.
        OPENAI_MODEL: OpenAI model name.
        GROQ_API_KEY: Required for Groq.
        GROQ_MODEL: Groq model name.
        OLLAMA_MODEL: Local Ollama model name.
        OLLAMA_BASE_URL: Base URL for the Ollama server.
    """
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()

    if provider == "openai":
        return call_openai_responses_api(prompt)
    if provider == "groq":
        return call_groq_chat_completions_api(prompt)
    if provider == "ollama":
        return call_ollama_generate_api(prompt)

    raise DirectLLMError(
        f"Unsupported LLM_PROVIDER '{provider}'. Use 'openai', 'groq', or 'ollama'."
    )


def call_openai_responses_api(prompt: str) -> RawLLMResponse:
    """Call OpenAI's Responses API with raw HTTP."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    timeout = _timeout_seconds()

    if not api_key:
        raise DirectLLMError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")

    payload = {
        "model": model,
        "input": prompt,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = httpx.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise DirectLLMError(f"OpenAI request failed: {exc}") from exc

    data = response.json()
    text = _extract_openai_text(data)
    return RawLLMResponse(provider="openai", model=model, text=text)


def call_groq_chat_completions_api(prompt: str) -> RawLLMResponse:
    """Call Groq's OpenAI-compatible Chat Completions API with raw HTTP."""
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    timeout = _timeout_seconds()

    if not api_key:
        raise DirectLLMError("GROQ_API_KEY is required when LLM_PROVIDER=groq.")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise DirectLLMError(f"Groq request failed: {exc}") from exc

    data = response.json()
    text = _extract_chat_completion_text(data, provider_name="Groq")
    return RawLLMResponse(provider="groq", model=model, text=text)


def call_ollama_generate_api(prompt: str) -> RawLLMResponse:
    """Call a local Ollama model with raw HTTP."""
    model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    timeout = _timeout_seconds()

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = httpx.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise DirectLLMError(f"Ollama request failed: {exc}") from exc

    data = response.json()
    text = data.get("response")
    if not isinstance(text, str) or not text.strip():
        raise DirectLLMError("Ollama response did not include generated text.")

    return RawLLMResponse(provider="ollama", model=model, text=text.strip())


def _extract_openai_text(data: dict[str, Any]) -> str:
    """Extract text from a raw OpenAI Responses API payload."""
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    # Keep the fallback explicit so learners can inspect the response shape.
    parts: list[str] = []
    for item in data.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())

    if parts:
        return "\n".join(parts)

    raise DirectLLMError("OpenAI response did not include generated text.")


def _extract_chat_completion_text(
    data: dict[str, Any], *, provider_name: str
) -> str:
    """Extract assistant text from an OpenAI-compatible chat completion payload."""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise DirectLLMError(f"{provider_name} response did not include choices.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise DirectLLMError(f"{provider_name} response choice was not an object.")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise DirectLLMError(f"{provider_name} response did not include a message.")

    text = message.get("content")
    if not isinstance(text, str) or not text.strip():
        raise DirectLLMError(f"{provider_name} response did not include generated text.")

    return text.strip()


def _timeout_seconds() -> float:
    """Read the HTTP timeout from the environment."""
    raw_timeout = os.getenv("HTTP_TIMEOUT_SECONDS")
    if raw_timeout is None:
        return DEFAULT_TIMEOUT_SECONDS

    try:
        return float(raw_timeout)
    except ValueError as exc:
        raise DirectLLMError("HTTP_TIMEOUT_SECONDS must be a number.") from exc
