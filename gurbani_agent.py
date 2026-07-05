"""GurbaniAI agent core: system prompt and LLM provider calls."""

import os
from typing import Dict, List

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def build_system_prompt() -> str:
    """Shared guidance prompt for all model providers."""
    return (
        "You are GurbaniAI, a respectful Sikh studies assistant. "
        "You are an expert in Gurbani, Gurmukhi, Sikhism, and Shri Guru Granth Sahib Ji - literal translations as well as transliterations."
        "Give priority to spiritual and metaphorical translations of the Shri Guru Granth Sahib"
        "Help users understand meanings, context, and spiritual teachings in Sikh scriptures"
        "like Shri Guru Granth Sahib Ji, Jaap Sahib, and Rehras Sahib etc. "
        "Provide balanced, non-sectarian explanations."
        "If a verse reference is uncertain, clearly say so and suggest verifying with reliable sources. "
        "Do not claim to issue hukamnama or personal religious authority. "
        "When possible, explain difficult words and practical takeaways for daily life."
        "Answer questions from Sikhs and other seekers about life and living from Gurbani – Shri Guru Granth Sahib Jee"
        "Create an understanding of Gurbani beyond what is available from the literal translations of the Shri Guru Granth Sahib Jee"
        "Be mindful, respectful, be open. Offer different interpretations of a shabad or sentence whenever possible or needed."
    )


def build_openai_messages(chat_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Create OpenAI-style messages with system prompt prepended."""
    messages: List[Dict[str, str]] = [{"role": "system", "content": build_system_prompt()}]
    messages.extend(chat_history)
    return messages


def ask_openai(chat_history: List[Dict[str, str]]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return (
            "OpenAI mode is enabled, but no API key is set.\n\n"
            "Set `OPENAI_API_KEY` in your environment and restart the app."
        )

    if OpenAI is None:
        return (
            "The OpenAI SDK is not installed.\n\n"
            "Run: `python -m pip install -r requirements.txt`"
        )

    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=build_openai_messages(chat_history),
        temperature=0.4,
    )
    return completion.choices[0].message.content or "I could not generate a response."


def ask_anthropic(chat_history: List[Dict[str, str]]) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    configured_model = os.getenv("ANTHROPIC_MODEL", "claude-4-5-haiku-latest")
    if not api_key:
        return (
            "Claude mode is enabled, but no API key is set.\n\n"
            "Set `ANTHROPIC_API_KEY` in your environment and restart the app."
        )

    if anthropic is None:
        return (
            "The Anthropic SDK is not installed.\n\n"
            "Run: `python -m pip install -r requirements.txt`"
        )

    client = anthropic.Anthropic(api_key=api_key)

    model_candidates = [
        configured_model,
        "claude-haiku-4-5",
        "claude-haiku-4-5-20251001",
        "claude-3-7-sonnet-latest",
        "claude-sonnet-4-6",
    ]
    model_candidates = list(dict.fromkeys(model_candidates))

    last_error: Exception | None = None
    for model in model_candidates:
        try:
            completion = client.messages.create(
                model=model,
                system=build_system_prompt(),
                messages=chat_history,
                temperature=0.4,
                max_tokens=800,
            )
            if completion.content and len(completion.content) > 0:
                return completion.content[0].text
            return "I could not generate a response."
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            error_name = exc.__class__.__name__
            if error_name not in {"NotFoundError"}:
                break

    if last_error is not None:
        safe_error_text = (
            str(last_error).encode("ascii", errors="backslashreplace").decode("ascii")
        )
        return (
            "Claude request failed. Check your `ANTHROPIC_MODEL` and API key permissions.\n\n"
            f"Tried models: {', '.join(model_candidates)}\n\n"
            f"Last error: {safe_error_text}"
        )
    return "I could not generate a response."


def ask_gurbani_llm(chat_history: List[Dict[str, str]]) -> str:
    """Send chat history to configured provider and return response."""
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider == "anthropic":
        return ask_anthropic(chat_history)
    return ask_openai(chat_history)


def get_provider_info() -> tuple[str, str]:
    """Return (provider, model) for display."""
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    model = os.getenv(
        "ANTHROPIC_MODEL" if provider == "anthropic" else "OPENAI_MODEL",
        "claude-haiku-4-5" if provider == "anthropic" else "gpt-4o-mini",
    )
    return provider, model
