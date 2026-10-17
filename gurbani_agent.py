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


# Used when ANTHROPIC_MODEL is not set. Only Opus models are offered, for answer accuracy.
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"

CLAUDE_MODELS: List[str] = [
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
]

# Generation params the Anthropic Messages API does not accept.
ANTHROPIC_UNSUPPORTED_PARAMS = frozenset({"frequency_penalty", "presence_penalty"})

# Opus 4.7 and newer Claude models reject sampling params with a 400; Opus 4.6 still accepts them.
CLAUDE_SAMPLING_PARAMS = frozenset({"temperature", "top_p", "top_k"})
CLAUDE_NO_SAMPLING_PREFIXES = (
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-5",
    "claude-fable",
    "claude-mythos",
)

# Anthropic requires max_tokens on every call; this is generous enough for a full answer.
DEFAULT_CLAUDE_MAX_TOKENS = 4096

# OpenAI reasoning models (o-series, gpt-5 family) reject sampling params and
# take max_completion_tokens instead of max_tokens. Their hidden reasoning also
# counts against that budget.
OPENAI_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")
OPENAI_REASONING_UNSUPPORTED = frozenset(
    {"temperature", "top_p", "frequency_penalty", "presence_penalty", "logit_bias", "stop"}
)

TRUNCATION_NOTE = (
    "\n\n_(Response stopped at the output length limit and may be incomplete. "
    "Raise `max_tokens` in Controls for a longer answer.)_"
)


def build_system_prompt() -> str:
    """Shared guidance prompt for all model providers."""
    return (
        "You are GurbaniAI, a respectful Sikh studies assistant. "
        "You are an expert in Gurbani, Gurmukhi, Sikhism, and Shri Guru Granth Sahib Ji - literal translations as well as transliterations. "
        "Give priority to spiritual and metaphorical translations of the Shri Guru Granth Sahib. "
        "Help users understand meanings, context, and spiritual teachings in Sikh scriptures "
        "like Shri Guru Granth Sahib Ji, Jaap Sahib, and Rehras Sahib etc. "
        "Provide balanced, non-sectarian explanations. "
        "If a verse reference is uncertain, clearly say so and suggest verifying with reliable sources. "
        "Do not claim to issue hukamnama or personal religious authority. "
        "When possible, explain difficult words and practical takeaways for daily life. "
        "Answer questions from seekers about life and living from Gurbani – Shri Guru Granth Sahib Jee. "
        "Try to create an understanding of Gurbani beyond what is available from the literal translations of the Shri Guru Granth Sahib Jee. "
        "Be mindful, respectful, be open. Offer different interpretations of a shabad or verse or a line in the scriptures - whenever possible or needed. "
        "Always provide references to the source of the shabad or verse, including page numbers and line numbers from Shri Guru Granth Sahib Jee or any other trusted verified sources. "
        "Always refer to religious scriptures with respect, preferably using the word \"Ji\" after name of the scripture or the name of the Guru."
    )


def build_openai_messages(
    chat_history: List[Dict[str, str]],
    system_prompt: str | None = None,
) -> List[Dict[str, str]]:
    """Create OpenAI-style messages with system prompt prepended."""
    prompt = (system_prompt or "").strip() or build_system_prompt()
    messages: List[Dict[str, str]] = [{"role": "system", "content": prompt}]
    messages.extend(chat_history)
    return messages


def openai_compatible_base_url() -> str | None:
    """Return OPENAI_BASE_URL when it points at a non-OpenAI server (llama.cpp, vLLM, ...).

    The OpenAI SDK picks up OPENAI_BASE_URL on its own; this only tells the app
    whether server-specific sampling params (top_k, min_p, mirostat) are safe to send.
    """
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if base_url and "api.openai.com" not in base_url:
        return base_url
    return None


def is_openai_reasoning_model(model: str) -> bool:
    return model.lower().startswith(OPENAI_REASONING_PREFIXES)


def claude_rejects_sampling(model: str) -> bool:
    return model.lower().startswith(CLAUDE_NO_SAMPLING_PREFIXES)


def build_openai_params(model: str, params: Dict | None) -> Dict:
    """Translate Controls params into kwargs the chosen OpenAI model accepts."""
    out = dict(params or {})
    if is_openai_reasoning_model(model):
        for key in OPENAI_REASONING_UNSUPPORTED:
            out.pop(key, None)
        if "max_tokens" in out:
            out["max_completion_tokens"] = out.pop("max_tokens")
    else:
        # reasoning_effort is only valid for reasoning models; sending it elsewhere returns a 400.
        out.pop("reasoning_effort", None)
    # With no max_tokens set, the model's own (large) output limit applies.
    return out


def ask_openai(
    chat_history: List[Dict[str, str]],
    *,
    system_prompt: str | None = None,
    generation_params: Dict | None = None,
    stream: bool = False,
    model: str | None = None,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key and openai_compatible_base_url():
        # Local OpenAI-compatible servers usually ignore the key, but the SDK requires one.
        api_key = "not-needed"
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
    kwargs: Dict = {
        "model": model,
        "messages": build_openai_messages(chat_history, system_prompt=system_prompt),
    }
    kwargs.update(build_openai_params(model, generation_params))

    if stream:
        try:
            chunks = client.chat.completions.create(stream=True, **kwargs)
        except Exception as exc:  # noqa: BLE001
            return (
                "OpenAI streaming request failed. Check Controls (stream / params).\n\n"
                f"Sent params: {generation_params or {}}\n\n"
                f"Error: {exc}"
            )
        parts: List[str] = []
        finish_reason = None
        for chunk in chunks:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                parts.append(delta)
            finish_reason = chunk.choices[0].finish_reason or finish_reason
        text = "".join(parts) or "I could not generate a response."
        return text + (TRUNCATION_NOTE if finish_reason == "length" else "")

    try:
        completion = client.chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001
        return (
            "OpenAI request failed. Check your Controls params (temperature, max_tokens, etc.).\n\n"
            f"Sent params: {generation_params or {}}\n\n"
            f"Error: {exc}"
        )
    choice = completion.choices[0]
    text = choice.message.content or "I could not generate a response."
    return text + (TRUNCATION_NOTE if choice.finish_reason == "length" else "")


def ask_anthropic(
    chat_history: List[Dict[str, str]],
    *,
    system_prompt: str | None = None,
    generation_params: Dict | None = None,
    model: str | None = None,
) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    configured_model = model or os.getenv("ANTHROPIC_MODEL", DEFAULT_CLAUDE_MODEL)
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

    # An explicitly chosen model is used alone so the user never gets a silent substitute.
    # Only the env/default model falls back through the list below.
    if model:
        model_candidates = [configured_model]
    else:
        # Fall back only to other Opus models so a missing model never quietly lowers quality.
        model_candidates = [configured_model, DEFAULT_CLAUDE_MODEL, "claude-opus-5"]
    model_candidates = list(dict.fromkeys(model_candidates))

    last_error: Exception | None = None
    prompt = (system_prompt or "").strip() or build_system_prompt()
    gen = {
        k: v
        for k, v in (generation_params or {}).items()
        if k not in ANTHROPIC_UNSUPPORTED_PARAMS
    }
    max_tokens = int(gen.pop("max_tokens", DEFAULT_CLAUDE_MAX_TOKENS))

    for candidate in model_candidates:
        candidate_gen = {
            k: v
            for k, v in gen.items()
            if not (k in CLAUDE_SAMPLING_PARAMS and claude_rejects_sampling(candidate))
        }
        try:
            completion = client.messages.create(
                model=candidate,
                system=prompt,
                messages=chat_history,
                max_tokens=max_tokens,
                **candidate_gen,
            )
            text = "".join(
                block.text for block in (completion.content or []) if getattr(block, "type", "") == "text"
            )
            text = text or "I could not generate a response."
            return text + (TRUNCATION_NOTE if completion.stop_reason == "max_tokens" else "")
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


def ask_gurbani_llm(
    chat_history: List[Dict[str, str]],
    *,
    system_prompt: str | None = None,
    generation_params: Dict | None = None,
    stream: bool = False,
    model: str | None = None,
) -> str:
    """Send chat history to configured provider and return response.

    Answers come from the model's own knowledge, guided by the system prompt;
    no external lookup or retrieval is used. `model` overrides the provider's
    env-configured model.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider == "anthropic":
        return ask_anthropic(
            chat_history,
            system_prompt=system_prompt,
            generation_params=generation_params,
            model=model,
        )
    return ask_openai(
        chat_history,
        system_prompt=system_prompt,
        generation_params=generation_params,
        stream=stream,
        model=model,
    )


def get_provider_info() -> tuple[str, str]:
    """Return (provider, model) for display."""
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    model = os.getenv(
        "ANTHROPIC_MODEL" if provider == "anthropic" else "OPENAI_MODEL",
        DEFAULT_CLAUDE_MODEL if provider == "anthropic" else "gpt-4o-mini",
    )
    return provider, model
