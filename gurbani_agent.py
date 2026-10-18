"""GurbaniAI agent core: system prompt and LLM provider calls."""

import json
import os
import re
from typing import Dict, List

import banidb

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

# Shown in the model picker in this order: newest to oldest.
CLAUDE_MODELS: List[str] = [
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-opus-4-8",
]


def is_haiku(model: str) -> bool:
    return "haiku" in model.lower()


def configured_claude_model() -> str:
    """ANTHROPIC_MODEL, or the default. Haiku is not offered in this app, so it is ignored."""
    model = os.getenv("ANTHROPIC_MODEL", "").strip()
    if not model or is_haiku(model):
        return DEFAULT_CLAUDE_MODEL
    return model

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

# Anthropic requires max_tokens on every call. Thinking on current Opus models counts
# against it too, so leave room for a full answer.
DEFAULT_CLAUDE_MAX_TOKENS = 16000

# OpenAI reasoning models (o-series, gpt-5 family) reject sampling params and
# take max_completion_tokens instead of max_tokens. Their hidden reasoning also
# counts against that budget.
OPENAI_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")
OPENAI_REASONING_UNSUPPORTED = frozenset(
    {"temperature", "top_p", "frequency_penalty", "presence_penalty", "logit_bias", "stop"}
)

TRUNCATION_NOTE = (
    "\n\n_(Response stopped at the output length limit and may be incomplete. "
    "Ask a narrower question, or ask it to continue.)_"
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
        "Always refer to religious scriptures with respect, preferably using the word \"Ji\" after name of the scripture or the name of the Guru. "
        "Never invent a pankti or citation. "
        "If you are not sure Gurbani answers the question, say so. "
        "Show Gurmukhi, then transliteration, then translation, then the source."
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
    """Translate generation params into kwargs the chosen OpenAI model accepts."""
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
                "OpenAI streaming request failed.\n\n"
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
            "OpenAI request failed. Check your `OPENAI_MODEL` and API key.\n\n"
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
    if model and is_haiku(model):
        model = None
    configured_model = model or configured_claude_model()
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


BANIDB_QUERY_REQUEST = (
    "Do not answer the question below. Instead, choose BaniDB search words for it.\n"
    "Reply with only a JSON object, no other text, in this form:\n"
    '{{"english": ["anger"], "gurmukhi": ["ਕਰੋਧ"]}}\n'
    "- english: up to 4 single English words likely to appear in English translations of the relevant lines.\n"
    "- gurmukhi: up to 4 single Gurmukhi words (Unicode) likely to appear in the relevant lines themselves.\n"
    "Use single words only, never phrases.\n\n"
    "Question: {question}"
)

BANIDB_ONLY_INSTRUCTIONS = (
    "BaniDB-only mode: the passages below were found in BaniDB for the user's latest question. "
    "Answer using only these passages. You may explain and connect them, but every line of Gurbani "
    "you quote, every citation, and every claim about what Gurbani says must come from these passages. "
    "Copy Gurmukhi, transliteration, translation, and source exactly as given. "
    "Do not add verses, page numbers, or citations from your own knowledge. "
    "If the passages do not answer the question, say so plainly."
)

# Opening words of the readable failure strings returned by ask_anthropic / ask_openai.
PROVIDER_FAILURE_PREFIXES = (
    "Claude mode is enabled",
    "The Anthropic SDK",
    "Claude request failed",
    "OpenAI mode is enabled",
    "The OpenAI SDK",
    "OpenAI request failed",
    "OpenAI streaming request failed",
)

_ENGLISH_STOPWORDS = frozenset(
    "a an and are about does do for from gurbani guru how i in is it me of on say says "
    "tell the to what when where which who why with".split()
)


def _fallback_banidb_queries(question: str) -> tuple[List[str], List[str]]:
    """Pick search words without the model, if its reply could not be parsed."""
    gurmukhi = re.findall(r"[਀-੿]+", question)
    english = [
        w for w in re.findall(r"[A-Za-z]+", question.lower())
        if len(w) > 2 and w not in _ENGLISH_STOPWORDS
    ]
    return list(dict.fromkeys(english))[:4], list(dict.fromkeys(gurmukhi))[:4]


def _plan_banidb_queries(question: str, reply: str) -> tuple[List[str], List[str]]:
    """Read the model's search-word reply; returns (english, gurmukhi)."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    try:
        plan = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        plan = {}
    english = [str(w).strip() for w in plan.get("english") or [] if str(w).strip()][:4]
    gurmukhi = [str(w).strip() for w in plan.get("gurmukhi") or [] if str(w).strip()][:4]
    if not english and not gurmukhi:
        return _fallback_banidb_queries(question)
    return english, gurmukhi


def ask_with_banidb(
    chat_history: List[Dict[str, str]],
    *,
    system_prompt: str | None = None,
    model: str | None = None,
) -> str:
    """Answer only from BaniDB passages: the model picks search words, BaniDB supplies the lines.

    If BaniDB fails or finds nothing, say so rather than answering from the model's own knowledge.
    """
    prompt = (system_prompt or "").strip() or build_system_prompt()
    question = next(
        (m["content"] for m in reversed(chat_history) if m.get("role") == "user"), ""
    )
    plan_reply = _ask_provider(
        [{"role": "user", "content": BANIDB_QUERY_REQUEST.format(question=question)}],
        system_prompt=prompt,
        model=model,
    )
    if plan_reply.startswith(PROVIDER_FAILURE_PREFIXES):
        return plan_reply
    english, gurmukhi = _plan_banidb_queries(question, plan_reply)
    searched = ", ".join(gurmukhi + english) or "(no search words)"

    try:
        verses = banidb.search_many(english, gurmukhi)
    except banidb.BaniDBError as exc:
        return (
            "BaniDB could not be reached, so no answer was given "
            "(BaniDB-only mode does not fall back to the model's own knowledge).\n\n"
            f"Error: {exc}"
        )
    if not verses:
        return (
            f"BaniDB found no lines for: {searched}.\n\n"
            "Try other words, a Gurmukhi word, or switch off **Use BaniDB Only**."
        )

    banidb_prompt = (
        prompt + "\n\n" + BANIDB_ONLY_INSTRUCTIONS + "\n\nBaniDB passages:\n\n"
        + banidb.format_passages(verses)
    )
    answer = _ask_provider(chat_history, system_prompt=banidb_prompt, model=model)
    if answer.startswith(PROVIDER_FAILURE_PREFIXES):
        return answer
    return answer + f"\n\n_Answered from BaniDB ({len(verses)} lines found for: {searched})._"


def ask_gurbani_llm(
    chat_history: List[Dict[str, str]],
    *,
    system_prompt: str | None = None,
    generation_params: Dict | None = None,
    stream: bool = False,
    model: str | None = None,
    banidb_only: bool = False,
) -> str:
    """Send chat history to configured provider and return response.

    By default answers come from the model's own knowledge, guided by the system
    prompt. With `banidb_only`, answers are built only from BaniDB search results
    (see `ask_with_banidb`). `model` overrides the provider's env-configured model.
    """
    if banidb_only:
        return ask_with_banidb(chat_history, system_prompt=system_prompt, model=model)
    return _ask_provider(
        chat_history,
        system_prompt=system_prompt,
        generation_params=generation_params,
        stream=stream,
        model=model,
    )


def _ask_provider(
    chat_history: List[Dict[str, str]],
    *,
    system_prompt: str | None = None,
    generation_params: Dict | None = None,
    stream: bool = False,
    model: str | None = None,
) -> str:
    """Call the provider chosen by LLM_PROVIDER and return its reply as a string."""
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
    if provider == "anthropic":
        return provider, configured_claude_model()
    return provider, os.getenv("OPENAI_MODEL", "gpt-4o-mini")
