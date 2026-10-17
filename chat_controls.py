"""Open WebUI–style chat controls panel (model params and additional instructions)."""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st

from gurbani_agent import CLAUDE_SAMPLING_PARAMS, claude_rejects_sampling, openai_compatible_base_url

DEFAULT_CHAT_PARAMS: Dict[str, Any] = {
    "stream_response": None,
    "function_calling": None,
    "reasoning_tags": None,
    "seed": None,
    "stop": None,
    "temperature": None,
    "reasoning_effort": None,
    "logit_bias": None,
    "max_tokens": None,
    "top_k": None,
    "top_p": None,
    "min_p": None,
    "frequency_penalty": None,
    "presence_penalty": None,
    "mirostat": None,
    "mirostat_eta": None,
    "mirostat_tau": None,
}

# llama.cpp-style sampling params. OpenAI's own API rejects them, so they are only
# sent (via extra_body) when OPENAI_BASE_URL points at a compatible server.
LOCAL_SAMPLING_PARAMS = ("top_k", "min_p", "mirostat", "mirostat_eta", "mirostat_tau")

# Matches <think>…</think>, <thinking>…</thinking> and <reasoning>…</reasoning> blocks.
_REASONING_BLOCK = re.compile(r"<(think|thinking|reasoning)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)


def unsent_params(provider: str, model: Optional[str] = None) -> frozenset:
    """Controls that are stored in the UI but not forwarded to the active provider/model."""
    # Answers come from the model's own knowledge; there are no tools to call.
    unsent = {"function_calling"}
    if provider == "anthropic":
        unsent.update({"min_p", "mirostat", "mirostat_eta", "mirostat_tau"})
        if model and claude_rejects_sampling(model):
            unsent.update(CLAUDE_SAMPLING_PARAMS)
    elif not openai_compatible_base_url():
        unsent.update(LOCAL_SAMPLING_PARAMS)
    return frozenset(unsent)


def split_reasoning(text: str) -> Tuple[str, str]:
    """Separate reasoning-tag blocks from the answer; returns (reasoning, answer)."""
    parts = [m.group(2).strip() for m in _REASONING_BLOCK.finditer(text)]
    answer = _REASONING_BLOCK.sub("", text)
    # Some local reasoning models omit the opening tag and emit only "</think>".
    head, sep, tail = answer.partition("</think>")
    if sep:
        parts.insert(0, head.strip())
        answer = tail
    return "\n\n".join(p for p in parts if p), answer.strip()


def init_controls_state() -> None:
    if "controls_open" not in st.session_state:
        st.session_state.controls_open = False
    if "chat_params" not in st.session_state:
        st.session_state.chat_params = copy.deepcopy(DEFAULT_CHAT_PARAMS)
    if "extra_instructions" not in st.session_state:
        st.session_state.extra_instructions = ""


def _toggle_param(key: str, enabled_value: Any) -> None:
    current = st.session_state.chat_params.get(key)
    if current is None:
        st.session_state.chat_params[key] = enabled_value
    else:
        st.session_state.chat_params[key] = None
        widget_key = f"ctrl_{key}"
        if widget_key in st.session_state:
            del st.session_state[widget_key]


def _param_mode_button(key: str, enabled_value: Any, *, button_key: str) -> None:
    current = st.session_state.chat_params.get(key)
    label = "Default" if current is None else "Custom"
    if st.button(label, key=button_key, width="stretch"):
        _toggle_param(key, enabled_value)
        st.rerun()


def get_effective_system_prompt(default_prompt: str) -> str:
    """Master prompt first, always; the user's additional instructions follow it."""
    extra = (st.session_state.get("extra_instructions") or "").strip()
    if extra:
        return default_prompt + "\n\nAdditional instructions:\n" + extra
    return default_prompt


def params_for_provider(provider: str, params: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
    """Map stored controls to provider SDK kwargs (only supported fields)."""
    out: Dict[str, Any] = {}
    if params.get("temperature") is not None:
        out["temperature"] = float(params["temperature"])
    if params.get("max_tokens") is not None:
        out["max_tokens"] = int(params["max_tokens"])
    if params.get("top_p") is not None:
        out["top_p"] = float(params["top_p"])
    if params.get("frequency_penalty") is not None:
        out["frequency_penalty"] = float(params["frequency_penalty"])
    if params.get("presence_penalty") is not None:
        out["presence_penalty"] = float(params["presence_penalty"])
    if params.get("seed") is not None and provider == "openai":
        out["seed"] = int(params["seed"])
    stop = params.get("stop")
    if stop is not None and str(stop).strip():
        sequences = [s.strip() for s in str(stop).split(",") if s.strip()]
        if sequences:
            if provider == "anthropic":
                out["stop_sequences"] = sequences
            else:
                out["stop"] = sequences
    if provider == "openai" and params.get("reasoning_effort") is not None:
        out["reasoning_effort"] = params["reasoning_effort"]
    if provider == "openai" and params.get("logit_bias") is not None:
        raw = str(params["logit_bias"]).strip()
        if raw:
            try:
                out["logit_bias"] = json.loads(raw)
            except json.JSONDecodeError:
                pass
    if provider == "anthropic" and params.get("top_k") is not None:
        out["top_k"] = int(params["top_k"])
    if provider == "openai" and openai_compatible_base_url():
        extra = {k: params[k] for k in LOCAL_SAMPLING_PARAMS if params.get(k) is not None}
        if extra:
            # extra_body merges these into the JSON request body as-is.
            out["extra_body"] = extra
    for key in unsent_params(provider, model):
        out.pop(key, None)
    return out


def should_stream(params: Dict[str, Any]) -> bool:
    sr = params.get("stream_response")
    if sr is None:
        return False
    return bool(sr)


def build_request_preview(
    provider: str, default_prompt: str, model: Optional[str] = None
) -> Tuple[str, Dict[str, Any], bool, List[str]]:
    """Return settings used for the *next* chat completion."""
    system_prompt = get_effective_system_prompt(default_prompt)
    gen_params = params_for_provider(provider, st.session_state.chat_params, model)
    stream = should_stream(st.session_state.chat_params)
    notes: List[str] = []
    skipped = unsent_params(provider, model)
    target = f"{provider} ({model})" if model else provider
    for key, val in st.session_state.chat_params.items():
        if val is not None and key in skipped:
            notes.append(f"{key} is set in UI but not sent to {target}.")
    if st.session_state.chat_params.get("reasoning_tags"):
        notes.append("reasoning_tags: `<think>` blocks in replies are shown in a collapsible Reasoning section.")
    return system_prompt, gen_params, stream, notes


def format_active_settings(provider: str, default_prompt: str, model: Optional[str] = None) -> str:
    system_prompt, gen_params, stream, notes = build_request_preview(provider, default_prompt, model)
    lines = [
        f"**Next message uses:** stream={'on' if stream else 'off'}",
    ]
    if gen_params:
        formatted = ", ".join(f"`{k}={v}`" for k, v in gen_params.items())
        lines.append(f"Params: {formatted}")
    else:
        lines.append("Params: provider defaults")
    lines.append("System prompt: GurbaniAI master prompt (always sent)")
    extra = (st.session_state.get("extra_instructions") or "").strip()
    if extra:
        extra_preview = extra.replace("\n", " ")
        if len(extra_preview) > 120:
            extra_preview = extra_preview[:117] + "..."
        lines.append(f"+ Additional instructions: _{extra_preview}_")
    for note in notes:
        lines.append(f"_{note}_")
    return "\n\n".join(lines)


def _bind_slider(key: str, label: str, min_v: float, max_v: float, step: float) -> None:
    if st.session_state.chat_params.get(key) is None:
        return
    widget_key = f"ctrl_{key}"
    default = float(st.session_state.chat_params[key])
    st.session_state.chat_params[key] = st.slider(
        label,
        min_value=min_v,
        max_value=max_v,
        value=default,
        step=step,
        key=widget_key,
        label_visibility="collapsed",
    )


def _bind_int(key: str, label: str, min_v: int, max_v: int) -> None:
    if st.session_state.chat_params.get(key) is None:
        return
    widget_key = f"ctrl_{key}"
    st.session_state.chat_params[key] = st.number_input(
        label,
        min_value=min_v,
        max_value=max_v,
        value=int(st.session_state.chat_params[key]),
        key=widget_key,
        label_visibility="collapsed",
    )


def _bind_text(key: str, label: str, placeholder: str = "") -> None:
    if st.session_state.chat_params.get(key) is None:
        return
    widget_key = f"ctrl_{key}"
    st.session_state.chat_params[key] = st.text_input(
        label,
        value=str(st.session_state.chat_params[key]),
        placeholder=placeholder,
        key=widget_key,
        label_visibility="collapsed",
    )


def _render_param_row(
    key: str,
    label: str,
    enabled_value: Any,
    editor: Optional[Callable[[], None]] = None,
) -> None:
    left, right = st.columns([3, 2], gap="small", vertical_alignment="center")
    with left:
        st.markdown(label)
    with right:
        _param_mode_button(key, enabled_value, button_key=f"btn_{key}")
    if editor and st.session_state.chat_params.get(key) is not None:
        editor()


def _render_advanced_params() -> None:
    _render_param_row("stream_response", "Stream Chat Response", True)
    if st.session_state.chat_params.get("stream_response") is not None:
        st.session_state.chat_params["stream_response"] = st.toggle(
            "Stream responses",
            value=bool(st.session_state.chat_params["stream_response"]),
            key="ctrl_stream_response",
        )

    _render_param_row("function_calling", "Function Calling", "native")
    if st.session_state.chat_params.get("function_calling") is not None:
        st.session_state.chat_params["function_calling"] = st.selectbox(
            "Mode",
            ["native", "legacy"],
            index=0 if st.session_state.chat_params["function_calling"] == "native" else 1,
            key="ctrl_function_calling",
            label_visibility="collapsed",
        )

    _render_param_row("reasoning_tags", "Reasoning Tags", True)
    if st.session_state.chat_params.get("reasoning_tags") is not None:
        st.session_state.chat_params["reasoning_tags"] = st.toggle(
            "Reasoning tags",
            value=bool(st.session_state.chat_params["reasoning_tags"]),
            key="ctrl_reasoning_tags",
            label_visibility="collapsed",
        )

    _render_param_row("seed", "Seed", 0, lambda: _bind_int("seed", "Seed", 0, 2_147_483_647))
    _render_param_row("stop", "Stop Sequence", "", lambda: _bind_text("stop", "Stop", "word1, word2"))
    _render_param_row("temperature", "Temperature", 0.8, lambda: _bind_slider("temperature", "Temperature", 0.0, 2.0, 0.05))
    _render_param_row("reasoning_effort", "Reasoning Effort", "medium")
    if st.session_state.chat_params.get("reasoning_effort") is not None:
        efforts = ["low", "medium", "high"]
        idx = (
            efforts.index(st.session_state.chat_params["reasoning_effort"])
            if st.session_state.chat_params["reasoning_effort"] in efforts
            else 1
        )
        st.session_state.chat_params["reasoning_effort"] = st.selectbox(
            "Effort",
            efforts,
            index=idx,
            key="ctrl_reasoning_effort",
            label_visibility="collapsed",
        )

    _render_param_row("logit_bias", "logit_bias", "{}", lambda: _bind_text("logit_bias", "logit_bias", '{"50256": -100}'))
    _render_param_row("max_tokens", "max_tokens", 4096, lambda: _bind_int("max_tokens", "max_tokens", 1, 8192))
    _render_param_row("top_k", "top_k", 40, lambda: _bind_int("top_k", "top_k", 0, 1000))
    _render_param_row("top_p", "top_p", 0.9, lambda: _bind_slider("top_p", "top_p", 0.0, 1.0, 0.05))
    _render_param_row("min_p", "min_p", 0.0, lambda: _bind_slider("min_p", "min_p", 0.0, 1.0, 0.05))
    _render_param_row(
        "frequency_penalty",
        "frequency_penalty",
        0.0,
        lambda: _bind_slider("frequency_penalty", "frequency_penalty", -2.0, 2.0, 0.05),
    )
    _render_param_row(
        "presence_penalty",
        "presence_penalty",
        0.0,
        lambda: _bind_slider("presence_penalty", "presence_penalty", -2.0, 2.0, 0.05),
    )
    _render_param_row("mirostat", "mirostat", 0, lambda: _bind_int("mirostat", "mirostat", 0, 2))
    _render_param_row("mirostat_eta", "mirostat_eta", 0.1, lambda: _bind_slider("mirostat_eta", "mirostat_eta", 0.0, 1.0, 0.05))
    _render_param_row("mirostat_tau", "mirostat_tau", 5.0, lambda: _bind_slider("mirostat_tau", "mirostat_tau", 0.0, 10.0, 0.5))


@st.fragment
def render_controls_panel() -> None:
    """Right-side controls panel; fragment rerun keeps the chat column stable."""
    with st.container(border=True):
        head_left, head_right = st.columns([5, 1], vertical_alignment="center")
        with head_left:
            st.subheader("Controls", anchor=False)
        with head_right:
            if st.button("✕", key="close_controls", help="Close controls", type="tertiary"):
                st.session_state.controls_open = False
                st.rerun()

        # Model params are the main content; the scroll area has no border of its own
        # so the panel reads as a single box.
        with st.container(height=440, border=False):
            _render_advanced_params()

        # Keep this outside the fixed-height scroll area: Streamlit lays that area out as a
        # flex column and squashes overflowing widgets like text areas to 0px height.
        # The text lives in `extra_instructions`, not in the widget's own key: Streamlit drops
        # a widget's value whenever a run doesn't draw it (panel closed, or a param button's
        # st.rerun() before this point), which would silently erase the instructions.
        st.text_area(
            "Additional instructions",
            value=st.session_state.extra_instructions,
            placeholder="Optional. Added after the GurbaniAI master prompt, which is always sent.",
            height=96,
            key="ctrl_extra_instructions",
            on_change=_save_extra_instructions,
        )

        # A callback runs before widgets are drawn, which is the only time Streamlit
        # allows resetting a widget's value (the Additional Instructions box).
        st.button("Reset all to default", width="stretch", key="reset_controls", on_click=_reset_controls)


def _save_extra_instructions() -> None:
    st.session_state.extra_instructions = st.session_state.ctrl_extra_instructions


def _reset_controls() -> None:
    st.session_state.chat_params = copy.deepcopy(DEFAULT_CHAT_PARAMS)
    st.session_state.extra_instructions = ""
    for k in list(st.session_state.keys()):
        if str(k).startswith(("ctrl_", "btn_")):
            del st.session_state[k]


def render_controls_toolbar() -> None:
    """Header button that opens the panel; fills its column so the label never wraps."""
    if st.button("Controls", key="open_controls_toolbar", width="stretch"):
        st.session_state.controls_open = True
        st.rerun()
