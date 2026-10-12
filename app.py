import streamlit as st

from chat_controls import (
    build_request_preview,
    format_active_settings,
    init_controls_state,
    render_controls_panel,
    render_controls_toolbar,
    split_reasoning,
)
from gurbani_agent import (
    CLAUDE_MODELS,
    ask_gurbani_llm,
    build_system_prompt,
    get_provider_info,
)

st.set_page_config(page_title="GurbaniAI", page_icon="🪯", layout="wide")

init_controls_state()

if "messages" not in st.session_state:
    st.session_state.messages = []
provider, current_model = get_provider_info()
default_prompt = build_system_prompt()


def render_model_picker() -> str:
    """Let the user choose a Claude model; returns the model for the next message."""
    # Keep the env-configured model first so a custom ANTHROPIC_MODEL is always selectable.
    options = list(dict.fromkeys([current_model, *CLAUDE_MODELS]))
    return st.selectbox(
        "Claude model",
        options=options,
        index=0,
        key="claude_model",
        help="Used for the next message. Your API key must have access to the model.",
    )


def render_sidebar() -> str:
    """Render the sidebar and return the model to use for the next message."""
    active_model = current_model
    with st.sidebar:
        st.subheader("How to use")
        st.markdown(
            "- Ask about a shabad, line, or theme.\n"
            "- Open **Controls** to add extra instructions and tune model params.\n"
            "- Changes apply to the **next** message you send (see preview below the title)."
        )
        st.divider()
        st.caption(f"Provider: `{provider}`")
        if provider == "anthropic":
            active_model = render_model_picker()
        else:
            st.caption(f"Model: `{current_model}`")
        st.divider()
        if st.button("Controls", help="Open parameter fine-tuning panel"):
            st.session_state.controls_open = True
            st.rerun()
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()
    return active_model


def render_message(role: str, content: str) -> None:
    """Render a chat message; with Reasoning Tags on, <think> blocks collapse into an expander."""
    if role == "assistant" and st.session_state.chat_params.get("reasoning_tags"):
        reasoning, answer = split_reasoning(content)
        if reasoning:
            with st.expander("Reasoning"):
                st.markdown(reasoning)
        st.markdown(answer)
    else:
        st.markdown(content)


def render_chat_column(active_model: str) -> None:
    header_left, header_right = st.columns([4, 1], vertical_alignment="center")
    with header_left:
        st.title("🪯 GurbaniAI")
        st.caption("Ask about meanings and teachings in Sikh scriptures.")
    with header_right:
        if not st.session_state.controls_open:
            render_controls_toolbar()

    with st.expander("Active settings for next message", expanded=st.session_state.controls_open):
        st.markdown(format_active_settings(provider, default_prompt))
        if provider == "anthropic":
            st.caption(f"Claude model: `{active_model}`")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            render_message(message["role"], message["content"])

    user_prompt = st.chat_input("Ask about Gurbani meaning, context, or life application...")
    if user_prompt:
        system_prompt, gen_params, use_stream, _notes = build_request_preview(provider, default_prompt)

        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            meta = f"_Using: {gen_params or 'provider defaults'}_"
            with st.spinner("Reflecting on Gurbani..."):
                answer = ask_gurbani_llm(
                    st.session_state.messages,
                    system_prompt=system_prompt,
                    generation_params=gen_params,
                    stream=use_stream,
                    model=active_model,
                )
            render_message("assistant", answer)
            if gen_params or st.session_state.get("extra_instructions"):
                st.caption(meta)

        st.session_state.messages.append({"role": "assistant", "content": answer})


active_model = render_sidebar()

if st.session_state.controls_open:
    chat_col, controls_col = st.columns([3, 2], gap="large")
    # Run controls first so widget values are in session_state before chat_input is handled.
    with controls_col:
        render_controls_panel()
    with chat_col:
        render_chat_column(active_model)
else:
    render_chat_column(active_model)
