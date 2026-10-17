import html

import streamlit as st

from gurbani_agent import (
    CLAUDE_MODELS,
    ask_gurbani_llm,
    build_system_prompt,
    get_provider_info,
)

st.set_page_config(page_title="GurbaniAI", page_icon="🪯", layout="wide")

# Session state lives on the server for this browser tab only; nothing is stored
# in cookies or browser storage, and a page refresh clears it.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "extra_instructions" not in st.session_state:
    st.session_state.extra_instructions = ""

provider, current_model = get_provider_info()
default_prompt = build_system_prompt()


def get_effective_system_prompt() -> str:
    """Master prompt first, always; the user's instructions for this session follow it."""
    extra = (st.session_state.get("extra_instructions") or "").strip()
    if extra:
        return default_prompt + "\n\nAdditional instructions:\n" + extra
    return default_prompt


@st.dialog("Master Prompt", width="large")
def master_prompt_dialog() -> None:
    """Let the user add their own instructions after the master prompt, for this session."""
    # A native title tooltip shows the full master prompt when hovering the info icon.
    st.markdown(
        f'**Master prompt** <span title="{html.escape(default_prompt, quote=True)}" '
        'style="cursor: help;">ⓘ</span>',
        unsafe_allow_html=True,
    )
    st.caption("The master prompt is always sent first. Hover the ⓘ icon to read it.")
    # The text is kept in `extra_instructions`, not the widget key: Streamlit drops a
    # widget's value once the dialog closes and the widget is no longer drawn.
    text = st.text_area(
        "Instructions",
        value=st.session_state.extra_instructions,
        placeholder="You can add your own prompt instructions on top of the master prompt.",
        help="You can add your own prompt instructions on top of the master prompt. "
        "They apply to this session only and are cleared when the page is refreshed.",
        height=180,
        key="mp_instructions_input",
    )
    save_col, clear_col = st.columns(2)
    if save_col.button("Save", type="primary", width="stretch"):
        st.session_state.extra_instructions = text.strip()
        st.rerun()
    if clear_col.button("Clear", width="stretch"):
        st.session_state.extra_instructions = ""
        st.rerun()


def render_model_picker() -> str:
    """Let the user choose a Claude model; returns the model for the next message."""
    # Keep CLAUDE_MODELS' newest-to-oldest order; a custom ANTHROPIC_MODEL not in the
    # list is added at the end so it stays selectable. The env/default model is preselected.
    options = list(dict.fromkeys([*CLAUDE_MODELS, current_model]))
    return st.selectbox(
        "Claude model",
        options=options,
        index=options.index(current_model),
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
            "- Use **Master Prompt** to add your own instructions for this session.\n"
            "- Switch on **Use BaniDB Only** to answer only from BaniDB search results."
        )
        st.divider()
        if st.button("Master Prompt", width="stretch"):
            master_prompt_dialog()
        if st.session_state.extra_instructions:
            st.caption("Your instructions are active for this session.")
        st.divider()
        st.caption(f"Provider: `{provider}`")
        if provider == "anthropic":
            active_model = render_model_picker()
        else:
            st.caption(f"Model: `{current_model}`")
        st.toggle(
            "Use BaniDB Only",
            key="banidb_only",
            help="Answers use only lines found in BaniDB. The model picks search words and "
            "explains the lines, but does not quote Gurbani from its own knowledge.",
        )
        st.divider()
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()
    return active_model


def render_chat_column(active_model: str) -> None:
    st.title("🪯 GurbaniAI")
    st.caption("Ask about meanings and teachings in Sikh scriptures.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Ask about Gurbani meaning, context, or life application...")
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        banidb_only = st.session_state.get("banidb_only", False)
        with st.chat_message("assistant"):
            spinner = "Searching BaniDB..." if banidb_only else "Reflecting on Gurbani..."
            with st.spinner(spinner):
                answer = ask_gurbani_llm(
                    st.session_state.messages,
                    system_prompt=get_effective_system_prompt(),
                    model=active_model,
                    banidb_only=banidb_only,
                )
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


render_chat_column(render_sidebar())
