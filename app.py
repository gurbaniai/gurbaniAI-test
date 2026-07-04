import streamlit as st

from gurbani_agent import ask_gurbani_llm, get_provider_info

st.set_page_config(page_title="GurbaniAI", page_icon="🪯", layout="centered")

st.title("🪯 GurbaniAI")
st.caption("Ask about meanings and teachings in Sikh scriptures.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("How to use")
    st.markdown(
        "- Ask about a shabad, line, or theme.\n"
        "- You can include transliteration or English translation.\n"
        "- For exact quotes, verify with trusted Sikh scripture references."
    )
    st.divider()
    provider, current_model = get_provider_info()
    st.caption(f"Provider: `{provider}`")
    st.caption(f"Model: `{current_model}`")
    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_prompt = st.chat_input("Ask about Gurbani meaning, context, or life application...")
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Reflecting on Gurbani..."):
            answer = ask_gurbani_llm(st.session_state.messages)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
