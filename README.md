# GurbaniAI

Streamlit web app to ask about meanings and teachings in Sikh scriptures.

## Setup

1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   - `python -m pip install -r requirements.txt`
3. Choose provider and set env vars.
   - OpenAI (default):
     - `$env:LLM_PROVIDER="openai"`
     - `$env:OPENAI_API_KEY="your_openai_key_here"`
     - Optional model override: `$env:OPENAI_MODEL="gpt-4o-mini"`
   - OpenAI-compatible local server (llama.cpp, vLLM, ...):
     - `$env:LLM_PROVIDER="openai"`
     - `$env:OPENAI_BASE_URL="http://localhost:8080/v1"`
     - `$env:OPENAI_MODEL="your-local-model"` (`OPENAI_API_KEY` is optional here)
     - Enables the top_k, min_p and mirostat Controls, which OpenAI's own API rejects
   - Claude:
     - `$env:LLM_PROVIDER="anthropic"`
     - `$env:ANTHROPIC_API_KEY="your_anthropic_key_here"`
     - Default model is `claude-sonnet-5-5`; optional override: `$env:ANTHROPIC_MODEL="claude-opus-5-5"`
4. Run the agent:
   - **CLI:** `python agent.py`
   - **Web UI:** `python -m streamlit run app.py`

The CLI agent runs in your terminal. The Streamlit app opens in your browser with a chat interface titled **GurbaniAI**.

## Project layout

- `gurbani_agent.py` — shared system prompt and LLM provider logic
- `agent.py` — interactive terminal chat agent
- `app.py` — Streamlit web UI (uses the same agent core)
- `chat_controls.py` — Open WebUI–style **Controls** panel (model params, plus additional instructions added after the master prompt)
- `.cursor/skills/gurbani-ai/` — Cursor skill so agents in this repo adopt the GurbaniAI persona
