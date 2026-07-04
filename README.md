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
   - Claude:
     - `$env:LLM_PROVIDER="anthropic"`
     - `$env:ANTHROPIC_API_KEY="your_anthropic_key_here"`
     - Optional model override: `$env:ANTHROPIC_MODEL="claude-haiku-4-5"`
4. Run the agent:
   - **CLI:** `python agent.py`
   - **Web UI:** `python -m streamlit run app.py`

The CLI agent runs in your terminal. The Streamlit app opens in your browser with a chat interface titled **GurbaniAI**.

## Project layout

- `gurbani_agent.py` — shared system prompt and LLM provider logic
- `agent.py` — interactive terminal chat agent
- `app.py` — Streamlit web UI (uses the same agent core)
- `.cursor/skills/gurbani-ai/` — Cursor skill so agents in this repo adopt the GurbaniAI persona
