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
   - Claude:
     - `$env:LLM_PROVIDER="anthropic"`
     - `$env:ANTHROPIC_API_KEY="your_anthropic_key_here"`
     - Default model is `claude-opus-5-5`; optional override: `$env:ANTHROPIC_MODEL="claude-opus-4-8"`
4. Run the agent:
   - **CLI:** `python agent.py`
   - **Web UI:** `python -m streamlit run app.py`

The CLI agent runs in your terminal. The Streamlit app opens in your browser with a chat interface titled **GurbaniAI**.

In the web UI sidebar:
- **Master Prompt** opens a popup where you can add your own instructions on top of the master prompt (hover the ⓘ icon to read it). They apply to your current session only and are not saved in the browser; refreshing the page clears them.
- **Use BaniDB Only** answers only from lines found by searching [BaniDB](https://www.banidb.com/). The model picks search words and explains the lines it gets back, but does not quote Gurbani from its own knowledge.

## Project layout

- `gurbani_agent.py` — shared system prompt, LLM provider logic, and the BaniDB-only answer flow
- `banidb.py` — BaniDB search client
- `agent.py` — interactive terminal chat agent
- `app.py` — Streamlit web UI (uses the same agent core)
- `.cursor/skills/gurbani-ai/` — Cursor skill so agents in this repo adopt the GurbaniAI persona
