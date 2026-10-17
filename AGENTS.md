# AGENTS.md

Guidance for AI coding agents working in this repository. For user-facing setup, see [README.md](README.md).

## Project

GurbaniAI is a Sikh studies assistant: a Streamlit chat app (plus a small CLI) that answers questions about Gurbani using Claude (Anthropic) or OpenAI / OpenAI-compatible models. Answers come from the model's own knowledge, guided by the master system prompt.

| File | Role |
|---|---|
| `gurbani_agent.py` | Core: master system prompt, provider calls (`ask_anthropic`, `ask_openai`), `ask_gurbani_llm` entry point |
| `chat_controls.py` | Controls panel (model params, additional instructions) and the mapping from UI params to provider kwargs |
| `app.py` | Streamlit UI: sidebar, Claude model picker, chat, reasoning display |
| `agent.py` | Terminal chat; uses `ask_gurbani_llm` with defaults only |
| `.cursor/skills/gurbani-ai/SKILL.md` | GurbaniAI persona for agents answering Gurbani questions (not for coding) |

## Setup and run

- Python 3.10+ (code uses `X | None` syntax). Install: `py -3 -m pip install -r requirements.txt`
- On this Windows machine `python` may resolve to the Microsoft Store stub; use `py -3` if `python` fails.
- Provider is chosen by env vars, read at startup:
  - `LLM_PROVIDER` = `openai` (default) or `anthropic`
  - `ANTHROPIC_API_KEY`, optional `ANTHROPIC_MODEL` (default `DEFAULT_CLAUDE_MODEL` = `claude-opus-5-5`; the owner offers only Opus models, no Sonnet or Haiku)
  - `OPENAI_API_KEY`, optional `OPENAI_MODEL` (default `gpt-4o-mini`)
  - `OPENAI_BASE_URL` for an OpenAI-compatible local server (llama.cpp, vLLM); the API key is then optional
- Web UI: `py -3 -m streamlit run app.py` · CLI: `py -3 agent.py`

## Secrets

- Never write API keys into any file, test, or commit. Keys come only from environment variables.
- `.gitignore` excludes `.env*`, `*.key`, and `__pycache__/`. Keep it that way.
- Error messages shown to users may include exception text; never add the key or request headers to them.

## Architecture rules

**One entry point.** UI and CLI call `ask_gurbani_llm(...)`. It picks the provider from `LLM_PROVIDER` and returns a plain string. Failures are returned as readable strings rather than raised, so the chat never crashes.

**Params flow:** `st.session_state.chat_params` → `chat_controls.params_for_provider()` → provider function. Only send what the target API accepts:
- Anthropic rejects `frequency_penalty` / `presence_penalty` (filtered via `ANTHROPIC_UNSUPPORTED_PARAMS`) and requires `max_tokens` (default `DEFAULT_CLAUDE_MAX_TOKENS` = 4096). `top_k` is sent only to Claude. Opus 4.7+ and other current Claude models reject `temperature` / `top_p` / `top_k`; `claude_rejects_sampling()` drops them for those models (Opus 4.6 still gets them).
- OpenAI reasoning models (`o1`/`o3`/`o4`/`gpt-5` prefixes) need `max_completion_tokens` and reject sampling params; `build_openai_params()` handles this. `reasoning_effort` is sent only to reasoning models.
- `top_k`, `min_p`, `mirostat*` go via `extra_body` only when `OPENAI_BASE_URL` points at a non-OpenAI server. OpenAI's own API rejects them.
- When adding a control, update `params_for_provider()` and `unsent_params()` so the "Active settings" preview stays truthful about what is sent.

**Message dicts stay `{"role", "content"}` only.** Provider APIs reject unknown fields. Keep any per-message UI metadata in separate session state, not in the message dicts.

**Truncation is surfaced, never silent.** Responses that stop at the token limit (`finish_reason == "length"` / `stop_reason == "max_tokens"`) get `TRUNCATION_NOTE` appended. Preserve this in any new response path.

**Model selection.** An explicitly chosen Claude model is used alone (no silent fallback). Only the env/default model falls back through the candidate list. Add selectable models to `CLAUDE_MODELS`.

## Knowledge source

- Answers come only from the model's own knowledge (Claude via the Anthropic API, or the configured OpenAI model), shaped by the master system prompt. This is a deliberate choice by the project owner.
- Do not add external knowledge sources: no Gurbani databases or APIs (BaniDB was tried and removed), no retrieval/RAG, no web search, no tool calling. Propose any such change to the owner first.
- The Controls "Function Calling" param is kept in the UI but never sent (listed in `unsent_params()`).

## Streamlit notes

- `render_controls_panel` is an `@st.fragment`; it renders before the chat column so widget values are in session state before `chat_input` is handled. Keep that order in `app.py`.
- Widget keys use prefixes (`ctrl_`, `btn_`) that "Reset all to default" relies on. New control widgets should follow them.
- The Controls panel is one bordered box: params in a fixed-height, borderless scroll area, then the Additional Instructions box, then Reset. The owner removed the Valves menu; don't bring it back.
- Don't put text areas (or other widgets that clip overflow) inside a fixed-height `st.container(height=...)` that can overflow: Streamlit's flex layout squashes them to 0px and they vanish. Keep them outside, as the Additional Instructions box is.
- Use `width="stretch"`, not the deprecated `use_container_width=True`. Don't nest `st.columns` inside narrow columns; the Controls button once collapsed to a vertical strip that way.

## Testing

There is no test suite yet. Verify changes without real API keys:
- Compile: `py -3 -m py_compile app.py gurbani_agent.py chat_controls.py agent.py`
- Provider logic: replace `gurbani_agent.anthropic.Anthropic` or `gurbani_agent.OpenAI` with a fake client that records kwargs and returns `SimpleNamespace` responses, then call `ask_gurbani_llm`.
- UI: use `streamlit.testing.v1.AppTest.from_file("app.py")`, seed `at.session_state`, run, and assert there are no `at.exception` entries.
- Set `PYTHONIOENCODING=utf-8` on Windows when printing Gurmukhi from scripts.
- Put scratch test scripts outside the repo unless you are adding a real test suite.

## Master prompt (critical)

`build_system_prompt()` is the project owner's master prompt and the core of the product. It defines the assistant's voice and approach: respectful, non-sectarian, spiritual and metaphorical meaning prioritized, multiple interpretations offered, no claims of religious authority (no hukamnama), and an explicit note when a verse reference is uncertain.
- Never change its wording, shorten it, or add to it without the owner's explicit approval.
- Every request must carry it. Keep the strings' spacing intact, since adjacent string literals are concatenated.
- Nothing may replace it. The Controls "Additional Instructions" box (`st.session_state.extra_instructions`) is appended after it by `get_effective_system_prompt()`.
