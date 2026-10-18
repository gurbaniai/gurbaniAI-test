# AGENTS.md

Guidance for AI coding agents working in this repository. For user-facing setup, see [README.md](README.md).

## Project

GurbaniAI is a Sikh studies assistant: a Streamlit chat app (plus a small CLI) that answers questions about Gurbani using Claude (Anthropic) or OpenAI / OpenAI-compatible models. By default answers come from the model's own knowledge, guided by the master system prompt. An optional "Use BaniDB Only" toggle answers only from BaniDB search results instead.

| File | Role |
|---|---|
| `gurbani_agent.py` | Core: master system prompt, provider calls (`ask_anthropic`, `ask_openai`), BaniDB-only flow (`ask_with_banidb`), `ask_gurbani_llm` entry point |
| `banidb.py` | BaniDB search client (standard library only) and passage formatting |
| `app.py` | Streamlit UI: sidebar (Master Prompt dialog, Claude model picker, BaniDB toggle), chat |
| `agent.py` | Terminal chat; uses `ask_gurbani_llm` with defaults only |
| `.cursor/skills/gurbani-ai/SKILL.md` | GurbaniAI persona for agents answering Gurbani questions (not for coding) |

## Setup and run

- Python 3.10+ (code uses `X | None` syntax). Install: `py -3 -m pip install -r requirements.txt`
- On this Windows machine `python` may resolve to the Microsoft Store stub; use `py -3` if `python` fails.
- Provider is chosen by env vars, read at startup:
  - `LLM_PROVIDER` = `openai` (default) or `anthropic`
  - `ANTHROPIC_API_KEY`, optional `ANTHROPIC_MODEL` (default `DEFAULT_CLAUDE_MODEL` = `claude-opus-5-5`; the owner offers only Opus models, no Sonnet or Haiku; a Haiku `ANTHROPIC_MODEL` is ignored by `configured_claude_model()`)
  - `OPENAI_API_KEY`, optional `OPENAI_MODEL` (default `gpt-4o-mini`)
  - `OPENAI_BASE_URL` for an OpenAI-compatible local server (llama.cpp, vLLM); the API key is then optional
- Web UI: `py -3 -m streamlit run app.py` · CLI: `py -3 agent.py`

## Secrets

- Never write API keys into any file, test, or commit. Keys come only from environment variables.
- `.gitignore` excludes `.env*`, `*.key`, and `__pycache__/`. Keep it that way.
- Error messages shown to users may include exception text; never add the key or request headers to them.

## Architecture rules

**One entry point.** UI and CLI call `ask_gurbani_llm(...)`. It picks the provider from `LLM_PROVIDER` and returns a plain string. Failures are returned as readable strings rather than raised, so the chat never crashes.

**No user-facing model params.** The owner removed the Controls panel; users must not see model params. Requests use defaults. Provider functions still filter what each API accepts, in case params are passed in code:
- Anthropic rejects `frequency_penalty` / `presence_penalty` (filtered via `ANTHROPIC_UNSUPPORTED_PARAMS`) and requires `max_tokens` (default `DEFAULT_CLAUDE_MAX_TOKENS` = 16000; thinking on current Opus models counts against it). Opus 4.7+ and other current Claude models reject `temperature` / `top_p` / `top_k`; `claude_rejects_sampling()` drops them for those models.
- OpenAI reasoning models (`o1`/`o3`/`o4`/`gpt-5` prefixes) need `max_completion_tokens` and reject sampling params; `build_openai_params()` handles this.

**Message dicts stay `{"role", "content"}` only.** Provider APIs reject unknown fields. Keep any per-message UI metadata in separate session state, not in the message dicts.

**Truncation is surfaced, never silent.** Responses that stop at the token limit (`finish_reason == "length"` / `stop_reason == "max_tokens"`) get `TRUNCATION_NOTE` appended. Preserve this in any new response path.

**Model selection.** An explicitly chosen Claude model is used alone (no silent fallback). Only the env/default model falls back through the candidate list. Add selectable models to `CLAUDE_MODELS`.

**Per-session state only.** The user's Master Prompt instructions (`st.session_state.extra_instructions`) and the BaniDB toggle live in Streamlit session state (server memory, per browser tab). The owner asked that nothing go in cookies or browser storage unless absolutely necessary; ask before adding any.

## Knowledge source

- Default mode: answers come only from the model's own knowledge, shaped by the master system prompt. This is a deliberate choice by the project owner.
- BaniDB-only mode (owner approved 2026-10-17, opt-in toggle): `ask_with_banidb` asks the model for single-word search terms (English translation search and Gurmukhi word search), runs them against `api.banidb.com/v2/search`, and has the model answer using only the returned passages. If BaniDB fails or finds nothing, it says so; it never falls back to the model's own knowledge.
- Do not add any other external knowledge source (other databases or APIs, web search, tool calling), and do not use BaniDB outside the toggle. Propose any such change to the owner first.

## Streamlit notes

- The Master Prompt button opens an `@st.dialog`. Its text area saves into `extra_instructions` on Save; don't rely on the widget's own key, since Streamlit drops widget values once the dialog closes.
- The master prompt is shown in a native `title` tooltip on the ⓘ icon (HTML-escaped).
- Don't put text areas (or other widgets that clip overflow) inside a fixed-height `st.container(height=...)` that can overflow: Streamlit's flex layout squashes them to 0px and they vanish.
- Use `width="stretch"`, not the deprecated `use_container_width=True`. Don't nest `st.columns` inside narrow columns.

## Testing

There is no test suite yet. Verify changes without real API keys:
- Compile: `py -3 -m py_compile app.py gurbani_agent.py banidb.py agent.py`
- Provider logic: replace `gurbani_agent.anthropic.Anthropic` or `gurbani_agent.OpenAI` with a fake client that records kwargs and returns `SimpleNamespace` responses, then call `ask_gurbani_llm`. For BaniDB mode, also replace `banidb.search` or let it hit the public API.
- UI: use `streamlit.testing.v1.AppTest.from_file("app.py")`, seed `at.session_state`, run with `timeout=60`, and assert there are no `at.exception` entries.
- Set `PYTHONIOENCODING=utf-8` on Windows when printing Gurmukhi from scripts.
- Put scratch test scripts outside the repo unless you are adding a real test suite.

## Master prompt (critical)

`build_system_prompt()` is the project owner's master prompt and the core of the product. It defines the assistant's voice and approach: respectful, non-sectarian, spiritual and metaphorical meaning prioritized, multiple interpretations offered, no claims of religious authority (no hukamnama), and an explicit note when a verse reference is uncertain.
- Never change its wording, shorten it, or add to it without the owner's explicit approval.
- Every request must carry it, including both BaniDB-mode requests. Keep the strings' spacing intact, since adjacent string literals are concatenated.
- Nothing may replace it. The user's Master Prompt instructions are appended after it by `get_effective_system_prompt()` in `app.py`; BaniDB-only instructions and passages are appended after that.
