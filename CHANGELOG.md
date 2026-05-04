# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.1] — 2026-05-03

### Fixed

- **`USER_AGENT` env var set after `WebBaseLoader` import** — `os.environ.setdefault("USER_AGENT", "ExoAgentApp/1.0")` moved to before the `langchain_community` imports in `exoagent.py`. LangChain checks the variable at import time, so the previous placement always triggered the warning.
- **`master_prompt` unavailable on `ExoModelList`** — the 1.0.0 refactor that decoupled `ExoModelList` from `ExoModel` removed `master_prompt` as a side effect. Restored by adding: default `@llm_function` tools (`call_create_list`, `call_update_list`); `llm_tools` property with reflection + caching; `use_tools` parameter on `run_llm`; `_get_master_prompt` builder; and `master_prompt` public method — mirroring the `ExoModel` pattern.
- **Installation instructions failed in zsh** — `pip install exomodel[extra]` commands in `README.md` now use quotes (`pip install "exomodel[extra]"`). zsh interprets bare square brackets as glob patterns.

### Tests

- **`test_exomodel_list_llm_tools`** — verifies that `call_create_list` and `call_update_list` are discovered by the `llm_tools` property.
- **`test_exomodel_list_master_prompt`** — verifies that `master_prompt` renders the prompt, calls `run_llm` with `mode="orchestrator"` and `use_tools=True`, and passes the return value through unchanged.

---

## [1.0.0] — 2026-05-03

First stable release. All correctness blockers from the beta period have been resolved, the public API surface is frozen, and the test suite has been expanded with unit, integration, and RAG pipeline coverage.

### Fixed — Correctness Bugs

- **`update_object` bypassed Pydantic validation** — replaced the raw `setattr` loop with `model_validate({**self.model_dump(), **filtered})` so all field validators, coercers, and `model_validator` hooks always run on LLM output.
- **Extraction schema marked all fields as required** — fields in the extraction schema are now `Optional[annotation]` with `None` as default. `update_object` filters out `None` values before merging, so unset LLM output fields never overwrite existing instance data.
- **`add_rag_source` did not invalidate the active agent** — `add_rag_source` now forwards the new source to the active `ExoAgent` via `add_rag_sources()` when an agent is already running, ensuring the new source is enqueued and picked up on the next RAG call.
- **`master_prompt.md` received raw Python `StructuredTool` repr as `tools_info`** — `__get_master_prompt` now builds a clean `"- name: description"` string from `self.llm_tools` before template substitution, so the orchestrator receives readable routing hints instead of Python object repr.
- **LLM config params passed in the wrong place** — `temperature` and `max_tokens` are now instance attributes on `ExoAgent`. `_init_agent` passes them directly to `init_chat_model(model_id, temperature=..., max_tokens=...)`. The `run()` config dict now only carries `recursion_limit`.
- **`retrieve_context` closure recreated on every RAG processing** — the retrieval tool is built once in `__init__` as `self._retrieve_context_tool`. `_process_pending_rag` simply assigns `self.rag_tools`. Added a direct-LLM recovery path in `run()` for providers that skip the final text turn after a tool call.
- **`ExoModelList.create_list` handled LLM response type inconsistently** — fixed as part of the `ExoModelList` redesign. `create_list` now checks `isinstance(result, BaseModel)` first (Pydantic path) and falls back to `isinstance(result, dict)`. The old `hasattr(result, 'items')` guard (always `True` for dicts) was removed.

### Fixed — Design Issues

- **`ExoModelList` incorrectly inherited from `ExoModel`** — `ExoModelList` now inherits from `BaseModel, Generic[T]` only. It holds an `ExoAgent` as a private attribute and inlines the three methods it actually needs (`_get_prompt_path`, `_load_prompt_template`, `run_llm`). `update_object`, `master_prompt`, `@llm_function` tools and all other `ExoModel` operations are no longer inherited.
- **`print()` used for all internal observability** — all `print()` calls replaced with `logging.getLogger(__name__)` across `exoagent.py` and `exomodel.py`. Internal traces use `logger.debug`, recoverable anomalies use `logger.warning`, errors use `logger.error`. Downstream apps control verbosity via `logging.getLogger("exomodel").setLevel(...)`.
- **`llm_tools` property rebuilt `StructuredTool` objects on every access** — added `_llm_tools_cache: Optional[List[Any]] = PrivateAttr(default=None)`. The property returns the cached list on subsequent calls and only runs the reflection loop once per instance.
- **Brittle string check for `ExoModelList` containers in extraction schema** — replaced `if "ListExoModel" in base_type.__name__` with `issubclass(base_type, ExoModelList)` via a local import inside the method (avoids circular import).
- **`to_ui` mixed HTML and implicit Markdown output** — added `format` parameter (`"html"` | `"markdown"` | `"plain"`) to `to_ui`, defaulting to `"html"` for backward compatibility. Extracted `_ui_bold` and `_ui_italic` static helpers. HTML output now properly escapes angle brackets; Markdown output uses `**bold**` and `*italic*`.

### Fixed — Missing Features

- **No error handling for RAG source loading failures** — each source load in `_process_pending_rag` is now wrapped in `try/except`. Failed sources are logged with `logger.warning` and skipped; the queue is always cleared so a broken source does not block subsequent RAG calls.
- **No `ExoModel.create` classmethod** — `ExoModel.create(prompt, **initial_values)` added as the canonical way to instantiate and immediately populate a model from a natural-language prompt. `prompt=` in `__init__` now emits a `DeprecationWarning`.
- **No configurable LLM parameters per call** — `ExoAgent.run()` and `ExoModel.run_llm()` now accept `temperature` and `max_tokens` as optional per-call overrides. When provided they override instance defaults and trigger an agent rebuild; when omitted the existing defaults apply unchanged.
- **No token/cost tracking** — added `_usage` dict on `ExoAgent` accumulating `input_tokens`, `output_tokens`, `total_tokens`, and `calls`. `_record_usage()` reads `usage_metadata` from the final `AIMessage` after each invoke. Exposed via `get_usage()` and `reset_usage()`.

### Fixed — Prompt Quality

- **`update_object.md` contradicted the code** — removed the "no markdown code blocks" prohibition from the prompt since the code already strips them via `re.sub`. Prompt now reads: *"Return ONLY a valid raw JSON object with no surrounding text, remarks, or comments."* The stripping fallback remains as a silent safety net.
- **`run_analysis.md` and `filling_instructions.md` hard-failed without RAG** — added `{rag_instruction}` placeholder to both prompt templates. The `_rag_instruction()` helper returns the RAG-grounded instruction when sources are configured, or a `[General Knowledge]` fallback when none are set, so both methods work in RAG-free environments.

### Fixed — Public API Stability

- **`ExoModelList` constructor signature finalized** — both usage patterns are now supported: `ExoModelList(item_class=T)` (positional, preserved for backward compatibility) and `class MyList(ExoModelList[T]): pass` (Pydantic-native, `model_validate` works). `item_class` is now optional; resolved at runtime via `__class_getitem__` when using the subclass pattern.
- **`prompt=` in `ExoModel.__init__` deprecated** — `ExoModel.create(prompt)` is the canonical entry point. Passing `prompt=` to `__init__` now emits a `DeprecationWarning` and will be removed in the next major version.
- **`llm_function` and `llm_action` decorators exported from the top-level package** — both are now in `__all__`. `llm_function` has stable, defined behavior. `llm_action` is exported but its runtime behavior is still being finalized.
- **Internal vs. public method boundary clarified** — `build_extraction_schema` renamed to `_build_extraction_schema` (implementation detail, not part of the public contract). `run_llm` confirmed as public API. The `_` prefix convention is now consistently applied across the codebase; no separate `API.md` is maintained at this stage — the public surface is communicated through docstrings and IDE autocomplete.

### Added — LLM Provider Agnosticism

- **Provider-optional dependency model** — `langchain-google-genai`, `langchain-anthropic`, `langchain-openai`, and `langchain-cohere` are now optional extras. Install only what you use:
  ```
  pip install exomodel[google]      # Gemini (default)
  pip install exomodel[anthropic]   # Claude
  pip install exomodel[openai]      # OpenAI / Azure OpenAI
  pip install exomodel[cohere]      # Cohere
  pip install exomodel[all]         # all providers
  ```
- **Fail-fast provider validation** — `ExoAgent.__init__` checks whether the required provider package is installed at startup and raises a clear `ImportError` with the exact `pip install` command to fix it, rather than failing deep inside a LangChain call.

### Added — Test Coverage

- **Prompt rendering tests** — `tests/integration/test_prompt_rendering.py`: 10 tests verify that every `_load_prompt_template` path renders without `KeyError`. No LLM required; runs in CI by default.
- **Smoke tests** — `tests/integration/test_smoke.py`: 4 end-to-end tests (`pytest.mark.integration`) exercising the full `ExoAgent → LLM → Pydantic` round-trip. Automatically skipped when no API key is present.
- **RAG pipeline tests** — `tests/integration/test_rag_pipeline.py`: 3 integration tests with real embeddings for full retrieval validation, gated by `pytest.mark.integration`.
- **RAG unit tests** — `tests/test_exoagent.py`: 8 mock-based tests covering early-return, text loading, metadata enrichment, source-type detection, failed source handling, all-fail guard, no-KB tool response, and score threshold filtering.
- **`to_ui` edge case tests** — `tests/test_exomodel.py`: 11 tests covering empty string/zero/None → "Not provided"; long strings truncated at 300 chars; HTML escaping of angle brackets; Markdown format skipping escaping; `Field(exclude=True)` fields hidden from output; empty nested list → "Empty list"; nested list count + label rendering; overflow display for lists with more than 5 items.

---

## [0.1.1-beta] — pre-release

Minor bug fixes and README updates.

## [0.1.0-beta] — pre-release

Initial beta release. Core classes (`ExoModel`, `ExoAgent`, `ExoModelList`) established. Basic RAG pipeline, `@llm_function` / `@llm_action` decorators, `to_ui`, `to_csv`, `update_object`, `master_prompt`.
