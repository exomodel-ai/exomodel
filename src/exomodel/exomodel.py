# Copyright 2026 Leandro Pessoa
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import io
import json
import logging
import os
import re
from typing import Any, Optional, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, PrivateAttr, create_model
from pydantic_core import PydanticUndefined

from .exoagent import ExoAgent

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='ExoModel')

def llm_action(func):
    """Tags a method as internally LLM-driven (introspection marker, not yet used by the runtime)."""
    func._is_llm_action = True
    return func

def llm_function(func):
    """Exposes a method as a LangChain tool available to the orchestrator in `master_prompt`."""
    func._is_llm_function = True
    return func

class ExoModel(BaseModel):
    """Base class that turns a Pydantic model into an LLM-powered entity.

    Subclass this and define fields normally. The class gains natural-language
    CRUD (`update_object`, `update_field`, `create`), self-analysis
    (`run_analysis`, `run_filling_instructions`), free-form reasoning
    (`run_object_prompt`), and an orchestrator that routes to any
    `@llm_function`-decorated method (`master_prompt`).

    RAG sources are declared at the class level via `get_rag_sources()`. All LLM
    calls are delegated to a lazily-created `ExoAgent` instance stored in
    `_exo_agent`.
    """
    _rag_sources: list[str] = PrivateAttr(default_factory=list)
    _exo_agent: Optional[ExoAgent] = PrivateAttr(default=None)
    _llm_tools_cache: Optional[list[Any]] = PrivateAttr(default=None)

    @llm_function
    def call_update_object(self, prompt: str):
        """
        Use this tool STRICTLY when the user explicitly commands to change, modify,
        or update the entity's fields based on an instruction.
        MANDATORY: Pass the user's update instruction as the 'prompt' argument.
        """
        logger.debug("AI tool call: call_update_object")
        return self.update_object(prompt)

    @llm_function
    def call_run_object_prompt(self, prompt: str):
        """
        Use this tool when the user asks to brainstorm, improve, give options, rewrite content,
        or answer complex questions based on the entity's data.
        MANDATORY: You must pass the user's original request as the 'prompt' argument.
        """
        logger.debug("AI tool call: call_run_object_prompt")
        return self.run_object_prompt(prompt)

    @llm_function
    def call_run_object_analysis(self):
        """
        Use this tool when the user asks for a critical analysis, evaluation, or deep review
        of the entity's current state and strategy.
        This tool requires NO arguments. Just invoke it.
        """
        logger.debug("AI tool call: call_run_object_analysis")
        return self.run_analysis()

    @llm_function
    def call_run_filling_instructions(self):
        """
        Use this tool when the user asks for guidance, best practices, rules,
        or instructions on how to fill out, complete, or improve the fields
        of this entity based on the reference material.
        This tool requires NO arguments. Just invoke it.
        """
        logger.debug("AI tool call: call_run_filling_instructions")
        return self.run_filling_instructions()

    @classmethod
    def get_rag_sources(cls) -> list[str]:
        """Override this in child classes to return knowledge files/URLs."""
        return []

    @property
    def llm_tools(self):
        """Returns all `@llm_function` methods on this instance as LangChain `StructuredTool` objects.

        Result is cached on first access — reflection only runs once per instance.
        """
        if self._llm_tools_cache is not None:
            return self._llm_tools_cache

        from langchain_core.tools import StructuredTool
        tools = []
        for attr_name in dir(type(self)):
            if attr_name.startswith('_') or attr_name.startswith('model_'):
                continue
            try:
                method = getattr(self, attr_name)
                if hasattr(method, "_is_llm_function"):
                    tools.append(StructuredTool.from_function(
                        func=method,
                        name=attr_name,
                        description=method.__doc__ or f"Executes {attr_name}",
                        return_direct=True
                    ))
            except Exception:
                continue

        self._llm_tools_cache = tools
        return self._llm_tools_cache

    def __init__(self, **data):
        import warnings
        prompt = data.pop("prompt", None)
        if prompt is not None:
            warnings.warn(
                "Passing prompt= to __init__ is deprecated and will be removed in 1.0.0. "
                "Use MyModel.create(prompt) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        super().__init__(**data)

        self._rag_sources = self.get_rag_sources()
        if self._rag_sources:
            logger.debug("Knowledge sources loaded: %s", self._rag_sources)

        if prompt:
            self.update_object(prompt)

    @classmethod
    def create(cls, prompt: str, **initial_values) -> "ExoModel":
        """Creates and populates a new instance entirely via LLM from a natural-language prompt.

        Args:
            prompt: Natural-language instruction used to populate the instance fields.
            **initial_values: Optional field values to set before the LLM call.

        Returns:
            A fully populated instance of the calling class.

        Example:
            proposal = Proposal.create("RetailFlow needs an AI strategy roadmap")
        """
        instance = cls(**initial_values)
        instance.update_object(prompt)
        return instance

    def add_rag_source(self, rag_source: str):
        """Appends a RAG source (URL, PDF path, or text file path) to this instance.

        Deduplicates silently. If an agent is already running, the source is
        forwarded immediately so the next RAG call picks it up without a restart.
        """
        if rag_source not in self._rag_sources:
            self._rag_sources.append(rag_source)
            if self._exo_agent is not None:
                self._exo_agent.add_rag_sources([rag_source])

    def get_json_schema(self):
        """Returns the Pydantic JSON schema."""
        return json.dumps(self.model_json_schema(), indent=2)

    def get_instance_json(self):
        """Returns the current instance data as JSON."""
        return self.model_dump_json(indent=2)

    def _get_prompt_path(self, filename: str) -> str:
        """Resolves the absolute path for a given prompt template file."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "prompt", filename)

    def _load_prompt_template(self, filename: str, **kwargs) -> str:
        """
        Loads a prompt template file and renders it with the given keyword arguments.

        Raises:
            FileNotFoundError: If the template file does not exist. This is an
                unrecoverable configuration error — passing the error message as
                a prompt to the LLM would produce nonsensical results.
            KeyError: If a required placeholder in the template is missing from kwargs.
        """
        file_path = self._get_prompt_path(filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Prompt template not found: '{file_path}'. "
                f"Ensure the 'prompt/' directory exists alongside exomodel.py "
                f"and contains '{filename}'."
            )

        with open(file_path, encoding='utf-8') as f:
            template = f.read()

        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise KeyError(
                f"Template '{filename}' requires placeholder {e} "
                f"which was not provided. Supplied keys: {list(kwargs.keys())}"
            ) from e

    # -------------------------------------------------------------------------
    # Private prompt builders — now use _load_prompt_template
    # -------------------------------------------------------------------------

    def __get_prompt_update_object(self, prompt: str) -> str:
        return self._load_prompt_template(
            "update_object.md",
            entity_name=self.__class__.__name__,
            prompt=prompt,
            obj_fields_info=self.get_fields_info()
        )

    def __get_prompt_update_field(self, field_name: str, prompt: str) -> str:
        return self._load_prompt_template(
            "update_field.md",
            field_name=field_name,
            entity_name=self.__class__.__name__,
            field_value=getattr(self, field_name),
            prompt=prompt
        )

    def __get_prompt_run_object_prompt(self, prompt: str) -> str:
        return self._load_prompt_template(
            "run_object_prompt.md",
            prompt=prompt,
            entity_name=self.__class__.__name__,
            json_schema=self.get_instance_json()
        )

    def _rag_instruction(self) -> str:
        """Returns the RAG grounding instruction injected into analysis/filling prompts."""
        if self._rag_sources:
            return (
                "Ground your analysis in the knowledge retrieved through the `retrieve_context` "
                "function. Use those definitions and standards as your source of truth."
            )
        return (
            "No knowledge base is configured. Use your domain expertise and clearly mark "
            "all conclusions as [General Knowledge]."
        )

    def __get_prompt_run_analysis(self) -> str:
        return self._load_prompt_template(
            "run_analysis.md",
            entity_name=self.__class__.__name__,
            json_schema=self.get_instance_json(),
            rag_instruction=self._rag_instruction()
        )

    def __get_prompt_filling_instructions(self) -> str:
        return self._load_prompt_template(
            "filling_instructions.md",
            entity_name=self.__class__.__name__,
            fields_info=self.get_fields_metadata(self.__class__),
            rag_instruction=self._rag_instruction()
        )

    def __get_master_prompt(self, prompt: str) -> str:
        tools_info = "\n".join(
            f"- {t.name}: {t.description}" for t in self.llm_tools
        )
        return self._load_prompt_template(
            "master_prompt.md",
            entity_name=self.__class__.__name__,
            prompt=prompt,
            obj_fields_info=self.get_fields_info(),
            tools_info=tools_info
        )

    # -------------------------------------------------------------------------
    # Public operations
    # -------------------------------------------------------------------------

    def update_object(self, prompt: str) -> dict:
        """Updates instance fields from a natural-language instruction via structured LLM output.

        Builds a dynamic extraction schema where every field is Optional so the
        LLM can leave unchanged fields as None. Only non-None values are merged
        back, ensuring unmentioned fields are never overwritten. The merge goes
        through `model_validate` so all Pydantic validators and coercers run on
        the LLM output.

        Args:
            prompt: Natural-language instruction describing what to change.

        Returns:
            A dict of field names → new values for every field actually updated.
            Empty dict on LLM failure or parse error.
        """
        specialized_prompt = self.__get_prompt_update_object(prompt)
        extraction_schema = self._build_extraction_schema()

        structured_output = self.run_llm(
            prompt=specialized_prompt,
            response_schema=extraction_schema,
            mode="hybrid"
        )

        logger.debug("Structured output: %s", structured_output)

        if not structured_output:
            logger.warning("LLM returned no output for update.")
            return {}

        try:
            if isinstance(structured_output, str):
                # Some providers return a raw string even when a schema is given.
                clean_json = re.sub(r"```json\s?|\s?```", "", structured_output).strip()
                updates = json.loads(clean_json)
            else:
                updates = structured_output.model_dump()

            # None = field intentionally omitted by the LLM in the Optional schema — skip it.
            cls_fields = type(self).model_fields
            filtered = {k: v for k, v in updates.items() if k in cls_fields and v is not None}
            validated = type(self).model_validate({**self.model_dump(), **filtered})
            self.__dict__.update(validated.__dict__)
            self.__pydantic_fields_set__.update(filtered.keys())

            return filtered

        except Exception as e:
            logger.error("Error synchronizing update: %s", e)
            return {}

    @classmethod
    def _build_extraction_schema(cls):
        """Generates a temporary Pydantic model used as the structured-output target for `update_object`.

        Every field is wrapped in `Optional[...]` with a None default so the LLM
        can omit fields it has no information about — preventing silent overwrites.
        Nested `ExoModel` / `ExoModelList` fields and non-primitive lists are
        excluded because the LLM cannot reliably reconstruct relational data in a
        single extraction call.
        """
        fields_for_ai = {}
        for name, field in cls.model_fields.items():
            if field.exclude:
                continue

            origin = get_origin(field.annotation)
            args = get_args(field.annotation)
            base_type = args[0] if origin in (Union, list, list) and args else field.annotation

            try:
                if isinstance(base_type, type):
                    from .exomodel_list import ExoModelList
                    if issubclass(base_type, ExoModel) or issubclass(base_type, ExoModelList):
                        continue
            except Exception:
                pass

            # Non-primitive lists (e.g. List[SomeModel]) are skipped — use targeted
            # nested update methods for those instead.
            if origin is list or origin is list:
                if base_type not in (str, int, float, bool):
                    continue

            default = field.default if field.default is not PydanticUndefined else None
            fields_for_ai[name] = (Optional[field.annotation], default)

        return create_model(f"{cls.__name__}Extraction", **fields_for_ai)

    def update_field(self, field_name: str, prompt: str):
        """Updates a single field from a natural-language prompt, bypassing full-schema extraction.

        Prefer this over `update_object` when only one field needs to change and
        you want to avoid the overhead of building and running the full extraction schema.

        Raises:
            ValueError: If `field_name` does not exist on the model.
        """
        if field_name not in type(self).model_fields:
            raise ValueError(f"Field '{field_name}' does not exist in the model.")

        prompt_llm = self.__get_prompt_update_field(field_name, prompt)
        result = self.run_llm(prompt_llm, mode="hybrid")
        setattr(self, field_name, result)
        return result

    def run_object_prompt(self, prompt: str):
        """Runs a free-form prompt grounded in the current instance data.

        Use for brainstorming, rewriting content, answering questions, or any
        task that needs awareness of the object's fields but should not modify them.
        """
        prompt_llm = self.__get_prompt_run_object_prompt(prompt)
        return self.run_llm(prompt_llm, mode="hybrid")

    def run_analysis(self):
        """Runs a deep critical analysis of the current instance state.

        Operates in `specialist` mode: the LLM is instructed to ground its
        evaluation in the RAG knowledge base when sources are configured, or
        fall back to general domain expertise with a [General Knowledge] marker.
        """
        prompt = self.__get_prompt_run_analysis()
        return self.run_llm(prompt, mode="specialist")

    def run_filling_instructions(self):
        """Returns field-by-field guidance on how to populate this entity correctly.

        Like `run_analysis`, runs in `specialist` mode and uses RAG sources when
        available so the instructions reflect domain-specific rules rather than
        generic best practices.
        """
        prompt = self.__get_prompt_filling_instructions()
        return self.run_llm(prompt, mode="specialist")

    def master_prompt(self, prompt: str):
        """Routes a user request to the appropriate `@llm_function` tool via an orchestrator agent.

        The agent sees the full tool list and decides autonomously which tool(s)
        to call. Use this as the single entry point for chat-style interactions
        where the action (update vs. analyse vs. custom tool) is not known in advance.
        """
        llm_prompt = self.__get_master_prompt(prompt)
        logger.debug("Master prompt initialized.")
        return self.run_llm(prompt=llm_prompt, mode="orchestrator", use_tools=True)

    def run_llm(self, prompt: str, response_schema: Any = None, mode: str = "generalist",
                use_tools: bool = False, temperature: Optional[float] = None,
                max_tokens: Optional[int] = None):
        """Sends a prompt to the underlying `ExoAgent` and returns its response.

        Lazily creates the agent on first call and registers any configured RAG
        sources. This is the single choke-point through which all LLM traffic
        flows, making it the right place to add cross-cutting concerns such as
        retries or observability.

        Args:
            prompt: The fully rendered prompt string to send.
            response_schema: Optional Pydantic model class for structured output.
            mode: Agent persona — ``"generalist"``, ``"specialist"``, ``"hybrid"``,
                  or ``"orchestrator"``.
            use_tools: When True, registers the instance's `llm_tools` with the agent.
            temperature: Per-call override for sampling temperature.
            max_tokens: Per-call override for the maximum response length.
        """
        if self._exo_agent is None:
            self._exo_agent = ExoAgent()
            if self._rag_sources:
                self._exo_agent.add_rag_sources(self._rag_sources)

        if use_tools:
            self._exo_agent.set_external_tools(self.llm_tools)
        else:
            self._exo_agent.set_external_tools([])

        return self._exo_agent.run(
            prompt=prompt,
            response_schema=response_schema,
            mode=mode,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def get_fields_info(self):
        """Returns a `- field: value` string for each non-excluded field, used in prompt templates."""
        info = []
        for name, field_info in type(self).model_fields.items():
            if field_info.exclude:
                continue
            value = getattr(self, name)
            info.append(f"- {name}: {value}")
        return "\n".join(info)

    @staticmethod
    def get_fields_metadata(model_class):
        """Returns a semicolon-delimited field summary (name, type, description) for prompt templates."""
        metadata = ""
        for name, field_info in model_class.model_fields.items():
            if name == "id" or field_info.exclude:
                continue

            type_name = getattr(field_info.annotation, '__name__', str(field_info.annotation))
            description = field_info.description or ""

            if description:
                metadata += f'{name} (type: {type_name}, info: {description}); '
            else:
                metadata += f'{name} (type: {type_name}); '
        return metadata

    def to_csv(self, delimiter: str = ";", include_header: bool = True) -> str:
        """Converts the current instance to a CSV row."""
        data = self.model_dump(exclude_unset=False)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data.keys(), delimiter=delimiter)
        if include_header:
            writer.writeheader()
        writer.writerow(data)
        return output.getvalue().strip()

    @staticmethod
    def _ui_bold(text: str, fmt: str) -> str:
        if fmt == "html":
            return f"<b>{text}</b>"
        if fmt == "markdown":
            return f"**{text}**"
        return text

    @staticmethod
    def _ui_italic(text: str, fmt: str) -> str:
        if fmt == "html":
            return f"<i>{text}</i>"
        if fmt == "markdown":
            return f"*{text}*"
        return text

    def to_ui(self, format: str = "html") -> str:
        """Generates a formatted string for UI display.

        :param format: Output format — ``"html"`` (Telegram), ``"markdown"`` (Discord/CLI),
                       or ``"plain"`` (no markup). Defaults to ``"html"``.
        """
        bold, italic = self._ui_bold, self._ui_italic
        lines = [
            bold(self.__class__.__name__.upper(), format),
            "━━━━━━━━━━━━━━━━━━━━\n"
        ]

        for name, field in type(self).model_fields.items():
            if field.exclude:
                continue

            value = getattr(self, name, None)
            clean_name = name.replace("_", " ").title()

            if value in [None, "", 0]:
                lines.append(f"⚪ {bold(clean_name + ':', format)} {italic('Not provided', format)}")
            elif hasattr(value, 'items') and isinstance(value.items, list):
                item_list = value.items
                if not item_list:
                    lines.append(f"⚪ {bold(clean_name + ':', format)} {italic('Empty list', format)}")
                else:
                    lines.append(f"🔵 {bold(clean_name + ':', format)} {len(item_list)} items registered")
                    limit = min(len(item_list), 5)
                    for idx, item in enumerate(item_list[:limit]):
                        item_label = getattr(item, 'name', getattr(item, 'title', f"Item {idx+1}"))
                        prefix = "└" if idx == limit - 1 else "├"
                        lines.append(f"    {prefix} 🔸 {italic(item_label, format)}")
                    if len(item_list) > 5:
                        lines.append(f"    └ {italic(f'...and {len(item_list) - 5} more', format)}")
            else:
                str_value = str(value)
                if format == "html":
                    str_value = str_value.replace("<", "&lt;").replace(">", "&gt;")
                if len(str_value) > 300:
                    str_value = str_value[:297] + "..."
                lines.append(f"🟢 {bold(clean_name + ':', format)} {str_value}")

        lines.append("\n━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines)

    def __repr__(self) -> str:
        name_val = getattr(self, 'name', 'unnamed')
        return f"<{self.__class__.__name__} name='{name_val}'>"
