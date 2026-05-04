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

import os
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field, PrivateAttr, create_model

from .exoagent import ExoAgent
from .exomodel import ExoModel, llm_function

T = TypeVar('T', bound='ExoModel')

class ExoModelList(BaseModel, Generic[T]):
    """Typed collection for generating, updating, and exporting batches of `ExoModel` instances.

    Use when you need to operate on multiple entities in a single LLM call — for example,
    turning a raw transcript into a list of structured records, or exporting pipeline
    results to CSV for downstream processing.

    Holds an `ExoAgent` via composition rather than inheriting `ExoModel`, so only
    list-relevant operations are exposed. RAG sources are inherited from the item class
    via `get_rag_sources()`.

    Two instantiation patterns are supported:

        # Subclass (recommended — Pydantic-native, model_validate works):
        class ShoppingList(ExoModelList[ShoppingItem]):
            pass
        sl = ShoppingList()

        # Direct instantiation (quick / inline use):
        sl = ExoModelList(item_class=ShoppingItem)
    """
    items: list[T] = Field(default_factory=list)

    _item_class: type[T] = PrivateAttr()
    _rag_sources: list[str] = PrivateAttr(default_factory=list)
    _exo_agent: Optional[ExoAgent] = PrivateAttr(default=None)
    _llm_tools_cache: Optional[list[Any]] = PrivateAttr(default=None)

    @classmethod
    def __class_getitem__(cls, item):
        """Captures the concrete type argument so `__init__` can resolve `_item_class` without an explicit kwarg.

        Fires for both ``class MyList(ExoModelList[T])`` and inline ``ExoModelList[T]``.
        TypeVar arguments (still-unresolved generics) are skipped.
        """
        alias = super().__class_getitem__(item)
        if isinstance(item, type) and issubclass(item, ExoModel):
            alias._item_class = item
        return alias

    def __init__(self, item_class: Optional[type[T]] = None, prompt: str = "", **data):
        """Initialises the list for a specific `ExoModel` subclass and optionally populates it.

        Args:
            item_class: Concrete `ExoModel` subclass this list will hold. Optional when
                        the class was declared as ``ExoModelList[ConcreteType]``.
            prompt: If provided, immediately calls ``create_list(prompt)`` after init.

        Raises:
            TypeError: If the item class cannot be resolved from either the argument
                       or the class-level attribute set by ``__class_getitem__``.
        """
        super().__init__(**data)

        # Runtime arg takes priority; fall back to the class-level attr set by
        # __class_getitem__ when the user declares ExoModelList[ConcreteType].
        resolved = item_class or getattr(self.__class__, "_item_class", None)
        if resolved is None:
            raise TypeError(
                "item_class must be provided as an argument or declared via "
                "ExoModelList[ItemClass] (e.g. class MyList(ExoModelList[MyItem]): pass)."
            )

        self._item_class = resolved
        self._rag_sources = self._item_class.get_rag_sources()
        self._exo_agent = None

        if prompt:
            self.create_list(prompt)

    # -------------------------------------------------------------------------
    # Default orchestrator tools
    # -------------------------------------------------------------------------

    @llm_function
    def call_create_list(self, prompt: str):
        """
        Use this tool when the user wants to generate or replace the entire list
        from a natural-language description.
        MANDATORY: Pass the user's original request as the 'prompt' argument.
        """
        return self.create_list(prompt)

    @llm_function
    def call_update_list(self, prompt: str):
        """
        Use this tool when the user wants to modify, add, remove, or adjust
        items in the existing list based on an instruction.
        MANDATORY: Pass the user's original request as the 'prompt' argument.
        """
        return self.update_list(prompt)

    # -------------------------------------------------------------------------
    # Agent plumbing (composition — mirrors ExoModel's implementation)
    # -------------------------------------------------------------------------

    @property
    def llm_tools(self):
        """Returns all `@llm_function` methods on this instance as LangChain `StructuredTool` objects."""
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

    def add_rag_source(self, rag_source: str):
        """Appends a RAG source and forwards it to the active agent if one exists.

        Deduplicates silently. Sources added here are merged with those inherited
        from the item class via ``get_rag_sources()``.
        """
        if rag_source not in self._rag_sources:
            self._rag_sources.append(rag_source)
            if self._exo_agent is not None:
                self._exo_agent.add_rag_sources([rag_source])

    def _get_prompt_path(self, filename: str) -> str:
        """Resolves the absolute path for a given prompt template file."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "prompt", filename)

    def _load_prompt_template(self, filename: str, **kwargs) -> str:
        """Loads a prompt template and renders it with the given keyword arguments.

        Raises:
            FileNotFoundError: If the template file is missing.
            KeyError: If a required ``{placeholder}`` is absent from ``kwargs``.
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

    def run_llm(self, prompt: str, response_schema: Any = None, mode: str = "generalist",
                use_tools: bool = False):
        """Sends a prompt to the underlying `ExoAgent`, lazily creating it on first call."""
        if self._exo_agent is None:
            self._exo_agent = ExoAgent()
            if self._rag_sources:
                self._exo_agent.add_rag_sources(self._rag_sources)

        if use_tools:
            self._exo_agent.set_external_tools(self.llm_tools)
        else:
            self._exo_agent.set_external_tools([])

        return self._exo_agent.run(prompt=prompt, response_schema=response_schema, mode=mode)

    def _get_master_prompt(self, prompt: str) -> str:
        tools_info = "\n".join(
            f"- {t.name}: {t.description}" for t in self.llm_tools
        )
        current_state = self.to_csv() or "(empty list)"
        return self._load_prompt_template(
            "master_prompt.md",
            entity_name=f"{self._item_class.__name__}List",
            prompt=prompt,
            obj_fields_info=current_state,
            tools_info=tools_info
        )

    def master_prompt(self, prompt: str):
        """Routes a user request to the appropriate `@llm_function` tool via an orchestrator agent.

        Use this as the single entry point for chat-style interactions where the action
        (create, update, or a custom tool) is not known in advance.
        """
        llm_prompt = self._get_master_prompt(prompt)
        return self.run_llm(prompt=llm_prompt, mode="orchestrator", use_tools=True)

    # -------------------------------------------------------------------------
    # Schema helpers
    # -------------------------------------------------------------------------

    def _build_list_schema(self) -> type[BaseModel]:
        """Wraps the item extraction schema in a container model with an ``items`` list field.

        The container is what the LLM sees as the structured output target, ensuring
        the response is always a JSON object with a top-level ``items`` array.
        """
        ItemSchema = self._item_class._build_extraction_schema()
        container_name = f"{self._item_class.__name__}ListContainer"
        return create_model(
            container_name,
            items=(list[ItemSchema], Field(
                default_factory=list,
                description=f"A list of {self._item_class.__name__} objects"
            ))
        )

    # -------------------------------------------------------------------------
    # Public operations
    # -------------------------------------------------------------------------

    def create_list(self, prompt: str) -> "ExoModelList[T]":
        """Generates and replaces the entire item list from a natural-language prompt.

        Sends the prompt to the LLM in hybrid mode with a structured output schema
        that forces the response into a typed ``items`` array. Each element is then
        cast to a full `ExoModel` instance so validators and defaults are applied.

        Returns ``self`` to allow method chaining.
        """
        response_schema = self._build_list_schema()
        prompt_llm = self._get_prompt_create_list(prompt=prompt)

        result = self.run_llm(
            prompt=prompt_llm,
            response_schema=response_schema,
            mode="hybrid"
        )

        extracted_items = []
        if result:
            if isinstance(result, BaseModel):
                extracted_items = result.items
            elif isinstance(result, dict):
                validated = response_schema(**result)
                extracted_items = validated.items

        self.items = [
            self._item_class(**item.model_dump())
            for item in extracted_items
        ]

        return self

    def update_list(self, prompt: str) -> "ExoModelList[T]":
        """Updates the list according to a natural-language instruction.

        Serialises the current list to CSV and prepends it to the prompt so the LLM
        has full context before regenerating. Delegates to ``create_list`` when the
        list is empty, since there is nothing to serialise.

        Note: the LLM regenerates the entire list on each call. For large lists this
        can be expensive — see EVOLUTION_PLAN item 2.5 for the planned targeted-update strategy.
        """
        if not self.items:
            return self.create_list(prompt)

        current_state_csv = self.to_csv()
        evolution_prompt = (
            f"CURRENT LIST STATE (CSV):\n{current_state_csv}\n\n"
            f"UPDATE INSTRUCTION: {prompt}"
        )

        return self.create_list(evolution_prompt)

    def _get_prompt_create_list(self, prompt: str) -> str:
        """Constructs the prompt for the LLM to generate the list."""
        entity_name = self._item_class.__name__

        fields_info = "\n".join(
            f"- {name}: {field.description or 'No description provided.'}"
            for name, field in self._item_class.model_fields.items()
            if not field.exclude
        )

        return self._load_prompt_template(
            "create_list.md",
            entity_name=entity_name,
            prompt=prompt,
            obj_fields_info=fields_info
        )

    def to_csv(self, delimiter: str = ";") -> str:
        """Serialises all items to a single CSV string with one header row.

        Returns an empty string when the list is empty. Delegates to each item's
        ``to_csv()`` method, so ``Field(exclude=True)`` fields are automatically omitted.
        """
        if not self.items:
            return ""

        output = [self.items[0].to_csv(delimiter=delimiter, include_header=True)]
        for item in self.items[1:]:
            output.append(item.to_csv(delimiter=delimiter, include_header=False))

        return "\n".join(output)

    def to_ui(self) -> str:
        """Returns an HTML-formatted string listing all items, suitable for Telegram or CLI display."""
        item_title = self._item_class.__name__.upper() if self._item_class else "ITEM"

        lines = [
            f"<b>{item_title} LIST</b>",
            f"<i>Total: {len(self.items)} items</i>",
            "━━━━━━━━━━━━━━━━━━━━\n"
        ]

        if not self.items:
            lines.append("⚪ <i>This list is currently empty.</i>")
        else:
            for i, item in enumerate(self.items, 1):
                lines.append(f"🔹 <b>ITEM #{i}</b>")
                for name, field in item.model_fields.items():
                    if field.exclude:
                        continue
                    val = getattr(item, name, "---")
                    clean_name = name.replace("_", " ").title()
                    lines.append(f"  ▪️ <b>{clean_name}:</b> {val}")

                if i < len(self.items):
                    lines.append("  " + "┈" * 15)

        lines.append("\n━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines)

    def __str__(self) -> str:
        count = len(self.items)
        return f"ExoModelList<{self._item_class.__name__}> (Count: {count})"

    def __repr__(self) -> str:
        return self.__str__()
