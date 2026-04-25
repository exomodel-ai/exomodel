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

from typing import TypeVar, Generic, List, Type, Any, Optional, Union
from pydantic import BaseModel, Field, PrivateAttr, create_model
from .exomodel import ExoModel

# Generic type constrained to ExoModel
T = TypeVar('T', bound='ExoModel')

class ExoModelList(ExoModel, Generic[T]):
    """
    ExoModelList: A specialized ExoModel that manages collections of other ExoModels.
    It handles bulk creation, updates, and CSV/UI representations for lists.
    """
    items: List[T] = Field(default_factory=list)
    
    _item_class: Type[T] = PrivateAttr()

    def __init__(self, item_class: Type[T], prompt: str = "", **data):
        """
        Initializes the list manager for a specific ExoModel subclass.
        :param item_class: The ExoModel class that this list will contain.
        :param prompt: Optional prompt to immediately populate the list.
        """
        # prompt is intentionally NOT passed to super().__init__() because
        # _item_class must be set before any LLM call. Passing prompt to super
        # would trigger update_object() before _item_class is defined, causing
        # an AttributeError. We handle prompt manually after setup is complete.
        super().__init__(**data)
        self._item_class = item_class
        
        # Inherit RAG sources from the item class itself
        self._rag_sources = self._item_class.get_rag_sources()
        
        if prompt:
            self.create_list(prompt)

    def _build_list_schema(self) -> Type[BaseModel]:
        """
        Dynamically constructs a list schema (envelope) containing only 
        the allowed fields from the item class for LLM extraction.

        Reuses build_extraction_schema() from ExoModel to ensure consistent
        filtering of complex types (nested ExoModels, non-primitive lists, etc.).
        """
        # Reuse parent's extraction schema — already filters nested ExoModels,
        # ListExoModel containers, and lists of non-primitive types correctly.
        ItemSchema = self._item_class.build_extraction_schema()

        container_name = f"{self._item_class.__name__}ListContainer"
        return create_model(
            container_name,
            items=(List[ItemSchema], Field(
                default_factory=list,
                description=f"A list of {self._item_class.__name__} objects"
            ))
        )

    def create_list(self, prompt: str) -> "ExoModelList[T]":
        """
        Populates the items list by processing a prompt through the LLM.
        """
        response_schema = self._build_list_schema()
        prompt_llm = self._get_prompt_create_list(prompt=prompt)

        # Use the internal run_llm logic from the parent ExoModel
        result = self.run_llm(
            prompt=prompt_llm, 
            response_schema=response_schema, 
            mode="hybrid"
        )

        extracted_items = []
        if result:
            if isinstance(result, dict):
                validated = response_schema(**result)
                extracted_items = validated.items
            elif hasattr(result, 'items'):
                extracted_items = result.items

        # Casting: Convert temporary schema objects back to the real ExoModel class
        self.items = [
            self._item_class(**item.model_dump()) 
            for item in extracted_items
        ]
        
        return self

    def update_list(self, prompt: str) -> "ExoModelList[T]":
        """
        Updates the internal list in-place based on a new instruction.
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

        # Uses _load_prompt_template() from ExoModel — raises FileNotFoundError
        # explicitly if the template is missing, instead of silently falling back
        # to a degraded prompt that would produce unpredictable LLM output.
        return self._load_prompt_template(
            "create_list.md",
            entity_name=entity_name,
            prompt=prompt,
            obj_fields_info=fields_info
        )

    def to_csv(self, delimiter: str = ";") -> str:
        """Converts the entire list to a single CSV string with headers."""
        if not self.items:
            return ""

        output = [self.items[0].to_csv(delimiter=delimiter, include_header=True)]
        for item in self.items[1:]:
            output.append(item.to_csv(delimiter=delimiter, include_header=False))

        return "\n".join(output)

    def to_ui(self) -> str:
        """Generates a high-quality UI representation for Telegram/CLI."""
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
                    if field.exclude: continue
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