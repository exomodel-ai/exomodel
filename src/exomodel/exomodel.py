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
import re
import json
import inspect
import io
import csv
from typing import TypeVar, Generic, List, Type, Any, Optional, Union, get_origin, get_args
from pydantic import BaseModel, Field, PrivateAttr, create_model, RootModel
from .exoagent import ExoAgent

T = TypeVar('T', bound='ExoModel')

def llm_action(func):
    """Decorator to mark methods that can be executed via run_method_from_prompt."""
    setattr(func, "_is_llm_action", True)
    return func

def llm_function(func):
    """Decorator to mark methods accessible by the Master Prompt (Orchestrator)."""
    setattr(func, "_is_llm_function", True)
    return func

class ExoModel(BaseModel):
    """
    ExoModel: A base class that utilizes an LLM to interact with class structures 
    via prompts, performing "CRUD" operations: create, update, and read.
    """
    _rag_sources: list[str] = PrivateAttr(default_factory=list)
    _exo_agent: Optional[ExoAgent] = PrivateAttr(default=None)

    @llm_function
    def call_update_object(self, prompt: str):
        """
        Use this tool STRICTLY when the user explicitly commands to change, modify, 
        or update the entity's fields based on an instruction.
        MANDATORY: Pass the user's update instruction as the 'prompt' argument.
        """
        print(f"[ExoModel] AI Tool Call: call_update_object")
        return self.update_object(prompt)

    @llm_function
    def call_run_object_prompt(self, prompt: str):
        """
        Use this tool when the user asks to brainstorm, improve, give options, rewrite content, 
        or answer complex questions based on the entity's data. 
        MANDATORY: You must pass the user's original request as the 'prompt' argument.
        """
        print(f"[ExoModel] AI Tool Call: call_run_object_prompt")
        return self.run_object_prompt(prompt)

    @llm_function
    def call_run_object_analysis(self):
        """
        Use this tool when the user asks for a critical analysis, evaluation, or deep review 
        of the entity's current state and strategy. 
        This tool requires NO arguments. Just invoke it.
        """
        print(f"[ExoModel] AI Tool Call: call_run_object_analysis")
        return self.run_analysis()

    @llm_function
    def call_run_filling_instructions(self):
        """
        Use this tool when the user asks for guidance, best practices, rules, 
        or instructions on how to fill out, complete, or improve the fields 
        of this entity based on the reference material.
        This tool requires NO arguments. Just invoke it.
        """
        print(f"[ExoModel] AI Tool Call: call_run_filling_instructions")
        return self.run_filling_instructions()

    @classmethod
    def get_rag_sources(cls) -> list[str]:
        """Override this in child classes to return knowledge files/URLs."""
        return []

    @property
    def llm_tools(self):
        """Scans the instance and returns methods decorated with @llm_function as LangChain tools."""
        from langchain_core.tools import StructuredTool
        tools = []
        for attr_name in dir(type(self)):
            # FILTRO CRUCIAL: ignora privados E metadados do Pydantic
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
        return tools

    def __init__(self, **data):
        # Extract prompt before Pydantic field processing
        prompt = data.pop("prompt", None)
        super().__init__(**data)

        # Load RAG sources from class definition
        self._rag_sources = self.get_rag_sources()
        if self._rag_sources:
            print(f"[ExoModel] Knowledge sources loaded: {self._rag_sources}")

        # If a prompt is provided during init, update the object immediately
        if prompt:
            self.update_object(prompt)

    def add_rag_source(self, rag_source: str):
        """Adds a source to the internal list (uniqueness enforced)."""
        if rag_source not in self._rag_sources:
            self._rag_sources.append(rag_source)

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

        with open(file_path, 'r', encoding='utf-8') as f:
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

    def __get_prompt_run_analysis(self) -> str:
        return self._load_prompt_template(
            "run_analysis.md",
            entity_name=self.__class__.__name__,
            json_schema=self.get_instance_json()
        )

    def __get_prompt_filling_instructions(self) -> str:
        return self._load_prompt_template(
            "filling_instructions.md",
            entity_name=self.__class__.__name__,
            fields_info=self.get_fields_metadata(self.__class__)
        )

    def __get_master_prompt(self, prompt: str) -> str:
        return self._load_prompt_template(
            "master_prompt.md",
            entity_name=self.__class__.__name__,
            prompt=prompt,
            obj_fields_info=self.get_fields_info(),
            tools_info=self.llm_tools
        )

    # -------------------------------------------------------------------------
    # Public operations
    # -------------------------------------------------------------------------

    def update_object(self, prompt: str) -> dict:
        """
        Updates the entity's fields based on a natural language prompt using 
        LLM Structured Output.
        
        This method builds a dynamic extraction schema, runs the LLM, and 
        synchronizes the returned data with the current instance fields. It is 
        designed to be resilient, handling both Pydantic objects and raw JSON 
        strings returned by the LLM.

        Args:
            prompt (str): The user's instruction or data for updating the object.

        Returns:
            dict: A dictionary containing the fields that were successfully updated.
        """
        # 1. Prepare context and schema
        specialized_prompt = self.__get_prompt_update_object(prompt)
        extraction_schema = self.build_extraction_schema()
        
        # 2. Execute LLM call
        structured_output = self.run_llm(
            prompt=specialized_prompt, 
            response_schema=extraction_schema, 
            mode="hybrid"
        )

        print(f"[ExoModel] Structured output: {structured_output}")
        
        if not structured_output:
            print(f"[ExoModel] Warning: LLM returned no output for update.")
            return {}

        try:
            # 3. Handle hybrid output types (Pydantic object vs. Raw String)
            if isinstance(structured_output, str):
                # Attempt to extract JSON from markdown blocks if present
                clean_json = re.sub(r"```json\s?|\s?```", "", structured_output).strip()
                updates = json.loads(clean_json)
            else:
                # Standard Pydantic model handling
                updates = structured_output.model_dump()        

            # 4. Synchronize updates with instance fields
            # We use type(self).model_fields to avoid Pydantic V2.11+ deprecation warnings
            updated_data = {}
            cls_fields = type(self).model_fields
            
            for field_name, field_value in updates.items():
                if field_name in cls_fields:
                    setattr(self, field_name, field_value)
                    updated_data[field_name] = field_value
            
            return updated_data

        except Exception as e:
            print(f"[ExoModel] Error synchronizing update: {e}")
            return {}

    @classmethod
    def build_extraction_schema(cls):
        """Builds a dynamic Pydantic model for LLM data extraction, filtering complex relations."""
        fields_for_ai = {}
        for name, field in cls.model_fields.items():
            if field.exclude:
                continue
                
            origin = get_origin(field.annotation)
            args = get_args(field.annotation)
            
            # Determine base type for List/Union/Optional
            base_type = args[0] if origin in (Union, list, List) and args else field.annotation
            
            try:
                if isinstance(base_type, type):
                    # Skip nested ExoModels or custom List containers
                    if issubclass(base_type, ExoModel):
                        continue
                    if "ListExoModel" in base_type.__name__:
                        continue
            except Exception:
                pass

            # Only allow simple lists of primitives for standard object updates
            if origin is list or origin is List:
                if base_type not in (str, int, float, bool):
                    continue

            fields_for_ai[name] = (field.annotation, ...)

        return create_model(f"{cls.__name__}Extraction", **fields_for_ai)

    def update_field(self, field_name: str, prompt: str):
        """Updates a specific field based on a prompt."""
        if field_name not in type(self).model_fields:
            raise ValueError(f"Field '{field_name}' does not exist in the model.")

        prompt_llm = self.__get_prompt_update_field(field_name, prompt)
        result = self.run_llm(prompt_llm, mode="hybrid")
        setattr(self, field_name, result)
        return result      

    def run_object_prompt(self, prompt: str):
        """Executes a general prompt regarding the object state."""
        prompt_llm = self.__get_prompt_run_object_prompt(prompt)
        return self.run_llm(prompt_llm, mode="hybrid")

    def run_analysis(self):
        """Performs a critical analysis of the object using RAG context."""
        prompt = self.__get_prompt_run_analysis()
        return self.run_llm(prompt, mode="specialist")

    def run_filling_instructions(self):
        """Retrieves filling guidelines and best practices."""
        prompt = self.__get_prompt_filling_instructions()
        return self.run_llm(prompt, mode="specialist")

    def master_prompt(self, prompt: str):
        """The Orchestrator prompt capable of executing other object tools."""
        llm_prompt = self.__get_master_prompt(prompt)
        print(f"[ExoModel] Master Prompt Initialized\n")
        return self.run_llm(prompt=llm_prompt, mode="orchestrator", use_tools=True)

    def run_llm(self, prompt: str, response_schema: Any = None, mode: str = "generalist", use_tools: bool = False):  
        """Unified interface with the ExoAgent."""
        if self._exo_agent is None:
            self._exo_agent = ExoAgent()
            if self._rag_sources:
                self._exo_agent.add_rag_sources(self._rag_sources)

        if use_tools:
            self._exo_agent.set_external_tools(self.llm_tools)
        else:
            self._exo_agent.set_external_tools([])
        
        return self._exo_agent.run(prompt=prompt, response_schema=response_schema, mode=mode)

    def get_fields_info(self):
        """Simplified string representation of current fields for LLM context."""
        info = []
        for name, field_info in type(self).model_fields.items():
            if field_info.exclude:
                continue
            value = getattr(self, name)
            info.append(f"- {name}: {value}")
        return "\n".join(info)

    @staticmethod
    def get_fields_metadata(model_class):
        """Returns string containing fields, types, and descriptions for a Pydantic class."""
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

    def to_ui(self) -> str:
        """Generates a formatted HTML/Markdown string for UI (Telegram/CLI)."""
        lines = [
            f"<b>{self.__class__.__name__.upper()}</b>",
            "━━━━━━━━━━━━━━━━━━━━\n"
        ]
        
        for name, field in type(self).model_fields.items():
            if field.exclude:
                continue
                
            value = getattr(self, name, None)
            clean_name = name.replace("_", " ").title()
            
            if value in [None, "", 0]:
                lines.append(f"⚪ <b>{clean_name}:</b> <i>Not provided</i>")
            elif hasattr(value, 'items') and isinstance(getattr(value, 'items'), list):
                # Logic for nested ExoModel list containers
                item_list = value.items
                if not item_list:
                    lines.append(f"⚪ <b>{clean_name}:</b> <i>Empty list</i>")
                else:
                    lines.append(f"🔵 <b>{clean_name}:</b> {len(item_list)} items registered")
                    limit = min(len(item_list), 5)
                    for idx, item in enumerate(item_list[:limit]):
                        item_label = getattr(item, 'name', getattr(item, 'title', f"Item {idx+1}"))
                        prefix = "└" if idx == limit - 1 else "├"
                        lines.append(f"    {prefix} 🔸 <i>{item_label}</i>")
                    if len(item_list) > 5:
                        lines.append(f"    └ <i>...and {len(item_list) - 5} more</i>")
            else:
                str_value = str(value).replace("<", "&lt;").replace(">", "&gt;")
                if len(str_value) > 300:
                    str_value = str_value[:297] + "..."
                lines.append(f"🟢 <b>{clean_name}:</b> {str_value}")
                
        lines.append("\n━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines)

    def __repr__(self) -> str:
        name_val = getattr(self, 'name', 'unnamed')
        return f"<{self.__class__.__name__} name='{name_val}'>"