"""
Prompt rendering tests — no LLM calls, no API keys required.
Verifies that every _load_prompt_template path renders without KeyError.
"""
import pytest

from exomodel.exomodel import ExoModel


class _SimpleModel(ExoModel):
    name: str = ""
    value: int = 0


@pytest.fixture
def model():
    m = _SimpleModel(name="Test", value=42)
    return m


def test_render_update_object(model):
    prompt = model._load_prompt_template(
        "update_object.md",
        entity_name="SimpleModel",
        obj_fields_info="- name: Test\n- value: 42",
        prompt="Change name to Foo",
    )
    assert "SimpleModel" in prompt
    assert "Change name to Foo" in prompt


def test_render_update_field(model):
    prompt = model._load_prompt_template(
        "update_field.md",
        entity_name="SimpleModel",
        field_name="name",
        field_value="Test",
        prompt="Set it to Foo",
    )
    assert "name" in prompt
    assert "Set it to Foo" in prompt


def test_render_run_analysis_with_rag(model):
    prompt = model._load_prompt_template(
        "run_analysis.md",
        entity_name="SimpleModel",
        json_schema='{"name": "Test", "value": 42}',
        rag_instruction="Ground your analysis in retrieve_context.",
    )
    assert "SimpleModel" in prompt
    assert "retrieve_context" in prompt


def test_render_run_analysis_without_rag(model):
    prompt = model._load_prompt_template(
        "run_analysis.md",
        entity_name="SimpleModel",
        json_schema='{"name": "Test", "value": 42}',
        rag_instruction="No knowledge base is configured. Use domain expertise.",
    )
    assert "No knowledge base" in prompt


def test_render_filling_instructions_with_rag(model):
    prompt = model._load_prompt_template(
        "filling_instructions.md",
        entity_name="SimpleModel",
        fields_info="- name: str\n- value: int",
        rag_instruction="Ground your analysis in retrieve_context.",
    )
    assert "SimpleModel" in prompt


def test_render_filling_instructions_without_rag(model):
    prompt = model._load_prompt_template(
        "filling_instructions.md",
        entity_name="SimpleModel",
        fields_info="- name: str\n- value: int",
        rag_instruction="No knowledge base is configured. Use domain expertise.",
    )
    assert "No knowledge base" in prompt


def test_render_master_prompt(model):
    prompt = model._load_prompt_template(
        "master_prompt.md",
        entity_name="SimpleModel",
        prompt="Do something",
        obj_fields_info="- name: Test\n- value: 42",
        tools_info="- update: updates fields",
    )
    assert "SimpleModel" in prompt
    assert "Do something" in prompt


def test_render_run_object_prompt(model):
    prompt = model._load_prompt_template(
        "run_object_prompt.md",
        entity_name="SimpleModel",
        json_schema='{"name": "Test", "value": 42}',
        prompt="Suggest a better name",
    )
    assert "SimpleModel" in prompt
    assert "Suggest a better name" in prompt


def test_render_create_list(model):
    prompt = model._load_prompt_template(
        "create_list.md",
        entity_name="SimpleModel",
        obj_fields_info="- name: str\n- value: int",
        prompt="Generate 3 items",
    )
    assert "SimpleModel" in prompt
    assert "Generate 3 items" in prompt


def test_missing_placeholder_raises_keyerror(model):
    with pytest.raises(KeyError):
        model._load_prompt_template(
            "update_object.md",
            entity_name="SimpleModel",
            # missing obj_fields_info and prompt
        )
