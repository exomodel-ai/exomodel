import pytest
import json
from unittest.mock import MagicMock, patch
from exomodel.exomodel import ExoModel, llm_function

class DummyUser(ExoModel):
    """A dummy user for testing."""
    name: str
    age: int
    active: bool = True

    @classmethod
    def get_rag_sources(cls) -> list[str]:
        return ["dummy_source.txt"]

    @llm_function
    def dummy_custom_tool(self):
        """This is a dummy tool for testing."""
        return "Tool executed"

def test_exomodel_initialization():
    """Test standard Pydantic initialization and hidden attributes."""
    user = DummyUser(name="Alice", age=30)
    
    assert user.name == "Alice"
    assert user.age == 30
    assert user.active is True
    
    # Check that rag_sources is loaded from get_rag_sources
    assert user._rag_sources == ["dummy_source.txt"]
    # Check that ExoAgent is deferred until needed
    assert user._exo_agent is None

def test_add_rag_source():
    """Test adding a custom RAG source."""
    user = DummyUser(name="Bob", age=25)
    user.add_rag_source("another_source.pdf")
    
    assert "another_source.pdf" in user._rag_sources
    assert "dummy_source.txt" in user._rag_sources

def test_get_json_schema():
    """Test retrieving Pydantic JSON schema."""
    user = DummyUser(name="Alice", age=30)
    schema_str = user.get_json_schema()
    schema_dict = json.loads(schema_str)
    
    assert "properties" in schema_dict
    assert "name" in schema_dict["properties"]
    assert "age" in schema_dict["properties"]

def test_get_instance_json():
    """Test retrieving the current state as JSON."""
    user = DummyUser(name="Alice", age=30)
    instance_json = user.get_instance_json()
    data = json.loads(instance_json)
    
    assert data["name"] == "Alice"
    assert data["age"] == 30

def test_to_csv():
    """Test CSV row generation."""
    user = DummyUser(name="Alice", age=30)
    
    csv_out = user.to_csv()
    assert "name;age;active" in csv_out
    assert "Alice;30;True" in csv_out

def test_to_ui():
    """Test UI representation generation."""
    user = DummyUser(name="Alice", age=30)
    ui_out = user.to_ui()
    
    assert "DUMMYUSER" in ui_out
    assert "Name:" in ui_out
    assert "Alice" in ui_out
    assert "Age:" in ui_out

def test_llm_tools_property():
    """Test that @llm_function decorated methods are converted to LangChain tools."""
    user = DummyUser(name="Alice", age=30)
    tools = user.llm_tools
    
    # Base ExoModel tools (4) + DummyUser custom tool (1)
    tool_names = [tool.name for tool in tools]
    assert "call_update_object" in tool_names
    assert "call_run_object_prompt" in tool_names
    assert "dummy_custom_tool" in tool_names

def test_get_fields_info():
    """Test field metadata generation for prompts."""
    user = DummyUser(name="Alice", age=30)
    fields_info = user.get_fields_info()
    
    assert "- name: Alice" in fields_info
    assert "- age: 30" in fields_info
    assert "- active: True" in fields_info

def test_repr():
    """Test string representation logic."""
    user = DummyUser(name="Alice", age=30)
    assert repr(user) == "<DummyUser name='Alice'>"

def test_update_object():
    """Test that update_object correctly updates object fields based on LLM output."""
    user = DummyUser(name="Alice", age=30)
    
    mock_structured_output = MagicMock()
    mock_structured_output.model_dump.return_value = {"name": "Alice Cooper", "age": 31, "active": False}
    
    with patch.object(DummyUser, 'run_llm', return_value=mock_structured_output) as mock_run_llm:
        updates = user.update_object("Alice got married and is now 31, and she is no longer active.")
        
        mock_run_llm.assert_called_once()
        
        assert user.name == "Alice Cooper"
        assert user.age == 31
        assert user.active is False
        assert updates == {"name": "Alice Cooper", "age": 31, "active": False}

def test_update_field():
    """Test that update_field correctly updates a specific field."""
    user = DummyUser(name="Alice", age=30)

    with patch.object(DummyUser, 'run_llm', return_value="Alice Cooper") as mock_run_llm:
        result = user.update_field("name", "Change her name to Alice Cooper")

        mock_run_llm.assert_called_once()
        assert user.name == "Alice Cooper"
        assert result == "Alice Cooper"

def test_update_field_invalid():
    """Test that update_field raises ValueError for non-existent fields."""
    user = DummyUser(name="Alice", age=30)
    
    with pytest.raises(ValueError, match="Field 'invalid_field' does not exist in the model."):
        user.update_field("invalid_field", "Do something")

def test_run_analysis():
    """Test that run_analysis invokes the LLM with the correct mode and returns the execution result."""
    user = DummyUser(name="Alice", age=30)
    
    with patch.object(DummyUser, 'run_llm', return_value="Analysis content") as mock_run_llm:
        result = user.run_analysis()
        
        mock_run_llm.assert_called_once()
        args, kwargs = mock_run_llm.call_args
        assert kwargs.get("mode") == "specialist"
        assert result == "Analysis content"

def test_run_filling_instructions():
    """Test that run_filling_instructions invokes the LLM with the correct mode and returns the execution result."""
    user = DummyUser(name="Alice", age=30)
    
    with patch.object(DummyUser, 'run_llm', return_value="Filling instructions content") as mock_run_llm:
        result = user.run_filling_instructions()
        
        mock_run_llm.assert_called_once()
        args, kwargs = mock_run_llm.call_args
        assert kwargs.get("mode") == "specialist"
        assert result == "Filling instructions content"
