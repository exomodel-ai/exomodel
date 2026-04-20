import pytest
from exomodel.exoagent import ExoAgent
from unittest.mock import patch, MagicMock
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader

def test_exoagent_initialization():
    agent = ExoAgent()
    assert agent.model_id is not None
    assert agent.emb_model is not None
    assert agent.sources_queue == []
    assert agent.vector_store is None
    assert agent.rag_tools == []
    assert agent.external_tools == []
    assert agent._agent is None
    assert agent._current_mode == "generalist"

def test_get_system_prompt():
    agent = ExoAgent()
    assert "helpful and direct virtual assistant" in agent._get_system_prompt("generalist").lower()
    assert "senior domain specialist" in agent._get_system_prompt("specialist").lower()
    assert "senior domain specialist" in agent._get_system_prompt("hybrid").lower()
    assert "orchestrator agent" in agent._get_system_prompt("orchestrator").lower()
    
    # Test fallback to generalist
    assert agent._get_system_prompt("unknown_mode") == agent._get_system_prompt("generalist")

def test_add_rag_sources():
    agent = ExoAgent()
    agent._agent = "mock_agent" # Set dummy to check reset
    agent.add_rag_sources(["https://example.com/docs"])
    
    assert "https://example.com/docs" in agent.sources_queue
    assert agent._agent is None # Should be reset

def test_set_external_tools():
    agent = ExoAgent()
    dummy_tool = "mock_tool"
    agent._agent = "mock_agent" # Set dummy to check reset
    agent.set_external_tools([dummy_tool])
    
    assert agent.external_tools == [dummy_tool]
    assert agent._agent is None # Should be reset
    assert dummy_tool in agent.all_tools

def test_get_loader():
    agent = ExoAgent()
    
    # Test PDF (Mockando a inicialização para não checar o arquivo)
    with patch("langchain_community.document_loaders.PyPDFLoader.__init__", return_value=None):
        pdf_loader = agent._get_loader("test.pdf")
        assert isinstance(pdf_loader, PyPDFLoader)
    
    # Test Web
    web_loader = agent._get_loader("https://example.com")
    assert isinstance(web_loader, WebBaseLoader)
    
    # Test Text (Mockando para evitar erro caso test.txt não exista)
    with patch("langchain_community.document_loaders.TextLoader.__init__", return_value=None):
        text_loader = agent._get_loader("test.txt")
        assert isinstance(text_loader, TextLoader)
