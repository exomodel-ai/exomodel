from unittest.mock import MagicMock, patch

from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_core.documents import Document

from exomodel.exoagent import ExoAgent


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


# ---------------------------------------------------------------------------
# _process_pending_rag tests
# ---------------------------------------------------------------------------

def _make_doc(content="hello world", source="test.txt"):
    return Document(page_content=content, metadata={"source": source})


def _mock_loader(docs):
    loader = MagicMock()
    loader.load.return_value = docs
    return loader


def test_rag_skips_when_queue_empty_and_store_exists():
    agent = ExoAgent()
    agent.vector_store = MagicMock()  # pretend store already built
    agent.sources_queue = []
    agent._process_pending_rag()
    # vector_store should not be touched
    agent.vector_store.add_documents.assert_not_called()


def test_rag_loads_text_source_and_creates_store():
    agent = ExoAgent()
    agent.sources_queue = ["notes.txt"]
    doc = _make_doc("ExoModel is a framework", source="notes.txt")

    mock_store = MagicMock()
    with patch.object(agent, "_get_loader", return_value=_mock_loader([doc])), \
         patch("exomodel.exoagent.InMemoryVectorStore", return_value=mock_store), \
         patch("exomodel.exoagent.init_embeddings", return_value=MagicMock()):
        agent._process_pending_rag()

    assert agent.sources_queue == []
    mock_store.add_documents.assert_called_once()
    assert agent.rag_tools != []


def test_rag_enriches_metadata():
    agent = ExoAgent()
    agent.sources_queue = ["notes.txt"]
    doc = _make_doc("content", source="notes.txt")

    captured = []
    mock_store = MagicMock()
    mock_store.add_documents.side_effect = lambda docs: captured.extend(docs)

    with patch.object(agent, "_get_loader", return_value=_mock_loader([doc])), \
         patch("exomodel.exoagent.InMemoryVectorStore", return_value=mock_store), \
         patch("exomodel.exoagent.init_embeddings", return_value=MagicMock()):
        agent._process_pending_rag()

    assert len(captured) > 0
    meta = captured[0].metadata
    assert meta["source"] == "notes.txt"
    assert meta["source_type"] == "text"
    assert "indexed_at" in meta


def test_rag_source_type_detection():
    agent = ExoAgent()
    doc_web = _make_doc(source="http://example.com/page")
    doc_pdf = _make_doc(source="report.pdf")
    doc_txt = _make_doc(source="notes.txt")

    agent.sources_queue = ["http://example.com/page", "report.pdf", "notes.txt"]

    captured = []
    mock_store = MagicMock()
    mock_store.add_documents.side_effect = lambda docs: captured.extend(docs)

    def fake_loader(source):
        if "example.com" in source:
            return _mock_loader([doc_web])
        if source.endswith(".pdf"):
            return _mock_loader([doc_pdf])
        return _mock_loader([doc_txt])

    with patch.object(agent, "_get_loader", side_effect=fake_loader), \
         patch("exomodel.exoagent.InMemoryVectorStore", return_value=mock_store), \
         patch("exomodel.exoagent.init_embeddings", return_value=MagicMock()):
        agent._process_pending_rag()

    types = {d.metadata["source_type"] for d in captured}
    assert "web" in types
    assert "pdf" in types
    assert "text" in types


def test_rag_failed_source_is_skipped_queue_cleared():
    agent = ExoAgent()
    agent.sources_queue = ["bad_source.txt", "notes.txt"]
    good_doc = _make_doc("good content", source="notes.txt")

    def fake_loader(source):
        if source == "bad_source.txt":
            m = MagicMock()
            m.load.side_effect = FileNotFoundError("not found")
            return m
        return _mock_loader([good_doc])

    mock_store = MagicMock()
    with patch.object(agent, "_get_loader", side_effect=fake_loader), \
         patch("exomodel.exoagent.InMemoryVectorStore", return_value=mock_store), \
         patch("exomodel.exoagent.init_embeddings", return_value=MagicMock()):
        agent._process_pending_rag()

    assert agent.sources_queue == []
    mock_store.add_documents.assert_called_once()


def test_rag_all_sources_fail_no_store_created():
    agent = ExoAgent()
    agent.sources_queue = ["bad.txt"]

    bad_loader = MagicMock()
    bad_loader.load.side_effect = FileNotFoundError("not found")

    with patch.object(agent, "_get_loader", return_value=bad_loader):
        agent._process_pending_rag()

    assert agent.sources_queue == []
    assert agent.vector_store is None
    assert agent.rag_tools == []


def test_rag_retrieve_context_returns_no_kb_when_store_none():
    agent = ExoAgent()
    result = agent._retrieve_context_tool.invoke({"query": "anything"})
    assert "No knowledge base" in result


def test_rag_retrieve_context_filters_by_score_threshold():
    agent = ExoAgent()
    # Mock vector store returning one result below threshold and one above
    low_doc = (_make_doc("irrelevant"), 0.3)
    high_doc = (_make_doc("very relevant content", source="docs.txt"), 0.9)
    mock_store = MagicMock()
    mock_store.similarity_search_with_score.return_value = [low_doc, high_doc]
    agent.vector_store = mock_store

    result = agent._retrieve_context_tool.invoke({"query": "test"})

    assert "very relevant content" in result
    assert "irrelevant" not in result
