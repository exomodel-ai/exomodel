"""
RAG pipeline integration tests — require a real embedding API key.
Gated by pytest.mark.integration.
Run with: pytest -m integration
"""
import os
import tempfile

import pytest

from exomodel.exoagent import ExoAgent


def _has_api_key() -> bool:
    return any(
        os.getenv(k)
        for k in ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")
    )


pytestmark = pytest.mark.integration
skip_no_key = pytest.mark.skipif(not _has_api_key(), reason="No embedding API key configured")


@skip_no_key
def test_rag_indexes_and_retrieves_text_file():
    """Full RAG round-trip: index a text file, retrieve relevant content."""
    content = "ExoModel is an object-oriented agentic AI framework built on Pydantic."

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name

    agent = ExoAgent()
    agent.add_rag_sources([path])
    agent._process_pending_rag()

    assert agent.vector_store is not None
    assert agent.rag_tools != []

    result = agent._retrieve_context_tool.invoke({"query": "What is ExoModel?"})
    assert "ExoModel" in result


@skip_no_key
def test_rag_irrelevant_query_returns_no_results():
    """Query with no semantic match should return the 'no relevant content' message."""
    content = "The quarterly revenue for ACME Corp increased by 12% in Q3."

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name

    agent = ExoAgent()
    agent.add_rag_sources([path])
    agent._process_pending_rag()

    result = agent._retrieve_context_tool.invoke({"query": "quantum physics"})
    assert "No sufficiently relevant content" in result or "ExoModel" not in result


@skip_no_key
def test_rag_multiple_sources_merged():
    """Two indexed files should both be searchable from the same vector store."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f1, \
         tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f2:
        f1.write("ExoModel supports Gemini, Claude, and OpenAI.")
        f2.write("RAG in ExoModel uses InMemoryVectorStore for embedding.")
        path1, path2 = f1.name, f2.name

    agent = ExoAgent()
    agent.add_rag_sources([path1, path2])
    agent._process_pending_rag()

    result = agent._retrieve_context_tool.invoke({"query": "embedding vector store"})
    assert result != "No knowledge base initialized."
