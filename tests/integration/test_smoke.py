"""
Smoke tests — hit the real LLM through the full ExoAgent → LLM → Pydantic round-trip.
Gated by pytest.mark.integration; skipped automatically when no API key is present.
Run with: pytest -m integration
"""
import os

import pytest

from exomodel.exoagent import ExoAgent
from exomodel.exomodel import ExoModel


def _has_api_key() -> bool:
    return any(
        os.getenv(k)
        for k in ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "COHERE_API_KEY")
    )


pytestmark = pytest.mark.integration
skip_no_key = pytest.mark.skipif(not _has_api_key(), reason="No LLM API key configured")


class _Item(ExoModel):
    name: str = ""
    color: str = ""


@skip_no_key
def test_smoke_update_object():
    """Full round-trip: natural language → ExoAgent → LLM → Pydantic fields."""
    item = _Item()
    item.update_object("A red apple called Ruby")
    assert item.name != ""
    assert item.color != ""


@skip_no_key
def test_smoke_run_object_prompt():
    """LLM answers a question about the entity state."""
    item = _Item(name="Ruby", color="red")
    answer = item.run_object_prompt("What is the name? Reply with just the name.")
    assert "Ruby" in answer


@skip_no_key
def test_smoke_exoagent_generalist():
    """ExoAgent in generalist mode returns a non-empty string."""
    agent = ExoAgent()
    result = agent.run("Reply with exactly: OK", mode="generalist")
    assert result.strip() != ""


@skip_no_key
def test_smoke_usage_tracking():
    """Token usage is recorded after a real LLM call."""
    agent = ExoAgent()
    agent.run("Say hello", mode="generalist")
    usage = agent.get_usage()
    assert usage["calls"] == 1
    assert usage["total_tokens"] > 0
