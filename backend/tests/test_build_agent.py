"""build_agent wires the policy context mode from settings. No network, no model load."""

import pytest

from app.config import get_settings
from app.main import build_agent


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("AGENT_PROMPT_VERSION", "v3")

    def load(**env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        get_settings.cache_clear()
        return get_settings()

    return load


def test_retrieval_mode(settings):
    agent, _ = build_agent(settings(POLICY_CONTEXT_MODE="retrieval",
                                    RETRIEVAL_SEARCH_MODE="vector"), None, None)  # fmt: skip
    assert agent.retriever is not None and agent.retriever.mode == "vector"
    assert "call search_policies" in agent.system_prompt
    assert "search_policies" in {t["function"]["name"] for t in agent.tools_unverified}


def test_full_text_mode(settings):
    agent, _ = build_agent(settings(POLICY_CONTEXT_MODE="full_text"), None, None)
    assert agent.retriever is None
    assert "<policy_documents>" in agent.system_prompt
    assert "# Refunds policy" in agent.system_prompt
    assert "search_policies" not in {t["function"]["name"] for t in agent.tools_verified}


def test_policy_mode_with_an_old_prompt_version_fails_at_startup(settings):
    with pytest.raises(ValueError, match="policy_context"):
        build_agent(settings(AGENT_PROMPT_VERSION="v2"), None, None)
