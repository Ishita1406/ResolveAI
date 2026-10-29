"""The terminal chat, with a fake agent and scripted keyboard input."""

import builtins

import pytest

from app.agent import cli
from tests.test_chat_ws import FakeAgent

pytestmark = pytest.mark.anyio


class FakeLLM:
    closed = False

    async def aclose(self):
        FakeLLM.closed = True


def _keyboard(monkeypatch, lines):
    feed = iter(lines)

    def fake_input(prompt=""):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr(builtins, "input", fake_input)


async def test_chat_session(monkeypatch, capsys):
    agent = FakeAgent()
    monkeypatch.setattr(cli, "build_agent", lambda *a: (agent, FakeLLM()))
    _keyboard(monkeypatch, ["hi", "", "/state", "/quit", "never sent"])
    assert await cli.main([]) == 0
    out = capsys.readouterr().out
    assert "Hello there." in out
    assert "turn 0 | not verified" in out
    assert [text for _, text in agent.received] == ["hi"]
    assert FakeLLM.closed


async def test_ctrl_d_exits(monkeypatch):
    monkeypatch.setattr(cli, "build_agent", lambda *a: (FakeAgent(), FakeLLM()))
    _keyboard(monkeypatch, [])
    assert await cli.main(["--memory"]) == 0


async def test_without_a_key_it_explains_and_exits(monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_agent", lambda *a: (None, None))
    assert await cli.main([]) == 2
    assert "GROQ_API_KEY is not set" in capsys.readouterr().err
