"""Observability: JSON logs with ids, the conversation trace, turn_metrics, the replay view,
the latency report, and the latency options (model routing, whole-reply TTS)."""

import asyncio
import base64
import io
import json
import logging
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr
from sqlalchemy import select

from app.agent.llm import Completion, TextDelta, ToolCall
from app.config import get_settings
from app.db.models import Conversation, TurnMetric
from app.main import create_app
from app.observability import conversation, latency, logs, metrics, tracing
from app.voice.audio import VAD_RATE, write_wav
from tests.agent_helpers import ScriptedLLM, make_agent, run_turn, say
from tests.test_voice_session import start
from tests.voice_helpers import UTTERANCES, noise, quiet, slice_ms, speech_sample

EMAIL, PIN = "meera.pillai@example.com", "302647"
_exporter = InMemorySpanExporter()


_installed = []


@pytest.fixture
def spans():
    if not _installed:  # the provider is process-wide: add the exporter once
        tracing.setup_tracing(get_settings(), processors=(SimpleSpanProcessor(_exporter),))
        _installed.append(True)
    _exporter.clear()
    yield _exporter
    _exporter.clear()


def named(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


def verify_then_answer(answer="Thanks, Meera. How can I help?"):
    """Two model calls with usage: verify the customer, then answer."""
    verify = ToolCall("c1", "verify_customer", json.dumps({"email": EMAIL, "postal_code": PIN}))
    usage = {"prompt_tokens": 100, "completion_tokens": 10}
    return ScriptedLLM(
        [Completion(text="", tool_calls=[verify], usage=usage, first_token_s=0.05)],
        [TextDelta(answer), Completion(text=answer, usage=usage, first_token_s=0.04)],
    )


# --- logs ---------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_json_log_lines_carry_the_conversation_and_trace_ids():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(logs.ContextFilter())
    handler.setFormatter(logs.JsonFormatter())
    log = logging.getLogger("test.observability")
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        log.info("outside")
        with logs.bound(conversation_id=7, session_id="s-1", turn_id="t-1"):
            with tracing.tracer.start_as_current_span("work"):
                log.info("inside", extra={"tool": "get_order"})

            async def child():  # tasks inherit the ids
                log.warning("from a task")

            await asyncio.create_task(child())
    finally:
        log.removeHandler(handler)
    outside, inside, task = (json.loads(line) for line in stream.getvalue().splitlines())
    assert outside["conversation_id"] is None and outside["message"] == "outside"
    assert (inside["conversation_id"], inside["session_id"], inside["turn_id"]) == (7, "s-1", "t-1")
    assert inside["tool"] == "get_order" and inside["level"] == "INFO"
    assert task["conversation_id"] == 7 and task["level"] == "WARNING"


# --- agent: trace and turn_metrics --------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.db
async def test_a_turn_is_traced_and_measured(tx_db, spans):
    llm = verify_then_answer()
    llm.model = "test-model"
    agent = make_agent(tx_db, llm, llm_prices={"test-model": {"input": 1.0, "output": 2.0}})
    session = await agent.start_session()
    events = await run_turn(agent, session.session_id, f"I'm {EMAIL}, PIN {PIN}")
    done = events[-1]
    await agent.end_session(session.session_id)
    await agent.end_session(session.session_id)  # a second end writes nothing new

    conversation_id = (await agent.store.load(session.session_id)).conversation_id
    trace_id, root_id = tracing.conversation_ids(conversation_id)
    (turn,) = named(spans, "turn")
    (root,) = named(spans, "conversation")
    calls = named(spans, "llm.call")
    assert {s.context.trace_id for s in spans.get_finished_spans()} == {trace_id}
    assert root.parent is None and root.context.span_id == root_id
    assert turn.parent.span_id == root_id  # every turn hangs off the conversation
    assert len(calls) == 2 and all(c.parent.span_id == turn.context.span_id for c in calls)
    assert len(named(spans, "llm.first_token")) == 2
    (tool,) = named(spans, "tool.verify_customer")
    assert tool.attributes["resolveai.tool.ok"] is True
    attrs = calls[0].attributes
    assert attrs["gen_ai.usage.input_tokens"] == 100
    assert attrs["langfuse.observation.type"] == "generation"
    assert attrs["langfuse.observation.metadata.prompt_version"] == "v2"
    assert attrs["gen_ai.usage.cost"] == pytest.approx((100 * 1.0 + 10 * 2.0) / 1e6)
    assert root.attributes["langfuse.session.id"] == str(conversation_id)

    row = await tx_db.get(TurnMetric, done.turn_id)
    assert (row.conversation_id, row.turn_index, row.outcome) == (conversation_id, 0, "completed")
    assert (row.llm_calls, row.prompt_tokens, row.completion_tokens) == (2, 200, 20)
    assert row.cost_usd == pytest.approx(2 * 120 / 1e6)
    assert row.llm_first_token_ms == 50.0 and row.tool_calls == 1
    assert row.model == "test-model" and row.prompt_version.startswith("v2+")
    assert row.trace_id == format(trace_id, "032x")
    assert row.time_to_first_audio_ms is None  # a text turn has no voice columns
    ended = await tx_db.scalar(
        select(Conversation.ended_at).where(Conversation.id == conversation_id)
    )
    assert ended is not None


@pytest.mark.anyio
@pytest.mark.db
async def test_a_voice_turn_adds_its_timings_to_the_same_row(tx_db, spans):
    llm = ScriptedLLM(say("Your order is on its way. Anything else?"))
    session, driver, _, _ = await start(tx_db, llm, script=["where is my order"])
    await driver.play(noise(800) + quiet(700))
    await driver.settle()
    await session.close()

    turn_metrics = session.metrics[0]
    row = await tx_db.get(TurnMetric, turn_metrics.turn_id)
    assert row.channel == "voice" and row.llm_calls == 1  # the agent's columns
    assert row.time_to_first_audio_ms == turn_metrics.first_audio_ms > 0  # and the voice's
    assert row.vad_ms >= 600 and row.tts_first_audio_ms is not None
    assert row.filler is False and row.interrupted is False
    stages = [row.vad_ms, row.stt_ms, row.agent_first_text_ms, row.first_sentence_ms,
              row.tts_first_audio_ms]  # fmt: skip
    assert sum(stages) == pytest.approx(row.time_to_first_audio_ms, abs=1.0)

    (turn,) = named(spans, "turn")  # the agent reused the voice session's turn span
    for name in ("vad.end_of_turn", "stt.final", "llm.call", "tts.first_audio"):
        (span,) = named(spans, name)
        assert span.parent.span_id == turn.context.span_id, name
    assert turn.attributes["resolveai.voice.time_to_first_audio_ms"] == row.time_to_first_audio_ms
    assert named(spans, "conversation")  # written when the call closed


@pytest.mark.anyio
@pytest.mark.db
async def test_whole_reply_tts_waits_for_the_end_of_the_reply(tx_db):
    llm = ScriptedLLM(say("Let me check. Your order is on its way."))
    session, driver, _, tts = await start(
        tx_db, llm, script=["where is my order"], tts_streaming="reply"
    )
    await driver.play(noise(800) + quiet(700))
    await driver.settle()
    await session.close()
    reply = [t for t in tts.spoken if t not in session._phrases]
    assert reply == ["Let me check. Your order is on its way."]  # one piece, at the end


# --- model routing ------------------------------------------------------------------------


class RoutingLLM(ScriptedLLM):
    def __init__(self, *steps):
        super().__init__(*steps)
        self.models: list[str | None] = []

    async def stream(self, messages, tools, *, stop=None, model=None):
        self.models.append(model)
        async for event in super().stream(messages, tools, stop=stop):
            yield event


@pytest.mark.anyio
@pytest.mark.db
async def test_the_fast_model_answers_only_before_verification(tx_db):
    verify = ToolCall("c1", "verify_customer", json.dumps({"email": EMAIL, "postal_code": PIN}))
    llm = RoutingLLM(
        say("Could I have your email and PIN?"),
        [Completion(text="", tool_calls=[verify])],
        say("Thanks, you're verified."),
    )
    agent = make_agent(tx_db, llm, fast_model="small-model")
    session = await agent.start_session()
    await run_turn(agent, session.session_id, "where is my order")
    await run_turn(agent, session.session_id, f"{EMAIL}, PIN {PIN}")
    # Unverified calls (including the one that verifies) go to the small model.
    assert llm.models == ["small-model", "small-model", None]


# --- replay view --------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.db
async def test_the_replay_view_shows_turns_tools_and_timings(tx_db, spans):
    agent = make_agent(tx_db, verify_then_answer())
    session = await agent.start_session()
    await run_turn(agent, session.session_id, f"I'm {EMAIL}, PIN {PIN}")
    conversation_id = (await agent.store.load(session.session_id)).conversation_id

    data = await conversation.load(tx_db, conversation_id, salt="s")
    (turn,) = data["turns"]
    assert turn["customer"].startswith("I'm meera")
    assert [c["tool"] for c in turn["tool_calls"]] == ["verify_customer"]
    assert turn["assistant"] == "Thanks, Meera. How can I help?"
    assert turn["timings"]["llm_calls"] == 2
    assert data["conversation"]["customer"]["name"] == "Meera Pillai"
    assert data["trace_id"] == tracing.trace_id_hex(conversation_id, "s")
    assert data["totals"]["prompt_tokens"] == 200
    text = conversation.render(data)
    assert "tool verify_customer [call]" in text and "llm first token 50 ms" in text
    assert await conversation.load(tx_db, 10**9, salt="s") is None


def test_policy_decisions_join_the_turn_they_happened_in():
    turns = [{"at": 10, "policy_decisions": []}, {"at": 20, "policy_decisions": []}]
    decision = {"decided_at": 15, "stage": "proposal", "tool_name": "issue_refund",
                "action_id": "a", "verdict": "reject", "rule": "refund.cap", "reason": None,
                "message": "Over the cap.", "amount_paise": 100, "policy_version": "1"}  # fmt: skip
    conversation._attach_decisions(turns, [SimpleNamespace(**decision)])
    assert turns[0]["policy_decisions"] == []
    assert turns[1]["policy_decisions"][0]["verdict"] == "reject"


@pytest.mark.db
def test_the_admin_api_is_guarded(test_db_settings):
    no_llm = test_db_settings.model_copy(update={"gemini_api_key": None})
    with TestClient(create_app(no_llm)) as client:  # the test client is loopback
        assert client.get("/admin/conversations").status_code == 200
        assert client.get("/admin/conversations/999999999").status_code == 404
        assert "Conversation replay" in client.get("/admin/replay").text
    guarded = no_llm.model_copy(update={"admin_token": SecretStr("s3cret")})
    with TestClient(create_app(guarded)) as client:
        assert client.get("/admin/conversations").status_code == 401
        wrong = {"Authorization": "Bearer nope"}
        assert client.get("/admin/conversations", headers=wrong).status_code == 401
        ok = {"Authorization": "Bearer s3cret"}
        assert client.get("/admin/conversations", headers=ok).status_code == 200


# --- metrics and the latency report -------------------------------------------------------


def test_cost_needs_a_price():
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 500_000}
    assert metrics.cost_usd("m", usage, {"m": {"input": 0.3, "output": 2.5}}) == 1.55
    assert metrics.cost_usd("unpriced", usage, {}) is None


def test_percentiles_interpolate_like_numpy():
    values = [100, 200, 300, 400, 1000]
    assert latency.percentile(values, 50) == 300
    assert latency.percentile(values, 95) == pytest.approx(880)
    assert latency.percentile([], 50) is None
    summary = latency.summarize([{"time_to_first_audio_ms": v} for v in values])
    assert summary["time_to_first_audio_ms"]["p99"] == pytest.approx(976)
    assert summary["vad_ms"]["n"] == 0


@pytest.mark.db
def test_latency_cli_runs_offline_and_writes_a_report(tmp_path, monkeypatch, test_db_settings):
    speech = speech_sample()  # real speech: the report uses the real Silero VAD
    for i, line in enumerate(["where is my order", "thanks"]):
        wav = tmp_path / f"turn-{i}.wav"
        write_wav(wav, slice_ms(speech, *UTTERANCES[i]) + quiet(300), VAD_RATE)
        wav.with_suffix(".txt").write_text(line)
    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()
    out = tmp_path / "report"
    wavs = [str(tmp_path / "turn-0.wav"), str(tmp_path / "turn-1.wav")]
    argv = ["run", *wavs, "--runs", "2", "--fake", "--fast", "--paced-llm", "50,20"]
    assert latency.main([*argv, "--out", str(out)]) == 0
    results = json.loads((out / "results.json").read_text())
    assert results["summary"]["time_to_first_audio_ms"]["n"] == 4
    assert results["config"]["llm"].startswith("paced")
    assert all(t["llm_calls"] == 1 for t in results["turns"])  # merged from turn_metrics
    assert "| **time_to_first_audio_ms** | 4 |" in (out / "report.md").read_text()


def test_request_spans_are_dropped_and_conversations_kept(spans):
    with tracing.tracer.start_as_current_span("GET /health"):
        with tracing.tracer.start_as_current_span("fastapi.endpoint"):
            pass
    assert spans.get_finished_spans() == ()
    tracing.end_conversation(1, datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC))
    (root,) = spans.get_finished_spans()
    assert root.name == "conversation" and root.attributes["langfuse.session.id"] == "1"


def test_langfuse_export_is_otlp_with_basic_auth():
    settings = get_settings().model_copy(
        update={"langfuse_public_key": "pk-lf-1", "langfuse_secret_key": SecretStr("sk-lf-2"),
                "langfuse_host": "https://langfuse.example/",
                "otel_exporter_otlp_endpoint": "http://localhost:4318",
                "otel_exporter_otlp_headers": "x-team=support"}
    )  # fmt: skip
    langfuse, collector = tracing.export_targets(settings)
    auth = base64.b64encode(b"pk-lf-1:sk-lf-2").decode()
    assert langfuse == (
        "langfuse",
        "https://langfuse.example/api/public/otel/v1/traces",
        {"Authorization": f"Basic {auth}"},
    )
    assert collector == ("otlp", "http://localhost:4318/v1/traces", {"x-team": "support"})
