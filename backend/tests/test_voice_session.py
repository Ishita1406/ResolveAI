"""The voice session end to end with fake STT/TTS, a deterministic VAD and a scripted LLM,
against the real agent and the test database."""

import asyncio

import pytest
from sqlalchemy import select

import app.agent.loop as agent_loop
from app.config import get_settings
from app.db.models import Ticket
from app.voice.session import DIDNT_CATCH, FILLERS, STT_TROUBLE, TTS_TROUBLE, VoiceSession
from app.voice.stt import FakeSttProvider
from app.voice.tts import FakeTtsProvider
from tests.agent_helpers import Pause, ScriptedLLM, call, make_agent, say
from tests.voice_helpers import Driver, EnergyScorer, noise, quiet

pytestmark = [pytest.mark.anyio, pytest.mark.db]


def settings(**overrides):
    base = {"end_of_turn_silence_ms": 600, "incomplete_turn_silence_ms": 1600,
            "filler_after_ms": 500, "stt_final_wait_ms": 100}  # fmt: skip
    return get_settings().model_copy(update={**base, **overrides})


async def start(tx_db, llm, *, script, tts=None, stt=None, real_time=False, **overrides):
    agent = make_agent(tx_db, llm)
    stt = stt or FakeSttProvider(script)
    tts = tts or FakeTtsProvider(ms_per_char=2.0)
    session = VoiceSession(agent, settings(**overrides), stt=stt, tts=tts, vad=EnergyScorer())
    driver = Driver(session, real_time=real_time)
    await session.start()
    await driver.until(lambda: len(session._phrases) == 6)  # fixed lines cached
    return session, driver, stt, tts


async def test_a_turn_is_heard_answered_and_spoken(tx_db):
    llm = ScriptedLLM(say("Your order VX-8015 is on its way. Anything else?"))
    session, driver, _, tts = await start(tx_db, llm, script=["where is order eight zero one five"])
    await driver.play(noise(800) + quiet(700))
    await driver.settle()

    assert driver.of("user")[0]["text"] == "where is order VX-8015"  # normalized for the agent
    assert driver.of("partial")  # partial transcripts streamed while speaking
    reply = [t for t in tts.spoken if t not in session._phrases]
    assert reply == ["Your order V X, 8, 0, 1, 5 is on its way.", "Anything else?"]  # read back
    assert session.metrics[0].first_audio_ms is not None
    assert any(driver.output)  # audio reached the transport
    await session.close()


async def test_speech_starts_before_the_reply_is_finished(tx_db):
    first, rest = "Let me look into that for you.", "Your parcel is overdue."
    step = say(f"{first} {rest}")
    step.insert(7, Pause(1.0))  # the LLM stalls right after the first sentence
    llm = ScriptedLLM(step)
    session, driver, _, tts = await start(tx_db, llm, script=["where is my parcel"])
    await driver.play(noise(600) + quiet(700))
    await driver.until(lambda: session.metrics and session.metrics[0].first_audio_ms)
    assert first in tts.spoken and rest not in tts.spoken  # spoken while the LLM still works
    await driver.settle()
    assert rest in tts.spoken
    await session.close()


async def test_a_pause_mid_reply_doesnt_silence_the_rest(tx_db):
    """Cartesia ends a context about a second after its last audio. A reply that pauses
    longer (a tool call, a model retry) must still be spoken after the pause."""
    reply = say("Let me check that for you. Your order is on its way.")
    reply.insert(6, Pause(1.0))  # after the first sentence
    llm = ScriptedLLM(reply)
    tts = FakeTtsProvider(ms_per_char=20.0, expire_after_s=0.3)
    session, driver, _, tts = await start(tx_db, llm, script=["where is it"], tts=tts)
    await driver.play(noise(600) + quiet(700))
    await driver.settle()
    assert tts.dropped == []
    assert [t for t in tts.spoken if "order" in t] == ["Your order is on its way."]
    await session.close()


async def test_barge_in_stops_playback_and_cancels_the_llm(tx_db):
    long_reply = say("I can help with that. " * 30)
    long_reply.insert(20, Pause(5.0))  # still generating when the customer interrupts
    llm = ScriptedLLM(long_reply, say("Sure, go ahead."))
    session, driver, _, tts = await start(tx_db, llm, script=["hello", "wait stop"],
                                          tts=FakeTtsProvider(ms_per_char=20.0))  # fmt: skip
    tts.first_chunk_delay_s = 0.3  # each sentence starts slowly: one is in flight at barge-in
    await driver.play(noise(600) + quiet(700))
    await driver.until(lambda: session.out.playing)
    await driver.play(noise(400))  # the customer talks over the agent
    assert "barge_in" in driver.kinds()
    assert not session.out.playing  # playback stopped at once
    assert llm.closed >= 1  # the in-flight LLM stream was closed (Step 5 cancellation)
    assert tts.cancelled >= 1
    await driver.play(quiet(700))
    await driver.settle()
    assert [e["text"] for e in driver.of("user")] == ["hello", "wait stop"]
    await session.close()


async def test_a_pause_mid_sentence_does_not_end_the_turn(tx_db):
    llm = ScriptedLLM(say("Thanks, let me check."))
    session, driver, _, _ = await start(
        tx_db, llm, script=["my order number is", "v x eight zero one five"]
    )
    # 1 s pause after "...is": longer than a finished turn needs, shorter than an open one.
    await driver.play(noise(700) + quiet(1000) + noise(800) + quiet(800))
    await driver.settle()
    assert [e["text"] for e in driver.of("user")] == ["my order number is VX-8015"]
    await session.close()


async def test_a_finished_sentence_ends_the_turn_after_a_short_pause(tx_db):
    llm = ScriptedLLM(say("Okay."), say("Okay again."))
    session, driver, _, _ = await start(tx_db, llm, script=["I want a refund", "for VX-8015"])
    await driver.play(noise(700) + quiet(1000) + noise(800) + quiet(800))
    await driver.settle()
    assert [e["text"] for e in driver.of("user")] == ["I want a refund", "for VX-8015"]
    await session.close()


async def test_a_slow_tool_gets_a_filler(tx_db, monkeypatch):
    real_call_tool = agent_loop.call_tool

    async def slow(*args, **kwargs):
        await asyncio.sleep(0.9)
        return await real_call_tool(*args, **kwargs)

    monkeypatch.setattr(agent_loop, "call_tool", slow)
    llm = ScriptedLLM(call("list_recent_orders"), say("I need to verify you first."))
    session, driver, _, _ = await start(tx_db, llm, script=["what did I order"], real_time=True)
    await driver.play(noise(500) + quiet(700))
    await driver.settle()
    fillers = driver.of("filler")
    assert len(fillers) == 1 and fillers[0]["text"] in FILLERS
    assert session.metrics[0].filler
    await session.close()


async def test_a_quick_tool_gets_no_filler(tx_db):
    llm = ScriptedLLM(call("list_recent_orders"), say("I need to verify you first."))
    session, driver, _, _ = await start(tx_db, llm, script=["what did I order"], real_time=True)
    await driver.play(noise(500) + quiet(700))
    await driver.settle()
    assert driver.of("filler") == []
    await session.close()


async def test_speech_with_no_transcript_asks_again(tx_db):
    llm = ScriptedLLM()
    session, driver, _, _ = await start(tx_db, llm, script=[""])
    await driver.play(noise(500) + quiet(700))
    await driver.settle()
    assert {"type": "say", "text": DIDNT_CATCH} in driver.events
    await session.close()


async def test_stt_failure_is_said_once_retried_and_recovers(tx_db):
    llm = ScriptedLLM(say("Sure."))
    stt = FakeSttProvider(["hello there"], fail_after_s=0.2)
    session, driver, *_ = await start(tx_db, llm, script=[], stt=stt)
    await driver.play(quiet(300))  # the stream drops
    await driver.until(lambda: "recovered" in driver.kinds())
    assert {"type": "say", "text": STT_TROUBLE} in driver.events and stt.connects == 2
    await driver.play(noise(500) + quiet(700))
    await driver.settle()
    assert driver.of("user")[0]["text"] == "hello there"
    await session.close()


async def test_stt_failing_twice_hands_the_call_over(tx_db):
    llm = ScriptedLLM()
    stt = FakeSttProvider(["hello"], fail_after_s=0.2)
    session, driver, *_ = await start(tx_db, llm, script=[], stt=stt)
    stt.fail_connects = 1  # the reconnect fails too
    await driver.play(quiet(300))
    await driver.until(lambda: "handed_over" in driver.kinds())
    handed = driver.of("handed_over")[0]
    assert handed["reference"].startswith("TCK-") and handed["spoken"]
    ticket = await tx_db.scalar(select(Ticket).where(Ticket.status == "escalated"))
    assert ticket.details["reason"] == "tool_failure"
    assert session.ended.is_set()
    await session.close()


async def test_tts_failure_is_said_once_retried_and_the_reply_finishes(tx_db):
    llm = ScriptedLLM(say("Your parcel is overdue. I can refund it."))
    tts = FakeTtsProvider(ms_per_char=2.0, fail_texts=["parcel is overdue"])
    session, driver, *_ = await start(tx_db, llm, script=["where is it"], tts=tts)
    await driver.play(noise(500) + quiet(700))
    await driver.settle()
    assert {"type": "say", "text": TTS_TROUBLE} in driver.events
    assert tts.connects == 2 and "recovered" in driver.kinds()
    assert tts.spoken.count("Your parcel is overdue.") == 2  # said again after reconnecting
    assert "handed_over" not in driver.kinds()
    await session.close()


async def test_tts_failing_twice_hands_the_call_over(tx_db):
    llm = ScriptedLLM(say("Your parcel is overdue."))
    tts = FakeTtsProvider(ms_per_char=2.0, fail_texts=["parcel is overdue", "parcel is overdue"])
    session, driver, *_ = await start(tx_db, llm, script=["where is it"], tts=tts)
    await driver.play(noise(500) + quiet(700))
    await driver.until(lambda: "handed_over" in driver.kinds())
    assert driver.of("handed_over")[0]["reference"].startswith("TCK-")
    await session.close()
