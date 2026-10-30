"""Voice building blocks: Silero VAD on real speech, audio helpers, output buffering,
transcript assembly, end-of-turn rules and speech text shaping."""

import asyncio
import base64
import hashlib
import json

import pytest

from app.config import get_settings
from app.voice.audio import Resampler, read_wav, write_wav
from app.voice.session import OutputBuffer, _Transcript
from app.voice.speech import SentenceChunker, for_speech, normalize_transcript
from app.voice.stt import SttEvent
from app.voice.tts import CartesiaTtsConnection, TtsError
from app.voice.turn import looks_complete, silence_needed_ms
from app.voice.vad import FRAME_BYTES, SileroVAD, VadTracker, ensure_model
from tests.voice_helpers import UTTERANCES, quiet, slice_ms, speech_sample


@pytest.fixture(scope="module")
def silero():
    s = get_settings()
    try:
        path = ensure_model(s.vad_model_url, s.vad_model_sha256, s.vad_cache_dir)
    except Exception as exc:
        pytest.skip(f"Silero model unavailable: {exc}")
    return SileroVAD(path)


def _frames(pcm: bytes) -> list[bytes]:
    return [pcm[i : i + FRAME_BYTES] for i in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES)]


# --- VAD ------------------------------------------------------------------------------------


def test_silero_hears_speech_and_not_silence(silero):
    speech = slice_ms(speech_sample(), *UTTERANCES[0])
    silero.reset()
    scores = [silero.score(f) for f in _frames(speech)]
    assert sum(p > 0.5 for p in scores) / len(scores) > 0.6
    silero.reset()
    assert max(silero.score(f) for f in _frames(quiet(1000))) < 0.1


def test_tracker_starts_after_min_speech_and_counts_silence(silero):
    tracker = VadTracker(silero, threshold=0.5, min_speech_ms=160)
    speech = slice_ms(speech_sample(), *UTTERANCES[1])
    updates = [tracker.update(f) for f in _frames(speech + quiet(800))]
    starts = [i for i, u in enumerate(updates) if u.speech_started]
    assert len(starts) == 1 and starts[0] >= 4  # 160 ms of speech = 5 frames of 32 ms
    assert updates[-1].silence_ms >= 700


def test_model_download_is_verified(tmp_path, monkeypatch):
    import io
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: io.BytesIO(b"tampered"))
    with pytest.raises(RuntimeError, match="sha256"):
        ensure_model("https://example.invalid/silero_vad.onnx", "0" * 64, tmp_path)
    assert not (tmp_path / "silero_vad.onnx").exists()  # a bad download is never kept
    good = hashlib.sha256(b"tampered").hexdigest()
    assert ensure_model("https://example.invalid/x", good, tmp_path).read_bytes() == b"tampered"


def test_cached_model_matches_its_pinned_hash():
    s = get_settings()
    path = s.vad_cache_dir / "silero_vad.onnx"
    if not path.exists():
        pytest.skip("model not downloaded yet")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == s.vad_model_sha256


# --- audio ----------------------------------------------------------------------------------


def test_resampling_keeps_duration(tmp_path):
    pcm = slice_ms(speech_sample(), 0, 1.0)
    up = Resampler(16_000, 48_000)(pcm)
    assert abs(len(up) / 2 / 48_000 - 1.0) < 0.02
    write_wav(tmp_path / "x.wav", up, 48_000)
    back = read_wav(tmp_path / "x.wav", 16_000)  # read_wav resamples to the rate asked for
    assert abs(len(back) / 2 / 16_000 - 1.0) < 0.02


def test_output_buffer_pads_with_silence_and_barge_in_drops_stale_audio():
    out = OutputBuffer(24_000)
    out.put(b"\x01\x00" * 10, out.epoch)
    assert out.read(40) == b"\x01\x00" * 10 + b"\x00" * 20
    stale = out.epoch
    out.put(b"\x02\x00" * 10, stale)
    out.clear()  # barge-in
    assert not out.playing
    out.put(b"\x03\x00" * 10, stale)  # late audio from the interrupted reply
    assert not out.playing


# --- transcript assembly --------------------------------------------------------------------


def test_transcript_joins_segments_and_never_repeats_handed_over_words():
    t = _Transcript()
    t.on_event(SttEvent("partial", "my order"))
    t.on_event(SttEvent("end", "my order number is"))
    t.on_event(SttEvent("partial", "VX-8015"))
    assert t.take() == "my order number is VX-8015"
    # The open segment continues after the turn was handed over: only new words count.
    t.on_event(SttEvent("partial", "VX-8015 and also"))
    t.on_event(SttEvent("end", "VX-8015 and also a refund"))
    assert t.take() == "and also a refund"


# --- end of turn ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "complete"),
    [
        ("I want a refund", True),
        ("where is my order VX-8015", True),
        ("my order number is", False),
        ("my order number is v x one five", False),  # 2 of 4 digits
        ("my PIN is three zero two", False),  # 3 of 6 digits
        ("my PIN is 302647", True),
        ("I was wondering if", False),
        ("um", False),
        ("", False),
    ],
)
def test_looks_complete(text, complete):
    assert looks_complete(text) is complete


def test_unfinished_words_get_more_silence():
    kw = {"complete_ms": 600, "incomplete_ms": 1600}
    assert silence_needed_ms("thanks", **kw) == 600
    assert silence_needed_ms("my order number is", **kw) == 1600


# --- speech text ----------------------------------------------------------------------------


def test_sentences_are_released_as_soon_as_they_end():
    c = SentenceChunker()
    assert c.feed("Your order is on its way") == []
    assert c.feed(". It should arrive") == ["Your order is on its way."]
    assert c.feed(" tomorrow.") == ["It should arrive tomorrow."]
    assert c.feed(" Version 2.5 is") == []  # a decimal point isn't a sentence end
    assert c.flush() == ["Version 2.5 is"]


def test_long_run_on_text_still_starts_speaking():
    c = SentenceChunker(soft_after=40, hard_after=80)
    out = c.feed("and then we would ship it again and again " * 3)
    assert out and all(len(s) <= 80 for s in out)


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("Order VX-8015 is late.", "Order V X, 8, 0, 1, 5 is late."),
        ("Ticket TCK-000012.", "Ticket T C K, 0, 0, 0, 0, 1, 2."),
        ("Refund ₹3,499 to you.", "Refund ₹3,499 to you."),
    ],
)
def test_ids_are_read_back_one_character_at_a_time(text, spoken):
    assert for_speech(text) == spoken


@pytest.mark.parametrize(
    ("heard", "agent_gets"),
    [
        ("my order is v x eight zero one five", "my order is VX-8015"),
        ("order number fifteen oh eight please", "order number VX-1508 please"),
        ("VX one five oh eight", "VX-1508"),
        ("my PIN is three zero two six four seven", "my PIN is three zero two six four seven"),
        ("I have one order and two questions", "I have one order and two questions"),
    ],
)
def test_spoken_order_ids_are_normalized_only_next_to_order_or_vx(heard, agent_gets):
    assert normalize_transcript(heard) == agent_gets


class _FakeCartesiaSocket:
    """Answers each transcript with one audio chunk, and the closing request with done.
    A `stalled` socket never answers at all."""

    def __init__(self, stalled: bool = False) -> None:
        self.stalled = stalled
        self.inbox: asyncio.Queue = asyncio.Queue()

    async def send(self, payload: str) -> None:
        msg = json.loads(payload)
        ctx = msg["context_id"]
        if self.stalled:
            return
        if not msg["continue"]:
            self.inbox.put_nowait(json.dumps({"type": "done", "context_id": ctx}))
        else:
            data = base64.b64encode(b"\x00\x00" * 240).decode()
            self.inbox.put_nowait(json.dumps({"type": "chunk", "context_id": ctx, "data": data}))

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        return await self.inbox.get()

    async def close(self) -> None:
        pass


async def _speak(socket: _FakeCartesiaSocket, steps) -> int:
    settings = get_settings().model_copy(update={"tts_first_audio_timeout_s": 0.3})
    connection = CartesiaTtsConnection(socket, settings)
    utterance = connection.utterance()
    chunks = 0

    async def pump() -> None:
        nonlocal chunks
        async for _ in utterance.audio():
            chunks += 1

    task = asyncio.create_task(pump())
    try:
        for step in steps:
            await step(utterance)
        await utterance.finish()
        await task
    finally:
        task.cancel()
        await connection.close()
    return chunks


@pytest.mark.anyio
async def test_cartesia_tts_waits_out_a_slow_llm():
    """A pause in the reply (a tool call, an LLM retry) isn't a TTS failure: once Cartesia
    has spoken what it was given, it owes nothing until more text arrives."""

    async def first(u):
        await u.say("Let me check that.")

    async def llm_stalls(u):
        await asyncio.sleep(0.8)  # well past the 0.3 s timeout

    async def second(u):
        await u.say("Your order is on its way.")

    assert await _speak(_FakeCartesiaSocket(), [first, llm_stalls, second]) == 2


@pytest.mark.anyio
async def test_cartesia_tts_times_out_when_text_gets_no_audio():
    async def stalled(u):
        await u.say("Hello there.")

    with pytest.raises(TtsError, match="timed out"):
        await _speak(_FakeCartesiaSocket(stalled=True), [stalled])
