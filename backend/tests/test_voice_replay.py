"""The voice pipeline on recorded speech: real Silero VAD on real speech audio, fake STT and
TTS, the echo LLM through the real agent loop. Also the replay CLI end to end."""

import json

import pytest

from app.config import get_settings
from app.voice.audio import VAD_RATE, read_wav, write_wav
from app.voice.replay import EchoLLM, main, replay
from app.voice.session import VoiceSession
from app.voice.stt import FakeSttProvider
from app.voice.tts import FakeTtsProvider
from app.voice.vad import SileroVAD, ensure_model
from tests.agent_helpers import make_agent
from tests.voice_helpers import UTTERANCES, quiet, slice_ms, speech_sample

pytestmark = pytest.mark.anyio


@pytest.fixture(scope="module")
def vad_path():
    s = get_settings()
    try:
        return ensure_model(s.vad_model_url, s.vad_model_sha256, s.vad_cache_dir)
    except Exception as exc:
        pytest.skip(f"Silero model unavailable: {exc}")


def _utterances(n: int) -> list[bytes]:
    speech = speech_sample()
    return [slice_ms(speech, *UTTERANCES[i]) + quiet(300) for i in range(n)]


def _session(tx_db, vad_path, script, **tts):
    agent = make_agent(tx_db, EchoLLM())
    settings = get_settings().model_copy(update={"stt_final_wait_ms": 100})
    return VoiceSession(
        agent,
        settings,
        stt=FakeSttProvider(script, energy_threshold=0.01),
        tts=FakeTtsProvider(**tts),
        vad=SileroVAD(vad_path),
    )


@pytest.mark.db
async def test_two_recorded_turns_are_heard_and_answered(tx_db, vad_path):
    session = _session(tx_db, vad_path, ["where is my order", "it is VX-8015"], ms_per_char=5.0)
    await session.start()
    result = await replay(session, _utterances(2))
    await session.close()
    assert [t["text"] for t in result.turns] == ["where is my order", "it is VX-8015"]
    assert all(t["first_audio_ms"] is not None and not t["interrupted"] for t in result.turns)
    assert any(result.agent_audio)


@pytest.mark.db
async def test_recorded_barge_in_interrupts_the_reply(tx_db, vad_path):
    session = _session(tx_db, vad_path, ["tell me everything", "stop"], ms_per_char=40.0)
    await session.start()
    result = await replay(session, _utterances(2), barge_in_after_ms=600)
    await session.close()
    assert [t["interrupted"] for t in result.turns] == [True, False]
    assert "barge_in" in [e["type"] for e in result.events]


def test_cli_runs_offline_and_writes_its_outputs(tmp_path, vad_path, monkeypatch, test_db_settings):
    speech = speech_sample()
    for i, line in enumerate(["where is my order", "thanks"]):
        wav = tmp_path / f"turn-{i}.wav"
        write_wav(wav, slice_ms(speech, *UTTERANCES[i]) + quiet(300), VAD_RATE)
        wav.with_suffix(".txt").write_text(line)
    out = tmp_path / "out"
    monkeypatch.setenv("GEMINI_API_KEY", "")  # --echo needs no key
    get_settings.cache_clear()
    assert main(["run", str(tmp_path / "turn-0.wav"), str(tmp_path / "turn-1.wav"),
                 "--fake", "--echo", "--out-dir", str(out)]) == 0  # fmt: skip
    summary = json.loads((out / "summary.json").read_text())
    assert [t["text"] for t in summary["turns"]] == ["where is my order", "thanks"]
    assert len(read_wav(out / "agent.wav", get_settings().tts_sample_rate)) > 0
    assert (out / "events.jsonl").read_text().count('"type": "user"') == 2
