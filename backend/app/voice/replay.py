"""Run the voice pipeline against recorded audio instead of a microphone.

    python -m app.voice.replay run a.wav b.wav --out-dir out/          # Cartesia + Groq
    python -m app.voice.replay run a.wav b.wav --fake --echo           # fully offline
    python -m app.voice.replay run a.wav b.wav --barge-in-after-ms 800 # b interrupts a's reply
    python -m app.voice.replay fixtures out/fixtures/                  # sample customer audio

Each file is one customer turn: it's played into the same VoiceSession the WebRTC path
uses (VAD, end of turn, STT, agent, TTS, barge-in, fillers), then silence until the reply
has been played, then the next file. With `--barge-in-after-ms N` the next file starts N ms
after the reply's first audio instead, to test interruption.

Outputs in --out-dir: agent.wav (everything played, on the call's timeline), customer.wav,
events.jsonl, and summary.json (each turn's text, latency to first audio, filler, whether it
was interrupted).

`--fake` uses the fake STT (transcripts from a .txt next to each .wav) and fake TTS (a tone),
so only the VAD and the agent are real. `--echo` replaces the LLM with one that repeats
what it heard, so no Groq key is needed; the agent loop itself is still the real one.
"""

import argparse
import asyncio
import contextlib
import json
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import create_async_engine

from app.agent.llm import Completion, TextDelta
from app.agent.session import InMemorySessionStore
from app.config import Settings, get_settings
from app.main import build_agent
from app.voice.audio import VAD_RATE, read_wav, silence, write_wav
from app.voice.session import VoiceSession
from app.voice.stt import CartesiaSttProvider, FakeSttProvider, SttProvider
from app.voice.tts import CartesiaTtsProvider, FakeTtsProvider, TtsProvider, synthesize
from app.voice.vad import SileroVAD, SpeechScorer, ensure_model

TICK_MS = 20


class EchoLLM:
    """A stand-in LLM: answers every turn with what it heard. No tools, no network."""

    model = "echo"

    async def stream(self, messages, tools, *, stop=None) -> AsyncIterator:
        heard = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        heard = heard.replace("<customer_message>", "").replace("</customer_message>", "")
        text = f"I heard: {heard.strip()}. How can I help?"
        for word in text.split(" "):
            yield TextDelta(word + " ")
        yield Completion(text=text)

    async def aclose(self) -> None:
        pass


@dataclass
class ReplayResult:
    turns: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    agent_audio: bytes = b""
    customer_audio: bytes = b""
    handed_over: str | None = None


async def replay(
    session: VoiceSession,
    inputs: list[bytes],
    *,
    barge_in_after_ms: float | None = None,
    real_time: bool = False,
    max_wait_s: float = 90.0,
) -> ReplayResult:
    """Play `inputs` (16 kHz pcm16) into a started session, as described above."""
    result = ReplayResult()
    previous = session.on_event

    def record(event: dict[str, Any]) -> None:
        result.events.append({**event, "t_ms": round(len(customer) / 2 / VAD_RATE * 1000)})
        previous(event)

    session.on_event = record
    customer = bytearray()
    agent = bytearray()
    in_step = VAD_RATE * 2 * TICK_MS // 1000
    out_step = session.out.sample_rate * 2 * TICK_MS // 1000

    async def tick(pcm: bytes = b"") -> None:
        pcm = pcm.ljust(in_step, b"\x00")
        customer.extend(pcm)
        session.feed(pcm)
        agent.extend(session.out.read(out_step))
        await asyncio.sleep(TICK_MS / 1000 if real_time else 0.001)

    async def wait(condition) -> None:
        deadline = time.monotonic() + max_wait_s
        while not condition() and time.monotonic() < deadline and not session.ended.is_set():
            await tick()

    for index, pcm in enumerate(inputs):
        turns_before = len(session.metrics)
        for i in range(0, len(pcm), in_step):
            await tick(pcm[i : i + in_step])
        last = index == len(inputs) - 1
        if barge_in_after_ms is not None and not last:
            # Wait for this turn's reply to start playing, then talk over it.
            await wait(lambda n=turns_before: len(session.metrics) > n and session.out.playing)
            for _ in range(int(barge_in_after_ms // TICK_MS)):
                await tick()
        else:
            await wait(lambda: session.idle)
            for _ in range(int(400 // TICK_MS)):  # a natural gap before the next turn
                await tick()
    await wait(lambda: session.idle)
    session.on_event = previous

    result.turns = [asdict(m) for m in session.metrics]
    for turn in result.turns:
        turn.pop("end_of_speech", None)
    result.agent_audio = bytes(agent)
    result.customer_audio = bytes(customer)
    result.handed_over = session.escalation_reference if session.ended.is_set() else None
    return result


def _vad(settings: Settings) -> SpeechScorer:
    path = ensure_model(settings.vad_model_url, settings.vad_model_sha256, settings.vad_cache_dir)
    return SileroVAD(path)


async def _run(args) -> int:
    settings = get_settings()
    inputs = [read_wav(Path(p)) for p in args.audio]
    if args.fake:
        script = []
        for p in args.audio:
            sidecar = Path(p).with_suffix(".txt")
            script.append(sidecar.read_text().strip() if sidecar.exists() else "")
        stt: SttProvider = FakeSttProvider(script, energy_threshold=0.01)
        tts: TtsProvider = FakeTtsProvider(sample_rate=settings.tts_sample_rate)
    else:
        stt, tts = CartesiaSttProvider(settings), CartesiaTtsProvider(settings)

    engine = create_async_engine(settings.database_url())
    redis = Redis.from_url(settings.redis_url)
    echo = EchoLLM() if args.echo else None
    agent, llm = build_agent(settings, engine, redis, llm=echo)
    if agent is None:
        print("GROQ_API_KEY is not set; use --echo to run without an LLM.", file=sys.stderr)
        return 2
    agent.store = InMemorySessionStore()
    if agent.retriever is not None:
        await agent.retriever.embedder.warm()
    session = VoiceSession(agent, settings, stt=stt, tts=tts, vad=_vad(settings))
    try:
        await session.start()
        result = await replay(
            session,
            inputs,
            barge_in_after_ms=args.barge_in_after_ms,
            real_time=args.real_time or not args.fake,
        )
    finally:
        await session.close()
        if llm is not None:
            await llm.aclose()
        await redis.aclose()
        await engine.dispose()

    out = Path(args.out_dir)
    write_wav(out / "agent.wav", result.agent_audio, settings.tts_sample_rate)
    write_wav(out / "customer.wav", result.customer_audio, VAD_RATE)
    (out / "events.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False, default=str) + "\n" for e in result.events)
    )
    summary = {"turns": result.turns, "handed_over": result.handed_over}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for turn in result.turns:
        flags = [f for f in ("filler", "interrupted") if turn[f]]
        print(f"customer: {turn['text']!r}  first audio: {turn['first_audio_ms']} ms  "
              f"{' '.join(flags)}")  # fmt: skip
    if result.handed_over:
        print(f"handed over: {result.handed_over}")
    print(f"Wrote {out}/agent.wav, customer.wav, events.jsonl, summary.json")
    return 0


DEFAULT_LINES = (
    "Hi, where is my order?",
    "My email is meera dot pillai at example dot com and my PIN is three zero two six four seven.",
    "It's order V X eight zero one five.",
)


async def _fixtures(args) -> int:
    """Customer lines as 16 kHz WAVs (with .txt transcripts), spoken by Cartesia TTS."""
    settings = get_settings().model_copy(update={"tts_sample_rate": VAD_RATE})
    out = Path(args.out_dir)
    connection = await CartesiaTtsProvider(settings).connect()
    try:
        for i, line in enumerate(args.lines or DEFAULT_LINES, start=1):
            pcm = await synthesize(connection, line)
            path = out / f"customer-{i:02d}.wav"
            write_wav(path, silence(300, VAD_RATE) + pcm + silence(300, VAD_RATE), VAD_RATE)
            path.with_suffix(".txt").write_text(line + "\n")
            print(f"Wrote {path}")
    finally:
        with contextlib.suppress(Exception):
            await connection.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Voice pipeline on recorded audio.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="play WAV files into the voice pipeline")
    run.add_argument("audio", nargs="+", help="WAV files, one customer turn each")
    run.add_argument("--out-dir", default="voice-replay")
    run.add_argument("--fake", action="store_true", help="fake STT/TTS (no Cartesia key)")
    run.add_argument("--echo", action="store_true", help="echo LLM (no Groq key)")
    run.add_argument("--barge-in-after-ms", type=float, help="each next file interrupts")
    run.add_argument("--real-time", action="store_true", help="pace audio in real time")
    fixtures = sub.add_parser("fixtures", help="synthesize customer WAVs with Cartesia TTS")
    fixtures.add_argument("out_dir")
    fixtures.add_argument("--line", dest="lines", action="append", help="repeatable")
    args = parser.parse_args(argv)
    return asyncio.run(_run(args) if args.command == "run" else _fixtures(args))


if __name__ == "__main__":
    sys.exit(main())
