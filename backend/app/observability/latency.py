"""Latency report: p50/p95/p99 time to first audio, with a per-stage breakdown.

    # N replays of recorded customer audio through the real voice pipeline
    python -m app.observability.latency run ../voice-fixtures/customer-0*.wav --runs 10 \\
        --label baseline --out ../latency/baseline
    python -m app.observability.latency run FILES --runs 3 --fake --echo   # offline
    # or over turns already stored (live calls), from turn_metrics
    python -m app.observability.latency db --since 2026-10-30 --channel voice

Each run is a new call (a new VoiceSession and conversation) on the same agent, LLM client
and provider configuration, as one server process would serve successive callers. Audio is
paced in real time, so the end-of-turn wait and every provider call take their real time.

Stages (ms), measured in the voice session; they add up to time to first audio when the
reply's own speech is heard first (a filler can be heard earlier):
  vad                customer's last voiced frame → end of turn decided (silence wait)
  stt                → final transcript
  agent_first_text   → the agent's first reply text (model calls and tools until then)
  first_sentence     → enough text for TTS
  tts_first_audio    → its first audio
and from the agent's own row: llm_first_token (first model call), llm (all calls), tool.

`--paced-llm FIRST_MS,TAIL_MS` replaces the model with one that answers every turn with
the same three-sentence reply: the first text after FIRST_MS, the rest spread over TAIL_MS.
The voice pipeline, the agent loop and its tools stay real. It isolates voice-side changes
from a model's run-to-run variance (and from free-tier quotas); take the delays from
measured model calls.

Outputs: report.md (tables) and results.json (every turn, for before/after comparisons).
"""

import argparse
import asyncio
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.agent.llm import Completion, TextDelta
from app.agent.session import InMemorySessionStore
from app.config import Settings, get_settings
from app.db.models import TurnMetric
from app.main import build_agent
from app.observability import tracing
from app.voice.audio import read_wav
from app.voice.replay import EchoLLM, _vad, replay
from app.voice.session import VoiceSession
from app.voice.stt import CartesiaSttProvider, FakeSttProvider
from app.voice.tts import CartesiaTtsProvider, FakeTtsProvider

VOICE_STAGES = (
    "vad_ms",
    "stt_ms",
    "agent_first_text_ms",
    "first_sentence_ms",
    "tts_first_audio_ms",
)
AGENT_STAGES = ("llm_first_token_ms", "llm_ms", "tool_ms")
ROWS = ("time_to_first_audio_ms", *VOICE_STAGES, *AGENT_STAGES)
AGENT_COLUMNS = (*AGENT_STAGES, "llm_calls", "prompt_tokens", "completion_tokens", "cached_tokens")


PACED_REPLY = (
    "I can help with that. Could you tell me the email or phone number on your account, "
    "and your six digit PIN code? Then I'll look up the order for you."
)


class PacedLLM:
    """A model stand-in with fixed timing: first text after `first_ms`, the rest of
    PACED_REPLY spread evenly over `tail_ms`. No tools, no network."""

    model = "paced"

    def __init__(self, first_ms: float, tail_ms: float) -> None:
        self.first_s = first_ms / 1000
        self.tail_s = tail_ms / 1000

    async def stream(self, messages, tools, *, stop=None):
        words = PACED_REPLY.split(" ")
        started = time.monotonic()
        await asyncio.sleep(self.first_s)
        first = time.monotonic() - started
        for i, word in enumerate(words):
            if i:
                await asyncio.sleep(self.tail_s / (len(words) - 1))
            yield TextDelta(word if i == 0 else " " + word)
        yield Completion(text=PACED_REPLY, first_token_s=first)

    async def aclose(self) -> None:
        pass


def percentile(values: list[float], p: float) -> float | None:
    """Linear interpolation between closest ranks (numpy's default)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def summarize(turns: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    out = {}
    for key in ROWS:
        values = [t[key] for t in turns if t.get(key) is not None]
        out[key] = {
            "n": len(values),
            "p50": _round(percentile(values, 50)),
            "p95": _round(percentile(values, 95)),
            "p99": _round(percentile(values, 99)),
            "mean": _round(sum(values) / len(values)) if values else None,
        }
    return out


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


def render(label: str, turns: list[dict[str, Any]], config: dict[str, Any]) -> str:
    stats = summarize(turns)
    lines = [f"# Latency: {label}", ""]
    lines += [f"- {k}: `{v}`" for k, v in config.items()]
    heard = [t for t in turns if t.get("time_to_first_audio_ms") is not None]
    fillers = sum(1 for t in turns if t.get("first_audio_source") == "filler")
    lines += [
        f"- turns: {len(turns)} ({len(heard)} with audio, {fillers} where a filler came first)",
        "",
        "| ms | n | p50 | p95 | p99 | mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ROWS:
        s = stats[key]
        cells = [_cell(s[c]) for c in ("p50", "p95", "p99", "mean")]
        name = f"**{key}**" if key == "time_to_first_audio_ms" else key
        lines.append(f"| {name} | {s['n']} | " + " | ".join(cells) + " |")
    by_text: dict[str, list[float]] = {}
    for t in heard:
        by_text.setdefault(t["text"], []).append(t["time_to_first_audio_ms"])
    if by_text:
        lines += ["", "Time to first audio by customer line:", "", "| line | n | p50 | p95 |",
                  "|---|---:|---:|---:|"]  # fmt: skip
        for text, values in by_text.items():
            lines.append(
                f"| {text[:60]} | {len(values)} | {_cell(percentile(values, 50))} | "
                f"{_cell(percentile(values, 95))} |"
            )
    return "\n".join(lines) + "\n"


def _cell(value: float | None) -> str:
    return "-" if value is None else f"{value:,.0f}"


async def _agent_rows(engine, turn_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not turn_ids:
        return {}
    async with AsyncSession(engine) as db:
        rows = await db.scalars(select(TurnMetric).where(TurnMetric.turn_id.in_(turn_ids)))
        return {r.turn_id: {c: getattr(r, c) for c in AGENT_COLUMNS} for r in rows}


def _config(settings: Settings, args) -> dict[str, Any]:
    return {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "runs": args.runs,
        "llm": (
            f"paced (first {args.paced_llm[0]:.0f} ms, tail {args.paced_llm[1]:.0f} ms)"
            if args.paced_llm
            else "echo"
            if args.echo
            else settings.gemini_model
        ),
        "fast_model": settings.llm_fast_model or "-",
        "reasoning_effort": settings.llm_reasoning_effort,
        "llm_keepalive_s": settings.llm_keepalive_s,
        "llm_prewarm": settings.llm_prewarm,
        "tts_streaming": settings.tts_streaming,
        "stt_tts": "fake" if args.fake else "cartesia",
        "files": " ".join(Path(p).name for p in args.audio),
    }


async def run(args) -> int:
    settings = get_settings()
    tracing.setup_tracing(settings)
    inputs = [read_wav(Path(p)) for p in args.audio]
    script = [
        Path(p).with_suffix(".txt").read_text().strip()
        if Path(p).with_suffix(".txt").exists()
        else ""
        for p in args.audio
    ]
    engine = create_async_engine(settings.database_url())
    redis = Redis.from_url(settings.redis_url)
    stand_in = PacedLLM(*args.paced_llm) if args.paced_llm else EchoLLM() if args.echo else None
    agent, llm = build_agent(settings, engine, redis, llm=stand_in)
    if agent is None:
        print("GEMINI_API_KEY is not set; use --echo to run without an LLM.", file=sys.stderr)
        return 2
    agent.store = InMemorySessionStore()
    if agent.retriever is not None:
        await agent.retriever.embedder.warm()
    vad = _vad(settings)
    turns: list[dict[str, Any]] = []
    try:
        for i in range(args.runs):
            if args.fake:
                stt = FakeSttProvider(script, energy_threshold=0.01)
                tts = FakeTtsProvider(sample_rate=settings.tts_sample_rate)
            else:
                stt, tts = CartesiaSttProvider(settings), CartesiaTtsProvider(settings)
            session = VoiceSession(agent, settings, stt=stt, tts=tts, vad=vad)
            vad.reset()
            started = time.monotonic()
            try:
                await session.start()
                result = await replay(session, inputs, real_time=not args.fast)
            finally:
                await session.close()
            for turn in result.turns:
                turns.append({**turn, "run": i, "conversation_id": session.conversation_id})
            handed = f", handed over {result.handed_over}" if result.handed_over else ""
            ttfa = [t["first_audio_ms"] for t in result.turns]
            print(f"run {i + 1}/{args.runs}: first audio {ttfa} ms "
                  f"({time.monotonic() - started:.0f} s){handed}", flush=True)  # fmt: skip
            if args.pause_s and i < args.runs - 1:
                await asyncio.sleep(args.pause_s)
        agent_rows = await _agent_rows(engine, [t["turn_id"] for t in turns])
    finally:
        if llm is not None:
            await llm.aclose()
        await redis.aclose()
        await engine.dispose()
        tracing.shutdown_tracing()
    for turn in turns:
        turn["time_to_first_audio_ms"] = turn.pop("first_audio_ms")
        turn.update(agent_rows.get(turn["turn_id"], {}))
    _write(Path(args.out), args.label, turns, _config(settings, args))
    return 0


async def from_db(args) -> int:
    settings = get_settings()
    engine = create_async_engine(settings.database_url())
    try:
        async with async_sessionmaker(engine)() as db:
            query = select(TurnMetric).order_by(TurnMetric.started_at)
            if args.since:
                query = query.where(TurnMetric.started_at >= datetime.fromisoformat(args.since))
            if args.channel:
                query = query.where(TurnMetric.channel == args.channel)
            if args.conversation:
                query = query.where(TurnMetric.conversation_id.in_(args.conversation))
            rows = (await db.scalars(query)).all()
    finally:
        await engine.dispose()
    columns = [c.name for c in TurnMetric.__table__.columns]
    turns = [{c: getattr(r, c) for c in columns} | {"text": "(stored)"} for r in rows]
    config = {
        "source": "turn_metrics",
        "since": args.since or "-",
        "channel": args.channel or "all",
    }
    _write(Path(args.out), args.label, turns, config)
    return 0


def _write(out: Path, label: str, turns: list[dict[str, Any]], config: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    report = render(label, turns, config)
    (out / "report.md").write_text(report)
    results = {"label": label, "config": config, "summary": summarize(turns), "turns": turns}
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(report)
    print(f"Wrote {out}/report.md and results.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="p50/p95/p99 time to first audio, by stage.")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="replay recorded audio N times")
    r.add_argument("audio", nargs="+", help="WAV files, one customer turn each")
    r.add_argument("--runs", type=int, default=5)
    r.add_argument("--label", default="latency")
    r.add_argument("--out", default="latency")
    r.add_argument("--fake", action="store_true", help="fake STT/TTS (no Cartesia key)")
    r.add_argument("--echo", action="store_true", help="echo LLM (no Gemini key)")
    r.add_argument("--fast", action="store_true", help="don't pace audio in real time")
    r.add_argument("--pause-s", type=float, default=0.0, help="wait between runs (rate limits)")
    r.add_argument(
        "--paced-llm",
        type=lambda v: tuple(float(x) for x in v.split(",")),
        metavar="FIRST_MS,TAIL_MS",
        help="a fixed-timing model stand-in (see above)",
    )
    d = sub.add_parser("db", help="report over stored turn_metrics")
    d.add_argument("--since", help="ISO date or datetime")
    d.add_argument("--channel", choices=["voice", "text"])
    d.add_argument("--conversation", type=int, action="append")
    d.add_argument("--label", default="stored turns")
    d.add_argument("--out", default="latency")
    args = parser.parse_args(argv)
    return asyncio.run(run(args) if args.command == "run" else from_db(args))


if __name__ == "__main__":
    sys.exit(main())
