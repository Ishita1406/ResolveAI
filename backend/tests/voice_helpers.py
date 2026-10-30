"""Audio and drivers for voice tests.

Two kinds of input audio:
- synthetic: a loud noise burst is "speech" and zeros are silence, scored by
  `EnergyScorer`. Deterministic and instant, for the turn-taking logic.
- real speech: slices of Silero's English sample (downloaded once into the cache, checked
  by hash), for the Silero VAD tests. Skipped if it can't be fetched.

`Driver` advances a call tick by tick (20 ms in, 20 ms out) faster than real time: the VAD
and the fake STT count audio, not wall-clock time. Timers that are wall-clock (fillers)
are exercised with `real_time=True`.
"""

import asyncio
import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from app.config import get_settings
from app.voice.audio import VAD_RATE, pcm16_to_float, read_wav, silence

SPEECH_URL = "https://models.silero.ai/vad_models/en.wav"
SPEECH_SHA256 = "89f17d9c94c4b31eb320f424628bcbc920abaddbee6e2760fd868bfb1d9a2e47"
# Utterances in that file (seconds), found with Silero itself.
UTTERANCES = [(0.0, 2.1), (2.65, 4.85), (4.95, 6.85), (20.3, 23.35)]


def noise(ms: float, *, amplitude: float = 0.3, seed: int = 0) -> bytes:
    n = int(VAD_RATE * ms / 1000)
    samples = np.random.default_rng(seed).uniform(-amplitude, amplitude, n)
    return (samples * 32767).astype("<i2").tobytes()


def quiet(ms: float) -> bytes:
    return silence(ms, VAD_RATE)


class EnergyScorer:
    """Speech probability 1.0 for loud frames, 0.0 otherwise."""

    def score(self, frame: bytes) -> float:
        return 1.0 if float(np.abs(pcm16_to_float(frame)).mean()) > 0.05 else 0.0

    def reset(self) -> None:
        pass


def speech_sample() -> bytes:
    """Silero's 60 s English sample as 16 kHz pcm16, or skip the test."""
    path = get_settings().vad_cache_dir / "en_example.wav"
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != SPEECH_SHA256:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(SPEECH_URL, timeout=30) as response:
                data = response.read()
        except Exception as exc:
            pytest.skip(f"speech sample unavailable: {exc}")
        if hashlib.sha256(data).hexdigest() != SPEECH_SHA256:
            pytest.skip("speech sample has an unexpected hash")
        path.write_bytes(data)
    return read_wav(Path(path))


def slice_ms(pcm: bytes, start_s: float, end_s: float) -> bytes:
    a, b = int(start_s * VAD_RATE) * 2, int(end_s * VAD_RATE) * 2
    return pcm[a:b]


class Driver:
    TICK_MS = 20

    def __init__(self, session, *, real_time: bool = False) -> None:
        self.session = session
        self.real_time = real_time
        self.events: list[dict] = []
        self.output = bytearray()
        session.on_event = self.events.append

    async def tick(self, pcm: bytes = b"") -> None:
        frame_in = len(quiet(self.TICK_MS))
        pcm = pcm.ljust(frame_in, b"\x00")
        self.session.feed(pcm)
        out_bytes = self.session.out.sample_rate * 2 * self.TICK_MS // 1000
        self.output += self.session.read_audio(out_bytes)
        await asyncio.sleep(self.TICK_MS / 1000 if self.real_time else 0.002)

    async def play(self, pcm: bytes) -> None:
        step = len(quiet(self.TICK_MS))
        for i in range(0, len(pcm), step):
            await self.tick(pcm[i : i + step])

    async def until(self, condition, *, timeout_ticks: int = 500) -> None:
        for _ in range(timeout_ticks):
            if condition():
                return
            await self.tick()
        raise AssertionError(f"condition not met; events: {[e['type'] for e in self.events]}")

    async def settle(self) -> None:
        """Until the agent has finished and everything queued has been played."""
        await self.until(lambda: self.session.idle)

    def kinds(self) -> list[str]:
        return [e["type"] for e in self.events]

    def of(self, kind: str) -> list[dict]:
        return [e for e in self.events if e["type"] == kind]
