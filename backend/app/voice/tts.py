"""Streaming text-to-speech.

Cartesia Sonic over one WebSocket per call. Each agent reply is a *context*: sentences are
pushed as they're ready (`continue: true`) and the reply ends with an empty transcript and
`continue: false`. Cancelling a context stops new generation, but Cartesia still sends
audio already being generated, so cancelled contexts' chunks are dropped here: once the
customer barges in, nothing more from that reply is ever played.
"""

import asyncio
import base64
import contextlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Protocol
from urllib.parse import urlencode

import numpy as np
from websockets.asyncio.client import ClientConnection, connect

from app.config import Settings


class TtsError(Exception):
    pass


class Utterance(Protocol):
    async def say(self, text: str) -> None: ...

    async def finish(self) -> None: ...

    async def cancel(self) -> None: ...

    def audio(self) -> AsyncIterator[bytes]: ...


class TtsConnection(Protocol):
    sample_rate: int

    def utterance(self) -> Utterance: ...

    async def close(self) -> None: ...


class TtsProvider(Protocol):
    async def connect(self) -> TtsConnection: ...


async def synthesize(connection: TtsConnection, text: str) -> bytes:
    """A whole phrase as one PCM buffer (for fillers and fallback lines cached at start)."""
    utterance = connection.utterance()
    await utterance.say(text)
    await utterance.finish()
    return b"".join([chunk async for chunk in utterance.audio()])


_END = object()


class CartesiaTtsProvider:
    def __init__(self, settings: Settings) -> None:
        if settings.cartesia_api_key is None or not settings.cartesia_api_key.get_secret_value():
            raise TtsError("CARTESIA_API_KEY is not set")
        self.settings = settings
        query = urlencode({"cartesia_version": settings.cartesia_version})
        self.url = f"{settings.cartesia_base_url}/tts/websocket?{query}"

    async def connect(self) -> TtsConnection:
        headers = {
            "X-API-Key": self.settings.cartesia_api_key.get_secret_value(),
            "Cartesia-Version": self.settings.cartesia_version,
        }
        try:
            ws = await asyncio.wait_for(
                connect(self.url, additional_headers=headers, max_size=2**24),
                self.settings.stt_connect_timeout_s,
            )
        except Exception as exc:
            raise TtsError(f"TTS connection failed: {type(exc).__name__}") from exc
        return CartesiaTtsConnection(ws, self.settings)


class CartesiaTtsConnection:
    def __init__(self, ws: ClientConnection, settings: Settings) -> None:
        self._ws = ws
        self.settings = settings
        self.sample_rate = settings.tts_sample_rate
        self._queues: dict[str, asyncio.Queue] = {}
        self._reader = asyncio.create_task(self._read())

    def _request(self, context_id: str, transcript: str, *, more: bool) -> str:
        # Every field but transcript and continue must stay the same within a context.
        return json.dumps(
            {
                "model_id": self.settings.cartesia_tts_model,
                "transcript": transcript,
                "voice": {"id": self.settings.cartesia_voice_id},
                "language": self.settings.cartesia_language,
                "output_format": {
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": self.sample_rate,
                },
                "context_id": context_id,
                "continue": more,
                # Whole sentences are sent, so there's nothing to wait for: speak at once.
                "max_buffer_delay_ms": 0,
            }
        )

    async def _send(self, payload: str) -> None:
        try:
            await self._ws.send(payload)
        except Exception as exc:
            raise TtsError(f"TTS send failed: {type(exc).__name__}") from exc

    async def _read(self) -> None:
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                queue = self._queues.get(msg.get("context_id", ""))
                if queue is None:
                    continue  # cancelled or finished: drop late audio
                kind = msg.get("type")
                if kind == "chunk" and msg.get("data"):
                    queue.put_nowait(base64.b64decode(msg["data"]))
                elif kind == "done":
                    queue.put_nowait(_END)
                elif kind == "error":
                    queue.put_nowait(TtsError(f"TTS error {msg.get('status_code')}: "
                                              f"{msg.get('message')}"))  # fmt: skip
        except Exception as exc:
            error = TtsError(f"TTS connection lost: {type(exc).__name__}")
        else:
            error = TtsError("TTS connection closed")
        for queue in self._queues.values():
            queue.put_nowait(error)

    def utterance(self) -> Utterance:
        context_id = uuid.uuid4().hex
        self._queues[context_id] = asyncio.Queue()
        return _CartesiaUtterance(self, context_id)

    async def close(self) -> None:
        self._reader.cancel()
        with contextlib.suppress(Exception):
            await self._ws.close()


class _CartesiaUtterance:
    """The timeout only runs while Cartesia owes us audio: from text sent until audio arrives,
    and after finish() until done. A context that has spoken everything it was given is
    silent until the LLM sends more, and that can take a while (a tool call, a 429 retry)."""

    def __init__(self, connection: CartesiaTtsConnection, context_id: str) -> None:
        self.c = connection
        self.context_id = context_id
        self._owed_since: float | None = None
        self._finished = False

    async def say(self, text: str) -> None:
        if self._owed_since is None:
            self._owed_since = time.monotonic()
        await self.c._send(self.c._request(self.context_id, text + " ", more=True))

    async def finish(self) -> None:
        self._finished = True
        self._owed_since = time.monotonic()
        await self.c._send(self.c._request(self.context_id, "", more=False))

    async def cancel(self) -> None:
        self.c._queues.pop(self.context_id, None)  # drop anything still arriving
        with contextlib.suppress(TtsError):
            await self.c._send(json.dumps({"context_id": self.context_id, "cancel": True}))

    async def audio(self) -> AsyncIterator[bytes]:
        queue = self.c._queues.get(self.context_id)
        if queue is None:
            return
        timeout = self.c.settings.tts_first_audio_timeout_s
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), min(timeout, 0.25))
                except TimeoutError as exc:
                    owed = self._owed_since
                    if owed is not None and time.monotonic() - owed > timeout:
                        raise TtsError("TTS timed out") from exc
                    continue
                if item is _END:
                    return
                if isinstance(item, Exception):
                    raise item
                # More audio may still be owed after finish(); otherwise we're caught up.
                self._owed_since = time.monotonic() if self._finished else None
                yield item
        finally:
            self.c._queues.pop(self.context_id, None)


class FakeTtsProvider:
    """For tests and offline replay: a quiet tone, about 60 ms per character, delivered in
    20 ms chunks with an optional delay. Records what was spoken.

    Failure injection: an utterance whose first sentence contains one of `fail_texts` fails
    after its first chunk (each entry is used once); `fail_connects` fails connecting."""

    def __init__(
        self,
        *,
        sample_rate: int = 24_000,
        ms_per_char: float = 60.0,
        first_chunk_delay_s: float = 0.0,
        fail_texts: list[str] | None = None,
        fail_connects: int = 0,
    ) -> None:
        self.sample_rate = sample_rate
        self.ms_per_char = ms_per_char
        self.first_chunk_delay_s = first_chunk_delay_s
        self.fail_texts = list(fail_texts or [])
        self.fail_connects = fail_connects
        self.spoken: list[str] = []  # every sentence sent for speech, in order
        self.cancelled = 0
        self.connects = 0

    async def connect(self) -> TtsConnection:
        self.connects += 1
        if self.fail_connects > 0:
            self.fail_connects -= 1
            raise TtsError("fake TTS: connection refused")
        return _FakeTtsConnection(self)


class _FakeTtsConnection:
    def __init__(self, provider: FakeTtsProvider) -> None:
        self.p = provider
        self.sample_rate = provider.sample_rate

    def utterance(self) -> Utterance:
        return _FakeUtterance(self.p)

    async def close(self) -> None:
        pass


class _FakeUtterance:
    def __init__(self, provider: FakeTtsProvider) -> None:
        self.p = provider
        self.fail = False
        self._said = 0
        self._queue: asyncio.Queue = asyncio.Queue()
        self._cancelled = False

    async def say(self, text: str) -> None:
        self._said += 1
        if self._said == 1:  # an utterance fails on its first sentence's text
            hit = next((t for t in self.p.fail_texts if t in text), None)
            if hit is not None:
                self.p.fail_texts.remove(hit)
                self.fail = True
        self.p.spoken.append(text)
        self._queue.put_nowait(text)

    async def finish(self) -> None:
        self._queue.put_nowait(_END)

    async def cancel(self) -> None:
        self._cancelled = True
        self.p.cancelled += 1
        self._queue.put_nowait(_END)

    async def audio(self) -> AsyncIterator[bytes]:
        first = True
        while True:
            item = await self._queue.get()
            if item is _END or self._cancelled:
                return
            samples = int(self.p.sample_rate * len(item) * self.p.ms_per_char / 1000)
            t = np.arange(samples) / self.p.sample_rate
            pcm = (0.1 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2").tobytes()
            step = self.p.sample_rate * 2 // 50  # 20 ms
            for i in range(0, len(pcm), step):
                if first and self.p.first_chunk_delay_s:
                    await asyncio.sleep(self.p.first_chunk_delay_s)
                if self._cancelled:
                    return
                yield pcm[i : i + step]
                if first and self.fail:
                    raise TtsError("fake TTS: connection lost")
                first = False
                await asyncio.sleep(0)
