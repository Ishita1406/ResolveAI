"""Streaming speech-to-text.

Cartesia Ink 2 on the turns endpoint (/stt/turns/websocket), the one that streams partial
transcripts (`turn.update`, cumulative within a segment). Its own turn detection is used
only as segmentation: whether the customer has finished is decided by our VAD and the
semantic check in turn.py, so a `turn.end` mid-sentence just starts another segment.
(The manual-finalize endpoint gives no partials with Ink 2, which the brief requires.)
"""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol
from urllib.parse import urlencode

from websockets.asyncio.client import ClientConnection, connect

from app.config import Settings
from app.voice.audio import VAD_RATE, pcm16_to_float


class SttError(Exception):
    pass


@dataclass(frozen=True)
class SttEvent:
    kind: Literal["start", "partial", "end"]
    text: str = ""  # cumulative within the current segment


class SttStream(Protocol):
    async def send(self, pcm: bytes) -> None: ...

    def events(self) -> AsyncIterator[SttEvent]: ...

    async def close(self) -> None: ...


class SttProvider(Protocol):
    async def connect(self) -> SttStream: ...


class CartesiaSttProvider:
    def __init__(self, settings: Settings) -> None:
        if settings.cartesia_api_key is None or not settings.cartesia_api_key.get_secret_value():
            raise SttError("CARTESIA_API_KEY is not set")
        self.settings = settings
        query = urlencode(
            {
                "model": settings.cartesia_stt_model,
                "encoding": "pcm_s16le",
                "sample_rate": VAD_RATE,
                "cartesia_version": settings.cartesia_version,
            }
        )
        self.url = f"{settings.cartesia_base_url}/stt/turns/websocket?{query}"

    async def connect(self) -> SttStream:
        headers = {
            "X-API-Key": self.settings.cartesia_api_key.get_secret_value(),
            "Cartesia-Version": self.settings.cartesia_version,
        }
        try:
            ws = await asyncio.wait_for(
                connect(self.url, additional_headers=headers, max_size=2**22),
                self.settings.stt_connect_timeout_s,
            )
        except (TimeoutError, OSError, Exception) as exc:
            raise SttError(f"STT connection failed: {type(exc).__name__}") from exc
        return CartesiaSttStream(ws)


class CartesiaSttStream:
    CHUNK_BYTES = VAD_RATE * 2 // 10  # ~100 ms, as Cartesia recommends

    def __init__(self, ws: ClientConnection) -> None:
        self._ws = ws
        self._buffer = bytearray()

    async def send(self, pcm: bytes) -> None:
        # The endpoint expects a continuous stream, silence included.
        self._buffer += pcm
        if len(self._buffer) >= self.CHUNK_BYTES:
            chunk, self._buffer = bytes(self._buffer), bytearray()
            try:
                await self._ws.send(chunk)
            except Exception as exc:
                raise SttError(f"STT send failed: {type(exc).__name__}") from exc

    async def events(self) -> AsyncIterator[SttEvent]:
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "turn.start":
                    yield SttEvent("start")
                elif kind in ("turn.update", "turn.eager_end"):
                    yield SttEvent("partial", msg.get("transcript", ""))
                elif kind == "turn.end":
                    yield SttEvent("end", msg.get("transcript", ""))
                elif kind == "error":
                    raise SttError(f"STT error {msg.get('status_code')}: {msg.get('message')}")
        except SttError:
            raise
        except Exception as exc:
            raise SttError(f"STT connection lost: {type(exc).__name__}") from exc
        raise SttError("STT connection closed")

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            await self._ws.send(json.dumps({"type": "close"}))
            await self._ws.close()


class FakeSttProvider:
    """For tests and offline replay. Each burst of sound (by energy) becomes the next
    scripted utterance: partials grow word by word while sound continues, and the segment
    ends after 300 ms of quiet, like a real streaming recognizer.

    `fail_connects` / `fail_after_s` inject failures."""

    def __init__(
        self,
        script: list[str],
        *,
        fail_connects: int = 0,
        fail_after_s: float | None = None,
        energy_threshold: float = 0.02,
    ) -> None:
        self.script = list(script)
        self.fail_connects = fail_connects
        self.fail_after_s = fail_after_s
        self.energy_threshold = energy_threshold
        self.connects = 0
        self.received_ms = 0.0

    async def connect(self) -> SttStream:
        self.connects += 1
        if self.fail_connects > 0:
            self.fail_connects -= 1
            raise SttError("fake STT: connection refused")
        return _FakeSttStream(self)


class _FakeSttStream:
    def __init__(self, provider: FakeSttProvider) -> None:
        self.p = provider
        self._queue: asyncio.Queue = asyncio.Queue()
        self._sound_ms = 0.0
        self._quiet_ms = 0.0
        self._active = False
        self._closed = False

    async def send(self, pcm: bytes) -> None:
        if self._closed:
            raise SttError("fake STT: closed")
        ms = len(pcm) / 2 / VAD_RATE * 1000
        self.p.received_ms += ms
        if self.p.fail_after_s is not None and self.p.received_ms >= self.p.fail_after_s * 1000:
            self.p.fail_after_s = None
            self._queue.put_nowait(SttError("fake STT: connection lost"))
            return
        loud = float(abs(pcm16_to_float(pcm)).mean()) > self.p.energy_threshold if pcm else False
        if loud:
            self._quiet_ms = 0.0
            if not self._active:
                self._active = True
                self._sound_ms = 0.0
                self._queue.put_nowait(SttEvent("start"))
            self._sound_ms += ms
            words = (self.p.script[0] if self.p.script else "").split()
            shown = max(1, min(len(words), int(self._sound_ms // 250) + 1))
            self._queue.put_nowait(SttEvent("partial", " ".join(words[:shown])))
        elif self._active:
            self._quiet_ms += ms
            if self._quiet_ms >= 300:
                self._active = False
                text = self.p.script.pop(0) if self.p.script else ""
                self._queue.put_nowait(SttEvent("end", text))

    async def events(self) -> AsyncIterator[SttEvent]:
        while True:
            item = await self._queue.get()
            if isinstance(item, Exception):
                raise item
            if item is None:
                return
            yield item

    async def close(self) -> None:
        self._closed = True
        self._queue.put_nowait(None)
