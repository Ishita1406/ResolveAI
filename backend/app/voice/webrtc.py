"""WebRTC transport with aiortc: audio in and out, plus a data channel for events.

aiortc is used for transport only (ICE, DTLS/SRTP, Opus). Everything about the call (VAD,
turn-taking, barge-in, the agent, safety) happens in VoiceSession.

    POST /voice/offer  {"sdp": ..., "type": "offer"}  ->  {"sdp": ..., "type": "answer",
                                                           "session_id": ...}
    GET  /voice        a minimal browser page to try it

Signalling is one HTTP request (no trickle ICE): aiortc gathers its candidates before
answering, and the page waits for its own before offering. Enough for localhost and a LAN;
across NATs set WEBRTC_ICE_SERVERS (STUN, or TURN).
"""

import asyncio
import contextlib
import fractions
import json
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import av
import numpy as np
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import MediaStreamError
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.voice.audio import VAD_RATE
from app.voice.session import VoiceSession
from app.voice.stt import CartesiaSttProvider, SttError, SttProvider
from app.voice.tts import CartesiaTtsProvider, TtsError, TtsProvider
from app.voice.vad import SileroVAD, SpeechScorer, ensure_model

log = logging.getLogger(__name__)
router = APIRouter()
STATIC = Path(__file__).parent / "static"
FRAME_MS = 20

Providers = tuple[SttProvider, TtsProvider, SpeechScorer]
ProviderFactory = Callable[[Settings], Awaitable[Providers]]


async def cartesia_providers(settings: Settings) -> Providers:
    """Production: Cartesia Ink and Sonic, and Silero VAD (downloaded on first use)."""
    stt, tts = CartesiaSttProvider(settings), CartesiaTtsProvider(settings)
    path = await asyncio.to_thread(
        ensure_model, settings.vad_model_url, settings.vad_model_sha256, settings.vad_cache_dir
    )
    return stt, tts, SileroVAD(path)


class SessionAudioTrack(MediaStreamTrack):
    """The agent's voice: 20 ms frames from the session's output buffer, paced in real time
    (silence when nothing is being said)."""

    kind = "audio"

    def __init__(self, session: VoiceSession) -> None:
        super().__init__()
        self.session = session
        self.rate = session.out.sample_rate
        self.samples = self.rate * FRAME_MS // 1000
        self._start: float | None = None
        self._timestamp = 0

    async def recv(self) -> av.AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError
        if self._start is None:
            self._start = time.monotonic()
        else:
            self._timestamp += self.samples
            wait = self._start + self._timestamp / self.rate - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
        pcm = self.session.out.read(self.samples * 2)
        frame = av.AudioFrame.from_ndarray(
            np.frombuffer(pcm, dtype="<i2").reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = self.rate
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self.rate)
        return frame


async def _consume(track: MediaStreamTrack, session: VoiceSession) -> None:
    """The customer's microphone, as 16 kHz mono pcm16, into the session."""
    resampler = av.AudioResampler(format="s16", layout="mono", rate=VAD_RATE)
    with contextlib.suppress(MediaStreamError):
        while True:
            frame = await track.recv()
            for out in resampler.resample(frame):
                session.feed(bytes(out.planes[0])[: out.samples * 2])


class Offer(BaseModel):
    sdp: str
    type: str


class Answer(BaseModel):
    sdp: str
    type: str
    session_id: str


@router.post("/voice/offer")
async def offer(body: Offer, request: Request) -> Answer:
    app = request.app
    agent = getattr(app.state, "agent", None)
    if agent is None:
        raise HTTPException(503, "The assistant isn't configured (GROQ_API_KEY).")
    settings = getattr(app.state, "settings", None) or get_settings()
    factory: ProviderFactory = getattr(app.state, "voice_providers", None) or cartesia_providers
    try:
        stt, tts, vad = await factory(settings)
    except (SttError, TtsError) as exc:
        raise HTTPException(503, f"Voice isn't available: {exc}") from exc

    session = VoiceSession(agent, settings, stt=stt, tts=tts, vad=vad)
    pc = RTCPeerConnection(
        RTCConfiguration(iceServers=[RTCIceServer(urls=u) for u in settings.webrtc_ice_servers])
    )
    calls: set = app.state.__dict__.setdefault("voice_calls", set())
    calls.add(pc)
    pc.addTrack(SessionAudioTrack(session))
    tasks: list[asyncio.Task] = []

    @pc.on("datachannel")
    def on_datachannel(channel) -> None:
        def send(event: dict) -> None:
            if channel.readyState == "open":
                channel.send(json.dumps(event, default=str))

        session.on_event = send

    @pc.on("track")
    def on_track(track: MediaStreamTrack) -> None:
        if track.kind == "audio":
            tasks.append(asyncio.create_task(_consume(track, session)))

    @pc.on("connectionstatechange")
    async def on_state() -> None:
        if pc.connectionState in ("failed", "closed"):
            await _hang_up(pc, session, tasks, calls)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=body.sdp, type=body.type))
    await pc.setLocalDescription(await pc.createAnswer())
    session_id = await session.start()
    return Answer(sdp=pc.localDescription.sdp, type=pc.localDescription.type, session_id=session_id)


async def _hang_up(pc, session: VoiceSession, tasks: list[asyncio.Task], calls: set) -> None:
    calls.discard(pc)
    for task in tasks:
        task.cancel()
    await session.close()
    await pc.close()


async def close_all(app) -> None:
    for pc in list(getattr(app.state, "voice_calls", ())):
        with contextlib.suppress(Exception):
            await pc.close()


@router.get("/voice", include_in_schema=False)
async def page() -> FileResponse:
    return FileResponse(STATIC / "index.html")
