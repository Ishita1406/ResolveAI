"""WebRTC end to end over loopback: an aiortc client sends audio to the real /voice/offer
endpoint, the server's VoiceSession (fake STT/TTS, scripted LLM) answers, and the client
gets the agent's audio back plus events on the data channel."""

import asyncio
import contextlib
import fractions
import json
import time

import av
import httpx
import numpy as np
import pytest
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError

from app.main import create_app
from app.voice.stt import FakeSttProvider
from app.voice.tts import FakeTtsProvider
from tests.agent_helpers import ScriptedLLM, say
from tests.voice_helpers import EnergyScorer

pytestmark = [pytest.mark.anyio, pytest.mark.db, pytest.mark.redis]


class Microphone(MediaStreamTrack):
    """0.6 s of loud noise ("speech"), then silence, at 48 kHz in 20 ms frames."""

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self.rate, self.samples, self.sent = 48_000, 960, 0
        self.start = None

    async def recv(self) -> av.AudioFrame:
        if self.start is None:
            self.start = time.monotonic()
        else:
            wait = self.start + self.sent / self.rate - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
        loud = self.sent < self.rate * 0.6
        rng = np.random.default_rng(self.sent)
        samples = rng.uniform(-0.3, 0.3, self.samples) if loud else np.zeros(self.samples)
        frame = av.AudioFrame.from_ndarray(
            (samples * 32767).astype("<i2").reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate, frame.pts = self.rate, self.sent
        frame.time_base = fractions.Fraction(1, self.rate)
        self.sent += self.samples
        return frame


async def test_a_call_over_webrtc(test_db_settings):
    settings = test_db_settings.model_copy(update={"stt_final_wait_ms": 100})
    app = create_app(settings, llm=ScriptedLLM(say("Hello, how can I help you today?")))
    tts = FakeTtsProvider(ms_per_char=20.0)

    async def fakes(_settings):
        return FakeSttProvider(["hi there"], energy_threshold=0.05), tts, EnergyScorer()

    async with app.router.lifespan_context(app):
        app.state.voice_providers = fakes
        client = RTCPeerConnection()
        events: list[dict] = []
        heard = asyncio.Event()
        client.addTrack(Microphone())
        channel = client.createDataChannel("events")
        channel.on("message", lambda m: events.append(json.loads(m)))

        @client.on("track")
        def on_track(track):
            async def listen():
                with contextlib.suppress(MediaStreamError):  # the call ended
                    while True:
                        frame = await track.recv()
                        if np.abs(frame.to_ndarray()).max() > 1000:
                            heard.set()

            asyncio.create_task(listen())  # noqa: RUF006 - ends with the connection

        await client.setLocalDescription(await client.createOffer())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            local = client.localDescription
            response = await http.post("/voice/offer", json={"sdp": local.sdp, "type": local.type})
            assert response.status_code == 200, response.text
            answer = response.json()
            await client.setRemoteDescription(RTCSessionDescription(answer["sdp"], answer["type"]))
            await asyncio.wait_for(heard.wait(), 15)  # the agent's voice reached the client
            page = await http.get("/voice")
            assert page.status_code == 200 and "RTCPeerConnection" in page.text
        kinds = [e["type"] for e in events]
        assert "user" in kinds and "token" in kinds
        assert next(e for e in events if e["type"] == "user")["text"] == "hi there"
        assert "Hello, how can I help you today?" in tts.spoken
        await client.close()


async def test_offer_without_an_agent_is_refused(test_db_settings, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    app = create_app(test_db_settings.model_copy(update={"gemini_api_key": None}))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            response = await http.post("/voice/offer", json={"sdp": "v=0", "type": "offer"})
    assert response.status_code == 503
