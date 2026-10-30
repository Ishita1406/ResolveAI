"""PCM helpers. Internally everything is 16-bit signed little-endian mono ("pcm16"):
16 kHz for VAD and STT, the TTS rate (24 kHz) for speech out, 48 kHz on the WebRTC side."""

import wave
from pathlib import Path

import av
import numpy as np

VAD_RATE = 16_000


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def silence(ms: float, rate: int) -> bytes:
    return b"\x00\x00" * int(rate * ms / 1000)


def duration_ms(pcm: bytes, rate: int) -> float:
    return len(pcm) / 2 / rate * 1000


class Resampler:
    """Mono pcm16 from one rate to another, streaming (keeps filter state between calls)."""

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        self.src_rate, self.dst_rate = src_rate, dst_rate
        self._resampler = (
            None
            if src_rate == dst_rate
            else av.AudioResampler(format="s16", layout="mono", rate=dst_rate)
        )

    def __call__(self, pcm: bytes) -> bytes:
        if self._resampler is None or not pcm:
            return pcm
        frame = av.AudioFrame.from_ndarray(
            np.frombuffer(pcm, dtype="<i2").reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = self.src_rate
        return _join(self._resampler.resample(frame))


def frame_to_mono_pcm16(frame: av.AudioFrame) -> bytes:
    """An incoming WebRTC frame (any layout and format) as mono pcm16 at its own rate."""
    resampler = av.AudioResampler(format="s16", layout="mono", rate=frame.sample_rate)
    return _join(resampler.resample(frame))


def _join(frames: list[av.AudioFrame]) -> bytes:
    # A plane can be padded past the samples it holds; keep exactly samples * 2 bytes.
    return b"".join(bytes(f.planes[0])[: f.samples * 2] for f in frames)


def read_wav(path: Path, rate: int = VAD_RATE) -> bytes:
    """A WAV file as mono pcm16 at `rate` (any channel count or rate in the file)."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        channels, src_rate = w.getnchannels(), w.getframerate()
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1).astype("<i2")
    return Resampler(src_rate, rate)(data.tobytes())


def write_wav(path: Path, pcm: bytes, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


class Chunker:
    """Re-chunks a pcm16 stream into fixed-size frames."""

    def __init__(self, frame_bytes: int) -> None:
        self.frame_bytes = frame_bytes
        self._buffer = bytearray()

    def feed(self, pcm: bytes) -> list[bytes]:
        self._buffer += pcm
        frames = []
        while len(self._buffer) >= self.frame_bytes:
            frames.append(bytes(self._buffer[: self.frame_bytes]))
            del self._buffer[: self.frame_bytes]
        return frames
