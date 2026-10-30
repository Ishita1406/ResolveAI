"""Voice activity detection with Silero VAD v5 (ONNX), run with onnxruntime (already a
dependency for embeddings), so no PyTorch.

`SileroVAD` scores 32 ms frames (512 samples at 16 kHz). `VadTracker` turns scores into
speech start/end with hysteresis: speech starts after `min_speech_ms` above the threshold
(so a cough or a click doesn't interrupt the agent) and silence is counted below a lower
threshold (so a dip mid-word doesn't end a turn).
"""

import hashlib
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from app.voice.audio import VAD_RATE, pcm16_to_float

FRAME_SAMPLES = 512  # Silero v5 at 16 kHz
FRAME_BYTES = FRAME_SAMPLES * 2
FRAME_MS = FRAME_SAMPLES / VAD_RATE * 1000  # 32 ms
_CONTEXT = 64  # v5 prepends the previous 64 samples to each window


class SpeechScorer(Protocol):
    def score(self, frame: bytes) -> float: ...

    def reset(self) -> None: ...


def ensure_model(url: str, sha256: str, cache_dir: Path) -> Path:
    """The model file in the cache, downloaded on first use and verified by hash."""
    path = cache_dir / "silero_vad.onnx"
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == sha256:
        return path
    cache_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https URL
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise RuntimeError(f"Silero VAD download has sha256 {digest}, expected {sha256}")
    path.write_bytes(data)
    return path


class SileroVAD:
    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort  # heavy import; only when voice is used

        options = ort.SessionOptions()
        options.inter_op_num_threads = options.intra_op_num_threads = 1
        self._session = ort.InferenceSession(str(model_path), sess_options=options)
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT), dtype=np.float32)

    def score(self, frame: bytes) -> float:
        """Speech probability of one 512-sample frame."""
        x = pcm16_to_float(frame).reshape(1, -1)
        if x.shape[1] != FRAME_SAMPLES:
            raise ValueError(f"Silero expects {FRAME_SAMPLES} samples, got {x.shape[1]}")
        window = np.concatenate([self._context, x], axis=1)
        with self._lock:
            out, self._state = self._session.run(
                None, {"input": window, "state": self._state, "sr": np.array(VAD_RATE)}
            )
        self._context = x[:, -_CONTEXT:]
        return float(out[0][0])


@dataclass
class VadUpdate:
    speech_started: bool = False  # speech began on this frame (after min_speech_ms)
    in_speech: bool = False
    silence_ms: float = 0.0  # silence since the last speech, while a turn is open


class VadTracker:
    def __init__(
        self, scorer: SpeechScorer, *, threshold: float = 0.5, min_speech_ms: float = 160
    ) -> None:
        self.scorer = scorer
        self.start_threshold = threshold
        self.end_threshold = max(threshold - 0.15, 0.05)
        self.min_speech_ms = min_speech_ms
        self.in_speech = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0

    def reset(self) -> None:
        self.scorer.reset()
        self.in_speech = False
        self._speech_ms = self._silence_ms = 0.0

    def update(self, frame: bytes) -> VadUpdate:
        p = self.scorer.score(frame)
        started = False
        if p >= self.start_threshold:
            self._speech_ms += FRAME_MS
            self._silence_ms = 0.0
            if not self.in_speech and self._speech_ms >= self.min_speech_ms:
                self.in_speech = started = True
        elif p < self.end_threshold:
            self._silence_ms += FRAME_MS
            if not self.in_speech:
                self._speech_ms = 0.0
        return VadUpdate(started, self.in_speech, self._silence_ms if self.in_speech else 0.0)

    def end_turn(self) -> None:
        """The turn was handed to the agent: the next speech is a new turn."""
        self.in_speech = False
        self._speech_ms = self._silence_ms = 0.0
