r"""One voice call: audio in, the agent in the middle, audio out.

    transport --pcm16 16 kHz--> feed() --> VAD (barge-in, end of speech)
                                      \--> STT stream (partials, segments)
    end of turn (VAD silence + semantic check) --> agent.respond(text)
        tokens --> sentences --> TTS context --> OutputBuffer --> transport reads 20 ms frames

Turn-taking, barge-in, fillers and failure handling are all here; the agent is called
exactly as the text channel calls it. Barge-in stops playback at once (the buffer is
emptied and anything still arriving from that reply is dropped) and cancels the agent's
turn, which is the Step 5 cancellation: the in-flight LLM stream is closed and only what
was already said is kept in history.

Failures: STT or TTS trouble is said once, the provider is reconnected once, and a second
failure hands the call to a person (escalation.py). LLM failures are the agent's own
(Step 5: "One moment, please", one retry, then a handoff), so they're spoken like any reply.

Timing: each turn is a span in the conversation's trace, starting at the customer's last
voiced frame, with child spans for the end-of-turn wait, the final transcript and TTS first
audio (the agent adds its own). Time to first audio and its stages go to turn_metrics.
"""

import asyncio
import contextlib
import logging
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.agent.loop import Agent
from app.config import Settings
from app.domain.enums import ConversationChannel
from app.observability import logs, tracing
from app.observability import metrics as metrics_store
from app.voice.audio import Chunker
from app.voice.escalation import escalate_voice_failure
from app.voice.speech import SentenceChunker, for_speech, normalize_transcript
from app.voice.stt import SttError, SttEvent, SttProvider, SttStream
from app.voice.tts import TtsConnection, TtsError, TtsProvider, Utterance, synthesize
from app.voice.turn import silence_needed_ms
from app.voice.vad import FRAME_BYTES, SpeechScorer, VadTracker

log = logging.getLogger(__name__)

FILLERS = ("One moment.", "Let me check that.", "Just a second.")
STT_TROUBLE = "Sorry, I'm having trouble hearing you. One moment."
TTS_TROUBLE = "Sorry, my audio cut out for a moment."
DIDNT_CATCH = "Sorry, I didn't catch that. Could you say it again?"
HANDOFF = (
    "I'm sorry, I'm having technical trouble on this call. I've passed it to a colleague "
    "who will follow up, and your reference is {reference}."
)
HANDOFF_FAILED = "I'm sorry, I'm having technical trouble. Please try again in a few minutes."


class _VoiceFailure(Exception):
    """A provider failed twice: hand the call over."""


class OutputBuffer:
    """Speech waiting to be played. The transport pulls fixed-size frames in real time.

    `clear()` is barge-in: it drops everything queued and bumps the epoch, so audio still
    arriving from the interrupted reply (tagged with the old epoch) is discarded."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.epoch = 0
        self._chunks: deque[bytes] = deque()
        self._bytes = 0
        self.played = bytearray()  # everything played, for replay recordings

    def put(self, pcm: bytes, epoch: int) -> None:
        if epoch == self.epoch and pcm:
            self._chunks.append(pcm)
            self._bytes += len(pcm)

    def clear(self) -> None:
        self.epoch += 1
        self._chunks.clear()
        self._bytes = 0

    @property
    def playing(self) -> bool:
        return self._bytes > 0

    def read(self, nbytes: int) -> bytes:
        """Exactly `nbytes` of audio, padded with silence when there's nothing to play."""
        out = bytearray()
        while self._chunks and len(out) < nbytes:
            chunk = self._chunks.popleft()
            take = nbytes - len(out)
            out += chunk[:take]
            if len(chunk) > take:
                self._chunks.appendleft(chunk[take:])
        self._bytes -= len(out)
        self.played += out
        return bytes(out) + b"\x00" * (nbytes - len(out))


@dataclass
class TurnMetrics:
    """One turn's timeline (time.monotonic() values) and time to first audio.

    end_of_speech → turn_end: the end-of-turn silence wait (VAD); → text_ready: the final
    transcript (STT); → first_token: the agent's first reply text; → tts_sent: the first text
    sent to TTS (a whole sentence, or the whole reply with tts_streaming "reply"); →
    first_audio: its first audio. A filler can be heard before the reply."""

    text: str
    end_of_speech: float  # the customer's last voiced frame
    turn_end: float = 0.0  # end of turn decided
    text_ready: float = 0.0  # final transcript handed to the agent
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    first_token: float | None = None
    first_sentence: float | None = None
    tts_sent: float | None = None  # the first sentence sent to TTS
    first_audio: float | None = None
    first_audio_ms: float | None = None  # end of speech → first agent audio of any kind
    first_audio_source: str | None = None  # "reply" or "filler"
    filler: bool = False
    interrupted: bool = False

    def stages(self) -> dict[str, float | None]:
        """Milliseconds per stage. They add up to first_audio_ms when the reply's own speech
        was heard first."""

        def ms(start: float | None, end: float | None) -> float | None:
            return round((end - start) * 1000, 1) if start and end else None

        return {
            "vad_ms": ms(self.end_of_speech, self.turn_end),
            "stt_ms": ms(self.turn_end, self.text_ready),
            "agent_first_text_ms": ms(self.text_ready, self.first_token),
            "first_sentence_ms": ms(self.first_token, self.tts_sent),
            "tts_first_audio_ms": ms(self.tts_sent, self.first_audio),
        }

    def summary(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "text": self.text,
            "first_audio_ms": self.first_audio_ms,
            "first_audio_source": self.first_audio_source,
            **self.stages(),
            "filler": self.filler,
            "interrupted": self.interrupted,
        }

    def mark_first_audio(self, source: str) -> bool:
        """True the first time any agent audio for this turn is queued."""
        if self.first_audio_ms is not None:
            return False
        self.first_audio = time.monotonic()
        self.first_audio_ms = round((self.first_audio - self.end_of_speech) * 1000, 1)
        self.first_audio_source = source
        return True


@dataclass
class _Transcript:
    """The customer's words in the current turn, from STT segments and partials."""

    segments: list[str] = field(default_factory=list)
    partial: str = ""
    raw_open: str = ""  # cumulative text of the open STT segment
    consumed: str = ""  # its prefix already handed to the agent
    segment_ended: asyncio.Event = field(default_factory=asyncio.Event)

    def text(self) -> str:
        return " ".join([*self.segments, self.partial]).strip()

    def _fresh(self, cumulative: str) -> str:
        if self.consumed and cumulative.startswith(self.consumed):
            return cumulative[len(self.consumed) :].strip()
        return cumulative.strip()

    def on_event(self, event: SttEvent) -> None:
        if event.kind == "partial":
            self.raw_open = event.text
            self.partial = self._fresh(event.text)
        elif event.kind == "end":
            fresh = self._fresh(event.text)
            if fresh:
                self.segments.append(fresh)
            self.partial = self.raw_open = self.consumed = ""
            self.segment_ended.set()

    def take(self) -> str:
        """Hand the turn's words over and start a new turn."""
        text = self.text()
        self.consumed = self.raw_open  # later updates of this open segment repeat it
        self.segments, self.partial = [], ""
        self.segment_ended.clear()
        return text


class VoiceSession:
    def __init__(
        self,
        agent: Agent,
        settings: Settings,
        *,
        stt: SttProvider,
        tts: TtsProvider,
        vad: SpeechScorer,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.agent = agent
        self.settings = settings
        self.stt_provider = stt
        self.tts_provider = tts
        self.vad = VadTracker(
            vad, threshold=settings.vad_threshold, min_speech_ms=settings.vad_min_speech_ms
        )
        self.on_event = on_event or (lambda event: None)
        self.session_id: str | None = None
        self.conversation_id: int | None = None
        self.out = OutputBuffer(settings.tts_sample_rate)
        self.metrics: list[TurnMetrics] = []
        self.ended = asyncio.Event()  # handed over or closed: no more turns
        self.escalation_reference: str | None = None

        self._in: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._stt: SttStream | None = None
        self._tts: TtsConnection | None = None
        self._phrases: dict[str, bytes] = {}
        self._transcript = _Transcript()
        self._turn_open = False
        self._turn_opened_at = 0.0
        self._closing_turn = False
        self._turn_task: asyncio.Task | None = None
        self._tasks: list[asyncio.Task] = []
        self._stt_failures = 0
        self._tts_failures = 0
        self._empty_turns = 0
        self._filler_index = 0
        self._stt_lock = asyncio.Lock()
        self._end_task: asyncio.Task | None = None
        self._writes: set[asyncio.Task] = set()  # turn metrics being saved

    # --- lifecycle ------------------------------------------------------------------------

    async def start(self) -> str:
        session = await self.agent.start_session(channel=ConversationChannel.VOICE)
        self.session_id = session.session_id
        self.conversation_id = session.conversation_id
        # Before any task is created, so every task of this call logs with these ids.
        logs.bind(conversation_id=self.conversation_id, session_id=self.session_id)
        try:
            self._tts = await self._connect_tts()
            self._stt = await self._connect_stt()
        except _VoiceFailure as failure:
            await self._hand_over(str(failure))
            return self.session_id
        self._tasks = [
            asyncio.create_task(self._input_loop()),
            asyncio.create_task(self._stt_loop()),
            asyncio.create_task(self._cache_phrases()),
        ]
        warm = getattr(self.agent.llm, "warm", None)
        if self.settings.llm_prewarm and warm is not None:
            # The customer is still speaking: open the LLM connection now, not in the turn.
            self._tasks.append(asyncio.create_task(warm()))
        self._emit({"type": "listening", "session_id": self.session_id})
        return self.session_id

    def feed(self, pcm16k: bytes) -> None:
        """Customer audio: mono pcm16 at 16 kHz, any chunk size."""
        self._in.put_nowait(pcm16k)

    async def close(self) -> None:
        self.ended.set()
        self._in.put_nowait(None)
        for task in [*self._tasks, self._turn_task, self._end_task]:
            if task and not task.done():
                task.cancel()
        for task in [*self._tasks, self._turn_task, self._end_task]:
            if task:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        for closer in (self._stt, self._tts):
            if closer is not None:
                with contextlib.suppress(Exception):
                    await closer.close()
        if self._writes:
            await asyncio.wait(self._writes, timeout=5)
        if self.session_id is not None:
            with contextlib.suppress(Exception):
                await self.agent.end_session(self.session_id)

    @property
    def busy(self) -> bool:
        """The agent is working on a reply or speech is still playing."""
        return bool(self._turn_task and not self._turn_task.done()) or self.out.playing

    @property
    def idle(self) -> bool:
        """Nothing in progress: no open turn, no reply being prepared, nothing playing."""
        return not (self._turn_open or self._closing_turn or self.busy)

    def _emit(self, event: dict[str, Any]) -> None:
        with contextlib.suppress(Exception):
            self.on_event(event)

    # --- providers ------------------------------------------------------------------------

    async def _connect_stt(self) -> SttStream:
        try:
            return await self.stt_provider.connect()
        except SttError as first:
            log.warning("STT connect failed: %s; retrying once", first)
            self._emit({"type": "error", "code": "stt_unavailable", "retrying": True})
            try:
                return await self.stt_provider.connect()
            except SttError as second:
                raise _VoiceFailure(f"speech recognition unavailable: {second}") from second

    async def _connect_tts(self) -> TtsConnection:
        try:
            return await self.tts_provider.connect()
        except TtsError as first:
            log.warning("TTS connect failed: %s; retrying once", first)
            try:
                return await self.tts_provider.connect()
            except TtsError as second:
                raise _VoiceFailure(f"speech synthesis unavailable: {second}") from second

    async def _cache_phrases(self) -> None:
        """Fixed lines synthesized up front: fillers play instantly, and the "audio cut out"
        apology can still be said when TTS has just failed."""
        for text in (*FILLERS, TTS_TROUBLE, STT_TROUBLE, DIDNT_CATCH):
            try:
                self._phrases[text] = await synthesize(self._tts, text)
            except TtsError:
                return

    async def _play_phrase(self, text: str) -> None:
        """A fixed line: from the cache if there, else synthesized now (if TTS works)."""
        pcm = self._phrases.get(text)
        if pcm is None:
            with contextlib.suppress(TtsError, AttributeError):
                pcm = await synthesize(self._tts, text)
        if pcm:
            self.out.put(pcm, self.out.epoch)
        self._emit({"type": "say", "text": text})

    # --- input: VAD and turn-taking -------------------------------------------------------

    async def _input_loop(self) -> None:
        chunker = Chunker(FRAME_BYTES)
        while (pcm := await self._in.get()) is not None:
            if self.ended.is_set():
                continue
            await self._send_to_stt(pcm)
            for frame in chunker.feed(pcm):
                update = self.vad.update(frame)
                if update.speech_started:
                    await self._on_speech_start()
                if self._turn_open and not self._closing_turn:
                    self._maybe_end_turn(update.silence_ms)

    async def _send_to_stt(self, pcm: bytes) -> None:
        if self._stt is None:
            return
        try:
            await self._stt.send(pcm)
        except SttError:
            pass  # the STT loop sees the same failure and recovers

    async def _on_speech_start(self) -> None:
        if self.busy:
            await self._interrupt()
        self._turn_open = True
        self._turn_opened_at = time.monotonic()
        self._emit({"type": "speech_start"})

    async def _interrupt(self) -> None:
        """Barge-in: silence now, then cancel the reply in progress."""
        self.out.clear()
        if self.metrics:
            self.metrics[-1].interrupted = True
        task, self._turn_task = self._turn_task, None
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._emit({"type": "barge_in"})

    def _maybe_end_turn(self, silence_ms: float) -> None:
        so_far = self._transcript.text()
        needed = silence_needed_ms(
            so_far,
            complete_ms=self.settings.end_of_turn_silence_ms,
            incomplete_ms=self.settings.incomplete_turn_silence_ms,
        )
        now = time.monotonic()
        too_long = now - self._turn_opened_at > self.settings.max_turn_s
        if silence_ms >= needed or too_long:
            self._closing_turn = True
            speech_end = now - silence_ms / 1000
            self._end_task = asyncio.create_task(self._end_turn(speech_end, now))

    async def _end_turn(self, speech_end: float, turn_end: float) -> None:
        try:
            self._turn_open = False
            self.vad.end_turn()
            transcript = self._transcript
            if transcript.partial and not transcript.segment_ended.is_set():
                # The last words may still be on their way from STT.
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        transcript.segment_ended.wait(), self.settings.stt_final_wait_ms / 1000
                    )
            text = normalize_transcript(transcript.take())
            if not text:
                await self._nothing_heard()
                return
            self._empty_turns = 0
            self._emit({"type": "user", "text": text})
            metrics = TurnMetrics(
                text=text, end_of_speech=speech_end, turn_end=turn_end, text_ready=time.monotonic()
            )
            self.metrics.append(metrics)
            span = None
            if self.conversation_id is not None:
                span = tracing.start_turn(
                    self.conversation_id, start=speech_end, channel="voice", turn_id=metrics.turn_id
                )
                with tracing.activate(span):
                    tracing.record_span("vad.end_of_turn", speech_end, turn_end)
                    tracing.record_span("stt.final", turn_end, metrics.text_ready)
            with tracing.activate(span) if span else contextlib.nullcontext():
                self._turn_task = asyncio.create_task(self._respond(text, metrics, span))
        finally:
            self._closing_turn = False

    async def _nothing_heard(self) -> None:
        """Speech with no transcript: ask once; twice in a row means STT is failing."""
        self._empty_turns += 1
        if self._empty_turns == 1:
            await self._play_phrase(DIDNT_CATCH)
            return
        self._empty_turns = 0
        await self._stt_trouble(SttError("speech detected but nothing transcribed, twice"))

    # --- STT ------------------------------------------------------------------------------

    async def _stt_loop(self) -> None:
        while not self.ended.is_set():
            stream = self._stt
            if stream is None:
                await asyncio.sleep(0.05)
                continue
            try:
                async for event in stream.events():
                    if stream is not self._stt:
                        break  # replaced by a reconnect
                    self._transcript.on_event(event)
                    if event.kind == "partial":
                        self._emit({"type": "partial", "text": self._transcript.text()})
                if stream is self._stt and not self.ended.is_set():
                    raise SttError("STT stream ended")
            except SttError as exc:
                if stream is self._stt and not await self._stt_trouble(exc):
                    return

    async def _stt_trouble(self, exc: Exception) -> bool:
        """Say so once and reconnect once; the second failure hands the call over.
        True if listening again."""
        async with self._stt_lock:  # the STT loop and an empty turn can both get here
            return await self._recover_stt(exc)

    async def _recover_stt(self, exc: Exception) -> bool:
        log.warning("STT failure: %s", exc)
        self._stt_failures += 1
        if self._stt_failures > 1:
            await self._hand_over(f"speech recognition failed twice: {exc}")
            return False
        await self._play_phrase(STT_TROUBLE)
        old, self._stt = self._stt, None
        if old is not None:
            with contextlib.suppress(Exception):
                await old.close()
        self._transcript = _Transcript()
        try:
            self._stt = await self.stt_provider.connect()
        except SttError as again:
            await self._hand_over(f"speech recognition failed twice: {again}")
            return False
        self._emit({"type": "recovered", "what": "stt"})
        return True

    # --- the agent's reply ----------------------------------------------------------------

    async def _respond(self, text: str, metrics: TurnMetrics, span=None) -> None:
        speaker = _Speaker(self, metrics)
        chunker = SentenceChunker()
        filler: asyncio.Task | None = None
        try:
            async for event in self.agent.respond(self.session_id, text, turn_id=metrics.turn_id):
                self._emit(event.model_dump(mode="json"))
                if event.type == "token":
                    if metrics.first_token is None and event.text.strip():
                        metrics.first_token = time.monotonic()
                    if filler and not filler.done():
                        filler.cancel()
                    for sentence in chunker.feed(event.text):
                        await speaker.say(sentence)
                elif event.type == "tool" and event.status == "started":
                    if filler is None or filler.done():
                        filler = asyncio.create_task(self._filler_after_delay(metrics))
                elif event.type == "tool" and filler and not filler.done():
                    filler.cancel()  # the tool was quick: no filler needed
                elif event.type == "escalated":
                    self.escalation_reference = event.reference
            for sentence in chunker.flush():
                await speaker.say(sentence)
            await speaker.finish()
        except _VoiceFailure as failure:
            await self._hand_over(str(failure))
        finally:
            if filler and not filler.done():
                filler.cancel()
            await speaker.close()
            self._finish_turn(metrics, span)

    def _finish_turn(self, metrics: TurnMetrics, span) -> None:
        """End the turn's span and save its voice timings (in the background, so a barge-in
        isn't held up by the database)."""
        stages = metrics.stages()
        log.info(
            "voice turn timings",
            extra={
                "time_to_first_audio_ms": metrics.first_audio_ms,
                "first_audio_source": metrics.first_audio_source,
                "interrupted": metrics.interrupted,
                **stages,
            },
        )
        if span is not None:
            span.set_attributes(
                {
                    f"resolveai.voice.{k}": v
                    for k, v in {**stages, "time_to_first_audio_ms": metrics.first_audio_ms}.items()
                    if v is not None
                }
                | {
                    "resolveai.voice.filler": metrics.filler,
                    "resolveai.voice.interrupted": metrics.interrupted,
                    "resolveai.voice.first_audio_source": metrics.first_audio_source or "none",
                }
            )
            span.end()
        if self.conversation_id is None:
            return
        task = asyncio.create_task(
            metrics_store.save_quietly(
                self.agent.db_factory,
                metrics.turn_id,
                conversation_id=self.conversation_id,
                channel="voice",
                time_to_first_audio_ms=metrics.first_audio_ms,
                filler=metrics.filler,
                interrupted=metrics.interrupted,
                **stages,
            )
        )
        self._writes.add(task)
        task.add_done_callback(self._writes.discard)

    async def _filler_after_delay(self, metrics: TurnMetrics) -> None:
        await asyncio.sleep(self.settings.filler_after_ms / 1000)
        if self.out.playing:
            return  # something is already being said
        text = FILLERS[self._filler_index % len(FILLERS)]
        self._filler_index += 1
        metrics.filler = True
        await self._play_phrase(text)
        if metrics.mark_first_audio("filler"):
            self._emit({"type": "first_audio", "ms": metrics.first_audio_ms, "source": "filler"})
        self._emit({"type": "filler", "text": text})

    # --- handing over ---------------------------------------------------------------------

    async def _hand_over(self, what_happened: str) -> None:
        if self.ended.is_set():
            return
        self.ended.set()
        task, self._turn_task = self._turn_task, None
        if task and not task.done() and task is not asyncio.current_task():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        reference = await escalate_voice_failure(self.agent, self.session_id, what_happened)
        self.escalation_reference = reference
        text = HANDOFF.format(reference=reference) if reference else HANDOFF_FAILED
        pcm = None
        with contextlib.suppress(TtsError, AttributeError):
            if self._tts is not None:
                pcm = await synthesize(self._tts, for_speech(text))
        if pcm:
            self.out.put(pcm, self.out.epoch)
        self._emit(
            {
                "type": "handed_over",
                "reference": reference,
                "reason": what_happened,
                "text": text,
                "spoken": bool(pcm),
            }
        )


class _Speaker:
    """Speaks one reply, a sentence at a time, into the output buffer.

    Each sentence gets its own TTS context, synthesized one after another. Cartesia ends a
    context about a second after its last audio, and a reply often pauses longer than that
    (a tool call, "One moment, please" while the model is retried): a context kept open
    across the pause would take the rest of the reply and say nothing. Synthesis runs faster
    than playback, so the next sentence is ready before the buffer runs dry.

    On a TTS failure: the apology is said once (from the phrase cache, so it works while
    TTS is down), TTS is reconnected once, and the reply so far is spoken again on the new
    connection. A second failure in the call hands it over."""

    def __init__(self, session: VoiceSession, metrics: TurnMetrics) -> None:
        self.s = session
        self.metrics = metrics
        # tts_streaming "reply": hold every sentence until the reply is complete (baseline).
        self.hold = session.settings.tts_streaming == "reply"
        self.epoch = session.out.epoch
        self.sentences: list[str] = []
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._current: Utterance | None = None
        self._pump: asyncio.Task | None = None
        self._failed: TtsError | None = None

    def _start(self) -> None:
        self._queue = asyncio.Queue()
        self._failed = None
        self._pump = asyncio.create_task(self._pump_audio())

    async def _pump_audio(self) -> None:
        try:
            while (sentence := await self._queue.get()) is not None:
                self._current = utterance = self.s._tts.utterance()
                first = self.metrics.tts_sent is None
                if first:
                    self.metrics.tts_sent = time.monotonic()
                await utterance.say(for_speech(sentence))
                await utterance.finish()
                async for chunk in utterance.audio():
                    if first:
                        first = False
                        tracing.record_span(
                            "tts.first_audio", self.metrics.tts_sent, time.monotonic()
                        )
                    if self.metrics.mark_first_audio("reply"):
                        self.s._emit(
                            {
                                "type": "first_audio",
                                "ms": self.metrics.first_audio_ms,
                                "source": "reply",
                            }
                        )
                    self.s.out.put(chunk, self.epoch)
                self._current = None
        except TtsError as exc:
            self._failed = exc

    async def say(self, sentence: str) -> None:
        if self.metrics.first_sentence is None:
            self.metrics.first_sentence = time.monotonic()
        self.sentences.append(sentence)
        if self.hold:
            return
        if self._pump is None:
            self._start()
        if self._failed:
            await self._recover(self._failed)  # speaks this sentence too
        else:
            self._queue.put_nowait(sentence)

    async def finish(self) -> None:
        if self.hold and self._pump is None and self.sentences:
            self._start()
            self._queue.put_nowait(" ".join(self.sentences))
        if self._pump is None:
            return
        self._queue.put_nowait(None)
        await self._pump
        if self._failed:
            await self._recover(self._failed)
            await self.finish()

    async def _recover(self, exc: TtsError) -> None:
        log.warning("TTS failure: %s", exc)
        self.s._tts_failures += 1
        if self.s._tts_failures > 1:
            raise _VoiceFailure(f"speech synthesis failed twice: {exc}")
        await self._stop()
        self.s.out.clear()  # don't finish half a sentence from the broken stream
        self.epoch = self.s.out.epoch
        await self.s._play_phrase(TTS_TROUBLE)
        old, self.s._tts = self.s._tts, None
        with contextlib.suppress(Exception):
            await old.close()
        try:
            self.s._tts = await self.s.tts_provider.connect()
        except TtsError as again:
            raise _VoiceFailure(f"speech synthesis failed twice: {again}") from again
        self.s._emit({"type": "recovered", "what": "tts"})
        self._start()
        for sentence in self.sentences:  # the reply so far, again
            self._queue.put_nowait(sentence)

    async def _stop(self) -> None:
        if self._current is not None:
            with contextlib.suppress(Exception):
                await self._current.cancel()
            self._current = None
        if self._pump and not self._pump.done():
            self._pump.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._pump

    async def close(self) -> None:
        await self._stop()
