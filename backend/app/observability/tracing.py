"""OpenTelemetry tracing: one trace per conversation, one span per turn.

    conversation                       (emitted when the call or chat ends)
    └─ turn                            (one per customer turn; voice: from end of speech)
       ├─ vad.end_of_turn              last voiced frame → end of turn decided   (voice)
       ├─ stt.final                    end of turn → final transcript            (voice)
       ├─ llm.call                     one model request (a Langfuse "generation")
       │  └─ llm.first_token           request sent → first streamed token
       ├─ tool.<name>                  each tool call, including prefetched and executed ones
       └─ tts.first_audio              first sentence sent → its first audio      (voice)

A conversation outlives any one request (a WebSocket can reconnect, a process can
restart), so its trace and root span ids are derived from the conversation id instead of
being held in memory: every turn finds the same trace. The root span itself is written
when the conversation ends, with those same ids (`_ForcedIds`). A conversation that never
ends cleanly still groups its turns in one trace; only the root span is missing.

Export is OTLP over HTTP, to Langfuse (LANGFUSE_* settings) or any collector
(OTEL_EXPORTER_OTLP_* settings). With neither set, spans are made but not exported.

Only conversation traces are kept. FastAPI starts a span for every request once an SDK is
installed (a health check would be a trace of its own); `_ConversationsOnly` drops every
root span except "conversation", and children follow their parent, so turns (whose parent
is the conversation's) are kept and request spans are not.
"""

import base64
import hashlib
import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.sdk.trace.sampling import Decision, ParentBased, Sampler, SamplingResult
from opentelemetry.trace import NonRecordingSpan, Span, SpanContext, TraceFlags

log = logging.getLogger(__name__)

tracer = trace.get_tracer("resolveai")

_forced_ids: ContextVar[tuple[int, int] | None] = ContextVar("forced_ids", default=None)
_turn_span: ContextVar[Span | None] = ContextVar("turn_span", default=None)
_provider: TracerProvider | None = None
_salt = "resolveai"  # set from the database name: conversation ids restart per database
capture_content = False  # LLM input/output on spans (customer PII): off unless configured


class _ForcedIds(RandomIdGenerator):
    """Random ids, except while `_forced_ids` is set: then those (for the root span)."""

    def generate_trace_id(self) -> int:
        forced = _forced_ids.get()
        return forced[0] if forced else super().generate_trace_id()

    def generate_span_id(self) -> int:
        forced = _forced_ids.get()
        return forced[1] if forced else super().generate_span_id()


class _ConversationsOnly(Sampler):
    """For root spans: keep "conversation", drop the rest (request spans)."""

    def should_sample(
        self, parent_context, trace_id, name, kind=None, attributes=None, *args, **kwargs
    ) -> SamplingResult:
        if name != "conversation":
            return SamplingResult(Decision.DROP)
        return SamplingResult(Decision.RECORD_AND_SAMPLE, attributes)  # keeps the attributes

    def get_description(self) -> str:
        return "ConversationsOnly"


def export_targets(settings) -> list[tuple[str, str, dict[str, str]]]:
    """(name, OTLP traces endpoint, headers) for each configured backend."""
    found = []
    secret = settings.langfuse_secret_key
    if settings.langfuse_public_key and secret is not None and secret.get_secret_value():
        auth = f"{settings.langfuse_public_key}:{secret.get_secret_value()}"
        endpoint = settings.langfuse_host.rstrip("/") + "/api/public/otel/v1/traces"
        headers = {"Authorization": "Basic " + base64.b64encode(auth.encode()).decode()}
        found.append(("langfuse", endpoint, headers))
    if settings.otel_exporter_otlp_endpoint:
        headers = dict(
            pair.split("=", 1)
            for pair in (settings.otel_exporter_otlp_headers or "").split(",")
            if "=" in pair
        )
        endpoint = settings.otel_exporter_otlp_endpoint.rstrip("/")
        if not endpoint.endswith("/v1/traces"):
            endpoint += "/v1/traces"
        found.append(("otlp", endpoint, headers))
    return found


def setup_tracing(settings, *, processors: tuple[SpanProcessor, ...] = ()) -> TracerProvider:
    """Install the tracer provider once per process; later calls only add `processors`
    (tests add an in-memory exporter)."""
    global _provider, _salt, capture_content
    _salt = settings.postgres_db
    capture_content = settings.otel_capture_content
    if _provider is None:
        _provider = TracerProvider(
            resource=Resource.create({"service.name": settings.otel_service_name}),
            id_generator=_ForcedIds(),
            sampler=ParentBased(root=_ConversationsOnly()),
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        for name, endpoint, headers in export_targets(settings):
            exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
            _provider.add_span_processor(BatchSpanProcessor(exporter))
            log.info("exporting traces", extra={"exporter": name, "endpoint": endpoint})
        trace.set_tracer_provider(_provider)
    for processor in processors:
        _provider.add_span_processor(processor)
    return _provider


def shutdown_tracing() -> None:
    """Flush spans still batched (on app shutdown and at the end of a CLI run)."""
    if _provider is not None:
        _provider.force_flush(timeout_millis=5000)


# --- conversation identity ----------------------------------------------------------------


def conversation_ids(conversation_id: int, salt: str | None = None) -> tuple[int, int]:
    """(trace id, root span id) for a conversation, the same in every process."""
    digest = hashlib.sha256(f"{salt or _salt}:conversation:{conversation_id}".encode()).digest()
    trace_id = int.from_bytes(digest[:16], "big") or 1
    span_id = int.from_bytes(digest[16:24], "big") or 1
    return trace_id, span_id


def trace_id_hex(conversation_id: int, salt: str | None = None) -> str:
    return format(conversation_ids(conversation_id, salt)[0], "032x")


def _conversation_context(conversation_id: int) -> otel_context.Context:
    trace_id, span_id = conversation_ids(conversation_id)
    parent = SpanContext(trace_id, span_id, is_remote=True, trace_flags=TraceFlags(1))
    return trace.set_span_in_context(NonRecordingSpan(parent), otel_context.Context())


def _session_attributes(conversation_id: int) -> dict[str, Any]:
    return {
        "resolveai.conversation_id": conversation_id,
        "langfuse.session.id": str(conversation_id),
    }


def end_conversation(
    conversation_id: int, started_at: datetime, ended_at: datetime, **attributes: Any
) -> None:
    """Write the conversation's root span, with the ids its turns already point to."""
    token = _forced_ids.set(conversation_ids(conversation_id))
    try:
        span = tracer.start_span(
            "conversation",
            context=otel_context.Context(),  # no parent: a root span
            start_time=_datetime_ns(started_at),
            attributes={
                **_session_attributes(conversation_id),
                "langfuse.trace.name": "conversation",
                **{f"resolveai.{k}": v for k, v in attributes.items() if v is not None},
            },
        )
    finally:
        _forced_ids.reset(token)
    span.end(end_time=_datetime_ns(ended_at))


# --- turns --------------------------------------------------------------------------------


def start_turn(conversation_id: int, *, start: float | None = None, **attributes: Any) -> Span:
    """A turn span under the conversation. `start` is a time.monotonic() value (voice turns
    start when the customer stopped speaking, before the agent is called)."""
    return tracer.start_span(
        "turn",
        context=_conversation_context(conversation_id),
        start_time=wall_ns(start) if start is not None else None,
        attributes={**_session_attributes(conversation_id), **_clean(attributes)},
    )


@contextmanager
def activate(span: Span) -> Iterator[Span]:
    """Make `span` the current span and the current turn, for this block and the tasks it
    creates. The span is not ended on exit."""
    token = _turn_span.set(span)
    try:
        with trace.use_span(span, end_on_exit=False):
            yield span
    finally:
        _turn_span.reset(token)


@contextmanager
def turn(conversation_id: int | None, **attributes: Any) -> Iterator[Span]:
    """The current turn span if a caller (the voice session) already opened one, else a
    new one, ended on exit."""
    current = _turn_span.get()
    if current is not None and current.is_recording():
        current.set_attributes(_clean(attributes))
        yield current
        return
    if conversation_id is None:
        yield trace.INVALID_SPAN
        return
    span = start_turn(conversation_id, **attributes)
    try:
        with activate(span):
            yield span
    finally:
        span.end()


def record_span(name: str, start: float, end: float, **attributes: Any) -> None:
    """A finished span from two time.monotonic() values, under the current span."""
    span = tracer.start_span(name, start_time=wall_ns(start), attributes=_clean(attributes))
    span.end(end_time=wall_ns(end))


# --- LLM generations ----------------------------------------------------------------------


def generation_attributes(
    *,
    model: str,
    usage: dict[str, Any],
    cost_usd: float | None,
    prompt_version: str,
    prompt_fingerprint: str,
    first_token_ms: float | None,
    attempt: int,
    finish_reason: str | None,
) -> dict[str, Any]:
    """OpenTelemetry GenAI conventions plus Langfuse's own keys, which take precedence in
    Langfuse (usage, cost and metadata as it expects them)."""
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    cached = cached_tokens(usage)
    attributes: dict[str, Any] = {
        "gen_ai.system": "gemini",
        "gen_ai.request.model": model,
        "gen_ai.response.model": model,
        "gen_ai.usage.input_tokens": prompt,
        "gen_ai.usage.output_tokens": completion,
        "gen_ai.response.finish_reasons": [finish_reason] if finish_reason else [],
        "langfuse.observation.type": "generation",
        "langfuse.observation.model.name": model,
        "langfuse.observation.usage_details": json.dumps(
            {
                "input": prompt,
                "output": completion,
                "total": prompt + completion,
                **({"input_cached_tokens": cached} if cached else {}),
            }
        ),
        "langfuse.observation.metadata.prompt_version": prompt_version,
        "langfuse.observation.metadata.prompt_fingerprint": prompt_fingerprint,
        "langfuse.observation.metadata.attempt": attempt,
        "resolveai.prompt.version": prompt_version,
        "resolveai.prompt.fingerprint": prompt_fingerprint,
        "resolveai.llm.cached_tokens": cached,
    }
    if first_token_ms is not None:
        attributes["resolveai.llm.first_token_ms"] = first_token_ms
    if cost_usd is not None:
        attributes["gen_ai.usage.cost"] = cost_usd
        attributes["langfuse.observation.cost_details"] = json.dumps({"total": cost_usd})
    return attributes


def cached_tokens(usage: dict[str, Any]) -> int:
    details = usage.get("prompt_tokens_details") or {}
    return int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0


# --- time ---------------------------------------------------------------------------------


def wall_ns(monotonic: float) -> int:
    """A time.monotonic() value as wall-clock nanoseconds (what span timestamps use)."""
    return time.time_ns() - int((time.monotonic() - monotonic) * 1e9)


def _datetime_ns(value: datetime) -> int:
    return int(value.timestamp() * 1e9)


def _clean(attributes: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in attributes.items() if v is not None}
