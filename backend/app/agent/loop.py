"""The agent loop: one customer message in, a streamed reply out.

Per turn:
  1. advance the session's turn counter (expires stale proposals) and reconcile writes
  2. if a proposal is awaiting confirmation, classify the customer's reply deterministically:
     yes -> confirm + execute; no -> cancel; anything else -> the model re-asks. A clear yes
     or no is answered from the outcome without the model (when the prompt set has the texts)
  2b. otherwise (with prefetch on) run the lookups the message plainly asks for, an email or
     phone plus PIN, and one order number, as if the model had called them; a message that is
     only verification details is answered without the model (prefetch.py)
  3. call the model with the tools; run read tools, turn write-tool calls into proposals
     (the model can only propose); repeat until the model answers without tool calls
  4. on timeout, iteration limit or model failure: hand off to a human with a summary

Each turn runs in its own task feeding a queue. The caller iterates events and may stop at
any time (barge-in); the turn task is then cancelled and cleans up: a confirmed write that
already started is allowed to finish, history keeps only what the customer actually heard,
and the session is saved.

Observability: each turn is a span in its conversation's trace (app/observability), with a
span per model call and per tool call, and its usage, cost and timings go to turn_metrics.
"""

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractAsyncContextManager, aclosing, contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from opentelemetry.trace import Span, Status, StatusCode
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import prefetch, transcript
from app.agent.confirmation import Reply, classify_reply
from app.agent.events import (
    ActionEvent,
    AgentEvent,
    DoneEvent,
    ErrorEvent,
    EscalationEvent,
    TokenEvent,
    ToolEvent,
)
from app.agent.grounding import CustomerTagGuard, ungrounded_identity
from app.agent.llm import (
    Completion,
    LLMClient,
    LLMError,
    TextDelta,
    ToolCall,
    tool_call_message,
)
from app.agent.prompts import PromptSet, render_system_prompt
from app.agent.session import AgentSession, SessionBusyError, SessionStore
from app.domain.enums import ConversationChannel, EscalationReason, TurnRole
from app.domain.formatting import format_inr
from app.observability import logs, metrics, tracing
from app.policy import gate
from app.policy.config import PolicyConfig
from app.policy.state import ActionState
from app.retrieval.search import PolicyRetriever
from app.tools.base import READ_TOOLS, WRITE_TOOLS
from app.tools.context import ToolContext
from app.tools.registry import call_tool
from app.tools.schema import llm_tool_definitions

log = logging.getLogger(__name__)

CUSTOMER_TAG = "customer_message"
UNVERIFIED_TOOLS = frozenset({"verify_customer", "escalate_to_human", "search_policies"})
# Replies are single spoken paragraphs. Longer runs of whitespace than this mean the model is
# degenerating (it has been seen padding a reply with spaces until the token cap).
MAX_BLANK_RUN = 8
INCOMPLETE = frozenset({"length", "degenerate"})
_WHITESPACE = re.compile(r"\s+")
_END = object()

DbFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]


@dataclass(frozen=True)
class AgentConfig:
    max_iterations: int = 6
    turn_timeout_s: float = 30.0
    history_turns: int = 10
    escalation_timeout_s: float = 10.0
    settle_timeout_s: float = 15.0
    llm_retries: int = 2  # per model call, only for retryable errors (429, 5xx, timeouts)
    min_answer_time_s: float = 5.0  # don't start a retry wait that leaves less than this
    history_tool_result_chars: int = 400  # tool results in older turns are cut to this
    prefetch: bool = False  # deterministic verify/get_order before the model (prefetch.py)
    # USD per million tokens by model, {"input": .., "output": ..}, for turn costs.
    llm_prices: dict[str, dict[str, float]] = field(default_factory=dict)
    # Model routing: this model for calls made before the customer is verified. Those only
    # ask for details, search policies or hand over; no account or money tools exist yet.
    fast_model: str | None = None


class _IterationLimitError(Exception):
    pass


@dataclass
class _Turn:
    """Everything one turn produces, kept apart from the session until it ends."""

    user_message: dict[str, Any]
    notices: list[str] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)  # model/tool exchange
    spoken: list[str] = field(default_factory=list)
    # JSON lines for the transcript: every tool call with its arguments and result.
    tool_log: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0}
    )
    protected: list[asyncio.Task] = field(default_factory=list)
    iterations: int = 0
    completed: bool = False
    interrupted: bool = False
    deadline: float = 0.0  # event-loop time by which the turn must finish
    held: bool = False  # "One moment, please." already said this turn
    done: DoneEvent | None = None  # sent only after the turn is saved
    redone: bool = False  # a cut-off reply was already asked for again this turn
    # Observability (time.monotonic() values and running totals for turn_metrics).
    turn_id: str = ""
    started: float = field(default_factory=time.monotonic)
    first_text_at: float | None = None
    llm_first_token_ms: float | None = None
    llm_ms: float = 0.0
    tool_ms: float = 0.0
    tool_calls: int = 0
    cached_tokens: int = 0
    cost_usd: float | None = None
    model: str | None = None


def wrap_customer_text(text: str) -> str:
    """Delimit customer text as data. Tags inside it are neutralized so it can't close the
    block and pose as system text."""
    cleaned = text.replace(f"<{CUSTOMER_TAG}>", "").replace(f"</{CUSTOMER_TAG}>", "")
    return f"<{CUSTOMER_TAG}>{cleaned}</{CUSTOMER_TAG}>"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Agent:
    def __init__(
        self,
        *,
        llm: LLMClient,
        store: SessionStore,
        db_factory: DbFactory,
        prompts: PromptSet,
        policy: PolicyConfig,
        config: AgentConfig | None = None,
        clock: Callable[[], datetime] = _utcnow,
        retriever: PolicyRetriever | None = None,
        policy_documents: str | None = None,
    ) -> None:
        """Policy text reaches the model through `retriever` (search_policies) or, in full-text
        mode, `policy_documents` in the system prompt; with neither, it gets no policy text."""
        self.llm = llm
        self.store = store
        self.db_factory = db_factory
        self.prompts = prompts
        self.config = config or AgentConfig()
        self.clock = clock
        self.retriever = retriever
        self.system_prompt = render_system_prompt(
            prompts, policy, retrieval=retriever is not None, policy_documents=policy_documents
        )
        # Before verification only verification, policy search and handoff are offered: the
        # rest would be refused anyway, and their schemas are most of each request's tokens.
        all_tools = [
            t
            for t in llm_tool_definitions()
            if retriever is not None or t["function"]["name"] != "search_policies"
        ]
        self.tools_unverified = [t for t in all_tools if t["function"]["name"] in UNVERIFIED_TOOLS]
        self.tools_verified = [t for t in all_tools if t["function"]["name"] != "verify_customer"]
        self._background: set[asyncio.Task] = set()

    # --- public ---------------------------------------------------------------------------

    async def start_session(self, channel: ConversationChannel = ConversationChannel.TEXT):
        now = self.clock()
        async with self.db_factory() as db:
            conversation_id = await transcript.start_conversation(db, channel=channel, now=now)
        session = AgentSession(
            conversation_id=conversation_id, channel=channel, created_at=now, updated_at=now
        )
        await self.store.save(session)
        return session

    async def respond(
        self, session_id: str, text: str, *, turn_id: str | None = None
    ) -> AsyncIterator[AgentEvent]:
        """Stream the reply to one customer message. Stop iterating to cancel the turn.

        `turn_id` names the turn in turn_metrics; the voice session passes its own so it can
        add its timings to the same row. A new one is made when not given."""
        queue: asyncio.Queue = asyncio.Queue()
        turn_id = turn_id or uuid.uuid4().hex
        task = asyncio.create_task(self._run_turn(session_id, text, turn_id, queue.put_nowait))
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        finished = False
        try:
            while (event := await queue.get()) is not _END:
                finished = finished or event.type == "done"
                yield event
        finally:
            # Barge-in cancels a reply in progress. After "done" there is nothing left to
            # interrupt, so a consumer that stops early must not cancel the turn's cleanup.
            if not task.done() and not finished:
                task.cancel()
            # The turn finishes its own cleanup even if this consumer is cancelled again.
            await asyncio.wait({task})

    # --- turn -----------------------------------------------------------------------------

    async def end_session(self, session_id: str) -> None:
        """The call or chat is over: record when, and write the conversation's root span."""
        session = await self.store.load(session_id)
        if session is None or session.conversation_id is None:
            return
        now = self.clock()
        async with self.db_factory() as db:
            first_end = await transcript.end_conversation(db, session.conversation_id, now=now)
        if not first_end:
            return  # the root span was written when it first ended
        tracing.end_conversation(
            session.conversation_id,
            session.created_at,
            now,
            channel=str(session.channel),
            verified=session.state.is_verified,
            escalation_reference=session.escalation_reference,
        )

    async def _run_turn(self, session_id: str, text: str, turn_id: str, emit: Callable) -> None:
        logs.bind(session_id=session_id, turn_id=turn_id)
        try:
            async with self.store.lock(session_id):
                session = await self.store.load(session_id)
                if session is None:
                    emit(ErrorEvent(code="session_not_found", message="Session not found."))
                    return
                logs.bind(conversation_id=session.conversation_id)
                with tracing.turn(
                    session.conversation_id, channel=str(session.channel), turn_id=turn_id
                ) as span:
                    await self._run_loaded(session, text, turn_id, emit, span)
        except SessionBusyError:
            emit(ErrorEvent(code="session_busy", message="Still working on the last message."))
        finally:
            emit(_END)

    async def _run_loaded(
        self, session: AgentSession, text: str, turn_id: str, emit: Callable, span: Span
    ) -> None:
        turn = _Turn(
            user_message={"role": "user", "content": wrap_customer_text(text)},
            deadline=asyncio.get_running_loop().time() + self.config.turn_timeout_s,
            turn_id=turn_id,
        )
        async with self.db_factory() as db:
            ctx = ToolContext(
                db=db,
                state=session.state,
                conversation_id=session.conversation_id,
                clock=self.clock,
                retriever=self.retriever,
            )
            try:
                try:
                    async with asyncio.timeout(self.config.turn_timeout_s):
                        await self._turn(session, ctx, turn, text, emit)
                except TimeoutError:
                    await self._fail(
                        session,
                        ctx,
                        turn,
                        emit,
                        EscalationReason.TOOL_FAILURE,
                        "The turn timed out.",
                    )
                except _IterationLimitError:
                    await self._fail(
                        session,
                        ctx,
                        turn,
                        emit,
                        EscalationReason.LOW_CONFIDENCE,
                        "The assistant could not reach an answer.",
                    )
                except LLMError as exc:
                    log.warning("LLM failure: %s", exc)
                    await self._fail(
                        session,
                        ctx,
                        turn,
                        emit,
                        EscalationReason.TOOL_FAILURE,
                        "The language model was unavailable.",
                    )
                except Exception:
                    log.exception("agent turn failed")
                    await self._fail(
                        session,
                        ctx,
                        turn,
                        emit,
                        EscalationReason.TOOL_FAILURE,
                        "An internal error occurred.",
                    )
            except asyncio.CancelledError:
                turn.interrupted = True
                raise
            finally:
                await self._settle(turn)
                self._commit_history(session, turn)
                session.updated_at = self.clock()
                turn_index = await self._record(db, session, turn, text)
                await self._record_metrics(session, turn, span, turn_index)
                await self.store.save(session)
                if turn.done is not None:
                    # "done" means the turn is saved: the client may send the next message.
                    emit(turn.done.model_copy(update={"turn_id": turn.turn_id}))

    async def _turn(
        self, session: AgentSession, ctx: ToolContext, turn: _Turn, text: str, emit: Callable
    ) -> None:
        for expired in gate.register_customer_turn(ctx):
            turn.notices.append(self.prompts.notice("expired", summary=expired.summary))
        await gate.reconcile(ctx)
        pending = self._awaiting(ctx)
        reply = await self._handle_confirmation(ctx, turn, text, emit)
        if reply:
            self._answer_directly(turn, emit, reply)
            return
        if self.config.prefetch and pending is None:
            if await self._prefetch(session, ctx, turn, text, emit):
                return

        while True:
            turn.iterations += 1
            if turn.iterations > self.config.max_iterations:
                raise _IterationLimitError
            completion, wrote_for_customer = await self._call_model(session, ctx, turn, emit)
            if wrote_for_customer:
                log.warning("model wrote a customer turn; discarded it and its tool calls")
                if not completion.text.strip():
                    turn.notices.append(self.prompts.notice("wrote_for_customer"))
                    continue
            assistant: dict[str, Any] = {"role": "assistant", "content": completion.text or ""}
            if completion.tool_calls:
                assistant["tool_calls"] = [tool_call_message(c) for c in completion.tool_calls]
            turn.messages.append(assistant)
            if (
                not completion.tool_calls
                and completion.finish_reason in INCOMPLETE
                and not turn.redone
            ):
                # Cut off mid-sentence: ask once for the whole answer rather than end on it.
                log.warning("model reply incomplete (%s); asking again", completion.finish_reason)
                turn.redone = True
                turn.messages.append(
                    {"role": "system", "content": self.prompts.notice("reply_cut_off")}
                )
                continue
            if not completion.tool_calls:
                turn.completed = True
                turn.done = DoneEvent(
                    reply=self._spoken(turn),
                    outcome="completed",
                    iterations=turn.iterations,
                    usage=turn.usage,
                )
                return
            for call in completion.tool_calls:
                content = await self._run_tool(session, ctx, turn, call, emit)
                turn.messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

    async def _handle_confirmation(
        self, ctx: ToolContext, turn: _Turn, text: str, emit: Callable
    ) -> str | None:
        """Apply a clear yes or no to the awaiting proposal. Returns the reply to speak when
        it can be said without the model (the outcome is all there is to tell)."""
        pending = self._awaiting(ctx)
        if pending is None:
            return None
        match classify_reply(text):
            case Reply.YES:
                confirmed = gate.confirm(ctx, pending.id)
                if not confirmed.ok:
                    return None
                with self._tool_span(turn, pending.tool, source="confirmed") as span:
                    result = await self._protected(turn, gate.execute(ctx, pending.id))
                    span.set_attribute("resolveai.tool.ok", result.ok)
                emit(self._action_event(ctx, pending.id))
                self._log_tool(
                    turn,
                    pending.tool,
                    ctx.state.actions[pending.id].arguments,
                    ok=result.ok,
                    payload={"error": _error(result.error)} if result.error else {},
                    phase="executed",
                )
                if result.ok and result.tool_result is not None:
                    turn.notices.append(
                        self.prompts.notice(
                            "executed",
                            summary=pending.summary,
                            result=json.dumps(result.tool_result.data, ensure_ascii=False),
                        )
                    )
                    return self._done_reply(pending.tool, result.tool_result.data)
                message = result.error.message if result.error else "unknown error"
                turn.notices.append(
                    self.prompts.notice(
                        "execution_failed", summary=pending.summary, message=message
                    )
                )
                return self._fixed_reply("not_done", summary=pending.summary, message=message)
            case Reply.NO:
                gate.cancel(ctx, pending.id)
                emit(self._action_event(ctx, pending.id))
                self._log_tool(
                    turn,
                    pending.tool,
                    ctx.state.actions[pending.id].arguments,
                    ok=True,
                    payload={},
                    phase="cancelled",
                )
                turn.notices.append(
                    self.prompts.notice("cancelled_by_customer", summary=pending.summary)
                )
                return self._fixed_reply("not_doing", summary=pending.summary)
            case Reply.AMBIGUOUS:
                pass  # the model sees the pending proposal in the turn context and re-asks
        return None

    def _fixed_reply(self, key: str, **values: object) -> str | None:
        """A reply spoken without the model, or None if this prompt version has no such text
        (then the model answers, as before)."""
        if key not in self.prompts.notices:
            return None
        return self.prompts.notice(key, **values)

    def _done_reply(self, tool: str, data: dict[str, Any]) -> str | None:
        """What a confirmed action did, from its result: references, amounts, deadlines."""
        main = self._fixed_reply(f"done_{tool}", **data)
        if main is None:
            return None
        parts = [main]
        if tool == "create_return":
            if data.get("return_by"):
                parts.append(self.prompts.notice("done_return_by", return_by=data["return_by"]))
            if data.get("refund_requires_human"):
                parts.append(
                    self.prompts.notice("done_refund_review", amount=data["refund_amount_display"])
                )
        if tool != "escalate_to_human":
            parts.append(self.prompts.notice("anything_else"))
        return " ".join(parts)

    def _answer_directly(self, turn: _Turn, emit: Callable, reply: str) -> None:
        """End the turn with a fixed reply: no model call."""
        for i, word in enumerate(reply.split(" ")):
            self._say(turn, emit, word if i == 0 else " " + word)
        turn.messages.append({"role": "assistant", "content": reply})
        turn.completed = True
        turn.done = DoneEvent(
            reply=self._spoken(turn), outcome="completed", iterations=0, usage=turn.usage
        )

    async def _prefetch(
        self, session: AgentSession, ctx: ToolContext, turn: _Turn, text: str, emit: Callable
    ) -> bool:
        """Run the tool calls this message plainly needs. True if the turn is answered."""
        found = prefetch.extract(text)
        state = ctx.state
        verified_now = False
        if (
            not state.is_verified
            and not state.is_locked
            and (found.email or found.phone or found.pin)
        ):
            # One detail may have come in an earlier message ("my email is ..." then the PIN).
            earlier = [prefetch.extract(t) for t in reversed(self._customer_texts(session, turn))]
            pin = next((f.pin for f in earlier if f.pin), None)
            who = next((f for f in earlier if f.email or f.phone), None)
            if pin and who:
                identity = {"email": who.email} if who.email else {"phone": who.phone}
                await self._prefetched(
                    session, ctx, turn, "verify_customer", {**identity, "postal_code": pin}, emit
                )
                verified_now = state.is_verified
        if (
            state.is_verified
            and found.order_number
            and found.order_number != session.current_order_number
        ):
            await self._prefetched(
                session, ctx, turn, "get_order", {"order_number": found.order_number}, emit
            )
            return False
        if verified_now and prefetch.only_identification(text, found, state.customer_name):
            first_name = (state.customer_name or "").split()[0]
            reply = self._fixed_reply("verified_ready", first_name=first_name)
            if reply:
                self._answer_directly(turn, emit, reply)
                return True
        return False

    async def _prefetched(
        self,
        session: AgentSession,
        ctx: ToolContext,
        turn: _Turn,
        name: str,
        args: dict[str, Any],
        emit: Callable,
    ) -> None:
        """A tool call the loop makes on the model's behalf, recorded as if the model had made
        it, so the model reads the result exactly as it would its own."""
        call = ToolCall(
            id=f"prefetch_{len(turn.messages)}",
            name=name,
            arguments=json.dumps(args),
            # Gemini checks thought signatures on past calls; this one has none to send.
            extra_content={"google": {"thought_signature": "skip_thought_signature_validator"}},
        )
        turn.messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call_message(call)],
            }
        )
        content = await self._run_tool(session, ctx, turn, call, emit, source="prefetch")
        turn.messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

    async def _call_model(
        self, session: AgentSession, ctx: ToolContext, turn: _Turn, emit: Callable
    ) -> tuple[Completion, bool]:
        """One model call, retried on rate limits and transient errors while time allows.

        Returns the completion and whether the model started writing a customer turn (then
        the text is cut before it and the tool calls are dropped).
        """
        messages = self._build_messages(session, ctx, turn)
        tools = self.tools_verified if ctx.state.is_verified else self.tools_unverified
        model = self.config.fast_model if not ctx.state.is_verified else None
        attempt = 0
        while True:
            spoke = [False]
            try:
                with tracing.tracer.start_as_current_span(
                    "llm.call", record_exception=False, set_status_on_exception=False
                ) as span:
                    started = time.monotonic()
                    try:
                        completion, wrote_for_customer = await self._stream_once(
                            messages, tools, turn, emit, spoke, model
                        )
                    except LLMError as exc:
                        span.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
                        span.set_attribute("resolveai.llm.http_status", exc.status or 0)
                        raise
                    self._note_llm(turn, span, completion, messages, started, attempt, model)
                return completion, wrote_for_customer
            except LLMError as exc:
                wait = self._retry_wait(exc, attempt, turn, spoke[0])
                if wait is None:
                    raise
                attempt += 1
                log.warning("model call failed (%s); retrying in %.1fs", exc.status, wait)
                if not turn.held:
                    turn.held = True
                    self._say(turn, emit, self.prompts.notice("please_hold") + " ")
                await asyncio.sleep(wait)

    def _retry_wait(self, exc: LLMError, attempt: int, turn: _Turn, spoke: bool) -> float | None:
        if not exc.retryable or spoke or attempt >= self.config.llm_retries:
            return None
        wait = exc.retry_after_s if exc.retry_after_s is not None else float(2**attempt)
        remaining = turn.deadline - asyncio.get_running_loop().time()
        if wait + self.config.min_answer_time_s > remaining:
            return None
        return wait

    async def _stream_once(
        self, messages, tools, turn: _Turn, emit: Callable, spoke: list[bool], model=None
    ) -> tuple[Completion, bool]:
        guard = CustomerTagGuard()
        said: list[str] = []
        completion: Completion | None = None

        def speak(text: str) -> None:
            last = turn.spoken[-1][-1:] if turn.spoken else ""
            text = _WHITESPACE.sub(" ", text)  # also folds U+202F etc. into plain spaces
            if not last or last.isspace():
                text = text.lstrip(" ")
            if not text:
                return
            if (
                last
                and not last.isspace()
                and (
                    not said  # a new model call: keep its words apart from the previous one
                    or (last in ".?!" and text[:1].isupper())  # "address?And" -> "address? And"
                )
            ):
                self._say(turn, emit, " ")
            said.append(text)
            spoke[0] = True
            self._say(turn, emit, text)

        blank_run = 0
        routed = {"model": model} if model else {}  # only routing clients take `model`
        stop = [f"<{CUSTOMER_TAG}>"]
        async with aclosing(self.llm.stream(messages, tools, stop=stop, **routed)) as stream:
            async for event in stream:
                if isinstance(event, TextDelta):
                    for ch in event.text:
                        blank_run = blank_run + 1 if ch.isspace() else 0
                        if blank_run > MAX_BLANK_RUN:
                            # Stop generating; aclosing closes the stream.
                            speak(guard.flush())
                            done = Completion(text="".join(said), finish_reason="degenerate")
                            return done, False
                    speak(guard.feed(event.text))
                    if guard.tripped:
                        break
                else:
                    completion = event
        if guard.tripped:
            return Completion(text="".join(said)), True
        speak(guard.flush())
        if completion is None:
            raise LLMError("model stream ended without a completion", retryable=True)
        return replace(completion, text="".join(said)), False

    async def _run_tool(
        self,
        session: AgentSession,
        ctx: ToolContext,
        turn: _Turn,
        call: ToolCall,
        emit: Callable,
        *,
        source: str | None = None,
    ) -> str:
        args = call.parsed_arguments()
        emit(ToolEvent(name=call.name, status="started"))
        with self._tool_span(turn, call.name, source=source or "model") as span:
            ok, payload = await self._tool_result(session, ctx, turn, call, args, emit)
            span.set_attribute("resolveai.tool.ok", ok)
            if not ok:
                error = payload.get("error") or {}
                span.set_attribute("resolveai.tool.error", str(error.get("code", "")))
        self._log_tool(turn, call.name, args, ok=ok, payload=payload, phase="call", source=source)
        emit(ToolEvent(name=call.name, status="finished", ok=ok))
        return json.dumps(payload, ensure_ascii=False, default=str)

    async def _tool_result(
        self,
        session: AgentSession,
        ctx: ToolContext,
        turn: _Turn,
        call: ToolCall,
        args: dict[str, Any],
        emit: Callable,
    ) -> tuple[bool, dict[str, Any]]:
        missing = (
            ungrounded_identity(args, self._customer_texts(session, turn))
            if call.name == "verify_customer"
            else None
        )
        if missing:
            # Never verify with details the model made up; it would also burn an attempt.
            ok = False
            payload: dict[str, Any] = {
                "error": {
                    "code": "not_said_by_customer",
                    "message": f"I still need your {missing}.",
                    "hint": f"The customer hasn't said this {missing}. Ask for it; never guess.",
                }
            }
        elif call.name in WRITE_TOOLS:
            # The model can only propose. Confirmation comes from the customer's next reply.
            proposed = await gate.propose(ctx, call.name, args)
            ok = proposed.ok
            if proposed.action is not None:
                emit(self._action_event(ctx, proposed.action.action_id))
            if ok and proposed.action is not None:
                payload = {
                    "status": "awaiting_confirmation",
                    "read_back": proposed.action.summary,
                    "amount": proposed.action.amount_display,
                    "instruction": self.prompts.notice(
                        "proposal_created", read_back=proposed.action.summary
                    ),
                }
            else:
                payload = {"status": "refused", "error": _error(proposed.error)}
        elif call.name in READ_TOOLS:
            result = await call_tool(ctx, call.name, args)
            ok = result.ok
            payload = {"ok": True, "data": result.data} if ok else {"error": _error(result.error)}
            if ok and isinstance(result.data, dict) and result.data.get("order_number"):
                session.current_order_number = result.data["order_number"]
        else:
            ok = False
            payload = {
                "error": {
                    "code": "unknown_tool",
                    "message": "Sorry, I can't do that.",
                    "hint": f"There is no tool named {call.name!r}.",
                }
            }
        return ok, payload

    async def _fail(
        self,
        session: AgentSession,
        ctx: ToolContext,
        turn: _Turn,
        emit: Callable,
        reason: EscalationReason,
        what_happened: str,
    ) -> None:
        """Hand off to a human and tell the customer, without relying on the model."""
        await ctx.db.rollback()
        customer_said = turn.user_message["content"]
        issue = f"{what_happened} Last customer message: {customer_said}"
        reference = None
        try:
            async with asyncio.timeout(self.config.escalation_timeout_s):
                result = await self._protected(
                    turn,
                    gate.escalate_now(
                        ctx,
                        reason=reason,
                        issue=issue[:2000],
                        order_number=session.current_order_number,
                        note=f"system: {what_happened}",
                    ),
                )
            if result.ok and result.tool_result is not None:
                reference = result.tool_result.data["ticket_reference"]
        except Exception:
            log.exception("system escalation failed")
        self._log_tool(
            turn,
            "escalate_to_human",
            {"reason": reason, "what_happened": what_happened, "reference": reference},
            ok=reference is not None,
            payload={},
            phase="system",
        )
        if reference:
            session.escalation_reference = reference
            emit(EscalationEvent(reference=reference, reason=reason))
            text = self.prompts.notice("fallback_escalated", reference=reference)
        else:
            text = self.prompts.notice("fallback_failed")
        if turn.spoken:
            self._say(turn, emit, " ")
        for word in text.split(" "):
            self._say(turn, emit, word + " ")
        turn.done = DoneEvent(
            reply=self._spoken(turn),
            outcome="escalated" if reference else "failed",
            iterations=turn.iterations,
            usage=turn.usage,
        )

    # --- helpers --------------------------------------------------------------------------

    @staticmethod
    def _log_tool(
        turn: _Turn,
        tool: str,
        arguments: dict[str, Any],
        *,
        ok: bool,
        payload: dict[str, Any],
        phase: str,
        source: str | None = None,
    ) -> None:
        """phase: "call" (the model called it; a write tool only proposes), "executed" or
        "cancelled" (the customer's yes or no to a proposal), or "system" (a handoff the
        loop made itself). source "prefetch": a call the loop made for the model."""
        error = payload.get("error") or {}
        entry = {
            "tool": tool,
            "phase": phase,
            "arguments": {k: v for k, v in arguments.items() if k != "idempotency_key"},
            "ok": ok,
            "status": payload.get("status"),
            "error": error.get("code") if isinstance(error, dict) else None,
        }
        if source:
            entry["source"] = source
        turn.tool_log.append(json.dumps(entry, ensure_ascii=False, default=str))

    def _say(self, turn: _Turn, emit: Callable, text: str) -> None:
        if turn.first_text_at is None and text.strip():
            turn.first_text_at = time.monotonic()
        turn.spoken.append(text)
        emit(TokenEvent(text=text))

    @staticmethod
    def _spoken(turn: _Turn) -> str:
        return "".join(turn.spoken).strip()

    @staticmethod
    def _customer_texts(session: AgentSession, turn: _Turn) -> list[str]:
        messages = [m for past in session.history for m in past] + [turn.user_message]
        return [
            m["content"].removeprefix(f"<{CUSTOMER_TAG}>").removesuffix(f"</{CUSTOMER_TAG}>")
            for m in messages
            if m.get("role") == "user"
        ]

    def _compact(self, message: dict[str, Any]) -> dict[str, Any]:
        """Older tool results only need to remind the model, not repeat every field."""
        limit = self.config.history_tool_result_chars
        content = message.get("content")
        if message.get("role") == "tool" and isinstance(content, str) and len(content) > limit:
            return {**message, "content": content[:limit] + "… (truncated)"}
        return message

    @staticmethod
    def _awaiting(ctx: ToolContext):
        waiting = [
            a for a in ctx.state.actions.values() if a.state is ActionState.AWAITING_CONFIRMATION
        ]
        return waiting[-1] if waiting else None

    @staticmethod
    def _action_event(ctx: ToolContext, action_id: str) -> ActionEvent:
        action = ctx.state.actions[action_id]
        return ActionEvent(
            action_id=action.id,
            tool=action.tool,
            state=action.state,
            summary=action.summary,
            amount_display=format_inr(action.amount_paise) if action.amount_paise else None,
        )

    async def _protected(self, turn: _Turn, coro):
        """Run a write so that cancelling the turn can't interrupt it halfway."""
        task = asyncio.ensure_future(coro)
        turn.protected.append(task)
        return await asyncio.shield(task)

    async def _settle(self, turn: _Turn) -> None:
        pending = [t for t in turn.protected if not t.done()]
        if pending:
            await asyncio.wait(pending, timeout=self.config.settle_timeout_s)

    def _build_messages(
        self, session: AgentSession, ctx: ToolContext, turn: _Turn
    ) -> list[dict[str, Any]]:
        state = ctx.state
        pending = self._awaiting(ctx)
        context = self.prompts.turn_context.render(
            verification=(
                self.prompts.notice("verified", first_name=(state.customer_name or "").split()[0])
                if state.is_verified
                else self.prompts.notice("not_verified")
            ),
            pending_action=(
                self.prompts.notice(
                    "pending_action",
                    summary=pending.summary,
                    amount=format_inr(pending.amount_paise) if pending.amount_paise else "none",
                )
                if pending
                else self.prompts.notice("no_pending_action")
            ),
        )
        recent = session.history[-self.config.history_turns :]
        history = [
            m if i == len(recent) - 1 else self._compact(m)
            for i, past in enumerate(recent)
            for m in past
        ]
        return [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "system", "content": context},
            turn.user_message,
            *({"role": "system", "content": n} for n in turn.notices),
            *turn.messages,
        ]

    def _commit_history(self, session: AgentSession, turn: _Turn) -> None:
        """Keep a valid message sequence. A completed turn keeps its tool exchange; any other
        turn keeps only what the customer said, the notices and what they actually heard."""
        notices = [{"role": "system", "content": n} for n in turn.notices]
        if turn.completed:
            entry = [turn.user_message, *notices, *turn.messages]
        else:
            heard = self._spoken(turn)
            if heard and turn.interrupted:
                heard = f"{heard} [interrupted by the customer]"
            entry = [turn.user_message, *notices]
            if heard:
                entry.append({"role": "assistant", "content": heard})
        session.history.append(entry)
        session.history = session.history[-self.config.history_turns :]

    # --- observability --------------------------------------------------------------------

    @contextmanager
    def _tool_span(self, turn: _Turn, name: str, *, source: str) -> Iterator[Span]:
        started = time.monotonic()
        attributes = {"resolveai.tool.name": name, "resolveai.tool.source": source}
        with tracing.tracer.start_as_current_span(f"tool.{name}", attributes=attributes) as span:
            try:
                yield span
            finally:
                turn.tool_calls += 1
                turn.tool_ms += (time.monotonic() - started) * 1000

    def _note_llm(
        self,
        turn: _Turn,
        span: Span,
        completion: Completion,
        messages: list[dict[str, Any]],
        started: float,
        attempt: int,
        routed_model: str | None = None,
    ) -> None:
        """Add one model call to the turn's totals and describe it on its span."""
        elapsed_ms = (time.monotonic() - started) * 1000
        first_ms = (
            round(completion.first_token_s * 1000, 1)
            if completion.first_token_s is not None
            else None
        )
        if first_ms is not None:
            tracing.record_span("llm.first_token", started, started + first_ms / 1000)
        model = routed_model or getattr(self.llm, "model", None) or "unknown"
        usage = completion.usage
        cost = metrics.cost_usd(model, usage, self.config.llm_prices)
        turn.usage["llm_calls"] += 1
        for key in ("prompt_tokens", "completion_tokens"):
            turn.usage[key] += int(usage.get(key) or 0)
        turn.cached_tokens += tracing.cached_tokens(usage)
        turn.llm_ms += elapsed_ms
        turn.model = model
        if turn.llm_first_token_ms is None:
            turn.llm_first_token_ms = first_ms
        if cost is not None:
            turn.cost_usd = (turn.cost_usd or 0.0) + cost
        span.set_attributes(
            tracing.generation_attributes(
                model=model,
                usage=usage,
                cost_usd=cost,
                prompt_version=self.prompts.version,
                prompt_fingerprint=self.prompts.fingerprint,
                first_token_ms=first_ms,
                attempt=attempt,
                finish_reason=completion.finish_reason,
            )
        )
        if tracing.capture_content:
            span.set_attribute("langfuse.observation.input", json.dumps(messages, default=str))
            calls = [{"name": c.name, "arguments": c.arguments} for c in completion.tool_calls]
            output = {"text": completion.text, "tool_calls": calls}
            span.set_attribute("langfuse.observation.output", json.dumps(output))

    async def _record_metrics(
        self, session: AgentSession, turn: _Turn, span: Span, turn_index: int | None
    ) -> None:
        if turn.done is not None:
            outcome = turn.done.outcome
        else:
            outcome = "interrupted" if turn.interrupted else "failed"
        values = {
            "outcome": outcome,
            "llm_calls": turn.usage["llm_calls"],
            "prompt_tokens": turn.usage["prompt_tokens"],
            "completion_tokens": turn.usage["completion_tokens"],
            "cached_tokens": turn.cached_tokens,
            "cost_usd": round(turn.cost_usd, 8) if turn.cost_usd is not None else None,
            "llm_first_token_ms": turn.llm_first_token_ms,
            "llm_ms": round(turn.llm_ms, 1),
            "tool_calls": turn.tool_calls,
            "tool_ms": round(turn.tool_ms, 1),
            "agent_ms": round((time.monotonic() - turn.started) * 1000, 1),
        }
        span.set_attributes(
            {f"resolveai.turn.{k}": v for k, v in values.items() if v is not None}
            | {"resolveai.prompt.version": self.prompts.version}
        )
        log.info("turn finished", extra={k: v for k, v in values.items() if v is not None})
        if session.conversation_id is None:
            return
        await metrics.save_quietly(
            self.db_factory,
            turn.turn_id,
            conversation_id=session.conversation_id,
            started_at=self.clock(),
            turn_index=turn_index,
            channel=str(session.channel),
            model=turn.model,
            prompt_version=self.prompts.fingerprint,
            trace_id=tracing.trace_id_hex(session.conversation_id),
            **values,
        )

    async def _record(
        self, db: AsyncSession, session: AgentSession, turn: _Turn, text: str
    ) -> int | None:
        """Write the turn to the transcript; returns the customer line's turn index."""
        if session.conversation_id is None:
            return None
        reply = self._spoken(turn) + (" [interrupted]" if turn.interrupted else "")
        entries = [(TurnRole.CUSTOMER, text)]
        entries += [(TurnRole.TOOL, line) for line in turn.tool_log]
        entries.append((TurnRole.ASSISTANT, reply.strip()))
        await db.rollback()
        return await transcript.record_turns(
            db, session.conversation_id, entries, state=session.state, now=self.clock()
        )


def _error(error) -> dict[str, Any]:
    if error is None:
        return {"code": "unknown", "message": "Sorry, something went wrong."}
    return {
        "code": error.code,
        "message": error.message,
        "hint": error.hint,
        "details": error.details,
    }
