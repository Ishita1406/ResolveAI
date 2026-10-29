"""The agent loop: one customer message in, a streamed reply out.

Per turn:
  1. advance the session's turn counter (expires stale proposals) and reconcile writes
  2. if a proposal is awaiting confirmation, classify the customer's reply deterministically:
     yes -> confirm + execute; no -> cancel; anything else -> the model re-asks
  3. call the model with the tools; run read tools, turn write-tool calls into proposals
     (the model can only propose); repeat until the model answers without tool calls
  4. on timeout, iteration limit or model failure: hand off to a human with a summary

Each turn runs in its own task feeding a queue. The caller iterates events and may stop at
any time (barge-in); the turn task is then cancelled and cleans up: a confirmed write that
already started is allowed to finish, history keeps only what the customer actually heard,
and the session is saved.
"""

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, aclosing
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import transcript
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
from app.agent.llm import Completion, LLMClient, LLMError, TextDelta, ToolCall
from app.agent.prompts import PromptSet, render_system_prompt
from app.agent.session import AgentSession, SessionBusyError, SessionStore
from app.domain.enums import ConversationChannel, EscalationReason, TurnRole
from app.domain.formatting import format_inr
from app.policy import gate
from app.policy.config import PolicyConfig
from app.policy.state import ActionState
from app.tools.base import READ_TOOLS, WRITE_TOOLS
from app.tools.context import ToolContext
from app.tools.registry import call_tool
from app.tools.schema import llm_tool_definitions

log = logging.getLogger(__name__)

CUSTOMER_TAG = "customer_message"
UNVERIFIED_TOOLS = frozenset({"verify_customer", "escalate_to_human"})
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


class _IterationLimitError(Exception):
    pass


@dataclass
class _Turn:
    """Everything one turn produces, kept apart from the session until it ends."""

    user_message: dict[str, Any]
    notices: list[str] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)  # model/tool exchange
    spoken: list[str] = field(default_factory=list)
    tool_log: list[str] = field(default_factory=list)
    protected: list[asyncio.Task] = field(default_factory=list)
    iterations: int = 0
    completed: bool = False
    interrupted: bool = False
    deadline: float = 0.0  # event-loop time by which the turn must finish
    held: bool = False  # "One moment, please." already said this turn
    redone: bool = False  # a cut-off reply was already asked for again this turn


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
    ) -> None:
        self.llm = llm
        self.store = store
        self.db_factory = db_factory
        self.prompts = prompts
        self.config = config or AgentConfig()
        self.clock = clock
        self.system_prompt = render_system_prompt(prompts, policy)
        # Before verification only verification and handoff are offered: the rest would be
        # refused anyway, and their schemas are most of each request's tokens.
        all_tools = llm_tool_definitions()
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

    async def respond(self, session_id: str, text: str) -> AsyncIterator[AgentEvent]:
        """Stream the reply to one customer message. Stop iterating to cancel the turn."""
        queue: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(self._run_turn(session_id, text, queue.put_nowait))
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        try:
            while (event := await queue.get()) is not _END:
                yield event
        finally:
            if not task.done():
                task.cancel()
            # The turn finishes its own cleanup even if this consumer is cancelled again.
            await asyncio.wait({task})

    # --- turn -----------------------------------------------------------------------------

    async def _run_turn(self, session_id: str, text: str, emit: Callable) -> None:
        try:
            async with self.store.lock(session_id):
                session = await self.store.load(session_id)
                if session is None:
                    emit(ErrorEvent(code="session_not_found", message="Session not found."))
                    return
                await self._run_loaded(session, text, emit)
        except SessionBusyError:
            emit(ErrorEvent(code="session_busy", message="Still working on the last message."))
        finally:
            emit(_END)

    async def _run_loaded(self, session: AgentSession, text: str, emit: Callable) -> None:
        turn = _Turn(
            user_message={"role": "user", "content": wrap_customer_text(text)},
            deadline=asyncio.get_running_loop().time() + self.config.turn_timeout_s,
        )
        async with self.db_factory() as db:
            ctx = ToolContext(
                db=db,
                state=session.state,
                conversation_id=session.conversation_id,
                clock=self.clock,
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
                await self._record(db, session, turn, text)
                await self.store.save(session)

    async def _turn(
        self, session: AgentSession, ctx: ToolContext, turn: _Turn, text: str, emit: Callable
    ) -> None:
        for expired in gate.register_customer_turn(ctx):
            turn.notices.append(self.prompts.notice("expired", summary=expired.summary))
        await gate.reconcile(ctx)
        await self._handle_confirmation(ctx, turn, text, emit)

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
                assistant["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments},
                    }
                    for c in completion.tool_calls
                ]
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
                emit(
                    DoneEvent(
                        reply=self._spoken(turn), outcome="completed", iterations=turn.iterations
                    )
                )
                return
            for call in completion.tool_calls:
                content = await self._run_tool(session, ctx, turn, call, emit)
                turn.messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

    async def _handle_confirmation(
        self, ctx: ToolContext, turn: _Turn, text: str, emit: Callable
    ) -> None:
        pending = self._awaiting(ctx)
        if pending is None:
            return
        match classify_reply(text):
            case Reply.YES:
                confirmed = gate.confirm(ctx, pending.id)
                if not confirmed.ok:
                    return
                result = await self._protected(turn, gate.execute(ctx, pending.id))
                emit(self._action_event(ctx, pending.id))
                if result.ok and result.tool_result is not None:
                    turn.notices.append(
                        self.prompts.notice(
                            "executed",
                            summary=pending.summary,
                            result=json.dumps(result.tool_result.data, ensure_ascii=False),
                        )
                    )
                else:
                    turn.notices.append(
                        self.prompts.notice(
                            "execution_failed",
                            summary=pending.summary,
                            message=result.error.message if result.error else "unknown error",
                        )
                    )
            case Reply.NO:
                gate.cancel(ctx, pending.id)
                emit(self._action_event(ctx, pending.id))
                turn.notices.append(
                    self.prompts.notice("cancelled_by_customer", summary=pending.summary)
                )
            case Reply.AMBIGUOUS:
                pass  # the model sees the pending proposal in the turn context and re-asks

    async def _call_model(
        self, session: AgentSession, ctx: ToolContext, turn: _Turn, emit: Callable
    ) -> tuple[Completion, bool]:
        """One model call, retried on rate limits and transient errors while time allows.

        Returns the completion and whether the model started writing a customer turn (then
        the text is cut before it and the tool calls are dropped).
        """
        messages = self._build_messages(session, ctx, turn)
        tools = self.tools_verified if ctx.state.is_verified else self.tools_unverified
        attempt = 0
        while True:
            spoke = [False]
            try:
                return await self._stream_once(messages, tools, turn, emit, spoke)
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
        self, messages, tools, turn: _Turn, emit: Callable, spoke: list[bool]
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
        async with aclosing(self.llm.stream(messages, tools, stop=[f"<{CUSTOMER_TAG}>"])) as stream:
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
        self, session: AgentSession, ctx: ToolContext, turn: _Turn, call: ToolCall, emit: Callable
    ) -> str:
        args = call.parsed_arguments()
        emit(ToolEvent(name=call.name, status="started"))
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
        turn.tool_log.append(f"{call.name}: {'ok' if ok else 'failed'}")
        emit(ToolEvent(name=call.name, status="finished", ok=ok))
        return json.dumps(payload, ensure_ascii=False, default=str)

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
        emit(
            DoneEvent(
                reply=self._spoken(turn),
                outcome="escalated" if reference else "failed",
                iterations=turn.iterations,
            )
        )

    # --- helpers --------------------------------------------------------------------------

    def _say(self, turn: _Turn, emit: Callable, text: str) -> None:
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

    async def _record(self, db: AsyncSession, session: AgentSession, turn: _Turn, text: str):
        if session.conversation_id is None:
            return
        reply = self._spoken(turn) + (" [interrupted]" if turn.interrupted else "")
        entries = [(TurnRole.CUSTOMER, text)]
        entries += [(TurnRole.TOOL, line) for line in turn.tool_log]
        entries.append((TurnRole.ASSISTANT, reply.strip()))
        await db.rollback()
        await transcript.record_turns(
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
