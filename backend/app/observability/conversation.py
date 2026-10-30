"""One conversation, turn by turn: what was said, the tool calls, the policy decisions and
the timings. Used by the replay view (/admin/replay) and its CLI:

    python -m app.observability.conversation 42          # readable
    python -m app.observability.conversation 42 --json   # the same data as the API
    python -m app.observability.conversation --list      # recent conversations

Tool calls come from the transcript's tool lines (the agent's JSON tool log). Policy
decisions and timings are joined to the turn they happened in: decisions by time (a
decision belongs to the first turn recorded at or after it), timings by the turn index the
agent stored with them.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.db.models import (
    Conversation,
    ConversationTurn,
    Customer,
    PolicyDecisionLog,
    Ticket,
    TurnMetric,
)
from app.domain.enums import TurnRole
from app.observability import tracing
from app.tools.common import ticket_ref

METRIC_FIELDS = (
    "turn_id",
    "outcome",
    "model",
    "prompt_version",
    "llm_calls",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "cost_usd",
    "llm_first_token_ms",
    "llm_ms",
    "tool_calls",
    "tool_ms",
    "agent_ms",
    "vad_ms",
    "stt_ms",
    "agent_first_text_ms",
    "first_sentence_ms",
    "tts_first_audio_ms",
    "time_to_first_audio_ms",
    "filler",
    "interrupted",
)


async def list_recent(db: AsyncSession, limit: int = 20) -> list[dict[str, Any]]:
    rows = await db.execute(
        select(Conversation, Customer.full_name)
        .outerjoin(Customer, Customer.id == Conversation.customer_id)
        .order_by(Conversation.id.desc())
        .limit(limit)
    )
    return [
        {
            "id": conversation.id,
            "channel": str(conversation.channel),
            "started_at": conversation.started_at,
            "ended_at": conversation.ended_at,
            "customer": name,
        }
        for conversation, name in rows
    ]


async def load(db: AsyncSession, conversation_id: int, *, salt: str) -> dict[str, Any] | None:
    conversation = await db.get(Conversation, conversation_id)
    if conversation is None:
        return None
    customer = (
        await db.get(Customer, conversation.customer_id) if conversation.customer_id else None
    )
    lines = (
        await db.scalars(
            select(ConversationTurn)
            .where(ConversationTurn.conversation_id == conversation_id)
            .order_by(ConversationTurn.turn_index)
        )
    ).all()
    decisions = (
        await db.scalars(
            select(PolicyDecisionLog)
            .where(PolicyDecisionLog.conversation_id == conversation_id)
            .order_by(PolicyDecisionLog.decided_at, PolicyDecisionLog.id)
        )
    ).all()
    timings = (
        await db.scalars(select(TurnMetric).where(TurnMetric.conversation_id == conversation_id))
    ).all()
    tickets = (
        await db.scalars(
            select(Ticket).where(Ticket.conversation_id == conversation_id).order_by(Ticket.id)
        )
    ).all()

    turns = _turns(lines)
    by_index = {m.turn_index: m for m in timings if m.turn_index is not None}
    for turn in turns:
        metric = by_index.get(turn["index"])
        turn["timings"] = {f: getattr(metric, f) for f in METRIC_FIELDS} if metric else None
    _attach_decisions(turns, decisions)

    return {
        "conversation": {
            "id": conversation.id,
            "channel": str(conversation.channel),
            "started_at": conversation.started_at,
            "verified_at": conversation.verified_at,
            "ended_at": conversation.ended_at,
            "customer": {"id": customer.id, "name": customer.full_name} if customer else None,
        },
        "trace_id": tracing.trace_id_hex(conversation_id, salt),
        "turns": turns,
        "tickets": [
            {
                "reference": ticket_ref(t.id),
                "category": str(t.category),
                "status": str(t.status),
                "summary": t.summary,
                "created_at": t.created_at,
            }
            for t in tickets
        ],
        "totals": _totals(timings),
    }


def _turns(lines: list[ConversationTurn]) -> list[dict[str, Any]]:
    """Group transcript lines into customer turns: what they said, the tool calls, the reply."""
    turns: list[dict[str, Any]] = []
    for line in lines:
        if line.role == TurnRole.CUSTOMER or not turns:
            turns.append(
                {
                    "index": line.turn_index,
                    "at": line.created_at,
                    "customer": line.content if line.role == TurnRole.CUSTOMER else None,
                    "tool_calls": [],
                    "assistant": None,
                    "policy_decisions": [],
                }
            )
            if line.role == TurnRole.CUSTOMER:
                continue
        turn = turns[-1]
        if line.role == TurnRole.TOOL:
            try:
                turn["tool_calls"].append(json.loads(line.content))
            except json.JSONDecodeError:
                turn["tool_calls"].append({"raw": line.content})
        elif line.role == TurnRole.ASSISTANT:
            turn["assistant"] = line.content
    return turns


def _attach_decisions(turns: list[dict[str, Any]], decisions: list[PolicyDecisionLog]) -> None:
    for decision in decisions:
        entry = {
            "at": decision.decided_at,
            "stage": decision.stage,
            "tool": decision.tool_name,
            "action_id": decision.action_id,
            "verdict": decision.verdict,
            "rule": decision.rule,
            "reason": decision.reason,
            "message": decision.message,
            "amount_paise": decision.amount_paise,
            "policy_version": decision.policy_version,
        }
        home = next((t for t in turns if t["at"] >= decision.decided_at), None)
        if home is None and turns:
            home = turns[-1]
        if home is not None:
            home["policy_decisions"].append(entry)


def _totals(timings: list[TurnMetric]) -> dict[str, Any]:
    def total(field: str) -> float:
        return sum(getattr(m, field) or 0 for m in timings)

    ttfa = sorted(m.time_to_first_audio_ms for m in timings if m.time_to_first_audio_ms)
    costs = [m.cost_usd for m in timings if m.cost_usd is not None]
    return {
        "turns_measured": len(timings),
        "llm_calls": int(total("llm_calls")),
        "prompt_tokens": int(total("prompt_tokens")),
        "completion_tokens": int(total("completion_tokens")),
        "cost_usd": round(sum(costs), 6) if costs else None,
        "time_to_first_audio_ms": {"min": ttfa[0], "max": ttfa[-1]} if ttfa else None,
    }


# --- text rendering -----------------------------------------------------------------------


def _fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:,.0f} ms"


def render(data: dict[str, Any]) -> str:
    c = data["conversation"]
    who = c["customer"]["name"] if c["customer"] else "not verified"
    out = [
        f"Conversation {c['id']} ({c['channel']}), customer: {who}",
        f"  started {c['started_at']}, ended {c['ended_at'] or '-'}",
        f"  trace {data['trace_id']}",
        "",
    ]
    for turn in data["turns"]:
        out.append(f"[{turn['index']}] customer: {turn['customer']}")
        for call in turn["tool_calls"]:
            state = "ok" if call.get("ok") else f"failed ({call.get('error')})"
            source = f", {call['source']}" if call.get("source") else ""
            args = json.dumps(call.get("arguments", {}), ensure_ascii=False)
            tag = f"[{call.get('phase')}{source}]"
            out.append(f"      tool {call.get('tool')} {tag} {args} → {state}")
        for d in turn["policy_decisions"]:
            verdict = f"{d['verdict']} ({d['rule']})"
            out.append(f"      policy {d['stage']} {d['tool']}: {verdict} {d['message']}")
        out.append(f"    agent: {turn['assistant']}")
        t = turn["timings"]
        if t:
            parts = [
                f"first audio {_fmt_ms(t['time_to_first_audio_ms'])}"
                if t["time_to_first_audio_ms"] is not None
                else None,
                f"vad {_fmt_ms(t['vad_ms'])}" if t["vad_ms"] is not None else None,
                f"stt {_fmt_ms(t['stt_ms'])}" if t["stt_ms"] is not None else None,
                f"llm first token {_fmt_ms(t['llm_first_token_ms'])}",
                f"llm {_fmt_ms(t['llm_ms'])} over {t['llm_calls']} call(s)",
                f"tools {_fmt_ms(t['tool_ms'])}",
                f"tts first audio {_fmt_ms(t['tts_first_audio_ms'])}"
                if t["tts_first_audio_ms"] is not None
                else None,
                f"{t['prompt_tokens']}+{t['completion_tokens']} tokens",
                f"${t['cost_usd']:.6f}" if t["cost_usd"] is not None else None,
                "filler" if t["filler"] else None,
                "interrupted" if t["interrupted"] else None,
            ]
            out.append("    timing: " + ", ".join(p for p in parts if p))
        out.append("")
    for ticket in data["tickets"]:
        out.append(f"ticket {ticket['reference']} ({ticket['category']}): {ticket['summary']}")
    return "\n".join(out)


def _json_default(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else str(value)


async def _main(args) -> int:
    from app.config import get_settings

    settings = get_settings()
    engine = create_async_engine(settings.database_url())
    try:
        async with AsyncSession(engine) as db:
            if args.list:
                for row in await list_recent(db, args.limit):
                    print(f"{row['id']:>6}  {row['channel']:<6} {row['started_at']}  "
                          f"{row['customer'] or '-'}")  # fmt: skip
                return 0
            data = await load(db, args.conversation_id, salt=settings.postgres_db)
    finally:
        await engine.dispose()
    if data is None:
        print(f"No conversation {args.conversation_id}.", file=sys.stderr)
        return 1
    print(json.dumps(data, indent=2, default=_json_default) if args.json else render(data))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show a conversation turn by turn.")
    parser.add_argument("conversation_id", type=int, nargs="?")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--list", action="store_true", help="recent conversations")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)
    if not args.list and args.conversation_id is None:
        parser.error("give a conversation id, or --list")
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())
