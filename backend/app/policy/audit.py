"""Decision log. Each decision is logged immediately and persisted to policy_decisions.

Rows are written after the caller's own transaction ends (flush_decisions), so a decision
that rejects a write is not rolled back together with that write.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from app.db.models import PolicyDecisionLog
from app.policy.config import get_policy_bundle
from app.policy.engine import PolicyDecision

if TYPE_CHECKING:
    from app.tools.context import ToolContext

log = logging.getLogger("app.policy.decisions")


class Stage(StrEnum):
    PROPOSAL = "proposal"
    EXECUTION = "execution"


@dataclass(frozen=True)
class DecisionRecord:
    decided_at: datetime
    stage: Stage
    tool_name: str
    action_id: str
    verdict: str
    rule: str
    reason: str | None
    message: str
    amount_paise: int | None
    policy_version: str
    facts: dict[str, Any]
    customer_id: int | None
    conversation_id: int | None


def record_decision(
    ctx: "ToolContext", decision: PolicyDecision, *, stage: Stage, tool: str, action_id: str
) -> DecisionRecord:
    record = DecisionRecord(
        decided_at=ctx.now(),
        stage=stage,
        tool_name=tool,
        action_id=action_id,
        verdict=decision.verdict.value,
        rule=decision.rule,
        reason=decision.reason.value if decision.reason else None,
        message=decision.message,
        amount_paise=decision.amount_paise,
        policy_version=get_policy_bundle()[1],
        facts=decision.facts,
        customer_id=ctx.state.customer_id,
        conversation_id=ctx.conversation_id,
    )
    log.info("policy decision %s", json.dumps(asdict(record), default=str, sort_keys=True))
    ctx.pending_decisions.append(record)
    return record


async def flush_decisions(ctx: "ToolContext") -> None:
    """Persist buffered decisions in their own transaction. Never fails the caller."""
    if not ctx.pending_decisions:
        return
    records, ctx.pending_decisions = ctx.pending_decisions, []
    try:
        ctx.db.add_all(PolicyDecisionLog(**asdict(r)) for r in records)
        await ctx.db.commit()
    except Exception:
        log.exception("could not persist %d policy decisions", len(records))
        await ctx.db.rollback()
