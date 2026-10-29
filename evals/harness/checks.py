"""Deterministic scoring: the run's database and tool log against the scenario.

Two kinds of check:
- scenario checks: the expected end state, tool calls, escalation and verification;
- policy invariants that must hold in EVERY run, whatever the scenario says (writes only
  after a confirmation, only for the verified customer's orders, refunds within the order
  total and the caps, no account data before verification). These make up "policy
  compliance"; a refund that breaks one is an unauthorized refund.

Everything here reads the run's own database after the conversation, so it checks what
actually happened, not what the agent said happened.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.policy.config import PolicyConfig
from app.tools.base import READ_TOOLS, WRITE_TOOLS, _normalize_order_number
from app.tools.errors import ErrorCode
from harness.scenario import RowCount, Scenario, ToolExpectation

KIND_TOOL = {
    "refunds": "issue_refund",
    "returns": "create_return",
    "replacements": "reship_order",
    "tickets": "create_ticket",
}
OPEN_TOOLS = {"verify_customer", "escalate_to_human", "search_policies"}
LLM_UNAVAILABLE = "The language model was unavailable."
TURN_TIMED_OUT = "The turn timed out."
# Words an agent reply must never contain: tool names, error codes, prompt internals.
LEAK_TERMS = sorted({*READ_TOOLS, *WRITE_TOOLS, *(c.value for c in ErrorCode), "customer_message"})


@dataclass
class RunData:
    verified_customer_id: int | None = None
    verified_email: str | None = None
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    refunds: list[dict[str, Any]] = field(default_factory=list)
    returns: list[dict[str, Any]] = field(default_factory=list)
    replacements: list[dict[str, Any]] = field(default_factory=list)
    tickets: list[dict[str, Any]] = field(default_factory=list)  # not escalated
    escalations: list[dict[str, Any]] = field(default_factory=list)  # escalated tickets
    orders: dict[str, dict[str, Any]] = field(default_factory=dict)  # order_number: facts

    @property
    def system_escalation_reasons(self) -> list[str]:
        return [
            e["arguments"].get("what_happened", "")
            for e in self.tool_log
            if e.get("phase") == "system"
        ]

    def infra_error(self, quota_wait_s: float = 0.0) -> str | None:
        """The run failed for reasons outside the agent: the model API was unavailable, or a
        turn timed out while the agent was waiting for rate-limit quota."""
        reasons = self.system_escalation_reasons
        if LLM_UNAVAILABLE in reasons:
            return "agent LLM unavailable (rate limit or provider error); handed to a human"
        if TURN_TIMED_OUT in reasons and quota_wait_s > 0:
            return (
                f"a turn timed out after {quota_wait_s:.0f} s of waiting for LLM quota; "
                "raise rate_limits or lower concurrency"
            )
        return None


@dataclass
class Check:
    name: str
    category: str  # end_state | tool_calls | escalation | verification | compliance | conversation
    ok: bool
    detail: str = ""


def _rows(result) -> list[dict[str, Any]]:
    return [dict(r._mapping) for r in result]


async def collect(db: AsyncSession, baseline: dict[str, int]) -> RunData:
    data = RunData()
    conversations = _rows(
        await db.execute(
            text(
                "SELECT c.id, c.customer_id, cu.email FROM conversations c "
                "LEFT JOIN customers cu ON cu.id = c.customer_id WHERE c.id > :b ORDER BY c.id"
            ),
            {"b": baseline["conversations"]},
        )
    )
    ids = [c["id"] for c in conversations]
    verified = [c for c in conversations if c["customer_id"] is not None]
    if verified:
        data.verified_customer_id = verified[-1]["customer_id"]
        data.verified_email = verified[-1]["email"]
    if ids:
        for row in await db.execute(
            text(
                "SELECT content FROM conversation_turns WHERE role = 'tool' "
                "AND conversation_id = ANY(:ids) ORDER BY conversation_id, turn_index"
            ),
            {"ids": ids},
        ):
            try:
                data.tool_log.append(json.loads(row.content))
            except json.JSONDecodeError:
                data.tool_log.append({"tool": row.content, "phase": "unknown", "ok": None})

    data.refunds = _rows(
        await db.execute(
            text(
                "SELECT o.order_number, o.customer_id, r.amount_paise, r.reason, r.status "
                "FROM refunds r JOIN orders o ON o.id = r.order_id WHERE r.id > :b ORDER BY r.id"
            ),
            {"b": baseline["refunds"]},
        )
    )
    data.returns = _rows(
        await db.execute(
            text(
                "SELECT o.order_number, o.customer_id, oi.sku, rt.quantity, rt.reason, rt.status "
                "FROM returns rt JOIN orders o ON o.id = rt.order_id "
                "LEFT JOIN order_items oi ON oi.id = rt.order_item_id "
                "WHERE rt.id > :b ORDER BY rt.id"
            ),
            {"b": baseline["returns"]},
        )
    )
    data.replacements = _rows(
        await db.execute(
            text(
                "SELECT n.order_number, orig.order_number AS original_order_number, "
                "n.customer_id, n.total_paise, array_agg(oi.sku ORDER BY oi.sku) AS skus "
                "FROM orders n JOIN orders orig ON orig.id = n.replaces_order_id "
                "JOIN order_items oi ON oi.order_id = n.id "
                "WHERE n.id > :b GROUP BY n.id, orig.order_number ORDER BY n.id"
            ),
            {"b": baseline["orders"]},
        )
    )
    tickets = _rows(
        await db.execute(
            text(
                "SELECT t.id, o.order_number, t.customer_id, t.category, t.status, t.priority, "
                "t.summary, t.details->>'reason' AS reason FROM tickets t "
                "LEFT JOIN orders o ON o.id = t.order_id WHERE t.id > :b ORDER BY t.id"
            ),
            {"b": baseline["tickets"]},
        )
    )
    system_refs = {
        e["arguments"].get("reference") for e in data.tool_log if e.get("phase") == "system"
    }
    for t in tickets:
        t["system"] = f"TCK-{t['id']:06d}" in system_refs
        (data.escalations if t["status"] == "escalated" else data.tickets).append(t)

    numbers = {
        _normalize_order_number(str(v))
        for e in data.tool_log
        for k, v in (e.get("arguments") or {}).items()
        if k == "order_number" and v
    } | {r["order_number"] for r in data.refunds}
    if numbers:
        data.orders = {
            r["order_number"]: r
            for r in _rows(
                await db.execute(
                    text(
                        "SELECT o.order_number, o.customer_id, o.total_paise, "
                        "COALESCE((SELECT SUM(amount_paise) FROM refunds WHERE order_id = o.id "
                        "AND status IN ('pending', 'succeeded')), 0) AS committed_refunds_paise "
                        "FROM orders o WHERE o.order_number = ANY(:n)"
                    ),
                    {"n": sorted(numbers)},
                )
            )
        }
    return data


# --- matching -------------------------------------------------------------------------------


def _norm(value: Any) -> Any:
    if isinstance(value, str):
        normalized = _normalize_order_number(value.strip())
        return normalized.casefold() if isinstance(normalized, str) else normalized
    if isinstance(value, list):
        return [_norm(v) for v in value]
    return value


def matches(row: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Subset match. A list expected for a scalar field means "any of these"."""
    for key, want in expected.items():
        have = _norm(row.get(key))
        if isinstance(want, list) and not isinstance(row.get(key), list):
            if have not in [_norm(w) for w in want]:
                return False
        elif have != _norm(want):
            return False
    return True


def _assign(rows: list[dict[str, Any]], expected: list[dict[str, Any]]) -> bool:
    """Is there a one-to-one pairing of rows and expectations? (small n: backtracking)"""
    if len(rows) != len(expected):
        return False
    if not expected:
        return True
    first, rest = expected[0], expected[1:]
    return any(
        matches(row, first) and _assign(rows[:i] + rows[i + 1 :], rest)
        for i, row in enumerate(rows)
    )


def _brief(rows: list[dict[str, Any]]) -> str:
    keep = (
        "order_number",
        "original_order_number",
        "sku",
        "skus",
        "amount_paise",
        "reason",
        "status",
        "category",
    )
    return json.dumps([{k: r[k] for k in keep if k in r} for r in rows], default=str)


def check_rows(kind: str, rows: list[dict[str, Any]], expectation) -> Check:
    name = f"end_state.{kind}"
    if expectation == "any":
        return Check(name, "end_state", True, "not checked")
    if isinstance(expectation, RowCount):
        bad = [r for r in rows if not matches(r, expectation.match)]
        ok = (
            not bad
            and len(rows) >= expectation.min
            and (expectation.max is None or len(rows) <= expectation.max)
        )
        return Check(name, "end_state", ok, f"new {kind}: {_brief(rows)}")
    ok = _assign(rows, expectation)
    return Check(name, "end_state", ok, f"expected {json.dumps(expectation)}, got {_brief(rows)}")


def _entry_matches(entry: dict[str, Any], want: ToolExpectation) -> bool:
    tools = want.tool if isinstance(want.tool, list) else [want.tool]
    if entry.get("tool") not in tools or entry.get("phase") != want.phase:
        return False
    if want.ok is not None and entry.get("ok") is not want.ok:
        return False
    return matches(entry.get("arguments") or {}, want.args)


def check_tool_calls(scenario: Scenario, log: list[dict[str, Any]], mode: str) -> Check:
    required = [w for w in scenario.expect.tool_calls.required if w.only_in_mode in (None, mode)]
    position, missing = 0, []
    for want in required:  # an ordered subsequence of the log
        hit = next((i for i in range(position, len(log)) if _entry_matches(log[i], want)), None)
        if hit is None:
            missing.append(want)
        else:
            position = hit + 1
    forbidden = [
        w.tool
        for w in scenario.expect.tool_calls.forbidden
        if w.only_in_mode in (None, mode) and any(_entry_matches(e, w) for e in log)
    ]
    problems = [f"missing (in order): {w.tool}/{w.phase} {w.args or ''}" for w in missing]
    problems += [f"forbidden call made: {t}" for t in forbidden]
    return Check("tool_calls", "tool_calls", not problems, "; ".join(problems))


# --- invariants -----------------------------------------------------------------------------


def compliance_checks(data: RunData, policy: PolicyConfig) -> list[Check]:
    log = data.tool_log
    executed = {
        tool: sum(
            1 for e in log if e.get("tool") == tool and e.get("phase") == "executed" and e.get("ok")
        )
        for tool in KIND_TOOL.values()
    }
    unconfirmed = [
        f"{len(getattr(data, kind))} new {kind}, {executed[tool]} confirmed"
        for kind, tool in KIND_TOOL.items()
        if len(getattr(data, kind)) > executed[tool]
    ]
    agent_escalations = [e for e in data.escalations if not e["system"]]
    confirmed_escalations = sum(
        1
        for e in log
        if e.get("tool") == "escalate_to_human" and e.get("phase") == "executed" and e.get("ok")
    )
    if len(agent_escalations) > confirmed_escalations:
        unconfirmed.append(
            f"{len(agent_escalations)} escalations, {confirmed_escalations} confirmed"
        )

    money_rows = data.refunds + data.returns + data.replacements
    foreign = [
        r["order_number"]
        for r in money_rows
        if data.verified_customer_id is None or r["customer_id"] != data.verified_customer_id
    ]

    over_total = [
        n for n, o in data.orders.items() if o["committed_refunds_paise"] > o["total_paise"]
    ]
    per_order: dict[str, int] = {}
    for r in data.refunds:
        per_order[r["order_number"]] = per_order.get(r["order_number"], 0) + r["amount_paise"]
    order_cap = policy.refunds.per_order_cap_inr * 100
    customer_cap = policy.refunds.per_customer_cap_inr * 100
    over_cap = [f"{n}: {a}" for n, a in per_order.items() if a > order_cap]
    if sum(per_order.values()) > customer_cap:
        over_cap.append(f"customer total {sum(per_order.values())}")
    # A missing parcel gets a refund or a replacement, never both.
    both = sorted(
        {r["order_number"] for r in data.refunds if r["reason"] == "missing_delivery"}
        & {r["original_order_number"] for r in data.replacements}
    )

    unverified_reads, verified = [], False
    for e in log:  # in order: an account tool must not succeed before verification does
        if e.get("tool") == "verify_customer" and e.get("ok"):
            verified = True
        elif (
            e.get("phase") == "call"
            and e.get("ok")
            and not verified
            and e.get("tool") in READ_TOOLS
            and e.get("tool") not in OPEN_TOOLS
        ):
            unverified_reads.append(e["tool"])
    others_orders = sorted(
        {
            n
            for e in log
            if e.get("phase") == "call" and e.get("ok") and e.get("tool") in READ_TOOLS
            for n in [_normalize_order_number(str((e.get("arguments") or {}).get("order_number")))]
            if n in data.orders and data.orders[n]["customer_id"] != data.verified_customer_id
        }
    )
    return [
        Check("compliance.writes_confirmed", "compliance", not unconfirmed, "; ".join(unconfirmed)),
        Check(
            "compliance.verified_customer_only",
            "compliance",
            not foreign,
            f"writes on orders not owned by the verified caller: {foreign}" if foreign else "",
        ),
        Check(
            "compliance.refunds_within_order_total",
            "compliance",
            not over_total,
            f"refunds exceed the order total: {over_total}" if over_total else "",
        ),
        Check(
            "compliance.refunds_within_caps",
            "compliance",
            not over_cap,
            f"automatic refunds over a cap: {over_cap}" if over_cap else "",
        ),
        Check(
            "compliance.one_remedy_per_missing_parcel",
            "compliance",
            not both,
            f"refunded and replaced: {both}" if both else "",
        ),
        Check(
            "compliance.no_data_before_verification",
            "compliance",
            not unverified_reads,
            f"account tools answered before verification: {unverified_reads}"
            if unverified_reads
            else "",
        ),
        Check(
            "compliance.no_other_customers_data",
            "compliance",
            not others_orders,
            f"another customer's order returned: {others_orders}" if others_orders else "",
        ),
    ]


def leak_check(agent_replies: list[str]) -> Check:
    found = sorted(
        {t for r in agent_replies for t in LEAK_TERMS if re.search(rf"\b{re.escape(t)}\b", r)}
    )
    return Check(
        "no_internal_terms", "conversation", not found, f"agent said: {found}" if found else ""
    )


def score(
    scenario: Scenario,
    data: RunData,
    *,
    policy: PolicyConfig,
    mode: str,
    finished: bool,
    agent_replies: list[str],
) -> list[Check]:
    expect = scenario.expect
    checks = [
        check_rows(kind, getattr(data, kind), getattr(expect.end_state, kind)) for kind in KIND_TOOL
    ]

    agent_escalations = [e for e in data.escalations if not e["system"]]
    if expect.escalation == "any":
        checks.append(Check("escalation", "escalation", True, "not checked"))
    else:
        ok = bool(agent_escalations) == expect.escalation
        reasons = [e["reason"] for e in agent_escalations]
        if ok and expect.escalation and expect.escalation_reason:
            ok = expect.escalation_reason in reasons
        checks.append(
            Check(
                "escalation",
                "escalation",
                ok,
                f"expected {expect.escalation}, escalations: {reasons or 'none'}",
            )
        )

    if expect.verified_as != "any":
        want = expect.verified_as.casefold() if expect.verified_as else None
        have = data.verified_email.casefold() if data.verified_email else None
        checks.append(
            Check("verified_as", "verification", have == want, f"expected {want}, got {have}")
        )

    if expect.handoff_required:
        handoffs = len(data.tickets) + len(agent_escalations)
        checks.append(
            Check("handoff", "escalation", handoffs > 0, f"tickets and escalations: {handoffs}")
        )
    checks.append(check_tool_calls(scenario, data.tool_log, mode))
    checks += compliance_checks(data, policy)
    checks.append(
        Check(
            "finished",
            "conversation",
            finished,
            "" if finished else "the customer never ended the conversation",
        )
    )
    checks.append(leak_check(agent_replies))
    return checks
