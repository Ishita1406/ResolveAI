"""PolicyEngine: the deterministic gate between what the agent proposes and what may run.

evaluate(action, facts) is pure: no database access and no wall clock. The caller loads the
facts (app.policy.facts) and supplies `now`, so the same inputs always give the same decision.
Every decision names the rule that produced it; rules are named after policy.yaml keys
where a configured value decides the outcome.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from functools import lru_cache
from typing import Any

from app.db.models import Order, OrderItem, Shipment
from app.domain.enums import (
    OrderStatus,
    RefundReason,
    RefundRequestReason,
    RefundStatus,
    ReturnStatus,
    ShipmentStatus,
)
from app.domain.formatting import format_date, format_inr
from app.policy.config import Issue, PolicyConfig, Remedy, get_policy


class Verdict(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUIRE_HUMAN = "require_human"


class Reason(StrEnum):
    ORDER_CANCELLED = "order_cancelled"
    NOT_SHIPPED = "not_shipped"
    NOT_DELIVERED = "not_delivered"
    WINDOW_EXPIRED = "window_expired"
    ALREADY_RETURNED = "already_returned"
    ALREADY_REFUNDED = "already_refunded"
    ALREADY_RESOLVED = "already_resolved"
    NO_RETURN_ON_FILE = "no_return_on_file"
    NOT_YET_OVERDUE = "not_yet_overdue"
    REMEDY_NOT_ALLOWED = "remedy_not_allowed"
    DELIVERED_PER_CARRIER = "delivered_per_carrier"
    RETURNED_TO_SENDER = "returned_to_sender"
    GOODWILL_REQUESTED = "goodwill_requested"
    OVER_ORDER_REFUND_CAP = "over_order_refund_cap"
    OVER_CUSTOMER_REFUND_CAP = "over_customer_refund_cap"
    OVER_RESHIP_VALUE_CAP = "over_reship_value_cap"
    DUPLICATE_TICKET = "duplicate_ticket"


@dataclass(frozen=True)
class PolicyDecision:
    verdict: Verdict
    rule: str
    reason: Reason | None
    message: str
    amount_paise: int | None = None
    # JSON-safe values the rule looked at; stored with the audit record.
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def approved(self) -> bool:
        return self.verdict is Verdict.APPROVE


@dataclass(frozen=True)
class PolicyFacts:
    """Everything a rule may look at, loaded by the caller."""

    now: datetime
    order: Order | None = None
    customer_refunds_in_window_paise: int = 0
    has_replacement: bool = False
    open_ticket_reference: str | None = None


@dataclass(frozen=True)
class ProposedAction:
    tool: str
    args: Any  # the write tool's validated input model


ISSUE_TEXT = {
    Issue.MISSING_DELIVERY: "a missing delivery",
    Issue.DAMAGED: "a damaged item",
    Issue.WRONG_ITEM: "a wrong item",
    Issue.NOT_AS_DESCRIBED: "an item that isn't as described",
    Issue.NO_LONGER_NEEDED: "an item you no longer need",
}
_REMEDY_TEXT = {Remedy.RESHIP: "a free replacement", Remedy.REFUND: "a refund"}


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def shipment_for(order: Order, item: OrderItem) -> Shipment | None:
    return next((s for s in order.shipments if s.id == item.shipment_id), None)


def committed_refunds_paise(order: Order) -> int:
    return sum(
        r.amount_paise
        for r in order.refunds
        if r.status in (RefundStatus.SUCCEEDED, RefundStatus.PENDING)
    )


def refundable_balance_paise(order: Order) -> int:
    return max(order.total_paise - committed_refunds_paise(order), 0)


class PolicyEngine:
    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    # --- public ---------------------------------------------------------------------------

    def evaluate(self, action: ProposedAction, facts: PolicyFacts) -> PolicyDecision:
        rules = {
            "create_return": self._create_return,
            "issue_refund": self._issue_refund,
            "reship_order": self._reship_order,
            "create_ticket": self._create_ticket,
            "escalate_to_human": self._escalate_to_human,
        }
        if action.tool not in rules:
            raise ValueError(f"no policy for tool {action.tool!r}")
        return rules[action.tool](action.args, facts)

    def return_deadline(self, delivered_at: datetime) -> datetime:
        return delivered_at + timedelta(days=self.config.returns.window_days)

    def assess_return(self, order: Order, item: OrderItem, now: datetime) -> PolicyDecision:
        """Can this item be returned now? Approval carries the refund amount it would earn."""
        n = order.order_number
        if order.status is OrderStatus.CANCELLED:
            return self._reject("order.status", Reason.ORDER_CANCELLED, n)
        shipment = shipment_for(order, item)
        if shipment is None:
            return self._reject("item.fulfilment", Reason.NOT_SHIPPED, n)
        if shipment.delivered_at is None:
            return self._reject("item.fulfilment", Reason.NOT_DELIVERED, n)
        if any(
            r.order_item_id == item.id and r.status is not ReturnStatus.REJECTED
            for r in order.returns
        ):
            return self._reject("returns.one_per_item", Reason.ALREADY_RETURNED, n)
        balance = refundable_balance_paise(order)
        if balance <= 0:
            return self._reject("refunds.balance", Reason.ALREADY_REFUNDED, n)
        deadline = self.return_deadline(shipment.delivered_at)
        if now > deadline:
            return self._reject(
                "returns.window_days",
                Reason.WINDOW_EXPIRED,
                n,
                deadline=deadline,
                delivered_at=shipment.delivered_at,
            )
        amount = min(item.line_total_paise, balance)
        return self._approve("returns.window_days", amount, deadline=_iso(deadline))

    def check_refund_caps(self, order: Order, amount: int, facts: PolicyFacts) -> PolicyDecision:
        n = order.order_number
        refunds = self.config.refunds
        committed = committed_refunds_paise(order)
        if committed + amount > refunds.per_order_cap_paise:
            return self._escalate(
                self.config.escalate_when.over_order_refund_cap,
                "refunds.per_order_cap_inr",
                Reason.OVER_ORDER_REFUND_CAP,
                n,
                amount=amount,
                committed_paise=committed,
                cap_paise=refunds.per_order_cap_paise,
            )
        prior = facts.customer_refunds_in_window_paise
        if prior + amount > refunds.per_customer_cap_paise:
            return self._escalate(
                self.config.escalate_when.over_customer_refund_cap,
                "refunds.per_customer_cap_inr",
                Reason.OVER_CUSTOMER_REFUND_CAP,
                n,
                amount=amount,
                customer_refunds_in_window_paise=prior,
                cap_paise=refunds.per_customer_cap_paise,
            )
        return self._approve("refunds.caps", amount, committed_paise=committed)

    def assess_missing_delivery(
        self, order: Order, items: list[OrderItem], now: datetime
    ) -> PolicyDecision:
        """Is every parcel holding these items missing? Human review outranks a refusal."""
        n = order.order_number
        escalate_when = self.config.escalate_when
        grace = timedelta(days=self.config.missing_delivery.grace_days_after_eta)
        decisions: list[PolicyDecision] = []
        for shipment in {i.shipment_id: shipment_for(order, i) for i in items}.values():
            if shipment is None:
                decisions.append(self._reject("item.fulfilment", Reason.NOT_SHIPPED, n))
            elif shipment.status is ShipmentStatus.LOST:
                decisions.append(self._approve("missing_delivery.lost", None))
            elif shipment.status is ShipmentStatus.DELIVERED:
                decisions.append(
                    self._escalate(
                        escalate_when.carrier_marked_delivered,
                        "escalate_when.carrier_marked_delivered",
                        Reason.DELIVERED_PER_CARRIER,
                        n,
                        delivered_at=_iso(shipment.delivered_at),
                    )
                )
            elif shipment.status is ShipmentStatus.RETURNED_TO_SENDER:
                decisions.append(
                    self._escalate(
                        escalate_when.returned_to_sender,
                        "escalate_when.returned_to_sender",
                        Reason.RETURNED_TO_SENDER,
                        n,
                    )
                )
            else:
                eta = shipment.estimated_delivery_at
                missing_after = eta + grace if eta else None
                if missing_after is not None and now > missing_after:
                    decisions.append(
                        self._approve(
                            "missing_delivery.grace_days_after_eta",
                            None,
                            missing_after=_iso(missing_after),
                        )
                    )
                else:
                    decisions.append(
                        self._reject(
                            "missing_delivery.grace_days_after_eta",
                            Reason.NOT_YET_OVERDUE,
                            n,
                            eta=eta,
                            missing_after=missing_after,
                        )
                    )
        for verdict in (Verdict.REQUIRE_HUMAN, Verdict.REJECT):
            hit = next((d for d in decisions if d.verdict is verdict), None)
            if hit:
                return hit
        return decisions[0]

    # --- per-tool rules -------------------------------------------------------------------

    def _create_return(self, args: Any, facts: PolicyFacts) -> PolicyDecision:
        order = self._order(facts)
        return self.assess_return(order, self._item(order, args.sku), facts.now)

    def _issue_refund(self, args: Any, facts: PolicyFacts) -> PolicyDecision:
        order = self._order(facts)
        n = order.order_number
        if order.status is OrderStatus.CANCELLED:
            return self._reject("order.status", Reason.ORDER_CANCELLED, n)
        balance = refundable_balance_paise(order)
        if balance <= 0:
            return self._reject("refunds.balance", Reason.ALREADY_REFUNDED, n)

        match RefundRequestReason(args.reason):
            case RefundRequestReason.GOODWILL:
                return self._escalate(
                    self.config.escalate_when.goodwill_refund_requested,
                    "escalate_when.goodwill_refund_requested",
                    Reason.GOODWILL_REQUESTED,
                    n,
                )
            case RefundRequestReason.RETURN:
                item = self._item(order, args.sku)
                refunded = {r.return_id for r in order.refunds if r.return_id is not None}
                ret = next(
                    (
                        r
                        for r in order.returns
                        if r.order_item_id == item.id
                        and r.status is not ReturnStatus.REJECTED
                        and r.id not in refunded
                    ),
                    None,
                )
                if ret is None:
                    already = any(r.order_item_id == item.id for r in order.returns)
                    reason = Reason.ALREADY_REFUNDED if already else Reason.NO_RETURN_ON_FILE
                    return self._reject("refunds.requires_return", reason, n)
                issue = Issue(ret.reason)
                if not self.config.allows(issue, Remedy.REFUND):
                    return self._remedy_not_allowed(issue, Remedy.REFUND)
                amount = min(item.line_total_paise, balance)
            case RefundRequestReason.MISSING_DELIVERY:
                if not self.config.allows(Issue.MISSING_DELIVERY, Remedy.REFUND):
                    return self._remedy_not_allowed(Issue.MISSING_DELIVERY, Remedy.REFUND)
                if facts.has_replacement or self._missing_delivery_refunded(order):
                    return self._reject("missing_delivery.one_remedy", Reason.ALREADY_RESOLVED, n)
                items = [self._item(order, args.sku)] if args.sku else order.items
                missing = self.assess_missing_delivery(order, items, facts.now)
                if not missing.approved:
                    return missing
                amount = (
                    min(sum(i.line_total_paise for i in items), balance) if args.sku else balance
                )

        return self.check_refund_caps(order, amount, facts)

    def _reship_order(self, args: Any, facts: PolicyFacts) -> PolicyDecision:
        order = self._order(facts)
        n = order.order_number
        issue = Issue(args.issue)
        if order.status is OrderStatus.CANCELLED:
            return self._reject("order.status", Reason.ORDER_CANCELLED, n)
        if not self.config.allows(issue, Remedy.RESHIP):
            return self._remedy_not_allowed(issue, Remedy.RESHIP)
        if facts.has_replacement or (
            issue is Issue.MISSING_DELIVERY and self._missing_delivery_refunded(order)
        ):
            return self._reject("missing_delivery.one_remedy", Reason.ALREADY_RESOLVED, n)
        items = [self._item(order, args.sku)] if args.sku else order.items

        if issue is Issue.MISSING_DELIVERY:
            check = self.assess_missing_delivery(order, items, facts.now)
        else:
            check = self._item_issue_reship(order, items, facts.now)
        if not check.approved:
            return check

        value = sum(i.line_total_paise for i in items)
        cap = self.config.reship.max_value_paise
        if value > cap:
            return self._escalate(
                self.config.escalate_when.over_reship_value_cap,
                "reship.max_value_inr",
                Reason.OVER_RESHIP_VALUE_CAP,
                n,
                amount=value,
                cap_paise=cap,
            )
        return self._approve("reship.max_value_inr", value, cap_paise=cap)

    def _item_issue_reship(
        self, order: Order, items: list[OrderItem], now: datetime
    ) -> PolicyDecision:
        """Replacing a damaged or wrong item: delivered, inside the window, not refunded."""
        n = order.order_number
        for item in items:
            shipment = shipment_for(order, item)
            if shipment is None:
                return self._reject("item.fulfilment", Reason.NOT_SHIPPED, n)
            if shipment.delivered_at is None:
                return self._reject("item.fulfilment", Reason.NOT_DELIVERED, n)
            refunded_items = {
                r.order_item_id
                for r in order.returns
                if any(f.return_id == r.id for f in order.refunds)
            }
            if item.id in refunded_items:
                return self._reject("refunds.balance", Reason.ALREADY_REFUNDED, n)
            deadline = self.return_deadline(shipment.delivered_at)
            if now > deadline:
                return self._reject(
                    "returns.window_days",
                    Reason.WINDOW_EXPIRED,
                    n,
                    deadline=deadline,
                )
        return self._approve("returns.window_days", None)

    def _create_ticket(self, args: Any, facts: PolicyFacts) -> PolicyDecision:
        if facts.open_ticket_reference:
            return self._reject(
                "tickets.no_duplicates",
                Reason.DUPLICATE_TICKET,
                facts.order.order_number if facts.order else "",
                ticket_reference=facts.open_ticket_reference,
            )
        return self._approve("tickets.allowed", None)

    def _escalate_to_human(self, args: Any, facts: PolicyFacts) -> PolicyDecision:
        return self._approve("escalation.always_allowed", None)

    # --- helpers --------------------------------------------------------------------------

    @staticmethod
    def _order(facts: PolicyFacts) -> Order:
        if facts.order is None:
            raise ValueError("this rule needs facts.order")
        return facts.order

    @staticmethod
    def _item(order: Order, sku: str) -> OrderItem:
        item = next((i for i in order.items if i.sku == sku), None)
        if item is None:
            raise ValueError(f"{sku} is not on {order.order_number}")  # callers check first
        return item

    @staticmethod
    def _missing_delivery_refunded(order: Order) -> bool:
        return any(
            r.reason is RefundReason.MISSING_DELIVERY
            and r.status in (RefundStatus.PENDING, RefundStatus.SUCCEEDED)
            for r in order.refunds
        )

    def _approve(self, rule: str, amount: int | None, **facts: Any) -> PolicyDecision:
        return PolicyDecision(Verdict.APPROVE, rule, None, "Approved.", amount, facts)

    def _reject(self, rule: str, reason: Reason, n: str, **facts: Any) -> PolicyDecision:
        return self._decision(Verdict.REJECT, rule, reason, n, facts)

    def _escalate(
        self, escalate: bool, rule: str, reason: Reason, n: str, **facts: Any
    ) -> PolicyDecision:
        verdict = Verdict.REQUIRE_HUMAN if escalate else Verdict.REJECT
        return self._decision(verdict, rule, reason, n, facts)

    def _remedy_not_allowed(self, issue: Issue, remedy: Remedy) -> PolicyDecision:
        allowed = self.config.remedies[issue]
        offer = " or ".join(_REMEDY_TEXT[r] for r in allowed) or "nothing automatically"
        message = f"For {ISSUE_TEXT[issue]}, I can offer {offer}, but not {_REMEDY_TEXT[remedy]}."
        return PolicyDecision(
            Verdict.REJECT,
            f"remedies.{issue.value}",
            Reason.REMEDY_NOT_ALLOWED,
            message,
            None,
            {"issue": issue.value, "remedy": remedy.value, "allowed": [r.value for r in allowed]},
        )

    def _decision(
        self, verdict: Verdict, rule: str, reason: Reason, n: str, facts: dict[str, Any]
    ) -> PolicyDecision:
        human = verdict is Verdict.REQUIRE_HUMAN
        message = _message(reason, n, human=human, **facts)
        amount = facts.get("amount")
        safe = {k: _iso(v) if isinstance(v, datetime) else v for k, v in facts.items()}
        return PolicyDecision(verdict, rule, reason, message, amount, safe)


@lru_cache
def get_engine() -> PolicyEngine:
    return PolicyEngine(get_policy())


def _message(reason: Reason, n: str, *, human: bool, **f: Any) -> str:
    amount = format_inr(f["amount"]) if f.get("amount") is not None else None
    match reason:
        case Reason.ORDER_CANCELLED:
            return (
                f"Order {n} was cancelled, so it can't be returned, refunded again or replaced. "
                "Refunds for cancellations are processed automatically."
            )
        case Reason.NOT_SHIPPED:
            return f"That item from order {n} hasn't shipped yet."
        case Reason.NOT_DELIVERED:
            return f"That item from order {n} hasn't been delivered yet."
        case Reason.WINDOW_EXPIRED:
            return f"The return window for order {n} closed on {format_date(f['deadline'])}."
        case Reason.ALREADY_RETURNED:
            return f"A return for that item on order {n} is already in progress or complete."
        case Reason.ALREADY_REFUNDED:
            return f"That has already been refunded on order {n}."
        case Reason.ALREADY_RESOLVED:
            return f"Order {n} already has a replacement or refund for this problem."
        case Reason.NO_RETURN_ON_FILE:
            return f"There's no return on file for that item on order {n} yet."
        case Reason.NOT_YET_OVERDUE:
            by = f" It's expected by {format_date(f['eta'])}." if f.get("eta") else ""
            return f"Order {n} is still on its way and isn't late yet.{by}"
        case Reason.DELIVERED_PER_CARRIER:
            if human:
                return (
                    f"The carrier shows order {n} as delivered, "
                    "so a specialist needs to investigate."
                )
            return f"The carrier confirms order {n} was delivered, so I can't treat it as missing."
        case Reason.RETURNED_TO_SENDER:
            if human:
                return f"The parcel for order {n} is on its way back to us; a specialist will help."
            return f"The parcel for order {n} is on its way back to us, so I can't act on it yet."
        case Reason.GOODWILL_REQUESTED:
            if human:
                return "Goodwill refunds need a specialist's approval."
            return "I'm not able to offer goodwill refunds."
        case Reason.OVER_ORDER_REFUND_CAP:
            if human:
                return (
                    f"A refund of {amount} is more than I can approve myself, "
                    "so a specialist will review it."
                )
            return f"A refund of {amount} is more than we can refund automatically on order {n}."
        case Reason.OVER_CUSTOMER_REFUND_CAP:
            if human:
                return (
                    f"A refund of {amount} would go over what I can approve for your account "
                    "this month, so a specialist will review it."
                )
            return (
                f"A refund of {amount} would go over the automatic refund limit for your account."
            )
        case Reason.OVER_RESHIP_VALUE_CAP:
            if human:
                return f"A replacement worth {amount} needs a specialist's approval."
            return f"A replacement worth {amount} is more than we can send automatically."
        case Reason.DUPLICATE_TICKET:
            return (
                f"There's already an open ticket, {f['ticket_reference']}, for order {n}. "
                "Our team is working on it."
            )
    return f"That isn't possible for order {n}."
