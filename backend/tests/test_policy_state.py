from datetime import timedelta

import pytest

from app.policy.config import get_policy
from app.policy.state import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    ActionState,
    InvalidTransitionError,
    PendingAction,
    args_hash,
)
from app.tools.write_tools import CreateTicketInput
from tests.policy_helpers import NOW

CONFIG = get_policy().confirmation
S = ActionState


def _action(state=S.PROPOSED, turn=5) -> PendingAction:
    return PendingAction(id="a1", tool="create_ticket", arguments={}, args_hash="h",
                         idempotency_key="act-a1", proposed_at=NOW, proposed_turn=turn,
                         summary="s", amount_paise=None, verdict="approve", rule="r",
                         state=state)  # fmt: skip


def test_happy_path_records_history():
    a = _action()
    for i, target in enumerate((S.AWAITING_CONFIRMATION, S.CONFIRMED, S.EXECUTED)):
        a.transition(target, at=NOW, turn=5 + i)
    assert a.state is S.EXECUTED
    assert [(t.source, t.target) for t in a.history] == [
        (S.PROPOSED, S.AWAITING_CONFIRMATION),
        (S.AWAITING_CONFIRMATION, S.CONFIRMED),
        (S.CONFIRMED, S.EXECUTED),
    ]


ALL_PAIRS = [(src, dst) for src in S for dst in S]


@pytest.mark.parametrize(("source", "target"), ALL_PAIRS)
def test_only_listed_transitions_are_allowed(source, target):
    a = _action(state=source)
    if target in ALLOWED_TRANSITIONS.get(source, ()):
        a.transition(target, at=NOW, turn=5)
        assert a.state is target
    else:
        with pytest.raises(InvalidTransitionError):
            a.transition(target, at=NOW, turn=5)
        assert a.state is source


def test_write_can_only_follow_confirmation():
    sources = {src for src, targets in ALLOWED_TRANSITIONS.items() if S.EXECUTED in targets}
    assert sources == {S.CONFIRMED}
    assert TERMINAL_STATES == {S.EXECUTED, S.CANCELLED, S.EXPIRED, S.REJECTED}


def test_expiry_by_turns_boundary():
    a = _action(state=S.AWAITING_CONFIRMATION, turn=5)
    n = CONFIG.expire_after_turns
    assert not a.is_expired(now=NOW, turn=5 + n, config=CONFIG)
    assert a.is_expired(now=NOW, turn=5 + n + 1, config=CONFIG)


def test_expiry_by_time_boundary():
    a = _action(state=S.AWAITING_CONFIRMATION)
    limit = timedelta(seconds=CONFIG.expire_after_seconds)
    assert not a.is_expired(now=NOW + limit, turn=5, config=CONFIG)
    assert a.is_expired(now=NOW + limit + timedelta(microseconds=1), turn=5, config=CONFIG)


@pytest.mark.parametrize("state", [s for s in S if s is not S.AWAITING_CONFIRMATION])
def test_only_awaiting_actions_expire(state):
    a = _action(state=state)
    assert not a.is_expired(now=NOW + timedelta(days=1), turn=99, config=CONFIG)


def test_args_hash_ignores_key_but_not_arguments():
    a = CreateTicketInput(category="other", summary="Same request.", idempotency_key="key-00001")
    b = CreateTicketInput(category="other", summary="Same request.", idempotency_key="key-00002")
    c = CreateTicketInput(category="other", summary="Different one.", idempotency_key="key-00001")
    assert args_hash(a) == args_hash(b) != args_hash(c)
