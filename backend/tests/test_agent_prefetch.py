"""Deterministic verify/get_order before the model: extraction, and the loop with it on."""

import json

import pytest

from app.agent.prefetch import Found, extract, only_identification
from app.config import POLICY_DOCS_DIR
from app.retrieval.chunking import policy_documents_text
from tests.agent_helpers import ScriptedLLM, call, make_agent, of_type, run_turn, say, tokens

MEERA = "meera.pillai@example.com"
MEERA_PIN = "302647"  # from the seed data

# --- extraction ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "found"),
    [
        (
            f"Hi, I'm {MEERA} and my PIN is {MEERA_PIN}. Where is order VX-8015?",
            Found(email=MEERA, pin=MEERA_PIN, order_number="VX-8015"),
        ),
        (  # as a voice transcript arrives
            "my email is meera dot pillai at example dot com, PIN three zero two six four seven",
            Found(email=MEERA, pin=MEERA_PIN),
        ),
        (  # a written ". " after the address isn't part of it
            f"{MEERA}. My PIN is {MEERA_PIN}.",
            Found(email=MEERA, pin=MEERA_PIN),
        ),
        ("what about order 1580?", Found(order_number="VX-1580")),
        ("vx 1508 please", Found(order_number="VX-1508")),
        ("is it VX-1508 or VX-1580?", Found()),  # two orders: the model asks which
        (
            "phone +91 79731 88087, pin double three four five six seven",
            Found(phone="7973188087", pin="334567"),
        ),
        ("PIN 30264", Found()),  # five digits is not a PIN
        # As speech-to-text wrote them in a live call:
        ("652432-5150.", Found(phone="6524325150")),
        ("VX441-0.", Found(order_number="VX-4410")),
        ("VX 44105", Found()),  # five digits is not an order number
        ("I'm at home and it never arrived", Found()),
    ],
)
def test_extract(text, found):
    assert extract(text) == found


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"hi, {MEERA} / {MEERA_PIN} thanks", True),
        (f"Hello, I'm Meera. My email is {MEERA} and my PIN code is {MEERA_PIN}", True),
        (f"{MEERA} {MEERA_PIN}, my speaker never arrived", False),
        (f"{MEERA} {MEERA_PIN} order VX-8015", False),
    ],
)
def test_only_identification(text, expected):
    assert only_identification(text, extract(text), "Meera Pillai") is expected


# --- in the loop -----------------------------------------------------------------------------


def _agent(tx_db, llm, *, v3: bool = True):
    documents = policy_documents_text(POLICY_DOCS_DIR) if v3 else None
    return make_agent(tx_db, llm, policy_documents=documents, prefetch=True)


def _tool_results(messages):
    return [json.loads(m["content"]) for m in messages if m["role"] == "tool"]


async def _tool_log(agent, session_id):
    session = await agent.store.load(session_id)
    return [json.loads(m["content"]) for m in session.history[-1] if m["role"] == "tool"], session


@pytest.mark.anyio
@pytest.mark.db
async def test_details_and_an_order_take_one_model_call(tx_db):
    llm = ScriptedLLM(say("Your speaker, order VX-8015, is still in transit."))
    agent = _agent(tx_db, llm)
    session = await agent.start_session()
    text = f"Hi, I'm {MEERA}, PIN {MEERA_PIN}. Where is order VX-8015?"
    events = await run_turn(agent, session.session_id, text)

    assert len(llm.calls) == 1
    verify, order = _tool_results(llm.last_messages())
    assert verify == {"ok": True, "data": {"verified": True, "first_name": "Meera"}}
    assert order["data"]["order_number"] == "VX-8015"
    # The one call already gets the verified toolset.
    assert "verify_customer" not in {t["function"]["name"] for t in llm.tools}
    assert [(e.name, e.status) for e in of_type(events, "tool")] == [
        ("verify_customer", "started"), ("verify_customer", "finished"),
        ("get_order", "started"), ("get_order", "finished"),
    ]  # fmt: skip
    saved = await agent.store.load(session.session_id)
    assert saved.state.is_verified and saved.current_order_number == "VX-8015"
    done = of_type(events, "done")[0]
    assert done.usage["llm_calls"] == 1


@pytest.mark.anyio
@pytest.mark.db
async def test_prefetched_calls_are_logged_as_calls_from_prefetch(tx_db):
    from sqlalchemy import select

    from app.db.models import ConversationTurn
    from app.domain.enums import TurnRole

    llm = ScriptedLLM(say("It's on its way."))
    agent = _agent(tx_db, llm)
    session = await agent.start_session()
    await run_turn(agent, session.session_id, f"{MEERA} {MEERA_PIN} order VX-8015")
    rows = await tx_db.scalars(
        select(ConversationTurn.content).where(
            ConversationTurn.conversation_id == session.conversation_id,
            ConversationTurn.role == TurnRole.TOOL,
        )
    )
    log = [json.loads(r) for r in rows]
    assert [(e["tool"], e["phase"], e["ok"], e["source"]) for e in log] == [
        ("verify_customer", "call", True, "prefetch"),
        ("get_order", "call", True, "prefetch"),
    ]


@pytest.mark.anyio
@pytest.mark.db
async def test_bare_details_are_answered_without_the_model(tx_db):
    llm = ScriptedLLM()  # any model call fails the test
    agent = _agent(tx_db, llm)
    session = await agent.start_session()
    events = await run_turn(agent, session.session_id, f"Hi, my email is {MEERA}, PIN {MEERA_PIN}")

    assert llm.calls == []
    assert tokens(events) == "Thanks, Meera, you're verified. How can I help you today?"
    done = of_type(events, "done")[0]
    assert (done.outcome, done.usage["llm_calls"]) == ("completed", 0)
    # The next model call sees a normal exchange: the tool call, its result, the reply.
    saved = await agent.store.load(session.session_id)
    assert [m["role"] for m in saved.history[-1]] == ["user", "assistant", "tool", "assistant"]


@pytest.mark.anyio
@pytest.mark.db
async def test_without_the_template_the_model_still_answers(tx_db):
    """Prompt sets without verified_ready (v2) keep the model's own reply."""
    llm = ScriptedLLM(say("Thanks Meera, what can I do for you?"))
    agent = _agent(tx_db, llm, v3=False)
    session = await agent.start_session()
    await run_turn(agent, session.session_id, f"{MEERA} {MEERA_PIN}")
    assert len(llm.calls) == 1 and len(_tool_results(llm.last_messages())) == 1


@pytest.mark.anyio
@pytest.mark.db
async def test_wrong_pin_goes_to_the_model_and_counts_as_an_attempt(tx_db):
    llm = ScriptedLLM(say("Those details don't match. Could you check them?"))
    agent = _agent(tx_db, llm)
    session = await agent.start_session()
    await run_turn(agent, session.session_id, f"{MEERA}, PIN 111111, where's VX-8015?")
    (verify,) = _tool_results(llm.last_messages())  # no order lookup while unverified
    assert verify["error"]["code"] == "verification_failed"
    saved = await agent.store.load(session.session_id)
    assert not saved.state.is_verified
    assert saved.state.failed_verification_attempts == 1


@pytest.mark.anyio
@pytest.mark.db
async def test_details_split_across_messages(tx_db):
    llm = ScriptedLLM(say("Thanks. And your PIN code?"))
    agent = _agent(tx_db, llm)
    session = await agent.start_session()
    await run_turn(agent, session.session_id, f"my email is {MEERA}")
    assert _tool_results(llm.last_messages()) == []  # an email alone: nothing to check yet
    events = await run_turn(agent, session.session_id, f"it's {MEERA_PIN}")
    assert tokens(events).startswith("Thanks, Meera, you're verified.")
    assert len(llm.calls) == 1


@pytest.mark.anyio
@pytest.mark.db
async def test_another_customers_order_is_refused_like_a_model_call(tx_db):
    llm = ScriptedLLM(say("I can't find that order on your account."))
    agent = _agent(tx_db, llm)
    session = await agent.start_session()
    # VX-1508 is Rahul's.
    await run_turn(agent, session.session_id, f"{MEERA} {MEERA_PIN}, order VX-1508")
    verify, order = _tool_results(llm.last_messages())
    assert verify["ok"] and "data" not in order
    assert "Rahul" not in json.dumps(order)


@pytest.mark.anyio
@pytest.mark.db
async def test_no_order_lookup_before_verification(tx_db):
    llm = ScriptedLLM(say("I'll need to verify you first."))
    agent = _agent(tx_db, llm)
    session = await agent.start_session()
    events = await run_turn(agent, session.session_id, "where is order VX-8015?")
    assert _tool_results(llm.last_messages()) == []
    assert of_type(events, "tool") == []


@pytest.mark.anyio
@pytest.mark.db
async def test_nothing_is_prefetched_while_a_proposal_awaits_an_answer(tx_db):
    llm = ScriptedLLM(
        say("Thanks, what can I do?"),
        call("issue_refund", order_number="VX-8015", reason="missing_delivery"),
        say("I'll refund ₹2,499 for VX-8015. Shall I go ahead?"),
        say("Which order do you mean?"),
    )
    agent = _agent(tx_db, llm)
    session = await agent.start_session()
    await run_turn(agent, session.session_id, f"{MEERA} {MEERA_PIN}, my speaker never came")
    await run_turn(agent, session.session_id, "refund VX-8015 please")
    assert (await agent.store.load(session.session_id)).state.actions  # awaiting a yes
    calls_before = len(llm.calls)
    await run_turn(agent, session.session_id, "hmm, what about order VX-1508?")
    assert len(llm.calls) == calls_before + 1
    last_turn = llm.last_messages()[-1]
    assert last_turn["role"] == "user"  # no prefetched tool exchange after the message


# --- a clear yes or no is answered without the model ----------------------------------------


async def _proposed(tx_db, llm):
    agent = _agent(tx_db, llm)
    session = await agent.start_session()
    await run_turn(agent, session.session_id, f"{MEERA} {MEERA_PIN}")
    await run_turn(agent, session.session_id, "my speaker VX-8015 never arrived, refund it")
    assert (await agent.store.load(session.session_id)).state.actions  # awaiting a yes
    return agent, session


@pytest.mark.anyio
@pytest.mark.db
async def test_yes_is_answered_from_the_result(tx_db):
    llm = ScriptedLLM(
        call("issue_refund", order_number="VX-8015", reason="missing_delivery"),
        say("I can refund ₹3,499 for VX-8015. Shall I go ahead?"),
    )
    agent, session = await _proposed(tx_db, llm)
    calls = len(llm.calls)
    events = await run_turn(agent, session.session_id, "yes")
    assert len(llm.calls) == calls  # no model call: the outcome is the whole answer
    reply = tokens(events)
    assert reply.startswith("Done. Your refund of ₹3,499 for order VX-8015 is on its way")
    assert "RF-" in reply and "5 to 7 business days" in reply
    assert reply.endswith("Is there anything else I can help with?")
    done = of_type(events, "done")[0]
    assert (done.outcome, done.usage["llm_calls"]) == ("completed", 0)


@pytest.mark.anyio
@pytest.mark.db
async def test_no_is_answered_without_the_model(tx_db):
    llm = ScriptedLLM(
        call("issue_refund", order_number="VX-8015", reason="missing_delivery"),
        say("I can refund ₹3,499 for VX-8015. Shall I go ahead?"),
    )
    agent, session = await _proposed(tx_db, llm)
    calls = len(llm.calls)
    events = await run_turn(agent, session.session_id, "no thanks")
    assert len(llm.calls) == calls
    assert tokens(events).startswith("Okay, I won't refund ₹3,499")
