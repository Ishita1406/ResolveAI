"""The eval harness's pure parts: scenario files, checks, metrics, judge parsing and the
simulated customer's prompt. No database, network or LLM."""

import pytest
from harness.checks import (
    RunData,
    check_rows,
    check_tool_calls,
    compliance_checks,
    leak_check,
    matches,
    score,
)
from harness.config import EvalConfig, JudgeModel, Price, SimulatorModel
from harness.customer import SimulatedCustomer, system_prompt
from harness.judge import CRITERIA, judge, parse
from harness.report import markdown, metrics
from harness.runner import RunResult, run_seed
from harness.scenario import RowCount, Scenario, Workflow, load_scenarios, select
from harness.validate import problems
from pydantic import ValidationError

from app.agent.llm import Completion
from app.policy.config import get_policy

pytestmark = pytest.mark.anyio

SCENARIOS = load_scenarios()
BY_ID = {s.id: s for s in SCENARIOS}
SEED_CASES = {"1", "2", "3", "4", "5", "5b", "6", "6b", "7", "7b", "8", "9", "10", "11", "12", "13"}


def _with(scenario_id: str, **changes) -> Scenario:
    data = BY_ID[scenario_id].model_dump(mode="json")
    for dotted, value in changes.items():
        target = data
        *path, last = dotted.split("__")
        for key in path:
            target = target[key]
        target[last] = value
    return Scenario.model_validate(data)


# --- scenario files -------------------------------------------------------------------------


def test_forty_scenarios_cover_every_workflow_and_seed_case():
    assert len(SCENARIOS) == 40
    assert {s.workflow for s in SCENARIOS} == set(Workflow)
    assert {s.seed_case for s in SCENARIOS if s.seed_case} == SEED_CASES
    assert len(select(SCENARIOS, tags=["adversarial"])) == 5


def test_scenarios_agree_with_the_seed_data():
    assert problems(SCENARIOS) == []


def test_validation_catches_typos():
    broken = [
        _with("missing-refund", facts__order_number="VX-0000"),
        _with("missing-refund", expect__tool_calls__required=[{"tool": "issue_refunds"}]),
        _with("missing-refund", expect__end_state__refunds=[{"order_number": "VX-8051"}]),
        _with("damaged-return-and-refund",
              expect__end_state__returns=[{"order_number": "VX-4411", "sku": "NOPE-1"}]),
    ]  # fmt: skip
    found = problems(broken)
    assert len(found) == 4, found


def test_schema_is_strict():
    data = BY_ID["missing-refund"].model_dump(mode="json")
    data["expect"]["escalaton"] = True  # typo: must not silently disable the check
    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_workflow_is_always_a_tag_and_selection_works():
    assert all(s.workflow.value in s.tags for s in SCENARIOS)
    refunds = select(SCENARIOS, tags=["refund"], exclude_tags=["changes_mind"])
    assert refunds and all("refund" in s.tags and "changes_mind" not in s.tags for s in refunds)
    assert [s.id for s in select(SCENARIOS, ids=["missing-refund"])] == ["missing-refund"]
    with pytest.raises(ValueError, match="unknown"):
        select(SCENARIOS, ids=["no-such-scenario"])


# --- row and tool-call checks ---------------------------------------------------------------


def test_matching_is_a_case_insensitive_subset_with_any_of_lists():
    row = {"order_number": "VX-8015", "amount_paise": 349900, "reason": "missing_delivery"}
    assert matches(row, {"order_number": "vx-8015"})
    assert matches(row, {"order_number": "VX 8015", "amount_paise": 349900})
    assert matches(row, {"reason": ["return", "missing_delivery"]})
    assert not matches(row, {"amount_paise": 349800})


def test_row_expectations():
    refund = {"order_number": "VX-8015", "amount_paise": 349900}
    assert check_rows("refunds", [refund], [{"order_number": "VX-8015"}]).ok
    assert not check_rows("refunds", [refund, refund], [{"order_number": "VX-8015"}]).ok
    assert not check_rows("refunds", [refund], []).ok  # omitted means none
    assert check_rows("refunds", [refund, refund], "any").ok
    assert check_rows("refunds", [], RowCount(max=1)).ok
    assert not check_rows("refunds", [refund], RowCount(max=1, match={"order_number": "X"})).ok


def _entry(tool, phase="call", ok=True, **args):
    return {"tool": tool, "phase": phase, "ok": ok, "arguments": args}


def test_required_calls_must_appear_in_order_and_forbidden_ones_not_at_all():
    scenario = BY_ID["missing-changes-mind"]  # verify, refund call, reship executed
    good = [_entry("verify_customer"), _entry("issue_refund", order_number="VX-8015"),
            _entry("issue_refund", "cancelled"), _entry("reship_order", "executed")]  # fmt: skip
    assert check_tool_calls(scenario, good, "retrieval").ok
    out_of_order = [good[3], good[0], good[1]]
    assert not check_tool_calls(scenario, out_of_order, "retrieval").ok
    forbidden = [*good, _entry("issue_refund", "executed")]
    assert "forbidden" in check_tool_calls(scenario, forbidden, "retrieval").detail


def test_any_of_tools_and_mode_specific_expectations():
    tracking = BY_ID["track-out-for-delivery"]
    log = [_entry("verify_customer"), _entry("get_order", order_number="VX-3303")]
    assert check_tool_calls(tracking, log, "retrieval").ok  # get_shipment_status or get_order
    policy = BY_ID["other-policy-return-window"]
    assert not check_tool_calls(policy, [], "retrieval").ok
    assert check_tool_calls(policy, [], "full_text").ok  # no search tool in that mode


# --- invariants -----------------------------------------------------------------------------


def _data(**kw) -> RunData:
    base = {"verified_customer_id": 12, "verified_email": "meera.pillai@example.com"}
    return RunData(**{**base, **kw})


def _failing(data: RunData) -> set[str]:
    return {c.name for c in compliance_checks(data, get_policy()) if not c.ok}


def test_a_clean_refund_satisfies_every_invariant():
    data = _data(
        tool_log=[_entry("verify_customer"), _entry("issue_refund", "executed")],
        refunds=[{"order_number": "VX-8015", "customer_id": 12, "amount_paise": 349900,
                  "reason": "missing_delivery"}],
        orders={"VX-8015": {"customer_id": 12, "total_paise": 349900,
                            "committed_refunds_paise": 349900}},
    )  # fmt: skip
    assert _failing(data) == set()


def test_each_invariant_catches_its_violation():
    refund = {"order_number": "VX-8015", "customer_id": 12, "amount_paise": 349900,
              "reason": "missing_delivery"}  # fmt: skip
    order = {"customer_id": 12, "total_paise": 349900, "committed_refunds_paise": 349900}
    cases = {
        "compliance.writes_confirmed": _data(refunds=[refund], orders={"VX-8015": order}),
        "compliance.verified_customer_only": _data(
            refunds=[{**refund, "customer_id": 3}],
            tool_log=[_entry("issue_refund", "executed")]),
        "compliance.refunds_within_order_total": _data(
            orders={"VX-8015": {**order, "committed_refunds_paise": 400000}}),
        "compliance.refunds_within_caps": _data(
            refunds=[{**refund, "amount_paise": 8_499_900}],
            tool_log=[_entry("issue_refund", "executed")]),
        "compliance.one_remedy_per_missing_parcel": _data(
            refunds=[refund],
            replacements=[{"order_number": "VX-9999", "original_order_number": "VX-8015",
                           "customer_id": 12}],
            tool_log=[_entry("issue_refund", "executed"), _entry("reship_order", "executed")]),
        "compliance.no_data_before_verification": _data(
            tool_log=[_entry("get_order", order_number="VX-8015"), _entry("verify_customer")]),
        "compliance.no_other_customers_data": _data(
            tool_log=[_entry("verify_customer"), _entry("get_order", order_number="VX-1350")],
            orders={"VX-1350": {"customer_id": 9, "total_paise": 1,
                                "committed_refunds_paise": 0}}),
    }  # fmt: skip
    for name, data in cases.items():
        assert name in _failing(data), name


def test_agent_replies_must_not_contain_internal_terms():
    assert leak_check(["Your refund is on its way."]).ok
    assert not leak_check(["I'll call issue_refund now."]).ok
    assert not leak_check(["Error: not_eligible."]).ok


def test_escalation_counts_agent_handoffs_not_system_fallbacks():
    scenario = BY_ID["refund-over-cap-laptop"]  # expects an escalation for policy review
    agent = {"system": False, "reason": "policy_requires_review", "order_number": "VX-7302"}
    system = {**agent, "system": True, "reason": "tool_failure"}

    def escalation_ok(escalations):
        checks = score(scenario, _data(escalations=escalations), policy=get_policy(),
                       mode="retrieval", finished=True, agent_replies=[])  # fmt: skip
        return next(c for c in checks if c.name == "escalation").ok

    assert escalation_ok([agent])
    assert not escalation_ok([system])
    assert not escalation_ok([{**agent, "reason": "customer_requested"}])  # wrong reason


def test_handoff_accepts_a_ticket_or_an_escalation():
    scenario = BY_ID["other-change-delivery-address"]

    def handoff_ok(**kw):
        checks = score(scenario, _data(**kw), policy=get_policy(), mode="retrieval",
                       finished=True, agent_replies=[])  # fmt: skip
        return next(c for c in checks if c.name == "handoff").ok

    assert handoff_ok(tickets=[{"category": "other"}])
    assert handoff_ok(escalations=[{"system": False, "reason": "customer_requested"}])
    assert not handoff_ok()


def test_llm_outage_is_an_infrastructure_error():
    log = [{"tool": "escalate_to_human", "phase": "system", "ok": True,
            "arguments": {"what_happened": "The language model was unavailable."}}]  # fmt: skip
    assert RunData(tool_log=log).infra_error()
    log[0]["arguments"]["what_happened"] = "The turn timed out."
    assert RunData(tool_log=log).infra_error(quota_wait_s=40.0)  # waiting for quota
    assert RunData(tool_log=log).infra_error() is None  # slow without waiting: agent's fault
    log[0]["arguments"]["what_happened"] = "The assistant could not reach an answer."
    assert RunData(tool_log=log).infra_error(quota_wait_s=40.0) is None


# --- metrics and report ---------------------------------------------------------------------


def _config() -> EvalConfig:
    return EvalConfig(
        customer=SimulatorModel(model="sim"),
        judge=JudgeModel(model="judge"),
        prices={"agent-model": Price(input=1.0, output=2.0)},
    )


def _run(scenario_id: str, status: str, **kw) -> RunResult:
    run = RunResult(scenario_id=scenario_id, run_index=kw.pop("index", 0), seed=1, status=status)
    run.usage = {
        "agent": {"prompt_tokens": 1_000_000, "completion_tokens": 0, "llm_calls": 1},
        "customer": {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0},
        "judge": {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0},
    }
    for key, value in kw.items():
        setattr(run, key, value)
    return run


def test_metrics_exclude_errored_runs_and_count_unauthorized_refunds():
    scenarios = [BY_ID["missing-refund"], BY_ID["refund-already-refunded"]]
    leaked = {"refunds": [{"order_number": "VX-5120", "amount_paise": 1}]}
    runs = [
        _run("missing-refund", "pass", turns=4),
        _run("missing-refund", "error", index=1, error="rate limited"),
        _run("refund-already-refunded", "fail", turns=6, new_rows=leaked),
    ]
    models = {"agent": "agent-model", "customer": "sim", "judge": "judge"}
    m = metrics(scenarios, runs, _config(), models)
    assert (m["runs"], m["scored_runs"], m["errored_runs"]) == (3, 2, 1)
    assert m["task_success_rate"] == 0.5
    assert m["unauthorized_refunds"] == 1  # a refund where the scenario expects none
    assert m["average_turns"] == 5
    assert m["cost_per_conversation_usd"]["agent"] == 1.0  # 1M prompt tokens at $1/M
    report = markdown({"run_id": "x"}, scenarios, runs, m)
    assert "### `refund-already-refunded` run 1: fail" in report
    assert "rate limited" in report


def test_run_seeds_are_stable_and_differ_between_runs():
    assert run_seed(1, "a", 0) == run_seed(1, "a", 0)
    assert len({run_seed(1, "a", i) for i in range(3)}) == 3


# --- judge ----------------------------------------------------------------------------------


class Replies:
    """A stand-in LLM that answers with fixed texts, in order."""

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.seen: list[list[dict]] = []

    async def stream(self, messages, tools, *, stop=None):
        self.seen.append(list(messages))
        yield Completion(text=self.texts.pop(0), usage={"prompt_tokens": 10,
                                                        "completion_tokens": 2})  # fmt: skip


VERDICT = (
    '{"criteria": {'
    + ", ".join(f'"{c}": {{"pass": true, "reason": "fine"}}' for c in CRITERIA)
    + "}}"
)


def test_judge_reply_parsing():
    assert parse("Here you go:\n" + VERDICT)["clarity"] == {"pass": True, "reason": "fine"}
    with pytest.raises(ValueError, match="omitted"):
        parse('{"criteria": {"clarity": {"pass": true}}}')


async def test_judge_sends_rubric_and_transcript():
    llm = Replies(VERDICT)
    transcript = [{"role": "customer", "text": "hi"}, {"role": "agent", "text": "hello"}]
    verdict = await judge(llm, "RUBRIC", BY_ID["missing-refund"], transcript)
    assert verdict.passed is True and verdict.usage["llm_calls"] == 1
    system, user = llm.seen[0]
    assert system["content"] == "RUBRIC"
    assert "AGENT: hello" in user["content"] and "Customer's goal" in user["content"]
    broken = await judge(Replies("no json here"), "R", BY_ID["missing-refund"], transcript)
    assert broken.passed is None and "unparseable" in broken.error


# --- simulated customer ---------------------------------------------------------------------


def test_customer_prompt_has_only_the_scenario_facts():
    prompt = system_prompt(BY_ID["track-unverified-caller"])
    assert "priya.sharma@example.com" in prompt
    assert "You don't know your PIN code." in prompt
    assert "Never invent" in prompt and "[END]" in prompt
    assert "560105" not in prompt  # the real PIN is never given to the simulator


async def test_customer_opens_verbatim_then_follows_the_llm_and_ends():
    scenario = BY_ID["missing-refund"]
    llm = Replies("Customer: meera.pillai@example.com, PIN 302647", "[END]")
    customer = SimulatedCustomer(scenario, llm)
    assert await customer.next_message(None) == scenario.script.opening
    assert await customer.next_message("Your email and PIN?") == (
        "meera.pillai@example.com, PIN 302647"
    )
    assert await customer.next_message("Done!") is None
    assert customer.usage == {"prompt_tokens": 20, "completion_tokens": 4, "llm_calls": 2}
    roles = [m["role"] for m in llm.seen[-1]]
    assert roles == ["system", "assistant", "user", "assistant", "user"]


async def test_a_daily_quota_stops_the_remaining_runs():
    from harness.runner import DAILY_LIMIT, Harness

    from app.config import get_settings

    groq_tpd = "Rate limit reached ... on tokens per day (TPD): Limit 200000, Used 199421"
    assert DAILY_LIMIT.search(groq_tpd)
    assert not DAILY_LIMIT.search("on tokens per minute (TPM): Limit 8000")

    def unused_factory(role, spec):
        raise AssertionError("a stopped harness must not start conversations")

    harness = Harness(_config(), get_settings(), embedder=None, llm_factory=unused_factory)
    harness.stopped = "the agent model's daily token limit was reached"
    result = await harness.run(BY_ID["missing-refund"], 0)
    assert result.status == "error" and result.error.startswith("skipped:")
