"""The harness end to end, with scripted LLMs: a real template database, a fresh copy per
run, the real app on a real port, WebSocket traffic, scoring and the report.

Two runs of the same refund scenario run concurrently and both pass: if they shared a
database, the second refund of VX-8015 would be refused as already resolved.
"""

import pytest
from harness.config import EvalConfig, JudgeModel, Price, SimulatorModel
from harness.judge import CRITERIA
from harness.report import markdown, metrics
from harness.runner import Harness, ModelSpec
from harness.scenario import load_scenarios
from redis.asyncio import Redis
from sqlalchemy import create_engine, text

from app.agent.llm import Completion
from tests.agent_helpers import ScriptedLLM, call, say
from tests.retrieval_helpers import HashingEmbedder

pytestmark = [pytest.mark.anyio, pytest.mark.db, pytest.mark.redis]

SCENARIOS = {s.id: s for s in load_scenarios()}
VERDICT = (
    '{"criteria": {'
    + ", ".join(f'"{c}": {{"pass": true, "reason": "ok"}}' for c in CRITERIA)
    + "}}"
)


class Replies:
    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)

    async def stream(self, messages, tools, *, stop=None):
        yield Completion(text=self.texts.pop(0), usage={"prompt_tokens": 50,
                                                        "completion_tokens": 5})  # fmt: skip


def refund_agent() -> ScriptedLLM:
    """What a good agent does for missing-refund, one model call per step."""
    return ScriptedLLM(
        say("I can help. What's your email and the PIN code of your address?"),
        call("verify_customer", email="meera.pillai@example.com", postal_code="302647"),
        call("issue_refund", order_number="VX-8015", reason="missing_delivery"),
        say("I can refund ₹3,499 to your original payment method for order VX-8015. "
            "Shall I go ahead?"),
        say("Done. Your refund is on its way and should arrive in 5 to 7 business days."),
    )  # fmt: skip


def replacement_customer() -> Replies:
    return Replies("meera.pillai@example.com, PIN 302647", "yes", "[END]")


@pytest.fixture
async def redis_up(test_db_settings):
    client = Redis.from_url(test_db_settings.redis_url)
    try:
        await client.ping()
    except Exception as exc:
        pytest.fail(f"Redis not reachable at REDIS_URL ({type(exc).__name__})")
    finally:
        await client.aclose()


def _config() -> EvalConfig:
    return EvalConfig(
        runs_per_scenario=2,
        concurrency=2,
        customer=SimulatorModel(model="sim"),
        judge=JudgeModel(model="judge"),
        prices={"agent-model": Price(input=0.15, output=0.60)},
    )


async def test_runs_are_isolated_scored_and_reported(test_db_settings, redis_up):
    settings = test_db_settings.model_copy(update={"groq_model": "agent-model"})
    script = {"missing-refund": refund_agent, "missing-replacement": refund_agent}

    def factory(role: str, spec: ModelSpec):
        assert spec.seed  # every client gets the run's seed
        if role == "customer":
            return replacement_customer()
        if role == "judge":
            return Replies(VERDICT)
        return script[current.pop(0)]()

    scenarios = [SCENARIOS["missing-refund"], SCENARIOS["missing-replacement"]]
    current = ["missing-refund", "missing-refund", "missing-replacement"]
    harness = Harness(_config(), settings, embedder=HashingEmbedder(), llm_factory=factory)
    await harness.prepare()
    try:
        both = await harness.run_all([scenarios[0]], runs=2)
        wrong = await harness.run(scenarios[1], 0)  # the agent refunds; a replacement was wanted
    finally:
        await harness.close()

    for run in both:
        assert run.status == "pass", [(c.name, c.detail) for c in run.checks if not c.ok]
        assert run.finished and run.turns == 3
        assert [r["amount_paise"] for r in run.new_rows["refunds"]] == [349900]
        assert run.usage["agent"]["llm_calls"] == 5
        assert run.usage["customer"]["llm_calls"] == 3
        assert run.judge.passed is True
        assert any(e["tool"] == "issue_refund" and e["phase"] == "executed" for e in run.tool_log)

    assert wrong.status == "fail"
    failing = {c.name for c in wrong.checks if not c.ok}
    assert {"end_state.replacements", "end_state.refunds", "tool_calls"} <= failing

    summary = metrics(scenarios, [*both, wrong], _config(), {
        "agent": "agent-model", "customer": "sim", "judge": "judge"})  # fmt: skip
    assert summary["task_success_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert summary["policy_compliance"] == 1.0
    report = markdown({"run_id": "test"}, scenarios, [*both, wrong], summary)
    assert "### `missing-replacement` run 1: fail" in report
    assert "**customer:** yes" in report  # the failure's transcript is in the report

    # Every run database and the template are gone.
    admin = create_engine(settings.database_url().set(database="postgres"))
    try:
        with admin.connect() as conn:
            left = conn.execute(
                text("SELECT datname FROM pg_database WHERE datname LIKE :p"),
                {"p": f"{settings.postgres_db}_eval_%"},
            ).all()
    finally:
        admin.dispose()
    assert left == []
