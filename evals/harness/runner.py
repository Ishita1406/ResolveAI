"""Run scenarios: each run gets a fresh database, its own app instance and a simulated
customer, talks over /ws/chat until the customer ends it (or max_turns), and is scored."""

import asyncio
import hashlib
import logging
import re
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.agent.llm import GroqClient, LLMClient, LLMError
from app.agent.ratelimit import TokenRateLimiter
from app.config import Settings
from app.db.seed import ANCHOR
from app.policy.config import PolicyConfig, get_policy
from app.retrieval.embedding import Embedder
from harness.checks import Check, RunData, collect, score
from harness.client import ChatClient
from harness.config import EVALS_DIR, EvalConfig
from harness.customer import SimulatedCustomer
from harness.databases import Databases
from harness.judge import Verdict, judge, load_rubric
from harness.scenario import Scenario
from harness.server import serve

log = logging.getLogger(__name__)

ZERO = {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0}
# A provider limit that won't reset within this harness run: stop instead of failing on.
DAILY_LIMIT = re.compile(r"per day|\bTPD\b|\bRPD\b", re.IGNORECASE)


@dataclass(frozen=True)
class ModelSpec:
    model: str
    temperature: float
    seed: int
    max_completion_tokens: int | None
    reasoning_effort: str | None


LLMFactory = Callable[[str, ModelSpec], LLMClient]  # role ("agent"|"customer"|"judge"), spec


@dataclass
class RunResult:
    scenario_id: str
    run_index: int
    seed: int
    status: str = "error"  # pass | fail | error
    checks: list[Check] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    new_rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    turns: int = 0
    finished: bool = False
    usage: dict[str, dict[str, int]] = field(default_factory=dict)  # agent, customer, judge
    judge: Verdict | None = None
    error: str | None = None
    duration_s: float = 0.0
    quota_wait_s: float = 0.0  # time the agent's LLM client waited for rate-limit quota

    def check(self, name: str) -> Check | None:
        return next((c for c in self.checks if c.name == name), None)


def run_seed(base: int, scenario_id: str, run_index: int) -> int:
    """Stable per run, different between runs: reproducible, yet 3 runs are 3 samples."""
    digest = hashlib.sha256(f"{base}:{scenario_id}:{run_index}".encode()).hexdigest()
    return int(digest[:8], 16)


def _add(total: dict[str, int], usage: dict[str, Any] | None) -> None:
    for key in ZERO:
        total[key] += int((usage or {}).get(key) or 0)


class GroqFactory:
    """One rate limiter per model, shared by every client of that model in this run."""

    def __init__(self, settings: Settings, config: EvalConfig) -> None:
        if settings.groq_api_key is None or not settings.groq_api_key.get_secret_value():
            raise SystemExit("GROQ_API_KEY is not set (see .env.example).")
        self.settings = settings
        self.limiters = {model: TokenRateLimiter(tpm) for model, tpm in config.rate_limits.items()}

    def __call__(self, role: str, spec: ModelSpec) -> LLMClient:
        return GroqClient(
            api_key=self.settings.groq_api_key.get_secret_value(),
            model=spec.model,
            base_url=self.settings.groq_base_url,
            temperature=spec.temperature,
            timeout_s=max(self.settings.llm_request_timeout_s, 60.0),
            max_completion_tokens=spec.max_completion_tokens,
            reasoning_effort=spec.reasoning_effort,
            seed=spec.seed,
            rate_limiter=self.limiters.get(spec.model),
        )


class Harness:
    def __init__(
        self,
        config: EvalConfig,
        settings: Settings,
        *,
        embedder: Embedder,
        llm_factory: LLMFactory,
        policy: PolicyConfig | None = None,
    ) -> None:
        self.config = config
        self.settings = settings
        self.embedder = embedder
        self.llm_factory = llm_factory
        self.policy = policy or get_policy()
        self.mode = config.policy_context_mode or settings.policy_context_mode
        self.agent_model = config.agent.model or settings.groq_model
        self.agent_temperature = (
            config.agent.temperature
            if config.agent.temperature is not None
            else settings.llm_temperature
        )
        self.databases = Databases(settings)
        self.rubric = load_rubric(EVALS_DIR / config.judge.rubric)
        self.stopped: str | None = None  # set when a daily quota runs out

    async def prepare(self) -> None:
        await self.databases.drop_leftovers()
        await self.databases.build_template(self.embedder)

    async def close(self) -> None:
        await self.databases.drop_template()
        self.databases.dispose()

    async def run_all(
        self, scenarios: list[Scenario], runs: int, *, on_result=None
    ) -> list[RunResult]:
        semaphore = asyncio.Semaphore(self.config.concurrency)

        async def one(scenario: Scenario, index: int) -> RunResult:
            async with semaphore:
                result = await self.run(scenario, index)
            if on_result:
                on_result(result)
            return result

        jobs = [one(s, i) for s in scenarios for i in range(runs)]
        return list(await asyncio.gather(*jobs))

    async def run(self, scenario: Scenario, index: int) -> RunResult:
        seed = run_seed(self.config.seed, scenario.id, index)
        result = RunResult(scenario_id=scenario.id, run_index=index, seed=seed)
        result.usage = {role: dict(ZERO) for role in ("agent", "customer", "judge")}
        if self.stopped:
            result.error = f"skipped: {self.stopped}"
            return result
        started = time.monotonic()
        name = await self.databases.create_run_db()
        clients: list[LLMClient] = []
        try:
            agent_llm = self.llm_factory(
                "agent",
                ModelSpec(
                    self.agent_model,
                    self.agent_temperature,
                    seed,
                    self.settings.llm_max_completion_tokens,
                    self.settings.llm_reasoning_effort or None,
                ),
            )
            c = self.config.customer
            customer_llm = self.llm_factory(
                "customer",
                ModelSpec(
                    c.model, c.temperature, seed, c.max_completion_tokens, c.reasoning_effort
                ),
            )
            clients += [agent_llm, customer_llm]
            settings = self.databases.settings_for(
                name,
                agent_turn_timeout_s=self.config.agent.turn_timeout_s,
                policy_context_mode=self.mode,
                groq_model=self.agent_model,
            )
            customer = SimulatedCustomer(scenario, customer_llm)
            await self._converse(scenario, settings, agent_llm, customer, result)
            _add(result.usage["customer"], customer.usage)

            engine = create_async_engine(settings.database_url())
            try:
                async with AsyncSession(engine) as db:
                    data = await collect(db, self.databases.baseline)
            finally:
                await engine.dispose()
            self._score(scenario, data, result, getattr(agent_llm, "rate_limit_wait_s", 0.0))
            provider_error = getattr(agent_llm, "last_error", None)
            if result.status == "error" and provider_error:
                result.error = f"{result.error}\nprovider: {provider_error}"
                if DAILY_LIMIT.search(provider_error):
                    self.stopped = "the agent model's daily token limit was reached"
            if self.config.judge.enabled and result.status != "error" and result.transcript:
                j = self.config.judge
                judge_llm = self.llm_factory(
                    "judge",
                    ModelSpec(
                        j.model, j.temperature, seed, j.max_completion_tokens, j.reasoning_effort
                    ),
                )
                clients.append(judge_llm)
                result.judge = await judge(judge_llm, self.rubric, scenario, result.transcript)
                _add(result.usage["judge"], result.judge.usage)
        except LLMError as exc:
            result.status, result.error = "error", f"simulator or judge LLM failed: {exc}"
            if DAILY_LIMIT.search(str(exc)):
                self.stopped = "the simulator or judge model's daily token limit was reached"
        except Exception as exc:
            log.exception("run %s/%s failed", scenario.id, index)
            result.status = "error"
            result.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
        finally:
            for client in clients:
                close = getattr(client, "aclose", None)
                if close:
                    await close()
            await self.databases.drop(name)
            result.duration_s = round(time.monotonic() - started, 2)
        return result

    async def _converse(
        self,
        scenario: Scenario,
        settings: Settings,
        agent_llm: LLMClient,
        customer: SimulatedCustomer,
        result: RunResult,
    ) -> None:
        started = time.monotonic()

        def clock():
            # The seed data is anchored at ANCHOR; time still passes during the run, so
            # expiry by seconds behaves as in production.
            return ANCHOR + timedelta(seconds=time.monotonic() - started)

        async with serve(settings, llm=agent_llm, embedder=self.embedder, clock=clock) as url:
            timeout = self.config.agent.turn_timeout_s + 30
            async with ChatClient(url, turn_timeout_s=timeout) as chat:
                reply: str | None = None
                for _ in range(scenario.max_turns):
                    message = await customer.next_message(reply)
                    if message is None:
                        result.finished = True
                        break
                    result.turns += 1
                    result.transcript.append({"role": "customer", "text": message})
                    turn = await chat.send(message)
                    if turn.done is not None:
                        _add(result.usage["agent"], turn.done.get("usage"))
                    result.transcript.append(
                        {
                            "role": "agent",
                            "text": turn.reply,
                            "tools": [
                                e["name"]
                                for e in turn.events
                                if e["type"] == "tool" and e["status"] == "started"
                            ],
                            "actions": [
                                f"{e['state']}: {e['summary']}"
                                for e in turn.events
                                if e["type"] == "action"
                            ],
                            "outcome": (turn.done or {}).get("outcome", "error"),
                        }
                    )
                    reply = turn.reply
                else:
                    # Out of turns: give the customer one chance to close the conversation.
                    result.finished = await customer.next_message(reply) is None

    def _score(
        self, scenario: Scenario, data: RunData, result: RunResult, quota_wait_s: float
    ) -> None:
        result.tool_log = data.tool_log
        result.new_rows = {
            kind: getattr(data, kind)
            for kind in ("refunds", "returns", "replacements", "tickets", "escalations")
        }
        result.checks = score(
            scenario,
            data,
            policy=self.policy,
            mode=self.mode,
            finished=result.finished,
            agent_replies=[t["text"] for t in result.transcript if t["role"] == "agent"],
        )
        result.quota_wait_s = round(quota_wait_s, 1)
        infra = data.infra_error(quota_wait_s)
        if infra:
            result.status, result.error = "error", infra
        else:
            result.status = "pass" if all(c.ok for c in result.checks) else "fail"
