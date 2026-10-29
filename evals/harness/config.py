"""evals/config.yaml: models, sampling, rate limits and prices for a harness run."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

EVALS_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = EVALS_DIR / "config.yaml"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentModel(Strict):
    model: str | None = None  # None: the app's GROQ_MODEL
    temperature: float | None = None  # None: the app's LLM_TEMPERATURE (what production uses)
    turn_timeout_s: float = 120.0


class SimulatorModel(Strict):
    model: str
    temperature: float = 0.6
    max_completion_tokens: int = 300
    reasoning_effort: str | None = "low"


class JudgeModel(Strict):
    enabled: bool = True
    model: str
    temperature: float = 0.0
    max_completion_tokens: int = 800
    reasoning_effort: str | None = "low"
    rubric: str = "rubrics/judge-v1.md"


class Price(Strict):
    input: float  # USD per million tokens
    output: float


class CIGate(Strict):
    max_success_drop: float = Field(0.10, ge=0, le=1)  # vs the stored baseline
    max_error_rate: float = Field(0.20, ge=0, le=1)  # more errored runs = inconclusive


class EvalConfig(Strict):
    runs_per_scenario: int = Field(3, ge=1)
    concurrency: int = Field(1, ge=1)
    seed: int = 20261029
    policy_context_mode: Literal["retrieval", "full_text"] | None = None  # None: app setting
    agent: AgentModel = Field(default_factory=AgentModel)
    customer: SimulatorModel
    judge: JudgeModel
    rate_limits: dict[str, int] = Field(default_factory=dict)  # tokens per minute, per model
    prices: dict[str, Price] = Field(default_factory=dict)
    ci: CIGate = Field(default_factory=CIGate)

    def cost_usd(self, model: str, usage: dict[str, int]) -> float | None:
        price = self.prices.get(model)
        if price is None:
            return None
        return (
            usage.get("prompt_tokens", 0) * price.input
            + usage.get("completion_tokens", 0) * price.output
        ) / 1_000_000


def load_config(path: Path = CONFIG_PATH) -> EvalConfig:
    return EvalConfig.model_validate(yaml.safe_load(path.read_text()))
