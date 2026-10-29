"""Scenario files: evals/scenarios/**/*.yaml, each a list under `scenarios:`.

One file per workflow keeps related cases together and scales to hundreds of scenarios;
ids are global and must be unique. The schema is strict (unknown keys are errors), so a typo
in an expectation can't silently turn a check off.
"""

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

SCENARIOS_DIR = Path(__file__).resolve().parents[1] / "scenarios"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Workflow(StrEnum):
    """The five workflows in the brief."""

    ORDER_TRACKING = "order_tracking"
    MISSING_DELIVERY = "missing_delivery"
    DAMAGED_PRODUCT = "damaged_product"
    RETURN_REFUND = "return_refund"
    OTHER = "other"


class Behavior(StrEnum):
    COOPERATIVE = "cooperative"
    ANGRY = "angry"
    IMPATIENT = "impatient"
    VAGUE = "vague"
    MISHEARS = "mishears"
    CHANGES_MIND = "changes_mind"
    PUSHY = "pushy"
    MANIPULATIVE = "manipulative"
    CONFUSED = "confused"


class Persona(Strict):
    name: str
    description: str
    behaviors: list[Behavior] = Field(default_factory=lambda: [Behavior.COOPERATIVE])


class Facts(Strict):
    """What the customer knows. They share it only when it makes sense, and never invent
    details that aren't here (a missing PIN means they don't know it)."""

    email: str | None = None
    phone: str | None = None
    postal_code: str | None = None
    order_number: str | None = None
    other: list[str] = Field(default_factory=list)


class Script(Strict):
    opening: str | None = None  # first message, sent verbatim (removes one source of variance)
    notes: list[str] = Field(default_factory=list)  # behaviour specifics for the simulator


class ToolExpectation(Strict):
    """Matches one entry of the agent's tool log (conversation_turns, role "tool").

    `args` is a subset match; strings compare case-insensitively. phase "call" is the model
    calling the tool (a write tool then only proposes); "executed" and "cancelled" are the
    customer's yes or no to a proposal.
    """

    tool: str | list[str]  # a list means any of these tools
    phase: Literal["call", "executed", "cancelled", "system"] = "call"
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool | None = None
    only_in_mode: Literal["retrieval", "full_text"] | None = None


class ToolCallExpectations(Strict):
    required: list[ToolExpectation] = Field(default_factory=list)  # in this order
    forbidden: list[ToolExpectation] = Field(default_factory=list)


class RowCount(Strict):
    min: int = 0
    max: int | None = None
    match: dict[str, Any] = Field(default_factory=dict)  # every counted row must match


# Per kind: "any" (not checked), a list (exactly these new rows, in any order, each a subset
# match), or a RowCount. Omitted means no new rows of that kind.
RowExpectation = Literal["any"] | list[dict[str, Any]] | RowCount


class EndState(Strict):
    refunds: RowExpectation = Field(default_factory=list)
    returns: RowExpectation = Field(default_factory=list)
    replacements: RowExpectation = Field(default_factory=list)
    tickets: RowExpectation = Field(default_factory=list)  # create_ticket, not escalations


class Expectations(Strict):
    # The email of the customer the conversation must end verified as; null: never verified.
    verified_as: Literal["any"] | str | None = "any"
    escalation: bool | Literal["any"] = False  # an agent handoff (escalate_to_human, executed)
    escalation_reason: str | None = None
    # A ticket (create_ticket) or an escalation, either one: for enquiries the agent can't
    # resolve itself, where both are acceptable outcomes.
    handoff_required: bool = False
    tool_calls: ToolCallExpectations = Field(default_factory=ToolCallExpectations)
    end_state: EndState = Field(default_factory=EndState)


class Scenario(Strict):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,80}$")
    title: str
    workflow: Workflow
    tags: list[str] = Field(default_factory=list)
    seed_case: str | None = None  # the row in docs/seed-cases.md it exercises
    persona: Persona
    goal: str
    facts: Facts = Field(default_factory=Facts)
    script: Script = Field(default_factory=Script)
    max_turns: int = Field(12, ge=1, le=40)
    expect: Expectations = Field(default_factory=Expectations)
    # What a good outcome looks like, in words, for the LLM judge (never for scoring).
    judge_notes: str = ""

    @model_validator(mode="after")
    def _tags_include_workflow(self):
        if self.workflow.value not in self.tags:
            self.tags.insert(0, self.workflow.value)
        return self


def load_scenarios(directory: Path = SCENARIOS_DIR) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for path in sorted(directory.rglob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        for raw in data.get("scenarios", []):
            try:
                scenarios.append(Scenario.model_validate(raw))
            except Exception as exc:
                raise ValueError(f"{path.name}: scenario {raw.get('id')!r}: {exc}") from exc
    ids = [s.id for s in scenarios]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate scenario ids: {', '.join(duplicates)}")
    return scenarios


def select(
    scenarios: list[Scenario],
    *,
    tags: list[str] | None = None,
    exclude_tags: list[str] | None = None,
    ids: list[str] | None = None,
) -> list[Scenario]:
    """Scenarios with ANY of `tags`, none of `exclude_tags`, and (if given) one of `ids`."""
    chosen = []
    for s in scenarios:
        if ids and s.id not in ids:
            continue
        if tags and not set(tags) & set(s.tags):
            continue
        if exclude_tags and set(exclude_tags) & set(s.tags):
            continue
        chosen.append(s)
    if ids:
        unknown = set(ids) - {s.id for s in scenarios}
        if unknown:
            raise ValueError(f"unknown scenario ids: {', '.join(sorted(unknown))}")
    return chosen
