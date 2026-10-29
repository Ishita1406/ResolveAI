"""LLM judge for qualities code can't check (clarity, empathy, honesty, spoken style).

The rubric lives in evals/rubrics/ and is versioned like the prompts. The judge never
decides whether a run passed: task success comes from the deterministic checks only, and
judge results are reported next to them.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.agent.llm import Completion, LLMClient, LLMError
from harness.scenario import Scenario

CRITERIA = ("clarity", "empathy", "honesty", "spoken_style")


@dataclass
class Verdict:
    criteria: dict[str, dict[str, Any]] = field(default_factory=dict)  # name: {pass, reason}
    error: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool | None:
        if self.error:
            return None
        return all(c["pass"] for c in self.criteria.values())


def judge_input(scenario: Scenario, transcript: list[dict[str, str]]) -> str:
    lines = [
        f"Scenario: {scenario.title}",
        f"Customer persona: {scenario.persona.name}. {scenario.persona.description}",
        f"Customer's goal: {scenario.goal}",
    ]
    if scenario.judge_notes:
        lines.append(f"What a good outcome looks like: {scenario.judge_notes}")
    lines += ["", "Conversation:"]
    lines += [f"{t['role'].upper()}: {t['text']}" for t in transcript]
    return "\n".join(lines)


def parse(text: str) -> dict[str, dict[str, Any]]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in the judge's reply")
    criteria = json.loads(match.group(0)).get("criteria", {})
    missing = [c for c in CRITERIA if c not in criteria]
    if missing:
        raise ValueError(f"judge omitted: {', '.join(missing)}")
    return {
        c: {"pass": bool(criteria[c].get("pass")), "reason": str(criteria[c].get("reason", ""))}
        for c in CRITERIA
    }


async def judge(
    llm: LLMClient, rubric: str, scenario: Scenario, transcript: list[dict[str, str]]
) -> Verdict:
    messages = [
        {"role": "system", "content": rubric},
        {"role": "user", "content": judge_input(scenario, transcript)},
    ]
    try:
        completion = None
        async for event in llm.stream(messages, []):
            if isinstance(event, Completion):
                completion = event
        if completion is None:
            raise LLMError("no completion", retryable=False)
        usage = {
            k: int(completion.usage.get(k) or 0) for k in ("prompt_tokens", "completion_tokens")
        }
        usage["llm_calls"] = 1
        try:
            return Verdict(criteria=parse(completion.text), usage=usage)
        except (ValueError, json.JSONDecodeError) as exc:
            return Verdict(error=f"unparseable judge reply: {exc}", usage=usage)
    except LLMError as exc:
        return Verdict(error=f"judge unavailable: {exc}")


def load_rubric(path: Path) -> str:
    return path.read_text()
