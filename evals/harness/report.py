"""Metrics, results.json and report.md."""

import json
import statistics
from collections import defaultdict
from dataclasses import asdict
from typing import Any

from harness.config import EvalConfig
from harness.judge import CRITERIA
from harness.runner import RunResult
from harness.scenario import RowCount, Scenario


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _category_ok(run: RunResult, category: str) -> bool:
    return all(c.ok for c in run.checks if c.category == category)


def run_cost(run: RunResult, config: EvalConfig, models: dict[str, str]) -> dict[str, float | None]:
    return {role: config.cost_usd(models[role], run.usage.get(role, {})) for role in models}


def metrics(
    scenarios: list[Scenario], runs: list[RunResult], config: EvalConfig, models: dict[str, str]
) -> dict[str, Any]:
    """Rates are over scored runs; runs that errored (infrastructure, not the agent) are
    counted separately and excluded, so a rate limit can't pass or fail an agent."""
    by_id = {s.id: s for s in scenarios}
    scored = [r for r in runs if r.status != "error"]
    costs = [run_cost(r, config, models) for r in scored]
    agent_costs = [c["agent"] for c in costs if c["agent"] is not None]
    harness_costs = [sum(v for k, v in c.items() if k != "agent" and v is not None) for c in costs]

    unauthorized = 0
    for r in scored:
        expected_none = by_id[r.scenario_id].expect.end_state.refunds == []
        if not _category_ok(r, "compliance") or expected_none:
            unauthorized += len(r.new_rows.get("refunds", []))

    per_scenario: dict[str, list[RunResult]] = defaultdict(list)
    for r in runs:
        per_scenario[r.scenario_id].append(r)
    fully_passing = [
        all(r.status == "pass" for r in rs)
        for rs in per_scenario.values()
        if all(r.status != "error" for r in rs)
    ]

    judged = [r.judge for r in scored if r.judge is not None and r.judge.error is None]
    judge = {c: _rate([j.criteria[c]["pass"] for j in judged]) for c in CRITERIA}

    workflows: dict[str, list[bool]] = defaultdict(list)
    for r in scored:
        workflows[by_id[r.scenario_id].workflow.value].append(r.status == "pass")

    return {
        "runs": len(runs),
        "scored_runs": len(scored),
        "errored_runs": len(runs) - len(scored),
        "task_success_rate": _rate([r.status == "pass" for r in scored]),
        "tool_call_accuracy": _rate([_category_ok(r, "tool_calls") for r in scored]),
        "policy_compliance": _rate([_category_ok(r, "compliance") for r in scored]),
        "escalation_correctness": _rate([_category_ok(r, "escalation") for r in scored]),
        "unauthorized_refunds": unauthorized,
        "scenarios_passing_every_run": _rate(fully_passing),
        "average_turns": round(statistics.mean(r.turns for r in scored), 2) if scored else None,
        "cost_per_conversation_usd": {
            "agent": round(statistics.mean(agent_costs), 6) if agent_costs else None,
            "simulator_and_judge": (
                round(statistics.mean(harness_costs), 6) if harness_costs else None
            ),
        },
        "tokens_per_conversation": {
            role: round(
                statistics.mean(
                    r.usage[role]["prompt_tokens"] + r.usage[role]["completion_tokens"]
                    for r in scored
                )
            )
            if scored
            else None
            for role in ("agent", "customer", "judge")
        },
        "judge_pass_rate": judge,
        "judged_runs": len(judged),
        "task_success_by_workflow": {w: _rate(v) for w, v in sorted(workflows.items())},
    }


def results_json(
    header: dict[str, Any],
    scenarios: list[Scenario],
    runs: list[RunResult],
    summary: dict[str, Any],
) -> dict[str, Any]:
    by_id: dict[str, list[RunResult]] = defaultdict(list)
    for r in sorted(runs, key=lambda r: (r.scenario_id, r.run_index)):
        by_id[r.scenario_id].append(r)
    return {
        **header,
        "summary": summary,
        "scenarios": [
            {
                "id": s.id,
                "title": s.title,
                "workflow": s.workflow.value,
                "tags": s.tags,
                "pass_rate": _rate(
                    [r.status == "pass" for r in by_id[s.id] if r.status != "error"]
                ),
                "runs": [_run_json(r) for r in by_id[s.id]],
            }
            for s in scenarios
            if s.id in by_id
        ],
    }


def _run_json(r: RunResult) -> dict[str, Any]:
    data = asdict(r)
    data["checks"] = [asdict(c) for c in r.checks]
    data["judge"] = asdict(r.judge) if r.judge else None
    return data


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def _expected_summary(s: Scenario) -> str:
    e = s.expect
    parts = []
    for kind in ("refunds", "returns", "replacements", "tickets"):
        want = getattr(e.end_state, kind)
        if want == "any":
            continue
        if isinstance(want, RowCount):
            parts.append(f"{kind} {want.min}..{want.max if want.max is not None else 'n'}")
        elif want:
            parts.append(f"{len(want)} {kind[:-1] if len(want) == 1 else kind}")
    if e.escalation is True:
        parts.append("escalation")
    return ", ".join(parts) or "no writes"


def markdown(
    header: dict[str, Any], scenarios: list[Scenario], runs: list[RunResult], summary: dict
) -> str:
    m = summary
    lines = ["# Agent evaluation report", ""]
    lines += [f"- **{k}:** {v}" for k, v in header.items() if not isinstance(v, dict)]
    lines += [
        "",
        "## Metrics",
        "",
        f"Rates are over {m['scored_runs']} scored runs; {m['errored_runs']} runs errored "
        "(infrastructure, e.g. rate limits) and are excluded.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Task success rate | {_pct(m['task_success_rate'])} |",
        f"| Tool-call accuracy | {_pct(m['tool_call_accuracy'])} |",
        f"| Policy compliance | {_pct(m['policy_compliance'])} |",
        f"| Unauthorized refunds | {m['unauthorized_refunds']} |",
        f"| Escalation correctness | {_pct(m['escalation_correctness'])} |",
        f"| Scenarios passing every run | {_pct(m['scenarios_passing_every_run'])} |",
        f"| Average customer turns | {m['average_turns']} |",
        f"| Agent cost per conversation | {_usd(m['cost_per_conversation_usd']['agent'])} |",
        "| Simulator + judge cost per conversation | "
        f"{_usd(m['cost_per_conversation_usd']['simulator_and_judge'])} |",
        f"| Agent tokens per conversation | {m['tokens_per_conversation']['agent']} |",
        "",
        f"LLM judge ({m['judged_runs']} runs; reported only, never part of task success):",
        "",
        "| Criterion | Pass rate |",
        "|---|---|",
        *(f"| {c} | {_pct(v)} |" for c, v in m["judge_pass_rate"].items()),
        "",
        "Task success by workflow:",
        "",
        "| Workflow | Success |",
        "|---|---|",
        *(f"| {w} | {_pct(v)} |" for w, v in m["task_success_by_workflow"].items()),
        "",
        "## Scenarios",
        "",
        "| Scenario | Workflow | Expected | Pass | Failing checks |",
        "|---|---|---|---|---|",
    ]
    by_id: dict[str, list[RunResult]] = defaultdict(list)
    for r in sorted(runs, key=lambda r: (r.scenario_id, r.run_index)):
        by_id[r.scenario_id].append(r)
    for s in scenarios:
        rs = by_id.get(s.id)
        if not rs:
            continue
        passed = sum(r.status == "pass" for r in rs)
        errored = sum(r.status == "error" for r in rs)
        failing = sorted({c.name for r in rs for c in r.checks if not c.ok})
        mark = f"{passed}/{len(rs) - errored}" + (f" ({errored} errored)" if errored else "")
        lines.append(
            f"| `{s.id}` | {s.workflow.value} | {_expected_summary(s)} | {mark} "
            f"| {', '.join(failing) or '-'} |"
        )

    failures = [r for r in runs if r.status != "pass"]
    if failures:
        lines += ["", "## Failures and errors", ""]
    for r in sorted(failures, key=lambda r: (r.scenario_id, r.run_index)):
        lines += [f"### `{r.scenario_id}` run {r.run_index + 1}: {r.status}", ""]
        if r.error:
            lines += ["```", r.error.strip(), "```", ""]
        for c in r.checks:
            if not c.ok:
                lines.append(f"- **{c.name}:** {c.detail}")
        if r.judge and r.judge.criteria:
            for name, v in r.judge.criteria.items():
                if not v["pass"]:
                    lines.append(f"- judge **{name}** failed: {v['reason']}")
        lines += ["", "<details><summary>Transcript</summary>", ""]
        for t in r.transcript:
            extra = ""
            if t["role"] == "agent":
                tools = f" [tools: {', '.join(t['tools'])}]" if t.get("tools") else ""
                acts = f" [actions: {'; '.join(t['actions'])}]" if t.get("actions") else ""
                extra = tools + acts
            lines.append(f"- **{t['role']}:** {t['text']}{extra}")
        lines += ["", "Tool log:", "", "```"]
        lines += [json.dumps(e, ensure_ascii=False) for e in r.tool_log] or ["(empty)"]
        lines += ["```", "", "</details>", ""]
    return "\n".join(lines) + "\n"
