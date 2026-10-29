"""python -m harness {run,list,validate,baseline,compare}. Run from backend/ with
PYTHONPATH=../evals (the Makefile's `eval` targets do this).

    run       play scenarios against the agent and write results.json + report.md
    list      show scenarios (filtered by --tags / --ids)
    validate  check scenario files against the seed data, offline
    baseline  turn a results.json into a baseline file to commit
    compare   the CI gate: a results.json against a baseline; exit 1 if it fails
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.agent.prompts import load_prompts
from app.config import get_settings
from app.retrieval.embedding import embedder_from_settings
from harness.baseline import compare, make_baseline
from harness.config import CONFIG_PATH, EVALS_DIR, load_config
from harness.report import markdown, metrics, results_json
from harness.runner import GroqFactory, Harness
from harness.scenario import load_scenarios, select
from harness.validate import problems


def _csv(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def _selected(args) -> list:
    return select(
        load_scenarios(),
        tags=_csv(args.tags),
        exclude_tags=_csv(args.exclude_tags),
        ids=_csv(args.ids),
    )


async def _run(args) -> int:
    config = load_config(args.config)
    updates = {}
    if args.runs:
        updates["runs_per_scenario"] = args.runs
    if args.concurrency:
        updates["concurrency"] = args.concurrency
    if args.policy_mode:
        updates["policy_context_mode"] = args.policy_mode
    if args.no_judge:
        updates["judge"] = config.judge.model_copy(update={"enabled": False})
    config = config.model_copy(update=updates)

    scenarios = _selected(args)
    if not scenarios:
        print("No scenarios match.", file=sys.stderr)
        return 2
    found = problems(scenarios)
    if found:
        print("Scenario problems:\n  " + "\n  ".join(found), file=sys.stderr)
        return 2

    settings = get_settings()
    embedder = embedder_from_settings(settings)
    await embedder.warm()
    harness = Harness(
        config, settings, embedder=embedder, llm_factory=GroqFactory(settings, config)
    )
    models = {
        "agent": harness.agent_model,
        "customer": config.customer.model,
        "judge": config.judge.model,
    }
    total = len(scenarios) * config.runs_per_scenario
    started = datetime.now(UTC)
    print(
        f"{len(scenarios)} scenarios x {config.runs_per_scenario} runs, "
        f"concurrency {config.concurrency}, policy mode {harness.mode}",
        flush=True,
    )
    done = 0

    def progress(result) -> None:
        nonlocal done
        done += 1
        failing = ", ".join(c.name for c in result.checks if not c.ok)
        detail = (
            f" ({failing})"
            if failing
            else f" ({result.error.splitlines()[0]})"
            if result.error
            else ""
        )
        print(
            f"[{done}/{total}] {result.scenario_id} run {result.run_index + 1}: "
            f"{result.status}{detail}",
            flush=True,
        )

    await harness.prepare()
    try:
        runs = await harness.run_all(scenarios, config.runs_per_scenario, on_result=progress)
    finally:
        await harness.close()

    header = {
        "run_id": started.strftime("%Y%m%dT%H%M%SZ"),
        "started_at": started.isoformat(timespec="seconds"),
        "duration_s": round((datetime.now(UTC) - started).total_seconds(), 1),
        "scenarios": len(scenarios),
        "runs_per_scenario": config.runs_per_scenario,
        "concurrency": config.concurrency,
        "seed": config.seed,
        "policy_context_mode": harness.mode,
        "prompt": load_prompts(settings.agent_prompt_version).fingerprint,
        "agent_model": f"{harness.agent_model} (temperature {harness.agent_temperature})",
        "customer_model": f"{config.customer.model} (temperature {config.customer.temperature})",
        "judge_model": config.judge.model if config.judge.enabled else "disabled",
        "filters": {"tags": args.tags, "exclude_tags": args.exclude_tags, "ids": args.ids},
        "stopped_early": harness.stopped,
    }
    summary = metrics(scenarios, runs, config, models)
    out = args.output_dir or EVALS_DIR / "results" / header["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(results_json(header, scenarios, runs, summary), indent=2, default=str)
    )
    (out / "report.md").write_text(markdown(header, scenarios, runs, summary))
    if harness.stopped:
        print(f"\nStopped early: {harness.stopped}. Remaining runs are marked as errors.")
    print(
        f"\nTask success {summary['task_success_rate']}, policy compliance "
        f"{summary['policy_compliance']}, errored runs {summary['errored_runs']}\n"
        f"Wrote {out / 'report.md'} and {out / 'results.json'}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "list", "validate"):
        p = sub.add_parser(name)
        p.add_argument("--tags", help="comma-separated: scenarios with any of these tags")
        p.add_argument("--exclude-tags", help="comma-separated: skip scenarios with these")
        p.add_argument("--ids", help="comma-separated scenario ids")
        if name == "run":
            p.add_argument("--runs", type=int, help="runs per scenario (config: 3)")
            p.add_argument("--concurrency", type=int, help="parallel runs (config: 1)")
            p.add_argument("--policy-mode", choices=["retrieval", "full_text"])
            p.add_argument("--no-judge", action="store_true")
            p.add_argument("--config", type=Path, default=CONFIG_PATH)
            p.add_argument("--output-dir", type=Path)
            p.add_argument("--verbose", action="store_true")
    p = sub.add_parser("baseline")
    p.add_argument("--results", type=Path, required=True, help="a results.json from `run`")
    p.add_argument("--suite", required=True, help="name, e.g. smoke or full")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--max-drop", type=float, help="allowed task-success drop (config: ci)")
    p.add_argument("--max-error-rate", type=float, help="errored-run share (config: ci)")
    p.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if getattr(args, "verbose", False) else logging.ERROR)

    if args.command == "list":
        for s in _selected(args):
            print(f"{s.id:48} {s.workflow.value:17} {','.join(s.tags)}")
        return 0
    if args.command == "validate":
        scenarios = _selected(args)
        found = problems(scenarios)
        print("\n".join(found) if found else f"{len(scenarios)} scenarios OK")
        return 1 if found else 0
    if args.command == "baseline":
        results = json.loads(args.results.read_text())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(make_baseline(results, args.suite), indent=2) + "\n")
        print(f"Wrote {args.output}")
        return 0
    if args.command == "compare":
        gate = load_config(args.config).ci
        if not args.baseline.exists():
            print(f"::warning::No baseline at {args.baseline}; only the hard gates apply.")
        verdict = compare(
            json.loads(args.results.read_text()),
            json.loads(args.baseline.read_text()) if args.baseline.exists() else None,
            max_drop=gate.max_success_drop if args.max_drop is None else args.max_drop,
            max_error_rate=(
                gate.max_error_rate if args.max_error_rate is None else args.max_error_rate
            ),
        )
        text = verdict.markdown()
        print(text)
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:  # shown on the workflow run's page
            with open(summary, "a") as f:
                f.write(text)
        return 0 if verdict.ok else 1
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
