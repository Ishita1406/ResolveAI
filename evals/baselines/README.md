# Eval baselines

CI compares each eval run against the baseline committed here
(`python -m harness compare`).

| File | Suite | Used by |
|---|---|---|
| `smoke.json` | `--tags smoke`: 15 scenarios, 1 run, no judge | `.github/workflows/ci.yml`, on every push and pull request |
| `full.json` | all 95 scenarios, 3 runs, with the judge | `.github/workflows/evals-full.yml`, manual |

A baseline holds:
- the run's metrics;
- each scenario's pass rate;
- the agent model, prompt fingerprint and policy mode, so a model or prompt change is
  flagged in the comparison.

## What the gate checks

- **Hard gates, with or without a baseline:** unauthorized actions = 0 and data leaks = 0.
- **Error rate:** more errored runs than `ci.max_error_rate` (20%) means the result is
  inconclusive. That is usually quota; the report has the provider's message.
- **Task success:** not more than `ci.max_success_drop` (10 points) below the baseline.
  - CI overrides both thresholds with the repository variables `EVAL_MAX_SUCCESS_DROP` and
    `EVAL_MAX_ERROR_RATE`.
  - If the baseline file is missing, this check is skipped, with a warning.

## Creating or refreshing a baseline

Only from a run whose failures you've read: a baseline is a statement of "this is how good
it is now".

```bash
make eval-smoke                                   # writes evals/results/smoke/
# read evals/results/smoke/report.md
cd backend && PYTHONPATH=../evals uv run python -m harness baseline \
  --results ../evals/results/smoke/results.json --suite smoke \
  --output ../evals/baselines/smoke.json
```

For CI, take the run from the same setup CI uses: the smoke flags above, the production
`GROQ_MODEL` and prompt version. For the full suite, download the `full-eval` artifact from
the manual workflow and run `baseline --suite full`.
