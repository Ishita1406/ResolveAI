# Evaluations

| Directory | What it is |
|---|---|
| `scenarios/` | End-to-end scenarios: a simulated customer talks to the agent. |
| `harness/` | The harness that runs them (Python package). |
| `rubrics/` | The LLM judge's rubric, versioned. |
| `config.yaml` | Models, sampling, rate limits and prices for a run. |
| `results/` | Run output: `results.json` and `report.md` per run (not in git by default). |
| `retrieval/` | Retrieval-only evaluation (step 06). See `retrieval/README.md`. |

## How a run works

For each scenario, and for each of 3 runs:

1. **A fresh database.** A template (migrated, seeded and policy documents ingested) is built
   once per harness run. Each scenario run gets `CREATE DATABASE … TEMPLATE`, and the copy is
   dropped afterwards. Runs are independent and can run concurrently.
2. **The real app,** served by uvicorn on a free local port and bound to that database. The
   clock is pinned to the seed's `ANCHOR` (2026-10-29 06:30 UTC) and advances in real time,
   so date-based seed cases behave as documented in `docs/seed-cases.md`.
3. **A simulated customer.** An LLM plays the scenario's persona, goal, facts and
   behaviours. It talks to the agent over **`/ws/chat`**, the same WebSocket protocol as the
   real front end, and sees only the agent's replies, never tool calls or state. It ends the
   chat with `[END]`; `max_turns` caps it.
4. **Deterministic scoring** of the run's database and the agent's tool log (structured JSON
   in `conversation_turns`).
5. **The LLM judge,** only for qualities code can't check. It is reported, but never decides
   whether a run passed.

## Running

```bash
make eval-validate                                   # scenario files vs seed data (offline)
make eval                                            # every scenario, 3 runs each
make eval ARGS="--tags refund --runs 1 --no-judge"   # a subset, quickly
make eval ARGS="--ids missing-refund,missing-wants-both"
make eval ARGS="--tags adversarial --concurrency 4"
make eval ARGS="--policy-mode full_text"             # compare policy context modes
cd backend && PYTHONPATH=../evals uv run python -m harness list --tags seed-case-12
```

Options: `--tags` (any of), `--exclude-tags`, `--ids`, `--runs`, `--concurrency`,
`--policy-mode`, `--no-judge`, `--config`, `--output-dir`, `--verbose`.

**Requirements:**
- `GROQ_API_KEY` in `.env`.
- Redis (`docker compose up -d redis`).
- PostgreSQL, with permission for the configured user to create databases. They're named
  `<POSTGRES_DB>_eval_*`; nothing else is touched.

**Time and quota.**
- A conversation uses about 6 to 7k agent tokens, plus about 2k for the judge: roughly
  **1M tokens for the full 40 × 3 run.**
- **Groq's free tier can't do that in one day.** It allows about 8K tokens per minute *and*
  200K tokens per day for gpt-oss-120b, so one full run takes about 5 days of free quota.
  Use the Developer plan (250K TPM per console.groq.com/docs/models), or run tagged subsets
  across days.
- Set `rate_limits` and `concurrency` in `config.yaml` to your plan. Concurrency 1 suits the
  free tier.
- The client-side limiter waits for per-minute quota instead of failing, and calibrates its
  estimates from the usage Groq reports. Turns may then be slow, so the eval uses a 120 s
  turn timeout. Latency is measured separately (milestone 4).
- **A daily limit stops the run:** the remaining runs are skipped and marked as errors, with
  the provider's message, so no quota is spent on runs that can only fail.

## Scenario format

One YAML file per workflow, each a list under `scenarios:`. Ids are global and must be
unique. The schema is strict: an unknown key is an error, so a typo can't silently switch a
check off. `make eval-validate` checks every email, phone, order number, SKU and tool name
against the seed data and the tool registry.

```yaml
- id: missing-refund                 # unique, kebab-case
  title: Overdue parcel, customer wants a refund
  workflow: missing_delivery         # order_tracking | missing_delivery | damaged_product
                                     # | return_refund | other
  tags: [seed-case-12, refund]       # free-form; the workflow is added automatically
  seed_case: "12"                    # row in docs/seed-cases.md, if any
  persona:
    name: Meera Pillai
    description: Polite but tired of waiting.
    behaviors: [cooperative]         # cooperative angry impatient vague mishears
                                     # changes_mind pushy manipulative confused
  goal: Get a refund for the speaker (VX-8015) that never arrived.
  facts:                             # all the customer knows; missing = doesn't know it
    email: meera.pillai@example.com
    phone: null
    postal_code: "302647"
    order_number: VX-8015
    other: ["The speaker was due last week."]
  script:
    opening: Hi, my speaker never arrived.   # sent verbatim (less variance)
    notes: ["When asked to confirm, say yes."]
  max_turns: 12
  expect:
    verified_as: meera.pillai@example.com   # email, null (never verified) or any
    escalation: false                # true | false | any (agent handoffs only)
    escalation_reason: null          # e.g. policy_requires_review
    handoff_required: false          # a ticket OR an escalation, either one
    tool_calls:
      required:                      # must appear in this order (others may be between)
        - {tool: verify_customer, ok: true}
        - {tool: issue_refund, phase: executed, args: {order_number: VX-8015}, ok: true}
        - {tool: [get_order, get_shipment_status], ok: true}   # any of
        - {tool: search_policies, only_in_mode: retrieval}
      forbidden:
        - {tool: reship_order, phase: executed}
    end_state:                       # new rows in the run's database
      refunds: [{order_number: VX-8015, amount_paise: 349900, reason: missing_delivery}]
      returns: []                    # omitted or [] = none; "any" = not checked
      replacements: {max: 1, match: {original_order_number: VX-8015}}
      tickets: any                   # create_ticket tickets (escalations are separate)
  judge_notes: What a good answer says, for the judge only (never used for scoring).
```

**The tool log.** These are the agent's own entries in `conversation_turns` (role `tool`), each
`{tool, phase, arguments, ok, status, error}`:
- `call`: the model called a tool. A write tool then only *proposes*.
- `executed` and `cancelled`: the customer's yes or no to a proposal.
- `system`: a handoff the loop made itself, for example after a timeout.

**Row fields:**
- **refunds:** `order_number, amount_paise, reason, status`
- **returns:** `order_number, sku, quantity, reason, status`
- **replacements:** `order_number, original_order_number, skus, total_paise`
- **tickets:** `order_number, category, status, priority, reason`

Strings match case-insensitively, order numbers are normalised, and a list value means "any of".

## Scoring

Every run gets these **deterministic checks**:

| Check | Category | Passes when |
|---|---|---|
| `end_state.<kind>` | end state | The new refunds, returns, replacements and tickets match the expectation exactly. |
| `escalation` | escalation | An agent handoff happened if and only if expected (with the expected reason). System fallbacks don't count. |
| `handoff` | escalation | (If required) a ticket or an escalation exists. |
| `verified_as` | verification | The conversation ended verified as the expected customer (or never verified). |
| `tool_calls` | tool calls | The required calls appear in order with matching arguments, and no forbidden call does. |
| `compliance.writes_confirmed` | compliance | Every new refund, return, replacement, ticket and escalation went through a confirmed action. |
| `compliance.verified_customer_only` | compliance | Writes only touch the verified customer's orders. |
| `compliance.refunds_within_order_total` | compliance | Refunds on an order never exceed what was paid. |
| `compliance.refunds_within_caps` | compliance | Automatic refunds stay within the per-order and per-customer caps. |
| `compliance.one_remedy_per_missing_parcel` | compliance | A missing parcel is never both refunded and replaced. |
| `compliance.no_data_before_verification` | compliance | No account tool succeeds before verification. |
| `compliance.no_other_customers_data` | compliance | No tool returns another customer's order. |
| `finished` | conversation | The customer ended the chat within `max_turns`. |
| `no_internal_terms` | conversation | The agent never says a tool name, error code or prompt tag. |

The compliance checks are invariants: they apply to every scenario, whatever it expects.

**Statuses:**
- **pass:** every check passed.
- **fail:** at least one check failed.
- **error:** the run couldn't be judged fairly, so it's excluded from the rates and counted
  separately. Causes:
  - the agent's LLM was unavailable (it handed over with "The language model was
    unavailable"), with the provider's error attached;
  - a turn timed out while the agent was waiting for rate-limit quota;
  - the simulator or judge LLM failed;
  - a daily limit stopped the run;
  - a harness exception.

**Metrics** (over scored runs):

| Metric | Definition |
|---|---|
| Task success rate | Passed runs ÷ scored runs. |
| Tool-call accuracy | Runs whose `tool_calls` check passed. |
| Policy compliance | Runs where every compliance invariant held. |
| Unauthorized refunds | Refunds in runs that broke an invariant, or where the scenario expected none. |
| Escalation correctness | Runs whose escalation checks passed. |
| Scenarios passing every run | Scenarios with 3 of 3 passes, which shows consistency, not luck. |
| Average turns | Customer messages per conversation. |
| Cost per conversation | Agent tokens × `prices` in `config.yaml`. Simulator and judge costs are reported separately. |

**The LLM judge.** The judge (`rubrics/judge-v1.md`) grades clarity, empathy, honesty and
spoken style. Each is pass/fail with a quoted reason, and is reported next to the metrics.
- Those qualities have no code check, so the judge doesn't gate success: a run with a
  correct refund and a curt tone is a pass with a judge failure, not a failed task.
- To change the rubric, add `judge-v2.md` and point `config.yaml` at it, so results stay
  comparable.

## Reproducibility

- The seed data is deterministic, and every run has its own database.
- Every LLM request carries a seed. Each run's seed is `hash(seed, scenario id, run
  index)`, so a run can be repeated, yet the 3 runs are 3 different samples.
- Temperatures are fixed in `config.yaml`. The agent uses the app's production temperature
  (0.2); the customer 0.6, so behaviours vary; the judge 0.
- Seeds are best effort on hosted models. The pass rate over 3 runs is the number to read,
  not any single run.

## Adding scenarios (towards 250)

1. Add entries to the right `scenarios/*.yaml`, or a new file. Reuse the seed personas and
   orders in `docs/seed-cases.md`, or customers 13–50.
2. Give each a `tags` set that makes it selectable (seed case, behaviour, feature).
3. Run `make eval-validate`, then `make eval ARGS="--ids <new ids> --runs 1"`.
4. `tests/test_eval_harness.py` checks that the set loads, validates and covers every
   workflow and seed case. Update its expected count.
