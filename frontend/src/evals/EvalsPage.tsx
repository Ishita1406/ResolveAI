import { api } from "../api/client";
import type { EvalResults, EvalRun, EvalScenario } from "../api/types";
import { LoadError } from "../components/LoadError";
import { count, percent, title, usd, when } from "../lib/format";
import { useLoad } from "../lib/useLoad";

// The brief's targets (docs/BRIEF.MD, "Metrics and targets"), shown next to each metric.
// They only colour the tiles; the CI gate (python -m harness compare) is what enforces.
interface Target {
  text: string;
  met: (v: number) => boolean;
}
const TARGETS: Record<string, Target> = {
  task_success_rate: { text: "target ≥ 85%", met: (v) => v >= 0.85 },
  tool_call_accuracy: { text: "target ≥ 95%", met: (v) => v >= 0.95 },
  policy_compliance: { text: "target 100%", met: (v) => v >= 1 },
  unauthorized_refunds: { text: "target 0", met: (v) => v === 0 },
  escalation_correctness: { text: "target ≥ 90%", met: (v) => v >= 0.9 },
};

export function EvalsPage({ run, onRun }: { run: string | null; onRun: (name: string | null) => void }) {
  const runs = useLoad(() => api.evalRuns(), "runs");
  const results = useLoad(() => (run ? api.evalRun(run) : api.latestEval()), run ?? "latest");

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Evaluation results</h1>
          <p>Simulated customers against the real agent, scored on the database it leaves behind. The latest run is shown unless you pick another.</p>
        </div>
        <label className="row small">
          <span className="muted">Run</span>
          <select value={run ?? ""} onChange={(e) => onRun(e.target.value || null)} aria-label="Eval run">
            <option value="">Latest</option>
            {runs.data?.map((r) => (
              <option key={r.name} value={r.name}>
                {r.name} · {percent(r.task_success_rate)} · {when(r.started_at)}
              </option>
            ))}
          </select>
        </label>
      </div>

      {results.error && <LoadError error={results.error} retry={results.reload} />}
      {results.loading && <div className="stack"><div className="skeleton" /><div className="skeleton" /></div>}
      {results.data && !results.loading && <Results data={results.data} />}
    </>
  );
}

function Metric({ label, value, raw, metric }: { label: string; value: string; raw?: number | null; metric?: string }) {
  const target = metric ? TARGETS[metric] : undefined;
  const tone = target && raw != null ? (target.met(raw) ? "ok" : "bad") : "";
  return (
    <div className={`metric ${tone}`}>
      <span className="label">{label}</span>
      <span className="value">{value}</span>
      {target && <span className="target">{target.text}</span>}
    </div>
  );
}

function failing(scenarios: EvalScenario[]) {
  return scenarios
    .map((s) => ({ scenario: s, bad: s.runs.filter((r) => r.status !== "pass") }))
    .filter((x) => x.bad.length > 0)
    .sort((a, b) => (a.scenario.pass_rate ?? 0) - (b.scenario.pass_rate ?? 0));
}

function Results({ data }: { data: EvalResults }) {
  const s = data.summary;
  const failures = failing(data.scenarios);
  const workflows = Object.entries(s.task_success_by_workflow);
  return (
    <div className="stack" style={{ gap: 18 }}>
      <section className="card">
        <div className="card-title">
          <div>
            <h2>{data.name}</h2>
            <div className="muted small">
              {when(data.started_at)} · {data.scenarios.length} scenarios × {data.runs_per_scenario} runs · {Math.round(data.duration_s / 60)} min
            </div>
          </div>
          {data.stopped_early ? <span className="badge warn">stopped early</span> : <span className="badge ok">complete</span>}
        </div>
        <div className="row small muted" style={{ marginBottom: 14 }}>
          <span className="badge plain">agent {data.agent_model}</span>
          <span className="badge plain">customer {data.customer_model}</span>
          <span className="badge plain">judge {data.judge_model}</span>
          <span className="badge plain">prompt {data.prompt}</span>
          <span className="badge plain">policy {data.policy_context_mode}</span>
        </div>
        {data.stopped_early && <div className="notice handover" style={{ marginTop: 0, marginBottom: 14 }}>{data.stopped_early}</div>}
        <div className="metrics">
          <Metric label="Task success" value={percent(s.task_success_rate)} raw={s.task_success_rate} metric="task_success_rate" />
          <Metric label="Tool-call accuracy" value={percent(s.tool_call_accuracy)} raw={s.tool_call_accuracy} metric="tool_call_accuracy" />
          <Metric label="Policy compliance" value={percent(s.policy_compliance)} raw={s.policy_compliance} metric="policy_compliance" />
          <Metric label="Unauthorized refunds" value={count(s.unauthorized_refunds)} raw={s.unauthorized_refunds} metric="unauthorized_refunds" />
          <Metric label="Escalation correctness" value={percent(s.escalation_correctness)} raw={s.escalation_correctness} metric="escalation_correctness" />
          <Metric label="Scenarios passing every run" value={percent(s.scenarios_passing_every_run)} />
          <Metric label="Runs scored / errored" value={`${s.scored_runs} / ${s.errored_runs}`} />
          <Metric label="Average turns" value={s.average_turns?.toFixed(1) ?? "–"} />
          <Metric label="Agent cost per conversation" value={usd(s.cost_per_conversation_usd.agent)} />
          <Metric label="Agent tokens per conversation" value={count(s.tokens_per_conversation.agent)} />
        </div>
      </section>

      <div className="split call">
        <section className="card">
          <div className="card-title"><h3>Task success by workflow</h3></div>
          <div className="stack">
            {workflows.length === 0 && <div className="muted small">No scored runs.</div>}
            {workflows.map(([workflow, rate]) => (
              <div key={workflow} className="bar-row">
                <span>{title(workflow)}</span>
                <div className="bar"><span style={{ width: percent(rate ?? 0) }} /></div>
                <b>{percent(rate)}</b>
              </div>
            ))}
          </div>
          {s.judged_runs > 0 && (
            <>
              <div className="card-title" style={{ marginTop: 20 }}><h3>Judge ({s.judged_runs} runs)</h3></div>
              <div className="stack">
                {Object.entries(s.judge_pass_rate).map(([criterion, rate]) => (
                  <div key={criterion} className="bar-row">
                    <span>{title(criterion)}</span>
                    <div className="bar"><span style={{ width: percent(rate ?? 0) }} /></div>
                    <b>{percent(rate)}</b>
                  </div>
                ))}
              </div>
            </>
          )}
        </section>

        <section className="card">
          <div className="card-title">
            <h3>Failing scenarios</h3>
            <span className={`badge ${failures.length ? "bad" : "ok"}`}>{failures.length} of {data.scenarios.length}</span>
          </div>
          {failures.length === 0 && <div className="empty">Every scenario passed every run. ✿</div>}
          <div className="stack">
            {failures.map(({ scenario, bad }) => (
              <details key={scenario.id} className="failure">
                <summary>
                  <span className="stack" style={{ gap: 0 }}>
                    <b>{scenario.title}</b>
                    <span className="muted small">{scenario.id} · {title(scenario.workflow)}</span>
                  </span>
                  <span className="badge bad">passed {percent(scenario.pass_rate)}</span>
                </summary>
                {bad.map((run) => <FailedRun key={run.run_index} run={run} />)}
              </details>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}

function FailedRun({ run }: { run: EvalRun }) {
  const failed = run.checks.filter((c) => !c.ok);
  const judged = run.judge ? Object.entries(run.judge.criteria).filter(([, v]) => !v.pass) : [];
  return (
    <div className="stack" style={{ gap: 6, paddingBottom: 12 }}>
      <div className="row small">
        <span className={`badge ${run.status === "error" ? "warn" : "bad"}`}>run {run.run_index + 1}: {run.status}</span>
        <span className="muted">{run.turns} turns · seed {run.seed}</span>
      </div>
      {run.error && <div className="notice error" style={{ marginTop: 0 }}>{run.error}</div>}
      {failed.map((c) => (
        <div key={c.name} className="check">
          <span className="badge bad">{c.category}</span>
          <span><b>{c.name}</b> <span className="detail">— {c.detail}</span></span>
        </div>
      ))}
      {judged.map(([criterion, verdict]) => (
        <div key={criterion} className="check">
          <span className="badge warn">judge</span>
          <span><b>{title(criterion)}</b> <span className="detail">— {verdict.reason}</span></span>
        </div>
      ))}
      {run.transcript.length > 0 && (
        <details>
          <summary className="small muted" style={{ cursor: "pointer" }}>Transcript ({run.transcript.length} lines)</summary>
          <div className="mini-transcript">
            {run.transcript.map((line, i) => (
              <div key={i}><b>{line.role === "customer" ? "Customer" : "Agent"}:</b> {line.text}</div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
