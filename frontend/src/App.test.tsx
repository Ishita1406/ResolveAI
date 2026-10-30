import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { parseRoute } from "./App";
import { EvalsPage } from "./evals/EvalsPage";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("routes", () => {
  it("parses hash routes", () => {
    expect(parseRoute("")).toEqual({ page: "call" });
    expect(parseRoute("#/conversations/42")).toEqual({ page: "conversations", id: 42 });
    expect(parseRoute("#/conversations/abc")).toEqual({ page: "conversations", id: null });
    expect(parseRoute("#/evals/live-smoke")).toEqual({ page: "evals", run: "live-smoke" });
  });
});

const results = {
  name: "smoke", run_id: "r1", started_at: "2026-10-30T10:00:00Z", duration_s: 600, runs_per_scenario: 1,
  concurrency: 1, seed: 1, policy_context_mode: "retrieval", prompt: "v3+abc", agent_model: "gemini",
  customer_model: "gemini-lite", judge_model: "disabled", stopped_early: null,
  summary: {
    runs: 2, scored_runs: 2, errored_runs: 0, task_success_rate: 0.5, tool_call_accuracy: 1,
    policy_compliance: 1, escalation_correctness: 1, unauthorized_refunds: 0, scenarios_passing_every_run: 0.5,
    average_turns: 3, cost_per_conversation_usd: { agent: 0.001, simulator_and_judge: 0 },
    tokens_per_conversation: { agent: 9000, customer: 800, judge: 0 }, judge_pass_rate: {}, judged_runs: 0,
    task_success_by_workflow: { missing_delivery: 0, order_tracking: 1 },
  },
  scenarios: [
    { id: "track-ok", title: "Where is my order", workflow: "order_tracking", tags: [], pass_rate: 1,
      runs: [{ scenario_id: "track-ok", run_index: 0, seed: 1, status: "pass", checks: [], transcript: [], tool_log: [], turns: 2, finished: true, judge: null, error: null, duration_s: 10 }] },
    { id: "missing-refund", title: "Overdue parcel refund", workflow: "missing_delivery", tags: [], pass_rate: 0,
      runs: [{ scenario_id: "missing-refund", run_index: 0, seed: 2, status: "fail", turns: 4, finished: true, judge: null, error: null, duration_s: 30,
        checks: [{ name: "end_state.refunds", category: "end_state", ok: false, detail: "expected a refund, got []" }],
        transcript: [{ role: "customer", text: "my parcel never came" }], tool_log: [] }] },
  ],
};

describe("eval dashboard", () => {
  it("shows the metrics against targets and only the failing scenarios", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string) =>
      new Response(JSON.stringify(url.endsWith("/latest") ? results : []), { status: 200 })));
    render(<EvalsPage run={null} onRun={() => undefined} />);
    expect(await screen.findByText("Overdue parcel refund")).toBeTruthy();
    expect(screen.queryByText("Where is my order")).toBeNull(); // passed: not listed
    expect(screen.getByText("end_state.refunds")).toBeTruthy();
    const success = screen.getByText("Task success").closest(".metric");
    expect(success?.className).toContain("bad"); // 50% < 85%
    expect(screen.getByText("Policy compliance").closest(".metric")?.className).toContain("ok");
  });
});
