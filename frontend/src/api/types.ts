// Shapes of what the backend sends. They mirror, field for field:
//   voice events     backend/app/voice/session.py (_emit) and app/agent/events.py
//   conversations    backend/app/observability/conversation.py (load, list_recent)
//   eval results     evals/harness/report.py (results_json), served by /admin/evals*
// The frontend only displays these; every decision is made by the backend.

// --- voice call --------------------------------------------------------------------------

export interface VoiceOffer {
  sdp: string;
  type: RTCSdpType;
}

export interface VoiceAnswer extends VoiceOffer {
  session_id: string;
}

export type ActionState =
  | "proposed"
  | "awaiting_confirmation"
  | "confirmed"
  | "executed"
  | "cancelled"
  | "expired"
  | "rejected";

export type VoiceEvent =
  | { type: "listening"; session_id: string }
  | { type: "speech_start" }
  | { type: "partial"; text: string }
  | { type: "user"; text: string }
  | { type: "token"; text: string }
  | { type: "tool"; name: string; status: "started" | "finished"; ok?: boolean | null }
  | {
      type: "action";
      action_id: string;
      tool: string;
      state: ActionState;
      summary: string;
      amount_display?: string | null;
    }
  | { type: "escalated"; reference: string; reason: string }
  | { type: "error"; code: string; message?: string; retrying?: boolean }
  | {
      type: "done";
      reply: string;
      outcome: "completed" | "escalated" | "failed";
      iterations: number;
      usage: Record<string, number>;
      turn_id?: string | null;
    }
  | { type: "barge_in" }
  | { type: "filler"; text: string }
  | { type: "say"; text: string }
  | { type: "first_audio"; ms: number; source?: "reply" | "filler" }
  | { type: "agent_audio"; playing: boolean }
  | { type: "recovered"; what: "stt" | "tts" }
  | {
      type: "handed_over";
      reference: string | null;
      reason: string;
      text: string;
      spoken: boolean;
    };

// --- conversations (replay) --------------------------------------------------------------

export interface ConversationSummary {
  id: number;
  channel: string;
  started_at: string;
  ended_at: string | null;
  customer: string | null;
}

export interface ToolCallEntry {
  tool?: string;
  phase?: "call" | "executed" | "cancelled" | "system";
  arguments?: Record<string, unknown>;
  ok?: boolean;
  status?: string | null;
  error?: string | null;
  source?: string;
  raw?: string; // a line that wasn't JSON
}

export interface PolicyDecision {
  at: string;
  stage: "proposal" | "execution";
  tool: string;
  action_id: string;
  verdict: "approve" | "reject" | "require_human";
  rule: string;
  reason: string | null;
  message: string;
  amount_paise: number | null;
  policy_version: string;
}

export interface TurnTimings {
  turn_id: string;
  outcome: string | null;
  model: string | null;
  prompt_version: string | null;
  llm_calls: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  cached_tokens: number | null;
  cost_usd: number | null;
  llm_first_token_ms: number | null;
  llm_ms: number | null;
  tool_calls: number | null;
  tool_ms: number | null;
  agent_ms: number | null;
  vad_ms: number | null;
  stt_ms: number | null;
  agent_first_text_ms: number | null;
  first_sentence_ms: number | null;
  tts_first_audio_ms: number | null;
  time_to_first_audio_ms: number | null;
  filler: boolean | null;
  interrupted: boolean | null;
}

export interface ReplayTurn {
  index: number;
  at: string;
  customer: string | null;
  tool_calls: ToolCallEntry[];
  assistant: string | null;
  policy_decisions: PolicyDecision[];
  timings: TurnTimings | null;
}

export interface ConversationReplay {
  conversation: {
    id: number;
    channel: string;
    started_at: string;
    verified_at: string | null;
    ended_at: string | null;
    customer: { id: number; name: string } | null;
  };
  trace_id: string;
  turns: ReplayTurn[];
  tickets: {
    reference: string;
    category: string;
    status: string;
    summary: string;
    created_at: string;
  }[];
  totals: {
    turns_measured: number;
    llm_calls: number;
    prompt_tokens: number;
    completion_tokens: number;
    cost_usd: number | null;
    time_to_first_audio_ms: { min: number; max: number } | null;
  };
}

// --- eval results ------------------------------------------------------------------------

export interface EvalRunListing {
  name: string;
  run_id: string | null;
  started_at: string | null;
  agent_model: string | null;
  runs: number | null;
  task_success_rate: number | null;
}

export interface EvalCheck {
  name: string;
  category: string;
  ok: boolean;
  detail: string;
}

export interface EvalTranscriptLine {
  role: "customer" | "agent";
  text: string;
  tools?: unknown[];
  actions?: unknown[];
  outcome?: string;
}

export interface EvalRun {
  scenario_id: string;
  run_index: number;
  seed: number;
  status: "pass" | "fail" | "error";
  checks: EvalCheck[];
  transcript: EvalTranscriptLine[];
  tool_log: ToolCallEntry[];
  turns: number;
  finished: boolean;
  judge: { criteria: Record<string, { pass: boolean; reason: string }> } | null;
  error: string | null;
  duration_s: number;
}

export interface EvalScenario {
  id: string;
  title: string;
  workflow: string;
  tags: string[];
  pass_rate: number | null;
  runs: EvalRun[];
}

export interface EvalSummary {
  runs: number;
  scored_runs: number;
  errored_runs: number;
  task_success_rate: number | null;
  tool_call_accuracy: number | null;
  policy_compliance: number | null;
  escalation_correctness: number | null;
  unauthorized_refunds: number;
  scenarios_passing_every_run: number | null;
  average_turns: number | null;
  cost_per_conversation_usd: { agent: number | null; simulator_and_judge: number | null };
  tokens_per_conversation: { agent: number; customer: number; judge: number };
  judge_pass_rate: Record<string, number | null>;
  judged_runs: number;
  task_success_by_workflow: Record<string, number | null>;
}

export interface EvalResults {
  name: string;
  run_id: string;
  started_at: string;
  duration_s: number;
  runs_per_scenario: number;
  concurrency: number;
  seed: number;
  policy_context_mode: string;
  prompt: string;
  agent_model: string;
  customer_model: string;
  judge_model: string;
  stopped_early: string | null;
  summary: EvalSummary;
  scenarios: EvalScenario[];
}
