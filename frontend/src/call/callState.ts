// What the call screen shows, folded from the server's events. Display state only: whether
// an action is allowed, what it costs and whether it ran all come from the backend.

import type { ActionState, VoiceEvent } from "../api/types";

export type Phase =
  | "idle" // no call
  | "connecting"
  | "listening" // waiting for the customer
  | "hearing" // the customer is speaking
  | "thinking" // the turn is with the agent, nothing audible yet
  | "speaking" // the agent is being heard
  | "ended";

export interface ActionCard {
  id: string;
  tool: string;
  state: ActionState;
  summary: string;
  amount: string | null;
}

export interface Line {
  id: number;
  role: "customer" | "agent" | "note";
  text: string;
  final: boolean;
  interrupted?: boolean;
  actionIds: string[]; // proposals made while this line was being spoken
  muted?: boolean; // a filler or fixed phrase, not the agent's answer
}

export interface CallState {
  phase: Phase;
  sessionId: string | null;
  lines: Line[];
  partial: string;
  actions: Record<string, ActionCard>;
  firstAudioMs: number | null;
  handedOver: { reference: string | null; text: string } | null;
  error: string | null;
  nextId: number;
}

export type CallAction =
  | { kind: "connecting" }
  | { kind: "connected"; sessionId: string }
  | { kind: "failed"; message: string }
  | { kind: "ended" }
  | { kind: "event"; event: VoiceEvent };

export const initialCallState: CallState = {
  phase: "idle",
  sessionId: null,
  lines: [],
  partial: "",
  actions: {},
  firstAudioMs: null,
  handedOver: null,
  error: null,
  nextId: 1,
};

const TOOL_LABELS: Record<string, string> = {
  issue_refund: "Refund",
  create_return: "Return",
  reship_order: "Replacement",
  create_ticket: "Support ticket",
  escalate_to_human: "Hand over to a colleague",
};

export function toolLabel(tool: string): string {
  return TOOL_LABELS[tool] ?? tool.replaceAll("_", " ");
}

const STATE_LABELS: Record<ActionState, string> = {
  proposed: "Proposed",
  awaiting_confirmation: "Waiting for your yes",
  confirmed: "Confirmed",
  executed: "Done",
  cancelled: "Cancelled",
  expired: "Expired",
  rejected: "Not allowed",
};

export function actionStateLabel(state: ActionState): string {
  return STATE_LABELS[state] ?? state;
}

function lastAgentLine(lines: Line[]): Line | undefined {
  const last = lines.at(-1);
  return last && last.role === "agent" && !last.final ? last : undefined;
}

function push(state: CallState, line: Omit<Line, "id">): CallState {
  return { ...state, lines: [...state.lines, { ...line, id: state.nextId }], nextId: state.nextId + 1 };
}

function updateLast(state: CallState, update: (line: Line) => Line): CallState {
  const lines = state.lines.slice();
  const last = lines.at(-1);
  if (last) lines[lines.length - 1] = update(last);
  return { ...state, lines };
}

/** The agent's open reply, created on first use. */
function withAgentLine(state: CallState): CallState {
  return lastAgentLine(state.lines)
    ? state
    : push(state, { role: "agent", text: "", final: false, actionIds: [] });
}

function closeAgentLine(state: CallState): CallState {
  return lastAgentLine(state.lines) ? updateLast(state, (l) => ({ ...l, final: true })) : state;
}

function onEvent(state: CallState, event: VoiceEvent): CallState {
  switch (event.type) {
    case "listening":
      return { ...state, phase: "listening", sessionId: event.session_id };
    case "speech_start":
      return { ...state, phase: "hearing" };
    case "partial":
      return { ...state, phase: "hearing", partial: event.text };
    case "user": {
      const closed = closeAgentLine({ ...state, partial: "" });
      return {
        ...push(closed, { role: "customer", text: event.text, final: true, actionIds: [] }),
        phase: "thinking",
      };
    }
    case "token": {
      const open = withAgentLine(state);
      return updateLast(open, (l) => ({ ...l, text: l.text + event.text }));
    }
    case "filler":
      return push(state, { role: "agent", text: event.text, final: true, actionIds: [], muted: true });
    case "say":
      return state; // fixed phrases (apologies) are announced by their own events
    case "action": {
      const card: ActionCard = {
        id: event.action_id,
        tool: event.tool,
        state: event.state,
        summary: event.summary,
        amount: event.amount_display ?? null,
      };
      const known = event.action_id in state.actions;
      const next = { ...state, actions: { ...state.actions, [card.id]: card } };
      if (known) return next; // a later state of a card already shown
      const open = withAgentLine(next);
      return updateLast(open, (l) => ({ ...l, actionIds: [...l.actionIds, card.id] }));
    }
    case "agent_audio":
      if (event.playing) return { ...state, phase: "speaking" };
      return state.phase === "speaking" ? { ...state, phase: "listening" } : state;
    case "barge_in": {
      const cut = lastAgentLine(state.lines)
        ? updateLast(state, (l) => ({ ...l, final: true, interrupted: true }))
        : state;
      return { ...cut, phase: "hearing" };
    }
    case "first_audio":
      return { ...state, firstAudioMs: event.ms };
    case "done":
      return closeAgentLine(state);
    case "escalated":
      return push(state, {
        role: "note",
        text: `Handed over to a colleague — reference ${event.reference}`,
        final: true,
        actionIds: [],
      });
    case "handed_over":
      return {
        ...push(closeAgentLine(state), {
          role: "note",
          text: event.reference
            ? `This call was passed to a colleague — reference ${event.reference}`
            : "This call couldn't continue. Please try again in a few minutes.",
          final: true,
          actionIds: [],
        }),
        handedOver: { reference: event.reference, text: event.text },
      };
    case "recovered":
      return push(state, {
        role: "note",
        text: event.what === "stt" ? "Reconnected to speech recognition." : "Audio reconnected.",
        final: true,
        actionIds: [],
      });
    case "error":
      return push(state, {
        role: "note",
        text: event.message ?? `Something went wrong (${event.code}).`,
        final: true,
        actionIds: [],
      });
    case "tool":
      return state; // shown through fillers and the reply; the replay view lists tools
  }
}

export function callReducer(state: CallState, action: CallAction): CallState {
  switch (action.kind) {
    case "connecting":
      return { ...initialCallState, phase: "connecting" };
    case "connected":
      return { ...state, sessionId: action.sessionId };
    case "failed":
      return { ...closeAgentLine(state), phase: "ended", error: action.message, partial: "" };
    case "ended":
      return { ...closeAgentLine(state), phase: "ended", partial: "" };
    case "event":
      return state.phase === "ended" ? state : onEvent(state, action.event);
  }
}
