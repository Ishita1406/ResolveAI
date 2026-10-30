import { describe, expect, it } from "vitest";
import type { VoiceEvent } from "../api/types";
import { type CallState, callReducer, initialCallState } from "./callState";

function play(...events: VoiceEvent[]): CallState {
  let state = callReducer(initialCallState, { kind: "connecting" });
  for (const event of events) state = callReducer(state, { kind: "event", event });
  return state;
}

describe("call state", () => {
  it("shows partial text live, then the final line, then the streamed reply", () => {
    let state = play({ type: "listening", session_id: "s1" }, { type: "speech_start" }, { type: "partial", text: "where is" });
    expect(state.phase).toBe("hearing");
    expect(state.partial).toBe("where is");
    state = play(
      { type: "listening", session_id: "s1" },
      { type: "partial", text: "where is my" },
      { type: "user", text: "where is my order" },
      { type: "token", text: "It's " },
      { type: "token", text: "on its way." },
    );
    expect(state.partial).toBe("");
    expect(state.phase).toBe("thinking");
    expect(state.lines.map((l) => [l.role, l.text, l.final])).toEqual([
      ["customer", "where is my order", true],
      ["agent", "It's on its way.", false],
    ]);
  });

  it("follows the agent's audio for the speaking indicator", () => {
    let state = play({ type: "listening", session_id: "s" }, { type: "user", text: "hi" });
    state = callReducer(state, { kind: "event", event: { type: "agent_audio", playing: true } });
    expect(state.phase).toBe("speaking");
    state = callReducer(state, { kind: "event", event: { type: "agent_audio", playing: false } });
    expect(state.phase).toBe("listening");
  });

  it("puts a proposed action next to the read-back and updates the same card", () => {
    const proposal: VoiceEvent = {
      type: "action", action_id: "a1", tool: "issue_refund", state: "awaiting_confirmation",
      summary: "Refund ₹3,499 for VX-8015", amount_display: "₹3,499",
    };
    let state = play(
      { type: "user", text: "refund please" },
      { type: "token", text: "I can refund ₹3,499. Shall I go ahead?" },
      proposal,
      { type: "done", reply: "", outcome: "completed", iterations: 1, usage: {} },
      { type: "user", text: "yes" },
      { ...proposal, state: "executed" },
    );
    const readBack = state.lines.find((l) => l.role === "agent");
    expect(readBack?.actionIds).toEqual(["a1"]);
    expect(state.actions.a1?.state).toBe("executed");
    expect(state.lines.filter((l) => l.actionIds.length).length).toBe(1); // one card, updated
  });

  it("marks an interrupted reply and ignores events after the call ends", () => {
    let state = play({ type: "user", text: "hi" }, { type: "token", text: "Hello there" }, { type: "barge_in" });
    expect(state.lines.at(-1)?.interrupted).toBe(true);
    expect(state.phase).toBe("hearing");
    state = callReducer(state, { kind: "ended" });
    state = callReducer(state, { kind: "event", event: { type: "user", text: "late" } });
    expect(state.phase).toBe("ended");
    expect(state.lines.some((l) => l.text === "late")).toBe(false);
  });

  it("shows a handover with its reference", () => {
    const state = play({
      type: "handed_over", reference: "TCK-000031", reason: "stt failed twice", text: "…", spoken: true,
    });
    expect(state.handedOver?.reference).toBe("TCK-000031");
    expect(state.lines.at(-1)?.text).toContain("TCK-000031");
  });
});
