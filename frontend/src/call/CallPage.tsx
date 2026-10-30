import { useEffect, useRef } from "react";
import { ms } from "../lib/format";
import { ActionCard } from "./ActionCard";
import type { Phase } from "./callState";
import { type MicPermission, useVoiceCall } from "./useVoiceCall";

const PHASE_TEXT: Record<Phase, [string, string]> = {
  idle: ["Ready when you are", "Start a call and say what you need."],
  connecting: ["Connecting…", "Setting up a secure audio line."],
  listening: ["Listening", "Go ahead, I'm all ears."],
  hearing: ["Hearing you", "Take your time."],
  thinking: ["One moment", "Looking into it."],
  speaking: ["Speaking", "Interrupt any time by talking."],
  ended: ["Call ended", "Start again whenever you like."],
};

const PERMISSION_TEXT: Partial<Record<MicPermission, string>> = {
  prompt: "Your browser will ask for the microphone when the call starts.",
  denied: "Microphone access is blocked for this site. Allow it in the browser's site settings.",
  unsupported: "This browser can't use the microphone here. Use a recent browser on https or localhost.",
};

export function CallPage() {
  const { state, permission, audioRef, start, end } = useVoiceCall();
  const scroller = useRef<HTMLDivElement | null>(null);
  const live = state.phase !== "idle" && state.phase !== "ended";
  const [label, hint] = PHASE_TEXT[state.phase];

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight });
  }, [state.lines, state.partial]);

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Talk to support</h1>
          <p>Ask about an order, a missing parcel, a damage or a refund. Anything that moves money is read back for your yes first.</p>
        </div>
      </div>

      <div className="split call">
        <section className="card" aria-label="Call">
          <div className="orb-wrap">
            <div className="orb" data-phase={state.phase} aria-hidden="true" />
          </div>
          <div className="phase-label" aria-live="polite">{label}</div>
          <div className="phase-hint">{hint}</div>

          <div className="call-controls">
            {live ? (
              <button className="btn end" onClick={end}>End call</button>
            ) : (
              <button className="btn" onClick={() => void start()} disabled={permission === "unsupported"}>
                {state.phase === "ended" ? "Call again" : "Start call"}
              </button>
            )}
          </div>

          <div className="stats">
            <span className={`badge ${permission === "granted" ? "ok" : permission === "denied" ? "bad" : "plain"}`}>
              Mic: {permission === "unknown" ? "not asked yet" : permission}
            </span>
            {state.firstAudioMs != null && (
              <span className="badge info">First audio {ms(state.firstAudioMs)}</span>
            )}
          </div>

          {!live && PERMISSION_TEXT[permission] && (
            <div className="notice info">{PERMISSION_TEXT[permission]}</div>
          )}
          {state.error && <div className="notice error" role="alert">{state.error}</div>}
          {state.handedOver && (
            <div className="notice handover">
              {state.handedOver.reference
                ? <>A colleague will follow up. Your reference is <b>{state.handedOver.reference}</b>.</>
                : "We couldn't continue this call. Please try again in a few minutes."}
            </div>
          )}
          <audio ref={audioRef} autoPlay />
        </section>

        <section className="card" aria-label="Transcript">
          <div className="card-title">
            <h2>Transcript</h2>
            {state.sessionId && <span className="muted small">session {state.sessionId.slice(0, 8)}</span>}
          </div>
          <div className="transcript" ref={scroller} aria-live="polite">
            {state.lines.length === 0 && !state.partial && (
              <div className="empty">
                <div style={{ fontSize: 28 }}>✿</div>
                <div>What you and the assistant say will appear here.</div>
              </div>
            )}
            {state.lines.map((line) =>
              line.role === "note" ? (
                <div key={line.id} className="note">{line.text}</div>
              ) : (
                <div key={line.id} className="stack" style={{ gap: 8 }}>
                  {(line.text || line.role === "customer") && (
                    <div
                      className={`bubble ${line.role}${line.muted ? " muted" : ""}${line.interrupted ? " interrupted" : ""}`}
                    >
                      <span className="who">{line.role === "customer" ? "You" : "Assistant"}</span>
                      {line.text}
                      {!line.final && line.role === "agent" && <span className="caret" />}
                    </div>
                  )}
                  {line.actionIds.map((id) => {
                    const card = state.actions[id];
                    return card ? <ActionCard key={id} card={card} /> : null;
                  })}
                </div>
              ),
            )}
            {state.partial && (
              <div className="bubble partial">
                <span className="who">You (live)</span>
                {state.partial}
              </div>
            )}
          </div>
        </section>
      </div>
    </>
  );
}
