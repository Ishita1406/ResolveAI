import { type ActionCard as Card, actionStateLabel, toolLabel } from "./callState";

const BADGE: Record<string, string> = {
  awaiting_confirmation: "warn",
  executed: "ok",
  confirmed: "info",
  cancelled: "plain",
  expired: "plain",
  rejected: "bad",
};

/** A proposed action, shown next to the agent's spoken read-back. The words, amount and
 * state are the backend's; the card only shows them. */
export function ActionCard({ card }: { card: Card }) {
  return (
    <div className="action-card" data-state={card.state} role="status" aria-live="polite">
      <div className="kind">{toolLabel(card.tool)}</div>
      <div className="summary">{card.summary}</div>
      {card.amount && <div className="amount">{card.amount}</div>}
      <div className="foot">
        <span className={`badge ${BADGE[card.state] ?? "info"}`}>{actionStateLabel(card.state)}</span>
        {card.state === "awaiting_confirmation" && (
          <span className="hint">Say “yes” to go ahead, or “no” to cancel.</span>
        )}
      </div>
    </div>
  );
}
