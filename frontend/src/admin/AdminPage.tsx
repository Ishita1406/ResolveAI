import { api } from "../api/client";
import type { ConversationReplay, PolicyDecision, ReplayTurn, ToolCallEntry } from "../api/types";
import { LoadError } from "../components/LoadError";
import { StageBar } from "../components/StageBar";
import { count, ms, title, usd, when } from "../lib/format";
import { useLoad } from "../lib/useLoad";

const VERDICT: Record<PolicyDecision["verdict"], string> = {
  approve: "ok",
  reject: "bad",
  require_human: "warn",
};

export function AdminPage({ selected, onSelect }: { selected: number | null; onSelect: (id: number) => void }) {
  const list = useLoad(() => api.conversations(50), "list");
  const replay = useLoad(selected != null ? () => api.conversation(selected) : null, selected);

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Conversations</h1>
          <p>Every call and chat, turn by turn: what was said, the tools the agent used, the policy engine's decisions, and how long each stage took.</p>
        </div>
        <button className="btn ghost small" onClick={list.reload}>Refresh</button>
      </div>

      <div className="split admin">
        <section className="card" aria-label="Recent conversations">
          <div className="card-title"><h2>Recent</h2></div>
          {list.error && <LoadError error={list.error} retry={list.reload} />}
          {list.loading && !list.data && <div className="stack"><div className="skeleton" /><div className="skeleton" /></div>}
          {list.data?.length === 0 && <div className="empty">No conversations yet. Start a call first.</div>}
          <div className="list">
            {list.data?.map((c) => (
              <button
                key={c.id}
                className="list-item"
                aria-current={c.id === selected}
                onClick={() => onSelect(c.id)}
              >
                <span className="stack" style={{ gap: 0 }}>
                  <span className="id">#{c.id} · {c.customer ?? "Unverified caller"}</span>
                  <span className="muted small">{when(c.started_at)}</span>
                </span>
                <span className={`badge ${c.channel === "voice" ? "info" : "plain"}`}>{c.channel}</span>
              </button>
            ))}
          </div>
        </section>

        <section className="card" aria-label="Replay">
          {selected == null && <div className="empty">Pick a conversation to replay it.</div>}
          {replay.error && <LoadError error={replay.error} retry={replay.reload} />}
          {replay.loading && <div className="stack"><div className="skeleton" /><div className="skeleton" /><div className="skeleton" /></div>}
          {replay.data && !replay.loading && <Replay data={replay.data} />}
        </section>
      </div>
    </>
  );
}

function Replay({ data }: { data: ConversationReplay }) {
  const c = data.conversation;
  const t = data.totals;
  return (
    <div className="stack" style={{ gap: 14 }}>
      <div className="card-title" style={{ marginBottom: 0 }}>
        <div>
          <h2>#{c.id} · {c.customer?.name ?? "Unverified caller"}</h2>
          <div className="muted small">
            {title(c.channel)} · {when(c.started_at)}{c.ended_at ? ` → ${when(c.ended_at)}` : " · still open"}
          </div>
        </div>
        {c.verified_at ? <span className="badge ok">Verified</span> : <span className="badge plain">Not verified</span>}
      </div>
      {t.turns_measured > 0 ? (
        <div className="row">
          <span className="badge plain">{count(t.llm_calls)} model calls</span>
          <span className="badge plain">{count(t.prompt_tokens + t.completion_tokens)} tokens</span>
          {t.cost_usd != null && <span className="badge plain">{usd(t.cost_usd)}</span>}
          {t.time_to_first_audio_ms && (
            <span className="badge info">First audio {ms(t.time_to_first_audio_ms.min)} – {ms(t.time_to_first_audio_ms.max)}</span>
          )}
        </div>
      ) : (
        <div className="muted small">No timings recorded for this conversation (it predates them).</div>
      )}
      <div className="muted small">trace <code>{data.trace_id}</code></div>
      {data.turns.length === 0 && <div className="empty">This conversation has no turns yet.</div>}
      {data.turns.map((turn) => <Turn key={turn.index} turn={turn} />)}
      {data.tickets.map((ticket) => (
        <div key={ticket.reference} className="notice handover">
          <b>{ticket.reference}</b> · {title(ticket.category)} · {ticket.status} — {ticket.summary}
        </div>
      ))}
    </div>
  );
}

function ToolChip({ call }: { call: ToolCallEntry }) {
  if (call.raw) return <span className="chip">{call.raw}</span>;
  const args = call.arguments && Object.keys(call.arguments).length ? JSON.stringify(call.arguments) : "";
  return (
    <span className={`chip${call.ok ? "" : " failed"}`} title={args}>
      <b>{call.tool}</b>
      <span>{call.phase}{call.source ? ` · ${call.source}` : ""}</span>
      <span>{call.ok ? "✓" : `✕ ${call.error ?? call.status ?? ""}`}</span>
    </span>
  );
}

function Turn({ turn }: { turn: ReplayTurn }) {
  const t = turn.timings;
  return (
    <article className="turn">
      <div className="turn-index">TURN {turn.index}</div>
      {turn.customer && (
        <div className="bubble customer" style={{ maxWidth: "100%" }}>
          <span className="who">Customer</span>
          {turn.customer}
        </div>
      )}
      {turn.tool_calls.length > 0 && (
        <div className="row">{turn.tool_calls.map((call, i) => <ToolChip key={i} call={call} />)}</div>
      )}
      {turn.policy_decisions.map((d, i) => (
        <div key={i} className="decision">
          <span className={`badge ${VERDICT[d.verdict]}`}>{d.verdict.replace("_", " ")}</span>
          <span>
            <b>{d.stage}</b> of {d.tool} · <code>{d.rule}</code> — {d.message}
          </span>
        </div>
      ))}
      {turn.assistant && (
        <div className="bubble agent" style={{ maxWidth: "100%" }}>
          <span className="who">Assistant</span>
          {turn.assistant}
        </div>
      )}
      {t && (
        <>
          <div className="timings">
            {t.time_to_first_audio_ms != null && <span>First audio <b>{ms(t.time_to_first_audio_ms)}</b></span>}
            <span>LLM first token <b>{ms(t.llm_first_token_ms)}</b></span>
            <span>LLM <b>{ms(t.llm_ms)}</b> · {count(t.llm_calls)} calls</span>
            <span>Tools <b>{ms(t.tool_ms)}</b> · {count(t.tool_calls)}</span>
            <span>Tokens <b>{count((t.prompt_tokens ?? 0) + (t.completion_tokens ?? 0))}</b></span>
            {t.cost_usd != null && <span>Cost <b>{usd(t.cost_usd)}</b></span>}
            {t.filler && <span className="badge plain">filler</span>}
            {t.interrupted && <span className="badge warn">interrupted</span>}
          </div>
          <StageBar timings={t} />
        </>
      )}
    </article>
  );
}
