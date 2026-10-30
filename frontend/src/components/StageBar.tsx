import type { TurnTimings } from "../api/types";
import { ms } from "../lib/format";

// The voice stages of time to first audio (backend/app/voice/session.py TurnMetrics.stages).
const STAGES = [
  ["vad_ms", "End of turn", "s-vad"],
  ["stt_ms", "Transcript", "s-stt"],
  ["agent_first_text_ms", "Agent to first words", "s-agent"],
  ["first_sentence_ms", "First sentence", "s-sentence"],
  ["tts_first_audio_ms", "Voice", "s-tts"],
] as const;

export function StageBar({ timings }: { timings: TurnTimings }) {
  const stages = STAGES.filter(([key]) => (timings[key] ?? 0) > 0);
  const total = stages.reduce((sum, [key]) => sum + (timings[key] ?? 0), 0);
  if (!total) return null;
  return (
    <div className="stack" style={{ gap: 6 }}>
      <div className="stagebar" role="img" aria-label={`Time to first audio ${ms(timings.time_to_first_audio_ms)}`}>
        {stages.map(([key, label, cls]) => (
          <span
            key={key}
            className={cls}
            style={{ width: `${(100 * (timings[key] ?? 0)) / total}%` }}
            title={`${label}: ${ms(timings[key])}`}
          />
        ))}
      </div>
      <div className="legend">
        {stages.map(([key, label, cls]) => (
          <span key={key}>
            <i className={cls} />
            {label} {ms(timings[key])}
          </span>
        ))}
      </div>
    </div>
  );
}
