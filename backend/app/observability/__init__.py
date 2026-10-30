"""Observability: structured logs, OpenTelemetry traces, per-turn metrics and reports.

- logs.py: JSON log lines carrying the conversation, session, turn and trace ids.
- tracing.py: one trace per conversation, one span per turn, child spans per stage;
  exported over OTLP (Langfuse or any collector).
- metrics.py: the turn_metrics table (time to first audio and the stage breakdown).
- latency.py: p50/p95/p99 report over N replays of recorded audio, or stored turns.
- conversation.py: a conversation's transcript, tool calls, policy decisions and timings.
"""
