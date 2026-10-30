# Latency: end of turn: after (fake STT/TTS)

- date: `2026-10-30T16:07:52+00:00`
- runs: `5`
- llm: `paced (first 3160 ms, tail 450 ms)`
- fast_model: `-`
- reasoning_effort: `none`
- llm_keepalive_s: `5.0`
- llm_prewarm: `False`
- tts_streaming: `sentence`
- stt_tts: `fake`
- files: `customer-01.wav customer-02.wav customer-03.wav`
- turns: 15 (15 with audio, 0 where a filler came first)

| ms | n | p50 | p95 | p99 | mean |
|---|---:|---:|---:|---:|---:|
| **time_to_first_audio_ms** | 15 | 3,833 | 3,860 | 3,860 | 2,768 |
| vad_ms | 15 | 608 | 608 | 608 | 608 |
| stt_ms | 15 | 0 | 0 | 0 | 0 |
| agent_first_text_ms | 15 | 3,162 | 3,187 | 3,188 | 2,117 |
| first_sentence_ms | 15 | 62 | 64 | 65 | 42 |
| tts_first_audio_ms | 15 | 1 | 2 | 2 | 1 |
| llm_first_token_ms | 10 | 3,161 | 3,162 | 3,162 | 3,161 |
| llm_ms | 15 | 3,633 | 3,641 | 3,643 | 2,424 |
| tool_ms | 15 | 4 | 24 | 24 | 8 |

Time to first audio by customer line:

| line | n | p50 | p95 |
|---|---:|---:|---:|
| Hi, where is my order? | 5 | 3,833 | 3,838 |
| My email is meera dot pillai at example dot com and my PIN i | 5 | 616 | 618 |
| It's order VX-8015. | 5 | 3,851 | 3,860 |
