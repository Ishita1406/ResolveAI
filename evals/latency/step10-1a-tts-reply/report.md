# Latency: before sentence streaming (whole-reply TTS)

- date: `2026-10-30T15:59:14+00:00`
- runs: `5`
- llm: `paced (first 3160 ms, tail 450 ms)`
- fast_model: `-`
- reasoning_effort: `none`
- llm_keepalive_s: `5.0`
- llm_prewarm: `False`
- tts_streaming: `reply`
- stt_tts: `cartesia`
- files: `customer-01.wav customer-02.wav customer-03.wav`
- turns: 8 (8 with audio, 0 where a filler came first)

| ms | n | p50 | p95 | p99 | mean |
|---|---:|---:|---:|---:|---:|
| **time_to_first_audio_ms** | 8 | 4,765 | 5,412 | 5,440 | 4,844 |
| vad_ms | 8 | 608 | 1,600 | 1,600 | 980 |
| stt_ms | 8 | 0 | 401 | 402 | 100 |
| agent_first_text_ms | 8 | 3,163 | 3,167 | 3,167 | 3,164 |
| first_sentence_ms | 8 | 488 | 494 | 494 | 489 |
| tts_first_audio_ms | 8 | 99 | 162 | 184 | 110 |
| llm_first_token_ms | 8 | 3,161 | 3,162 | 3,162 | 3,161 |
| llm_ms | 8 | 3,632 | 3,634 | 3,634 | 3,632 |
| tool_ms | 8 | 0 | 4 | 5 | 2 |

Time to first audio by customer line:

| line | n | p50 | p95 |
|---|---:|---:|---:|
| Hi, where is my order | 3 | 5,345 | 5,437 |
| My email is mira. pillai@example.com and my PIN is 302647. | 3 | 4,364 | 4,367 |
| It's order VX801-5. | 2 | 4,765 | 4,767 |
