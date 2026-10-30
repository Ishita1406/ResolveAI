# Latency: baseline

- date: `2026-10-30T15:56:06+00:00`
- runs: `5`
- llm: `paced (first 3160 ms, tail 450 ms)`
- fast_model: `-`
- reasoning_effort: `none`
- llm_keepalive_s: `5.0`
- llm_prewarm: `False`
- tts_streaming: `sentence`
- stt_tts: `cartesia`
- files: `customer-01.wav customer-02.wav customer-03.wav`
- turns: 15 (15 with audio, 0 where a filler came first)

| ms | n | p50 | p95 | p99 | mean |
|---|---:|---:|---:|---:|---:|
| **time_to_first_audio_ms** | 15 | 4,316 | 4,949 | 4,952 | 4,373 |
| vad_ms | 15 | 608 | 1,600 | 1,600 | 939 |
| stt_ms | 15 | 0 | 401 | 401 | 104 |
| agent_first_text_ms | 15 | 3,162 | 3,169 | 3,170 | 3,164 |
| first_sentence_ms | 15 | 63 | 65 | 65 | 63 |
| tts_first_audio_ms | 15 | 90 | 153 | 187 | 103 |
| llm_first_token_ms | 15 | 3,161 | 3,162 | 3,162 | 3,161 |
| llm_ms | 15 | 3,635 | 3,638 | 3,640 | 3,634 |
| tool_ms | 15 | 0 | 5 | 6 | 2 |

Time to first audio by customer line:

| line | n | p50 | p95 |
|---|---:|---:|---:|
| Hi, where is my order | 5 | 4,918 | 4,952 |
| My email is mira. pillai@example.com and my PIN is 302647. | 5 | 3,930 | 4,142 |
| It's order VX801-5. | 5 | 4,316 | 4,409 |
