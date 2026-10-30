# Latency: end of turn: before (fake STT/TTS)

- date: `2026-10-30T16:03:35+00:00`
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
| **time_to_first_audio_ms** | 15 | 3,852 | 4,829 | 4,831 | 3,099 |
| vad_ms | 15 | 608 | 1,600 | 1,600 | 939 |
| stt_ms | 15 | 0 | 0 | 0 | 0 |
| agent_first_text_ms | 15 | 3,162 | 3,182 | 3,182 | 2,116 |
| first_sentence_ms | 15 | 63 | 65 | 65 | 43 |
| tts_first_audio_ms | 15 | 1 | 2 | 2 | 1 |
| llm_first_token_ms | 10 | 3,161 | 3,162 | 3,162 | 3,161 |
| llm_ms | 15 | 3,634 | 3,643 | 3,645 | 2,425 |
| tool_ms | 15 | 5 | 18 | 19 | 6 |

Time to first audio by customer line:

| line | n | p50 | p95 |
|---|---:|---:|---:|
| Hi, where is my order? | 5 | 4,828 | 4,830 |
| My email is meera dot pillai at example dot com and my PIN i | 5 | 618 | 619 |
| It's order VX-8015. | 5 | 3,852 | 3,855 |
