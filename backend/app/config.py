from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

# Repo-root .env for local runs; in Docker, compose injects the variables directly.
ROOT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
POLICY_DOCS_DIR = Path(__file__).resolve().parents[2] / "docs" / "policies"
EVAL_RESULTS_DIR = Path(__file__).resolve().parents[2] / "evals" / "results"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_ENV_FILE, extra="ignore")

    app_env: str = "development"

    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_user: str
    postgres_password: SecretStr = SecretStr("")
    postgres_db: str = "resolveai"
    # Dropped and rebuilt by the test suite; must differ from postgres_db and contain "test".
    test_postgres_db: str = "resolveai_test"

    redis_url: str = "redis://127.0.0.1:6379/0"

    health_check_timeout_s: float = 2.0

    # LLM (Gemini, OpenAI-compatible chat completions). No key means the agent is disabled.
    gemini_api_key: SecretStr | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    gemini_model: str = "gemini-3.6-flash"
    llm_temperature: float = 0.2
    llm_request_timeout_s: float = 20.0
    # Includes thinking tokens on Gemini, so raise it if reasoning is turned up.
    llm_max_completion_tokens: int = 800
    # "none" turns thinking off (Gemini 3 Flash rejects "minimal"): fastest first token.
    llm_reasoning_effort: str | None = "none"
    # Best-effort reproducibility; the provider may still vary. None sends no seed.
    llm_seed: int | None = None
    # Client-side tokens-per-minute budget. None means no limiter.
    llm_tokens_per_minute: int | None = None
    # Latency (docs/progress/step-10-observability.md has the measurements):
    # How long an idle HTTPS connection to the LLM is kept for the next call. httpx's 5 s
    # default closed it between voice turns, so each turn paid for a new TLS connection.
    llm_keepalive_s: float = 60.0
    # Open the LLM connection when a voice call starts, so the first turn doesn't pay for it.
    llm_prewarm: bool = True
    # A smaller model for turns before verification (asking for details, policy questions;
    # no account tools exist yet). Unset: one model for everything.
    llm_fast_model: str | None = None
    # USD per million tokens by model, as JSON: {"gemini-3.6-flash": {"input": .., "output": ..}}.
    # Unpriced models get no cost from us (Langfuse prices known models itself).
    llm_prices: dict[str, dict[str, float]] = {}

    # Agent loop
    agent_prompt_version: str = "v3"
    agent_max_iterations: int = 6
    agent_turn_timeout_s: float = 30.0
    agent_history_turns: int = 10
    # Run verify_customer / get_order straight from the customer's message when it plainly
    # holds the details, instead of a model call each (app/agent/prefetch.py).
    agent_prefetch: bool = True
    session_ttl_s: int = 3600

    # Policy documents. "retrieval": the agent calls search_policies (hybrid search over
    # ingested chunks). "full_text": every policy document goes into the system prompt.
    policy_context_mode: Literal["retrieval", "full_text"] = "retrieval"
    policy_docs_dir: Path = POLICY_DOCS_DIR
    # Local ONNX model via fastembed; downloaded once into the cache dir. The vector column
    # is 384-dimensional, so a model of another size needs a migration.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: Path = Path.home() / ".cache" / "resolveai" / "fastembed"
    # BGE's instruction for short queries against passages; "" embeds the query as is.
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    # What search_policies uses; vector and text exist for comparison (see evals/retrieval/).
    retrieval_search_mode: Literal["hybrid", "vector", "text"] = "hybrid"
    retrieval_top_k: int = 3  # passages returned to the agent
    retrieval_candidates: int = 20  # per ranking, before fusion
    retrieval_rrf_k: int = 60  # reciprocal rank fusion constant

    # Voice (app/voice/). Cartesia Ink (streaming STT) and Sonic (streaming TTS), per the brief.
    cartesia_api_key: SecretStr | None = None
    cartesia_base_url: str = "wss://api.cartesia.ai"
    cartesia_version: str = "2026-08-14"
    cartesia_stt_model: str = "ink-2"
    cartesia_tts_model: str = "sonic-3.6"
    # The example voice from Cartesia's API reference; set CARTESIA_VOICE_ID to your own.
    cartesia_voice_id: str = "a0e99841-438c-4a64-b679-ae501e7d6091"
    cartesia_language: str = "en"
    tts_sample_rate: int = 24000
    # Silero VAD v5 (ONNX, run with onnxruntime), downloaded once and checked by hash.
    vad_model_url: str = (
        "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx"
    )
    vad_model_sha256: str = "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f"
    vad_cache_dir: Path = Path.home() / ".cache" / "resolveai" / "silero"
    vad_threshold: float = 0.5  # speech probability that starts speech
    vad_min_speech_ms: int = 160  # speech this long starts a turn (and interrupts the agent)
    # End of turn: silence needed after speech. Longer when the words so far sound unfinished
    # ("my order number is ...", a half-said PIN), so customers aren't cut off mid-sentence.
    end_of_turn_silence_ms: int = 600
    incomplete_turn_silence_ms: int = 1600
    max_turn_s: float = 30.0
    filler_after_ms: int = 500  # say a short filler if a tool runs longer than this
    # "sentence": speak each sentence as soon as the model finishes it. "reply": wait for the
    # whole reply (the slower baseline, kept for comparison).
    tts_streaming: Literal["sentence", "reply"] = "sentence"
    stt_connect_timeout_s: float = 5.0
    stt_final_wait_ms: int = 400  # after our end of turn, wait this long for the last words
    tts_first_audio_timeout_s: float = 5.0
    # ICE servers for WebRTC, e.g. ["stun:stun.l.google.com:19302"]. Empty: host candidates
    # only, which is enough on localhost and a LAN.
    webrtc_ice_servers: list[str] = []

    # Observability (app/observability/). Logs: JSON lines on stdout, or "text".
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"
    # Traces go over OTLP/HTTP to Langfuse when its keys are set, and/or to any collector at
    # OTEL_EXPORTER_OTLP_ENDPOINT (e.g. http://localhost:4318). Neither: not exported.
    otel_service_name: str = "resolveai"
    otel_exporter_otlp_endpoint: str | None = None
    otel_exporter_otlp_headers: str | None = None  # "key=value,key2=value2"
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    # Put model input and output on LLM spans. Off: they hold customer details (PII).
    otel_capture_content: bool = False
    # /admin/* (conversation replay, eval results). Set: requests need
    # "Authorization: Bearer <token>". Unset: only requests from this machine are served.
    admin_token: SecretStr | None = None
    # Eval harness output (evals/results/<run>/results.json), read by the eval dashboard.
    eval_results_dir: Path = EVAL_RESULTS_DIR

    def database_url(self, driver: str = "postgresql+psycopg") -> URL:
        # URL.create (not a string) so passwords with special characters need no escaping,
        # and a host starting with "/" is passed through to libpq as a Unix socket directory.
        return URL.create(
            drivername=driver,
            username=self.postgres_user,
            password=self.postgres_password.get_secret_value() or None,
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
