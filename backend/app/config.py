from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

# Repo-root .env for local runs; in Docker, compose injects the variables directly.
ROOT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
POLICY_DOCS_DIR = Path(__file__).resolve().parents[2] / "docs" / "policies"


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

    # LLM (Groq, OpenAI-compatible chat completions). No key means the agent is disabled.
    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.2
    llm_request_timeout_s: float = 20.0
    llm_max_completion_tokens: int = 400
    llm_reasoning_effort: str | None = "low"

    # Agent loop
    agent_prompt_version: str = "v3"
    agent_max_iterations: int = 6
    agent_turn_timeout_s: float = 30.0
    agent_history_turns: int = 10
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
