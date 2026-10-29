from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

# Repo-root .env for local runs; in Docker, compose injects the variables directly.
ROOT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


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
