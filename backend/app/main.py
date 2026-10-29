import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agent.llm import GroqClient
from app.agent.loop import Agent, AgentConfig
from app.agent.prompts import load_prompts
from app.agent.session import RedisSessionStore
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.config import Settings, get_settings
from app.policy.config import get_policy

log = logging.getLogger(__name__)


def build_agent(settings: Settings, engine, redis: Redis) -> tuple[Agent | None, GroqClient | None]:
    """The agent, or (None, None) when no LLM key is configured."""
    if settings.groq_api_key is None or not settings.groq_api_key.get_secret_value():
        log.warning("GROQ_API_KEY is not set; /ws/chat will report the agent as unavailable")
        return None, None
    llm = GroqClient(
        api_key=settings.groq_api_key.get_secret_value(),
        model=settings.groq_model,
        base_url=settings.groq_base_url,
        temperature=settings.llm_temperature,
        timeout_s=settings.llm_request_timeout_s,
        max_completion_tokens=settings.llm_max_completion_tokens,
        reasoning_effort=settings.llm_reasoning_effort or None,
    )
    config = AgentConfig(
        max_iterations=settings.agent_max_iterations,
        turn_timeout_s=settings.agent_turn_timeout_s,
        history_turns=settings.agent_history_turns,
    )
    store = RedisSessionStore(
        redis, ttl_s=settings.session_ttl_s, lock_timeout_s=config.turn_timeout_s + 30
    )
    agent = Agent(
        llm=llm,
        store=store,
        db_factory=async_sessionmaker(engine),
        prompts=load_prompts(settings.agent_prompt_version),
        policy=get_policy(),
        config=config,
    )
    return agent, llm


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.engine = create_async_engine(settings.database_url(), pool_pre_ping=True)
    app.state.redis = Redis.from_url(settings.redis_url)
    app.state.agent, llm = build_agent(settings, app.state.engine, app.state.redis)
    try:
        yield
    finally:
        if llm is not None:
            await llm.aclose()
        await app.state.redis.aclose()
        await app.state.engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(title="ResolveAI", lifespan=lifespan)
    app.include_router(health_router)
    app.include_router(chat_router)
    return app


app = create_app()
