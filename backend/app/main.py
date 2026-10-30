import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agent.llm import GeminiClient, LLMClient
from app.agent.loop import Agent, AgentConfig
from app.agent.prompts import load_prompts
from app.agent.ratelimit import TokenRateLimiter
from app.agent.session import RedisSessionStore
from app.api.admin import router as admin_router
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.config import Settings, get_settings
from app.observability import tracing
from app.observability.logs import configure_logging
from app.policy.config import get_policy
from app.retrieval.chunking import policy_documents_text
from app.retrieval.embedding import Embedder, embedder_from_settings
from app.retrieval.search import PolicyRetriever
from app.voice import webrtc

log = logging.getLogger(__name__)

Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(UTC)


def build_llm(settings: Settings) -> GeminiClient | None:
    """The configured Gemini client, or None when no key is set."""
    if settings.gemini_api_key is None or not settings.gemini_api_key.get_secret_value():
        return None
    limiter = (
        TokenRateLimiter(settings.llm_tokens_per_minute) if settings.llm_tokens_per_minute else None
    )
    return GeminiClient(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=settings.gemini_model,
        base_url=settings.gemini_base_url,
        temperature=settings.llm_temperature,
        timeout_s=settings.llm_request_timeout_s,
        max_completion_tokens=settings.llm_max_completion_tokens,
        reasoning_effort=settings.llm_reasoning_effort or None,
        seed=settings.llm_seed,
        rate_limiter=limiter,
        keepalive_s=settings.llm_keepalive_s,
    )


def build_agent(
    settings: Settings,
    engine,
    redis: Redis,
    *,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    clock: Clock = _utcnow,
) -> tuple[Agent | None, LLMClient | None]:
    """The agent and the LLM client it owns, or (None, None) when no LLM is configured.

    `llm`, `embedder` and `clock` let the eval harness share a rate-limited client and one
    loaded embedding model across many app instances, and pin the clock to the seed data.
    An injected `llm` stays the caller's to close, so (agent, None) is returned for it.
    """
    owned = None
    if llm is None:
        llm = owned = build_llm(settings)
    if llm is None:
        log.warning("GEMINI_API_KEY is not set; /ws/chat will report the agent as unavailable")
        return None, None
    config = AgentConfig(
        max_iterations=settings.agent_max_iterations,
        turn_timeout_s=settings.agent_turn_timeout_s,
        history_turns=settings.agent_history_turns,
        prefetch=settings.agent_prefetch,
        llm_prices=settings.llm_prices,
        fast_model=settings.llm_fast_model,
    )
    retriever, documents = None, None
    if settings.policy_context_mode == "retrieval":
        retriever = PolicyRetriever(
            embedder or embedder_from_settings(settings),
            mode=settings.retrieval_search_mode,
            top_k=settings.retrieval_top_k,
            candidates=settings.retrieval_candidates,
            rrf_k=settings.retrieval_rrf_k,
        )
    else:
        documents = policy_documents_text(settings.policy_docs_dir)
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
        clock=clock,
        retriever=retriever,
        policy_documents=documents,
    )
    return agent, owned


def create_app(
    settings: Settings | None = None,
    *,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    clock: Clock = _utcnow,
) -> FastAPI:
    """The API. Production calls it with no arguments; the eval harness passes its own
    settings (a fresh database per scenario), LLM client, embedder and clock."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved = settings or get_settings()
        app.state.settings = resolved
        if settings is None:  # the real server; tests and the eval harness keep their logging
            configure_logging(resolved.log_level, resolved.log_format)
        tracing.setup_tracing(resolved)
        app.state.engine = create_async_engine(resolved.database_url(), pool_pre_ping=True)
        app.state.redis = Redis.from_url(resolved.redis_url)
        app.state.agent, owned_llm = build_agent(
            resolved, app.state.engine, app.state.redis, llm=llm, embedder=embedder, clock=clock
        )
        if app.state.agent is not None and app.state.agent.retriever is not None:
            await app.state.agent.retriever.embedder.warm()  # don't make the first caller wait
        try:
            yield
        finally:
            await webrtc.close_all(app)
            if owned_llm is not None:
                await owned_llm.aclose()
            await app.state.redis.aclose()
            await app.state.engine.dispose()
            tracing.shutdown_tracing()

    app = FastAPI(title="ResolveAI", lifespan=lifespan)
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(webrtc.router)
    app.include_router(admin_router)
    return app


app = create_app()
