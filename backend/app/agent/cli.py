"""Chat with the agent in the terminal.

    python -m app.agent.cli                    # new session, stored in Redis
    python -m app.agent.cli --session <id>     # resume a session
    python -m app.agent.cli --memory           # keep the session in memory (no Redis)

Ctrl-C while the assistant is replying interrupts that reply (like barge-in).
Ctrl-C or Ctrl-D at the prompt exits. /state shows verification and pending actions.
"""

import argparse
import asyncio
import contextlib
import logging
import signal
import sys

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import create_async_engine

from app.agent.session import InMemorySessionStore
from app.config import get_settings
from app.main import build_agent

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def _print_event(event) -> None:
    match event.type:
        case "token":
            print(event.text, end="", flush=True)
        case "tool" if event.status == "started":
            print(f"{DIM}[{event.name}…]{RESET} ", end="", flush=True)
        case "action":
            amount = f" {event.amount_display}" if event.amount_display else ""
            print(f"\n{DIM}[action {event.state}: {event.summary}{amount}]{RESET}", flush=True)
        case "escalated":
            print(f"\n{DIM}[handed to a human: {event.reference}]{RESET}", flush=True)
        case "error":
            print(f"\n{DIM}[error {event.code}: {event.message}]{RESET}", flush=True)


async def _show_state(agent, session_id: str) -> None:
    session = await agent.store.load(session_id)
    state = session.state
    who = state.customer_name if state.is_verified else "not verified"
    print(f"{DIM}session {session_id} | turn {state.turn} | {who}")
    for action in state.actions.values():
        print(f"  {action.state:<22} {action.summary}")
    print(RESET, end="")


async def _turn(agent, session_id: str, text: str) -> None:
    async def consume() -> None:
        async for event in agent.respond(session_id, text):
            _print_event(event)

    task = asyncio.create_task(consume())
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, task.cancel)
    try:
        await task
    except asyncio.CancelledError:
        print(f" {DIM}[interrupted]{RESET}", end="")
    finally:
        loop.remove_signal_handler(signal.SIGINT)
    print()


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with the ResolveAI agent.")
    parser.add_argument("--session", help="resume this session id")
    parser.add_argument("--memory", action="store_true", help="don't use Redis for sessions")
    parser.add_argument("--verbose", action="store_true", help="show server warnings")
    args = parser.parse_args(argv)
    # Server logs (retries, provider errors) would interleave with the chat; keep them quiet.
    logging.basicConfig(level=logging.WARNING if args.verbose else logging.ERROR)

    settings = get_settings()
    engine = create_async_engine(settings.database_url())
    redis = Redis.from_url(settings.redis_url)
    agent, llm = build_agent(settings, engine, redis)
    if agent is None:
        print("GEMINI_API_KEY is not set. Add it to .env (see .env.example).", file=sys.stderr)
        await redis.aclose()
        await engine.dispose()
        return 2
    if args.memory:
        agent.store = InMemorySessionStore()
    if agent.retriever is not None:
        await agent.retriever.embedder.warm()
    try:
        session = await agent.store.load(args.session) if args.session else None
        if session is None:
            session = await agent.start_session()
        print(f"{BOLD}ResolveAI{RESET} session {session.session_id} "
              f"(model {settings.gemini_model}). Ctrl-C interrupts, Ctrl-D quits.")  # fmt: skip
        while True:
            try:
                text = await asyncio.to_thread(input, f"{BOLD}you>{RESET} ")
            except EOFError:
                break
            text = text.strip()
            if not text:
                continue
            if text in ("/quit", "/exit"):
                break
            if text == "/state":
                await _show_state(agent, session.session_id)
                continue
            print(f"{BOLD}agent>{RESET} ", end="", flush=True)
            await _turn(agent, session.session_id, text)
    finally:
        await llm.aclose()
        await redis.aclose()
        await engine.dispose()
    return 0


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(asyncio.run(main()))
