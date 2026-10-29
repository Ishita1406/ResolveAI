"""A client-side tokens-per-minute budget, shared by every LLM client that holds it.

Groq rejects a request (429) when the tokens used in the last minute plus the request's
estimate (prompt + max_completion_tokens) exceed the limit. Waiting here first keeps
concurrent conversations (the eval harness, or several sessions on a small plan) from
failing on 429 and retrying. Estimates are rough; a 429 can still happen and is retried by
the agent as before.
"""

import asyncio
import json
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any


def estimate_prompt_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> int:
    """About 4 characters per token. Callers correct it with observed usage (JSON-heavy
    prompts tokenize differently), then add the completion budget, which Groq counts up
    front (1,024 when not set)."""
    chars = len(json.dumps(messages, ensure_ascii=False)) + len(json.dumps(tools))
    return chars // 4


class TokenRateLimiter:
    def __init__(
        self,
        tokens_per_minute: int,
        *,
        window_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if tokens_per_minute <= 0:
            raise ValueError("tokens_per_minute must be positive")
        self.tokens_per_minute = tokens_per_minute
        self.window_s = window_s
        self._clock = clock
        self._sleep = sleep
        self._spent: deque[list] = deque()  # [time, tokens]; tokens corrected by settle()
        self._lock = asyncio.Lock()  # waiters are served in arrival order
        self.waited_s = 0.0  # total time spent waiting, for reports

    async def acquire(self, tokens: int) -> list:
        """Wait until `tokens` fit in the last minute's budget, then book them. Returns the
        booking, to correct with the real usage via settle()."""
        # A request bigger than the whole budget can only ever go alone.
        tokens = min(tokens, self.tokens_per_minute)
        async with self._lock:
            while True:
                now = self._clock()
                while self._spent and self._spent[0][0] <= now - self.window_s:
                    self._spent.popleft()
                used = sum(t for _, t in self._spent)
                if used + tokens <= self.tokens_per_minute:
                    booking = [now, tokens]
                    self._spent.append(booking)
                    return booking
                wait = max(self._spent[0][0] + self.window_s - now, 0.05)
                self.waited_s += wait
                await self._sleep(wait)

    def settle(self, booking: list, tokens: int) -> None:
        """Replace an estimate with what the provider reported (estimates run high)."""
        booking[1] = tokens
