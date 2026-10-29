"""The client-side tokens-per-minute budget, on a fake clock."""

import pytest

from app.agent.ratelimit import TokenRateLimiter, estimate_prompt_tokens

pytestmark = pytest.mark.anyio


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _limiter(tpm: int, t: FakeTime) -> TokenRateLimiter:
    return TokenRateLimiter(tpm, clock=t.clock, sleep=t.sleep)


async def test_requests_within_the_budget_go_straight_through():
    t = FakeTime()
    limiter = _limiter(8000, t)
    await limiter.acquire(3000)
    await limiter.acquire(5000)
    assert t.sleeps == []


async def test_over_budget_waits_until_the_oldest_request_leaves_the_window():
    t = FakeTime()
    limiter = _limiter(8000, t)
    await limiter.acquire(5000)
    t.now = 10.0
    await limiter.acquire(4000)  # 9000 > 8000: wait for the first to age out at t=60
    assert t.sleeps == [50.0] and t.now == 60.0
    assert limiter.waited_s == 50.0


async def test_a_request_larger_than_the_budget_still_goes_alone():
    t = FakeTime()
    limiter = _limiter(1000, t)
    await limiter.acquire(5000)
    assert t.sleeps == []


def test_prompt_estimate_is_about_four_characters_per_token():
    messages = [{"role": "user", "content": "x" * 400}]
    assert 100 <= estimate_prompt_tokens(messages, []) <= 120


def test_budget_must_be_positive():
    with pytest.raises(ValueError):
        TokenRateLimiter(0)


async def test_settling_with_real_usage_frees_budget():
    t = FakeTime()
    limiter = _limiter(8000, t)
    booking = await limiter.acquire(6000)  # an estimate
    limiter.settle(booking, 2000)  # what the provider reported
    await limiter.acquire(5000)
    assert t.sleeps == []
