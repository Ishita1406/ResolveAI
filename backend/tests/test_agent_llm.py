"""GeminiClient against a mocked HTTP transport (no network)."""

import asyncio
import json

import httpx
import pytest

from app.agent.llm import Completion, GeminiClient, LLMError, TextDelta, tool_call_message

pytestmark = pytest.mark.anyio


def _sse(*chunks: dict) -> bytes:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()


def _delta(**delta) -> dict:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}


class Recorder:
    def __init__(self, response: httpx.Response | Exception):
        self.response = response
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _client(handler) -> GeminiClient:
    return GeminiClient(api_key="test-key", model="test-model", base_url="https://llm.test/v1",
                        transport=httpx.MockTransport(handler))  # fmt: skip


async def _collect(client, tools=None):
    return [e async for e in client.stream([{"role": "user", "content": "hi"}], tools or [])]


async def test_text_is_streamed_then_completed():
    rec = Recorder(httpx.Response(200, content=_sse(_delta(content="Hel"), _delta(content="lo")),
                                  headers={"content-type": "text/event-stream"}))  # fmt: skip
    events = await _collect(_client(rec))
    assert events[:2] == [TextDelta("Hel"), TextDelta("lo")]
    completion = events[2]
    assert completion == Completion(text="Hello", first_token_s=completion.first_token_s)
    assert completion.first_token_s is not None and completion.first_token_s >= 0
    request = rec.requests[0]
    assert request.url == "https://llm.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    body = json.loads(request.content)
    assert (body["model"], body["stream"]) == ("test-model", True)
    assert body["stream_options"] == {"include_usage": True}  # Gemini sends usage on request
    assert "tools" not in body


async def test_tool_call_fragments_are_assembled():
    frag1 = {"index": 0, "id": "call_a", "type": "function",
             "function": {"name": "get_order", "arguments": '{"order_'}}  # fmt: skip
    frag2 = {"index": 0, "function": {"arguments": 'number": "VX-1508"}'}}
    frag3 = {"index": 1, "id": "call_b", "function": {"name": "list_recent_orders",
                                                      "arguments": "{}"}}  # fmt: skip
    finish = {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
              "usage": {"total_tokens": 42}}  # fmt: skip
    rec = Recorder(httpx.Response(200, content=_sse(
        _delta(tool_calls=[frag1]), _delta(tool_calls=[frag2]), _delta(tool_calls=[frag3]),
        finish)))  # fmt: skip
    tools = [{"type": "function", "function": {"name": "get_order", "parameters": {}}}]
    (completion,) = await _collect(_client(rec), tools)
    assert [(c.id, c.name, c.parsed_arguments()) for c in completion.tool_calls] == [
        ("call_a", "get_order", {"order_number": "VX-1508"}),
        ("call_b", "list_recent_orders", {}),
    ]
    assert completion.finish_reason == "tool_calls"
    assert completion.usage == {"total_tokens": 42}
    body = json.loads(rec.requests[0].content)
    assert body["tools"] == tools and body["tool_choice"] == "auto"


@pytest.mark.parametrize(("status", "retryable"), [(429, True), (503, True), (400, False),
                                                   (401, False)])  # fmt: skip
async def test_http_errors_are_typed(status, retryable):
    rec = Recorder(httpx.Response(status, json={"error": {"message": "nope"}}))
    with pytest.raises(LLMError) as exc:
        await _collect(_client(rec))
    assert (exc.value.status, exc.value.retryable) == (status, retryable)
    assert "test-key" not in str(exc.value)


async def test_timeouts_are_retryable_errors():
    with pytest.raises(LLMError) as exc:
        await _collect(_client(Recorder(httpx.ReadTimeout("slow"))))
    assert exc.value.retryable


def test_malformed_tool_arguments_parse_to_empty():
    from app.agent.llm import ToolCall

    assert ToolCall("c", "get_order", "{oops").parsed_arguments() == {}
    assert ToolCall("c", "get_order", "[1, 2]").parsed_arguments() == {}


class _SlowStream(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = False

    async def __aiter__(self):
        yield _sse(_delta(content="first"))[: -len(b"data: [DONE]\n\n")]
        await asyncio.sleep(10)
        yield b""

    async def aclose(self):
        self.closed = True


async def test_closing_the_stream_closes_the_http_response():
    body = _SlowStream()
    client = _client(lambda request: httpx.Response(200, stream=body))
    stream = client.stream([], [])
    assert await anext(stream) == TextDelta("first")
    await stream.aclose()  # barge-in: stop generating
    assert body.closed


async def test_request_carries_limits_reasoning_and_stop():
    rec = Recorder(httpx.Response(200, content=_sse(_delta(content="ok"))))
    client = GeminiClient(api_key="k", model="gemini-3.8-flash", base_url="https://llm.test/v1",
                          max_completion_tokens=800, reasoning_effort="low",
                          transport=httpx.MockTransport(rec))  # fmt: skip
    _ = [e async for e in client.stream([], [], stop=["<customer_message>"])]
    body = json.loads(rec.requests[0].content)
    assert body["max_completion_tokens"] == 800
    assert body["reasoning_effort"] == "low"
    assert "include_reasoning" not in body  # Groq-only; Gemini rejects unknown fields
    assert body["stop"] == ["<customer_message>"]


async def test_reasoning_params_are_omitted_for_other_models():
    rec = Recorder(httpx.Response(200, content=_sse(_delta(content="ok"))))
    _ = await _collect(_client(rec))
    body = json.loads(rec.requests[0].content)
    assert not {"reasoning_effort", "stop"} & set(body)


async def test_rate_limit_reports_retry_after():
    rec = Recorder(httpx.Response(429, headers={"retry-after": "15.8"}, json={"error": {}}))
    with pytest.raises(LLMError) as exc:
        await _collect(_client(rec))
    assert (exc.value.retryable, exc.value.retry_after_s) == (True, 15.8)


async def test_rate_limit_reads_gemini_retry_delay_from_the_body():
    error = [{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "x" * 600,
              "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                           "retryDelay": "18s"}]}}]  # fmt: skip
    rec = Recorder(httpx.Response(429, json=error))
    with pytest.raises(LLMError) as exc:
        await _collect(_client(rec))
    assert (exc.value.retryable, exc.value.retry_after_s) == (True, 18.0)


async def test_gemini_tool_calls_without_index_and_signatures_are_kept():
    sig = {"google": {"thought_signature": "abc"}}
    call_a = {
        "id": "a",
        "type": "function",
        "extra_content": sig,
        "function": {"name": "get_order", "arguments": '{"order_number": "VX-1"}'},
    }
    call_b = {"id": "b", "type": "function",
              "function": {"name": "list_recent_orders", "arguments": "{}"}}  # fmt: skip
    rec = Recorder(httpx.Response(200, content=_sse(
        _delta(tool_calls=[call_a]), _delta(tool_calls=[call_b]))))  # fmt: skip
    (completion,) = await _collect(_client(rec))
    assert [(c.id, c.name, c.extra_content) for c in completion.tool_calls] == [
        ("a", "get_order", sig),
        ("b", "list_recent_orders", None),
    ]
    assert tool_call_message(completion.tool_calls[0])["extra_content"] == sig
    assert "extra_content" not in tool_call_message(completion.tool_calls[1])


async def test_seed_is_sent_only_when_set():
    rec = Recorder(httpx.Response(200, content=_sse(_delta(content="ok"))))
    client = GeminiClient(api_key="k", model="m", base_url="https://llm.test/v1", seed=42,
                          transport=httpx.MockTransport(rec))  # fmt: skip
    _ = [e async for e in client.stream([], [])]
    assert json.loads(rec.requests[0].content)["seed"] == 42
    _ = await _collect(_client(rec))
    assert "seed" not in json.loads(rec.requests[1].content)


async def test_rate_limiter_is_consulted_before_each_request():
    class Limiter:
        def __init__(self):
            self.asked: list[int] = []

        async def acquire(self, tokens: int) -> None:
            self.asked.append(tokens)

    limiter = Limiter()
    rec = Recorder(httpx.Response(200, content=_sse(_delta(content="ok"))))
    client = GeminiClient(api_key="k", model="m", base_url="https://llm.test/v1",
                          max_completion_tokens=400, rate_limiter=limiter,
                          transport=httpx.MockTransport(rec))  # fmt: skip
    _ = await _collect(client)
    assert len(limiter.asked) == 1 and limiter.asked[0] > 400  # prompt estimate + completion


async def test_prompt_estimate_learns_from_reported_usage():
    class Limiter:
        def __init__(self):
            self.asked, self.settled = [], []

        async def acquire(self, tokens):
            self.asked.append(tokens)
            return [0, tokens]

        def settle(self, booking, tokens):
            self.settled.append(tokens)

    usage = {"prompt_tokens": 1, "completion_tokens": 2}
    body = _sse(_delta(content="ok"), {"choices": [], "usage": usage})
    limiter = Limiter()
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=body))
    client = GeminiClient(api_key="k", model="m", base_url="https://llm.test/v1",
                          max_completion_tokens=10, rate_limiter=limiter,
                          transport=transport)  # fmt: skip
    messages = [{"role": "user", "content": "x" * 4000}]
    for _ in range(3):
        _ = [e async for e in client.stream(messages, [])]
    assert limiter.settled == [3, 3, 3]  # bookings corrected to the real usage
    assert limiter.asked[0] > limiter.asked[1] > limiter.asked[2]  # estimate shrinks to fit


async def test_a_routed_call_names_its_own_model():
    rec = Recorder(httpx.Response(200, content=_sse(_delta(content="ok"))))
    _ = [e async for e in _client(rec).stream([], [], model="small-model")]
    assert json.loads(rec.requests[0].content)["model"] == "small-model"


async def test_warm_opens_a_connection_without_spending_tokens():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        return httpx.Response(503)  # a failure is ignored: the real call will report it

    await _client(handler).warm()
    assert seen == [("GET", "/v1/models")]
