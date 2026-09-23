from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import httpx
import pytest
from anthropic import RateLimitError

import agent

# ---------- fakes ----------


def text_block(text):
    return NS(type="text", text=text)


def tool_block(name, input, id="tool_1"):
    return NS(type="tool_use", name=name, input=input, id=id)


def message(content, stop_reason, input_tokens=100, output_tokens=50):
    return NS(
        content=content,
        stop_reason=stop_reason,
        usage=NS(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class FakeStream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    @property
    def text_stream(self):
        return (b.text for b in self.msg.content if b.type == "text")

    def get_final_message(self):
        return self.msg


class FakeMessages:
    """Returns scripted responses in order, one per call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def stream(self, **kwargs):
        self.calls += 1
        next_item = self.responses.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return FakeStream(next_item)


# ---------- fixtures ----------


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch):
    monkeypatch.setattr(agent, "langfuse", MagicMock())


def script(monkeypatch, *responses):
    fake = FakeMessages(responses)
    monkeypatch.setattr(agent, "claude", NS(messages=fake))
    return fake


def final_result(events):
    return next(e for e in events if e["type"] == "result")


# ---------- tests ----------


def test_answers_without_tools(monkeypatch):
    script(monkeypatch, message([text_block("Hello")], "end_turn"))
    result = final_result(agent.run_agent_stream("hi"))
    assert result["answer"] == "Hello"
    assert result["error"] is None
    assert result["tokens"] == 150
    assert result["searches"] == 0


def test_searches_then_answers(monkeypatch):
    fake = script(
        monkeypatch,
        message([tool_block("search_web", {"query": "golf 2015"})], "tool_use"),
        message([text_block("The Golf uses 7 L/100km")], "end_turn"),
    )
    queries = []
    monkeypatch.setattr(
        agent, "search_web", lambda q: queries.append(q) or "fake results"
    )

    events = list(agent.run_agent_stream("golf?"))

    assert queries == ["golf 2015"]
    assert {"type": "searching", "query": "golf 2015"} in events
    assert final_result(events)["searches"] == 1
    assert fake.calls == 2


def test_failed_search_does_not_kill_the_agent(monkeypatch):
    script(
        monkeypatch,
        message([tool_block("search_web", {"query": "golf"})], "tool_use"),
        message([text_block("I couldn't search")], "end_turn"),
    )

    def broken_search(query):
        raise RuntimeError("tavily down")

    monkeypatch.setattr(agent, "search_web", broken_search)

    result = final_result(list(agent.run_agent_stream("golf?")))

    assert result["error"] is None
    assert result["answer"] == "I couldn't search"


def test_budget_exceeded(monkeypatch):
    monkeypatch.setattr(agent, "TOKEN_BUDGET", 100)
    script(monkeypatch, message([text_block("x")], "end_turn", input_tokens=200))

    result = final_result(list(agent.run_agent_stream("hi")))

    assert result["error"] == "budget_exceeded"


def test_provider_error_becomes_result_event(monkeypatch):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = RateLimitError(
        "slow down", response=httpx.Response(429, request=request), body=None
    )
    script(monkeypatch, error)

    result = final_result(list(agent.run_agent_stream("hi")))

    assert result["error"] == "rate_limited"
    assert result["answer"] is None


def test_max_iterations(monkeypatch):
    monkeypatch.setattr(agent, "MAX_ITERATIONS", 2)
    monkeypatch.setattr(agent, "search_web", lambda q: "results")
    script(
        monkeypatch,
        message([tool_block("search_web", {"query": "a"})], "tool_use"),
        message([tool_block("search_web", {"query": "b"})], "tool_use"),
    )

    result = final_result(list(agent.run_agent_stream("hi")))

    assert result["error"] == "max_iterations"
