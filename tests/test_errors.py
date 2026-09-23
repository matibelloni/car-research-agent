import httpx2
from anthropic import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
)

from agent import classify_error


def make_status_error(cls, status_code: int):
    """Builds an SDK status exception without hitting the network."""
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status_code, request=request)
    return cls("boom", response=response, body=None)


def test_rate_limit():
    error = make_status_error(RateLimitError, 429)
    assert classify_error(error) == "rate_limited"


def test_server_error():
    error = make_status_error(InternalServerError, 529)
    assert classify_error(error) == "provider_unavailable"


def test_auth_error_is_not_retryable():
    error = make_status_error(AuthenticationError, 401)
    assert classify_error(error) == "provider_error"


def test_connection_error():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    assert classify_error(APIConnectionError(request=request)) == "provider_unreachable"


def test_timeout_is_a_connection_error():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    assert classify_error(APITimeoutError(request=request)) == "provider_unreachable"
