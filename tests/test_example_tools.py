"""T5 acceptance test: shipped example tools (final_answer, calculate, http_get).

Per plan §13 T5 and §2 section 10 (Example tools).

Covers:
  - final_answer is a bare function (not @tool-decorated) that returns its input
  - calculate uses simpleeval.SimpleEval, no raw eval/exec; safe against
    attribute/method access and __import__ injection
  - http_get is async, uses httpx.AsyncClient, returns response text
  - All three are importable from the top-level tinyagent namespace
  - calculate and http_get carry JSON schemas via @tool (final_answer does not)
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
from typing import Any

import httpx
import pytest

import tinyagent
from tinyagent import calculate, final_answer, http_get


# ---------------------------------------------------------------------
# Imports — every shipped tool resolves from the top-level namespace
# ---------------------------------------------------------------------
def test_final_answer_importable_from_top_level() -> None:
    """`from tinyagent import final_answer` resolves to a callable."""
    assert callable(final_answer)
    assert final_answer is tinyagent.final_answer


def test_calculate_importable_from_top_level() -> None:
    """`from tinyagent import calculate` resolves to a callable with a schema."""
    assert callable(calculate)
    assert calculate is tinyagent.calculate


def test_http_get_importable_from_top_level() -> None:
    """`from tinyagent import http_get` resolves to a coroutine function with a schema."""
    assert inspect.iscoroutinefunction(http_get)
    assert http_get is tinyagent.http_get


# ---------------------------------------------------------------------
# final_answer — bare termination tool
# ---------------------------------------------------------------------
def test_final_answer_returns_input_unchanged() -> None:
    """`final_answer(answer)` returns `answer` exactly (no eval, no transform)."""
    assert final_answer("hello") == "hello"


def test_final_answer_with_empty_string() -> None:
    """`final_answer("")` returns "" — empty string is a valid answer."""
    assert final_answer("") == ""


def test_final_answer_with_multiline() -> None:
    """`final_answer` preserves multiline content verbatim."""
    payload = "line1\nline2\nline3"
    assert final_answer(payload) == payload


def test_final_answer_is_not_tool_decorated() -> None:
    """final_answer is the loop terminator — intentionally NOT a @tool.

    The agent loop recognises it by name, not by schema, so adding a JSON
    schema would cause the LLM to call it through the normal tool
    dispatch path rather than the termination branch.
    """
    assert not getattr(final_answer, "is_tool", False)
    assert not hasattr(final_answer, "tool_schema")


# ---------------------------------------------------------------------
# calculate — safe expression evaluator
# ---------------------------------------------------------------------
def test_calculate_basic_arithmetic() -> None:
    """calculate respects standard operator precedence (* before +)."""
    result = calculate("2 + 3 * 4")
    assert result == "14"


def test_calculate_parentheses_and_floats() -> None:
    """calculate handles parentheses and produces float results as strings."""
    result = calculate("(1 + 2) * 3.5")
    # str(float) may give e.g. "10.5" — keep it loose but exact.
    assert result == str(10.5)


def test_calculate_math_constants() -> None:
    """calculate exposes standard math constants like pi and e."""
    # pi^2 should round to a small float
    result = calculate("pi * 2")
    assert float(result) == pytest.approx(6.283185307179586, abs=1e-9)


def test_calculate_division_by_zero_returns_error_string() -> None:
    """`1/0` is an arithmetic error; calculate returns a string, NOT a raise."""
    result = calculate("1 / 0")
    assert isinstance(result, str)
    assert result.startswith("Error:"), f"expected 'Error:' prefix, got {result!r}"


def test_calculate_blocks_dunder_import() -> None:
    """`__import__('os')` is blocked — calculate must NOT execute code.

    This is the security-critical test: a naive `eval()` implementation
    would let an attacker reach the `os` module via the `__import__`
    builtin. simpleeval.SimpleEval has no `__import__` available, so the
    expression is a FunctionNotDefined error and the function returns an
    error string.
    """
    result = calculate("__import__('os').system('rm -rf /')")
    assert isinstance(result, str)
    assert result.startswith("Error:"), f"expected 'Error:' prefix, got {result!r}"
    # The text must not contain anything that looks like a successful
    # shell command output — the safest assertion is "no traceback surfaced"
    # but a stricter one is that the error mentions the function name.
    assert "__import__" in result or "not defined" in result.lower()


def test_calculate_blocks_attribute_access() -> None:
    """`().__class__.__bases__[0].__subclasses__()` style probes are blocked."""
    # Several probe payloads — at least one of them MUST be rejected.
    payloads: list[str] = [
        "().__class__.__bases__",
        "[].__class__",
        "{}.update",
        "''.__class__.__mro__",
    ]
    for payload in payloads:
        result = calculate(payload)
        assert isinstance(result, str)
        assert result.startswith("Error:"), (
            f"payload {payload!r} did not return an error string: {result!r}"
        )


def test_calculate_syntax_error_returns_error_string() -> None:
    """Malformed input is also an error string, not a raise."""
    result = calculate("not valid python +")
    assert isinstance(result, str)
    assert result.startswith("Error:")


def test_calculate_empty_string_returns_error_string() -> None:
    """An empty expression is a parse error, not a raise."""
    result = calculate("")
    assert isinstance(result, str)
    assert result.startswith("Error:")


def test_calculate_carries_tool_schema() -> None:
    """calculate is @tool-decorated, so it carries a JSON schema with name=calculate."""
    assert getattr(calculate, "is_tool", False) is True
    schema: dict[str, Any] = calculate.tool_schema
    assert schema["name"] == "calculate"
    assert schema["parameters"]["properties"]["expression"]["type"] == "string"
    assert "expression" in schema["parameters"]["required"]


# ---------------------------------------------------------------------
# http_get — async HTTP fetch
# ---------------------------------------------------------------------
def test_http_get_carries_tool_schema() -> None:
    """http_get is @tool-decorated, so it carries a JSON schema with name=http_get."""
    assert getattr(http_get, "is_tool", False) is True
    schema: dict[str, Any] = http_get.tool_schema
    assert schema["name"] == "http_get"
    props = schema["parameters"]["properties"]
    assert props["url"]["type"] == "string"
    assert "url" in schema["parameters"]["required"]
    # timeout has a default — not in required.
    assert "timeout" not in schema["parameters"]["required"]
    assert props["timeout"]["default"] == 10.0


def test_http_get_is_coroutine_function() -> None:
    """http_get is async; calling it returns a coroutine."""
    coro = http_get("https://example.com")
    assert asyncio.iscoroutine(coro)
    coro.close()


def test_http_get_returns_response_text() -> None:
    """http_get fetches via httpx.AsyncClient and returns the response body."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="hello from mock server")

    transport = httpx.MockTransport(handler)

    async def _run() -> str:
        # Monkey-patch httpx.AsyncClient for the duration of this test.
        original_client = httpx.AsyncClient

        class _PatchedClient(original_client):  # type: ignore[misc, valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                kwargs["transport"] = transport
                super().__init__(*args, **kwargs)

        httpx.AsyncClient = _PatchedClient  # type: ignore[misc]
        try:
            # @tool-wrapped functions are typed as Any; the result IS a
            # str per the tool's contract. Annotate the local as str to
            # give mypy a concrete type for the assertions below.
            result: str = await http_get("https://example.com/test")
            return result
        finally:
            httpx.AsyncClient = original_client  # type: ignore[misc]

    result = asyncio.run(_run())
    assert result == "hello from mock server"


def test_http_get_truncates_oversized_response() -> None:
    """http_get caps response text at ~4KB so a giant page can't blow up the agent."""

    def handler(request: httpx.Request) -> httpx.Response:
        # 8 KB body — well over the 4 KB cap.
        return httpx.Response(200, text="x" * 8192)

    transport = httpx.MockTransport(handler)

    async def _run() -> str:
        original_client = httpx.AsyncClient

        class _PatchedClient(original_client):  # type: ignore[misc, valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                kwargs["transport"] = transport
                super().__init__(*args, **kwargs)

        httpx.AsyncClient = _PatchedClient  # type: ignore[misc]
        try:
            # @tool-wrapped functions are typed as Any; the result IS a
            # str per the tool's contract. Annotate the local as str to
            # give mypy a concrete type for the assertions below.
            result: str = await http_get("https://example.com/big")
            return result
        finally:
            httpx.AsyncClient = original_client  # type: ignore[misc]

    result = asyncio.run(_run())
    # Truncated to at most ~4 KB (4096 chars). Allow the 4096-4099 range
    # because some implementations add a single truncation marker.
    assert len(result) <= 4099, f"expected <=~4KB, got {len(result)} chars"
    # The first ~4096 chars are still the body, so a long "xxxx..." string
    # is expected to dominate the result.
    assert result.startswith("x" * 100)


def test_http_get_returns_error_string_on_failure() -> None:
    """A network error is returned as an 'Error: ...' string, not raised."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("mock connect failure")

    transport = httpx.MockTransport(handler)

    async def _run() -> str:
        original_client = httpx.AsyncClient

        class _PatchedClient(original_client):  # type: ignore[misc, valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                kwargs["transport"] = transport
                super().__init__(*args, **kwargs)

        httpx.AsyncClient = _PatchedClient  # type: ignore[misc]
        try:
            # @tool-wrapped functions are typed as Any; the result IS a
            # str per the tool's contract. Annotate the local as str to
            # give mypy a concrete type for the assertions below.
            result: str = await http_get("https://unreachable.example.com/")
            return result
        finally:
            httpx.AsyncClient = original_client  # type: ignore[misc]

    result = asyncio.run(_run())
    assert isinstance(result, str)
    assert result.startswith("Error:"), f"expected 'Error:' prefix, got {result!r}"


# ---------------------------------------------------------------------
# __all__ — example tools are still listed after the implementation lands
# ---------------------------------------------------------------------
def test_example_tools_listed_in_module_all() -> None:
    """The shipped example tools remain in tinyagent.__all__ after T5."""
    module = importlib.import_module("tinyagent")
    for name in ("calculate", "http_get", "final_answer"):
        assert name in module.__all__, (
            f"{name!r} missing from tinyagent.__all__ after T5 implementation"
        )
