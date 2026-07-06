"""T4 acceptance test: @tool decorator + _wrap_no_exception + _cast_argument.

Per plan §13 T4 and §2 section 9 (Tool helpers).
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any

from tinyagent import _cast_argument, tool


# ---------------------------------------------------------------------
# @tool decorator — sync callable
# ---------------------------------------------------------------------
def test_tool_decorator_sync_no_parens() -> None:
    """`@tool` (no-paren form) wraps a sync callable and attaches a schema."""

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    # Schema attached as a function attribute.
    assert hasattr(add, "tool_schema")
    schema: dict[str, Any] = add.tool_schema
    assert schema["name"] == "add"
    assert schema["description"] == "Add two integers."
    assert schema["parameters"]["type"] == "object"
    assert schema["parameters"]["properties"]["a"]["type"] == "integer"
    assert schema["parameters"]["properties"]["b"]["type"] == "integer"
    assert set(schema["parameters"]["required"]) == {"a", "b"}


def test_tool_decorator_sync_with_parens() -> None:
    """`@tool()` (paren, no-kwarg form) is equivalent to `@tool`."""

    @tool()
    def greet(name: str) -> str:
        return f"hello {name}"

    assert greet.tool_schema["name"] == "greet"
    assert greet.tool_schema["parameters"]["properties"]["name"]["type"] == "string"
    assert greet.tool_schema["parameters"]["required"] == ["name"]


def test_tool_decorator_with_kwargs() -> None:
    """`@tool(name=...)` (paren, kwarg form) is supported and stores kwargs."""

    @tool(name="renamed")
    def internal_name(x: int) -> int:
        return x

    # The kwarg form should still build a valid schema.
    assert internal_name.tool_schema["parameters"]["properties"]["x"]["type"] == "integer"
    assert internal_name.tool_schema["parameters"]["required"] == ["x"]


# ---------------------------------------------------------------------
# @tool decorator — async callable
# ---------------------------------------------------------------------
def test_tool_decorator_async() -> None:
    """`@tool` wraps an async callable and remains awaitable."""

    @tool
    async def fetch(url: str, timeout: float = 5.0) -> str:
        """Fetch a URL asynchronously."""
        return f"fetched {url}"

    assert inspect.iscoroutinefunction(fetch)
    schema: dict[str, Any] = fetch.tool_schema
    assert schema["name"] == "fetch"
    assert schema["description"] == "Fetch a URL asynchronously."
    props = schema["parameters"]["properties"]
    assert props["url"]["type"] == "string"
    assert props["timeout"]["type"] == "number"
    # Default is reflected in the property entry.
    assert props["timeout"]["default"] == 5.0
    # Required list excludes parameters with defaults.
    assert schema["parameters"]["required"] == ["url"]


# ---------------------------------------------------------------------
# Schema generation: types and required flags
# ---------------------------------------------------------------------
def test_schema_types_for_primitives() -> None:
    """Type hints map to JSON-Schema primitive types."""

    @tool
    def example(
        s: str,
        i: int,
        f: float,
        b: bool,
        items: list[str],
        mapping: dict[str, int],
    ) -> None:
        """Mixed-type example."""

    props = example.tool_schema["parameters"]["properties"]
    assert props["s"]["type"] == "string"
    assert props["i"]["type"] == "integer"
    assert props["f"]["type"] == "number"
    assert props["b"]["type"] == "boolean"
    assert props["items"]["type"] == "array"
    assert props["mapping"]["type"] == "object"
    # All parameters have no default → all required.
    assert set(example.tool_schema["parameters"]["required"]) == {
        "s", "i", "f", "b", "items", "mapping",
    }


def test_schema_required_excludes_params_with_defaults() -> None:
    """Parameters with defaults are NOT in the required list."""

    @tool
    def f(a: int, b: int = 10, c: str = "x") -> None:
        return None

    required = f.tool_schema["parameters"]["required"]
    assert required == ["a"], f"expected only 'a' required, got {required}"


def test_schema_includes_default_values() -> None:
    """Default values are reflected in the property entry."""

    @tool
    def f(a: int, b: int = 10, c: str = "x") -> None:
        return None

    props = f.tool_schema["parameters"]["properties"]
    assert props["a"].get("default") is None
    assert props["b"]["default"] == 10
    assert props["c"]["default"] == "x"


# ---------------------------------------------------------------------
# Decorated function is still callable with original args
# ---------------------------------------------------------------------
def test_decorated_function_callable_sync() -> None:
    """The decorated sync function is invoked exactly as the original."""

    @tool
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5
    assert add(a=10, b=20) == 30


def test_decorated_function_callable_async() -> None:
    """The decorated async function returns a coroutine when called."""

    @tool
    async def fetch(url: str) -> str:
        return f"got {url}"

    result = fetch("https://example.com")
    # fetch(...) returns a coroutine because it's still async.
    assert asyncio.iscoroutine(result)
    assert asyncio.run(result) == "got https://example.com"


def test_decorated_function_preserves_signature() -> None:
    """inspect.signature on the decorated function matches the original."""

    @tool
    def original(a: int, b: str = "x") -> bool:
        return True

    sig = inspect.signature(original)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ["a", "b"]
    assert sig.parameters["b"].default == "x"


# ---------------------------------------------------------------------
# _cast_argument — primitive coercion
# ---------------------------------------------------------------------
def test_cast_argument_str() -> None:
    """str cast preserves the string and stringifies non-string inputs."""
    assert _cast_argument("hello", str) == "hello"
    assert _cast_argument(42, str) == "42"
    assert _cast_argument(3.14, str) == "3.14"
    assert _cast_argument(True, str) == "True"


def test_cast_argument_int() -> None:
    """int cast handles string-to-int and float-to-int (truncation)."""
    assert _cast_argument(5, int) == 5
    assert _cast_argument("42", int) == 42
    assert _cast_argument(3.7, int) == 3
    assert _cast_argument(True, int) == 1


def test_cast_argument_float() -> None:
    """float cast handles string-to-float and int-to-float."""
    assert _cast_argument(2, float) == 2.0
    assert _cast_argument("3.14", float) == 3.14
    assert _cast_argument(0, float) == 0.0


def test_cast_argument_bool() -> None:
    """bool cast recognises string truthy/falsy values."""
    assert _cast_argument(True, bool) is True
    assert _cast_argument(False, bool) is False
    assert _cast_argument("true", bool) is True
    assert _cast_argument("false", bool) is False
    assert _cast_argument("1", bool) is True
    assert _cast_argument("0", bool) is False
    assert _cast_argument("yes", bool) is True
    assert _cast_argument("no", bool) is False


def test_cast_argument_list() -> None:
    """list cast: JSON string parses to list, existing list passes through."""
    assert _cast_argument([1, 2, 3], list) == [1, 2, 3]
    assert _cast_argument("[1, 2, 3]", list) == [1, 2, 3]
    assert _cast_argument(("a", "b"), list) == ["a", "b"]


def test_cast_argument_dict() -> None:
    """dict cast: JSON string parses to dict, existing dict passes through."""
    assert _cast_argument({"a": 1}, dict) == {"a": 1}
    assert _cast_argument('{"a": 1}', dict) == {"a": 1}


def test_cast_argument_unknown_annotation_passthrough() -> None:
    """Unknown annotations return the value unchanged."""

    class Custom:
        pass

    obj = Custom()
    assert _cast_argument(obj, Custom) is obj
    # Non-primitive annotation: leave value alone.
    assert _cast_argument("anything", object) == "anything"
