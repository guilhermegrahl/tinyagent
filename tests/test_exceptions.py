"""T3 acceptance test: exception hierarchy for tinyagent.

Per plan §2 section 6, §13 T3, and the cross-cutting risk #9 in plan §13.

Covers:
- Every library exception subclasses ``AgentError`` (single umbrella catch).
- The five exception classes are constructible with a message.
- ``except AgentError:`` catches every subclass (umbrella semantics).
- The public symbols are importable from the ``tinyagent`` top-level.
"""
from __future__ import annotations

import importlib

import pytest


# ---------------------------------------------------------------------
# Subclass-relationship tests (plan §13 cross-cutting risk #9)
# ---------------------------------------------------------------------
def test_agent_error_subclasses_exception() -> None:
    """``AgentError`` is itself an ``Exception`` subclass — callers can catch it as Exception."""
    tinyagent = importlib.import_module("tinyagent")
    assert issubclass(tinyagent.AgentError, Exception)


def test_agent_cancel_subclasses_agent_error() -> None:
    """``AgentCancel`` is raised from hooks to terminate the loop; subclasses ``AgentError``."""
    tinyagent = importlib.import_module("tinyagent")
    assert issubclass(tinyagent.AgentCancel, tinyagent.AgentError)


def test_tool_not_found_error_subclasses_agent_error() -> None:
    """``ToolNotFoundError`` is raised by the dispatcher for unknown tool names; subclasses AgentError."""
    tinyagent = importlib.import_module("tinyagent")
    assert issubclass(tinyagent.ToolNotFoundError, tinyagent.AgentError)


def test_mcp_connection_error_subclasses_agent_error() -> None:
    """``MCPConnectionError`` wraps MCP subprocess death / EOF on stdin; subclasses AgentError (M8)."""
    tinyagent = importlib.import_module("tinyagent")
    assert issubclass(tinyagent.MCPConnectionError, tinyagent.AgentError)


def test_mcp_protocol_error_subclasses_agent_error() -> None:
    """``MCPProtocolError`` wraps invalid UTF-8 / malformed JSON-RPC; subclasses AgentError (M8)."""
    tinyagent = importlib.import_module("tinyagent")
    assert issubclass(tinyagent.MCPProtocolError, tinyagent.AgentError)


# ---------------------------------------------------------------------
# Constructibility tests — every exception takes a string message
# ---------------------------------------------------------------------
def test_agent_error_constructible_with_message() -> None:
    """``AgentError(msg)`` constructs with a string message and exposes it via ``str(e)``."""
    tinyagent = importlib.import_module("tinyagent")
    exc = tinyagent.AgentError("boom")
    assert str(exc) == "boom"
    assert isinstance(exc, Exception)


def test_agent_cancel_constructible_with_message() -> None:
    """``AgentCancel(msg)`` constructs and surfaces the message."""
    tinyagent = importlib.import_module("tinyagent")
    exc = tinyagent.AgentCancel("aborted by hook")
    assert str(exc) == "aborted by hook"


def test_tool_not_found_error_constructible_with_message() -> None:
    """``ToolNotFoundError(name)`` constructs with a descriptive message."""
    tinyagent = importlib.import_module("tinyagent")
    exc = tinyagent.ToolNotFoundError("tool 'foo' is not registered")
    assert "foo" in str(exc)


def test_mcp_connection_error_constructible_with_message() -> None:
    """``MCPConnectionError(msg)`` constructs (subprocess death / EOF on stdin)."""
    tinyagent = importlib.import_module("tinyagent")
    exc = tinyagent.MCPConnectionError("subprocess died: EOF on stdin")
    assert "EOF" in str(exc)


def test_mcp_protocol_error_constructible_with_message() -> None:
    """``MCPProtocolError(msg)`` constructs (invalid UTF-8 / malformed JSON-RPC)."""
    tinyagent = importlib.import_module("tinyagent")
    exc = tinyagent.MCPProtocolError("invalid UTF-8 on stdout")
    assert "UTF-8" in str(exc)


# ---------------------------------------------------------------------
# Umbrella-catch tests — `except AgentError` must catch every subclass
# ---------------------------------------------------------------------
def test_agent_cancel_caught_by_agent_error() -> None:
    """``except AgentError`` catches ``AgentCancel`` (umbrella contract)."""
    tinyagent = importlib.import_module("tinyagent")
    with pytest.raises(tinyagent.AgentError):
        raise tinyagent.AgentCancel("abort")


def test_tool_not_found_error_caught_by_agent_error() -> None:
    """``except AgentError`` catches ``ToolNotFoundError`` (umbrella contract)."""
    tinyagent = importlib.import_module("tinyagent")
    with pytest.raises(tinyagent.AgentError):
        raise tinyagent.ToolNotFoundError("unknown tool")


def test_mcp_connection_error_caught_by_agent_error() -> None:
    """``except AgentError`` catches ``MCPConnectionError`` (umbrella contract)."""
    tinyagent = importlib.import_module("tinyagent")
    with pytest.raises(tinyagent.AgentError):
        raise tinyagent.MCPConnectionError("subprocess died")


def test_mcp_protocol_error_caught_by_agent_error() -> None:
    """``except AgentError`` catches ``MCPProtocolError`` (umbrella contract)."""
    tinyagent = importlib.import_module("tinyagent")
    with pytest.raises(tinyagent.AgentError):
        raise tinyagent.MCPProtocolError("bad frame")


# ---------------------------------------------------------------------
# Public-API import tests — symbols resolve from `tinyagent` top-level
# ---------------------------------------------------------------------
def test_exception_classes_importable_from_top_level() -> None:
    """Every exception class is a module-level attribute of ``tinyagent`` (public API surface)."""
    tinyagent = importlib.import_module("tinyagent")
    for name in (
        "AgentError",
        "AgentCancel",
        "ToolNotFoundError",
        "MCPConnectionError",
        "MCPProtocolError",
    ):
        assert hasattr(tinyagent, name), f"tinyagent.{name} must be importable from top level"
        cls = getattr(tinyagent, name)
        assert isinstance(cls, type), f"tinyagent.{name} must be a class"


def test_exception_classes_have_docstrings() -> None:
    """Every exception class has a docstring explaining when it is raised (task acceptance criterion)."""
    tinyagent = importlib.import_module("tinyagent")
    for name in (
        "AgentError",
        "AgentCancel",
        "ToolNotFoundError",
        "MCPConnectionError",
        "MCPProtocolError",
    ):
        cls = getattr(tinyagent, name)
        doc = cls.__doc__
        assert doc, f"tinyagent.{name} must have a non-empty docstring"
        assert doc.strip(), f"tinyagent.{name} docstring must contain non-whitespace content"
