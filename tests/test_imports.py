"""T1 acceptance test: tinyagent is importable and every __all__ symbol resolves.

Per plan §13 T1 and §10 (canonical __all__ list).
"""
from __future__ import annotations

import importlib
import sys
from typing import Any


EXPECTED_ALL: list[str] = [
    # core
    "TinyAgent",
    "AgentConfig",
    "tool",
    # MCP
    "MCPServer",
    "add_mcp_server",
    "MCPTool",
    # callbacks
    "CallbackRegistry",
    "Context",
    "ToolCall",
    # tracing
    "AgentTrace",
    "AgentSpan",
    "TokenInfo",
    "CostInfo",
    # exceptions
    "AgentError",
    "AgentCancel",
    "ToolNotFoundError",
    # example tools
    "calculate",
    "http_get",
    "final_answer",
    # test-helper exports
    "PROVIDER_KEY_ENV",
    "PROVIDER_EXTRA_ENV",
    # misc
    "__version__",
]


def test_import_top_level() -> None:
    """`import tinyagent` succeeds at the top level (flat layout, §3)."""
    sys.modules.pop("tinyagent", None)
    module = importlib.import_module("tinyagent")
    assert module is not None
    # Module must define __all__ per plan §10.
    assert hasattr(module, "__all__"), "tinyagent must define __all__"
    assert isinstance(module.__all__, list)


def test_all_symbols_importable() -> None:
    """Every name in tinyagent.__all__ resolves via `from tinyagent import name`."""
    tinyagent = importlib.import_module("tinyagent")
    assert hasattr(tinyagent, "__all__")
    for name in tinyagent.__all__:
        assert hasattr(tinyagent, name), (
            f"tinyagent.__all__ lists {name!r} but the module has no such attribute"
        )
        value: Any = getattr(tinyagent, name)
        assert value is not None, f"tinyagent.{name} resolved to None"


def test_all_contents_match_canonical() -> None:
    """__all__ is exactly the canonical list from plan §10 (set equality, order-insensitive)."""
    tinyagent = importlib.import_module("tinyagent")
    assert set(tinyagent.__all__) == set(EXPECTED_ALL), (
        f"__all__ drift: module has {sorted(tinyagent.__all__)}, "
        f"canonical has {sorted(EXPECTED_ALL)}"
    )


def test_version_string() -> None:
    """__version__ is a non-empty string (per spec: 0.1.0)."""
    tinyagent = importlib.import_module("tinyagent")
    assert isinstance(tinyagent.__version__, str)
    assert tinyagent.__version__, "tinyagent.__version__ must be a non-empty string"


def test_public_exceptions_subclass_agent_error() -> None:
    """User-facing exception hierarchy: AgentCancel and ToolNotFoundError subclass AgentError.

    Locks the contract so downstream tasks (T3, T4, ...) preserve it.
    """
    tinyagent = importlib.import_module("tinyagent")
    agent_error = tinyagent.AgentError
    assert issubclass(tinyagent.AgentCancel, agent_error)
    assert issubclass(tinyagent.ToolNotFoundError, agent_error)
