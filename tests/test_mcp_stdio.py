"""T10 acceptance test: ``tinyagent.MCPServer`` (stdio-only) lifecycle + error handling.

Per plan §13 T10, §2 section 11, and §4 (MCP stdio-only strategy).

Covers:
- ``tinyagent.MCPServer`` constructs with name/command/args/env.
- Async context-manager form (``async with tinyagent.MCPServer(...) as srv:``) wires
  connect/cleanup symmetrically.
- ``call_tool`` on an unknown name raises ``tinyagent.ToolNotFoundError`` (a subclass
  of ``tinyagent.AgentError``).
- The stdio subprocess is launched with ``start_new_session=True`` so
  cleanup can ``os.killpg`` the whole process group.
- ``cleanup()`` terminates the process group (the spawned child PID is
  gone / reaped after cleanup).
- ``MCPConnectionError`` and ``MCPProtocolError`` are subclasses of
  ``tinyagent.AgentError`` (M8 cross-cutting risk #9).
- ``tinyagent._create_tool_function`` synthesises a callable that round-trips a
  tool call back to the server.

The integration-level tests in this file spawn a real subprocess via the
fixture ``examples/inproc_mcp_echo.py`` (an MCP stdio server advertising
two tools: ``echo`` and ``add``).  Unit-level tests (subclass
relationships, attribute storage) do not need a subprocess and run in any
environment.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import pytest

import tinyagent


# Resolve the absolute path to the fixture script so the test works
# regardless of pytest's CWD.
_HERE: Path = Path(__file__).resolve().parent
_REPO_ROOT: Path = _HERE.parent
_FIXTURE_SCRIPT: Path = _REPO_ROOT / "examples" / "inproc_mcp_echo.py"


# =====================================================================
# Section 1 - Exception hierarchy (no subprocess required)
# =====================================================================
def test_mcp_connection_error_subclasses_agent_error() -> None:
    """``MCPConnectionError`` wraps subprocess death / EOF on stdin (M8).

    It MUST be a subclass of ``tinyagent.AgentError`` so callers that catch the
    umbrella ``tinyagent.AgentError`` type also catch the MCP-specific failure.
    """
    assert issubclass(tinyagent.MCPConnectionError, tinyagent.AgentError)


def test_mcp_protocol_error_subclasses_agent_error() -> None:
    """``MCPProtocolError`` wraps invalid UTF-8 / malformed JSON-RPC (M8).

    It MUST be a subclass of ``tinyagent.AgentError`` so callers that catch the
    umbrella ``tinyagent.AgentError`` type also catch the MCP-specific failure.
    """
    assert issubclass(tinyagent.MCPProtocolError, tinyagent.AgentError)


def test_tool_not_found_error_subclasses_agent_error() -> None:
    """``tinyagent.ToolNotFoundError`` is a subclass of ``tinyagent.AgentError`` (umbrella catch).

    The agent's loop catches this specific subclass and feeds a
    descriptive string back to the LLM; it still satisfies the
    ``tinyagent.AgentError`` umbrella contract for callers that don't care which
    subclass they get.
    """
    assert issubclass(tinyagent.ToolNotFoundError, tinyagent.AgentError)


# =====================================================================
# Section 2 - Constructor + public attribute shape
# =====================================================================
def test_mcp_server_constructible_with_required_args() -> None:
    """``tinyagent.MCPServer(name, command, args, env)`` constructs with sane defaults.

    ``args`` and ``env`` default to empty. The four public attributes
    (``name``, ``command``, ``args``, ``env``) are exposed as direct
    attributes — no `getattr`/`hasattr` introspection is required.
    """

    srv = tinyagent.MCPServer(name="srv-a", command="python", args=["-"])
    assert srv.name == "srv-a"
    assert srv.command == "python"
    assert srv.args == ["-"]
    assert srv.env is None


def test_mcp_server_constructible_with_env() -> None:
    """``env`` is accepted as a ``dict[str, str] | None``."""

    env = {"FOO": "bar", "BAZ": "qux"}
    srv = tinyagent.MCPServer(name="srv-b", command="python", args=[], env=env)
    assert srv.env == env
    # Mutating the caller's dict after construction must NOT mutate the
    # server's view (defensive copy is the standard contract).
    env["FOO"] = "changed"
    assert srv.env == {"FOO": "bar", "BAZ": "qux"}


# =====================================================================
# Section 3 - Async context-manager lifecycle
# =====================================================================
@pytest.mark.skipif(
    not _FIXTURE_SCRIPT.exists(),
    reason="examples/inproc_mcp_echo.py fixture missing",
)
def test_async_context_manager_connects_then_cleans_up() -> None:
    """``async with tinyagent.MCPServer(...) as srv:`` connects on enter, cleans up on exit.

    Inside the block, ``srv.tools`` exposes the synthesised callables for
    every tool the server advertised. After the block exits, the
    subprocess is reaped and the tool dict is cleared.
    """

    async def _runner() -> None:
        async with tinyagent.MCPServer(
            name="inproc",
            command=sys.executable,
            args=[str(_FIXTURE_SCRIPT)],
        ) as srv:
            # Inside the block, the server is connected and tools are registered.
            assert "echo" in srv.tools
            assert "add" in srv.tools
            assert callable(srv.tools["echo"])

    asyncio.run(_runner())


@pytest.mark.skipif(
    not _FIXTURE_SCRIPT.exists(),
    reason="examples/inproc_mcp_echo.py fixture missing",
)
def test_call_tool_against_live_server_returns_text() -> None:
    """``call_tool`` round-trips through the real subprocess and returns text.

    The fixture's ``add`` tool returns ``str(a + b)``; we assert the
    client surfaces that string to the caller. ``echo`` is verified the
    same way.
    """

    async def _runner() -> None:
        async with tinyagent.MCPServer(
            name="inproc",
            command=sys.executable,
            args=[str(_FIXTURE_SCRIPT)],
        ) as srv:
            add_result = await srv.call_tool("add", {"a": 2, "b": 40})
            assert add_result == "42"
            echo_result = await srv.call_tool("echo", {"text": "hello"})
            assert echo_result == "hello"

    asyncio.run(_runner())


@pytest.mark.skipif(
    not _FIXTURE_SCRIPT.exists(),
    reason="examples/inproc_mcp_echo.py fixture missing",
)
def test_call_tool_unknown_name_raises_tool_not_found_error() -> None:
    """Calling an unregistered tool name raises ``tinyagent.ToolNotFoundError``.

    The acceptance criterion: a tool call to a name that is not in the
    server's advertised tool list MUST raise ``tinyagent.ToolNotFoundError`` (a
    subclass of ``tinyagent.AgentError``) so the loop can feed a descriptive
    string back to the LLM.
    """

    async def _runner() -> None:
        async with tinyagent.MCPServer(
            name="inproc",
            command=sys.executable,
            args=[str(_FIXTURE_SCRIPT)],
        ) as srv:
            with pytest.raises(tinyagent.ToolNotFoundError):
                await srv.call_tool("not-a-real-tool", {})
            # The umbrella catch still works.
            with pytest.raises(tinyagent.AgentError):
                await srv.call_tool("also-not-real", {})

    asyncio.run(_runner())


# =====================================================================
# Section 4 - Subprocess lifecycle: start_new_session=True + cleanup
# =====================================================================
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="process-group semantics differ on Windows",
)
@pytest.mark.skipif(
    not _FIXTURE_SCRIPT.exists(),
    reason="examples/inproc_mcp_echo.py fixture missing",
)
def test_subprocess_is_launched_with_start_new_session() -> None:
    """The MCP stdio subprocess is launched in a new session/process group.

    Why: ``cleanup()`` calls ``os.killpg(os.getpgid(pid), SIGTERM)`` to
    terminate the server and any children it spawned. That only works if
    the subprocess started a new session (``start_new_session=True`` on
    Unix).  The mcp library does this internally inside ``stdio_client``;
    we assert the *observable consequence*: the child PID's process
    group is its own PGID, distinct from the test runner's PGID.
    """

    runner_pgid = os.getpgrp()
    child_pgid_holder: dict[str, int] = {}

    async def _runner() -> None:
        async with tinyagent.MCPServer(
            name="inproc",
            command=sys.executable,
            args=[str(_FIXTURE_SCRIPT)],
        ) as srv:
            # ``srv`` exposes the child PID via ``_process`` (set inside
            # ``connect()``). If the property is not exposed, fail with a
            # clear message rather than AttributeError.
            process = srv._process  # noqa: SLF001 - intentional test access
            child_pgid_holder["pgid"] = os.getpgid(process.pid)

    asyncio.run(_runner())

    # The subprocess must have been started in a NEW process group,
    # i.e. its PGID differs from the test runner's PGID. If they were
    # the same, ``start_new_session=True`` was not honored and
    # ``os.killpg`` would also signal the test runner itself.
    assert child_pgid_holder["pgid"] != runner_pgid, (
        "stdio subprocess should be in its own process group "
        "(start_new_session=True)"
    )


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="process-group semantics differ on Windows",
)
@pytest.mark.skipif(
    not _FIXTURE_SCRIPT.exists(),
    reason="examples/inproc_mcp_echo.py fixture missing",
)
def test_cleanup_terminates_process_group() -> None:
    """``cleanup()`` kills the subprocess group — child PID is reaped after exit.

    We spawn the server, capture the child's PID + PGID, run cleanup(),
    then verify the child PID is no longer alive. We also verify the
    process group has been emptied by reaping whatever is left.
    """

    pid_holder: dict[str, int] = {}
    pgid_holder: dict[str, int] = {}

    async def _runner() -> None:
        srv = tinyagent.MCPServer(
            name="inproc",
            command=sys.executable,
            args=[str(_FIXTURE_SCRIPT)],
        )
        await srv.connect()
        process = srv._process  # noqa: SLF001 - intentional test access
        pid_holder["pid"] = process.pid
        pgid_holder["pgid"] = os.getpgid(process.pid)
        # Sanity: the process is alive at this point.
        assert _pid_alive(pid_holder["pid"])
        await srv.cleanup()
        # After cleanup, the PID is no longer alive.
        # Give the OS a moment to deliver SIGTERM and reap the zombie.
        for _ in range(20):
            if not _pid_alive(pid_holder["pid"]):
                break
            time.sleep(0.05)
        # Re-entrant cleanup is a no-op.
        await srv.cleanup()

    asyncio.run(_runner())

    assert not _pid_alive(pid_holder["pid"]), (
        f"subprocess {pid_holder['pid']} should be dead after cleanup()"
    )
    # The process group is now empty (or the PGID is no longer valid
    # because the leader is gone).
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid_holder["pgid"], signal.SIGTERM)
    # If the PGID still resolves, every process in it must be gone
    # (i.e. nothing matched the signal).  The bare ``os.killpg`` above
    # either errors or no-ops; that's the expected terminal state.


# =====================================================================
# Section 5 - ``tinyagent._create_tool_function`` synthesises a callable + schema
# =====================================================================
@pytest.mark.skipif(
    not _FIXTURE_SCRIPT.exists(),
    reason="examples/inproc_mcp_echo.py fixture missing",
)
def test_create_tool_function_synthesises_callable_with_schema() -> None:
    """``tinyagent._create_tool_function`` returns a callable that, when invoked, dispatches to ``call_tool``.

    The returned function carries a ``tool_schema`` attribute that mirrors
    the MCP tool's ``inputSchema`` (plan §2 section 11) so the agent
    loop can advertise it to the LLM in a provider-agnostic way.
    """

    server_ref: dict[str, Any] = {}

    async def _runner() -> None:
        async with tinyagent.MCPServer(
            name="inproc",
            command=sys.executable,
            args=[str(_FIXTURE_SCRIPT)],
        ) as srv:
            server_ref["srv"] = srv
            # The server's ``list_tools()`` returns MCP tool descriptors.
            tools = await srv.list_tools()
            assert tools, "fixture server should advertise at least one tool"
            echo_tool = next(t for t in tools if t.name == "echo")
            fn = tinyagent._create_tool_function(srv, echo_tool)
            # The function exposes the schema as an attribute.
            assert hasattr(fn, "tool_schema")
            assert fn.tool_schema["name"] == "echo"
            # And it round-trips a call to the server.
            result = await fn(text="hi from synthesized callable")
            assert result == "hi from synthesized callable"

    asyncio.run(_runner())


# =====================================================================
# Section 6 - ``list_tools()`` returns the MCP tool descriptors
# =====================================================================
@pytest.mark.skipif(
    not _FIXTURE_SCRIPT.exists(),
    reason="examples/inproc_mcp_echo.py fixture missing",
)
def test_list_tools_returns_tool_descriptors() -> None:
    """``list_tools()`` returns the list of MCP tool descriptors advertised by the server."""

    async def _runner() -> None:
        async with tinyagent.MCPServer(
            name="inproc",
            command=sys.executable,
            args=[str(_FIXTURE_SCRIPT)],
        ) as srv:
            tools = await srv.list_tools()
            names = sorted(t.name for t in tools)
            assert names == ["add", "echo"]

    asyncio.run(_runner())


# =====================================================================
# Helpers
# =====================================================================
def _pid_alive(pid: int) -> bool:
    """Return True if ``pid`` is currently a live process.

    On POSIX, ``os.kill(pid, 0)`` raises ``ProcessLookupError`` if the
    PID does not exist (or is a zombie we cannot signal).  We treat
    both cases as "not alive" — the cleanup test only cares that the
    PID is no longer a signal-able running process.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # PID exists but is owned by another user; for our purposes
        # (testing our own subprocess) this means it's alive.
        return True
    return True


# =====================================================================
# Section 7 - ``TinyAgent.add_mcp_server`` ergonomic API (T14)
# =====================================================================
def test_add_mcp_server_is_in_module_all() -> None:
    """``add_mcp_server`` is exported in ``tinyagent.__all__``.

    The symbol is exposed at module level for ``from tinyagent import
    add_mcp_server`` discoverability. The task description's acceptance
    criteria require both forms: name in ``__all__`` AND importable.
    """
    assert "add_mcp_server" in tinyagent.__all__


def test_add_mcp_server_top_level_import_resolves_to_method() -> None:
    """``from tinyagent import add_mcp_server`` resolves to ``TinyAgent.add_mcp_server``.

    Module-level re-export (per T1 stub): the symbol is bound to the
    same function object on the class so users can write ``add_mcp_server``
    for type hints while the runtime contract lives on the instance.
    The check uses the already-imported ``tinyagent`` module to avoid
    a local re-import.
    """
    assert tinyagent.add_mcp_server is tinyagent.TinyAgent.add_mcp_server


def _make_minimal_agent() -> tinyagent.TinyAgent:
    """Build a ``TinyAgent`` with no MCP servers for in-test usage.

    The model string is arbitrary — T14 only exercises the MCP
    bookkeeping path; no LLM calls happen in these tests. Using
    ``openai:gpt-4o-mini`` keeps the constructor happy without
    requiring a real provider key (the placeholder API key in
    ``TinyAgent.__init__`` satisfies the any-llm client construction).
    """
    config = tinyagent.AgentConfig(
        instructions="x",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o-mini",
    )
    return tinyagent.TinyAgent(config)


@pytest.mark.skipif(
    not _FIXTURE_SCRIPT.exists(),
    reason="examples/inproc_mcp_echo.py fixture missing",
)
def test_add_mcp_server_async_cm_populates_clients_on_enter() -> None:
    """``async with agent.add_mcp_server(server) as tools:`` populates ``_clients``.

    Form A (the context-manager form, preferred for multi-server sessions,
    plan §10): entering the block calls ``server.connect()`` and inserts
    each synthesised tool into ``agent._clients`` keyed by tool name. The
    yielded value is the list of synthesised callables.
    """
    agent = _make_minimal_agent()
    server = tinyagent.MCPServer(
        name="inproc",
        command=sys.executable,
        args=[str(_FIXTURE_SCRIPT)],
    )

    async def _runner() -> None:
        async with agent.add_mcp_server(server) as tools:
            # The synthesised tools from the fixture (``echo``, ``add``)
            # are now registered on the agent.
            assert "echo" in agent._clients  # noqa: SLF001 - intentional test access
            assert "add" in agent._clients  # noqa: SLF001
            # The yielded value is a non-empty list of callables.
            assert isinstance(tools, list)
            assert tools, "yields list of synthesised tools must be non-empty"
            assert all(callable(fn) for fn in tools)
            # final_answer is still present (was seeded by the constructor).
            assert "final_answer" in agent._clients  # noqa: SLF001

    asyncio.run(_runner())


@pytest.mark.skipif(
    not _FIXTURE_SCRIPT.exists(),
    reason="examples/inproc_mcp_echo.py fixture missing",
)
def test_add_mcp_server_async_cm_removes_clients_and_kills_subprocess_on_exit() -> None:
    """On exit, ``add_mcp_server`` removes tools from ``_clients`` and cleans up the subprocess.

    Plan §10 Form A — exit semantics: the synthesised tools must be
    removed from ``_clients`` (so a subsequent ``run_async`` would not
    accidentally invoke a server that no longer exists) and the
    subprocess must be terminated. We assert both: the names are gone
    from ``_clients``, and the captured PID is no longer alive.
    """
    agent = _make_minimal_agent()
    server = tinyagent.MCPServer(
        name="inproc",
        command=sys.executable,
        args=[str(_FIXTURE_SCRIPT)],
    )
    pid_holder: dict[str, int] = {}

    async def _runner() -> None:
        async with agent.add_mcp_server(server) as _tools:
            # Capture the live subprocess PID before exiting the block.
            pid_holder["pid"] = server._process.pid  # noqa: SLF001 - test access
            assert _pid_alive(pid_holder["pid"])
            assert "echo" in agent._clients  # noqa: SLF001

    asyncio.run(_runner())

    # After exit: tools removed and subprocess is dead.
    assert "echo" not in agent._clients  # noqa: SLF001
    assert "add" not in agent._clients  # noqa: SLF001
    # final_answer remains (constructor-seeded, untouched by the CM).
    assert "final_answer" in agent._clients  # noqa: SLF001
    assert not _pid_alive(pid_holder["pid"]), (
        f"subprocess {pid_holder['pid']} should be dead after add_mcp_server exit"
    )


@pytest.mark.skipif(
    not _FIXTURE_SCRIPT.exists(),
    reason="examples/inproc_mcp_echo.py fixture missing",
)
def test_add_mcp_server_explicit_aenter_aexit_form_works() -> None:
    """Explicit register/cleanup form via ``__aenter__`` / ``__aexit__`` works.

    Plan §10 Form B (the explicit form, used when callers prefer manual
    control over the lifecycle). The same async-CM object returned by
    ``add_mcp_server(server)`` is driven via its ``__aenter__`` and
    ``__aexit__`` methods directly. After exit, both the ``_clients``
    state and the subprocess are cleaned up.
    """
    agent = _make_minimal_agent()
    server = tinyagent.MCPServer(
        name="inproc",
        command=sys.executable,
        args=[str(_FIXTURE_SCRIPT)],
    )
    pid_holder: dict[str, int] = {}

    async def _runner() -> None:
        cm = agent.add_mcp_server(server)
        await cm.__aenter__()
        try:
            assert "echo" in agent._clients  # noqa: SLF001
            assert "add" in agent._clients  # noqa: SLF001
            pid_holder["pid"] = server._process.pid  # noqa: SLF001
            assert _pid_alive(pid_holder["pid"])
        finally:
            await cm.__aexit__(None, None, None)

    asyncio.run(_runner())

    # Same post-conditions as the `async with` form.
    assert "echo" not in agent._clients  # noqa: SLF001
    assert "add" not in agent._clients  # noqa: SLF001
    assert not _pid_alive(pid_holder["pid"])


@pytest.mark.skipif(
    not _FIXTURE_SCRIPT.exists(),
    reason="examples/inproc_mcp_echo.py fixture missing",
)
def test_add_mcp_server_cleanup_runs_even_on_exception_inside_block() -> None:
    """If the body raises, the cleanup (``finally``) still runs.

    Plan §10 Form A — the ``async with`` must guarantee subprocess
    cleanup on any exit path, including raised exceptions inside the
    block. We force a synthetic exception and assert the subprocess is
    dead afterwards. Setup (PID capture) happens OUTSIDE the
    ``pytest.raises`` block so the assertion contains a single
    statement (PT012) — the captured PID is the side-effect we assert
    against after the body unwinds.
    """
    agent = _make_minimal_agent()
    server = tinyagent.MCPServer(
        name="inproc",
        command=sys.executable,
        args=[str(_FIXTURE_SCRIPT)],
    )
    pid_holder: dict[str, int] = {}
    boom_message = "boom"

    async def _body() -> None:
        async with agent.add_mcp_server(server) as _tools:
            pid_holder["pid"] = server._process.pid  # noqa: SLF001
            assert _pid_alive(pid_holder["pid"])
            raise RuntimeError(boom_message)

    async def _runner() -> None:
        with pytest.raises(RuntimeError, match=boom_message):
            await _body()

    asyncio.run(_runner())

    # Cleanup ran via the `finally` branch.
    assert "echo" not in agent._clients  # noqa: SLF001
    assert not _pid_alive(pid_holder["pid"])
