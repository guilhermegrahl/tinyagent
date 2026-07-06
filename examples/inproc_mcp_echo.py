# SPDX-License-Identifier: Apache-2.0
"""In-process MCP echo/calc stdio server — fixture for T10 tests.

Speaks MCP JSON-RPC over stdio. Exposes two tools:

- ``echo``: returns its argument unchanged.
- ``add``: sums two integer arguments.

This script is *only* intended as a fixture for the T10 test suite
(``tests/test_mcp_stdio.py``). It uses the low-level ``mcp.server.Server``
API so the test can spawn it as a real subprocess and exercise the
``MCPServer`` lifecycle (connect / list_tools / call_tool / cleanup) end
to end.

The script never daemonises — it runs as a long-lived process and
exits only when the parent closes its stdin (which is the MCP spec's
graceful-shutdown signal). On SIGTERM / SIGKILL the process dies
without cleanup, which is what the T10 cleanup tests verify.

Usage in tests:

    proc = subprocess.Popen(
        [sys.executable, "examples/inproc_mcp_echo.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    server = MCPServer(
        name="inproc",
        command=sys.executable,
        args=["examples/inproc_mcp_echo.py"],
    )
"""
from __future__ import annotations

import asyncio
import sys

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server


# A single module-level Server instance. mcp wires the request handlers
# to it via decorator methods, then the ``run`` coroutine drives the
# stdio transport.
server: Server = Server("inproc-mcp-echo")


@server.list_tools()
async def _list_tools() -> list[types.Tool]:
    """Advertise the two fixture tools to the client."""
    return [
        types.Tool(
            name="echo",
            description="Echo the input string back to the caller.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
                "required": ["text"],
            },
        ),
        types.Tool(
            name="add",
            description="Add two integers and return the sum.",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        ),
    ]


@server.call_tool()
async def _call_tool(
    name: str, arguments: dict[str, object]
) -> list[types.TextContent]:
    """Dispatch a tool call to the matching fixture implementation."""
    if name == "echo":
        text = str(arguments.get("text", ""))
        return [types.TextContent(type="text", text=text)]
    if name == "add":
        a = int(arguments.get("a", 0))  # type: ignore[arg-type]
        b = int(arguments.get("b", 0))  # type: ignore[arg-type]
        return [types.TextContent(type="text", text=str(a + b))]
    # Mirror the MCP spec: an unknown tool name is a protocol error,
    # not a tool-level success. mcp propagates this back to the caller
    # as a CallToolResult with isError=True; the T10 tests assert the
    # MCPServer surfaces it as ``ToolNotFoundError`` on the client side.
    raise ValueError(f"unknown tool: {name!r}")


async def _main() -> None:
    """Run the server over stdio. Returns when stdin closes."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except (KeyboardInterrupt, SystemExit):
        # Graceful exit on Ctrl-C / parent SIGTERM. Errors during
        # shutdown are not interesting for the test fixture.
        pass
