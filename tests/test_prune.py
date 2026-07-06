"""T12b acceptance test: pair-preserving message prune.

Per plan §2 section 14, §9 (algorithm), §13 T12b, and round-2 M5 (pair preservation).

The invariant is: for every tool-role message that survives pruning, its
parent assistant message (with matching ``tool_call_id``) also survives. The
algorithm walks units of (assistant + its tool follow-ups) and never breaks
an (assistant, [tool, ...]) pairing.

Tests cover:
  - Empty history is no-op
  - History shorter than keep_last_n is unchanged
  - System message preserved at index 0
  - Every surviving tool message has its parent assistant (no orphans)
  - Pruning drops whole (assistant, [tool, tool]) blocks from the head, not partial blocks
  - Edge case: trailing assistant message with no tools is preserved as-is
"""
from __future__ import annotations

from typing import Any

import tinyagent


# ---------------------------------------------------------------------------
# Helpers — fixtures for building paired message histories
# ---------------------------------------------------------------------------
def _assistant_with_tools(
    tool_call_ids: list[str], content: str = ""
) -> dict[str, Any]:
    """Build an assistant message that emitted ``tool_call_ids``."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": tc_id,
                "type": "function",
                "function": {"name": "calculate", "arguments": "{}"},
            }
            for tc_id in tool_call_ids
        ],
    }


def _tool_result(tool_call_id: str, content: str) -> dict[str, Any]:
    """Build a tool-role message echoing ``tool_call_id``."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }


def _user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def _assistant_text(text: str) -> dict[str, Any]:
    """Assistant with no tool_calls (text-only)."""
    return {"role": "assistant", "content": text}


def _system(text: str) -> dict[str, Any]:
    return {"role": "system", "content": text}


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------
def test_prune_function_is_exposed() -> None:
    """`_prune_messages_keeping_pairs` is accessible on the tinyagent module.

    The function is private (leading underscore) but module-importable so
    the TDD test can drive it directly. Exposing via the module surface
    keeps the contract auditable from one place.
    """
    assert hasattr(tinyagent, "_prune_messages_keeping_pairs")
    assert callable(tinyagent._prune_messages_keeping_pairs)


# ---------------------------------------------------------------------------
# Empty / smaller-than-budget histories
# ---------------------------------------------------------------------------
def test_empty_history_is_noop() -> None:
    """An empty messages list returns the same empty list (no mutation)."""
    out = tinyagent._prune_messages_keeping_pairs([], keep_last_n=5)
    assert out == [], f"empty history must be a no-op, got {out!r}"


def test_history_shorter_than_keep_last_n_is_unchanged() -> None:
    """A body that fits within `keep_last_n` is returned intact."""
    body = [
        _assistant_with_tools(["call_a"]),
        _tool_result("call_a", "42"),
        _assistant_with_tools(["call_b"]),
        _tool_result("call_b", "43"),
    ]
    out = tinyagent._prune_messages_keeping_pairs(body, keep_last_n=10)
    assert out == body, f"shorter body must be unchanged, got {out!r}"


def test_shorter_than_keep_last_n_with_system_is_unchanged() -> None:
    """A system-prompted body that fits within keep_last_n is returned intact."""
    body: list[dict[str, Any]] = [_system("you are helpful")]
    body.append(_assistant_with_tools(["call_a"]))
    body.append(_tool_result("call_a", "42"))

    out = tinyagent._prune_messages_keeping_pairs(body, keep_last_n=10)
    assert out == body


# ---------------------------------------------------------------------------
# System message preservation
# ---------------------------------------------------------------------------
def test_system_message_survives_at_index_zero() -> None:
    """When keep_last_n < body length, the system message still leads the result."""
    body: list[dict[str, Any]] = [_system("you are helpful")]
    # Build ten assistant+tool pairs so the body strictly exceeds keep_last_n=4.
    for i in range(10):
        tc_id = f"call_{i}"
        body.append(_assistant_with_tools([tc_id]))
        body.append(_tool_result(tc_id, str(i)))

    out = tinyagent._prune_messages_keeping_pairs(body, keep_last_n=4)

    assert out, "prune must return a non-empty list"
    assert out[0] == _system("you are helpful"), (
        f"system message must remain at index 0, got {out[0]!r}"
    )
    assert out[0].get("role") == "system"


def test_system_message_absent_when_not_in_history() -> None:
    """When the input has no leading system message, no synthetic one is added."""
    body: list[dict[str, Any]] = [
        _assistant_with_tools(["call_a"]),
        _tool_result("call_a", "42"),
        _assistant_with_tools(["call_b"]),
        _tool_result("call_b", "43"),
        _assistant_with_tools(["call_c"]),
        _tool_result("call_c", "44"),
    ]
    out = tinyagent._prune_messages_keeping_pairs(body, keep_last_n=2)

    # After pruning the head, we expect the latest two pairs.
    # None of them should be a system message.
    assert all(m.get("role") != "system" for m in out), (
        f"no synthetic system message should be injected, got {out!r}"
    )


# ---------------------------------------------------------------------------
# Pair-preservation invariant (round-2 M5 — the central rule)
# ---------------------------------------------------------------------------
def _assert_pair_invariant(messages: list[dict[str, Any]]) -> None:
    """For every tool-role message, the matching assistant message exists.

    Walks the history and, per tool-role message, scans backwards for the
    first assistant message with ``tool_calls`` containing a matching
    ``tool_call_id``. If none exists within the window, the invariant is
    violated (orphan tool message).
    """
    for idx, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        target_id: str | None = msg.get("tool_call_id")
        assert target_id is not None, (
            f"tool message at index {idx} missing tool_call_id: {msg!r}"
        )
        # Scan backwards from idx-1 to find the parent assistant.
        found_parent = False
        for j in range(idx - 1, -1, -1):
            candidate = messages[j]
            if candidate.get("role") != "assistant":
                continue
            tool_calls = candidate.get("tool_calls") or []
            if any(tc.get("id") == target_id for tc in tool_calls):
                found_parent = True
                break
        assert found_parent, (
            f"orphan tool message at index {idx} "
            f"(tool_call_id={target_id!r}); no parent assistant found"
        )


def test_pair_invariant_holds_after_pruning() -> None:
    """After pruning, every tool message has its parent assistant message.

    Builds a long history (15 paired units) and prunes to keep_last_n=4.
    Then asserts the invariant across the pruned list.
    """
    body: list[dict[str, Any]] = []
    for i in range(15):
        tc_id = f"call_{i:02d}"
        body.append(_assistant_with_tools([tc_id]))
        body.append(_tool_result(tc_id, f"result_{i}"))

    out = tinyagent._prune_messages_keeping_pairs(body, keep_last_n=4)

    _assert_pair_invariant(out)


def test_pair_invariant_holds_with_multi_tool_units() -> None:
    """Pruning drops whole (assistant, [tool, tool, tool]) units — no partial blocks.

    Builds history with units of varying tool-result cluster sizes and
    asserts that no surviving tool message is an orphan after pruning.
    """
    body: list[dict[str, Any]] = []
    # Unit 1: 1 tool result
    body.append(_assistant_with_tools(["a1"]))
    body.append(_tool_result("a1", "1"))
    # Unit 2: 2 tool results
    body.append(_assistant_with_tools(["b1", "b2"]))
    body.append(_tool_result("b1", "2a"))
    body.append(_tool_result("b2", "2b"))
    # Unit 3: 3 tool results
    body.append(_assistant_with_tools(["c1", "c2", "c3"]))
    body.append(_tool_result("c1", "3a"))
    body.append(_tool_result("c2", "3b"))
    body.append(_tool_result("c3", "3c"))
    # Unit 4: 1 tool result
    body.append(_assistant_with_tools(["d1"]))
    body.append(_tool_result("d1", "4"))
    # Unit 5: 2 tool results
    body.append(_assistant_with_tools(["e1", "e2"]))
    body.append(_tool_result("e1", "5a"))
    body.append(_tool_result("e2", "5b"))
    # Unit 6: 1 tool result
    body.append(_assistant_with_tools(["f1"]))
    body.append(_tool_result("f1", "6"))

    out = tinyagent._prune_messages_keeping_pairs(body, keep_last_n=4)

    _assert_pair_invariant(out)
    # Unit 6 (the most recent) should still be in there.
    assert _tool_result("f1", "6") in out
    # The earliest unit's tools must NOT survive (pruning dropped them).
    assert _tool_result("a1", "1") not in out


# ---------------------------------------------------------------------------
# Block-level dropping (never split a block)
# ---------------------------------------------------------------------------
def test_pruning_drops_whole_blocks_not_partial_blocks() -> None:
    """Head-dropping a partial (assistant + 1 tool) is forbidden.

    Builds a body whose head unit is (assistant_with_calls=[a], tool=[a]).
    keep_last_n=2 forces the algorithm to drop the head. The head unit
    has 2 messages (assistant + 1 tool); dropping only the assistant and
    keeping the tool would orphan the tool — the implementation MUST
    drop the WHOLE head unit instead.
    """
    body: list[dict[str, Any]] = []
    # Unit to drop (head): 1 assistant + 1 tool
    body.append(_assistant_with_tools(["a"]))
    body.append(_tool_result("a", "head"))
    # Two units to keep
    body.append(_assistant_with_tools(["b"]))
    body.append(_tool_result("b", "middle"))
    body.append(_assistant_with_tools(["c"]))
    body.append(_tool_result("c", "tail"))

    out = tinyagent._prune_messages_keeping_pairs(body, keep_last_n=2)

    # The result should NOT contain an orphan tool message.
    _assert_pair_invariant(out)
    # Head tool result must be gone — its parent assistant is also gone.
    assert _tool_result("a", "head") not in out
    # All two messages of the head unit are dropped:
    head_unit_msgs = [
        body[0],  # assistant with calls=[a]
        body[1],  # tool result for a
    ]
    for m in head_unit_msgs:
        assert m not in out, f"head-unit member survived pruning: {m!r}"


def test_pruning_drops_only_head_not_middle() -> None:
    """Pruning drops earlier units; the body order is otherwise preserved."""
    body: list[dict[str, Any]] = []
    for i in range(8):
        tc_id = f"call_{i}"
        body.append(_assistant_with_tools([tc_id]))
        body.append(_tool_result(tc_id, f"r_{i}"))

    out = tinyagent._prune_messages_keeping_pairs(body, keep_last_n=4)

    # Length: keep_last_n=4 units means assistant + tool for each.
    assert len(out) == 8, f"expected 8 messages in pruned list, got {len(out)}"
    # Oldest four tool_call_ids ('call_0'..'call_3') must be absent.
    for i in range(4):
        assert _tool_result(f"call_{i}", f"r_{i}") not in out, (
            f"unit {i} survived pruning — head drop was incomplete"
        )
    # Newest four must be present and in original order.
    expected_tail: list[dict[str, Any]] = []
    for i in range(4, 8):
        expected_tail.append(_assistant_with_tools([f"call_{i}"]))
        expected_tail.append(_tool_result(f"call_{i}", f"r_{i}"))
    assert out == expected_tail, (
        f"tail order/contents wrong after pruning.\n"
        f"expected={expected_tail!r}\nactual={out!r}"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_trailing_assistant_without_tools_is_preserved() -> None:
    """An assistant message with NO tool_calls is its own unit (trailing-text case).

    The plan §8 trailing-text fallback can leave the history with a final
    assistant message that has no tool_calls. Pruning must treat that as
    a standalone unit, not split/merge it with anything.
    """
    body: list[dict[str, Any]] = []
    # Old paired unit that should be pruned.
    body.append(_assistant_with_tools(["old"]))
    body.append(_tool_result("old", "old_result"))
    # Newer paired unit (kept).
    body.append(_assistant_with_tools(["kept"]))
    body.append(_tool_result("kept", "kept_result"))
    # Trailing assistant with no tool_calls.
    body.append(_assistant_text("the answer is 42"))

    out = tinyagent._prune_messages_keeping_pairs(body, keep_last_n=2)

    _assert_pair_invariant(out)
    # The trailing assistant message must survive.
    assert _assistant_text("the answer is 42") in out
    # The new paired unit survives.
    assert _tool_result("kept", "kept_result") in out
    # The old paired unit is gone.
    assert _tool_result("old", "old_result") not in out


def test_keep_last_n_zero_returns_only_system() -> None:
    """keep_last_n=0 drops every body unit; only the system message survives."""
    body: list[dict[str, Any]] = [_system("you are helpful")]
    body.append(_assistant_with_tools(["a"]))
    body.append(_tool_result("a", "1"))
    body.append(_assistant_with_tools(["b"]))
    body.append(_tool_result("b", "2"))

    out = tinyagent._prune_messages_keeping_pairs(body, keep_last_n=0)

    assert out == [_system("you are helpful")], (
        f"keep_last_n=0 must keep only the system message, got {out!r}"
    )


def test_does_not_mutate_input_list() -> None:
    """The prune function must NOT mutate its input list (defensive contract)."""
    body: list[dict[str, Any]] = []
    for i in range(6):
        tc_id = f"call_{i}"
        body.append(_assistant_with_tools([tc_id]))
        body.append(_tool_result(tc_id, f"r_{i}"))
    snapshot = list(body)
    _ = tinyagent._prune_messages_keeping_pairs(body, keep_last_n=2)
    assert body == snapshot, "input list was mutated by prune"


# ---------------------------------------------------------------------------
# Behavioral sanity: a synthetic OpenAI-style guard would accept the result
# ---------------------------------------------------------------------------
def _would_openai_accept(messages: list[dict[str, Any]]) -> bool:
    """Simulate OpenAI/Anthropic pairing check.

    A message history is acceptable iff every tool message has at least
    one preceding assistant message whose ``tool_calls`` include the
    ``tool_call_id`` of that tool message. This mirrors what real
    providers enforce (and is the exact invariant the round-2 M5
    pruning bug violated).
    """
    for idx, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        target_id = msg.get("tool_call_id")
        if target_id is None:
            return False
        found = False
        for j in range(idx - 1, -1, -1):
            cand = messages[j]
            if cand.get("role") != "assistant":
                continue
            tcs = cand.get("tool_calls") or []
            if any(tc.get("id") == target_id for tc in tcs):
                found = True
                break
        if not found:
            return False
    return True


def test_openai_style_pairing_guard_accepts_pruned_history() -> None:
    """A pair-checking guard accepts the pruned history (no orphans)."""
    body: list[dict[str, Any]] = [_system("sys")]
    for i in range(20):
        tc_id = f"tc_{i:02d}"
        body.append(_assistant_with_tools([tc_id]))
        body.append(_tool_result(tc_id, f"res_{i}"))

    out = tinyagent._prune_messages_keeping_pairs(body, keep_last_n=6)

    assert _would_openai_accept(out), (
        f"pruned history fails OpenAI-style pairing check: {out!r}"
    )
    # keep_last_n=6 means at most 6 assistant units survive.
    surviving_assistants = [
        m for m in out if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    assert len(surviving_assistants) <= 6, (
        f"pruning kept more assistant units than keep_last_n allows: "
        f"{len(surviving_assistants)}"
    )
