"""T6 acceptance test: CallbackRegistry (5 hooks, dict-backed, pinned-loop bridge).

Per plan §5, §2 section 7, and §13 T6 (round-3 M3 storage model).

The CallbackRegistry is the only public API for hook registration. The registry
uses ONE storage model: `self._hooks: dict[str, list[Callable]]`. Users call
`register_*` methods. The previous `cb.before_llm_call.append(fn)` form
(round-2 attribute storage) is **dropped** and locked out by a regression test.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING

import pytest

import tinyagent

if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ctx_stub() -> object:
    """Build a minimal stand-in for Context (T7 will fill in the real type).

    T6 treats Context as `object` (per plan §0 C5 contract: "for this task,
    treat it as `object` and pass it through"). The test asserts the registry
    passes the ctx by reference to hooks.
    """
    return object()


def _null_hook(ctx: object) -> None:
    """No-op sync hook for tests that just need a registered function."""


def _capturing_hook_factory(
    sink: list[object],
) -> Callable[[object], None]:
    """Return a sync hook that appends its ctx to `sink`."""

    def _hook(ctx: object) -> None:
        sink.append(ctx)

    return _hook


def _ordered_hook_factory(order: list[int], label: int) -> Callable[[object], None]:
    """Return a sync hook that appends `label` to `order` on call."""

    def _hook(ctx: object) -> None:
        order.append(label)

    return _hook


# ---------------------------------------------------------------------------
# Construction & storage (round-3 M3: dict-backed, no attribute storage)
# ---------------------------------------------------------------------------
def test_callback_registry_is_exported() -> None:
    """CallbackRegistry is part of the public API (plan §10 __all__)."""
    assert hasattr(tinyagent, "CallbackRegistry")
    cb = tinyagent.CallbackRegistry()
    assert isinstance(cb, tinyagent.CallbackRegistry)


def test_uses_slots_for_hooks_and_loop() -> None:
    """__slots__ = ('_hooks', '_loop') — fixed memory layout, no __dict__.

    Locks the public class shape so the registry's storage is auditable from
    `__slots__` alone. Adding a new storage attribute requires editing the
    class, which is the explicit surface for that change.
    """
    assert hasattr(tinyagent.CallbackRegistry, "__slots__"), (
        "CallbackRegistry must declare __slots__ for a fixed memory layout"
    )
    assert tinyagent.CallbackRegistry.__slots__ == ("_hooks", "_loop"), (
        f"CallbackRegistry.__slots__ drifted: {tinyagent.CallbackRegistry.__slots__}"
    )
    cb = tinyagent.CallbackRegistry()
    # The instance has no __dict__ — __slots__ eliminates it.
    assert "__dict__" not in dir(cb), (
        "CallbackRegistry with __slots__ should not have a per-instance __dict__"
    )


def test_internal_storage_is_dict_of_lists() -> None:
    """self._hooks is a dict[str, list[Callable]] keyed by hook name.

    Locks the canonical storage shape from §0 C5. The dispatch methods read
    via self._hooks.get(name, ()), and registration writes via
    self._hooks[name].append(fn). White-box test — touches the private
    `_hooks` attribute on purpose.
    """
    cb = tinyagent.CallbackRegistry()
    assert isinstance(cb._hooks, dict), (
        f"self._hooks must be a dict, got {type(cb._hooks).__name__}"
    )
    # Every canonical hook name has an entry (empty list).
    for name in (
        "before_llm_call",
        "after_llm_call",
        "before_tool_execution",
        "after_tool_execution",
        "on_error",
    ):
        assert name in cb._hooks, f"self._hooks missing canonical key {name!r}"
        assert cb._hooks[name] == [], f"self._hooks[{name!r}] should start empty"


def test_loop_starts_unpinned() -> None:
    """self._loop is None at construction (no loop pinned yet)."""
    cb = tinyagent.CallbackRegistry()
    assert cb._loop is None, f"self._loop should start as None, got {cb._loop!r}"


# ---------------------------------------------------------------------------
# Regression: the cb.before_llm_call.append(fn) form MUST NOT exist (M3)
# ---------------------------------------------------------------------------
def test_attribute_storage_form_does_not_exist() -> None:
    """Regression guard: cb.before_llm_call MUST raise AttributeError.

    Round-3 M3: the attribute-style form (`cb.before_llm_call.append(fn)`)
    is dropped. This test pins the negative — if a future refactor
    reintroduces attribute storage, this test fails and the regression is
    caught at CI time, not in production.
    """
    cb = tinyagent.CallbackRegistry()
    for name in (
        "before_llm_call",
        "after_llm_call",
        "before_tool_execution",
        "after_tool_execution",
        "on_error",
    ):
        with pytest.raises(AttributeError):
            cb.__getattribute__(name)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def test_register_before_llm_call_appends_to_dict() -> None:
    """register_before_llm_call(fn) writes to self._hooks['before_llm_call']."""
    cb = tinyagent.CallbackRegistry()
    cb.register_before_llm_call(_null_hook)
    assert cb._hooks["before_llm_call"] == [_null_hook]


def test_register_after_llm_call_appends_to_dict() -> None:
    """register_after_llm_call(fn) writes to self._hooks['after_llm_call']."""
    cb = tinyagent.CallbackRegistry()
    cb.register_after_llm_call(_null_hook)
    assert cb._hooks["after_llm_call"] == [_null_hook]


def test_register_before_tool_execution_appends_to_dict() -> None:
    """register_before_tool_execution(fn) writes to self._hooks['before_tool_execution']."""
    cb = tinyagent.CallbackRegistry()
    cb.register_before_tool_execution(_null_hook)
    assert cb._hooks["before_tool_execution"] == [_null_hook]


def test_register_after_tool_execution_appends_to_dict() -> None:
    """register_after_tool_execution(fn) writes to self._hooks['after_tool_execution']."""
    cb = tinyagent.CallbackRegistry()
    cb.register_after_tool_execution(_null_hook)
    assert cb._hooks["after_tool_execution"] == [_null_hook]


def test_register_on_error_appends_to_dict() -> None:
    """register_on_error(fn) writes to self._hooks['on_error']."""
    cb = tinyagent.CallbackRegistry()
    cb.register_on_error(_null_hook)
    assert cb._hooks["on_error"] == [_null_hook]


def test_registration_is_additive() -> None:
    """Multiple register_* calls on the same hook append (no replacement)."""
    cb = tinyagent.CallbackRegistry()

    def hook_a(ctx: object) -> None:
        pass

    def hook_b(ctx: object) -> None:
        pass

    def hook_c(ctx: object) -> None:
        pass

    cb.register_before_llm_call(hook_a)
    cb.register_before_llm_call(hook_b)
    cb.register_before_llm_call(hook_c)
    assert cb._hooks["before_llm_call"] == [hook_a, hook_b, hook_c]


# ---------------------------------------------------------------------------
# Dispatch — sync hooks
# ---------------------------------------------------------------------------
def test_dispatch_runs_registered_sync_hook() -> None:
    """dispatch('before_llm_call', ctx) invokes the registered sync hook."""
    cb = tinyagent.CallbackRegistry()
    calls: list[object] = []
    cb.register_before_llm_call(_capturing_hook_factory(calls))
    ctx = _ctx_stub()
    cb.dispatch("before_llm_call", ctx)
    assert calls == [ctx], f"sync hook should have been called once with ctx, got {calls!r}"


def test_dispatch_passes_ctx_to_sync_hook() -> None:
    """Sync hook receives the exact ctx object that was passed to dispatch."""
    cb = tinyagent.CallbackRegistry()
    captured: list[object] = []
    cb.register_after_llm_call(_capturing_hook_factory(captured))
    sentinel = object()  # distinguishable from any other object
    cb.dispatch("after_llm_call", sentinel)
    assert captured == [sentinel], (
        f"sync hook should receive the dispatched ctx by reference, got {captured!r}"
    )


def test_dispatch_with_no_registered_hooks_is_noop() -> None:
    """dispatch on an unpopulated hook name does nothing and does not raise."""
    cb = tinyagent.CallbackRegistry()
    # No exception expected; hook list is empty.
    cb.dispatch("before_llm_call", _ctx_stub())
    cb.dispatch("on_error", _ctx_stub())


def test_dispatch_unknown_name_is_noop() -> None:
    """dispatch with a name not in self._hooks is a no-op (dict.get default)."""
    cb = tinyagent.CallbackRegistry()
    # None of the canonical names are populated, but the method should be
    # defensive against unknown names too.
    cb.dispatch("nonexistent_hook", _ctx_stub())  # no raise


def test_multiple_hooks_run_in_registration_order() -> None:
    """Multiple hooks registered for the same event run in append order."""
    cb = tinyagent.CallbackRegistry()
    order: list[int] = []
    cb.register_before_tool_execution(_ordered_hook_factory(order, 1))
    cb.register_before_tool_execution(_ordered_hook_factory(order, 2))
    cb.register_before_tool_execution(_ordered_hook_factory(order, 3))
    cb.dispatch("before_tool_execution", _ctx_stub())
    assert order == [1, 2, 3], f"hooks must run in registration order, got {order!r}"


# ---------------------------------------------------------------------------
# Dispatch — async hooks (pinned loop bridge, cross-thread pattern)
# ---------------------------------------------------------------------------
def test_dispatch_awaits_async_hook_when_loop_pinned() -> None:
    """Async hook is awaited correctly in dispatch when the loop is pinned.

    Pins a loop in a separate thread (NOT the calling thread — the same-thread
    case deadlocks under run_coroutine_threadsafe). The driver thread runs
    the loop; dispatch schedules the coroutine via run_coroutine_threadsafe
    and blocks on future.result() until the driver delivers the result.
    """
    pinned_loop = asyncio.new_event_loop()
    cb = tinyagent.CallbackRegistry()
    cb._loop = pinned_loop
    observed: list[str] = []
    hook_done = threading.Event()

    async def async_hook(ctx: object) -> None:
        await asyncio.sleep(0)
        observed.append("ran")
        hook_done.set()

    cb.register_before_llm_call(async_hook)

    driver = threading.Thread(target=pinned_loop.run_forever, daemon=True)
    driver.start()
    try:
        # dispatch is a sync method that bridges to the pinned loop.
        cb.dispatch("before_llm_call", _ctx_stub())
    finally:
        pinned_loop.call_soon_threadsafe(pinned_loop.stop)
        driver.join(timeout=5.0)
        pinned_loop.close()

    assert hook_done.is_set(), "async hook did not complete within timeout"
    assert observed == ["ran"], f"async hook should have been awaited and run, got {observed!r}"


def test_dispatch_sync_uses_run_coroutine_threadsafe_when_loop_pinned() -> None:
    """dispatch_sync bridges async hooks via the pinned loop.

    Runs in a worker thread that has no event loop. The pinned loop lives in
    the main thread. We assert that an async hook ran on the pinned loop
    (verified by checking the running loop is the pinned one) and that
    dispatch_sync returned only after the hook completed.
    """
    main_thread_loop = asyncio.new_event_loop()
    try:
        cb = tinyagent.CallbackRegistry()
        cb._loop = main_thread_loop
        hook_ran_on_loop: list[asyncio.AbstractEventLoop] = []
        hook_done = threading.Event()

        async def async_hook(ctx: object) -> None:
            hook_ran_on_loop.append(asyncio.get_running_loop())
            hook_done.set()

        cb.register_after_llm_call(async_hook)

        driver = threading.Thread(target=main_thread_loop.run_forever, daemon=True)
        driver.start()
        try:

            def worker() -> None:
                # Worker thread has no event loop of its own.
                cb.dispatch_sync("after_llm_call", _ctx_stub())

            t = threading.Thread(target=worker)
            t.start()
            # Drive the pinned loop from the test thread by waiting for the
            # hook_done event (the driver thread runs the loop continuously).
            for _ in range(500):
                if hook_done.is_set():
                    break
                time.sleep(0.01)
            t.join(timeout=5.0)
        finally:
            main_thread_loop.call_soon_threadsafe(main_thread_loop.stop)
            driver.join(timeout=2.0)

        assert hook_done.is_set(), "async hook did not run within timeout"
        assert hook_ran_on_loop == [main_thread_loop], (
            f"async hook should have run on the pinned loop, got {hook_ran_on_loop!r}"
        )
    finally:
        main_thread_loop.close()


def test_dispatch_sync_raises_assertion_when_loop_not_pinned() -> None:
    """dispatch_sync asserts self._loop is pinned before iterating hooks.

    The assertion prevents silent coroutine drops (peer-review M3 round-1
    closure). A user that calls dispatch_sync without a pinned loop hits the
    assertion at the dispatch call site, not at GC time.
    """
    cb = tinyagent.CallbackRegistry()

    async def async_hook(ctx: object) -> None:
        pass

    cb.register_before_llm_call(async_hook)
    with pytest.raises(AssertionError):
        cb.dispatch_sync("before_llm_call", _ctx_stub())


# ---------------------------------------------------------------------------
# clear() — test helper
# ---------------------------------------------------------------------------
def test_clear_empties_all_hook_lists() -> None:
    """clear() empties every entry in self._hooks (test helper)."""
    cb = tinyagent.CallbackRegistry()
    cb.register_before_llm_call(_null_hook)
    cb.register_after_llm_call(_null_hook)
    cb.register_before_tool_execution(_null_hook)
    cb.register_after_tool_execution(_null_hook)
    cb.register_on_error(_null_hook)
    # Sanity: every list has one entry.
    for name in (
        "before_llm_call",
        "after_llm_call",
        "before_tool_execution",
        "after_tool_execution",
        "on_error",
    ):
        assert len(cb._hooks[name]) == 1, f"setup failed: {name!r} not populated"

    cb.clear()

    for name in (
        "before_llm_call",
        "after_llm_call",
        "before_tool_execution",
        "after_tool_execution",
        "on_error",
    ):
        assert cb._hooks[name] == [], (
            f"after clear(), self._hooks[{name!r}] should be empty, got {cb._hooks[name]!r}"
        )


def test_clear_preserves_dict_keys() -> None:
    """clear() empties the lists but keeps the canonical keys in self._hooks.

    The hook-name keys are the registry's structural contract — clearing must
    not delete them, only empty the per-hook lists.
    """
    cb = tinyagent.CallbackRegistry()
    cb.clear()
    for name in (
        "before_llm_call",
        "after_llm_call",
        "before_tool_execution",
        "after_tool_execution",
        "on_error",
    ):
        assert name in cb._hooks, f"clear() must preserve the {name!r} key in self._hooks"


# ---------------------------------------------------------------------------
# Return value handling
# ---------------------------------------------------------------------------
def test_dispatch_discards_sync_hook_return_value() -> None:
    """Sync hook return value is discarded by dispatch (fire-and-forget)."""

    def hook_returning_value(ctx: object) -> str:
        return "ignored return value"

    cb = tinyagent.CallbackRegistry()
    cb.register_before_llm_call(hook_returning_value)
    # No assertion failure means dispatch did not try to use the return.
    cb.dispatch("before_llm_call", _ctx_stub())


def test_dispatch_discards_async_hook_return_value() -> None:
    """Async hook return value is discarded by dispatch (fire-and-forget)."""
    pinned_loop = asyncio.new_event_loop()
    cb = tinyagent.CallbackRegistry()
    cb._loop = pinned_loop

    async def async_hook(ctx: object) -> str:
        return "ignored return value"

    cb.register_before_llm_call(async_hook)

    driver = threading.Thread(target=pinned_loop.run_forever, daemon=True)
    driver.start()
    try:
        cb.dispatch("before_llm_call", _ctx_stub())  # no assertion failure
    finally:
        pinned_loop.call_soon_threadsafe(pinned_loop.stop)
        driver.join(timeout=5.0)
        pinned_loop.close()
