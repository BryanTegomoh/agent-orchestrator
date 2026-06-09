"""Tests for the dependency-aware task graph."""

import threading

import pytest

from agent_orchestrator.taskgraph import (
    SelfReportError,
    TaskGraph,
    TaskGraphError,
    TaskState,
    UnknownParentError,
)


def test_independent_tasks_are_ready():
    g = TaskGraph()
    a = g.add("a", "worker")
    b = g.add("b", "worker")
    ready = {t.id for t in g.ready()}
    assert ready == {a, b}
    assert g[a].state is TaskState.READY


def test_child_is_pending_until_parents_complete():
    g = TaskGraph()
    a = g.add("a", "worker")
    c = g.add("c", "worker", parents=[a])
    assert g[c].state is TaskState.PENDING
    assert [t.id for t in g.ready()] == [a]
    g.complete(a, "result-a")
    assert g[c].state is TaskState.READY


def test_unknown_parent_raises():
    g = TaskGraph()
    with pytest.raises(UnknownParentError):
        g.add("c", "worker", parents=["does-not-exist"])


def test_ready_orders_by_priority_then_id():
    g = TaskGraph()
    low = g.add("low", "worker", priority=0)
    high = g.add("high", "worker", priority=5)
    assert [t.id for t in g.ready()] == [high, low]


def test_self_report_gate_rejects_phantom_ids():
    g = TaskGraph()
    planner = g.add("planner", "worker")
    with pytest.raises(SelfReportError):
        g.complete(planner, "done", created=["t999"])


def test_self_report_gate_accepts_real_ids():
    g = TaskGraph()
    planner = g.add("planner", "worker")
    child = g.add("child", "worker", parents=[planner])
    g.complete(planner, "done", created=[child])
    assert g[planner].state is TaskState.DONE


def test_run_executes_in_dependency_order_with_handoff():
    g = TaskGraph()
    cost = g.add("cost", "researcher", body="estimate cost")
    perf = g.add("perf", "researcher", body="estimate perf")
    synth = g.add("synthesize", "analyst", parents=[cost, perf])

    order: list[str] = []

    def execute(task, parent_outputs):
        order.append(task.id)
        if task.id == synth:
            assert set(parent_outputs) == {cost, perf}
            return "synth:" + "+".join(sorted(parent_outputs.values()))
        return f"{task.id}-out"

    outputs = g.run(execute)

    assert order.index(synth) > order.index(cost)
    assert order.index(synth) > order.index(perf)
    assert outputs[synth] == "synth:" + "+".join(sorted([f"{cost}-out", f"{perf}-out"]))
    assert g.is_complete()


def test_run_is_fail_loud_and_leaves_descendants_unreachable():
    g = TaskGraph()
    a = g.add("a", "worker")
    b = g.add("b", "worker", parents=[a])

    def execute(task, parent_outputs):
        raise RuntimeError("boom")

    with pytest.raises(TaskGraphError):
        g.run(execute)

    assert g[a].state is TaskState.FAILED
    assert g[b].state is TaskState.PENDING  # never promoted; surfaced as unreachable


def test_parallel_wave_runs_concurrently():
    g = TaskGraph()
    a = g.add("a", "worker")
    b = g.add("b", "worker")

    # Both tasks must be in flight at once to pass the barrier; sequential
    # execution would time out, fail both tasks, and raise.
    barrier = threading.Barrier(2, timeout=5)

    def execute(task, parent_outputs):
        barrier.wait()
        return task.id

    outputs = g.run(execute, max_workers=2)
    assert set(outputs) == {a, b}


def test_parallel_run_still_respects_dependencies():
    g = TaskGraph()
    a = g.add("a", "worker")
    b = g.add("b", "worker")
    child = g.add("child", "worker", parents=[a, b])

    order: list[str] = []
    lock = threading.Lock()

    def execute(task, parent_outputs):
        with lock:
            order.append(task.id)
        if task.id == child:
            assert set(parent_outputs) == {a, b}
        return task.id

    g.run(execute, max_workers=4)
    assert order.index(child) > order.index(a)
    assert order.index(child) > order.index(b)


def test_parallel_failure_is_still_fail_loud():
    g = TaskGraph()
    g.add("a", "worker")
    g.add("b", "worker")

    def execute(task, parent_outputs):
        if task.title == "b":
            raise RuntimeError("boom")
        return "ok"

    with pytest.raises(TaskGraphError):
        g.run(execute, max_workers=2)


def test_invalid_max_workers_raises():
    g = TaskGraph()
    g.add("a", "worker")
    with pytest.raises(ValueError):
        g.run(lambda t, p: "x", max_workers=0)
