"""Tests for the dependency-aware task graph."""

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
