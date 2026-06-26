"""Tests for bounded autonomy: grants, owner parks, briefs, and the ledger."""

import pytest

from agent_orchestrator.authority import DecisionBrief, NeedsOwner
from agent_orchestrator.ledger import Ledger
from agent_orchestrator.taskgraph import (
    OwnerDecisionRequired,
    TaskGraph,
    TaskGraphError,
    TaskState,
)


def ok(task, parent_outputs):
    return f"{task.id}-out"


# ── Permission grants ────────────────────────────────────────────────────────


def test_ungranted_task_parks_and_lanes_continue():
    g = TaskGraph()
    free = g.add("fix bug", "worker")
    gated = g.add("publish release", "worker", requires=["release"])

    with pytest.raises(OwnerDecisionRequired) as exc:
        g.run(ok)

    assert g[free].state is TaskState.DONE        # independent lane finished
    assert g[gated].state is TaskState.BLOCKED    # parked, not failed
    assert exc.value.outputs == {free: f"{free}-out"}
    (brief,) = exc.value.briefs
    assert "release" in brief.what
    assert brief.task_id == gated


def test_granted_task_runs():
    g = TaskGraph()
    gated = g.add("publish release", "worker", requires=["release"])
    outputs = g.run(ok, granted={"release"})
    assert outputs[gated] == f"{gated}-out"


def test_grants_are_exact_no_hierarchy():
    g = TaskGraph()
    g.add("merge it", "worker", requires=["merge"])
    with pytest.raises(OwnerDecisionRequired):
        g.run(ok, granted={"push"})  # push does not imply merge


def test_rerun_with_broader_grants_resumes():
    g = TaskGraph()
    gated = g.add("publish release", "worker", requires=["release"])
    with pytest.raises(OwnerDecisionRequired):
        g.run(ok)
    outputs = g.run(ok, granted={"release"})      # same graph, broader grants
    assert outputs[gated] == f"{gated}-out"
    assert g.is_complete()


def test_descendants_of_parked_task_wait():
    g = TaskGraph()
    gated = g.add("deploy", "worker", requires=["deploy"])
    child = g.add("announce", "worker", parents=[gated])
    with pytest.raises(OwnerDecisionRequired):
        g.run(ok)
    assert g[child].state is TaskState.PENDING


# ── Executor-raised owner parks ──────────────────────────────────────────────


def test_needs_owner_parks_with_brief():
    g = TaskGraph()
    t = g.add("choose storage backend", "worker")

    def execute(task, parent_outputs):
        raise NeedsOwner(
            what="Postgres and SQLite both fit; this is a product call.",
            recommendation="Postgres for the multi-writer requirement.",
            choices=("postgres", "sqlite"),
        )

    with pytest.raises(OwnerDecisionRequired) as exc:
        g.run(execute)
    (brief,) = exc.value.briefs
    assert brief.choices == ("postgres", "sqlite")
    assert g[t].state is TaskState.BLOCKED
    assert g[t].blocked_on == ()  # not a grant problem; needs explicit unblock


def test_unblock_then_rerun_completes():
    g = TaskGraph()
    t = g.add("choose storage backend", "worker")
    asked = {"n": 0}

    def execute(task, parent_outputs):
        asked["n"] += 1
        if asked["n"] == 1:
            raise NeedsOwner(what="product call")
        return "postgres it is"

    with pytest.raises(OwnerDecisionRequired):
        g.run(execute)
    g.unblock(t)
    assert g.run(execute) == {t: "postgres it is"}


def test_unblock_requires_blocked_state():
    g = TaskGraph()
    t = g.add("a", "worker")
    with pytest.raises(TaskGraphError):
        g.unblock(t)


def test_failure_outranks_blocked_in_reporting():
    g = TaskGraph()
    g.add("boom", "worker")
    g.add("gated", "worker", requires=["x"])

    def execute(task, parent_outputs):
        raise RuntimeError("boom")

    with pytest.raises(TaskGraphError) as exc:
        g.run(execute)
    assert not isinstance(exc.value, OwnerDecisionRequired)
    assert "blocked on owner" in str(exc.value)


# ── Briefs ───────────────────────────────────────────────────────────────────


def test_brief_render_includes_choices():
    brief = DecisionBrief(
        task_id="t1",
        title="ship it",
        what="Releases v2 to all users.",
        recommendation="Ship.",
        choices=("ship", "hold"),
    )
    text = brief.render()
    assert "ship | hold" in text
    assert "Releases v2" in text


def test_brief_to_dict_is_json_safe():
    brief = DecisionBrief(
        task_id="t1",
        title="ship it",
        what="Releases v2 to all users.",
        why_now="CI is green.",
        proof="73 tests passed.",
        tradeoffs="Rollback is manual.",
        recommendation="Ship.",
        choices=("ship", "hold"),
    )
    assert brief.to_dict() == {
        "task_id": "t1",
        "title": "ship it",
        "what": "Releases v2 to all users.",
        "why_now": "CI is green.",
        "proof": "73 tests passed.",
        "tradeoffs": "Rollback is manual.",
        "recommendation": "Ship.",
        "choices": ["ship", "hold"],
    }


# ── Ledger ───────────────────────────────────────────────────────────────────


def test_ledger_appends_and_reads(tmp_path):
    led = Ledger(tmp_path / "run.jsonl")
    led.record("task_started", task="t1", detail="fix bug")
    led.record("task_done", task="t1")
    events = led.events()
    assert [e.event for e in events] == ["task_started", "task_done"]
    assert events[0].task == "t1"
    assert events[0].at  # timestamped


def test_ledger_skips_malformed_lines(tmp_path):
    path = tmp_path / "run.jsonl"
    led = Ledger(path)
    led.record("task_done", task="t1")
    with path.open("a", encoding="utf-8") as f:
        f.write("not json\n")
    led.record("task_done", task="t2")
    assert [e.task for e in led.events()] == ["t1", "t2"]


def test_graph_events_reach_ledger(tmp_path):
    led = Ledger(tmp_path / "run.jsonl")
    g = TaskGraph()
    a = g.add("a", "worker")
    gated = g.add("gated", "worker", requires=["x"])

    with pytest.raises(OwnerDecisionRequired):
        g.run(ok, on_event=lambda event, task: led.record(event, task=task.id))

    seen = {(e.event, e.task) for e in led.events()}
    assert ("task_started", a) in seen
    assert ("task_done", a) in seen
    assert ("task_blocked", gated) in seen
