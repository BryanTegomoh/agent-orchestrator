"""
Dependency-aware task graph for multi-step agent orchestration.

A ``TaskGraph`` is a directed acyclic graph of tasks. Each task names the agent
expected to run it and, optionally, the parent tasks whose output it needs. A
task becomes runnable only once every parent has completed, so a dependency is
declared once, at creation, and the scheduler enforces ordering. Independent
tasks surface together and can be executed concurrently.

The design avoids the failure modes that bite naive orchestrators:

  * Dependencies are declared at creation (``add(..., parents=[...])``). There is
    no create-then-link step, and therefore no window in which a child can run
    before its inputs exist.
  * A parent reference that does not resolve raises immediately. A task that can
    never become runnable is a defect, not a silent dead end.
  * The graph is acyclic by construction: a task may depend only on tasks that
    already exist, so a cycle cannot be formed.
  * A task that reports spawning children has that report checked against the
    graph; ids that do not exist are rejected. An agent does not get to claim
    work it did not do.
  * Execution is fail-loud: if any task fails or is left unreachable, ``run``
    raises rather than returning a partial result that reads as success.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from functools import partial

from .authority import DecisionBrief, NeedsOwner, missing_grants_brief


class TaskState(Enum):
    PENDING = "pending"   # waiting on one or more parents
    READY = "ready"       # every parent done; runnable now
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"   # parked on an owner decision; not a failure


class TaskGraphError(Exception):
    """Base class for task-graph errors."""


class UnknownParentError(TaskGraphError):
    """A task referenced a parent that does not exist."""


class SelfReportError(TaskGraphError):
    """A task reported creating child tasks that do not exist."""


class OwnerDecisionRequired(TaskGraphError):
    """
    All runnable work finished, but tasks are parked on owner decisions.
    Carries one DecisionBrief per parked task and the outputs of everything
    that completed; resolve with grants or unblock(), then run again.
    """

    def __init__(self, briefs: list[DecisionBrief], outputs: dict[str, str]):
        self.briefs = briefs
        self.outputs = outputs
        rendered = "\n".join(b.render() for b in briefs)
        super().__init__(
            f"{len(briefs)} task(s) await an owner decision:\n{rendered}\n"
            "Resolve with broader grants or unblock(), then run the graph again."
        )


@dataclass
class Task:
    id: str
    title: str
    assignee: str
    body: str = ""
    parents: tuple[str, ...] = ()
    priority: int = 0
    requires: tuple[str, ...] = ()      # grants this task needs to run
    state: TaskState = TaskState.PENDING
    result: str | None = None
    error: str | None = None
    brief: DecisionBrief | None = None  # set when parked on an owner decision
    blocked_on: tuple[str, ...] = ()    # missing grants; empty for executor parks


# An executor turns a runnable task (plus its parents' outputs) into a result.
Executor = Callable[["Task", "dict[str, str]"], str]


class TaskGraph:
    """A DAG of agent tasks with dependency-gated scheduling."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._counter = 0

    # ── Construction ─────────────────────────────────────────────────────────

    def add(
        self,
        title: str,
        assignee: str,
        *,
        body: str = "",
        parents: Iterable[str] = (),
        priority: int = 0,
        requires: Iterable[str] = (),
        task_id: str | None = None,
    ) -> str:
        """
        Add a task and return its id.

        Every parent must already exist; an unknown parent raises
        ``UnknownParentError``. Because a task can depend only on tasks that
        already exist, the graph stays acyclic by construction.
        """
        parent_ids = tuple(parents)
        for pid in parent_ids:
            if pid not in self._tasks:
                raise UnknownParentError(
                    f"parent {pid!r} does not exist; create it before its child"
                )
        tid = task_id or self._next_id()
        if tid in self._tasks:
            raise TaskGraphError(f"task id {tid!r} already exists")
        self._tasks[tid] = Task(
            id=tid,
            title=title,
            assignee=assignee,
            body=body,
            parents=parent_ids,
            priority=priority,
            requires=tuple(requires),
            state=TaskState.READY if not parent_ids else TaskState.PENDING,
        )
        return tid

    def _next_id(self) -> str:
        self._counter += 1
        return f"t{self._counter}"

    # ── Inspection ───────────────────────────────────────────────────────────

    def __getitem__(self, task_id: str) -> Task:
        return self._tasks[task_id]

    def __len__(self) -> int:
        return len(self._tasks)

    def tasks(self) -> list[Task]:
        return list(self._tasks.values())

    def ready(self) -> list[Task]:
        """Runnable tasks now, highest priority first (ties broken by id)."""
        runnable = [t for t in self._tasks.values() if t.state is TaskState.READY]
        return sorted(runnable, key=lambda t: (-t.priority, t.id))

    def parent_outputs(self, task_id: str) -> dict[str, str]:
        """Map of parent id to result for the given task's completed parents."""
        outputs: dict[str, str] = {}
        for pid in self._tasks[task_id].parents:
            result = self._tasks[pid].result
            if result is not None:
                outputs[pid] = result
        return outputs

    def is_complete(self) -> bool:
        return all(t.state is TaskState.DONE for t in self._tasks.values())

    def unfinished(self) -> list[Task]:
        return [
            t
            for t in self._tasks.values()
            if t.state not in (TaskState.DONE, TaskState.FAILED)
        ]

    def failed(self) -> list[Task]:
        return [t for t in self._tasks.values() if t.state is TaskState.FAILED]

    def blocked(self) -> list[Task]:
        return [t for t in self._tasks.values() if t.state is TaskState.BLOCKED]

    def snapshot(self) -> dict[str, object]:
        """
        JSON-safe graph state for logs, dashboards, or external control planes.

        This is an export surface, not a persistence or resume mechanism. The
        graph still owns execution state in memory; callers that need durable
        execution should use a workflow engine.
        """
        tasks = []
        for task in self._tasks.values():
            tasks.append(
                {
                    "id": task.id,
                    "title": task.title,
                    "assignee": task.assignee,
                    "body": task.body,
                    "parents": list(task.parents),
                    "priority": task.priority,
                    "requires": list(task.requires),
                    "state": task.state.value,
                    "result": task.result,
                    "error": task.error,
                    "brief": task.brief.to_dict() if task.brief is not None else None,
                    "blocked_on": list(task.blocked_on),
                }
            )
        return {
            "tasks": tasks,
            "states": {task.id: task.state.value for task in self._tasks.values()},
            "dependencies": {
                task.id: list(task.parents) for task in self._tasks.values()
            },
            "briefs": [
                task.brief.to_dict()
                for task in self.blocked()
                if task.brief is not None
            ],
        }

    def unblock(self, task_id: str) -> None:
        """Owner override: return a parked task to the runnable pool."""
        task = self._tasks[task_id]
        if task.state is not TaskState.BLOCKED:
            raise TaskGraphError(f"task {task_id!r} is {task.state.value}, not blocked")
        task.brief = None
        task.blocked_on = ()
        task.state = TaskState.READY

    # ── State transitions ────────────────────────────────────────────────────

    def complete(self, task_id: str, result: str, *, created: Iterable[str] = ()) -> None:
        """
        Mark a task done and promote any children whose parents are now complete.

        ``created`` lets a task report the children it spawned (relevant when the
        task is itself an agent that plans more work). Reported ids are checked
        against the graph and a phantom id raises ``SelfReportError``.
        """
        for cid in created:
            if cid not in self._tasks:
                raise SelfReportError(
                    f"task {task_id!r} reported creating {cid!r}, which does not exist"
                )
        task = self._tasks[task_id]
        task.result = result
        task.state = TaskState.DONE
        self._promote()

    def fail(self, task_id: str, error: str) -> None:
        task = self._tasks[task_id]
        task.error = error
        task.state = TaskState.FAILED

    def _promote(self) -> None:
        for task in self._tasks.values():
            if task.state is TaskState.PENDING and all(
                self._tasks[p].state is TaskState.DONE for p in task.parents
            ):
                task.state = TaskState.READY

    # ── Execution ────────────────────────────────────────────────────────────

    def run(
        self,
        executor: Executor,
        *,
        max_workers: int = 1,
        granted: Iterable[str] = (),
        on_event: Callable[[str, Task], None] | None = None,
    ) -> dict[str, str]:
        """
        Execute the graph in dependency order and return each task's output.

        Ready tasks surface in waves of mutually independent tasks. With the
        default ``max_workers=1`` each wave runs sequentially; a higher value
        runs the tasks in a wave concurrently in a thread pool, in which case
        the executor must be thread-safe. Graph state is only mutated from the
        calling thread either way.

        ``granted`` is the set of permission grants for this run. A task whose
        ``requires`` exceed the grants is parked as BLOCKED with a brief, and
        independent lanes keep running; an executor may park a task the same
        way by raising ``NeedsOwner``. Re-running with broader grants resumes
        grant-parked tasks automatically; ``unblock()`` resumes the rest.

        ``on_event`` receives (event, task) for task_started, task_done,
        task_failed, and task_blocked, always from the calling thread.

        The run never reads as more successful than it was: failures raise
        ``TaskGraphError``; otherwise parked tasks raise
        ``OwnerDecisionRequired`` carrying the briefs and completed outputs.
        """
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        grants = frozenset(granted)

        def emit(event: str, task: Task) -> None:
            if on_event is not None:
                on_event(event, task)

        def settle(task: Task, future_result: Callable[[], str]) -> None:
            try:
                result = future_result()
            except NeedsOwner as ask:
                self._park(task, ask.to_brief(task.id, task.title), missing=())
                emit("task_blocked", task)
                return
            except Exception as exc:
                self.fail(task.id, str(exc))
                emit("task_failed", task)
                return
            self.complete(task.id, result)
            emit("task_done", task)

        # A re-run with broader grants resumes tasks parked only on grants.
        for task in self.blocked():
            if task.blocked_on and set(task.blocked_on) <= grants:
                self.unblock(task.id)

        while True:
            wave = []
            for task in self.ready():
                missing = tuple(sorted(set(task.requires) - grants))
                if missing:
                    self._park(
                        task,
                        missing_grants_brief(task.id, task.title, missing),
                        missing=missing,
                    )
                    emit("task_blocked", task)
                else:
                    wave.append(task)
            if not wave:
                break

            for task in wave:
                task.state = TaskState.RUNNING
                emit("task_started", task)

            if max_workers == 1 or len(wave) == 1:
                for task in wave:
                    settle(task, partial(executor, task, self.parent_outputs(task.id)))
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures: dict[Future[str], Task] = {
                        pool.submit(executor, task, self.parent_outputs(task.id)): task
                        for task in wave
                    }
                    for future in as_completed(futures):
                        settle(futures[future], future.result)

        return self._account()

    # ── Run accounting ───────────────────────────────────────────────────────

    def _park(self, task: Task, brief: DecisionBrief, missing: tuple[str, ...]) -> None:
        task.brief = brief
        task.blocked_on = missing
        task.state = TaskState.BLOCKED

    def _account(self) -> dict[str, str]:
        """Final reckoning for run(): outputs, or a loud, accurate exception."""
        failed = self.failed()
        blocked = self.blocked()
        pending = [t for t in self._tasks.values() if t.state is TaskState.PENDING]

        if failed:
            parts = ["failed: " + ", ".join(f"{t.id} ({t.error})" for t in failed)]
            if blocked:
                parts.append("blocked on owner: " + ", ".join(t.id for t in blocked))
            if pending:
                parts.append("unreachable: " + ", ".join(t.id for t in pending))
            raise TaskGraphError("; ".join(parts))

        outputs = {
            t.id: t.result
            for t in self._tasks.values()
            if t.state is TaskState.DONE and t.result is not None
        }
        if blocked:
            raise OwnerDecisionRequired(
                briefs=[t.brief for t in blocked if t.brief is not None],
                outputs=outputs,
            )
        if pending:
            raise TaskGraphError("unreachable: " + ", ".join(t.id for t in pending))
        return outputs
