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
from dataclasses import dataclass
from enum import Enum


class TaskState(Enum):
    PENDING = "pending"   # waiting on one or more parents
    READY = "ready"       # every parent done; runnable now
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class TaskGraphError(Exception):
    """Base class for task-graph errors."""


class UnknownParentError(TaskGraphError):
    """A task referenced a parent that does not exist."""


class SelfReportError(TaskGraphError):
    """A task reported creating child tasks that do not exist."""


@dataclass
class Task:
    id: str
    title: str
    assignee: str
    body: str = ""
    parents: tuple[str, ...] = ()
    priority: int = 0
    state: TaskState = TaskState.PENDING
    result: str | None = None
    error: str | None = None


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

    def run(self, executor: Executor) -> dict[str, str]:
        """
        Execute the graph in dependency order and return each task's output.

        Ready tasks are surfaced in waves; the reference loop runs each wave
        sequentially, but the tasks in a wave are independent and a deployment
        is free to run them concurrently. Execution is fail-loud: if any task
        fails (or is left unreachable because an ancestor failed), this raises
        ``TaskGraphError`` instead of returning a partial result.
        """
        outputs: dict[str, str] = {}
        while True:
            wave = self.ready()
            if not wave:
                break
            for task in wave:
                task.state = TaskState.RUNNING
                try:
                    result = executor(task, self.parent_outputs(task.id))
                except Exception as exc:
                    self.fail(task.id, str(exc))
                    continue
                self.complete(task.id, result)
                outputs[task.id] = result

        failed = self.failed()
        unreachable = self.unfinished()
        if failed or unreachable:
            parts: list[str] = []
            if failed:
                parts.append(
                    "failed: " + ", ".join(f"{t.id} ({t.error})" for t in failed)
                )
            if unreachable:
                parts.append("unreachable: " + ", ".join(t.id for t in unreachable))
            raise TaskGraphError("; ".join(parts))
        return outputs
