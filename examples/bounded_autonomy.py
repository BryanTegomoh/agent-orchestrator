"""
Example: bounded autonomy on a maintainer queue.

A maintainer-style graph where work proceeds only inside its grants: the bug
fix runs freely, the dependency bump needs "push", and the release needs
"release". The run stops at its authorized boundary with decision-ready
briefs, the owner grants more, and the same graph resumes. Every lifecycle
event lands in an append-only ledger. No API keys needed.
"""

from pathlib import Path

from agent_orchestrator import Ledger, NeedsOwner, OwnerDecisionRequired, TaskGraph

LEDGER_PATH = Path("/tmp/example-bounded-autonomy.jsonl")
LEDGER_PATH.unlink(missing_ok=True)
ledger = Ledger(LEDGER_PATH)


def execute(task, parent_outputs):
    if task.id == "choose-name" and "decision:" not in task.body:
        # A product call mid-task: park it for the owner instead of guessing.
        raise NeedsOwner(
            what="The CLI flag could be --workers or --jobs; both are defensible.",
            proof="Both spellings pass the test suite.",
            recommendation="--workers, matching the library API.",
            choices=("--workers", "--jobs"),
        )
    return f"{task.title}: done"


g = TaskGraph()
fix = g.add("fix the off-by-one in retry logic", "engineer", task_id="fix-bug")
name = g.add("pick the new CLI flag name", "engineer", task_id="choose-name")
deps = g.add("bump patch dependencies", "engineer", task_id="bump-deps", requires=["push"])
rel = g.add(
    "publish the patch release", "engineer",
    task_id="release", parents=[fix, deps], requires=["release"],
)

record = lambda event, task: ledger.record(event, task=task.id, detail=task.title)  # noqa: E731

print("=== Run 1: granted={push} ===")
try:
    g.run(execute, granted={"push"}, on_event=record)
except OwnerDecisionRequired as ask:
    print(f"completed anyway: {sorted(ask.outputs)}")
    print("awaiting owner:")
    for brief in ask.briefs:
        print(brief.render())
print()

print("=== Owner decides: --workers, and grants release ===")
g["choose-name"].body += "\ndecision: --workers"   # the decision flows into the task
g.unblock("choose-name")

print("=== Run 2: granted={push, release} ===")
outputs = g.run(execute, granted={"push", "release"}, on_event=record)
print(f"all done: {sorted(outputs)}")
print()

print("=== Ledger ===")
for e in ledger.events():
    print(f"  {e.at}  {e.event:13s} {e.task}")
