"""
Example: dependency-aware orchestration with a task graph.

Two research tasks run independently, a synthesis task waits for both, and a
memo task waits for the synthesis. Dependencies are declared at creation, so the
scheduler runs each task only after its inputs exist. Runs with stub agents and
no API keys.
"""

from agent_orchestrator import Orchestrator, TaskGraph
from agent_orchestrator.taskgraph import TaskGraphError

orch = Orchestrator(primary_model="your/reasoning-model", memory_dir="/tmp/example-graph-memory")
orch.set_llm_caller(lambda prompt, model: f"[primary {model}] {prompt[:50]}")


# Stub specialists. In production these call real LLMs / tools.
def researcher(task: str, model: str) -> str:
    return f"[researcher:{model}] findings for: {task.splitlines()[-1][:48]}"


def analyst(task: str, model: str) -> str:
    # The analyst's prompt already contains the researchers' outputs as context.
    return f"[analyst] recommendation synthesized from {task.count('[researcher')} input(s)"


def writer(task: str, model: str) -> str:
    return "[writer] 1-page memo drafted from the recommendation"


orch.register_agent("researcher", researcher)
orch.register_agent("analyst", analyst)
orch.register_agent("writer", writer)

g = TaskGraph()
cost = g.add("research: cost", "researcher", body="Estimate 3-year migration cost.")
perf = g.add("research: performance", "researcher", body="Estimate query latency at scale.")
synth = g.add("synthesize recommendation", "analyst", parents=[cost, perf])
memo = g.add("draft decision memo", "writer", parents=[synth])

print("=== Task graph ===")
for t in g.tasks():
    deps = ", ".join(t.parents) if t.parents else "(none)"
    print(f"  {t.id}: {t.title}  [assignee={t.assignee}, parents={deps}]")
print()

print("=== Execution (dependency order) ===")
outputs = orch.run_graph(g)
for task_id, output in outputs.items():
    print(f"  {task_id}: {output}")
print()

# Fail-loud: an unknown assignee raises instead of sitting in the queue forever.
print("=== Unknown assignee fails loud ===")
broken = TaskGraph()
broken.add("orphan task", "nonexistent-agent", body="nobody can run this")
try:
    orch.run_graph(broken)
except TaskGraphError as exc:
    print(f"  raised TaskGraphError: {exc}")
