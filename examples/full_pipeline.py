"""
Example: a closed loop end to end.

A fleet (task graph) produces a draft through dependency-gated steps, with the
independent research tasks running in parallel. A judge loop then revises the
draft until it meets explicit acceptance criteria. The loop ends in an explicit
verdict either way: "done" only when the judge accepts, "exhausted" when the
budget runs out.

Runs with stub agents and no API keys; swap the stubs for real LLM calls.
"""

from agent_orchestrator import Orchestrator, TaskGraph, Verdict, run_goal

# ── Fleet: research in parallel, then draft ─────────────────────────────────────

orch = Orchestrator(
    primary_model="your/reasoning-model",
    memory_dir="/tmp/example-pipeline-memory",
)
orch.set_llm_caller(lambda prompt, model: f"[primary {model}] {prompt[:40]}")


def researcher(task: str, model: str) -> str:
    topic = task.splitlines()[-1]
    if "cost" in topic:
        return "Finding: migration cost is roughly $40k over three years."
    return "Finding: p95 latency stays under 20ms at the expected load."


def writer(task: str, model: str) -> str:
    # First draft is deliberately incomplete: it drops the latency finding,
    # so the judge loop below has real work to do.
    return "Recommendation: migrate. Cost is acceptable at roughly $40k."


orch.register_agent("researcher", researcher)
orch.register_agent("writer", writer)

g = TaskGraph()
cost = g.add("research: cost", "researcher", body="Estimate 3-year migration cost.")
perf = g.add("research: latency", "researcher", body="Estimate p95 latency at expected load.")
draft = g.add("draft recommendation", "writer", parents=[cost, perf])

print("=== Fleet: parallel research, gated draft ===")
outputs = orch.run_graph(g, max_workers=2)   # cost and perf run concurrently
first_draft = outputs[draft]
print(f"  {cost}: {outputs[cost]}")
print(f"  {perf}: {outputs[perf]}")
print(f"  draft: {first_draft}")
print()

# ── Judge loop: revise the draft until it meets the acceptance criteria ────────

GOAL = "Recommendation memo. Acceptance: mentions both the cost and the latency findings."


def revise(goal: str, feedback: str) -> str:
    # A real worker would be an LLM call that revises the draft per feedback.
    if "latency" in feedback:
        return first_draft + " Latency stays under 20ms at the expected load."
    return first_draft


def judge(goal: str, attempt: str) -> Verdict:
    missing = [term for term in ("cost", "latency") if term not in attempt.lower()]
    if missing:
        return Verdict(done=False, feedback=f"missing the {missing[0]} finding")
    return Verdict(done=True)


print("=== Judge loop: revise until acceptance ===")
result = run_goal(GOAL, revise, judge, max_turns=5)
print(f"  status: {result.status}")
print(f"  turns:  {result.turns}")
print(f"  final:  {result.output}")
