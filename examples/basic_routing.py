"""
Example: Basic task routing.

Demonstrates how the orchestrator classifies tasks and routes them to
the appropriate model without calling an LLM.
"""

from agent_orchestrator import Orchestrator, TaskRouter, TaskType

# ── 1. Direct router usage ─────────────────────────────────────────────────────
router = TaskRouter()

tasks = [
    "Write a Python script to parse JSON from stdin and output a summary",
    "Compare the tradeoffs between RAG and fine-tuning for domain adaptation",
    "What is the latest news on AI regulation in the EU today?",
    "Draft a cold email to a VP of Engineering at a fintech startup",
]

print("=== Task Routing Decisions ===\n")
for task in tasks:
    decision = router.route(task)
    print(f"Task: {task[:70]}...")
    print(f"  Type:       {decision.task_type.value}")
    print(f"  Model:      {decision.model}")
    print(f"  Agent:      {decision.agent or '(direct LLM call)'}")
    print(f"  Confidence: {decision.confidence:.0%}")
    print(f"  Rationale:  {decision.rationale}")
    print()

# ── 2. Full orchestrator with stub agents ──────────────────────────────────────
print("=== Orchestrator with Sub-Agents ===\n")

orch = Orchestrator(
    primary_model="google/gemini-3.1-pro-preview",
    memory_dir="/tmp/example-memory",
)

# Register stub agents (in production these would call real LLMs/APIs)
def coding_agent(task: str, model: str) -> str:
    return f"[coding-agent via {model}] Wrote code for: {task[:60]}..."

def research_agent(task: str, model: str) -> str:
    return f"[research-agent via {model}] Found info on: {task[:60]}..."

orch.register_agent("coding-agent", coding_agent)
orch.register_agent("research-agent", research_agent)

for task in tasks[:2]:
    result = orch.run(task)
    print(f"Task: {task[:60]}...")
    print(f"  Routed to: {result.routing.model}")
    print(f"  Output:    {result.output}")
    print(f"  Took:      {result.duration_ms}ms")
    print(f"  Success:   {result.success}")
    print()
