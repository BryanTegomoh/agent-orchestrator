"""
Example: a peer panel for a high-stakes decision.

Two "models" get the same question independently; neither sees the other's
answer. The synthesizer sees both, treats their disagreement as signal, and
merges the best of each. Runs with stub functions and no API keys.

In production each panelist wraps a different provider (for example a Claude
reasoning model and a peer model from another vendor), so their failure modes
do not correlate.
"""

from agent_orchestrator import Opinion, run_panel

QUESTION = "Should the service split its monolith before or after the traffic migration?"


# Stub panelists standing in for two models from different vendors.
def deep_reasoner(task: str) -> str:
    return "Split after migration: the migration needs a stable artifact to cut over."


def peer_engineer(task: str) -> str:
    return "Split the auth module first: it blocks both efforts and is the smallest seam."


def synthesizer(task: str, opinions: list[Opinion]) -> str:
    views = "\n".join(f"- {o.panelist}: {o.output}" for o in opinions)
    # A real synthesizer is an LLM call; this stub just shows what it would see.
    return (
        f"Question: {task}\n"
        f"Independent views:\n{views}\n"
        "Synthesis: migrate on the stable monolith, carving out only auth first."
    )


print("=== Peer panel: blind opinions, synthesized together ===")
result = run_panel(
    QUESTION,
    {"deep-reasoner": deep_reasoner, "peer-engineer": peer_engineer},
    synthesizer,
    max_workers=2,
)
print(result.output)
print()
print("=== Opinions recorded for the audit trail ===")
for opinion in result.opinions:
    print(f"  {opinion.panelist}: ok={opinion.ok}")
