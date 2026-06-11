"""
Bounded autonomy: permission scopes and owner-decision gates.

An orchestrator should know exactly what it is allowed to do, stop at that
boundary, and ask with a decision-ready brief instead of a vague status. Two
pieces implement that:

  * Tasks declare the grants they need (``requires=("push",)``); the scheduler
    parks any task whose grants were not given. Grants are exact strings with
    no hierarchy, on purpose: "push" does not imply "merge", and "merge" does
    not imply "release".
  * An executor that hits a judgment call mid-task raises ``NeedsOwner`` with
    a brief; the scheduler parks the task and keeps running independent lanes.

A parked task is not a failure and never reads as one. The graph finishes all
runnable work, then raises ``OwnerDecisionRequired`` (from taskgraph) carrying
one ``DecisionBrief`` per parked task, so the owner decides from a prepared
position: what changes, why now, what was already proven, the tradeoffs, a
recommendation, and the exact choices. The shape follows completed-staff-work
practice: do everything within your authority first, then present a decision,
not a problem.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionBrief:
    """A decision-ready ask: enough context to choose without re-investigating."""

    task_id: str
    title: str
    what: str                      # plain language: what changes and who benefits
    why_now: str = ""
    proof: str = ""                # evidence already gathered (tests, repro, CI)
    tradeoffs: str = ""
    recommendation: str = ""
    choices: tuple[str, ...] = ()  # the exact options on the table

    def render(self) -> str:
        """One readable block per brief, for logs and reports."""
        lines = [f"[{self.task_id}] {self.title}", f"  what: {self.what}"]
        for label, value in (
            ("why now", self.why_now),
            ("proof", self.proof),
            ("tradeoffs", self.tradeoffs),
            ("recommendation", self.recommendation),
        ):
            if value:
                lines.append(f"  {label}: {value}")
        if self.choices:
            lines.append("  choices: " + " | ".join(self.choices))
        return "\n".join(lines)


@dataclass
class NeedsOwner(Exception):
    """
    Raised by an executor when a task reaches a judgment call only the owner
    can make: a product choice, an irreversible action, missing credentials.
    The task parks as blocked; it is not a failure.
    """

    what: str
    why_now: str = ""
    proof: str = ""
    tradeoffs: str = ""
    recommendation: str = ""
    choices: tuple[str, ...] = ()

    def to_brief(self, task_id: str, title: str) -> DecisionBrief:
        return DecisionBrief(
            task_id=task_id,
            title=title,
            what=self.what,
            why_now=self.why_now,
            proof=self.proof,
            tradeoffs=self.tradeoffs,
            recommendation=self.recommendation,
            choices=self.choices,
        )


def missing_grants_brief(
    task_id: str, title: str, missing: tuple[str, ...]
) -> DecisionBrief:
    """Brief for a task parked because its required grants were not given."""
    grants = ", ".join(missing)
    return DecisionBrief(
        task_id=task_id,
        title=title,
        what=f"Requires grants not given to this run: {grants}.",
        recommendation=f"Re-run with {grants} granted, or unblock the task explicitly.",
        choices=(f"grant: {grants}", "unblock without granting", "leave parked"),
    )
