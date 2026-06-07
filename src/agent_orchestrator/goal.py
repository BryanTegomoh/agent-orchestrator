"""
Goal loop: run a worker until a judge accepts the result or the budget is spent.

The worker produces an attempt, a judge scores it against the goal, and the loop
repeats with the judge's feedback until the judge accepts or ``max_turns`` is
reached. When the budget is exhausted without acceptance, the result status is
``"exhausted"``, never a false ``"done"``: an unmet goal is reported, not hidden.

This is the same shape as an agent that keeps working until an acceptance check
passes, with two deliberate properties: the worker keeps full context across
turns (the judge's feedback is fed back in), and failure surfaces rather than
being silently swallowed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

GoalStatus = Literal["done", "exhausted"]


@dataclass
class Verdict:
    """A judge's assessment of one attempt."""

    done: bool
    feedback: str = ""


@dataclass
class GoalResult:
    status: GoalStatus
    turns: int
    output: str
    history: list[Verdict] = field(default_factory=list)


# A worker turns (goal, latest feedback) into an attempt.
Worker = Callable[[str, str], str]
# A judge scores (goal, attempt) and returns a verdict.
Judge = Callable[[str, str], Verdict]


def run_goal(
    goal: str,
    worker: Worker,
    judge: Judge,
    *,
    max_turns: int = 20,
) -> GoalResult:
    """
    Drive ``worker`` toward ``goal`` until ``judge`` accepts or the budget ends.

    Returns a ``GoalResult`` whose ``status`` is ``"done"`` only if the judge
    accepted an attempt. If the budget runs out first, ``status`` is
    ``"exhausted"`` so the caller cannot mistake an unmet goal for success.
    """
    if max_turns < 1:
        raise ValueError("max_turns must be >= 1")

    feedback = ""
    output = ""
    history: list[Verdict] = []
    for turn in range(1, max_turns + 1):
        output = worker(goal, feedback)
        verdict = judge(goal, output)
        history.append(verdict)
        if verdict.done:
            return GoalResult("done", turn, output, history)
        feedback = verdict.feedback

    return GoalResult("exhausted", max_turns, output, history)
