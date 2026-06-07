"""Tests for the goal loop."""

import pytest

from agent_orchestrator.goal import Verdict, run_goal


def test_accepts_immediately():
    result = run_goal(
        "ship it",
        worker=lambda goal, feedback: "attempt",
        judge=lambda goal, output: Verdict(done=True),
    )
    assert result.status == "done"
    assert result.turns == 1


def test_feedback_is_fed_back_and_accepts_later():
    seen = {"turns": 0}

    def worker(goal: str, feedback: str) -> str:
        return f"attempt[{feedback}]"

    def judge(goal: str, output: str) -> Verdict:
        seen["turns"] += 1
        return Verdict(done=seen["turns"] >= 3, feedback=f"revise {seen['turns']}")

    result = run_goal("ship it", worker, judge, max_turns=5)
    assert result.status == "done"
    assert result.turns == 3
    assert "revise 2" in result.output  # turn 3 ran on turn 2's feedback


def test_exhaustion_never_reports_false_done():
    result = run_goal(
        "ship it",
        worker=lambda goal, feedback: "attempt",
        judge=lambda goal, output: Verdict(done=False, feedback="no"),
        max_turns=4,
    )
    assert result.status == "exhausted"
    assert result.turns == 4
    assert len(result.history) == 4


def test_invalid_budget_raises():
    with pytest.raises(ValueError):
        run_goal(
            "ship it",
            worker=lambda goal, feedback: "",
            judge=lambda goal, output: Verdict(done=True),
            max_turns=0,
        )
