"""Tests for the panel."""

import pytest

from agent_orchestrator.panel import Opinion, PanelError, run_panel


def synth_join(task: str, opinions: list[Opinion]) -> str:
    return " | ".join(o.output for o in opinions)


def test_panelists_are_blind_and_synthesizer_sees_all():
    received: dict[str, str] = {}

    def make_panelist(name: str, answer: str):
        def panelist(task: str) -> str:
            received[name] = task  # each panelist sees only the task
            return answer

        return panelist

    seen_by_synth: list[str] = []

    def synthesizer(task: str, opinions: list[Opinion]) -> str:
        seen_by_synth.extend(o.output for o in opinions)
        return "merged"

    result = run_panel(
        "decide",
        {"a": make_panelist("a", "answer-a"), "b": make_panelist("b", "answer-b")},
        synthesizer,
    )
    assert received == {"a": "decide", "b": "decide"}  # no cross-contamination
    assert seen_by_synth == ["answer-a", "answer-b"]
    assert result.output == "merged"


def test_opinions_keep_declaration_order_when_parallel():
    result = run_panel(
        "q",
        {"first": lambda t: "1", "second": lambda t: "2", "third": lambda t: "3"},
        synth_join,
        max_workers=3,
    )
    assert [o.panelist for o in result.opinions] == ["first", "second", "third"]
    assert result.output == "1 | 2 | 3"


def test_failed_panelist_breaks_default_quorum():
    def broken(task: str) -> str:
        raise RuntimeError("model unavailable")

    with pytest.raises(PanelError, match="1 of 2"):
        run_panel("q", {"ok": lambda t: "fine", "down": broken}, synth_join)


def test_quorum_tolerates_recorded_failure():
    def broken(task: str) -> str:
        raise RuntimeError("model unavailable")

    result = run_panel(
        "q",
        {"ok": lambda t: "fine", "down": broken},
        synth_join,
        quorum=1,
    )
    assert result.output == "fine"  # synthesizer saw only the successful opinion
    failed = [o for o in result.opinions if not o.ok]
    assert len(failed) == 1 and failed[0].panelist == "down"
    assert failed[0].error == "model unavailable"


def test_single_panelist_raises():
    with pytest.raises(ValueError, match="at least two"):
        run_panel("q", {"only": lambda t: "x"}, synth_join)


def test_invalid_quorum_raises():
    panel = {"a": lambda t: "x", "b": lambda t: "y"}
    with pytest.raises(ValueError, match="quorum"):
        run_panel("q", panel, synth_join, quorum=3)
    with pytest.raises(ValueError, match="quorum"):
        run_panel("q", panel, synth_join, quorum=0)
