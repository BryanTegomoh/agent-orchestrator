"""
Panel: independent opinions on one question, synthesized blind.

For a high-stakes decision, put the same question to several panelists in
parallel and give every answer to a synthesizer. Two deliberate properties:
panelists never see each other's output, and only the synthesizer sees all of
them. A panelist shown another's answer anchors on it and the panel converges
on the first plausible take; kept blind, independent answers diverge exactly
where the question is genuinely hard, and that disagreement is the signal the
synthesizer needs.

This is the peer pattern, not the judge pattern. ``run_goal`` puts one worker
under an independent judge and iterates; ``run_panel`` widens a single
decision across models with different failure modes (different vendors,
different training data) and merges the best of each. Use a judge to converge
on a known standard; use a panel when no single model's blind spots should
decide alone.

Fail-loud contract: a panelist that raises is recorded, not hidden, and if
fewer panelists answer than the quorum requires, the panel raises rather than
synthesizing from thin evidence.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

__all__ = ["Opinion", "PanelError", "PanelResult", "run_panel"]


class PanelError(RuntimeError):
    """Raised when too few panelists answer to meet the quorum."""


@dataclass
class Opinion:
    """One panelist's independent answer, or its recorded failure."""

    panelist: str
    output: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class PanelResult:
    output: str
    opinions: list[Opinion] = field(default_factory=list)


# A panelist answers the task with no knowledge of the other panelists.
Panelist = Callable[[str], str]
# A synthesizer turns (task, successful opinions) into the final answer.
Synthesizer = Callable[[str, list[Opinion]], str]


def run_panel(
    task: str,
    panelists: dict[str, Panelist],
    synthesizer: Synthesizer,
    *,
    quorum: int | None = None,
    max_workers: int = 1,
) -> PanelResult:
    """
    Ask every panelist ``task`` independently, then synthesize the answers.

    Each panelist receives only the task, never another panelist's output;
    the synthesizer receives every successful opinion. A panelist that raises
    is recorded as a failed ``Opinion``. If fewer than ``quorum`` panelists
    succeed (default: all of them), raises ``PanelError`` so a thin panel is
    never mistaken for a full one.

    ``max_workers > 1`` asks panelists concurrently; opinions are returned in
    declaration order either way. Panelists must be thread-safe when run
    concurrently.
    """
    if len(panelists) < 2:
        raise ValueError("a panel needs at least two panelists; call the model directly for one")
    required = len(panelists) if quorum is None else quorum
    if not 1 <= required <= len(panelists):
        raise ValueError(f"quorum must be between 1 and {len(panelists)}, got {required}")

    def ask(name: str, panelist: Panelist) -> Opinion:
        try:
            return Opinion(panelist=name, output=panelist(task))
        except Exception as exc:
            return Opinion(panelist=name, output="", error=str(exc))

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {name: pool.submit(ask, name, fn) for name, fn in panelists.items()}
            opinions = [futures[name].result() for name in panelists]
    else:
        opinions = [ask(name, fn) for name, fn in panelists.items()]

    successful = [opinion for opinion in opinions if opinion.ok]
    if len(successful) < required:
        failures = "; ".join(
            f"{opinion.panelist}: {opinion.error}" for opinion in opinions if not opinion.ok
        )
        raise PanelError(
            f"only {len(successful)} of {len(panelists)} panelists answered, "
            f"quorum is {required} ({failures})"
        )

    return PanelResult(output=synthesizer(task, successful), opinions=opinions)
