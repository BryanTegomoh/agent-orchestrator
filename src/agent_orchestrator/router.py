"""
Task router: classifies incoming tasks and routes them to the right model.

Core philosophy: different models have different strengths. The orchestrator's
job is to route intelligently, not to do everything itself.

Routing logic:
  - Code / debugging / scripts        → GPT-5.4 Pro (SWE-bench leader)
  - Structured reasoning / agentic    → Gemini 3.1 Pro (GPQA + APEX leader)
  - Real-time data / news             → Grok (live search)
  - Long-form writing / synthesis     → Claude Opus (200K context)
  - Fast classification / triage      → Haiku / Flash (cost-efficient)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskType(Enum):
    CODE = "code"
    REASONING = "reasoning"
    RESEARCH = "research"
    WRITING = "writing"
    TRIAGE = "triage"
    UNKNOWN = "unknown"


@dataclass
class RoutingDecision:
    task_type: TaskType
    model: str
    agent: Optional[str]          # named sub-agent if applicable
    rationale: str
    confidence: float             # 0.0 – 1.0
    fallback_model: str


# ── Keyword signals per task type ─────────────────────────────────────────────
_CODE_SIGNALS = re.compile(
    r"\b(code|script|function|debug|refactor|implement|python|javascript|typescript"
    r"|rust|sql|bash|dockerfile|api|endpoint|test|unit test|pr|pull request"
    r"|git|repository|package|module|class|algorithm|parse|compile)\b",
    re.IGNORECASE,
)

_REASONING_SIGNALS = re.compile(
    r"\b(analyze|compare|evaluate|strategy|architecture|design|tradeoff|decision"
    r"|plan|structure|orchestrat|workflow|pipeline|reason|synthesize|assess"
    r"|recommend|optimize|prioritize)\b",
    re.IGNORECASE,
)

_RESEARCH_SIGNALS = re.compile(
    r"\b(search|find|news|latest|current|today|recent|trending|what is|who is"
    r"|look up|fetch|browse|web|internet|article|paper|study|research)\b",
    re.IGNORECASE,
)

_WRITING_SIGNALS = re.compile(
    r"\b(write|draft|email|letter|message|post|tweet|summarize|explain|describe"
    r"|essay|report|document|readme|proposal|cover letter|bio|profile)\b",
    re.IGNORECASE,
)


class TaskRouter:
    """
    Routes tasks to the appropriate model based on keyword signals and heuristics.

    The router is intentionally simple: O(1) per decision, no LLM call required.
    For ambiguous tasks, the orchestrator falls back to asking the primary model
    to self-classify before routing.
    """

    def __init__(self, model_map: dict[str, str] | None = None):
        self.model_map = model_map or {
            TaskType.CODE: "openai/gpt-5.4-pro",
            TaskType.REASONING: "google/gemini-3.1-pro-preview-customtools",
            TaskType.RESEARCH: "x-ai/grok-4-1",
            TaskType.WRITING: "anthropic/claude-opus-4-6",
            TaskType.TRIAGE: "anthropic/claude-haiku-4-5",
            TaskType.UNKNOWN: "google/gemini-3.1-pro-preview",
        }
        self.fallback_model = "google/gemini-3.1-pro-preview"

    # ── Public API ─────────────────────────────────────────────────────────────

    def route(self, task: str) -> RoutingDecision:
        """Classify a task string and return a routing decision."""
        task_type, confidence = self._classify(task)
        model = self.model_map.get(task_type, self.fallback_model)

        return RoutingDecision(
            task_type=task_type,
            model=model,
            agent=self._agent_for(task_type),
            rationale=self._rationale(task_type),
            confidence=confidence,
            fallback_model=self.fallback_model,
        )

    # ── Classification ─────────────────────────────────────────────────────────

    def _classify(self, task: str) -> tuple[TaskType, float]:
        scores: dict[TaskType, int] = {
            TaskType.CODE: len(_CODE_SIGNALS.findall(task)),
            TaskType.REASONING: len(_REASONING_SIGNALS.findall(task)),
            TaskType.RESEARCH: len(_RESEARCH_SIGNALS.findall(task)),
            TaskType.WRITING: len(_WRITING_SIGNALS.findall(task)),
        }
        best_type = max(scores, key=lambda t: scores[t])
        best_score = scores[best_type]

        if best_score == 0:
            return TaskType.UNKNOWN, 0.5

        total = sum(scores.values()) or 1
        confidence = min(best_score / total + 0.3, 1.0)  # floor at 30% for any match
        return best_type, round(confidence, 2)

    def _agent_for(self, task_type: TaskType) -> Optional[str]:
        return {
            TaskType.CODE: "coding-agent",
            TaskType.RESEARCH: "research-agent",
        }.get(task_type)

    def _rationale(self, task_type: TaskType) -> str:
        return {
            TaskType.CODE: "Code tasks routed to GPT-5.4 Pro (top SWE-bench score)",
            TaskType.REASONING: "Structured reasoning routed to Gemini 3.1 Pro (GPQA 94.3%)",
            TaskType.RESEARCH: "Real-time data routed to Grok (live search access)",
            TaskType.WRITING: "Long-form writing routed to Claude Opus (200K context)",
            TaskType.TRIAGE: "Fast classification routed to Haiku (cost-efficient)",
            TaskType.UNKNOWN: "Unknown task type, using primary model",
        }.get(task_type, "No rationale available")
