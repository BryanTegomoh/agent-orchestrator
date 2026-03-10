"""
Prompt injection defense for multi-agent systems.

Threat model: external content (web pages, emails, tool outputs, user-pasted text)
may contain instructions designed to hijack the agent's behavior. This filter
classifies incoming content before the orchestrator processes it.

Risk levels:
  LOW     - safe to process
  MEDIUM  - flag to operator before processing
  HIGH    - block, log, alert
  CRITICAL - block immediately, do not process at all
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FilterResult:
    risk_level: RiskLevel
    matched_patterns: list[str]
    recommendation: str
    safe_to_process: bool
    sanitized_content: Optional[str] = None


# ── Pattern banks ──────────────────────────────────────────────────────────────
# These are NOT exhaustive — the goal is catching common injection vectors,
# not building a complete adversarial NLP system.

_CRITICAL_PATTERNS = [
    # Direct instruction override attempts
    re.compile(r"ignore (all )?(previous|prior|above|earlier) instructions?", re.IGNORECASE),
    re.compile(r"disregard (your )?(previous|prior|system|all) (instructions?|rules?|guidelines?)", re.IGNORECASE),
    re.compile(r"you are now (a |an )?(different|new|other|jailbroken)", re.IGNORECASE),
    re.compile(r"forget (everything|all) (you were|you've been) told", re.IGNORECASE),
    re.compile(r"(act|pretend|behave|respond) as (if you are|like you are|though you are) (a |an )?(DAN|evil|uncensored|unrestricted)", re.IGNORECASE),
    # Persona hijacking
    re.compile(r"your (new|true|real) (name|identity|persona|role) is", re.IGNORECASE),
    re.compile(r"from now on (you are|you will be|act as)", re.IGNORECASE),
    re.compile(r"\[SYSTEM\].*override", re.IGNORECASE | re.DOTALL),
]

_HIGH_PATTERNS = [
    re.compile(r"<\|system\|>", re.IGNORECASE),
    re.compile(r"<INST>.*</INST>", re.IGNORECASE | re.DOTALL),
    re.compile(r"\[\[SYSTEM\]\]", re.IGNORECASE),
    re.compile(r"assistant:?\s*I will now", re.IGNORECASE),
    re.compile(r"reveal (your|the) (system prompt|instructions|configuration)", re.IGNORECASE),
    re.compile(r"print (your|the) (system prompt|instructions|configuration)", re.IGNORECASE),
    re.compile(r"exfiltrate|data exfil", re.IGNORECASE),
]

_MEDIUM_PATTERNS = [
    re.compile(r"what (are|were) your instructions?", re.IGNORECASE),
    re.compile(r"repeat (the|your) (system|original) (prompt|message|instructions?)", re.IGNORECASE),
    re.compile(r"(jailbreak|jailbroken|DAN mode)", re.IGNORECASE),
    re.compile(r"hypothetically (speaking|if you (were|could))", re.IGNORECASE),
    re.compile(r"for (educational|research) purposes? only", re.IGNORECASE),
    re.compile(r"sudo (mode|override)", re.IGNORECASE),
]


class ContentFilter:
    """
    Screens external content for prompt injection signals before the
    orchestrator processes it.

    Usage:
        cf = ContentFilter()
        result = cf.screen(scraped_webpage_content)
        if not result.safe_to_process:
            alert_operator(result)
    """

    def __init__(self, strict_mode: bool = False):
        """
        strict_mode: if True, MEDIUM-level content is also blocked (not just flagged).
        """
        self.strict_mode = strict_mode

    def screen(self, content: str, source: str = "unknown") -> FilterResult:
        """Screen content and return a risk assessment."""
        matched: list[str] = []
        highest_risk = RiskLevel.LOW

        for pattern in _CRITICAL_PATTERNS:
            if pattern.search(content):
                matched.append(f"CRITICAL: {pattern.pattern[:60]}")
                highest_risk = RiskLevel.CRITICAL

        if highest_risk != RiskLevel.CRITICAL:
            for pattern in _HIGH_PATTERNS:
                if pattern.search(content):
                    matched.append(f"HIGH: {pattern.pattern[:60]}")
                    if highest_risk.value != RiskLevel.CRITICAL.value:
                        highest_risk = RiskLevel.HIGH

        if highest_risk not in (RiskLevel.CRITICAL, RiskLevel.HIGH):
            for pattern in _MEDIUM_PATTERNS:
                if pattern.search(content):
                    matched.append(f"MEDIUM: {pattern.pattern[:60]}")
                    if highest_risk == RiskLevel.LOW:
                        highest_risk = RiskLevel.MEDIUM

        safe = self._is_safe(highest_risk)
        recommendation = self._recommendation(highest_risk, source)

        return FilterResult(
            risk_level=highest_risk,
            matched_patterns=matched,
            recommendation=recommendation,
            safe_to_process=safe,
            sanitized_content=self._sanitize(content) if safe and matched else None,
        )

    def screen_tool_output(self, tool_name: str, output: str) -> FilterResult:
        """Screen output from an external tool (web search, browser, API call)."""
        result = self.screen(output, source=f"tool:{tool_name}")
        # Tool outputs get slightly more scrutiny — add context to recommendation
        if result.risk_level != RiskLevel.LOW:
            result.recommendation = (
                f"Tool '{tool_name}' returned potentially injected content. "
                + result.recommendation
            )
        return result

    # ── Internals ──────────────────────────────────────────────────────────────

    def _is_safe(self, risk: RiskLevel) -> bool:
        if risk == RiskLevel.LOW:
            return True
        if risk == RiskLevel.MEDIUM:
            return not self.strict_mode
        return False

    def _recommendation(self, risk: RiskLevel, source: str) -> str:
        return {
            RiskLevel.LOW: "Content is clean. Safe to process.",
            RiskLevel.MEDIUM: f"Potential injection signal in content from '{source}'. Flag to operator.",
            RiskLevel.HIGH: f"Injection attempt likely in content from '{source}'. Block and log.",
            RiskLevel.CRITICAL: f"Active injection attempt in content from '{source}'. Block immediately, do not process.",
        }[risk]

    def _sanitize(self, content: str) -> str:
        """Light sanitization for MEDIUM-risk content when strict_mode is off."""
        # Strip potential fake system delimiters
        content = re.sub(r"<\|.*?\|>", "", content)
        content = re.sub(r"\[\[.*?\]\]", "", content)
        return content.strip()
