"""
Orchestrator: the primary brain of a multi-agent system.

Core principle: the orchestrator thinks, routes, and synthesizes.
It never does grunt work directly — that's what sub-agents are for.

Execution flow:
  1. Screen incoming content for injection attempts
  2. Recall relevant memory for context
  3. Route the task to the appropriate model/agent
  4. Execute via the selected model
  5. Persist the result to memory
  6. Return synthesized output

The orchestrator itself runs on the highest-capability reasoning model.
Sub-agents run on task-optimized models (code, search, writing, etc.).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from .memory import MemoryManager
from .router import RoutingDecision, TaskRouter
from .security import ContentFilter

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    task: str
    routing: RoutingDecision
    output: str
    duration_ms: int
    memory_written: bool = False
    injection_blocked: bool = False
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None and not self.injection_blocked


class Orchestrator:
    """
    Routes tasks to specialized sub-agents and synthesizes results.

    Sub-agents are registered as callables: (task: str, model: str) -> str.
    If no sub-agent is registered for a task type, the orchestrator calls the
    primary model directly.

    Example:
        orch = Orchestrator(
            primary_model="google/gemini-3.1-pro-preview",
            memory_dir="./memory",
        )
        orch.register_agent("coding-agent", my_coding_agent_fn)
        result = orch.run("Write a Python function to parse JSON safely")
    """

    def __init__(
        self,
        primary_model: str,
        memory_dir: str = "./memory",
        strict_security: bool = False,
        model_map: dict[str, str] | None = None,
    ):
        self.primary_model = primary_model
        self.router = TaskRouter(model_map=model_map)
        self.memory = MemoryManager(memory_dir)
        self.security = ContentFilter(strict_mode=strict_security)
        self._agents: dict[str, Callable[[str, str], str]] = {}
        self._llm_caller: Callable[[str, str], str] = self._default_llm_call

    # ── Agent registry ─────────────────────────────────────────────────────────

    def register_agent(self, name: str, fn: Callable[[str, str], str]) -> None:
        """Register a sub-agent by name. fn(task, model) -> output."""
        self._agents[name] = fn
        logger.info("Registered agent: %s", name)

    def set_llm_caller(self, fn: Callable[[str, str], str]) -> None:
        """
        Override the default LLM caller. fn(prompt, model) -> response.
        Use this to inject your actual LLM client (litellm, openai, anthropic, etc.).
        """
        self._llm_caller = fn

    # ── Core execution ─────────────────────────────────────────────────────────

    def run(self, task: str, persist: bool = True) -> TaskResult:
        """
        Execute a task end-to-end:
          screen → recall → route → execute → persist → return
        """
        start = time.monotonic()

        # 1. Security screen
        screen_result = self.security.screen(task, source="user")
        if not screen_result.safe_to_process:
            logger.warning("Task blocked by security filter: %s", screen_result.recommendation)
            return TaskResult(
                task=task,
                routing=self.router.route(task),
                output="",
                duration_ms=self._elapsed_ms(start),
                injection_blocked=True,
                error=screen_result.recommendation,
            )

        # 2. Recall relevant memory
        context = self.memory.recall(task)

        # 3. Route
        routing = self.router.route(task)
        logger.info(
            "Task routed: type=%s model=%s confidence=%.2f",
            routing.task_type.value,
            routing.model,
            routing.confidence,
        )

        # 4. Build prompt
        prompt = self._build_prompt(task, context)

        # 5. Execute via sub-agent or direct LLM call
        try:
            if routing.agent and routing.agent in self._agents:
                output = self._agents[routing.agent](prompt, routing.model)
            else:
                output = self._llm_caller(prompt, routing.model)
        except Exception as exc:
            logger.error("Execution error: %s", exc)
            # Try fallback model once
            try:
                output = self._llm_caller(prompt, routing.fallback_model)
            except Exception as fallback_exc:
                return TaskResult(
                    task=task,
                    routing=routing,
                    output="",
                    duration_ms=self._elapsed_ms(start),
                    error=str(fallback_exc),
                )

        # 6. Persist
        memory_written = False
        if persist and output:
            self.memory.update_active_context(
                f"[{routing.task_type.value}] {task[:80]}{'...' if len(task) > 80 else ''}"
            )
            memory_written = True

        return TaskResult(
            task=task,
            routing=routing,
            output=output,
            duration_ms=self._elapsed_ms(start),
            memory_written=memory_written,
        )

    def run_batch(self, tasks: list[str]) -> list[TaskResult]:
        """Run multiple tasks sequentially, sharing memory context."""
        return [self.run(task) for task in tasks]

    # ── Delegation helper ──────────────────────────────────────────────────────

    def delegate(self, task: str, to_agent: str, model: str | None = None) -> str:
        """
        Explicitly delegate a task to a named agent, bypassing routing.
        Useful when the orchestrator makes a high-confidence routing decision
        before calling run().
        """
        if to_agent not in self._agents:
            raise ValueError(f"Agent '{to_agent}' not registered. Call register_agent() first.")
        effective_model = model or self.primary_model
        return self._agents[to_agent](task, effective_model)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _build_prompt(self, task: str, context: str) -> str:
        if context and context != "No relevant memory found.":
            return (
                f"Relevant context from memory:\n{context}\n\n"
                f"---\n\nTask: {task}"
            )
        return task

    def _default_llm_call(self, prompt: str, model: str) -> str:
        """
        Placeholder LLM caller. Replace with your actual client via set_llm_caller().

        Example with litellm:
            import litellm
            orch.set_llm_caller(
                lambda prompt, model: litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                ).choices[0].message.content
            )
        """
        logger.warning(
            "Using default (stub) LLM caller. Call set_llm_caller() to connect a real LLM. "
            "Model requested: %s",
            model,
        )
        return f"[STUB] Would call {model} with: {prompt[:100]}..."

    @staticmethod
    def _elapsed_ms(start: float) -> int:
        return int((time.monotonic() - start) * 1000)
