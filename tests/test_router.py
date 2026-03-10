"""Tests for the task router."""

import pytest
from agent_orchestrator.router import TaskRouter, TaskType


@pytest.fixture
def router():
    return TaskRouter()


def test_routes_code_tasks(router):
    decision = router.route("Write a Python function to parse JSON from stdin")
    assert decision.task_type == TaskType.CODE
    assert decision.confidence > 0.5
    assert "gpt" in decision.model.lower()


def test_routes_reasoning_tasks(router):
    decision = router.route("Compare the architecture tradeoffs between RAG and fine-tuning")
    assert decision.task_type == TaskType.REASONING
    assert decision.confidence > 0.5


def test_routes_research_tasks(router):
    decision = router.route("What is the latest news on AI regulation today?")
    assert decision.task_type == TaskType.RESEARCH
    assert "grok" in decision.model.lower()


def test_routes_writing_tasks(router):
    decision = router.route("Draft an email to a VP of Engineering at a startup")
    assert decision.task_type == TaskType.WRITING


def test_unknown_task_returns_fallback(router):
    decision = router.route("7x9")
    assert decision.task_type == TaskType.UNKNOWN
    assert decision.model == router.fallback_model


def test_code_task_has_agent(router):
    decision = router.route("Implement a binary search in Rust")
    assert decision.agent == "coding-agent"


def test_rationale_is_present(router):
    decision = router.route("Debug this Python script")
    assert len(decision.rationale) > 0


def test_custom_model_map():
    custom_router = TaskRouter(model_map={TaskType.CODE: "my-custom/model"})
    decision = custom_router.route("Write a SQL query")
    assert decision.model == "my-custom/model"
