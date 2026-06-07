"""
agent-orchestrator: a typed reference implementation of three multi-agent patterns:
task-based model routing, two-layer persistent memory, and prompt-injection screening.
"""

from .goal import GoalResult, Verdict, run_goal
from .memory import MemoryManager
from .orchestrator import Orchestrator
from .router import TaskRouter, TaskType
from .security import ContentFilter
from .semantic_memory import EXTRACTION_PROMPT, MemoryFragment, RecallResult, SemanticMemory
from .taskgraph import (
    SelfReportError,
    Task,
    TaskGraph,
    TaskGraphError,
    TaskState,
    UnknownParentError,
)

__all__ = [
    # Orchestration
    "Orchestrator",
    "TaskGraph",
    "Task",
    "TaskState",
    "TaskGraphError",
    "UnknownParentError",
    "SelfReportError",
    # Goal loop
    "run_goal",
    "Verdict",
    "GoalResult",
    # Routing
    "TaskRouter",
    "TaskType",
    # Memory
    "MemoryManager",
    "SemanticMemory",
    "MemoryFragment",
    "RecallResult",
    "EXTRACTION_PROMPT",
    # Screening
    "ContentFilter",
]
__version__ = "0.3.0"
