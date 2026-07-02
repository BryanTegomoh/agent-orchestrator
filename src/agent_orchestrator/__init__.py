"""
agent-orchestrator: a typed reference implementation of three multi-agent patterns:
task-based model routing, two-layer persistent memory, and prompt-injection screening.
"""

from .authority import DecisionBrief, NeedsOwner
from .goal import GoalResult, Verdict, run_goal
from .ledger import Ledger, LedgerEvent
from .memory import MemoryManager
from .orchestrator import Orchestrator
from .panel import Opinion, PanelError, PanelResult, run_panel
from .router import TaskRouter, TaskType
from .security import ContentFilter
from .semantic_memory import EXTRACTION_PROMPT, MemoryFragment, RecallResult, SemanticMemory
from .taskgraph import (
    OwnerDecisionRequired,
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
    # Bounded autonomy
    "DecisionBrief",
    "NeedsOwner",
    "OwnerDecisionRequired",
    "Ledger",
    "LedgerEvent",
    # Goal loop
    "run_goal",
    "Verdict",
    "GoalResult",
    # Panel
    "run_panel",
    "Opinion",
    "PanelResult",
    "PanelError",
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
__version__ = "0.6.0"
