"""
agent-orchestrator: a typed reference implementation of three multi-agent patterns:
task-based model routing, two-layer persistent memory, and prompt-injection screening.
"""

from .memory import MemoryManager
from .orchestrator import Orchestrator
from .router import TaskRouter, TaskType
from .security import ContentFilter
from .semantic_memory import EXTRACTION_PROMPT, MemoryFragment, RecallResult, SemanticMemory

__all__ = [
    "Orchestrator",
    "TaskRouter",
    "TaskType",
    "MemoryManager",
    "SemanticMemory",
    "MemoryFragment",
    "RecallResult",
    "ContentFilter",
    "EXTRACTION_PROMPT",
]
__version__ = "0.2.0"
