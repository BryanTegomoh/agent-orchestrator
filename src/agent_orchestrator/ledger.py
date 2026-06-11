"""
Append-only run ledger: the audit trail of what the orchestrator did.

Records meaningful lifecycle events (task started, done, failed, parked on an
owner decision, grants used) as one JSON object per line. Append-only by
design: entries are never rewritten, so the file is a faithful history even
after a crash. Record decisions and outcomes, not routine polling, and never
write secrets into it.

The ledger is an audit trail, not a checkpoint: it tells you what happened,
it does not restore graph state.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LedgerEvent:
    at: str          # ISO timestamp
    event: str       # e.g. task_started, task_done, task_failed, task_blocked
    task: str | None
    detail: str


class Ledger:
    """
    Append-only JSONL event log, safe to share across parallel waves.

    Usage:
        ledger = Ledger("./memory/run-ledger.jsonl")
        ledger.record("task_done", task="t3", detail="synthesis complete")
        for e in ledger.events():
            print(e.at, e.event, e.task, e.detail)
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, event: str, task: str | None = None, detail: str = "") -> None:
        """Append one event. Never include secrets in detail."""
        row: dict[str, Any] = {
            "at": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            "task": task,
            "detail": detail,
        }
        line = json.dumps(row, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def events(self) -> list[LedgerEvent]:
        """All recorded events, oldest first. Malformed lines are skipped."""
        if not self.path.exists():
            return []
        out: list[LedgerEvent] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                out.append(
                    LedgerEvent(
                        at=str(row.get("at", "")),
                        event=str(row.get("event", "")),
                        task=row.get("task"),
                        detail=str(row.get("detail", "")),
                    )
                )
            except (json.JSONDecodeError, AttributeError):
                continue
        return out
