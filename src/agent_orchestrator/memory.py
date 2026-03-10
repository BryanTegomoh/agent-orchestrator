"""
Persistent memory manager for multi-session agent continuity.

Architecture:
  - MEMORY.md        → lean index, loaded every session (kept under 200 lines)
  - memory/<topic>.md → deep topic files, loaded on demand
  - active-context.md → rolling 7-day state (what's in flight right now)

Bloat rule: any section in MEMORY.md that grows past MAX_SECTION_LINES
is auto-split into a dedicated topic file with a one-line pointer.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

MAX_SECTION_LINES = 30
ACTIVE_CONTEXT_DAYS = 7


class MemoryManager:
    """
    Manages structured memory files for cross-session agent continuity.

    Usage:
        mem = MemoryManager("/path/to/workspace/memory")
        mem.recall("job search")          # load relevant topic files
        mem.write("job search", content)  # persist a fact
        mem.compact()                     # split bloated sections
    """

    def __init__(self, memory_dir: str | Path):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.memory_dir / "MEMORY.md"
        self.active_context_file = self.memory_dir / "active-context.md"
        self._ensure_index()

    # ── Public API ─────────────────────────────────────────────────────────────

    def recall(self, query: str) -> str:
        """
        Return memory content relevant to the query.
        Searches the index and loads matching topic files.
        """
        results: list[str] = []

        # Always include active context
        if self.active_context_file.exists():
            results.append(f"## Active Context\n{self.active_context_file.read_text()}")

        # Search topic files for keyword matches
        for topic_file in self.memory_dir.glob("*.md"):
            if topic_file.name in ("MEMORY.md", "active-context.md"):
                continue
            content = topic_file.read_text()
            if self._matches(query, content):
                results.append(f"## {topic_file.stem}\n{content}")

        return "\n\n---\n\n".join(results) if results else "No relevant memory found."

    def write(self, topic: str, content: str, section: str | None = None) -> Path:
        """
        Persist content to memory. Creates or updates a topic file.
        If section is provided, updates that section within MEMORY.md.
        """
        if section:
            return self._update_index_section(section, content)

        topic_slug = self._slugify(topic)
        topic_file = self.memory_dir / f"{topic_slug}.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        if topic_file.exists():
            existing = topic_file.read_text()
            updated = f"{existing}\n\n<!-- updated {timestamp} -->\n{content}"
        else:
            updated = f"# {topic}\n\n<!-- created {timestamp} -->\n{content}"

        topic_file.write_text(updated)
        self._upsert_index_pointer(topic, topic_file)
        return topic_file

    def update_active_context(self, entry: str) -> None:
        """Append an entry to the rolling active-context file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        line = f"- [{timestamp}] {entry}\n"

        existing = self.active_context_file.read_text() if self.active_context_file.exists() else ""
        lines = existing.splitlines(keepends=True)

        # Keep only recent entries (rolling window)
        cutoff = datetime.now().timestamp() - (ACTIVE_CONTEXT_DAYS * 86400)
        recent = [ln for ln in lines if not self._is_expired(ln, cutoff)]
        self.active_context_file.write_text("".join(recent) + line)

    def compact(self) -> list[str]:
        """
        Split any bloated MEMORY.md sections into dedicated topic files.
        Returns list of sections that were split.
        """
        if not self.index_file.exists():
            return []

        content = self.index_file.read_text()
        section_pattern = re.compile(r"^(## .+)$", re.MULTILINE)
        positions = [(m.start(), m.group(1)) for m in section_pattern.finditer(content)]

        new_parts: list[str] = []
        prev_end = 0
        splits: list[str] = []

        for i, (start, header) in enumerate(positions):
            new_parts.append(content[prev_end:start])
            end = positions[i + 1][0] if i + 1 < len(positions) else len(content)
            section_body = content[start:end]
            body_lines = section_body.splitlines()

            if len(body_lines) > MAX_SECTION_LINES:
                slug = self._slugify(header.lstrip("# "))
                out_file = self.memory_dir / f"{slug}.md"
                if not out_file.exists():
                    out_file.write_text(f"# {header.lstrip('# ')}\n\n" + "\n".join(body_lines[1:]))
                    pointer = (
                        f"{header}\n"
                        f"> Full detail in [{out_file.name}]({out_file.name}) "
                        f"(auto-split {datetime.now().strftime('%Y-%m-%d')}).\n\n"
                    )
                    new_parts.append(pointer)
                    splits.append(header.strip())
                else:
                    new_parts.append(section_body)
            else:
                new_parts.append(section_body)

            prev_end = end

        if splits:
            self.index_file.write_text("".join(new_parts))

        return splits

    # ── Internals ──────────────────────────────────────────────────────────────

    def _ensure_index(self) -> None:
        if not self.index_file.exists():
            self.index_file.write_text(
                f"# Memory Index\n\n"
                f"_Created {datetime.now().strftime('%Y-%m-%d')}. "
                f"Sections over {MAX_SECTION_LINES} lines are auto-split into topic files._\n\n"
            )

    def _update_index_section(self, section: str, content: str) -> Path:
        existing = self.index_file.read_text()
        header = f"## {section}"
        if header in existing:
            flags = re.MULTILINE | re.DOTALL
            pattern = re.compile(rf"^## {re.escape(section)}.+?(?=^## |\Z)", flags)
            updated = pattern.sub(f"{header}\n\n{content}\n\n", existing)
        else:
            updated = existing + f"\n{header}\n\n{content}\n\n"
        self.index_file.write_text(updated)
        return self.index_file

    def _upsert_index_pointer(self, topic: str, topic_file: Path) -> None:
        existing = self.index_file.read_text()
        pointer_line = f"- [{topic}]({topic_file.name})"
        if topic_file.name not in existing:
            self.index_file.write_text(existing.rstrip() + f"\n{pointer_line}\n")

    def _matches(self, query: str, content: str) -> bool:
        query_terms = query.lower().split()
        content_lower = content.lower()
        return any(term in content_lower for term in query_terms)

    def _slugify(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

    def _is_expired(self, line: str, cutoff: float) -> bool:
        match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]", line)
        if not match:
            return False
        try:
            ts = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M").timestamp()
            return ts < cutoff
        except ValueError:
            return False
