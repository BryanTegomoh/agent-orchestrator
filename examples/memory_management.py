"""
Example: Persistent memory management across sessions.

Shows how to write, recall, and compact memory so agents maintain
continuity without carrying stale context in their prompts.
"""

import shutil
from pathlib import Path

from agent_orchestrator import MemoryManager

# Use a temp directory so this example is self-contained
MEMORY_DIR = Path("/tmp/example-memory-demo")
if MEMORY_DIR.exists():
    shutil.rmtree(MEMORY_DIR)

mem = MemoryManager(MEMORY_DIR)

# ── Write some memory ──────────────────────────────────────────────────────────
mem.write("job search", "Applied to Anthropic Biosafety Research Scientist (Feb 2026).")
mem.write("job search", "Following up with OpenAI Biological Risk Research Lead.")
mem.write("immigration", "EB-1A RFE response due April 20, 2026. Premium processing active.")
mem.write("projects", "Three handbooks live: biosecurity, public health AI, physician AI.")
mem.update_active_context("Researching multi-agent memory patterns.")
mem.update_active_context("Drafted cold outreach to three biosecurity organizations.")

print("=== Files created ===")
for f in sorted(MEMORY_DIR.glob("*.md")):
    print(f"  {f.name} ({f.stat().st_size} bytes)")
print()

# ── Recall relevant context ────────────────────────────────────────────────────
print("=== Recall: 'job search' ===")
result = mem.recall("job search")
print(result[:400])
print("...\n")

print("=== Recall: 'immigration visa' ===")
result = mem.recall("immigration visa")
print(result[:400])
print("...\n")

# ── Compaction (auto-split bloated sections) ───────────────────────────────────
# Write a deliberately long section to trigger auto-split
long_content = "\n".join(f"- Meeting note {i}: discussed strategy item #{i}" for i in range(1, 40))
mem._update_index_section("Meeting Notes", long_content)

splits = mem.compact()
if splits:
    print(f"=== Compacted {len(splits)} bloated section(s) ===")
    for s in splits:
        print(f"  Split: '{s}'")
    print()

print("=== Files after compaction ===")
for f in sorted(MEMORY_DIR.glob("*.md")):
    lines = len(f.read_text().splitlines())
    print(f"  {f.name} ({lines} lines)")

# Cleanup
shutil.rmtree(MEMORY_DIR)
