"""Tests for the file-based memory manager."""

from datetime import datetime, timedelta

import pytest

from agent_orchestrator.memory import MAX_SECTION_LINES, MemoryManager


@pytest.fixture
def mem(tmp_path):
    return MemoryManager(tmp_path / "memory")


def test_write_creates_topic_file_and_index_pointer(mem):
    path = mem.write("project-alpha", "API design phase complete.")
    assert path.exists()
    assert "API design phase complete." in path.read_text()
    assert path.name in mem.index_file.read_text()


def test_write_appends_to_existing_topic(mem):
    mem.write("notes", "first entry")
    path = mem.write("notes", "second entry")
    content = path.read_text()
    assert "first entry" in content
    assert "second entry" in content


def test_recall_loads_matching_topic(mem):
    mem.write("storage", "Chose LanceDB for the vector layer.")
    mem.write("hiring", "Phone screen scheduled for Friday.")
    result = mem.recall("lancedb")
    assert "LanceDB" in result
    assert "Phone screen" not in result


def test_recall_matches_whole_words_only(mem):
    mem.write("changelog", "updated the build pipeline yesterday")
    result = mem.recall("date")
    # "date" must not match inside "updated"
    assert "build pipeline" not in result


def test_recall_includes_active_context(mem):
    mem.update_active_context("Closed issue #142.")
    result = mem.recall("anything-unrelated")
    assert "Closed issue #142." in result


def test_active_context_expires_old_entries(mem):
    old = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M")
    mem.active_context_file.write_text(f"- [{old}] ancient entry\n")
    mem.update_active_context("fresh entry")
    content = mem.active_context_file.read_text()
    assert "ancient entry" not in content
    assert "fresh entry" in content


def test_active_context_preserves_untimestamped_lines(mem):
    mem.active_context_file.write_text("manual note without a timestamp\n")
    mem.update_active_context("fresh entry")
    content = mem.active_context_file.read_text()
    assert "manual note without a timestamp" in content


def test_compact_splits_oversized_sections(mem):
    body = "\n".join(f"- line {i}" for i in range(MAX_SECTION_LINES + 5))
    mem.index_file.write_text(f"# Memory Index\n\n## Meeting Notes\n{body}\n")
    splits = mem.compact()
    assert splits == ["## Meeting Notes"]
    topic = mem.memory_dir / "meeting-notes.md"
    assert topic.exists()
    assert "line 3" in topic.read_text()
    # The index now points at the topic file instead of holding the body.
    index = mem.index_file.read_text()
    assert "meeting-notes.md" in index
    assert "- line 3" not in index


def test_compact_leaves_small_sections_alone(mem):
    mem.index_file.write_text("# Memory Index\n\n## Small\n- one line\n")
    assert mem.compact() == []
    assert "- one line" in mem.index_file.read_text()
