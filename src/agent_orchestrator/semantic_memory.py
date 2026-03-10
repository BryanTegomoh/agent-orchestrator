"""
Semantic memory layer: vector search over long-term agent memory.

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │                  Semantic Memory Stack                   │
  │                                                          │
  │  MemoryManager (file layer)  →  fast, structured recall  │
  │  SemanticMemory  (vector layer) →  fuzzy, cross-session  │
  └─────────────────────────────────────────────────────────┘

Two-layer memory model:
  1. File layer (MemoryManager): MEMORY.md index + topic files.
     Good for: structured facts, active context, topic lookups.

  2. Vector layer (SemanticMemory): LanceDB table of embedded memory
     fragments. Good for: "what did we decide about X six weeks ago?"
     — queries that keyword search can't answer.

The vector layer ingests session transcripts, extracts memorable facts
via an LLM, embeds them, and stores them in LanceDB. At recall time,
a query is embedded and the top-k most semantically similar memories
are returned regardless of when they were created.

Usage:
    mem = SemanticMemory(db_path="./memory/lancedb")
    mem.ingest_session("2026-03-09", transcript_text, extract_fn=llm_extractor)
    results = mem.query("decisions about job search strategy")
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryFragment:
    """A single extracted memory, ready to embed and store."""
    id: str
    agent: str
    session_date: str
    content: str
    category: str           # fact | decision | preference | entity | context
    source_session: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    embedding: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecallResult:
    fragment: MemoryFragment
    score: float            # cosine similarity, 0.0–1.0


# ── Extraction prompt template ─────────────────────────────────────────────────
EXTRACTION_PROMPT = """
You are extracting memorable facts from an AI assistant session transcript.

Extract ONLY information that would be useful for a future session to know:
- Decisions made (what was decided and why)
- Preferences stated (how the user likes things done)
- Facts about the user (skills, constraints, goals, relationships)
- Project state (what was completed, what is in progress)
- Named entities (people, organizations, tools relevant to the user)

Return a JSON array. Each item must have:
  - "content": the fact as a clear, standalone sentence
  - "category": one of [fact, decision, preference, entity, context]

If nothing is worth preserving, return [].

Transcript:
{transcript}
"""


class SemanticMemory:
    """
    Vector-backed long-term memory using LanceDB.

    Ingests session transcripts → extracts facts via LLM → embeds → stores.
    Queries return semantically similar memory fragments across all sessions.

    Requires:
        pip install lancedb
        pip install openai  # or any embedding provider

    Example:
        import openai

        def embed(texts: list[str]) -> list[list[float]]:
            resp = openai.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            return [r.embedding for r in resp.data]

        def extract(transcript: str) -> list[dict]:
            resp = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(transcript=transcript)}],
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)

        mem = SemanticMemory("./memory/lancedb", embed_fn=embed)
        mem.ingest_session("2026-03-09", transcript, extract_fn=extract)
        results = mem.query("job search decisions last month")
    """

    TABLE_NAME = "memories"
    EMBEDDING_DIM = 1536        # text-embedding-3-small default
    TOP_K_DEFAULT = 5

    def __init__(
        self,
        db_path: str | Path,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        agent_id: str = "default",
    ):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.embed_fn = embed_fn
        self.agent_id = agent_id
        self._db = None
        self._table = None

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def ingest_session(
        self,
        session_date: str,
        transcript: str,
        extract_fn: Callable[[str], list[dict]],
        source_label: str = "",
    ) -> list[MemoryFragment]:
        """
        Extract memorable facts from a session transcript and store them.

        extract_fn: (transcript: str) -> list[{"content": str, "category": str}]
        Returns the list of MemoryFragments that were ingested.
        """
        if not self.embed_fn:
            raise RuntimeError("embed_fn is required for ingestion. See class docstring.")

        raw_facts = extract_fn(transcript)
        if not raw_facts:
            logger.info("No memorable facts extracted from session %s", session_date)
            return []

        fragments = [
            MemoryFragment(
                id=str(uuid.uuid4()),
                agent=self.agent_id,
                session_date=session_date,
                content=f["content"],
                category=f.get("category", "fact"),
                source_session=source_label or session_date,
            )
            for f in raw_facts
            if f.get("content")
        ]

        texts = [f.content for f in fragments]
        embeddings = self.embed_fn(texts)
        for fragment, embedding in zip(fragments, embeddings):
            fragment.embedding = embedding

        self._upsert(fragments)
        logger.info("Ingested %d memory fragments from session %s", len(fragments), session_date)
        return fragments

    def ingest_fact(self, content: str, category: str = "fact", metadata: dict | None = None) -> MemoryFragment:
        """Ingest a single fact directly (no transcript, no LLM extraction)."""
        if not self.embed_fn:
            raise RuntimeError("embed_fn is required. See class docstring.")

        fragment = MemoryFragment(
            id=str(uuid.uuid4()),
            agent=self.agent_id,
            session_date=datetime.now().strftime("%Y-%m-%d"),
            content=content,
            category=category,
            source_session="direct",
            metadata=metadata or {},
        )
        fragment.embedding = self.embed_fn([content])[0]
        self._upsert([fragment])
        return fragment

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def query(
        self,
        query: str,
        top_k: int = TOP_K_DEFAULT,
        category_filter: str | None = None,
        min_score: float = 0.0,
    ) -> list[RecallResult]:
        """
        Semantic search over all stored memory fragments.

        Returns the top-k most similar fragments to the query, optionally
        filtered by category and minimum similarity score.
        """
        if not self.embed_fn:
            raise RuntimeError("embed_fn is required for queries. See class docstring.")

        query_embedding = self.embed_fn([query])[0]

        try:
            table = self._get_table()
        except Exception:
            return []

        try:
            import lancedb  # type: ignore[import]
            q = table.search(query_embedding).limit(top_k)
            if category_filter:
                q = q.where(f"category = '{category_filter}'")
            rows = q.to_list()
        except Exception as exc:
            logger.error("LanceDB query failed: %s", exc)
            return []

        results = []
        for row in rows:
            # LanceDB returns _distance (L2); convert to similarity score
            distance = row.get("_distance", 1.0)
            score = max(0.0, 1.0 - distance)
            if score < min_score:
                continue
            fragment = MemoryFragment(
                id=row.get("id", ""),
                agent=row.get("agent", ""),
                session_date=row.get("session_date", ""),
                content=row.get("content", ""),
                category=row.get("category", "fact"),
                source_session=row.get("source_session", ""),
                created_at=row.get("created_at", ""),
            )
            results.append(RecallResult(fragment=fragment, score=round(score, 3)))

        return results

    def format_for_context(self, results: list[RecallResult], max_chars: int = 2000) -> str:
        """Format recall results into a context string for an LLM prompt."""
        if not results:
            return ""

        lines = ["Relevant memory from prior sessions:"]
        total = 0
        for r in results:
            line = f"- [{r.fragment.category}] {r.fragment.content}"
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line)

        return "\n".join(lines)

    # ── Stats ──────────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Number of memory fragments stored."""
        try:
            return self._get_table().count_rows()
        except Exception:
            return 0

    # ── Internals ──────────────────────────────────────────────────────────────

    def _get_table(self):
        """Lazy-connect to LanceDB and open the memories table."""
        if self._table is not None:
            return self._table

        try:
            import lancedb  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "lancedb is required for SemanticMemory. Run: pip install lancedb"
            ) from e

        self._db = lancedb.connect(str(self.db_path))
        existing = self._db.table_names()

        if self.TABLE_NAME not in existing:
            schema = self._make_schema()
            self._table = self._db.create_table(self.TABLE_NAME, schema=schema)
        else:
            self._table = self._db.open_table(self.TABLE_NAME)

        return self._table

    def _upsert(self, fragments: list[MemoryFragment]) -> None:
        table = self._get_table()
        rows = [
            {
                "id": f.id,
                "agent": f.agent,
                "session_date": f.session_date,
                "content": f.content,
                "category": f.category,
                "source_session": f.source_session,
                "created_at": f.created_at,
                "vector": f.embedding,
            }
            for f in fragments
            if f.embedding is not None
        ]
        if rows:
            table.add(rows)

    def _make_schema(self):
        try:
            import pyarrow as pa  # type: ignore[import]
        except ImportError as e:
            raise ImportError("pyarrow is required: pip install pyarrow") from e

        return pa.schema([
            pa.field("id", pa.utf8()),
            pa.field("agent", pa.utf8()),
            pa.field("session_date", pa.utf8()),
            pa.field("content", pa.utf8()),
            pa.field("category", pa.utf8()),
            pa.field("source_session", pa.utf8()),
            pa.field("created_at", pa.utf8()),
            pa.field("vector", pa.list_(pa.float32(), self.EMBEDDING_DIM)),
        ])
