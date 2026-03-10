"""
Example: Semantic memory with LanceDB.

Shows how to ingest session facts, query by meaning (not just keywords),
and combine semantic recall with the file-based MemoryManager.

Requires: pip install "agent-orchestrator[lancedb]"
"""

# ── Stub embedding and extraction functions ────────────────────────────────────
# Replace these with real implementations connected to your embedding provider.

import math
import random


def stub_embed(texts: list[str]) -> list[list[float]]:
    """Stub: returns deterministic fake embeddings. Replace with real API call."""
    results = []
    for text in texts:
        random.seed(hash(text) % (2**32))
        results.append([random.gauss(0, 1) for _ in range(1536)])
    return results


def stub_extract(transcript: str) -> list[dict]:
    """Stub: returns hardcoded facts. Replace with real LLM extraction."""
    return [
        {"content": "User prefers concise responses without em dashes or filler phrases.", "category": "preference"},
        {"content": "User is building a multi-agent orchestration system in Python.", "category": "fact"},
        {"content": "Decided to use LanceDB for semantic memory over Chroma for better local performance.", "category": "decision"},
        {"content": "User is targeting roles at AI safety organizations.", "category": "context"},
    ]


# ── Demo ───────────────────────────────────────────────────────────────────────

import shutil
from pathlib import Path

from agent_orchestrator.semantic_memory import SemanticMemory, EXTRACTION_PROMPT

DB_PATH = Path("/tmp/example-semantic-memory")
if DB_PATH.exists():
    shutil.rmtree(DB_PATH)

mem = SemanticMemory(db_path=DB_PATH, embed_fn=stub_embed, agent_id="main")

# Ingest a session
print("=== Ingesting session transcript ===")
fake_transcript = "Today we discussed memory architecture. Decided on LanceDB for the vector layer."
fragments = mem.ingest_session(
    session_date="2026-03-09",
    transcript=fake_transcript,
    extract_fn=stub_extract,
    source_label="session-2026-03-09",
)
print(f"Ingested {len(fragments)} fragments")
for f in fragments:
    print(f"  [{f.category}] {f.content[:80]}")
print()

# Add a few more facts directly
mem.ingest_fact("User prefers LM Studio for running local models.", category="preference")
mem.ingest_fact("Primary embedding model: text-embedding-3-small (1536 dim).", category="fact")

print(f"Total fragments stored: {mem.count()}\n")

# Semantic query — finds relevant memories by meaning, not keyword
print("=== Semantic query: 'communication style preferences' ===")
results = mem.query("communication style preferences", top_k=3)
for r in results:
    print(f"  [score={r.score:.3f}] [{r.fragment.category}] {r.fragment.content[:80]}")
print()

print("=== Semantic query: 'vector database choice' ===")
results = mem.query("vector database choice", top_k=3)
for r in results:
    print(f"  [score={r.score:.3f}] [{r.fragment.category}] {r.fragment.content[:80]}")
print()

# Format for LLM context injection
print("=== Formatted for prompt context ===")
results = mem.query("memory and storage architecture", top_k=4)
print(mem.format_for_context(results))

# Cleanup
shutil.rmtree(DB_PATH)


# ── Real implementation (commented out) ───────────────────────────────────────
"""
To connect real providers:

import openai
import anthropic

client_oai = openai.OpenAI()
client_ant = anthropic.Anthropic()

def embed(texts: list[str]) -> list[list[float]]:
    resp = client_oai.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [r.embedding for r in resp.data]

def extract(transcript: str) -> list[dict]:
    import json
    resp = client_ant.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": EXTRACTION_PROMPT.format(transcript=transcript),
        }]
    )
    return json.loads(resp.content[0].text)

mem = SemanticMemory(
    db_path="./memory/lancedb",
    embed_fn=embed,
    agent_id="main",
)
mem.ingest_session("2026-03-09", transcript_text, extract_fn=extract)
results = mem.query("what did we decide about the job search strategy?")
"""
