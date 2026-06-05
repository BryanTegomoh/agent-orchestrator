"""
Example: Semantic memory with LanceDB.

Shows the ingest -> store -> query -> format mechanics end to end. To keep this
runnable with no API keys, it uses a deterministic bag-of-words stub for
embeddings, so retrieval here ranks by token overlap, not by meaning. Plug in a
real embedding model (text-embedding-3-small, nomic-embed-text, etc.) for
genuine semantic recall; the API is identical.

Requires: pip install -e ".[lancedb]"
"""

# ── Stub embedding and extraction functions ────────────────────────────────────
# Replace these with real implementations connected to your embedding provider.

import hashlib
import shutil
from pathlib import Path

from agent_orchestrator.semantic_memory import SemanticMemory


def stub_embed(texts: list[str]) -> list[list[float]]:
    """
    Deterministic bag-of-words stub: hashes each token into a fixed-width vector.
    This is a lexical placeholder, NOT a semantic model. Swap in a real embedding
    provider (same signature) for meaning-based recall.
    """
    results = []
    for text in texts:
        vec = [0.0] * 1536
        for token in text.lower().split():
            token = token.strip(".,:;!?()[]\"'")
            if not token:
                continue
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % 1536
            vec[idx] += 1.0
        results.append(vec)
    return results


def stub_extract(transcript: str) -> list[dict]:
    """Stub: returns hardcoded facts. Replace with real LLM extraction."""
    return [
        {"content": "User prefers concise responses, no filler.", "category": "preference"},
        {"content": "Building a multi-agent orchestration system in Python.", "category": "fact"},
        {"content": "Chose LanceDB over Chroma for local performance.", "category": "decision"},
        {"content": "User is targeting roles at AI safety organizations.", "category": "context"},
    ]


# ── Demo ───────────────────────────────────────────────────────────────────────

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

# Query the store. With the bag-of-words stub these rank by token overlap; a
# real embedding model would rank by meaning, surfacing paraphrases too.
print("=== Query: 'LanceDB Chroma local performance' ===")
results = mem.query("LanceDB Chroma local performance", top_k=3)
for r in results:
    print(f"  [score={r.score:.3f}] [{r.fragment.category}] {r.fragment.content[:80]}")
print()

print("=== Query: 'multi-agent orchestration in Python' ===")
results = mem.query("multi-agent orchestration in Python", top_k=3)
for r in results:
    print(f"  [score={r.score:.3f}] [{r.fragment.category}] {r.fragment.content[:80]}")
print()

# Format for LLM context injection
print("=== Formatted for prompt context ===")
results = mem.query("local models LM Studio", top_k=4)
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
