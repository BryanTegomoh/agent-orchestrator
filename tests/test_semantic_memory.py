"""Tests for the semantic memory layer.

The pure score-conversion math is tested unconditionally. The end-to-end
LanceDB round-trip is skipped when lancedb is not installed, but is designed
to FAIL under the old L2-default behavior and PASS once the search uses the
cosine metric, so CI (which installs lancedb via the dev extra) regression-
guards the fix.
"""

import pytest

from agent_orchestrator.semantic_memory import SemanticMemory

DIM = SemanticMemory.EMBEDDING_DIM


def _vec(*pairs: tuple[int, float]) -> list[float]:
    v = [0.0] * DIM
    for idx, val in pairs:
        v[idx] = val
    return v


def _make_embed(mapping: dict[str, list[float]]):
    def embed(texts: list[str]) -> list[list[float]]:
        return [mapping.get(t, _vec((10, 1.0))) for t in texts]

    return embed


# ── Pure conversion math (no lancedb needed) ────────────────────────────────────


def test_distance_to_score_identical_is_one():
    assert SemanticMemory._distance_to_score(0.0) == 1.0


def test_distance_to_score_midpoint():
    assert SemanticMemory._distance_to_score(0.5) == 0.5


def test_distance_to_score_orthogonal_is_zero():
    assert SemanticMemory._distance_to_score(1.0) == 0.0


def test_distance_to_score_clamps_negative():
    # Cosine distance can reach 2.0 (anti-correlated); score must not go negative.
    assert SemanticMemory._distance_to_score(2.0) == 0.0
    assert SemanticMemory._distance_to_score(1.5) == 0.0


# ── End-to-end round trip (requires lancedb) ────────────────────────────────────


def test_query_scores_are_meaningful(tmp_path):
    pytest.importorskip("lancedb")
    pytest.importorskip("pyarrow")

    # Stored "fruit" vector and the query vector point the same direction but
    # have different magnitudes: cosine similarity is 1.0, yet the L2 distance
    # between them is 2.0. Under the old L2 default the score clamped to 0.0;
    # under the cosine metric it is ~1.0.
    mapping = {
        "Apples are a healthy fruit.": _vec((0, 3.0)),
        "Car engines burn fuel.": _vec((1, 3.0)),
        "fruit": _vec((0, 1.0)),
    }
    mem = SemanticMemory(db_path=tmp_path / "db", embed_fn=_make_embed(mapping))
    mem.ingest_fact("Apples are a healthy fruit.", category="fact")
    mem.ingest_fact("Car engines burn fuel.", category="fact")

    results = mem.query("fruit", top_k=2)
    assert results, "expected at least one result"
    assert "fruit" in results[0].fragment.content.lower()
    assert results[0].score > 0.5  # would be 0.0 under the L2 bug
    assert results[-1].score < results[0].score

    # min_score filtering only works once scores are meaningful.
    filtered = mem.query("fruit", top_k=2, min_score=0.5)
    assert len(filtered) == 1
    assert "fruit" in filtered[0].fragment.content.lower()


def test_count_round_trips(tmp_path):
    pytest.importorskip("lancedb")
    mem = SemanticMemory(
        db_path=tmp_path / "db",
        embed_fn=_make_embed({"a": _vec((0, 1.0)), "b": _vec((1, 1.0))}),
    )
    assert mem.count() == 0
    mem.ingest_fact("a")
    mem.ingest_fact("b")
    assert mem.count() == 2
