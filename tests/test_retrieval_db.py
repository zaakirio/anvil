"""Integration tests against the real Postgres + pgvector store (keyless, fallback
embedder). Skipped automatically when the database is unreachable."""

import pytest
from conftest import requires_db

from anvil import db
from anvil.evals.retrieval import load_goldens, validate_labels
from anvil.retrieval import search

pytestmark = requires_db


def _corpus_ingested() -> bool:
    with db.connect() as conn:
        row = conn.execute("SELECT count(*) FROM chunks").fetchone()
        return row[0] > 0


@pytest.fixture(autouse=True)
def needs_corpus():
    if not _corpus_ingested():
        pytest.skip("corpus not ingested; run `anvil ingest --embedder fallback`")


def test_hybrid_search_finds_cors_docs():
    results = search("How do I allow cross-origin requests?", top_k=5, embedder_kind="fallback")
    assert results
    assert any(c.doc_slug == "tutorial-cors" for c in results)
    assert all(c.rerank_score is not None for c in results)


def test_rerank_scores_are_finite():
    import math

    results = search("How do I upload a file?", top_k=5, embedder_kind="fallback")
    assert all(math.isfinite(c.rerank_score) for c in results)


def test_golden_labels_all_exist_in_corpus():
    validate_labels(load_goldens())


def test_embedder_mismatch_is_rejected(monkeypatch):
    from anvil import retrieval as retrieval_mod

    class WrongEmbedder:
        name = "openai/text-embedding-3-small"
        dim = 1536

        def embed_query(self, text):
            return [0.0] * 1536

    monkeypatch.setattr(retrieval_mod, "get_embedder", lambda kind: WrongEmbedder())
    with pytest.raises(RuntimeError, match="ingested with embedder"):
        search("anything", embedder_kind="openai")
