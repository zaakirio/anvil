from dataclasses import dataclass

from anvil import config, db
from anvil.embeddings import Embedder, get_embedder
from anvil.rerank import get_reranker


@dataclass
class RetrievedChunk:
    id: str
    doc_slug: str
    heading_path: str
    content: str
    fused_score: float
    rerank_score: float | None = None


def _fts_candidates(conn, query: str, limit: int) -> list[str]:
    rows = conn.execute(
        """
        SELECT id FROM chunks
        WHERE tsv @@ websearch_to_tsquery('english', %s)
        ORDER BY ts_rank_cd(tsv, websearch_to_tsquery('english', %s)) DESC
        LIMIT %s
        """,
        (query, query, limit),
    ).fetchall()
    return [r[0] for r in rows]


def _vector_candidates(conn, query_vec: list[float], limit: int) -> list[str]:
    from pgvector import Vector

    rows = conn.execute(
        "SELECT id FROM chunks ORDER BY embedding <=> %s LIMIT %s",
        (Vector(query_vec), limit),
    ).fetchall()
    return [r[0] for r in rows]


def _rrf(rankings: list[list[str]], k: int) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _check_embedder(conn, embedder: Embedder) -> None:
    ingested_with = db.get_meta(conn, "embedder")
    if ingested_with and ingested_with != embedder.name:
        raise RuntimeError(
            f"corpus was ingested with embedder '{ingested_with}' but query uses "
            f"'{embedder.name}'; re-run `anvil ingest --embedder ...` to match"
        )


def search(
    query: str,
    top_k: int = config.DEFAULT_TOP_K,
    embedder_kind: str = "auto",
    rerank: bool = True,
) -> list[RetrievedChunk]:
    """Hybrid retrieval: Postgres FTS + pgvector cosine, RRF fusion, cross-encoder rerank."""
    embedder = get_embedder(embedder_kind)
    with db.connect() as conn:
        _check_embedder(conn, embedder)
        fts = _fts_candidates(conn, query, config.CANDIDATES_PER_ARM)
        vec = _vector_candidates(conn, embedder.embed_query(query), config.CANDIDATES_PER_ARM)
        fused = _rrf([fts, vec], config.RRF_K)
        pool_size = config.RERANK_POOL if rerank else top_k
        pool_ids = [cid for cid, _ in sorted(fused.items(), key=lambda kv: -kv[1])[:pool_size]]
        if not pool_ids:
            return []
        rows = conn.execute(
            "SELECT id, doc_slug, heading_path, content FROM chunks WHERE id = ANY(%s)",
            (pool_ids,),
        ).fetchall()

    by_id = {r[0]: r for r in rows}
    pool = [
        RetrievedChunk(id=cid, doc_slug=by_id[cid][1], heading_path=by_id[cid][2],
                       content=by_id[cid][3], fused_score=fused[cid])
        for cid in pool_ids if cid in by_id
    ]
    if rerank:
        scores = get_reranker().score(query, [c.content for c in pool])
        for chunk, score in zip(pool, scores, strict=True):
            chunk.rerank_score = score
        pool.sort(key=lambda c: -c.rerank_score)
    return pool[:top_k]
