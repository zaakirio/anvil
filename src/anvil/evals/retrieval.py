"""Retrieval evals: golden queries with labeled relevant-chunk id prefixes.

A retrieved chunk counts as relevant for a label when its id equals the label or is a
split part of it ("<label>/2"). Metrics: recall@5, recall@10, MRR, nDCG@10.
Runs keyless with the fallback embedder so CI can gate on it.
"""

import json
import math
from datetime import UTC, datetime
from pathlib import Path

from anvil import config, db
from anvil.retrieval import search


def load_goldens(path: Path | None = None) -> list[dict]:
    path = path or config.GOLDENS_DIR / "retrieval.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _matches(chunk_id: str, label: str) -> bool:
    return chunk_id == label or chunk_id.startswith(label + "/")


def validate_labels(goldens: list[dict]) -> None:
    with db.connect() as conn:
        ids = {r[0] for r in conn.execute("SELECT id FROM chunks").fetchall()}
    missing = []
    for g in goldens:
        for label in g["relevant"]:
            if not any(_matches(cid, label) for cid in ids):
                missing.append((g["id"], label))
    if missing:
        detail = "\n".join(f"  {qid}: {label}" for qid, label in missing)
        raise ValueError(f"golden labels not present in ingested corpus:\n{detail}")


def _recall_at(ranked: list[str], labels: list[str], k: int) -> float:
    covered = {lab for lab in labels if any(_matches(cid, lab) for cid in ranked[:k])}
    return len(covered) / len(labels)


def _mrr(ranked: list[str], labels: list[str]) -> float:
    for i, cid in enumerate(ranked):
        if any(_matches(cid, lab) for lab in labels):
            return 1.0 / (i + 1)
    return 0.0


def _ndcg_at(ranked: list[str], labels: list[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, cid in enumerate(ranked[:k])
        if any(_matches(cid, lab) for lab in labels)
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(labels), k)))
    return dcg / ideal if ideal else 0.0


def run_retrieval_eval(
    embedder_kind: str = "auto",
    rerank: bool = True,
    goldens_path: Path | None = None,
    out_path: Path | None = None,
) -> dict:
    goldens = load_goldens(goldens_path)
    validate_labels(goldens)

    per_query = []
    for g in goldens:
        ranked = [c.id for c in search(g["query"], top_k=10, embedder_kind=embedder_kind, rerank=rerank)]
        per_query.append(
            {
                "id": g["id"],
                "query": g["query"],
                "relevant": g["relevant"],
                "retrieved": ranked,
                "recall_at_5": _recall_at(ranked, g["relevant"], 5),
                "recall_at_10": _recall_at(ranked, g["relevant"], 10),
                "mrr": _mrr(ranked, g["relevant"]),
                "ndcg_at_10": _ndcg_at(ranked, g["relevant"], 10),
            }
        )

    n = len(per_query)
    metrics = {
        m: round(sum(q[m] for q in per_query) / n, 4)
        for m in ("recall_at_5", "recall_at_10", "mrr", "ndcg_at_10")
    }
    from anvil.embeddings import get_embedder

    result = {
        "kind": "retrieval",
        "ts": datetime.now(UTC).isoformat(),
        "embedder": get_embedder(embedder_kind).name,
        "rerank": rerank,
        "queries": n,
        "metrics": metrics,
        "per_query": per_query,
    }
    config.ensure_dirs()
    out = out_path or config.RESULTS_DIR / f"retrieval-{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    result["out_path"] = str(out)
    return result
