from pathlib import Path

from pgvector import Vector

from anvil import config, db
from anvil.chunking import chunk_markdown
from anvil.embeddings import get_embedder


def ingest_corpus(corpus_dir: Path = config.CORPUS_DIR, embedder_kind: str = "auto") -> dict:
    embedder = get_embedder(embedder_kind)
    paths = sorted(corpus_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no markdown docs found in {corpus_dir}")

    db.init_schema()
    docs = 0
    all_chunks = []
    doc_rows = []
    for path in paths:
        slug, title, chunks = chunk_markdown(path)
        doc_rows.append((slug, title, str(path)))
        all_chunks.extend(chunks)
        docs += 1

    embeddings = embedder.embed_documents([c.content for c in all_chunks])

    # One transaction so a crash mid-ingest cannot leave a truncated index with stale meta.
    with db.connect() as conn, conn.transaction():
        conn.execute("TRUNCATE documents CASCADE")
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO documents (slug, title, path) VALUES (%s, %s, %s)", doc_rows
            )
            cur.executemany(
                "INSERT INTO chunks (id, doc_slug, heading_path, content, embedding) "
                "VALUES (%s, %s, %s, %s, %s)",
                [
                    (c.id, c.doc_slug, c.heading_path, c.content, Vector(e))
                    for c, e in zip(all_chunks, embeddings, strict=True)
                ],
            )
        db.set_meta(conn, "embedder", embedder.name)
        db.set_meta(conn, "embedding_dim", str(embedder.dim))
    return {"documents": docs, "chunks": len(all_chunks), "embedder": embedder.name}
