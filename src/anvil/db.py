from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector

from anvil import config

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    slug TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_slug TEXT NOT NULL REFERENCES documents(slug) ON DELETE CASCADE,
    heading_path TEXT NOT NULL,
    content TEXT NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding vector
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_slug);

CREATE TABLE IF NOT EXISTS ingest_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@contextmanager
def connect():
    with psycopg.connect(config.DATABASE_URL, autocommit=True) as conn:
        register_vector(conn)
        yield conn


def init_schema() -> None:
    with psycopg.connect(config.DATABASE_URL, autocommit=True) as conn:
        conn.execute(SCHEMA)


def set_meta(conn, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO ingest_meta (key, value) VALUES (%s, %s) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        (key, value),
    )


def get_meta(conn, key: str) -> str | None:
    row = conn.execute("SELECT value FROM ingest_meta WHERE key = %s", (key,)).fetchone()
    return row[0] if row else None
